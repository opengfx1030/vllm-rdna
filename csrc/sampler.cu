#include "core/cuda_compat.h"
#include "dispatch_utils.h"
#include "torch_utils.h"

#include <cstring>
#include <cstdlib>
#include <strings.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

// The tie-stable top-k selection (VLLM_RDNA2_MOE_DETERMINISTIC=1) is compiled
// into a second translation unit, csrc/sampler_deterministic.cu, which defines
// VLLM_TOPK_DETERMINISTIC_TU and includes this file. This unit then holds
// exactly the kernels it held before the selection existed: same names, same
// machine code, same place in the code object. Instantiating the deterministic
// kernels here instead moved every default kernel and widened the lookup table
// of the default jobs' shared arrays, and the default path measured 1.4-2.1 %
// slower on tie-heavy rows of 4096 and 8192 columns.
//
// VLLM_TOPK_INLINE_CALL: a call the compiler must inline (HIP only). The
// deterministic jobs measured faster inlined into their kernels than called.
#if defined(VLLM_TOPK_DETERMINISTIC_TU) && defined(USE_ROCM)
  #define VLLM_TOPK_INLINE_CALL [[clang::always_inline]]
#else
  #define VLLM_TOPK_INLINE_CALL
#endif

namespace vllm {

#ifdef VLLM_TOPK_DETERMINISTIC_TU
static constexpr bool kDeterministicTies = true;
#else
static constexpr bool kDeterministicTies = false;
#endif

template <typename scalar_t>
__global__ void apply_repetition_penalties_kernel(
    scalar_t* __restrict__ logits,         // [num_seqs, vocab_size]
    const bool* __restrict__ prompt_mask,  // [num_seqs, vocab_size]
    const bool* __restrict__ output_mask,  // [num_seqs, vocab_size]
    const scalar_t* __restrict__ repetition_penalties,  // [num_seqs]
    const int num_seqs, const int vocab_size, const int tile_size) {
  // Each block handles one sequence and a tile of vocab
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);

  // Load repetition penalty for this sequence
  const scalar_t penalty = repetition_penalties[seq_idx];

  // Each thread processes multiple vocab items within the tile
  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    const bool is_repeated = prompt_mask[idx] || output_mask[idx];
    if (is_repeated) {
      scalar_t logit = logits[idx];
      if (logit > 0) {
        logits[idx] = logit / penalty;
      } else {
        logits[idx] = logit * penalty;
      }
    }
  }
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (step == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads,
                                   const T* in, idxT len, Func f) {
  // Use dynamic WARP_SIZE from cuda_compat.h to support both
  // Wave64 (MI300X/gfx942) and Wave32 (Strix Halo/gfx1151) architectures
  constexpr int kWarpSize = WARP_SIZE;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt =
        (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
               sizeof(T))
            : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, kWarpSize > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // kWarpSize no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

// The column the strided loops (stride1 != 1) hand to the per-element
// callbacks of processHistogramStep. vectorized_process hands them the column
// relative to rowStart, and a split block stores rowStart plus that column: a
// column of the whole row, which the merge kernel reads back as it is. The
// strided loops walk absolute columns, so a split block passes them on
// relative too; stored as they are, the columns of every block after the first
// would count rowStart twice. The other kernels store the absolute column and
// subtract rowStart when they write the row out.
template <bool multipleBlocksPerRow>
static inline __device__ int stridedLoopIdx(int idx, int rowStart) {
  return multipleBlocksPerRow ? idx - rowStart : idx;
}

// Deterministic tie membership (the second translation unit only).
//
// True when every index a block stores is below 2^21, so the top digit of
// its tie key is zero. Merged rows store original columns and -1 padding.
template <bool mergeBlocks>
static inline __device__ bool storedIndicesBelow2Pow21(int rowStart,
                                                       int rowEnd) {
  return !mergeBlocks &&
         static_cast<uint32_t>(rowStart) + static_cast<uint32_t>(rowEnd) <=
             (1u << 21);
}

// Writes the `need` candidates with the lowest keys to the output slots
// [base, base + need), in no set order. forEachCandidate(visit) calls
// visit(key, store) once per candidate; store(dstIdx) writes it out.
// Candidates with one key (only a merged row's -1 padding repeats a key)
// are interchangeable.
//
// A radix select over the 32-bit keys, 11, 11 and 10 bits from the top (the
// top digit is skipped when keysBelow2Pow21). The digit counts do not depend
// on the order threads reach the atomics, so the written set does not
// either. Scratch: smemFinal.histo and the four shared ints. The caller syncs
// before the call (nobody still reads them) and reads the slots after it.
template <int kNumThreadsPerBlock, int kNumBins, typename SmemFinalType,
          typename ForEachCandidate>
__device__ void emitLowestKeys(ForEachCandidate&& forEachCandidate, int need,
                               int base, bool keysBelow2Pow21,
                               SmemFinalType& smemFinal, int* smemDigit,
                               int* smemBelow, int* smemDst, int* smemDstLast) {
  static_assert(kNumBins >= 2048 && kNumBins % kNumThreadsPerBlock == 0);
  constexpr int kBinsPerThread = kNumBins / kNumThreadsPerBlock;
  // Digit 0 is key bits 21-31, digit 1 bits 10-20, digit 2 bits 0-9.
  auto digitShift = [](int level) {
    return level == 0 ? 21 : level == 1 ? 10 : 0;
  };
  auto digitMask = [](int level) { return level == 2 ? 0x3ffu : 0x7ffu; };
  const int firstLevel = keysBelow2Pow21 ? 1 : 0;
  uint32_t keyPrefix = 0;  // the threshold digits chosen so far
  int threshold = 0;       // the threshold digit of the previous level
  int below = 0;           // the candidates on the prefix under it
  int emitted = 0;         // the slots taken under earlier threshold digits
  for (int level = firstLevel; level <= 3; ++level) {
    // One pass: write the candidates under the previous threshold digit and
    // count the ones on it per digit of this level. Level 3 has no digit
    // left: the ones on it share one key; write them while slots remain.
    if (level < 3) {
      for (int b = threadIdx.x; b < kNumBins; b += kNumThreadsPerBlock) {
        smemFinal.histo.data[b] = 0;
      }
      __syncthreads();
    }
    forEachCandidate([&](uint32_t key, auto&& store) {
      if (level > firstLevel) {
        const int prev = level - 1;
        // The digits above the previous one must match the chosen prefix.
        if (prev > firstLevel &&
            ((key ^ keyPrefix) >> digitShift(prev - 1)) != 0) {
          return;
        }
        const int digit =
            static_cast<int>((key >> digitShift(prev)) & digitMask(prev));
        if (digit < threshold) {
          store(base + emitted + atomicAdd(smemDst, 1));
          return;
        }
        if (digit > threshold) {
          return;
        }
        if (level == 3) {
          const int dst = emitted + below + atomicAdd(smemDstLast, 1);
          if (dst < need) {
            store(base + dst);
          }
          return;
        }
      }
      atomicAdd(
          &smemFinal.histo.data[(key >> digitShift(level)) & digitMask(level)],
          1);
    });
    __syncthreads();
    if (level == 3) {
      return;
    }
    emitted += below;
    // This level's threshold digit: the first at which the running count
    // reaches the slots still free.
    const int slots = need - emitted;
    int counts[kBinsPerThread];
    int sum = 0;
#pragma unroll
    for (int j = 0; j < kBinsPerThread; ++j) {
      counts[j] = smemFinal.histo.data[threadIdx.x * kBinsPerThread + j];
      sum += counts[j];
    }
    int before = 0, total = 0;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(sum, before, total);
#pragma unroll
    for (int j = 0; j < kBinsPerThread; ++j) {
      if (before < slots && before + counts[j] >= slots) {
        *smemDigit = threadIdx.x * kBinsPerThread + j;
        *smemBelow = before;
      }
      before += counts[j];
    }
    if (threadIdx.x == 0) {
      *smemDst = 0;
      *smemDstLast = 0;
    }
    __syncthreads();
    threshold = *smemDigit;
    below = *smemBelow;
    keyPrefix |= static_cast<uint32_t>(threshold) << digitShift(level);
  }
}

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks,
          bool useTopK512Optimization, bool deterministicTies,
          typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices, const float* logits, int rowEnd, uint32_t& logitPattern,
    int& thresholdBinIdx, SmemOutputType& smemOutput, int* smemThresholdBinIdx,
    int* smemFinalDstIdx, int* smemFinalBinSize, int* smemFoundTopKValues,
    SmemFinalType& smemFinal, int stride1, int rowStart, int topK) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                   << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff)
                    << patternShift;
  }

  int negativeInfinityCount = 0;
  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (useTopK512Optimization && __float_as_uint(logit) == 0xff800000u) {
        ++negativeInfinityCount;
      } else {
        atomicAdd(&smemFinal.histo.data[binIdx], 1);
      }
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  if constexpr (useTopK512Optimization) {
    if (negativeInfinityCount > 0) {
      const auto bin = extractBinIdx<step>(-INFINITY);
      atomicAdd(&smemFinal.histo.data[bin], negativeInfinityCount);
    }
  }
  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining
    // rounds
    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];
  // Resolve a -inf cutoff with exact tie emission, keeping sort padding out.
  const bool finalBinFits =
      smemFinalBinSize[0] <= kNumFinalItems &&
      !(useTopK512Optimization && step < 3 &&
        isPartialMatch<patternShift>(-INFINITY, logitPattern) &&
        thresholdBinIdx == extractBinIdx<step>(-INFINITY));

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      // Only write elements with binIdx < thresholdBinIdx when:
      // 1. This is step 0 and the threshold bin is small enough (no step 1)
      // 2. This is step >= 1 (where pattern matching filters correctly)
      // This prevents duplicates when step 0 and step 1 both run.
      bool shouldWriteDirectly = (step == 0 && finalBinFits) || (step >= 1);
      if (binIdx < thresholdBinIdx && shouldWriteDirectly) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput[dstIdx] = idx + rowStart;
          reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
        } else {
          smemOutput[dstIdx] = idx;
        }
      }
      if constexpr (step < 3) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx && finalBinFits) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      } else if constexpr (!deterministicTies) {
        // Default: the threads that reach the counter first keep their slot.
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 3
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < topK) {
            if constexpr (mergeBlocks) {
              smemOutput[dstIdx] = indices[idx];
            } else if constexpr (multipleBlocksPerRow) {
              smemOutput[dstIdx] = idx + rowStart;
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
            } else {
              smemOutput[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                       rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd;
         idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, stridedLoopIdx<multipleBlocksPerRow>(idx, rowStart));
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  if constexpr (deterministicTies && step == 3) {
    // The threshold bin holds equal logits (all 32 bits match), more of them
    // than the slots left. The default pass above takes a slot per element
    // with an atomic, so which of them are kept changes run to run; select the
    // ones with the lowest stored indices instead.
    //
    // idx is relative to rowStart in the vectorized loop and in a split
    // block's strided loop, absolute in the strided loop of the other kernels
    // (stridedLoopIdx).
    auto forEachElement = [&](auto&& f) {
      if (stride1 == 1) {
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart,
                           rowEnd - rowStart, f);
      } else {
        for (int idx = rowStart + threadIdx.x; idx < rowEnd;
             idx += kNumThreadsPerBlock) {
          f(logits[idx * stride1],
            stridedLoopIdx<multipleBlocksPerRow>(idx, rowStart));
        }
      }
    };
    // The index this block outputs for element idx. It is also the tie key.
    auto storedIndex = [&](int idx) {
      if constexpr (mergeBlocks) {
        return indices[idx];
      } else if constexpr (multipleBlocksPerRow) {
        return idx + rowStart;
      } else {
        return idx;
      }
    };
    auto emit = [&](int dstIdx, float logit, int idx) {
      smemOutput[dstIdx] = storedIndex(idx);
      if constexpr (multipleBlocksPerRow && !mergeBlocks) {
        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
      }
    };
    const int base = smemFoundTopKValues[0];
    emitLowestKeys<kNumThreadsPerBlock, kNumBins>(
        [&](auto&& visit) {
          forEachElement([&](float logit, int idx) {
            if (isPartialMatch<patternShift>(logit, logitPattern) &&
                extractBinIdx<3>(logit) ==
                    static_cast<uint32_t>(thresholdBinIdx)) {
              visit(static_cast<uint32_t>(storedIndex(idx)),
                    [&](int dstIdx) { emit(dstIdx, logit, idx); });
            }
          });
        },
        topK - base, base,
        storedIndicesBelow2Pow21<mergeBlocks>(rowStart, rowEnd), smemFinal,
        smemThresholdBinIdx, smemFinalBinSize, smemFinalDstIdx,
        smemFoundTopKValues);
  }

  // Check if we should continue to next step
  return !finalBinFits;
}

// Follows half - 11 - 11 - 10 bit iterations
// Keep the adaptive kernel's device instantiation separate from legacy calls.
//
// kDeterministicTies (true in the second translation unit, see the top of the
// file): equal logits at the cut keep the lowest stored indices (the column;
// for a merged row, the column its split block stored), so the selected set
// does not depend on the order threads reach the shared-memory atomics. The
// order of the output indices still does. The default keeps the atomic-order
// selection, which costs less on tie-heavy rows. The gfx950
// device-length-aware instantiations always keep it.
template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false,
          bool deviceLengthAware = false, bool useTopK512Optimization = false>
static __device__ void topKPerRowJob(const int* indices, const float* logits,
                                     int rowStart, int rowEnd, int* outIndices,
                                     float* outLogits, int stride1, int topK) {
  static_assert(!deviceLengthAware || multipleBlocksPerRow != mergeBlocks);
  static_assert(!useTopK512Optimization || kNumThreadsPerBlock == 1024);
  static constexpr bool deterministicTies =
      kDeterministicTies && !deviceLengthAware && !useTopK512Optimization;
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = useTopK512Optimization ? 1024 : 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage =
      std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  // Shared memory to store the selected indices.
  // If we are processing using multiple blocks, we need to store the logits and
  // indices.
  extern __shared__ int32_t smemOutput[];

  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the
  // previous iterations
  __shared__ int smemFoundTopKValues[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= topK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      if constexpr (multipleBlocksPerRow) {
        outIndices[rowIt] = rowIt + rowStart;
        // Consecutive columns are stride1 apart when the logits are strided.
        outLogits[rowIt] = logits[(rowIt + rowStart) * stride1];
      } else {
        outIndices[rowIt] = rowIt;
      }
    }
    for (int rowIt = rowLen + threadIdx.x; rowIt < topK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
      }
    }

    return;
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits of half representation
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                           multipleBlocksPerRow, mergeBlocks,
                           useTopK512Optimization, deterministicTies>(
          indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
          smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
          smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks,
                             useTopK512Optimization, deterministicTies>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 2: Process next 11 bits
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                             multipleBlocksPerRow, mergeBlocks,
                             useTopK512Optimization, deterministicTies>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
            smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (continueToNextStep) {
    // Step 3: Process last 10 bits
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems,
                         multipleBlocksPerRow, mergeBlocks,
                         useTopK512Optimization, deterministicTies>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize,
        smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to
    // sort the final items The logits of the elements to be sorted in the final
    // pass.
    auto insertionSort = [&]() {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0];
           i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];
        [[maybe_unused]] uint32_t key = 0;
        if constexpr (deterministicTies) {
          key = static_cast<uint32_t>(smemFinal.items.indices[i]);
        }
        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if constexpr (deterministicTies) {
            // Equal logits are ranked by stored index (lowest first; slot
            // order between copies of one index), so which of them make the
            // cut does not depend on the order the bin was filled in.
            if (logit < otherLogit) {
              outIndex++;
            } else if (logit == otherLogit) {
              const auto otherKey =
                  static_cast<uint32_t>(smemFinal.items.indices[j]);
              outIndex += otherKey < key || (otherKey == key && j < i);
            }
          } else {
            if (logit < otherLogit || (logit == otherLogit && i < j)) {
              outIndex++;
            }
          }
        }
        // Store if outIndex is in bounds
        if (outIndex + baseIdx < topK) {
          smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] =
                smemFinal.items.logits[i];
          }
        }
      }
    };
    if constexpr (useRadixSort) {
      if (useTopK512Optimization && smemFinalDstIdx[0] <= 128) {
        insertionSort();
      } else {
        // Sorting with radix sort
        float finalLogits[kNumFinalItemsPerThread];
        // The indices of the elements to be sorted in the final pass.
        int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          finalLogits[ii] = useTopK512Optimization ? -INFINITY : -FLT_MAX;
          if constexpr (useTopK512Optimization) finalIndices[ii] = -1;
        }

        // Read the elements from SMEM.
#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
          if (srcIdx < smemFinalDstIdx[0]) {
            finalLogits[ii] = smemFinal.items.logits[srcIdx];
            finalIndices[ii] = smemFinal.items.indices[srcIdx];
          }
        }
        // Make sure the shared memory has been read.
        __syncthreads();

        // Sort the elements.
        FinalSort(smemFinal.finalSort)
            .SortDescendingBlockedToStriped(finalLogits, finalIndices);

        // Copy the data back to the shared memory storage.
        int baseIdx = smemFoundTopKValues[0];

#pragma unroll
        for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
          int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
          int dstIdx = baseIdx + srcIdx;

          if (dstIdx < topK) {
            smemOutput[dstIdx] = finalIndices[ii];
            if constexpr (multipleBlocksPerRow) {
              reinterpret_cast<float*>(smemOutput + topK)[dstIdx] =
                  finalLogits[ii];
            }
          }
        }

        if constexpr (deterministicTies) {
          // The sort leaves equal logits in the order the bin was filled in,
          // by atomic slot, which changes run to run. When equal logits
          // straddle the cut, refill the slots of the tied ones before it
          // with the lowest stored indices among all tied ones. The bin fills
          // [baseIdx, topK) with the first `cut` sorted items.
          const int numItems = smemFinalDstIdx[0];
          const int cut = topK - baseIdx;
          // The radix-sorted logit bits at the cut and right after it.
          __shared__ uint32_t smemCutBits[2];
#pragma unroll
          for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
            const int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
            if (srcIdx == cut - 1 || srcIdx == cut) {
              smemCutBits[srcIdx - cut + 1] = __float_as_uint(finalLogits[ii]);
            }
          }
          __syncthreads();
          if (cut < numItems && smemCutBits[0] == smemCutBits[1]) {
            const uint32_t tieBits = smemCutBits[0];
            auto tiedAt = [&](int ii) {
              const int pos = ii * kNumThreadsPerBlock + threadIdx.x;
              return pos < numItems &&
                     __float_as_uint(finalLogits[ii]) == tieBits;
            };
            int keep = 0;
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
              const int pos = ii * kNumThreadsPerBlock + threadIdx.x;
              keep += __syncthreads_count(pos < cut && tiedAt(ii));
            }
            emitLowestKeys<kNumThreadsPerBlock, kNumBins>(
                [&](auto&& visit) {
#pragma unroll
                  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                    if (tiedAt(ii)) {
                      visit(static_cast<uint32_t>(finalIndices[ii]),
                            [&](int dstIdx) {
                              smemOutput[dstIdx] = finalIndices[ii];
                              if constexpr (multipleBlocksPerRow) {
                                reinterpret_cast<float*>(smemOutput +
                                                         topK)[dstIdx] =
                                    finalLogits[ii];
                              }
                            });
                    }
                  }
                },
                keep, topK - keep,
                storedIndicesBelow2Pow21<mergeBlocks>(rowStart, rowEnd),
                smemFinal, smemThresholdBinIdx, smemFinalBinSize,
                smemFinalDstIdx, smemFoundTopKValues);
          }
        }
      }
    } else {
      insertionSort();
    }
    __syncthreads();
  }

  // Store to global memory.
  for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
    if constexpr (multipleBlocksPerRow) {
      outIndices[i] = smemOutput[i];
      outLogits[i] = reinterpret_cast<float*>(smemOutput + topK)[i];
    } else {
      if (stride1 == 1) {
        // stride1 == 1 will use vectorized_process, which indexes already skip
        // the rowStart.
        outIndices[i] = smemOutput[i];
      } else {
        outIndices[i] = smemOutput[i] - rowStart;
      }
    }
  }
}

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowPrefill(
    const float* logits, const int* rowStarts, const int* rowEnds,
    int* outIndices, int stride0, int stride1, const int topK,
    const int offsetIndex) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int rowStart = rowStarts[rowIdx];
  int rowEnd = rowEnds[rowIdx];

  // Local pointers to this block
  outIndices += static_cast<int64_t>(rowIdx) * topK;
  logits += static_cast<int64_t>(rowIdx) * stride0;

  VLLM_TOPK_INLINE_CALL topKPerRowJob<kNumThreadsPerBlock, kNumBins,
                                      useRadixSort>(
      nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

template <int kNumThreadsPerBlock, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode(
    const float* logits, const int* seqLens, int* outIndices, int stride0,
    int stride1, const int topK, int next_n, int seqLensIs2D = 0,
    float* outLogits = nullptr, const int numBlocksToMerge = 0,
    const int* indices = nullptr) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x;

  // The range of logits within the row.
  int rowStart = 0;
  int batch_idx = rowIdx / next_n;
  int next_n_idx = rowIdx % next_n;
  // seqLensIs2D=0: 1D seqLens — all rows in a batch share the same seq_len;
  //               kernel computes per-row effective length via offset.
  // seqLensIs2D=1: 2D seqLens — each logit row has its own pre-computed
  //               effective length (flat index rowIdx = b*next_n + j maps
  //               directly to seqLens[b, j] in C-contiguous layout).
  int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch_idx];
  int rowEnd =
      seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + next_n_idx + 1);

  // Local pointers to this block
  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  } else if constexpr (multipleBlocksPerRow) {
    const auto blockSize = rowEnd / gridDim.y;  // 16384 / 2 = 8192
    rowStart = blockSize * blockIdx.y;          // 8192 * 1 = 8192
    rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    outLogits +=
        static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
  } else if constexpr (mergeBlocks) {
    rowEnd = numBlocksToMerge * topK;
    indices += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  }
  logits += static_cast<int64_t>(rowIdx) * stride0;

  VLLM_TOPK_INLINE_CALL topKPerRowJob<kNumThreadsPerBlock, kNumBins,
                                      useRadixSort, multipleBlocksPerRow,
                                      mergeBlocks>(
      indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
}

#ifdef USE_ROCM
// Measured native-kernel policy for gfx950, FP32, and k=1024. This
// active-column split crossover is independent of the Python AITER/native
// backend crossover; the two policies can be retuned separately. At exactly
// 64K, an eager call may select AITER while a captured native replay uses two
// splits.
constexpr int kGfx950C4AShortRowNumBlocks = 3;
constexpr int kGfx950C4ALongRowNumBlocks = 2;
constexpr int kGfx950C4ATwoSplitMinRowLength = 64 * 1024;

template <int kNumThreadsPerBlock, bool mergeBlocks = false>
static __global__
__launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecodeDeviceLengthAware(
    const float* logits, const int* seqLens, int* outIndices, int stride0,
    int stride1, const int topK, int next_n, int seqLensIs2D = 0,
    float* outLogits = nullptr, const int* indices = nullptr) {
  static constexpr int kNumBins = 2048;
  int rowIdx = blockIdx.x;
  int batch_idx = rowIdx / next_n;
  int next_n_idx = rowIdx % next_n;
  int seq_len = seqLensIs2D ? seqLens[rowIdx] : seqLens[batch_idx];
  int rowEnd =
      seqLensIs2D ? max(0, seq_len) : max(0, seq_len - next_n + next_n_idx + 1);
  const int activeBlocks = rowEnd >= kGfx950C4ATwoSplitMinRowLength
                               ? kGfx950C4ALongRowNumBlocks
                               : kGfx950C4AShortRowNumBlocks;
  int rowStart = 0;

  if constexpr (!mergeBlocks) {
    if (blockIdx.y >= activeBlocks) return;
    const auto blockSize = rowEnd / activeBlocks;
    rowStart = blockSize * blockIdx.y;
    rowEnd = activeBlocks == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices +=
        static_cast<int64_t>(rowIdx) * kGfx950C4AShortRowNumBlocks * topK +
        blockIdx.y * topK;
    outLogits +=
        static_cast<int64_t>(rowIdx) * kGfx950C4AShortRowNumBlocks * topK +
        blockIdx.y * topK;
  } else {
    rowEnd = activeBlocks * topK;
    indices +=
        static_cast<int64_t>(rowIdx) * kGfx950C4AShortRowNumBlocks * topK;
    outIndices += static_cast<int64_t>(rowIdx) * topK;
  }
  logits += static_cast<int64_t>(rowIdx) * stride0;

  topKPerRowJob<kNumThreadsPerBlock, kNumBins, true, !mergeBlocks, mergeBlocks,
                true>(indices, logits, rowStart, rowEnd, outIndices, outLogits,
                      stride1, topK);
}

// The grid and workspace stay fixed while graph replays change device lengths.
__device__ int gfx950TopK512ActiveBlocks(int rowLength, int numRows,
                                         int maxBlocks) {
  int activeBlocks = 1;
  if (numRows <= 32 && rowLength > 16 * 1024) {
    activeBlocks = rowLength <= 128 * 1024   ? 8
                   : rowLength <= 256 * 1024 ? 10
                                             : 16;
  } else if (numRows <= 128 && rowLength > 64 * 1024) {
    activeBlocks = rowLength <= 512 * 1024 ? 4 : 10;
  } else if (numRows > 128 && numRows <= 256 && rowLength > 256 * 1024) {
    activeBlocks = numRows <= 160 ? 3 : 2;
  }
  // Every block in a merged row must contain at least k real indices.
  return min(min(activeBlocks, maxBlocks), max(1, rowLength / 512));
}

template <int kNumThreadsPerBlock, bool mergeBlocks = false>
static __global__
__launch_bounds__(kNumThreadsPerBlock) void topKPerRowDecode512DeviceLengthAware(
    const float* logits, const int* seqLens, int* outIndices,
    int* outIndicesAux, float* outLogitsAux, int stride0, int next_n,
    int seqLensIs2D, int maxBlocks) {
  #if defined(__gfx950__)
  constexpr int topK = 512;
  constexpr int kNumBins = 2048;
  const int rowIdx = blockIdx.x;
  const int rowEnd =
      seqLensIs2D
          ? max(0, seqLens[rowIdx])
          : max(0, seqLens[rowIdx / next_n] - next_n + rowIdx % next_n + 1);
  const int activeBlocks =
      gfx950TopK512ActiveBlocks(rowEnd, gridDim.x, maxBlocks);
  outIndices += static_cast<int64_t>(rowIdx) * topK;
  if constexpr (mergeBlocks) {
    if (activeBlocks == 1) return;
    const int64_t offset = static_cast<int64_t>(rowIdx) * maxBlocks * topK;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, true, false, true, false,
                  true>(outIndicesAux + offset, outLogitsAux + offset, 0,
                        activeBlocks * topK, outIndices, nullptr, 1, topK);
  } else {
    if (blockIdx.y >= activeBlocks) return;
    logits += static_cast<int64_t>(rowIdx) * stride0;
    if (activeBlocks == 1) {
      topKPerRowJob<kNumThreadsPerBlock, kNumBins, true, false, false, false,
                    true>(nullptr, logits, 0, rowEnd, outIndices, nullptr, 1,
                          topK);
      return;
    }
    const int blockSize = rowEnd / activeBlocks;
    const int rowStart = blockSize * blockIdx.y;
    const int blockRowEnd =
        blockIdx.y + 1 == activeBlocks ? rowEnd : rowStart + blockSize;
    const int64_t offset =
        (static_cast<int64_t>(rowIdx) * maxBlocks + blockIdx.y) * topK;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, true, true, false, false,
                  true>(nullptr, logits, rowStart, blockRowEnd,
                        outIndicesAux + offset, outLogitsAux + offset, 1, topK);
  }
  #endif
}
#endif  // USE_ROCM

#ifndef VLLM_TOPK_DETERMINISTIC_TU
// VLLM_RDNA2_MOE_DETERMINISTIC (vllm/envs.py): "1" or "true", any case. Besides
// the RDNA2 MoE epilogue it selects the top-k kernels that keep the lowest
// stored indices among tied logits at the cut (sampler_deterministic.cu). Read
// on the first launch and constant for the rest of the process, so a CUDA graph
// captured at any point replays the variant an eager call would run.
static bool topKDeterministicTies() {
  static const bool enabled = [] {
    const char* value = std::getenv("VLLM_RDNA2_MOE_DETERMINISTIC");
    return value != nullptr &&
           (std::strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0);
  }();
  return enabled;
}
#endif  // !VLLM_TOPK_DETERMINISTIC_TU

}  // namespace vllm

#ifndef VLLM_TOPK_DETERMINISTIC_TU
void apply_repetition_penalties_(
    torch::stable::Tensor& logits,  // [num_seqs, vocab_size], in-place
    const torch::stable::Tensor& prompt_mask,  // [num_seqs, vocab_size]
    const torch::stable::Tensor& output_mask,  // [num_seqs, vocab_size]
    const torch::stable::Tensor& repetition_penalties) {  // [num_seqs]
  STD_TORCH_CHECK(logits.is_contiguous());
  STD_TORCH_CHECK(prompt_mask.is_contiguous());
  STD_TORCH_CHECK(output_mask.is_contiguous());
  STD_TORCH_CHECK(repetition_penalties.is_contiguous());

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get number of SMs on the current device
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         logits.get_device_index());

  // Compute tile_num and tile_size
  int tile_num =
      std::min(vocab_size, std::max(1, (sms + num_seqs - 1) / num_seqs));
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  // Each block handles one sequence and a tile of vocab
  dim3 grid(num_seqs, tile_num);
  dim3 block(std::min(tile_size, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        vllm::apply_repetition_penalties_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                logits.mutable_data_ptr<scalar_t>(),
                prompt_mask.const_data_ptr<bool>(),
                output_mask.const_data_ptr<bool>(),
                repetition_penalties.const_data_ptr<scalar_t>(), num_seqs,
                vocab_size, tile_size);
      });
}
#endif  // !VLLM_TOPK_DETERMINISTIC_TU

#ifndef VLLM_TOPK_DETERMINISTIC_TU
// Defined by sampler_deterministic.cu.
void top_k_per_row_decode_deterministic(const torch::stable::Tensor& logits,
                                        int64_t next_n,
                                        const torch::stable::Tensor& seqLens,
                                        torch::stable::Tensor& indices,
                                        int64_t numRows, int64_t stride0,
                                        int64_t stride1, int64_t topK);
void top_k_per_row_prefill_deterministic(const torch::stable::Tensor& logits,
                                         const torch::stable::Tensor& rowStarts,
                                         const torch::stable::Tensor& rowEnds,
                                         torch::stable::Tensor& indices,
                                         int64_t numRows, int64_t stride0,
                                         int64_t stride1, int64_t topK);
#endif

// The two launchers are compiled twice. Here they are the ops; in
// sampler_deterministic.cu (VLLM_TOPK_DETERMINISTIC_TU) the same bodies are the
// *_deterministic entry points the ops forward to when
// VLLM_RDNA2_MOE_DETERMINISTIC is set.
#ifdef VLLM_TOPK_DETERMINISTIC_TU
void top_k_per_row_decode_deterministic(
#else
void top_k_per_row_decode(
#endif
    const torch::stable::Tensor& logits, int64_t next_n,
    const torch::stable::Tensor& seqLens, torch::stable::Tensor& indices,
    int64_t numRows, int64_t stride0, int64_t stride1, int64_t topK) {
#ifndef VLLM_TOPK_DETERMINISTIC_TU
  if (vllm::topKDeterministicTies()) {
    top_k_per_row_decode_deterministic(logits, next_n, seqLens, indices,
                                       numRows, stride0, stride1, topK);
    return;
  }
#endif
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kSplitWorkThreshold = 200 * 1000;
  constexpr int kNumThreadsPerBlock = 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const auto numColumns = logits.size(1);

  // True if seqLens is 2D (B, next_n): each logit row has its own pre-computed
  // effective seq_len. False if seqLens is 1D (B,): all rows in a batch share
  // the same seq_len and the kernel computes the per-row offset itself.
  int seqLensIs2D = seqLens.dim() == 2 ? 1 : 0;

#ifdef USE_ROCM
  const bool useGfx950DeviceLengthAwareTopK512 =
      topK == 512 && stride1 == 1 && numRows > 0 && numRows <= 384 &&
      numColumns <= 1024 * 1024 &&
      std::strncmp(get_device_prop()->gcnArchName, "gfx950", 6) == 0;
  if (useGfx950DeviceLengthAwareTopK512) {
    int multipleBlocksPerRowConfig = 1;
    if (numRows <= 32 && numColumns > 16 * 1024) {
      multipleBlocksPerRowConfig = 16;
    } else if (numRows <= 128 && numColumns > 64 * 1024) {
      multipleBlocksPerRowConfig = 10;
    } else if (numRows <= 160 && numColumns > 256 * 1024) {
      multipleBlocksPerRowConfig = 3;
    } else if (numRows <= 256 && numColumns > 256 * 1024) {
      multipleBlocksPerRowConfig = 2;
    }

    constexpr int kNumThreadsPerSplitBlock = 1024;
    if (multipleBlocksPerRowConfig == 1) {
      vllm::topKPerRowDecode512DeviceLengthAware<kNumThreadsPerSplitBlock>
          <<<numRows, kNumThreadsPerSplitBlock, 2 * topK * sizeof(int32_t),
             stream>>>(logits.const_data_ptr<float>(),
                       seqLens.const_data_ptr<int>(),
                       indices.mutable_data_ptr<int>(), nullptr, nullptr,
                       static_cast<int>(stride0), static_cast<int>(next_n),
                       seqLensIs2D, multipleBlocksPerRowConfig);
      return;
    }
    const auto outIndicesAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Int, std::nullopt, logits.device());
    const auto outLogitsAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Float, std::nullopt, logits.device());
    vllm::topKPerRowDecode512DeviceLengthAware<kNumThreadsPerSplitBlock>
        <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerSplitBlock,
           2 * topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(),
            outIndicesAux.mutable_data_ptr<int>(),
            outLogitsAux.mutable_data_ptr<float>(), static_cast<int>(stride0),
            static_cast<int>(next_n), seqLensIs2D, multipleBlocksPerRowConfig);
    constexpr int kNumThreadsPerBlockMerge = 1024;
    vllm::topKPerRowDecode512DeviceLengthAware<kNumThreadsPerBlockMerge, true>
        <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t), stream>>>(
            nullptr, seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(),
            outIndicesAux.mutable_data_ptr<int>(),
            outLogitsAux.mutable_data_ptr<float>(), 0, static_cast<int>(next_n),
            seqLensIs2D, multipleBlocksPerRowConfig);
    return;
  }
#endif

#ifdef USE_ROCM
  // gfx10x (RDNA2): the insertion-sort path is ~16x slower than the radix
  // one at 8 rows x 4096 live columns (443 vs 27 us) and never faster.
  const bool preferRadix =
      std::strncmp(get_device_prop()->gcnArchName, "gfx10", 5) == 0;
#else
  constexpr bool preferRadix = false;
#endif
  if (numColumns < kSortingAlgorithmThreshold && !preferRadix) {
    // Use insertion sort
    vllm::topKPerRowDecode<kNumThreadsPerBlock, false>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), seqLensIs2D);
  } else if (numColumns < kSplitWorkThreshold) {
    // From this threshold, use radix sort instead
    vllm::topKPerRowDecode<kNumThreadsPerBlock, true>
        <<<numRows, kNumThreadsPerBlock, topK * sizeof(int32_t), stream>>>(
            logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
            static_cast<int>(stride1), static_cast<int>(topK),
            static_cast<int>(next_n), seqLensIs2D);
  } else {
    // Long sequences are run in two steps
#ifdef USE_ROCM
    const bool useGfx950DeviceLengthAwareTopK1024 =
        topK == 1024 && numRows > 64 && numRows <= 256 &&
        std::strncmp(get_device_prop()->gcnArchName, "gfx950", 6) == 0;
    if (useGfx950DeviceLengthAwareTopK1024) {
      // Keep three blocks for short FULL-graph replays, but let each row use
      // two blocks once its live device length reaches the long-context
      // crossover. The grid and workspace shapes remain capture-stable.
      constexpr int multipleBlocksPerRowConfig =
          vllm::kGfx950C4AShortRowNumBlocks;
      const auto outIndicesAux = torch::stable::empty(
          {numRows, multipleBlocksPerRowConfig, topK},
          torch::headeronly::ScalarType::Int, std::nullopt, logits.device());
      const auto outLogitsAux = torch::stable::empty(
          {numRows, multipleBlocksPerRowConfig, topK},
          torch::headeronly::ScalarType::Float, std::nullopt, logits.device());

      constexpr int kNumThreadsPerSplitBlock = 1024;
      vllm::topKPerRowDecodeDeviceLengthAware<kNumThreadsPerSplitBlock>
          <<<dim3(numRows, multipleBlocksPerRowConfig),
             kNumThreadsPerSplitBlock, 2 * topK * sizeof(int32_t), stream>>>(
              logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
              outIndicesAux.mutable_data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK),
              static_cast<int>(next_n), seqLensIs2D,
              outLogitsAux.mutable_data_ptr<float>());
      constexpr int kNumThreadsPerBlockMerge = 1024;
      vllm::topKPerRowDecodeDeviceLengthAware<kNumThreadsPerBlockMerge, true>
          <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t),
             stream>>>(
              outLogitsAux.const_data_ptr<float>(),
              seqLens.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
              multipleBlocksPerRowConfig * topK, 1, static_cast<int>(topK),
              static_cast<int>(next_n), seqLensIs2D, nullptr,
              outIndicesAux.const_data_ptr<int>());
      return;
    }
#endif

#ifdef USE_ROCM
    const bool useGfx950TopK1024 =
        topK == 1024 && numRows <= 64 &&
        std::strncmp(get_device_prop()->gcnArchName, "gfx950", 6) == 0;
#else
    constexpr bool useGfx950TopK1024 = false;
#endif
    int multipleBlocksPerRowConfig = 10;
    if (useGfx950TopK1024) {
      if (numRows <= 32) {
        multipleBlocksPerRowConfig = 8;
      } else {
        multipleBlocksPerRowConfig = 4;
      }
    }

    const auto outIndicesAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Int, std::nullopt, logits.device());
    const auto outLogitsAux = torch::stable::empty(
        {numRows, multipleBlocksPerRowConfig, topK},
        torch::headeronly::ScalarType::Float, std::nullopt, logits.device());

    if (useGfx950TopK1024) {
      constexpr int kNumThreadsPerSplitBlock = 1024;
      vllm::topKPerRowDecode<kNumThreadsPerSplitBlock, true, true>
          <<<dim3(numRows, multipleBlocksPerRowConfig),
             kNumThreadsPerSplitBlock, 2 * topK * sizeof(int32_t), stream>>>(
              logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
              outIndicesAux.mutable_data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK),
              static_cast<int>(next_n), seqLensIs2D,
              outLogitsAux.mutable_data_ptr<float>());
    } else {
      vllm::topKPerRowDecode<kNumThreadsPerBlock, true, true>
          <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock,
             2 * topK * sizeof(int32_t), stream>>>(
              logits.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
              outIndicesAux.mutable_data_ptr<int>(), static_cast<int>(stride0),
              static_cast<int>(stride1), static_cast<int>(topK),
              static_cast<int>(next_n), seqLensIs2D,
              outLogitsAux.mutable_data_ptr<float>());
    }

    constexpr int kNumThreadsPerBlockMerge = 1024;
    vllm::topKPerRowDecode<kNumThreadsPerBlockMerge, true, false, true>
        <<<numRows, kNumThreadsPerBlockMerge, topK * sizeof(int32_t), stream>>>(
            outLogitsAux.const_data_ptr<float>(), seqLens.const_data_ptr<int>(),
            indices.mutable_data_ptr<int>(), multipleBlocksPerRowConfig * topK,
            1, static_cast<int>(topK), static_cast<int>(next_n), seqLensIs2D,
            nullptr, multipleBlocksPerRowConfig,
            outIndicesAux.const_data_ptr<int>());
  }
}

#ifdef VLLM_TOPK_DETERMINISTIC_TU
void top_k_per_row_prefill_deterministic(
#else
void top_k_per_row_prefill(
#endif
    const torch::stable::Tensor& logits, const torch::stable::Tensor& rowStarts,
    const torch::stable::Tensor& rowEnds, torch::stable::Tensor& indices,
    int64_t numRows, int64_t stride0, int64_t stride1, int64_t topK) {
#ifndef VLLM_TOPK_DETERMINISTIC_TU
  if (vllm::topKDeterministicTies()) {
    top_k_per_row_prefill_deterministic(logits, rowStarts, rowEnds, indices,
                                        numRows, stride0, stride1, topK);
    return;
  }
#endif
  constexpr int kSortingAlgorithmThreshold = 12288;
  constexpr int kNumThreadsPerBlock = 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  int numInsertionBlocks =
      std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
  vllm::topKPerRowPrefill<kNumThreadsPerBlock, false>
      <<<numInsertionBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
         stream>>>(logits.const_data_ptr<float>(),
                   rowStarts.const_data_ptr<int>(),
                   rowEnds.const_data_ptr<int>(),
                   indices.mutable_data_ptr<int>(), static_cast<int>(stride0),
                   static_cast<int>(stride1), static_cast<int>(topK), 0);

  if (numRows > kSortingAlgorithmThreshold) {
    int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
    vllm::topKPerRowPrefill<kNumThreadsPerBlock, true>
        <<<numRadixBlocks, kNumThreadsPerBlock, topK * sizeof(int32_t),
           stream>>>(
            logits.const_data_ptr<float>(), rowStarts.const_data_ptr<int>(),
            rowEnds.const_data_ptr<int>(), indices.mutable_data_ptr<int>(),
            static_cast<int>(stride0), static_cast<int>(stride1),
            static_cast<int>(topK), kSortingAlgorithmThreshold);
  }
}
