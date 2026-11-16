// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

// No borrowed pointers survive a call. Tensor handles retain the mmap storage;
// callers must not mutate shard contents or metadata concurrently with gather.
class PleInt4Table : public torch::CustomClassHolder {
 public:
  PleInt4Table(std::vector<at::Tensor> packed,
               std::vector<at::Tensor> scales, int64_t rows_per_shard)
      : packed_(std::move(packed)),
        scales_(std::move(scales)),
        rows_per_shard_(rows_per_shard) {
    TORCH_CHECK(rows_per_shard_ > 0,
                "PleInt4Table: rows_per_shard must be positive");
    TORCH_CHECK(!packed_.empty() && packed_.size() == scales_.size(),
                "PleInt4Table: nonempty matching shard lists required");
    TORCH_CHECK(packed_.size() <=
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max() /
                                          rows_per_shard_),
                "PleInt4Table: total row count overflow");
    total_rows_ = static_cast<int64_t>(packed_.size()) * rows_per_shard_;
    check_tensor(packed_[0], at::kByte, 2, "packed");
    const int64_t packed_width = packed_[0].size(1);
    TORCH_CHECK(packed_width > 0 && packed_width % 8 == 0 &&
                    packed_width <= std::numeric_limits<int64_t>::max() / 2,
                "PleInt4Table: width must be positive and divisible by 16");
    width_ = packed_width * 2;
    validate_shards();
  }

  void gather(const at::Tensor& ids, const at::Tensor& out) const {
    check_tensor(ids, at::kLong, 1, "ids");
    check_tensor(out, at::kFloat, 2, "out");
    TORCH_CHECK(out.size(0) == ids.size(0) && out.size(1) == width_,
                "PleInt4Table: out shape must match ids count and table width");

    // Recheck handles rather than trusting constructor-time sizes/pointers:
    // resize_, set_, or storage resizing through another alias may change them.
    validate_shards();
    check_no_overlap(out, ids);
    for (size_t shard = 0; shard < packed_.size(); ++shard) {
      check_no_overlap(out, packed_[shard]);
      check_no_overlap(out, scales_[shard]);
    }

    const int64_t* indices = ids.const_data_ptr<int64_t>();
    const int64_t count = ids.size(0);
    // Validate every ID before the first write, including late invalid IDs.
    for (int64_t i = 0; i < count; ++i) {
      TORCH_CHECK(indices[i] >= 0 && indices[i] < total_rows_,
                  "PleInt4Table: row ID out of bounds: ", indices[i]);
    }

    float* output = out.data_ptr<float>();
    const int64_t packed_width = width_ / 2;
    const int64_t groups = width_ / 16;
    for (int64_t i = 0; i < count; ++i) {
      const int64_t shard = indices[i] / rows_per_shard_;
      const int64_t row = indices[i] % rows_per_shard_;
      const uint8_t* packed = packed_[shard].const_data_ptr<uint8_t>() +
                              row * packed_width;
      const c10::Half* scales = scales_[shard].const_data_ptr<c10::Half>() +
                                row * groups;
      float* dest = output + i * width_;
      for (int64_t group = 0; group < groups; ++group) {
        const float scale = static_cast<float>(scales[group]);
        for (int64_t byte = 0; byte < 8; ++byte) {
          const uint8_t bits = packed[group * 8 + byte];
          // Promote the half scale first: multiplication is FP32, not FP16.
          // Do not fold away multiplication by zero (signed zero/NaN matter).
          dest[group * 16 + byte * 2] =
              static_cast<float>(static_cast<int>(bits & 15) - 8) * scale;
          dest[group * 16 + byte * 2 + 1] =
              static_cast<float>(static_cast<int>(bits >> 4) - 8) * scale;
        }
      }
    }
  }

 private:
  static void check_tensor(const at::Tensor& tensor, at::ScalarType dtype,
                           int64_t dimensions, const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.device().is_cpu() &&
                    tensor.layout() == at::kStrided &&
                    tensor.scalar_type() == dtype &&
                    tensor.dim() == dimensions && tensor.is_contiguous() &&
                    !tensor.is_conj() && !tensor.is_neg(),
                "PleInt4Table: ", name,
                " must have the required CPU dtype, rank and contiguous layout");
    const auto bytes = tensor.nbytes();
    const auto storage_bytes = tensor.storage().nbytes();
    const auto offset = static_cast<uint64_t>(tensor.storage_offset());
    const auto item_bytes = tensor.element_size();
    TORCH_CHECK(offset <= storage_bytes / item_bytes &&
                    bytes <= storage_bytes - offset * item_bytes,
                "PleInt4Table: ", name, " storage is smaller than its metadata");
  }

  void validate_shards() const {
    for (size_t shard = 0; shard < packed_.size(); ++shard) {
      check_tensor(packed_[shard], at::kByte, 2, "packed");
      check_tensor(scales_[shard], at::kHalf, 2, "scales");
      TORCH_CHECK(packed_[shard].size(0) == rows_per_shard_ &&
                      packed_[shard].size(1) == width_ / 2 &&
                      scales_[shard].size(0) == rows_per_shard_ &&
                      scales_[shard].size(1) == width_ / 16,
                  "PleInt4Table: shard shape must match rows_per_shard and "
                  "group16 table width");
    }
  }

  static void check_no_overlap(const at::Tensor& out,
                               const at::Tensor& input) {
    if (out.numel() == 0 || input.numel() == 0) return;
    // Compare actual byte spans, not Storage identities: from_numpy/from_blob
    // can create distinct Storage objects covering the same backing memory.
    const auto a = reinterpret_cast<uintptr_t>(out.const_data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(input.const_data_ptr());
    const bool overlap = a <= b ? b - a < out.nbytes()
                                : a - b < input.nbytes();
    TORCH_CHECK(!overlap, "PleInt4Table: out must not overlap ids or shards");
  }

  std::vector<at::Tensor> packed_;
  std::vector<at::Tensor> scales_;
  int64_t rows_per_shard_;
  int64_t total_rows_;
  int64_t width_;
};

}  // namespace

// Standalone CPU registration as well as ROCm legacy _C: no Python entrypoint.
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.class_<PleInt4Table>("PleInt4Table")
      .def(torch::init<std::vector<at::Tensor>, std::vector<at::Tensor>, int64_t>())
      .def("gather", &PleInt4Table::gather);
}
