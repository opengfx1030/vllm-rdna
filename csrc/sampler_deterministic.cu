// The top-k kernels of sampler.cu with tied logits resolved to the lowest
// stored indices (VLLM_RDNA2_MOE_DETERMINISTIC=1): sampler.cu compiled a second
// time with VLLM_TOPK_DETERMINISTIC_TU defined, which exports
// top_k_per_row_{decode,prefill}_deterministic and nothing else. Kept out of
// sampler.cu so the default kernels stay exactly as they were; see the note at
// the top of that file.
#define VLLM_TOPK_DETERMINISTIC_TU
#include "sampler.cu"
