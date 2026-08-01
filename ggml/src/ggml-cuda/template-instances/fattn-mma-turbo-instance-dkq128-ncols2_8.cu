// Hand-created turbo MMA prefill GQA instances (ncols2 == 8, gqa_ratio >= 8).
// Do NOT run generate_cu_files.py over it - that script deletes all *.cu including
// the hand-created turbo instances.
//
// Only ncols1 = 16 (ncols = 128) is compiled to keep shared_Q within limits
// across all architectures and head dims. See fattn-mma-turbo.cuh for details.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(128, 128, 16, 8, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_CASE(128, 128, 16, 8, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_CASE(128, 128, 16, 8, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
