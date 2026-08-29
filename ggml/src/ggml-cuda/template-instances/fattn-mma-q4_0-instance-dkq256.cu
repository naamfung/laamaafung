// Hand-created Q4_0 MMA decode instances (inline-dequant Q4_0 KV path).
// Do NOT run generate_cu_files.py over the template-instances dir.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-q4_0.cuh"

DECL_FATTN_MMA_Q4_0_CASE(256, 256, 1, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_MMA_Q4_0_CASE(256, 256, 2, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_MMA_Q4_0_CASE(256, 256, 4, 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);