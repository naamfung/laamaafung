// TurboQuant2/3 TCQ CUDA flash attention vec kernel instantiation

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ);
