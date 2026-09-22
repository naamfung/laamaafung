/*
 * TurboQuant TCQ (Trellis-Coded Quantization) CUDA kernels
 * Based on: spiritbuun/llama-cpp-turboquant-cuda
 *
 * Implements GGML_TYPE_TURBO3_TCQ (3-bit TCQ, 512-state Viterbi, 3.25 bpv)
 *         and GGML_TYPE_TURBO2_TCQ (2-bit TCQ, 256-state Viterbi, 2.25 bpv)
 *
 * Codebooks: product_mono GLA training on Qwen3.5-27B FWHT-rotated KV activations.
 * Encode: Viterbi optimal path with right-shift bitshift trellis.
 * Decode: state_t = read_N_bits(qs, t*k), recon_t = codebook[state_t] * norm.
 *
 * If you copy these codebooks, credit spiritbuun!
 */

#pragma once

#include "common.cuh"
#include "turbo-quant.cuh"

// ---- Quantization ratios for dequantize_block template (stride = half block) ----
#define QR_TURBO3_TCQ 2
#define QR_TURBO2_TCQ 2

// =====================================================================================
// TCQ norm alpha constants
// =====================================================================================
// K alpha = 1.0 (no scaling), V alpha = 1.04 (KLD-optimal at 2K context).
// Override via TURBO_TCQ_ALPHA (K) and TURBO_TCQ_ALPHA_V (V) env vars.
// Alpha is applied at encode time (baked into fp16 norm).
static __constant__ float d_tcq_norm_alpha   = 1.0f;
static __constant__ float d_tcq_norm_alpha_v = 1.04f;

// =====================================================================================
// TURBO3_TCQ: 3-bit TCQ codebook (512-state bitshift trellis, k=3, L=9)
// =====================================================================================
// product_mono/iter080, CUDA GLA product-aware training, 100 iters on
// Qwen3.5-27B FWHT-rotated KV activations.
// Decode: state_t = read_9_bits(qs, t*3)

static __constant__ float d_turbo3_tcq_codebook[512] = {
    -0.14559399f, -0.09062801f, -0.054925077f, -0.03699251f, -0.006363985f, +0.026264573f, +0.067378916f, +0.121981815f,
    -0.18648055f, -0.106522456f, -0.052047577f, -0.011695214f, +0.021953275f, +0.059698727f, +0.09831437f, +0.16083933f,
    -0.16390342f, -0.12639847f, -0.09513180f, -0.05938352f, -0.028396897f, +0.005973862f, +0.049104784f, +0.11334257f,
    -0.25952467f, -0.079778515f, -0.036024813f, +0.0003641268f, +0.031858794f, +0.073280424f, +0.11835553f, +0.19738495f,
    -0.14218009f, -0.10224814f, -0.062498566f, -0.027066832f, +0.00393002f, +0.04069300f, +0.08257346f, +0.14548601f,
    -0.18673635f, -0.13438253f, -0.088401966f, -0.05205436f, -0.02032501f, +0.012399545f, +0.05127183f, +0.10316186f,
    -0.10807011f, -0.065903045f, -0.032206114f, -0.0062006037f, +0.020679146f, +0.04422085f, +0.08313074f, +0.16821936f,
    -0.22979105f, -0.14431947f, -0.07689272f, -0.02755307f, +0.009225173f, +0.046684854f, +0.08834142f, +0.13766693f,
    -0.22114082f, -0.12612148f, -0.06890522f, -0.016128855f, +0.03691900f, +0.08474852f, +0.14940020f, +0.23229980f,
    -0.14933491f, -0.099693604f, -0.06738499f, -0.037100967f, -0.009332986f, +0.023535024f, +0.060272533f, +0.109464675f,
    -0.20200425f, -0.07398328f, -0.038700905f, -0.01714807f, +0.011161969f, +0.04528101f, +0.08902637f, +0.19573534f,
    -0.16645233f, -0.124482535f, -0.089342155f, -0.04427387f, -0.007353691f, +0.028033108f, +0.066108435f, +0.15552913f,
    -0.22295763f, -0.059887577f, -0.018804537f, +0.020141022f, +0.059682943f, +0.097920544f, +0.14080113f, +0.25698325f,
    -0.14248224f, -0.089685425f, -0.050101686f, -0.017257255f, +0.011412255f, +0.040830314f, +0.07400172f, +0.11997315f,
    -0.18649384f, -0.113997504f, -0.067775466f, -0.033394672f, +0.006586988f, +0.05312057f, +0.10433043f, +0.22344802f,
    -0.16138338f, -0.108194515f, -0.07600300f, -0.05135381f, -0.023365447f, +0.0087320795f, +0.045431953f, +0.09113002f,
    -0.12630440f, -0.07225349f, -0.032280035f, +0.0029231994f, +0.019239848f, +0.05081419f, +0.077840395f, +0.121695265f,
    -0.08928155f, -0.044983763f, -0.009889568f, +0.020831043f, +0.05684458f, +0.09409702f, +0.13867535f, +0.19084482f,
    -0.14182915f, -0.11380146f, -0.06904074f, -0.002002765f, +0.034864165f, +0.070399575f, +0.11403063f, +0.15394832f,
    -0.10876417f, -0.056122433f, -0.02267638f, +0.011113975f, +0.039639056f, +0.074084364f, +0.10155376f, +0.12540291f,
    -0.17693359f, -0.13940524f, -0.10049578f, -0.06796275f, -0.036915872f, +0.00062823476f, +0.042142134f, +0.17906062f,
    -0.09253492f, -0.04290128f, -0.006311852f, +0.023908244f, +0.049849935f, +0.078770354f, +0.10818172f, +0.15166481f,
    -0.12429565f, -0.07392063f, -0.029114135f, +0.0059440783f, +0.042675965f, +0.08425635f, +0.13836108f, +0.18634140f,
    -0.11795639f, -0.07033707f, -0.034163877f, -0.0008773357f, +0.03334606f, +0.07188203f, +0.12216825f, +0.17097956f,
    -0.18718453f, -0.14090346f, -0.097799584f, -0.059522875f, -0.019208657f, +0.03079176f, +0.09334672f, +0.15811224f,
    -0.27198875f, -0.16546582f, -0.11433405f, -0.06933013f, -0.04026183f, -0.0061146915f, +0.029263576f, +0.07322499f,
    -0.18471734f, -0.102074504f, -0.06492570f, -0.034418534f, -0.009636157f, +0.023043344f, +0.05751496f, +0.09905984f,
    -0.22826399f, -0.15946552f, -0.09913176f, -0.06585259f, -0.03252090f, +0.001313243f, +0.03556729f, +0.21612854f,
    -0.13243781f, -0.087299444f, -0.049820945f, -0.016216082f, +0.01799807f, +0.057916876f, +0.09001349f, +0.13221787f,
    -0.19516511f, -0.120894566f, -0.076130204f, -0.051442243f, -0.029535033f, -0.0020043184f, +0.029452588f, +0.075566076f,
    -0.27272871f, -0.15841717f, -0.105432935f, -0.06792948f, -0.024532158f, +0.014960791f, +0.054415092f, +0.101517834f,
    -0.21153601f, -0.15015371f, -0.08676790f, -0.04414934f, -0.0042129597f, +0.033762872f, +0.07589151f, +0.12768789f,
    -0.090428725f, -0.037582967f, +0.0013173596f, +0.03900247f, +0.06840049f, +0.116906695f, +0.16584939f, +0.25382105f,
    -0.13446195f, -0.07865091f, -0.039625354f, -0.0028398742f, +0.03019514f, +0.06799379f, +0.11850997f, +0.17521496f,
    -0.11350345f, -0.058599845f, -0.017512511f, +0.019431496f, +0.055897832f, +0.093173414f, +0.14820710f, +0.22092152f,
    -0.15165758f, -0.08869354f, -0.04974287f, -0.01705474f, +0.013134752f, +0.04367713f, +0.07733791f, +0.12430801f,
    -0.09329869f, -0.04673005f, -0.00045857552f, +0.042781368f, +0.07802363f, +0.11887439f, +0.16250038f, +0.28612965f,
    -0.12571070f, -0.07786012f, -0.03843933f, -0.0075433915f, +0.025822964f, +0.066053316f, +0.12021536f, +0.18341768f,
    -0.16079275f, -0.04921760f, -0.006114644f, +0.026215268f, +0.05699377f, +0.09813471f, +0.16080129f, +0.23786584f,
    -0.09980837f, -0.048535258f, -0.0096120685f, +0.025387142f, +0.05979822f, +0.09875251f, +0.14474337f, +0.20324114f,
    -0.15846540f, -0.09938028f, -0.061492465f, -0.03523542f, -0.0061364113f, +0.024916094f, +0.06037314f, +0.106796466f,
    -0.20557843f, -0.123237535f, -0.07734871f, -0.044549115f, -0.017114898f, +0.01616654f, +0.049574375f, +0.092319444f,
    -0.19221115f, -0.14642999f, -0.091701314f, -0.055265956f, -0.021026207f, +0.017720066f, +0.05786183f, +0.110154524f,
    -0.09956386f, -0.03870283f, +0.003052007f, +0.034851722f, +0.06256365f, +0.09628840f, +0.13979156f, +0.16582295f,
    -0.18026546f, -0.12448310f, -0.07424377f, -0.03954519f, -0.01221123f, +0.028641058f, +0.100819774f, +0.18240699f,
    -0.21520759f, -0.15573645f, -0.09820838f, -0.051450998f, -0.012993679f, +0.021135861f, +0.058727216f, +0.105848536f,
    -0.11207385f, -0.08335689f, -0.048542723f, -0.023198519f, +0.0039304253f, +0.037778318f, +0.07813917f, +0.13106476f,
    -0.17849164f, -0.120988995f, -0.078016765f, -0.043093704f, -0.016565649f, +0.015182641f, +0.050754096f, +0.09595712f,
    -0.22132620f, -0.13407415f, -0.065785654f, -0.013291034f, +0.032098345f, +0.07478225f, +0.12431934f, +0.19174045f,
    -0.095454164f, -0.051898945f, -0.015116375f, -0.012596778f, +0.018636847f, +0.05006925f, +0.087654814f, +0.13754296f,
    -0.15254061f, -0.09576059f, -0.052086458f, -0.01596074f, +0.017607626f, +0.04778498f, +0.08950204f, +0.14901252f,
    -0.26057002f, -0.12472382f, -0.074396215f, -0.03764066f, +0.0011168446f, +0.061569117f, +0.10793752f, +0.19771695f,
    -0.08661132f, -0.045195263f, -0.016098704f, +0.012780116f, +0.040476497f, +0.074102715f, +0.074102715f, +0.12635531f,
    -0.14047913f, -0.059587404f, -0.016261123f, +0.019801628f, +0.053541403f, +0.096650146f, +0.15005490f, +0.21051759f,
    -0.22986396f, -0.11964334f, -0.07266585f, -0.026522418f, +0.018169926f, +0.058630653f, +0.100647695f, +0.15919648f,
    -0.13251697f, -0.077567816f, -0.042766172f, -0.011389967f, +0.01831755f, +0.05304656f, +0.09620367f, +0.15567583f,
    -0.119819686f, -0.06772876f, -0.028123451f, +0.00876240f, +0.014405836f, +0.048829112f, +0.08422175f, +0.13823749f,
    -0.16379014f, -0.08956941f, -0.041652776f, +0.008921398f, +0.05473602f, +0.10037984f, +0.16022855f, +0.23457925f,
    -0.115844205f, -0.05939626f, -0.020390417f, +0.01374377f, +0.044976473f, +0.07873563f, +0.12207942f, +0.18412720f,
    -0.19048831f, -0.07587487f, -0.03220580f, -0.00011795067f, +0.02721784f, +0.04380719f, +0.07886723f, +0.13193911f,
    -0.13935551f, -0.092902906f, -0.052706074f, -0.017797327f, +0.015312965f, +0.056098964f, +0.11203423f, +0.24448302f,
    -0.17986591f, -0.10738580f, -0.06376371f, -0.026595421f, +0.00842492f, +0.04272362f, +0.08608052f, +0.15240218f,
    -0.10953678f, -0.057022586f, -0.012483291f, +0.024463262f, +0.06076792f, +0.09776234f, +0.12983681f, +0.18648379f,
    -0.16471463f, -0.089491285f, -0.037574016f, +0.004444791f, +0.039293647f, +0.07845859f, +0.12893885f, +0.23508036f
};

// =====================================================================================
// TURBO2_TCQ: 2-bit TCQ codebook (256-state bitshift trellis, k=2, L=8)
// =====================================================================================
// product_mono/iter090, CUDA GLA product-aware training, 100 iters on
// Qwen3.5-27B FWHT-rotated KV activations.
// Decode: state_t = read_8_bits(qs, t*2)

static __constant__ float d_turbo2_tcq_codebook[256] = {
    -0.18030643f, -0.11009848f, -0.04742626f, +0.02894132f, -0.10523465f, -0.031312924f, +0.031491395f, +0.12263535f,
    -0.15660362f, -0.055477407f, +0.0046675834f, +0.06166081f, -0.07506216f, -0.016963918f, +0.043737844f, +0.116496615f,
    -0.08632783f, -0.022493735f, +0.041032985f, +0.10660284f, -0.06274858f, -0.0036939639f, +0.02095157f, +0.07539709f,
    -0.09802641f, -0.008419088f, +0.059072323f, +0.17311879f, -0.093109086f, -0.02654333f, +0.014827672f, +0.07793592f,
    -0.031235758f, +0.01271591f, +0.08752262f, +0.17246453f, -0.14595252f, -0.07227624f, +0.013628688f, +0.08131674f,
    -0.036909282f, +0.0018896917f, +0.05209119f, +0.12407892f, -0.13689458f, -0.06054520f, +0.0064648795f, +0.07551241f,
    -0.18980840f, -0.110128626f, -0.046503957f, +0.026387159f, -0.034967307f, +0.04810357f, +0.072072044f, +0.14355458f,
    -0.10182410f, -0.02907887f, +0.014033012f, +0.083419636f, -0.056140676f, +0.008405868f, +0.066070884f, +0.14037225f,
    -0.117427245f, -0.047159385f, +0.016928354f, +0.08142885f, -0.029359628f, +0.045608785f, +0.10559447f, +0.20061271f,
    -0.040425077f, +0.029068163f, +0.08408973f, +0.13628258f, -0.16633821f, -0.10711727f, -0.04196669f, +0.027895834f,
    -0.0054065837f, +0.058898676f, +0.12688550f, +0.18268861f, -0.16287325f, -0.11218357f, -0.07165227f, -0.009524379f,
    -0.24026902f, -0.073219374f, -0.0005165726f, +0.05959821f, -0.05532953f, +0.027044486f, +0.09425678f, +0.15356481f,
    -0.14381111f, -0.10563502f, -0.037867088f, +0.023611993f, -0.03624307f, +0.049588434f, +0.12192037f, +0.23462485f,
    -0.14990251f, -0.09659304f, -0.05886742f, +0.014878461f, -0.009889551f, +0.06910514f, +0.12120181f, +0.22596690f,
    -0.08290075f, -0.009009629f, +0.066151775f, +0.12188313f, -0.11591514f, -0.06952189f, -0.031633306f, +0.023740824f,
    -0.20510401f, -0.103369795f, +0.09148037f, +0.17268716f, -0.16597997f, -0.09207068f, -0.032810967f, +0.024847647f,
    -0.02487482f, +0.049298953f, +0.09624215f, +0.14217524f, -0.18418685f, -0.10147012f, -0.05841265f, +0.008057022f,
    -0.14269894f, -0.092456274f, -0.026881337f, +0.049792137f, -0.019881032f, +0.030333601f, +0.09736802f, +0.17764080f,
    -0.19579841f, -0.114739306f, -0.026823774f, +0.07466014f, -0.09001050f, -0.041468445f, +0.028473806f, +0.08870695f,
    -0.019396419f, +0.042828932f, +0.10885327f, +0.13335012f, -0.15005013f, -0.074581385f, -0.028608415f, +0.03848942f,
    -0.09687270f, -0.057059396f, +0.0077843578f, +0.06302297f, -0.23247094f, -0.14509225f, -0.032651436f, +0.027010715f,
    -0.047595482f, +0.06280303f, +0.114691675f, +0.17124057f, -0.21092793f, -0.13704823f, -0.07340412f, +0.0039013291f,
    -0.062834196f, +0.012601906f, +0.012601906f, +0.08721347f, -0.13256435f, -0.024173854f, +0.07723171f, +0.14801070f,
    -0.06471605f, -0.0017903054f, -0.0017903054f, +0.058302354f, -0.09731802f, -0.03400696f, +0.02762442f, +0.08986137f,
    -0.08288722f, -0.019051429f, +0.045709886f, +0.15211061f, -0.09507891f, -0.015612489f, +0.025347246f, +0.087257534f,
    -0.066236064f, -0.0047936034f, +0.06386274f, +0.15401669f, -0.105809286f, -0.051802177f, +0.01073050f, +0.08292137f,
    -0.11884470f, -0.04404144f, +0.02550729f, +0.02550729f, -0.01731189f, +0.062161792f, +0.12127554f, +0.21981733f,
    -0.17066145f, -0.11660990f, -0.049425896f, +0.021293938f, -0.04711412f, +0.026577346f, +0.055197213f, +0.12541275f,
    -0.028268812f, +0.015206398f, +0.09002519f, +0.12699963f, -0.10059831f, -0.026676945f, +0.059903253f, +0.13054545f,
    -0.09582803f, -0.033371232f, +0.010346129f, +0.066766635f, -0.09964944f, -0.028686784f, +0.021184925f, +0.09120017f,
    -0.16957201f, -0.07594450f, +0.04172865f, +0.18313301f, -0.051526368f, +0.011877304f, +0.011877304f, +0.07956263f,
    -0.13432936f, -0.05269006f, +0.03536416f, +0.117640756f, -0.022776067f, +0.042032316f, +0.10472976f, +0.18042557f
};

// =====================================================================================
// TURBO3_TCQ: Viterbi encoder kernel (3-bit, 512 states, right-shift trellis)
// =====================================================================================
// 512 threads per block (one per trellis state), one block per 128-element group.
// Double-buffered cost arrays + global memory backtrace (128 syncs/group).

template<typename idx_t>
static __global__ void __launch_bounds__(512, 1) k_set_rows_turbo3_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf, const int64_t group_offset,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int64_t group = blockIdx.x + group_offset;
    if (group >= ne_total_groups) return;

    const int sid = threadIdx.x; // state index 0..511

    // Compute source and destination pointers
    const int64_t i_base = group * QK_TURBO3_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_tcq * dst_blk = (block_turbo3_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                                  + (i00 / QK_TURBO3_TCQ);

    // Shared memory layout (~5KB):
    // x[128]     : rotated+normalized input (reused as outputs[] after Viterbi)
    // cost[512]  : path costs buffer A (reused for reductions)
    // cost_b[512]: path costs buffer B (double-buffering eliminates 2/3 of syncs)
    // Backtrace: bt_buf in global memory, 128x512 bytes per block
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ float cost_b[512];
    __shared__ int warp_min_idx[16];
    __shared__ float warp_min_cost[16];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    // InnerQ: calibrate on original (unscaled) values
    if (d_innerq_calibrating && sid < 128) {
        atomicAdd(&d_innerq_sq_accum[sid], x[sid] * x[sid]);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    // InnerQ: apply channel scale (only when active)
    if (d_innerq_active && sid < 128) {
        x[sid] *= d_innerq_scale[sid];
    }
    __syncthreads();

    // Norm reduction (parallel over 512 threads, only first 128 contribute)
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT: signs1 -> butterfly -> signs2 (shared memory, parallel)
    if (sid < 128) x[sid] *= TURBO_WHT_SIGNS1[sid];
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        if (sid < 64) {
            int j = (sid / h) * (2 * h) + (sid % h);
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * TURBO_WHT_SIGNS2[sid];
    __syncthreads();

    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: double-buffered cost (1 sync/step)
    // Backtrace in global memory: byte-packed, no nibble conflicts
    uint8_t * bt = bt_buf + (int64_t)blockIdx.x * (128 * 512);
    cost[sid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        // Double-buffer: even steps read cost/write cost_b, odd steps read cost_b/write cost
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis: ns = (prev >> 3) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 3) | p, for p = 0..7
        int base_prev = (sid & 0x3F) << 3;
        float dist = xt - d_turbo3_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 8; p++) {
            float c = cost_rd[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        cost_wr[sid] = best + dist;
        bt[t * 512 + sid] = (uint8_t)best_p;
        __syncthreads();
    }
    // After 128 steps (even count): final costs are in cost[]

    // Warp argmin over 512 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid == 0) {
        float best = warp_min_cost[0];
        int best_idx = warp_min_idx[0];
        for (int w = 1; w < 16; w++) {
            if (warp_min_cost[w] < best) { best = warp_min_cost[w]; best_idx = warp_min_idx[w]; }
        }
        shared_initial_state = best_idx;
    }
    __syncthreads();

    // Backtrack (inherently sequential, reads global bt)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = bt[t * 512 + state];
            state = ((state & 0x3F) << 3) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Parallel recon norm: t>=2 can compute state directly from 3 outputs (3 shifts of 3 = 9 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 2) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 3) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 2] & 0x7)
                      | (((int)outputs[sid - 1] & 0x7) << 3)
                      | (((int)outputs[sid]     & 0x7) << 6);
        }
        float c = d_turbo3_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= innerq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Serial bitpack -- byte-alignment prevents parallel atomicOr
    if (sid == 0) {
        for (int j = 0; j < 49; j++) dst_blk->qs[j] = 0;
        dst_blk->qs[0] = (uint8_t)((shared_initial_state >> 3) & 0x3F);
        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 3;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x7;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
            if (bit_off > 5) dst_blk->qs[byte_idx + 1] |= (uint8_t)(out >> (8 - bit_off));
        }
        dst_blk->norm = __float2half(corrected_norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// =====================================================================================
// TURBO3_TCQ: dequantize (for non-FA get_rows path)
// =====================================================================================

static __device__ __forceinline__
void dequantize_turbo3_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 9-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.x = d_turbo3_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.y = d_turbo3_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBO2_TCQ: Viterbi encoder kernel (2-bit, 256 states, right-shift trellis)
// =====================================================================================
// 256 threads per block (one per trellis state), one block per 128-element group.
// Double-buffered cost arrays + global memory backtrace (128 syncs/group).

template<typename idx_t>
static __global__ void __launch_bounds__(256, 1) k_set_rows_turbo2_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf, const int64_t group_offset,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int iq_is_k,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x + group_offset;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    // Compute source and destination pointers
    const int64_t i_base = int64_t(grp) * QK_TURBO2_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_tcq * dst_blk = (block_turbo2_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBO2_TCQ);

    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ float cost_b[256];   // double-buffering for Viterbi
    __shared__ int warp_min_idx[8];
    __shared__ float warp_min_cost[8];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    // InnerQ: calibrate on original (unscaled) values
    if (d_innerq_calibrating && sid < 128) {
        atomicAdd(&d_innerq_sq_accum[sid], x[sid] * x[sid]);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    // InnerQ: apply channel scale (only when active)
    if (d_innerq_active && sid < 128) {
        x[sid] *= d_innerq_scale[sid];
    }
    __syncthreads();

    // Norm reduction (parallel over 256 threads, only first 128 contribute)
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT: signs1 -> butterfly -> signs2 (shared memory, parallel)
    if (sid < 128) x[sid] *= TURBO_WHT_SIGNS1[sid];
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        if (sid < 64) {
            int j = (sid / h) * (2 * h) + (sid % h);
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * TURBO_WHT_SIGNS2[sid];
    __syncthreads();

    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: double-buffered cost (1 sync/step)
    uint8_t * bt = bt_buf + (int64_t)blockIdx.x * (128 * 256);
    cost[sid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis (k=2, L=8): ns = (prev >> 2) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 2) | p, for p = 0..3
        int base_prev = (sid & 0x3F) << 2;
        float dist = xt - d_turbo2_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 4; p++) {
            float c = cost_rd[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        cost_wr[sid] = best + dist;
        bt[t * 256 + sid] = (uint8_t)best_p;
        __syncthreads();
    }
    // After 128 steps (even count): final costs in cost[]

    // Warp argmin over 256 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid == 0) {
        float best = warp_min_cost[0];
        int best_idx = warp_min_idx[0];
        for (int w = 1; w < 8; w++) {
            if (warp_min_cost[w] < best) { best = warp_min_cost[w]; best_idx = warp_min_idx[w]; }
        }
        shared_initial_state = best_idx;
    }
    __syncthreads();

    // Backtrack (inherently sequential, reads global bt)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = bt[t * 256 + state];
            state = ((state & 0x3F) << 2) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Parallel recon norm: t>=3 can compute state directly from 4 outputs (4 shifts of 2 = 8 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 3) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 2) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 3] & 0x3)
                      | (((int)outputs[sid - 2] & 0x3) << 2)
                      | (((int)outputs[sid - 1] & 0x3) << 4)
                      | (((int)outputs[sid]     & 0x3) << 6);
        }
        float c = d_turbo2_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= iq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Serial bitpack -- byte-alignment prevents parallel atomicOr
    if (sid == 0) {
        for (int j = 0; j < 33; j++) dst_blk->qs[j] = 0;
        dst_blk->qs[0] = (uint8_t)((shared_initial_state >> 2) & 0x3F);
        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 2;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x3;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
        }
        dst_blk->norm = __float2half(corrected_norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// =====================================================================================
// TURBO2_TCQ: dequantize (for non-FA get_rows path)
// =====================================================================================

static __device__ __forceinline__
void dequantize_turbo2_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 8-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.x = d_turbo2_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.y = d_turbo2_tcq_codebook[state] * norm;
    }
}
