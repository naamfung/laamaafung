#! /bin/sh
# 稀疏 35B-A3B 模型（Qwen3.6-35B-A3B）llama-bench 全面性能测试脚本（POSIX /bin/sh）
# 用途：用各个构建版本自带的 llama-bench.exe，在固定推理参数下，遍历版本
#       去对比不同 --load-mode，以及 GGML_SCHED_PREFETCH_EXPERTS / GGML_CUDA_REGISTER_HOST
#       这两个环境开关（各为独立维度，取 on/off 两态）的性能，并把原始数据落盘供后续分析。
#       对 MoE 稀疏模型，这两个环境开关（尤其是 GGML_SCHED_PREFETCH_EXPERTS）
#       是本次对比的重点。
# 用法：./bench-sparse-35b-a3b.sh 或  ./bench-sparse-35b-a3b.sh <ncmoe 列表>
#       （可选位置参数指定要测的 n-cpu-moe 值，如 "34" 只测 34，或 "30 32 34"；
#         省略时按 NCMOE_MIN..NCMOE_MAX 闭区间全扫）
# 说明：
#   - 每个待测目录需存在 bin/Release/llama-bench.exe，缺失会被跳过。
#   - 若某构建的 llama-bench 不支持当前 -ctv（V cache 类型，如部分非 TurboQuant
#     的构建不接受 turbo4），llama-bench 会立即报 "invalid parameter" 并退出；本脚本
#     会自动改用 CTV_FALLBACK 重跑该组合，并在 summary.tsv / .err 里标注实际 ctv。
#   - llama-bench 用合成 prompt 的长度来测速（不看正文），因此这里不需要
#     真实对话提示词；如需测真实 generation 质量 / 首 token 延迟，请改用 llama-server。
#   - 结果落盘：
#       bench-logs/sparse-35b-a3b/<时间戳>/
#           README.txt            本次运行的可复现配置
#           summary.tsv           每组合: build/load_mode/ncmoe/prefetch/reg_host/ctv/rc/json/err
#           raw/<slug>.json       该组合的 JSON 结果（机读）
#           raw/<slug>.err        stderr 日志（含调用命令 + markdown 可读汇总）

set -u

# ---------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")" && pwd)"

BENCH_REL="bin/Release/llama-bench.exe"

# 待测构建版本目录，用空格分隔（路径不含空格方可直接列于此）
BUILD_DIRS="$ROOT/build-master $ROOT/build-v17 $ROOT/build-v18 $ROOT/build-v19-testing $ROOT/build-v20-testing"

# 模型（稀疏 A3B MoE）；默认用下方路径，可用环境变量 MODEL 覆盖（如 MODEL=<路径> ./bench-sparse-35b-a3b.sh）
: "${MODEL:=C:/WorkModels/Qwen3.6-35B-A3B/LuffyTheFox/Hermes3.6-35B-A3B-Uncensored-Genesis-V12-MTP-APEX-Compact.gguf}"

# llama-bench 固定参数（尽量贴合日常 server 用法）
NGL=99              # 全层 offload 到 GPU
FA=on               # flash attention
CTK=q8_0            # K cache 类型
CTV=turbo4          # V cache 类型（若某构建不支持会自动回退 CTV_FALLBACK）
CTV_FALLBACK=f16    # 回退用 V cache 类型（f16 全构建通用）
THREADS=10          # 运行线程数（与日常 server 一致）
NCMOE_MIN=26         # n-cpu-moe 扫描下界（闭区间）
NCMOE_MAX=36         # n-cpu-moe 扫描上界（闭区间）
BATCH=2048          # prompt eval 模型 batch（MoE 下 batch 可大些，这里保持统一以对比）
UBATCH=256          # ubatch
PROMPTS="128,512"        # prompt 长度序列（稀疏模型 prompt eval 较慢，序列取小）
NGEN="128,256"           # 生成 token 序列
REPS=3               # llama-bench 内部重复次数
# 设备。注意：llama-bench 的 -dev 按后端设备名称逐字匹配（如 CUDA 为 "CUDA0"，
# 大小写敏感，不能照搬 server 的小写 "cuda0"）；默认 auto 会随 -ngl 自动选 GPU。
DEV=auto

# 要对比的 --load-mode（mmap=默认基线 / mlock-ram=读入RAM+mlock / mmap+mlock=mmap+mlock）
LOAD_MODES="mmap mlock-ram mmap+mlock"

# GGML_SCHED_PREFETCH_EXPERTS / GGML_CUDA_REGISTER_HOST 各为独立维度（on/off）
PREFETCH_STATES="on off"
REGISTER_HOST_STATES="on off"

# 由命令行参数或默认区间生成 n-cpu-moe 待测值列表（n-cpu-moe 也是一维组合）
NCMOES="$*"
if [ -z "$NCMOES" ]; then
  _n=$NCMOE_MIN
  while [ "$_n" -le "$NCMOE_MAX" ]; do
    NCMOES="$NCMOES $_n"
    _n=$((_n + 1))
  done
fi

# ---------------------------------------------------------------
# 输出目录（每次运行独立时间戳，避免覆盖）
# ---------------------------------------------------------------
LOGROOT="$ROOT/bench-logs/sparse-35b-a3b"
RUNDIR="$LOGROOT/$(date +%Y%m%d-%H%M%S)"
RAWDIR="$RUNDIR/raw"
mkdir -p "$RAWDIR"

cat > "$RUNDIR/README.txt" <<EOF
本次运行的可复现配置 (sparse-35b-a3b)
=====================================
model       $MODEL
ngl         $NGL
fa          $FA
ctk         $CTK
ctv         $CTV
ctv_fallback $CTV_FALLBACK
threads     $THREADS
ncmoe       $NCMOES
batch       $BATCH
ubatch      $UBATCH
prompts     $PROMPTS
ngen        $NGEN
reps        $REPS
device      $DEV
load_modes   $LOAD_MODES
prefetch     $PREFETCH_STATES
reg_host     $REGISTER_HOST_STATES

说明：llama-bench 以合成 prompt 长度测速，不使用真实对话提示词。
ctv: 某构建不支持 -ctv $CTV 时会自动改用 $CTV_FALLBACK 重跑并在 summary.tsv 标注。
结果：raw/<slug>.json 为该组合的机读结果；raw/<slug>.err 为可读日志。
EOF

echo "结果将写入: $RUNDIR"

# 单次 llama-bench：$1=ctv  $2=json输出  $3=err输出；结果追加进 summary.tsv
run_bench() {
  _ctv="$1"; _jout="$2"; _eout="$3"
  set -- "$bench" -m "$MODEL" \
    -ngl "$NGL" -fa "$FA" \
    -ctk "$CTK" -ctv "$_ctv" \
    -t "$THREADS" -ncmoe "$ncmoe" \
    -b "$BATCH" -ub "$UBATCH" \
    -p "$PROMPTS" -n "$NGEN" -r "$REPS" \
    -dev "$DEV" -lm "$lm" -o json -oe md
  {
    echo "=== ${bname}__lm=${lm}__ncmoe=${ncmoe}__pf=${pf}__rh=${rh} (ctv=$_ctv) ==="
    echo "env: $env_desc"
    printf 'cmd: %s\n' "$@"
    echo "=================================================="
  } >> "$_eout"
  "$@" 2>>"$_eout" >>"$_jout"
  _rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$bname" "$lm" "$ncmoe" "$pf" "$rh" "$_ctv" "$_rc" "${_jout##*/}" "${_eout##*/}" >> "$RUNDIR/summary.tsv"
  return "$_rc"
}

# 初始化汇总表头
printf 'build\tload_mode\tncmoe\tprefetch\treg_host\tctv\trc\tjson\terr\n' > "$RUNDIR/summary.tsv"

# ---------------------------------------------------------------
# 遍历：版本 x load-mode x n-cpu-moe x prefetch x reg_host
# ---------------------------------------------------------------
for build in $BUILD_DIRS; do
  bench="$build/$BENCH_REL"
  if [ ! -f "$bench" ]; then
    echo "[SKIP] 缺少 llama-bench: $bench"
    continue
  fi
  bname="$(basename "$build")"
  bname="${bname#build-}"   # 构建版本名

  for lm in $LOAD_MODES; do
    for pf in $PREFETCH_STATES; do
      for rh in $REGISTER_HOST_STATES; do
        for ncmoe in $NCMOES; do
          slug="${bname}__lm=${lm}__ncmoe=${ncmoe}__pf=${pf}__rh=${rh}"
          json_out="$RAWDIR/$slug.json"
          err_out="$RAWDIR/$slug.err"

          # 设置待对比的环境变量（pf/rh 各为独立维度）
          GGML_SCHED_PREFETCH_EXPERTS=1; export GGML_SCHED_PREFETCH_EXPERTS
          if [ "$pf" = "off" ]; then
            unset GGML_SCHED_PREFETCH_EXPERTS
          fi
          GGML_CUDA_REGISTER_HOST=1;     export GGML_CUDA_REGISTER_HOST
          if [ "$rh" = "off" ]; then
            unset GGML_CUDA_REGISTER_HOST
          fi
          env_desc="GGML_SCHED_PREFETCH_EXPERTS=${GGML_SCHED_PREFETCH_EXPERTS:-off} GGML_CUDA_REGISTER_HOST=${GGML_CUDA_REGISTER_HOST:-off}"

          used_ctv="$CTV"
          echo "[RUN] build=$bname load_mode=$lm ncmoe=$ncmoe prefetch=$pf reg_host=$rh ctk=$CTK ctv=$used_ctv"
          start=$(date +%s)
          run_bench "$used_ctv" "$json_out" "$err_out"; rc=$?
          end=$(date +%s)
          # 主 ctv 不被该构建支持时，改用回退类型重跑一次
          if [ "$rc" -ne 0 ] && [ -n "$CTV_FALLBACK" ] && grep -q "invalid parameter" "$err_out"; then
            used_ctv="$CTV_FALLBACK"
            json_out="$RAWDIR/${slug}__ctv=${used_ctv}.json"
            err_out="$RAWDIR/${slug}__ctv=${used_ctv}.err"
            echo "  [FALLBACK] build=$bname 不支持 -ctv $CTV，改测 -ctv $used_ctv"
            start=$(date +%s)
            run_bench "$used_ctv" "$json_out" "$err_out"; rc=$?
            end=$(date +%s)
          fi
          echo "  rc=$rc  ncmoe=$ncmoe  ctk=$CTK ctv=$used_ctv  ${json_out##*/}  ${err_out##*/}  (${start}s -> ${end}s)"
        done
      done
    done
  done
done

echo "完成。结果目录: $RUNDIR"