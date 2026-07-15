#!/bin/sh
# 用法: ./build.sh [--clean] [--ui] [--no-ccache] [--jobs N] [target...]
#   --clean     删除 build/ 后全量编译
#   --ui        启用界面编译 (LLAMA_BUILD_UI=ON)
#   --no-ccache 禁用 ccache (MSVC + ccache 偶发崩溃时使用)
#   --jobs N    并行任务数 (默认 4, 避免高并行导致 cl.exe access violation 崩溃)
#   target      可选 cmake 目标, 如 llama-server; 不指定则编译全部

set -e

CLEAN=0
BUILD_UI=OFF
USE_CCACHE=ON
JOBS=4
TARGETS=""

# 手动解析参数 (POSIX sh 无 getopts 长选项支持)
i=0
while [ $i -lt $# ]; do
    i=$((i + 1))
    eval "arg=\${$i}"
    case "$arg" in
        --clean)     CLEAN=1 ;;
        --ui)        BUILD_UI=ON ;;
        --no-ccache) USE_CCACHE=OFF ;;
        --jobs)
            i=$((i + 1))
            eval "JOBS=\${$i}"
            ;;
        --jobs=*)
            JOBS="${arg#--jobs=}"
            ;;
        *)           TARGETS="$TARGETS $arg" ;;
    esac
done

# 去掉前导空格
TARGETS="${TARGETS# }"

if [ "$CLEAN" -eq 1 ]; then
    rm -rf build
fi

# CMAKE_SUPPRESS_REGENERATION=ON: 禁用 VS 生成器的 ZERO_CHECK (Checking Build System)
# 步骤, 避免并行构建时 stamp-check 竞态触发 MSB8066 (exit code 0x100000CB)
# 本脚本已在构建前显式 configure, 无需 VS 自动重新配置
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_BUILD_UI="$BUILD_UI" \
    -DGGML_CCACHE="$USE_CCACHE" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_SUPPRESS_REGENERATION=ON

if [ -z "$TARGETS" ]; then
    cmake --build build -j "$JOBS" --config Release
else
    # 故意不加引号, 让 TARGETS 按空格分词为多个 --target 参数
    # shellcheck disable=SC2086
    cmake --build build -j "$JOBS" --config Release --target $TARGETS
fi
