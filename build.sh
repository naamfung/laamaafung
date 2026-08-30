#!/bin/sh
# 用法: ./build.sh [--clean] [--ui] [--no-ccache] [--jobs N] [target...]
#   --clean     删除构建目录后全量编译
#   --ui        启用界面编译 (LLAMA_BUILD_UI=ON)
#   --no-ccache 禁用 ccache (MSVC + ccache 偶发崩溃时使用)
#   --jobs N    并行任务数 (默认 1, 避免 MSVC 并行编译时 cl.exe access violation 崩溃;
#               可用 --jobs 2 等提升速度, 但若遇 0x1000000A 错误请回退到 1)
#   target      可选 cmake 目标, 如 llama-server; 不指定则编译全部
#
# 构建目录默认按当前 git 分支命名 (如 build-master, build-v12),
# 非 git 仓库或分支获取失败时回退到 build

set -e

CLEAN=0
BUILD_UI=OFF
USE_CCACHE=ON
JOBS=1
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

# 按当前 git 分支命名构建目录, 获取失败时回退到 build
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ]; then
    BUILD_DIR="build-$BRANCH"
else
    BUILD_DIR="build"
fi

if [ "$CLEAN" -eq 1 ]; then
    rm -rf "$BUILD_DIR"
fi

# MSYS_NO_PATHCONV=1: 防止 git-bash/msys2 把 /DWIN32 等 MSVC flags 误当作 Unix 路径
# 转换成 Windows 路径 (如 /DWIN32 -> D:/winkit/share/PortableGit/DWIN32)
# 该环境变量不影响 cmake 内部行为, 仅抑制 shell 层路径转换
export MSYS_NO_PATHCONV=1

# CMAKE_SUPPRESS_REGENERATION=ON: 禁用 VS 生成器的 ZERO_CHECK (Checking Build System)
# 步骤, 避免并行构建时 stamp-check 竞态触发 MSB8066 (exit code 0x100000CB)
# 本脚本已在构建前显式 configure, 无需 VS 自动重新配置
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_BUILD_UI="$BUILD_UI" \
    -DGGML_CCACHE="$USE_CCACHE" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_SUPPRESS_REGENERATION=ON

if [ -z "$TARGETS" ]; then
    cmake --build "$BUILD_DIR" -j "$JOBS" --config Release
else
    # 故意不加引号, 让 TARGETS 按空格分词为多个 --target 参数
    # shellcheck disable=SC2086
    cmake --build "$BUILD_DIR" -j "$JOBS" --config Release --target $TARGETS
fi
