# 按当前 git 分支命名构建目录, 获取失败时回退到 build
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ]; then
    BUILD_DIR="build-$BRANCH"
else
    BUILD_DIR="build"
fi

rm -rf "$BUILD_DIR"

# 清理跨分支共享残留的预构建前端资源, 避免误用不匹配版本的静态页面
rm -rf tools/ui/dist

# BUILD_UI=ON 编译界面
cmake -B "$BUILD_DIR" -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release

# cmake --build build --config Release --target llama-server --parallel
cmake --build "$BUILD_DIR" -j --config Release