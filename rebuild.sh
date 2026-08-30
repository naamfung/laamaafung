# 清理跨分支共享残留的预构建前端资源, 避免误用不匹配版本的静态页面
rm -rf tools/ui/dist

# BUILD_UI=ON 编译界面
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release

# cmake --build build --config Release --target llama-server --parallel
cmake --build build -j --config Release