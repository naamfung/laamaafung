# BUILD_UI=ON 编译界面
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release

# cmake --build build --config Release --target llama-server --parallel
cmake --build build -j --config Release