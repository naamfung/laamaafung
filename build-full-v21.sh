# 按当前 git 分支命名构建目录, 获取失败时回退到 build
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ -n "$BRANCH" ] && [ "$BRANCH" != "HEAD" ]; then
    BUILD_DIR="build-$BRANCH"
else
    BUILD_DIR="build"
fi

set -e

rm -rf "$BUILD_DIR"

# 清理跨分支共享残留的预构建前端资源, 避免误用不匹配版本的静态页面
rm -rf tools/ui/dist

# LLAMA_KVMEM=ON + ROOT 指向仓库根: 构建含 KVMem 的全部程序
# BUILD_UI=ON 编译界面
#
# CMAKE_SUPPRESS_REGENERATION=ON (重要, 勿删):
#   Visual Studio 生成器会往每个 .vcxproj 里塞一条"构建系统自检"规则
#       cmake -S <src> -B <build> --check-stamp-file <dir>/CMakeFiles/generate.stamp
#   并额外生成 ZERO_CHECK 工程
#       cmake ... --check-stamp-list CMakeFiles/generate.stamp.list  (覆盖全部 53 个目录)
#   这些规则每次构建都会重写目录戳文件 (先写 generate.stamp.tmp<rand> 再
#   MoveFileEx 覆盖 generate.stamp), 即使配置完全没变也一样。
#   加上 cmake --build -j (MSBuild /m) 的并行, 同一批戳文件会被数十个进程
#   同时抢写; 只要某个戳文件被防病毒/搜索索引/其它进程瞬时占住, 覆盖失败:
#       CMake error : Cannot restore timestamp "...generate.stamp": 拒绝访问。
#   MSBuild 随即以 MSB8066 中断该工程 (v21 曾因此丢掉 test-model-load-cancel.exe),
#   而且该进程会顺势回退到"重新配置"分支, 在本就并行编译时跑一次 configure,
#   于是日志里出现 Configuring incomplete / 工程文件被改写的隐患。
#   本脚本每次都是 rm -rf 构建目录后全量配置, 不依赖增量自检, 故直接关闭它。
cmake -B "$BUILD_DIR" -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$PWD" -DCMAKE_SUPPRESS_REGENERATION=ON

# cmake --build "$BUILD_DIR" --config Release --target llama-server --parallel
cmake --build "$BUILD_DIR" -j --config Release

# 收尾自检: MSB8066 只会中断单个工程, 不会让 --build 整体失败, 容易漏掉静默缺失的产物
missing=0
for exe in llama-server llama-cli llama-bench llama-perplexity llama-quantize llama-kvmem-server test-model-load-cancel; do
    if [ ! -f "$BUILD_DIR/bin/Release/$exe.exe" ]; then
        echo "构建自检: 缺少产物 $BUILD_DIR/bin/Release/$exe.exe" >&2
        missing=1
    fi
done
[ "$missing" -eq 0 ] || exit 1
echo "构建完成: $BUILD_DIR/bin/Release"
