// laamaafung 构建器 —— build-full-v21.sh 的 Go 重写版。
//
// 为什么用 Go：编译出的二进制执行时不经过 shell，不受 git-bash 环境、代理变量、
// env 清洗差异的干扰。此前 SHELL 路径实测踩过的坑，这里全部内置处理：
//   - 代理变量（HTTP_PROXY 等）会让 MSBuild 报 MSB6001 —— 子进程环境强制剥离；
//   - vs_cmake.sh 的 `env -i` 清洗过头的反面教训：FindCUDAToolkit 探到版本却
//     找不到 CUDA_CUDART 库 —— 本构建器保留完整环境（仅剥离代理），CUDA 探测正常；
//   - `$PWD` 是 POSIX 形式（/g/...），Windows 原生 CMake 不认 —— 用 Go 原生
//     路径（G:\...）传 LLAMA_KVMEM_ROOT；
//   - git-bash 的 rm -rf 大目录可能被环境安全钩子拦截 —— Go 的 os.RemoveAll
//     原生删除，不走 shell；另提供 -keep 增量模式（只清 CMake 缓存）。
//
// 2026-09-23 新增（ccache + Ninja）：
//   - MSVC 开发环境自建：本机 cmd.exe 被安全策略禁用、vcvars 无法调用，因此这里
//     直接用 Go 探测 VS 安装 / MSVC 工具集 / Windows SDK 版本，手工拼出
//     INCLUDE / LIB / PATH（等价 vcvars64.bat 的关键部分），Ninja 生成器即可用。
//   - ccache：Ninja 是 CMake 里少数会真正执行 CMAKE_<LANG>_COMPILER_LAUNCHER 的
//     Windows 生成器（Visual Studio 生成器会**静默忽略** launcher，实测 build 树内
//     无任何 ccache 引用）。
//   - **只给 CUDA 挂 ccache**：实测 ccache 4.13.6 包装本机本地化 MSVC 时必崩
//     （cl.exe 输出中文 GBK，ccache 按 UTF-8 解析 → std::filesystem
//     "Illegal byte sequence"；设 CCACHE_MSVC_DEP_PREFIX / VSLANG=1033 都无效），
//     而包装 nvcc 完全正常且能命中。CUDA 实例正是耗时大头，所以这样配置既安全又有收益。
//     因此必须同时 -DGGML_CCACHE=OFF：ggml 的自身接法会设全局 RULE_LAUNCH_COMPILE，
//     Ninja 下会连 cl.exe 一起包 → 必崩。
//
// 用法：
//   builder.exe                  全量构建（默认 -j8；新建目录时 Ninja+ccache）
//   builder.exe -j 12            指定并行度
//   builder.exe -keep            不删构建目录（增量/重配置续编）
//   builder.exe -arch 86         覆盖 CUDA 架构（默认 native；等价 sh 的 CUDA_ARCH）
//   builder.exe -gen vs         强制 Visual Studio 生成器（ccache 自动停用）
//   builder.exe -gen ninja      强制 Ninja（换生成器需先 clean）
//   builder.exe -no-ccache       关闭 ccache
//   builder.exe -ccache-all      也给 C/CXX 挂 ccache（本机本地化 MSVC 下会崩，仅调试用）
//   builder.exe -C <dir>         指定仓库根（默认取 builder.exe 所在目录；用于 worktree）
//   builder.exe -list            打印探测到的工具链/环境后退出
//   builder.exe clean            仅清理构建目录与 ui/dist
//
// 环境变量 CUDA_ARCH 等价于 -arch；两处都设置时 -arch 优先。
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	colorRed    = "\033[0;31m"
	colorGreen  = "\033[0;32m"
	colorYellow = "\033[0;33m"
	colorBlue   = "\033[0;34m"
	colorNC     = "\033[0m"
)

func printInfo(msg string)    { fmt.Print(colorBlue + "[INFO] " + msg + colorNC + "\n") }
func printSuccess(msg string) { fmt.Print(colorGreen + "[ OK ] " + msg + colorNC + "\n") }
func printWarning(msg string) { fmt.Print(colorYellow + "[WARN] " + msg + colorNC + "\n") }
func printError(msg string)   { fmt.Print(colorRed + "[FAIL] " + msg + colorNC + "\n") }

// proxyVars 传给 MSBuild 的环境里必须剥离的变量（MSB6001 的根因）。
var proxyVars = []string{
	"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
	"http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
	"NO_PROXY", "no_proxy",
}

// childExtraEnv 由 main 在解析参数后填好，run()/git 调用时合并进子进程环境。
var childExtraEnv []string

// cleanEnv 返回剥离代理变量后的环境副本，再叠加 childExtraEnv（MSVC / ccache）。
func cleanEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		blocked := false
		for _, p := range proxyVars {
			if strings.HasPrefix(kv, p+"=") {
				blocked = true
				break
			}
		}
		if !blocked {
			env = append(env, kv)
		}
	}
	return append(env, childExtraEnv...)
}

// run 执行命令（工作目录固定为仓库根）。
func run(repoRoot, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = repoRoot
	cmd.Env = cleanEnv()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	printInfo("$ " + name + " " + strings.Join(args, " "))
	start := time.Now()
	if err := cmd.Run(); err != nil {
		printError(fmt.Sprintf("%s 失败（耗时 %s）: %v", name, time.Since(start).Round(time.Second), err))
		return err
	}
	printInfo(fmt.Sprintf("%s 完成（耗时 %s）", name, time.Since(start).Round(time.Second)))
	return nil
}

// transientPatterns 是并行编译时的环境瞬时竞争特征串：
//   - nvcc 在 %TEMP% 下生成的 tmpxft_*_cudafe1.cpp 被抢/丢失 →
//     "c1xx: fatal error C1083: 无法打开源文件 ... tmpxft_..."
//   - VS 生成器的构建系统自检戳文件被防病毒/索引器瞬时占住 → MSB8066
// 两类都是重跑即过，不应让人工介入。
var transientPatterns = []string{"tmpxft_", "MSB8066", "Cannot restore timestamp", "MSB6001"}

// logHasTransient 检查编译日志尾部是否含瞬时竞争特征。
func logHasTransient(logPath string) bool {
	b, err := os.ReadFile(logPath)
	if err != nil {
		return false
	}
	// 只看尾部 256KB，避免大日志全量扫描
	if len(b) > 256*1024 {
		b = b[len(b)-256*1024:]
	}
	s := string(b)
	for _, p := range transientPatterns {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

// runLogged 执行命令并把输出同时写到控制台与日志文件（tee）。
func runLogged(repoRoot, logPath, name string, args ...string) error {
	f, err := os.Create(logPath)
	if err != nil {
		printWarning("无法创建日志文件（仅输出到控制台）: " + err.Error())
		return run(repoRoot, name, args...)
	}
	defer f.Close()

	cmd := exec.Command(name, args...)
	cmd.Dir = repoRoot
	cmd.Env = cleanEnv()
	w := io.MultiWriter(os.Stdout, f)
	cmd.Stdout, cmd.Stderr = w, w
	printInfo("$ " + name + " " + strings.Join(args, " ") + "   (日志: " + logPath + ")")
	start := time.Now()
	if err := cmd.Run(); err != nil {
		printError(fmt.Sprintf("%s 失败（耗时 %s）: %v", name, time.Since(start).Round(time.Second), err))
		return err
	}
	printInfo(fmt.Sprintf("%s 完成（耗时 %s）", name, time.Since(start).Round(time.Second)))
	return nil
}

// runWithRetry 在遇到瞬时竞争（见 transientPatterns）时自动重跑，最多 attempts 次。
// 有了 ccache，重跑只会重做失败的那几个编译单元，代价很小。
func runWithRetry(repoRoot, logPath, name string, attempts int, args ...string) error {
	for i := 1; i <= attempts; i++ {
		err := runLogged(repoRoot, logPath, name, args...)
		if err == nil {
			return nil
		}
		if i < attempts && logHasTransient(logPath) {
			printWarning(fmt.Sprintf("检测到并行编译的瞬时竞争（nvcc 临时文件 / 构建戳文件），自动重试 %d/%d", i, attempts-1))
			continue
		}
		return err
	}
	return nil
}

// runCapture 执行命令并捕获输出（用于探测类调用，不打印）。
func runCapture(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = cleanEnv()
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// gitBranch 取当前分支名；失败（如 detached HEAD）返回空串。
func gitBranch(repoRoot string) string {
	out, err := runCapture(repoRoot, "git", "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

// buildDirName 按分支命名构建目录，取不到分支时回退 "build"。
func buildDirName(repoRoot string) string {
	if b := gitBranch(repoRoot); b != "" {
		return "build-" + b
	}
	return "build"
}

// ---------------------------------------------------------------------------
// MSVC 工具链探测（替代 vcvars64.bat）
// ---------------------------------------------------------------------------

const sdkBaseWin = `C:\Program Files (x86)\Windows Kits\10`

func maxDirEntry(base string) string {
	ents, err := os.ReadDir(base)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names) // 版本号零填充一致，字典序即版本序
	return names[len(names)-1]
}

// findMSVC 探测 VS 安装根、MSVC 工具集版本、Windows SDK 版本。
func findMSVC() (vsRoot, msvcVer, sdkVer string) {
	for _, c := range []string{
		`C:\Program Files\Microsoft Visual Studio\2022\Community`,
		`C:\Program Files\Microsoft Visual Studio\2022\Professional`,
		`C:\Program Files\Microsoft Visual Studio\2022\Enterprise`,
		`C:\Program Files\Microsoft Visual Studio\2022\BuildTools`,
		`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`,
		`C:\Program Files\Microsoft Visual Studio\2019\Community`,
		`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`,
	} {
		if v := maxDirEntry(filepath.Join(c, "VC", "Tools", "MSVC")); v != "" {
			vsRoot, msvcVer = c, v
			break
		}
	}
	if vsRoot == "" {
		return "", "", ""
	}
	sdkVer = maxDirEntry(filepath.Join(sdkBaseWin, "Include"))
	return vsRoot, msvcVer, sdkVer
}

// msvcEnvVars 由探测结果拼出 vcvars64.bat 的关键环境变量。
func msvcEnvVars(vsRoot, msvcVer, sdkVer string) []string {
	vc := filepath.Join(vsRoot, "VC", "Tools", "MSVC", msvcVer)
	include := strings.Join([]string{
		filepath.Join(vc, "include"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "ucrt"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "um"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "shared"),
		filepath.Join(sdkBaseWin, "Include", sdkVer, "winrt"),
	}, ";")
	lib := strings.Join([]string{
		filepath.Join(vc, "lib", "x64"),
		filepath.Join(sdkBaseWin, "Lib", sdkVer, "ucrt", "x64"),
		filepath.Join(sdkBaseWin, "Lib", sdkVer, "um", "x64"),
	}, ";")
	path := strings.Join([]string{
		filepath.Join(vc, "bin", "Hostx64", "x64"),
		filepath.Join(vsRoot, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "Ninja"),
		filepath.Join(sdkBaseWin, "bin", sdkVer, "x64"),
		os.Getenv("PATH"),
	}, string(os.PathListSeparator))
	return []string{
		"INCLUDE=" + include,
		"LIB=" + lib,
		"PATH=" + path,
		"VCToolsInstallDir=" + vc + `\`,
		"VCINSTALLDIR=" + filepath.Join(vsRoot, "VC") + `\`,
		"WindowsSdkDir=" + sdkBaseWin + `\`,
		"WindowsSDKVersion=" + sdkVer + `\`,
		// cl.exe 默认输出中文，ccache 解析编译器输出时会崩；同时 nvcc 的英文输出更稳。
		"VSLANG=1033",
	}
}

// findCcache 定位 ccache.exe：先 PATH，再常见便携安装目录。
func findCcache() string {
	if p, err := exec.LookPath("ccache"); err == nil {
		return p
	}
	for _, pat := range []string{
		filepath.Join(`D:\winkit\share`, "ccache-*", "ccache.exe"),
		filepath.Join(`C:\winkit\share`, "ccache-*", "ccache.exe"),
		filepath.Join(`D:\tools`, "ccache-*", "ccache.exe"),
	} {
		if ms, err := filepath.Glob(pat); err == nil && len(ms) > 0 {
			sort.Strings(ms)
			return ms[len(ms)-1]
		}
	}
	return ""
}

// ccacheEnvVars 组装 ccache 运行环境。
func ccacheEnvVars(repoRoot, ccacheDir string) []string {
	if ccacheDir == "" {
		// 放在仓库的兄弟目录：多个 worktree/分支共享同一份缓存，且不污染 git 工作区
		ccacheDir = filepath.Join(filepath.Dir(repoRoot), ".ccache")
	}
	os.MkdirAll(ccacheDir, 0o755)
	return []string{
		"CCACHE_DIR=" + ccacheDir,
		"CCACHE_MAXSIZE=20G",
		// MSVC/nvcc 的 include 时间戳与绝对路径会让 ccache 拒绝缓存，放宽检查
		"CCACHE_SLOPPINESS=time_macros,include_file_ctime,include_file_mtime",
		"CCACHE_BASEDIR=" + filepath.Dir(repoRoot),
	}
}

func ccacheStats(ccachePath string) {
	if ccachePath == "" {
		return
	}
	cmd := exec.Command(ccachePath, "-s")
	cmd.Env = cleanEnv()
	out, err := cmd.Output()
	if err != nil {
		return
	}
	txt := string(out)
	if i := strings.Index(txt, "Local storage:"); i > 0 {
		txt = txt[:i]
	}
	printInfo("ccache 统计:\n" + strings.TrimRight(txt, "\n"))
}

// ---------------------------------------------------------------------------
// 构建流程
// ---------------------------------------------------------------------------

// detectGenerator 读已有构建目录的 CMakeCache，返回其生成器名（无则空串）。
// 注意：CMake 不允许在既有目录上更换生成器，所以已有目录必须沿用原生成器。
func detectGenerator(buildDir string) string {
	b, err := os.ReadFile(filepath.Join(buildDir, "CMakeCache.txt"))
	if err != nil {
		return ""
	}
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "CMAKE_GENERATOR:INTERNAL=") {
			return strings.TrimSpace(strings.TrimPrefix(l, "CMAKE_GENERATOR:INTERNAL="))
		}
	}
	return ""
}

// cleanArtifacts 删除构建目录与跨分支共享的预构建前端资源。
// keep=true 时做增量重配置：**Ninja（单配置）的对象文件就落在 buildDir/CMakeFiles/ 下，
// 删掉等于全量重编，所以 Ninja 的 -keep 不做任何删除**；VS（多配置）对象在
// <target>.dir/Release/ 下，清缓存不影响对象。
func cleanArtifacts(repoRoot, buildDir, generator string, keep bool) {
	if keep {
		if strings.Contains(generator, "Ninja") {
			printInfo("-keep 模式（Ninja）: 保留全部构建状态，仅重跑 configure + build（真正增量）")
		} else {
			printInfo("-keep 模式: 仅清 CMake 缓存（保留构建目录增量）")
			os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeCache.txt"))
			os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeFiles"))
		}
	} else {
		printInfo("删除构建目录 " + buildDir)
		if err := os.RemoveAll(filepath.Join(repoRoot, buildDir)); err != nil {
			printWarning("删除构建目录失败（继续）: " + err.Error())
		}
	}
	// 清理跨分支共享残留的预构建前端资源, 避免误用不匹配版本的静态页面
	uiDist := filepath.Join(repoRoot, "tools", "ui", "dist")
	if _, err := os.Stat(uiDist); err == nil {
		printInfo("删除 tools/ui/dist（跨分支共享残留）")
		os.RemoveAll(uiDist)
	}
}

// configure 组装并执行 CMake 配置。
func configure(repoRoot, buildDir, cudaArch, generator, ccachePath, logPath string, ccacheAll bool, forceCcache bool, extra []string) error {
	args := []string{"-B", buildDir}
	if generator != "" {
		args = append(args, "-G", generator)
	}
	args = append(args,
		"-DCMAKE_CUDA_ARCHITECTURES="+cudaArch,
		"-DGGML_CUDA=ON",
		"-DGGML_NATIVE=ON",
		"-DGGML_CUDA_FA=ON",
		"-DGGML_CUDA_FA_ALL_QUANTS=ON",
		"-DCMAKE_BUILD_TYPE=Release",
		// LLAMA_KVMEM=ON + ROOT 指向仓库根: 构建含 KVMem 的全部程序
		"-DLLAMA_KVMEM=ON",
		"-DLLAMA_KVMEM_ROOT="+repoRoot,
		// CMAKE_SUPPRESS_REGENERATION=ON (重要, 勿删):
		//   VS 生成器会往每个 .vcxproj 塞"构建系统自检"规则并生成 ZERO_CHECK 工程，
		//   并行构建时数十个进程抢写同一批 generate.stamp，被防病毒/索引器瞬时
		//   占住就报 "Cannot restore timestamp: 拒绝访问" → MSB8066 中断工程。
		//   本构建器全量配置不依赖增量自检，直接关闭。
		"-DCMAKE_SUPPRESS_REGENERATION=ON",
	)

	// ccache：只挂 CUDA。ggml 自带的 GGML_CCACHE 会设全局 RULE_LAUNCH_COMPILE，
	// Ninja 下会连 cl.exe 一起包（本机本地化 MSVC + ccache 必崩），必须关掉。
	ccacheOn := forceCcache && ccachePath != "" && strings.Contains(generator, "Ninja")
	if forceCcache && ccachePath == "" {
		printWarning("未找到 ccache.exe，本次不使用 ccache")
	}
	if forceCcache && ccachePath != "" && !strings.Contains(generator, "Ninja") {
		printWarning("当前生成器不是 Ninja，ccache 无法生效（VS 生成器会忽略 launcher）")
	}
	args = append(args, "-DGGML_CCACHE=OFF")
	if ccacheOn {
		args = append(args, "-DCMAKE_CUDA_COMPILER_LAUNCHER="+ccachePath)
		if ccacheAll {
			printWarning("-ccache-all: 也给 C/CXX 挂 ccache（本机本地化 MSVC 下预计会崩，仅调试用）")
			args = append(args,
				"-DCMAKE_C_COMPILER_LAUNCHER="+ccachePath,
				"-DCMAKE_CXX_COMPILER_LAUNCHER="+ccachePath)
		}
	}

	args = append(args, extra...)
	return runWithRetry(repoRoot, logPath, "cmake", 2, args...)
}

// build 执行编译。瞬时竞争（nvcc 临时文件等）允许再重试两次。
func build(repoRoot, buildDir, logPath string, jobs int) error {
	return runWithRetry(repoRoot, logPath, "cmake", 3,
		"--build", buildDir, fmt.Sprintf("-j%d", jobs), "--config", "Release")
}

// verifyArtifacts 收尾自检: MSB8066 只会中断单个工程，不会让 --build 整体
// 失败，容易漏掉静默缺失的产物。多配置生成器产物在 bin/Release/，
// 单配置（Ninja）在 bin/。
func verifyArtifacts(repoRoot, buildDir string) error {
	candidates := []string{
		filepath.Join(repoRoot, buildDir, "bin", "Release"),
		filepath.Join(repoRoot, buildDir, "bin"),
	}
	binDir := ""
	for _, d := range candidates {
		if _, err := os.Stat(filepath.Join(d, "llama-cli.exe")); err == nil {
			binDir = d
			break
		}
	}
	if binDir == "" {
		binDir = candidates[0]
	}
	missing := 0
	for _, exe := range []string{
		"llama-server", "llama-cli", "llama-bench", "llama-perplexity",
		"llama-quantize", "llama-kvmem-server", "test-model-load-cancel",
	} {
		p := filepath.Join(binDir, exe+".exe")
		if _, err := os.Stat(p); err != nil {
			printError("构建自检: 缺少产物 " + p)
			missing++
		} else {
			printSuccess("产物 " + exe + ".exe")
		}
	}
	if missing > 0 {
		return fmt.Errorf("缺少 %d 个产物", missing)
	}
	printSuccess("构建完成: " + binDir)
	return nil
}

func main() {
	jobs := flag.Int("j", 8, "并行编译度")
	keep := flag.Bool("keep", false, "保留构建目录（仅清 CMake 缓存做增量重配置）")
	arch := flag.String("arch", "", "CUDA 架构（默认 native；亦可用环境变量 CUDA_ARCH）")
	clean := flag.Bool("clean", false, "只清理不构建")
	root := flag.String("C", "", "仓库根（默认 builder.exe 所在目录）")
	gen := flag.String("gen", "auto", "生成器: auto|ninja|vs")
	noCcache := flag.Bool("no-ccache", false, "关闭 ccache")
	ccacheAll := flag.Bool("ccache-all", false, "也给 C/CXX 挂 ccache（调试用）")
	ccacheDir := flag.String("ccache-dir", "", "ccache 缓存目录（默认 <仓库父目录>/.ccache）")
	listOnly := flag.Bool("list", false, "打印探测到的工具链后退出")
	flag.Parse()

	// 定位仓库根：默认以可执行文件所在目录为准（builder.exe 与 builder.go 同放仓库根）。
	repoRoot := ""
	if *root != "" {
		if abs, err := filepath.Abs(*root); err == nil {
			repoRoot = abs
		}
	} else {
		exePath, err := os.Executable()
		if err != nil {
			exePath = os.Args[0]
		}
		repoRoot, _ = filepath.Abs(filepath.Dir(exePath))
		// 供 `go run builder.go` 调试时使用当前目录（exe 在临时目录里）。
		if strings.Contains(filepath.Base(exePath), "go-build") ||
			filepath.Base(exePath) == "builder.test.exe" {
			if wd, err := os.Getwd(); err == nil {
				repoRoot = wd
			}
		}
	}
	if _, err := os.Stat(filepath.Join(repoRoot, "ggml")); err != nil {
		printError("仓库根不像 llama.cpp 源码树（缺少 ggml/）: " + repoRoot)
		os.Exit(1)
	}

	if envArch := os.Getenv("CUDA_ARCH"); *arch == "" && envArch != "" {
		*arch = envArch
	}
	if *arch == "" {
		*arch = "native"
	}

	// ---- 工具链探测 ----
	vsRoot, msvcVer, sdkVer := findMSVC()
	ccachePath := findCcache()
	if *noCcache {
		ccachePath = ""
	}

	// 子进程环境：MSVC 变量 + ccache 变量
	childExtraEnv = nil
	if vsRoot != "" {
		childExtraEnv = append(childExtraEnv, msvcEnvVars(vsRoot, msvcVer, sdkVer)...)
	}
	if ccachePath != "" {
		childExtraEnv = append(childExtraEnv, ccacheEnvVars(repoRoot, *ccacheDir)...)
	}

	buildDir := buildDirName(repoRoot)
	detectedGen := detectGenerator(filepath.Join(repoRoot, buildDir))

	// 选生成器：ccache 只在 Ninja 下生效，所以新目录默认 Ninja。
	chosenGen := detectedGen
	passGen := ""
	switch {
	case detectedGen != "":
		// 已有构建目录：必须沿用原生成器（CMake 不允许换）
		if *gen == "ninja" && !strings.Contains(detectedGen, "Ninja") {
			printError(fmt.Sprintf("已有构建目录用的是 %q，无法直接换成 Ninja；请先执行: builder.exe clean", detectedGen))
			os.Exit(1)
		}
		if *gen == "vs" && strings.Contains(detectedGen, "Ninja") {
			printError(fmt.Sprintf("已有构建目录用的是 %q，无法直接换成 VS；请先执行: builder.exe clean", detectedGen))
			os.Exit(1)
		}
	default:
		switch *gen {
		case "ninja":
			chosenGen, passGen = "Ninja", "Ninja"
		case "vs":
			chosenGen = "Visual Studio 17 2022"
		default:
			if vsRoot != "" && ccachePath != "" {
				chosenGen, passGen = "Ninja", "Ninja"
			} else if vsRoot != "" {
				chosenGen, passGen = "Ninja", "Ninja"
			} else {
				chosenGen = "Visual Studio 17 2022"
			}
		}
	}

	// ---- 打印摘要 ----
	printInfo(fmt.Sprintf("仓库根: %s", repoRoot))
	br := gitBranch(repoRoot)
	if br == "" {
		br = "(detached)"
	}
	printInfo(fmt.Sprintf("构建目录: %s（分支: %s）", buildDir, br))
	printInfo(fmt.Sprintf("CUDA 架构: %s, 并行度: %d", *arch, *jobs))
	if vsRoot != "" {
		printInfo(fmt.Sprintf("MSVC: %s (%s) / SDK %s", vsRoot, msvcVer, sdkVer))
	} else {
		printWarning("未探测到 VS 安装：Ninja 构建可能找不到 cl.exe")
	}
	if chosenGen == "" {
		chosenGen = "Visual Studio 17 2022"
	}
	printInfo(fmt.Sprintf("生成器: %s", chosenGen))
	if ccachePath != "" && strings.Contains(chosenGen, "Ninja") {
		printSuccess("ccache: " + ccachePath + "（仅 CUDA；缓存 " + func() string {
			if *ccacheDir != "" {
				return *ccacheDir
			}
			return filepath.Join(filepath.Dir(repoRoot), ".ccache")
		}() + "）")
	} else if ccachePath != "" {
		printWarning("ccache: 已找到但当前生成器非 Ninja（VS 生成器会忽略 launcher），本次不生效")
	} else {
		printWarning("ccache: 未找到 ccache.exe（放在 PATH 或 D:\\winkit\\share\\ccache-*）")
	}

	if *listOnly {
		printInfo("环境变量（子进程）：")
		for _, kv := range childExtraEnv {
			printInfo("  " + kv)
		}
		return
	}

	// 环境自检提示（有代理变量时说明剥离动作，避免 MSB6001 复发时排查无门）。
	for _, p := range proxyVars {
		if os.Getenv(p) != "" {
			printWarning(fmt.Sprintf("检测到 %s，已对子进程剥离（防 MSB6001）", p))
			break
		}
	}

	cleanArtifacts(repoRoot, buildDir, chosenGen, *keep)
	if *clean {
		printSuccess("清理完成")
		return
	}

	os.MkdirAll(filepath.Join(repoRoot, buildDir), 0o755)
	cfgLog := filepath.Join(repoRoot, buildDir, "builder-configure.log")
	buildLog := filepath.Join(repoRoot, buildDir, "builder-build.log")

	start := time.Now()
	if err := configure(repoRoot, buildDir, *arch, passGen, ccachePath, cfgLog, *ccacheAll, !*noCcache, flag.Args()); err != nil {
		printError("configure 失败，详见 " + cfgLog)
		os.Exit(1)
	}
	if err := build(repoRoot, buildDir, buildLog, *jobs); err != nil {
		printError("编译失败，详见 " + buildLog)
		os.Exit(1)
	}
	if err := verifyArtifacts(repoRoot, buildDir); err != nil {
		os.Exit(1)
	}
	ccacheStats(ccachePath)
	printSuccess(fmt.Sprintf("总耗时 %s", time.Since(start).Round(time.Second)))
}
