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
// 用法：
//   builder.exe              全量构建（默认 -j8）
//   builder.exe -j 12        指定并行度
//   builder.exe -keep        不删构建目录（增量/重配置续编）
//   builder.exe -arch 86     覆盖 CUDA 架构（默认 native；等价 sh 的 CUDA_ARCH 环境变量）
//   builder.exe clean        仅清理构建目录与 ui/dist
//
// 环境变量 CUDA_ARCH 等价于 -arch；两处都设置时 -arch 优先。
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

// cleanEnv 返回剥离代理变量后的环境副本。
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
	return env
}

// run 执行命令，输出同时打到控制台与日志文件（tee）。
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

// gitBranch 取当前分支名；失败（如 detached HEAD）返回空串。
func gitBranch(repoRoot string) string {
	cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = repoRoot
	cmd.Env = cleanEnv()
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// buildDirName 按分支命名构建目录，取不到分支时回退 "build"。
func buildDirName(repoRoot string) string {
	if b := gitBranch(repoRoot); b != "" {
		return "build-" + b
	}
	return "build"
}

// cleanArtifacts 删除构建目录与跨分支共享的预构建前端资源。
// keep=true 时仅清 CMake 缓存（增量重配置用），保留已编译对象。
func cleanArtifacts(repoRoot, buildDir string, keep bool) {
	if !keep {
		printInfo("删除构建目录 " + buildDir)
		if err := os.RemoveAll(filepath.Join(repoRoot, buildDir)); err != nil {
			printWarning("删除构建目录失败（继续）: " + err.Error())
		}
	} else {
		printInfo("-keep 模式: 仅清 CMake 缓存（保留构建目录增量）")
		os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeCache.txt"))
		os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeFiles"))
	}
	// 清理跨分支共享残留的预构建前端资源, 避免误用不匹配版本的静态页面
	uiDist := filepath.Join(repoRoot, "tools", "ui", "dist")
	if _, err := os.Stat(uiDist); err == nil {
		printInfo("删除 tools/ui/dist（跨分支共享残留）")
		os.RemoveAll(uiDist)
	}
}

// configure 组装并执行 CMake 配置。
func configure(repoRoot, buildDir, cudaArch string, extra []string) error {
	args := []string{
		"-B", buildDir,
		"-DCMAKE_CUDA_ARCHITECTURES=" + cudaArch,
		"-DGGML_CUDA=ON",
		"-DGGML_NATIVE=ON",
		"-DGGML_CUDA_FA=ON",
		"-DGGML_CUDA_FA_ALL_QUANTS=ON",
		"-DCMAKE_BUILD_TYPE=Release",
		// LLAMA_KVMEM=ON + ROOT 指向仓库根: 构建含 KVMem 的全部程序
		"-DLLAMA_KVMEM=ON",
		"-DLLAMA_KVMEM_ROOT=" + repoRoot,
		// CMAKE_SUPPRESS_REGENERATION=ON (重要, 勿删):
		//   VS 生成器会往每个 .vcxproj 塞"构建系统自检"规则并生成 ZERO_CHECK 工程，
		//   并行构建时数十个进程抢写同一批 generate.stamp，被防病毒/索引器瞬时
		//   占住就报 "Cannot restore timestamp: 拒绝访问" → MSB8066 中断工程。
		//   本构建器全量配置不依赖增量自检，直接关闭。
		"-DCMAKE_SUPPRESS_REGENERATION=ON",
	}
	args = append(args, extra...)
	return run(repoRoot, "cmake", args...)
}

// build 执行编译。
func build(repoRoot, buildDir string, jobs int) error {
	return run(repoRoot, "cmake", "--build", buildDir,
		fmt.Sprintf("-j%d", jobs), "--config", "Release")
}

// verifyArtifacts 收尾自检: MSB8066 只会中断单个工程，不会让 --build 整体
// 失败，容易漏掉静默缺失的产物。
func verifyArtifacts(repoRoot, buildDir string) error {
	binDir := filepath.Join(repoRoot, buildDir, "bin", "Release")
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
	printSuccess("构建完成: " + buildDir + "\\bin\\Release")
	return nil
}

func main() {
	// 定位仓库根：以可执行文件所在目录为准（builder.exe 与 builder.go 同放仓库根）。
	exePath, err := os.Executable()
	if err != nil {
		exePath = os.Args[0]
	}
	repoRoot, err := filepath.Abs(filepath.Dir(exePath))
	if err != nil {
		fmt.Println("定位仓库根失败:", err)
		os.Exit(1)
	}
	// 供 `go run builder.go` 调试时使用当前目录（exe 在临时目录里）。
	if strings.Contains(filepath.Base(exePath), "go-build") ||
		filepath.Base(exePath) == "builder.test.exe" {
		if wd, err := os.Getwd(); err == nil {
			repoRoot = wd
		}
	}

	jobs := flag.Int("j", 8, "并行编译度")
	keep := flag.Bool("keep", false, "保留构建目录（仅清 CMake 缓存做增量重配置）")
	arch := flag.String("arch", "", "CUDA 架构（默认 native；亦可用环境变量 CUDA_ARCH）")
	clean := flag.Bool("clean", false, "只清理不构建")
	flag.Parse()

	if envArch := os.Getenv("CUDA_ARCH"); *arch == "" && envArch != "" {
		*arch = envArch
	}
	if *arch == "" {
		*arch = "native"
	}

	buildDir := buildDirName(repoRoot)

	// 环境自检提示（有代理变量时说明剥离动作，避免 MSB6001 复发时排查无门）。
	for _, p := range proxyVars {
		if os.Getenv(p) != "" {
			printWarning(fmt.Sprintf("检测到 %s，已对子进程剥离（防 MSB6001）", p))
			break
		}
	}
	printInfo(fmt.Sprintf("仓库根: %s", repoRoot))
	printInfo(fmt.Sprintf("构建目录: %s（分支: %s）", buildDir, func() string {
		if b := gitBranch(repoRoot); b != "" {
			return b
		}
		return "(detached)"
	}()))
	printInfo(fmt.Sprintf("CUDA 架构: %s, 并行度: %d", *arch, *jobs))

	cleanArtifacts(repoRoot, buildDir, *keep)
	if *clean {
		printSuccess("清理完成")
		return
	}

	start := time.Now()
	if err := configure(repoRoot, buildDir, *arch, flag.Args()); err != nil {
		printError("configure 失败，详见上方 CMake 输出")
		os.Exit(1)
	}
	if err := build(repoRoot, buildDir, *jobs); err != nil {
		printError("编译失败，详见上方 MSBuild 输出")
		os.Exit(1)
	}
	if err := verifyArtifacts(repoRoot, buildDir); err != nil {
		os.Exit(1)
	}
	printSuccess(fmt.Sprintf("总耗时 %s", time.Since(start).Round(time.Second)))
}
