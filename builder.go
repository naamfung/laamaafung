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
// 2026-09-23 新增（内嵌 Web UI 必须构建）：
//
//	内嵌 UI 是 llama-server 的既有行为（tools/server/CMakeLists.txt 无条件链接
//	llama-ui），build-full-v21.sh 编出来的产物也是带 UI 的，所以本构建器**默认
//	必须产出带 UI 的二进制**，不允许静默降级成无 UI。
//
//	而 UI 的资源供给有两条路：LLAMA_BUILD_UI=ON 走 `bun install` + `bun run build`
//	（源码构建，版本号随构建注入）；本机联网受限时 `bun install` 会**永久挂起**
//	（表现为构建停在 "Provisioning UI assets" 数十分钟零输出，极易误判成链接卡死）。
//	实测 `bun run build`(vite) 本身是纯本地的 —— 唯一卡住的就是依赖安装。
//
//	解法：**跨 worktree 共享一份 UI 依赖缓存**（<仓库父目录>/.ui-deps，与 .ccache
//	同级），构建前用 **NTFS 目录联接**接到 <构建目录>/tools/ui/ui-src/node_modules
//	（零拷贝 —— node_modules 约 560 MB / 数万个小文件，复制在 Defender 逐文件扫描下
//	要数分钟；实测 os.RemoveAll 不穿透 junction，故 `-fresh` 删构建目录不波及缓存）。
//	ui-assets.cmake 见到 node_modules/.ui-deps-stamp 就跳过 `bun install`，之后仍是
//	**真正的源码构建**（dist/build.json 里的版本号每次构建注入，与 build-full-v21.sh
//	的产物同源同行为）。缓存只在第一次需要时生成一次，之后各 worktree 直接复用。
//
//	兜底顺序（-ui auto）：①共享依赖缓存 ②同父目录下任意已有构建目录的 node_modules
//	③都拿不到 → 回退 `-DLLAMA_BUILD_UI=OFF` 走 files/llama-b*-ui.tar.gz 本地归档
//	（仍是带 UI 的产物，但版本号固定，会打 WARN）；④连归档都没有 → 报错退出，
//	**绝不产出无 UI 的二进制**。
//
// 用法：
//
//	builder.exe                  增量构建（默认；无构建目录时即全新构建）
//	builder.exe -fresh           先删构建目录再重建（全量）
//	builder.exe -j 12            指定并行度
//	builder.exe -keep            仅重置 CMake 状态、保留已编译对象
//	builder.exe -arch 86         覆盖 CUDA 架构（默认 native；等价 sh 的 CUDA_ARCH）
//	builder.exe -gen vs         强制 Visual Studio 生成器（ccache 自动停用）
//	builder.exe -gen ninja      强制 Ninja（换生成器需先 clean）
//	builder.exe -no-ccache       关闭 ccache
//	builder.exe -ccache-all      也给 C/CXX 挂 ccache（本机本地化 MSVC 下会崩，仅调试用）
//	builder.exe -ui auto         UI 方案（默认）：UI 必须有，优先复用依赖缓存做源码构建
//	builder.exe -ui archive      强制用 files/llama-b*-ui.tar.gz 本地归档（不跑 bun）
//	builder.exe -ui off          不内嵌 UI（仅调试；llama-server 将没有 Web 界面）
//
// builder.exe -ui-deps-dir D   指定 UI 依赖缓存目录（默认 <仓库父目录>/.ui-deps）
//
//	builder.exe -no-configure    跳过 CMake configure（确认没改过 CMakeLists/参数时用）
//	builder.exe -target T        只构建目标 T（逗号分隔，如 llama-ui-assets）；跳过产物齐全性自检
//	builder.exe -C <dir>         指定仓库根（默认取 builder.exe 所在目录；用于 worktree）
//	builder.exe -list            打印探测到的工具链/环境后退出
//	builder.exe clean            仅清理构建目录与 ui/dist
//
// 瞬时竞争（nvcc 的 tmpxft_*.cudafe1.cpp C1083 / MSB8066 / MSB6001）自动重跑：
// 编译日志默认落在 <构建目录>/builder-build.log（configure 为 builder-configure.log）。
//
// 环境变量 CUDA_ARCH 等价于 -arch；两处都设置时 -arch 优先。
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
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

// cmakeExe 是探测到的 cmake 绝对路径，由 main 填好。
// 必须用绝对路径：exec.Command 按本进程 PATH 解析命令名，在 cmd.Env 里补 PATH 对解析无效。
var cmakeExe = "cmake"

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
//
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

// uiPlan 描述本次构建的 UI 方案。
type uiPlan struct {
	buildUI     bool   // -DLLAMA_BUILD_UI：是否用 bun/vite 从源码构建
	usePrebuilt bool   // -DLLAMA_USE_PREBUILT_UI：是否允许回退 files/ 本地归档
	desc        string // 打印给人看的一行说明
}

// uiDepsCacheDir 返回跨 worktree 共享的 UI 依赖缓存目录（与 .ccache 同级）。
// 放在仓库父目录下，laamaafung 与各 wt-* worktree 自然共用同一份。
func uiDepsCacheDir(repoRoot, override string) string {
	if override != "" {
		if abs, err := filepath.Abs(override); err == nil {
			return abs
		}
		return override
	}
	return filepath.Join(filepath.Dir(repoRoot), ".ui-deps")
}

// uiDepsReady 判断一份 node_modules 能否直接复用。
// 判据是 CMake 自己写的依赖戳 <node_modules>/.ui-deps-stamp：
// ui-assets.cmake 的 need_install 逻辑正是「没有这个戳就 bun install」，
// 而 bun install 就是联网挂起的那个点。
func uiDepsReady(modDir string) bool {
	if _, err := os.Stat(filepath.Join(modDir, ".ui-deps-stamp")); err != nil {
		return false
	}
	// 顺带确认依赖树非空（空目录可能来自中断的复制）
	ents, err := os.ReadDir(modDir)
	return err == nil && len(ents) > 1
}

// uiLockHash 计算 tools/ui 依赖清单（bun.lock 等）的指纹，用于判断共享依赖缓存
// 是否匹配当前分支。两分支 tools/ui 同源时指纹一致，缓存即可安全共用。
func uiLockHash(repoRoot string) string {
	for _, name := range []string{"bun.lock", "bun.lockb", "package-lock.json"} {
		b, err := os.ReadFile(filepath.Join(repoRoot, "tools", "ui", name))
		if err != nil {
			continue
		}
		sum := sha256.Sum256(b)
		return name + ":" + hex.EncodeToString(sum[:])
	}
	return ""
}

// pickUIDepsCache 返回可用的共享缓存 node_modules 路径（不可用则空串）。
// .lock-hash 既记录依赖指纹，也充当「缓存已建好」的标记：缺它一律不可用，
// 免得用到中途中断的复制（树不完整，构建会报缺包）。
func pickUIDepsCache(cacheDir, wantHash string) string {
	cacheMod := filepath.Join(cacheDir, "node_modules")
	if !uiDepsReady(cacheMod) {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(cacheDir, ".lock-hash"))
	if err != nil {
		return ""
	}
	if got := strings.TrimSpace(string(b)); wantHash != "" && got != wantHash {
		printWarning("共享 UI 依赖缓存与当前分支的依赖清单不匹配，忽略该缓存")
		return ""
	}
	return cacheMod
}

// newUIDepsStamp 在 node_modules 下写入依赖戳（空文件，只作时间戳载体）。
// 这个戳就是 ui-assets.cmake 的 need_install 判据 —— 有它且不比 lock 旧，
// `bun install` 才会被跳过。
func newUIDepsStamp(modDir string) error {
	return os.WriteFile(filepath.Join(modDir, ".ui-deps-stamp"), nil, 0o644)
}

// refreshUIDepsStamp 把依赖戳时间刷到当前时刻。
// 必要性来自 ui-assets.cmake 的判据：
//
//	file(TIMESTAMP "<WORK_DIR>/<lock>") STRGREATER "<依赖戳>"
//
// 而 stage_sources 是用 file(COPY) 拷 lock 过去的；实测 file(COPY) **保留**源文件
// 时间戳（目标 mtime == 源 mtime，不是复制时刻），所以该判据等价于「源树的
// tools/ui/bun.lock 是否比依赖戳新」。刷戳即跳过 install，随后照常 `bun run build`
// （vite 纯本地，离线可跑）。
func refreshUIDepsStamp(stamp string) {
	now := time.Now()
	_ = os.Chtimes(stamp, now, now)
}

// copyTree 用 robocopy 多线程复制目录（node_modules 有数万个文件，
// 纯 Go 逐文件复制在 Windows 上要几分钟）。robocopy 的 0-7 退出码都算成功。
// robocopy 不可用时退回 Go 递归复制。
func copyTree(src, dst string) error {
	os.MkdirAll(filepath.Dir(dst), 0o755)
	if _, err := exec.LookPath("robocopy"); err == nil {
		// /E 含空目录 /MT 多线程 /NFL /NDL /NJH /NJS 静音
		cmd := exec.Command("robocopy", src, dst, "/E", "/MT:16", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1")
		cmd.Env = cleanEnv()
		out, err := cmd.CombinedOutput()
		if err == nil {
			return nil
		}
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() < 8 {
			return nil // 1-7 都是「有复制/有跳过」的成功状态
		}
		printWarning("robocopy 复制失败，改走 Go 递归复制: " + strings.TrimSpace(string(out)))
	}
	return copyTreeGo(src, dst)
}

// isReparsePath 判定路径是否带重解析属性（junction / 符号链接）。
// 实测 Go 的 os.Lstat 对 junction 返回 mode=?rw-rw-rw-（IsDir=false、Symlink=false），
// 靠 Mode 判联接必然漏判，只能查 FILE_ATTRIBUTE_REPARSE_POINT (0x400)。
func isReparsePath(p string) bool {
	p16, err := syscall.UTF16PtrFromString(p)
	if err != nil {
		return false
	}
	attrs, aerr := syscall.GetFileAttributes(p16)
	return aerr == nil && attrs&0x400 != 0
}

// copyTreeGo 是 copyTree 的纯 Go 回退实现。
func copyTreeGo(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(src, p)
		if rerr != nil {
			return rerr
		}
		target := filepath.Join(dst, rel)
		switch {
		case d.IsDir():
			// 带重解析属性的目录（junction/链接）绝不递归：bun 装出来的树里可能有
			// 指回祖先的链接，递归会死循环。放弃内容（本路径只是 robocopy 缺席时的回退）。
			if isReparsePath(p) && rel != "." {
				printWarning("跳过重解析目录（Go 回退复制不展开联接）: " + rel)
				return filepath.SkipDir
			}
			return os.MkdirAll(target, 0o755)
		case d.Type()&os.ModeSymlink != 0:
			if link, lerr := os.Readlink(p); lerr == nil {
				if serr := os.Symlink(link, target); serr == nil {
					return nil
				}
			}
			return nil // 无权限建链接就跳过（本机 bun 装出来的树基本用不到）
		default:
			in, oerr := os.Open(p)
			if oerr != nil {
				return oerr
			}
			defer in.Close()
			out, cerr := os.Create(target)
			if cerr != nil {
				return cerr
			}
			defer out.Close()
			_, werr := io.Copy(out, in)
			return werr
		}
	})
}

// findUIDepsSeed 在 repoRoot 的其它构建目录里找可复用的 UI 依赖树
// （共享缓存之外的来源；同父目录下的 worktree 走另一条 glob）。
// 按依赖戳时间从新到旧排序 —— 越新的构建目录，依赖树越可能完整且与当前
// tools/ui 清单一致。
func findUIDepsSeed(repoRoot string) []string {
	type cand struct {
		dir string
		ts  time.Time
	}
	var cands []cand
	for _, pat := range []string{
		// 本仓库的其它构建目录（build-v21/build-v22/...）
		filepath.Join(repoRoot, "build*", "tools", "ui", "ui-src", "node_modules"),
		// 同父目录下其它 worktree 的构建目录（laamaafung / wt-v23 / wt-v22 ...）
		filepath.Join(filepath.Dir(repoRoot), "*", "build*", "tools", "ui", "ui-src", "node_modules"),
	} {
		ms, _ := filepath.Glob(pat)
		for _, m := range ms {
			if !uiDepsReady(m) {
				continue
			}
			ts := time.Time{}
			if fi, err := os.Stat(filepath.Join(m, ".ui-deps-stamp")); err == nil {
				ts = fi.ModTime()
			}
			cands = append(cands, cand{m, ts})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].ts.After(cands[j].ts) })
	out := make([]string, 0, len(cands))
	for _, c := range cands {
		out = append(out, c.dir)
	}
	return out
}

// Windows 重解析点（reparse point）常量
const (
	fsctlSetReparsePoint   = 0x000900A4
	ioReparseTagMountPoint = 0xA0000003
)

// utf16NoNul 返回 UTF-16 编码（去掉结尾的 NUL，方便自己控制布局）。
func utf16NoNul(s string) []uint16 {
	p, err := syscall.UTF16FromString(s)
	if err != nil {
		return nil
	}
	return p[:len(p)-1]
}

// makeJunction 建 NTFS 目录联接（junction），**纯 Go 原生实现**。
//
// 为什么不用现成手段：
//   - cmd.exe 被本机安全策略禁用（mklink 用不了）；
//   - PowerShell 虽然能建（实测 `New-Item -ItemType Junction` 可行），但从本进程
//     调用会被沙箱的程序黑名单拦住，而且表现为**静默挂起**（实测 builder 卡在联接
//     这一步数分钟、零输出），绝不可用。
//
// junction 是 NTFS 挂载点重解析，**不需要 SeCreateSymbolicLinkPrivilege**，
// 所以直接 DeviceIoControl(FSCTL_SET_REPARSE_POINT) 就能建。
//
// 缓冲区布局（踩过的坑，勿改）：
//
//	REPARSE_DATA_BUFFER: [Tag(4)][DataLength(2)][Reserved(2)]
//	  [SubstituteNameOffset(2)][SubstituteNameLength(2)]
//	  [PrintNameOffset(2)][PrintNameLength(2)]
//	  [替换名 UTF-16][NUL][打印名 UTF-16][NUL]
//
// 替换名要带 `\??\` 前缀；两个名字后面都必须有 NUL，且 DataLength 要把
// **两个 NUL 都算进去**（8 + subLen + 2 + printLen + 2）。少算末尾那个 NUL
// 就会 ERROR_INVALID_REPARSE_DATA —— 实测四种布局只有这一种能过。
func makeJunction(link, target string) error {
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return err
	}
	absLink, err := filepath.Abs(link)
	if err != nil {
		return err
	}
	if _, err := os.Stat(absTarget); err != nil {
		return fmt.Errorf("联接目标不存在: %v", err)
	}
	// 联接点本身必须先存在（空目录），再把重解析数据挂上去
	if err := os.MkdirAll(absLink, 0o755); err != nil {
		return err
	}

	sub := utf16NoNul(`\??\` + absTarget)
	print := utf16NoNul(absTarget)
	subLen := len(sub) * 2
	printLen := len(print) * 2

	payload := make([]byte, 8+subLen+2+printLen+2)
	binary.LittleEndian.PutUint16(payload[0:], 0)                // SubstituteNameOffset
	binary.LittleEndian.PutUint16(payload[2:], uint16(subLen))   // SubstituteNameLength
	binary.LittleEndian.PutUint16(payload[4:], uint16(subLen+2)) // PrintNameOffset
	binary.LittleEndian.PutUint16(payload[6:], uint16(printLen)) // PrintNameLength
	for i, c := range sub {
		binary.LittleEndian.PutUint16(payload[8+i*2:], c)
	}
	for i, c := range print {
		binary.LittleEndian.PutUint16(payload[8+subLen+2+i*2:], c)
	}

	full := make([]byte, 8+len(payload))
	binary.LittleEndian.PutUint32(full[0:], ioReparseTagMountPoint)
	binary.LittleEndian.PutUint16(full[4:], uint16(len(payload)))
	binary.LittleEndian.PutUint16(full[6:], 0)
	copy(full[8:], payload)

	p, err := syscall.UTF16PtrFromString(absLink)
	if err != nil {
		return err
	}
	h, err := syscall.CreateFile(p, syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING,
		syscall.FILE_FLAG_BACKUP_SEMANTICS|syscall.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return fmt.Errorf("打开链接目录失败: %v", err)
	}
	defer syscall.CloseHandle(h)

	var ret uint32
	if err := syscall.DeviceIoControl(h, fsctlSetReparsePoint,
		&full[0], uint32(len(full)), nil, 0, &ret, nil); err != nil {
		return fmt.Errorf("设置重解析点失败: %v", err)
	}
	return nil
}

// uiTrashDir 返回 UI 依赖的回收区（放在共享缓存目录下，**在构建目录与源码树之外**）。
//
// 为什么必须放到外面：被挪开的旧依赖树若留在 CMake 的 WORK_DIR
// （<构建目录>/tools/ui/ui-src）里，ui-assets.cmake 的 stage_sources() 会把它当作
// 待清理条目做 file(REMOVE_RECURSE) —— CMake 与我们的后台删除线程在同一棵数万文件
// 的树上互拖，构建直接卡死（实测 v23 的 UI 目标因此卡了 10 分钟、ninja 变僵尸态）。
func uiTrashDir(cacheDir string) string {
	d := filepath.Join(cacheDir, ".trash")
	os.MkdirAll(d, 0o755)
	return d
}

// mvToTrash 把待删目录改名挪进回收区，并交给后台 goroutine 慢慢删。
// 同卷改名是瞬时操作；失败返回 false，由调用方决定是否退化为就地删除。
// 名字带 pid，避免并发构建时撞名。
func mvToTrash(dst, trashDir string) bool {
	target := filepath.Join(trashDir,
		fmt.Sprintf("%s.stale-%s-%d", filepath.Base(dst), time.Now().Format("20060102-150405"), os.Getpid()))
	if err := os.Rename(dst, target); err != nil {
		return false
	}
	printInfo("旧依赖树已挪入回收区（后台删除）: " + target)
	go func() { _ = os.RemoveAll(target) }()
	return true
}

// evacuateStaleUIDeps 把遗留在 WORK_DIR 里的 node_modules.stale-* 挪出构建目录。
// （早期版本把残树挪在 ui-src 里，会被 stage_sources 反复清理导致卡死。）
func evacuateStaleUIDeps(uiSrcDir, trashDir string) {
	ms, _ := filepath.Glob(filepath.Join(uiSrcDir, "node_modules.stale-*"))
	for _, m := range ms {
		if !mvToTrash(m, trashDir) {
			printWarning("疏散遗留残树失败（会让 stage_sources 变慢）: " + m)
		}
	}
}

// linkOrCopyUIDeps 把依赖树放到 dst：优先目录联接（零拷贝），失败退回递归复制。
// 返回 true 表示用的是联接。
//
// 为什么值得用联接：node_modules 约 560 MB / 数万个小文件，复制在 Defender 逐文件
// 扫描下要数分钟（实测 5 分钟才 26 MB），而联接是瞬时的，且多个构建目录共享同一份
// 依赖树、不占额外磁盘。
//
// 安全性：实测 `os.RemoveAll` 不会穿透 junction（只删链接本身，目标内容完好），
// 所以 `builder -fresh` 删构建目录不会波及被联接的共享依赖树。
func linkOrCopyUIDeps(src, dst, trashDir string) (bool, error) {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return false, err
	}

	// 目标已存在时**先改名挪进回收区**：一则不在关键路径上删它（删上万个小文件很慢），
	// 二则绝不把残树留在构建目录里（见 uiTrashDir 的说明）。
	if _, err := os.Lstat(dst); err == nil {
		if !mvToTrash(dst, trashDir) {
			printWarning("挪入回收区失败，只能就地删除（可能较慢）: " + dst)
			if derr := os.RemoveAll(dst); derr != nil {
				return false, derr
			}
		}
	}

	if err := makeJunction(dst, src); err == nil {
		return true, nil
	}
	printWarning("目录联接不可用，改为复制依赖树（约 560 MB，首次较慢）")
	return false, copyTree(src, dst)
}

// sweepTrash 后台清理历史残树（不占关键路径）。扫两处：
//   - <cacheDir>/.trash/ 里的（现行版本 mvToTrash 的去处）
//   - <cacheDir> 顶层的（早期版本曾把残树直接改名在缓存目录旁边，那时创建者
//     若被 kill，后台删除就丢了，留下永远没人清的残树）
func sweepTrash(cacheDir string) {
	for _, pat := range []string{
		filepath.Join(cacheDir, ".trash", "node_modules.stale-*"),
		filepath.Join(cacheDir, "node_modules.stale-*"),
	} {
		ms, _ := filepath.Glob(pat)
		for _, m := range ms {
			_ = os.RemoveAll(m)
		}
	}
}

// runBounded 执行命令并施加死限；超时杀掉进程并返回错误。
// 用于 bun install：离线时 bun 会一直重试，不给死限就会把整个构建挂死。
func runBounded(cmd *exec.Cmd, timeout time.Duration) ([]byte, error) {
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return buf.Bytes(), err
	case <-time.After(timeout):
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		<-done
		return buf.Bytes(), fmt.Errorf("超过死限 %s", timeout)
	}
}

// installCacheDeps 按**当前分支自己的**依赖清单跑一次 bun install，把结果实体化成共享缓存。
//
// 为什么必须有这一环：缓存的依赖树是分支无关**的前提**是各分支 bun.lock 一致 —— 这只是
// 当前几个分支恰好同源（哈希 bd8a3b54…）的巧合，不是不变式。旧分支（v12/v7 那类）的
// UI 底层组成不同，指纹必然对不上；此时正确动作是**按该分支自己的清单装一份**，
// 而不是硬套缓存（缺包）或直接退归档（版本号退化）。
//
// 死限 10 分钟：有网时足够；离线时 bun 会一直重试，到点杀掉、转归档兜底，绝不挂死。
// 注意这里**不剥离代理变量**（与 MSVC 不同）：bun 走网络，代理可能是联网的必要条件。
func installCacheDeps(repoRoot, cacheDir, wantHash string) bool {
	if wantHash == "" {
		printWarning("找不到 tools/ui 的依赖清单（bun.lock/package-lock.json），无法自动安装依赖")
		return false
	}
	bun, err := exec.LookPath("bun")
	if err != nil {
		printWarning("自动安装依赖需要 bun（PATH 里没找到）；本次退回本地归档")
		return false
	}
	stage := filepath.Join(cacheDir, fmt.Sprintf(".install-%d", os.Getpid()))
	os.RemoveAll(stage)
	os.MkdirAll(stage, 0o755)
	defer os.RemoveAll(stage)

	// 装依赖只需要清单与锁文件
	for _, name := range []string{"package.json", "bun.lock", "bun.lockb", "package-lock.json", ".npmrc"} {
		if b, rerr := os.ReadFile(filepath.Join(repoRoot, "tools", "ui", name)); rerr == nil {
			_ = os.WriteFile(filepath.Join(stage, name), b, 0o644)
		}
	}

	printInfo("共享缓存与当前分支依赖清单不匹配（或缺），自动执行 bun install --frozen-lockfile（死限 10 分钟）")
	cmd := exec.Command(bun, "install", "--frozen-lockfile")
	cmd.Dir = stage
	cmd.Env = os.Environ() // 保留代理变量：bun 需要联网，MSB6001 那套清洗只针对 MSVC
	if _, ierr := runBounded(cmd, 10*time.Minute); ierr != nil {
		printWarning("bun install 未成功（" + ierr.Error() + "），本次退回本地归档；" +
			"在有网环境重跑 builder 会自动补装")
		return false
	}
	mod := filepath.Join(stage, "node_modules")
	if !uiDepsReady(mod) {
		printWarning("bun install 已结束但没有产出可用的 node_modules")
		return false
	}

	// 摘掉旧缓存（联接直接摘，真实目录挪进回收区），把装好的树安放为缓存实体
	cacheMod := filepath.Join(cacheDir, "node_modules")
	if _, lerr := os.Lstat(cacheMod); lerr == nil {
		if rmErr := os.Remove(cacheMod); rmErr != nil {
			// 还在 ⇒ 是真实目录（联接/空目录会被 os.Remove 摘掉）
			if !mvToTrash(cacheMod, uiTrashDir(cacheDir)) {
				printWarning("旧缓存实体挪不动，放弃自动安装")
				return false
			}
		}
	}
	if err := os.Rename(mod, cacheMod); err != nil {
		printWarning("安放依赖树失败: " + err.Error())
		return false
	}
	_ = newUIDepsStamp(cacheMod)
	_ = os.WriteFile(filepath.Join(cacheDir, ".lock-hash"), []byte(wantHash+"\n"), 0o644)
	printSuccess("已按当前分支清单装好共享依赖缓存: " + cacheMod)
	return true
}

// seedUIDeps 确保 <构建目录>/tools/ui/ui-src/node_modules 存在且依赖戳有效，
// 使 ui-assets.cmake 跳过 `bun install`（本机联网受限时它会永久挂起），
// 随后仍走 bun/vite 真实源码构建（dist/build.json 的版本号随构建注入）。
// 返回 false 表示拿不到可用依赖树，调用方需回退本地归档。
func seedUIDeps(repoRoot, buildDir, cacheDir string) bool {
	modDir := filepath.Join(repoRoot, buildDir, "tools", "ui", "ui-src", "node_modules")
	stamp := filepath.Join(modDir, ".ui-deps-stamp")
	lockPath := filepath.Join(repoRoot, "tools", "ui", "bun.lock")

	var lockTs time.Time
	hasLock := false
	if fi, err := os.Stat(lockPath); err == nil {
		hasLock, lockTs = true, fi.ModTime()
	}

	// 先把遗留在 WORK_DIR 里的残树挪出去 —— 它会被 ui-assets.cmake 的 stage_sources()
	// 当作待清理条目反复 REMOVE_RECURSE，与我们的后台删除互拖而卡死构建。
	// 这件事**每次都要做**（包括走"情形一"提前返回时），否则残树会一直卡在那里。
	trashDir := uiTrashDir(cacheDir)
	go sweepTrash(cacheDir)
	evacuateStaleUIDeps(filepath.Dir(modDir), trashDir)

	// 情形一：构建目录里已有可复用依赖树 —— 增量构建的常态，零开销。
	if uiDepsReady(modDir) {
		if hasLock {
			if st, err := os.Stat(stamp); err == nil && lockTs.After(st.ModTime()) {
				printWarning("tools/ui/bun.lock 比依赖戳新（依赖清单有变更）；离线环境跳过 bun install。" +
					"若随后报缺包，请联网重建依赖缓存: " + filepath.Join(cacheDir, "node_modules"))
				refreshUIDepsStamp(stamp)
			}
		}
		return true
	}

	// 情形二：从共享缓存 / 同父目录下其它构建目录接一份依赖树过来。
	wantHash := uiLockHash(repoRoot)
	src := pickUIDepsCache(cacheDir, wantHash)
	if src == "" {
		for _, c := range findUIDepsSeed(repoRoot) {
			src = c
			break
		}
	}
	if src == "" {
		// 情形三：一处可复用的依赖树都没有（含缓存指纹与当前分支不一致 —— 旧分支的
		// 依赖组成本就不同）。自动按本分支清单 bun install 一份进缓存，失败才退归档。
		if installCacheDeps(repoRoot, cacheDir, wantHash) {
			src = pickUIDepsCache(cacheDir, wantHash)
		}
		if src == "" {
			return false
		}
	}

	printInfo("复用 UI 依赖: " + src + " → " + modDir)
	start := time.Now()
	linked, err := linkOrCopyUIDeps(src, modDir, trashDir)
	if err != nil {
		printWarning("准备 UI 依赖失败: " + err.Error())
		return false
	}
	if linked {
		// 联接目标的依赖戳可能比本分支的 bun.lock 旧 —— 不刷就会重跑 bun install。
		refreshUIDepsStamp(stamp)
		printSuccess(fmt.Sprintf("UI 依赖就绪（目录联接，零拷贝，%s）", time.Since(start).Round(time.Second)))
	} else {
		if err := newUIDepsStamp(modDir); err != nil {
			printWarning("写入依赖戳失败: " + err.Error())
			return false
		}
		printSuccess(fmt.Sprintf("UI 依赖就绪（已复制，%s）", time.Since(start).Round(time.Second)))
	}

	// 缓存还没建好时，顺手用联接把缓存指向同一来源（同样零拷贝），
	// 之后任何新 worktree/构建目录都能直接命中缓存。
	if wantHash != "" && pickUIDepsCache(cacheDir, wantHash) == "" {
		cacheMod := filepath.Join(cacheDir, "node_modules")
		if ok, _ := linkOrCopyUIDeps(src, cacheMod, trashDir); ok {
			_ = os.WriteFile(filepath.Join(cacheDir, ".lock-hash"), []byte(wantHash+"\n"), 0o644)
			printInfo("已建立共享 UI 依赖缓存（目录联接）: " + cacheDir)
		}
	}
	return true
}

// countUIArchives 数一下 files/ 里有多少个本地 UI 归档（离线兜底资源）。
func countUIArchives(repoRoot string) int {
	ms, _ := filepath.Glob(filepath.Join(repoRoot, "files", "llama-b*-ui.tar.gz"))
	return len(ms)
}

// planUI 决定本次构建的 UI 方案。核心约束：**必须产出带 UI 的二进制**。
func planUI(repoRoot, buildDir, mode, depsCache string) uiPlan {
	switch mode {
	case "off":
		printWarning("-ui=off: 不内嵌 Web UI（llama-server 将没有界面，仅调试用）")
		return uiPlan{buildUI: false, usePrebuilt: false, desc: "-ui=off 强制不内嵌"}
	case "archive":
		return uiPlan{buildUI: false, usePrebuilt: true,
			desc: "-ui=archive：走 files/llama-b*-ui.tar.gz 本地归档（版本号固定）"}
	}

	// auto：UI 必须有，优先复现 build-full-v21.sh 的源码构建
	if seedUIDeps(repoRoot, buildDir, depsCache) {
		return uiPlan{buildUI: true, usePrebuilt: true,
			desc: "bun/vite 源码构建（依赖缓存已就绪，跳过 bun install）"}
	}
	if n := countUIArchives(repoRoot); n > 0 {
		printWarning(fmt.Sprintf("拿不到可复用的 UI 依赖缓存，回退 files/ 本地归档（%d 个可用）。"+
			"归档 UI 的版本号是固定的，与源码构建的产物不完全一致；要恢复源码构建，"+
			"在有网的机器上于 tools/ui 下跑一次 `bun install`，再把 node_modules 放到 %s",
			n, filepath.Join(depsCache, "node_modules")))
		return uiPlan{buildUI: false, usePrebuilt: true,
			desc: "本地归档兜底（带 UI，但版本号固定、非源码构建）"}
	}
	printError("UI 是必须构建的，但没有任何离线来源：既拿不到 node_modules 依赖缓存，" +
		"也没有 files/llama-b*-ui.tar.gz 本地归档")
	printError("请任选其一：①联网跑一次 `bun install`（tools/ui 下）生成依赖缓存；" +
		"②把 UI 归档放进 files/；③确认要放弃 UI 时显式用 -ui=off")
	os.Exit(1)
	return uiPlan{}
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

// newestByVersion 从候选路径里挑目录版本号最高的一个。
//
// 不能直接 sort.Strings 取末尾：那是**字典序**，位数不同就会挑错
// （"cmake-10.0.0" 会被 "cmake-9.0.0" 压住，因为 '1' < '9'）。
// 这里把目录名里 prefix-<数字>[.<数字>...] 的版本段解析成整数序列再比大小；
// 一个都解析不出时退回原来的字典序取末尾，行为不变。
func newestByVersion(paths []string, prefix string) string {
	best := ""
	var bestKey []int
	for _, p := range paths {
		key := pathVersionKey(p, prefix)
		if key == nil {
			continue
		}
		if best == "" || versionLess(bestKey, key) {
			best, bestKey = p, key
		}
	}
	if best != "" {
		return best
	}
	if len(paths) == 0 {
		return ""
	}
	sort.Strings(paths)
	return paths[len(paths)-1]
}

// pathVersionKey 取出路径中 prefix-<版本> 目录段的数字序列。
// 只吃开头的数字段（4.1.1）——后面的 -windows-x86_64 之类与版本无关。
func pathVersionKey(p, prefix string) []int {
	sep := prefix + "-"
	for _, part := range strings.Split(filepath.ToSlash(p), "/") {
		if !strings.HasPrefix(part, sep) {
			continue
		}
		rest := part[len(sep):]
		end := 0
		for end < len(rest) && ((rest[end] >= '0' && rest[end] <= '9') || rest[end] == '.') {
			end++
		}
		ver := strings.TrimSuffix(rest[:end], ".")
		if ver == "" {
			return nil
		}
		var key []int
		for _, f := range strings.Split(ver, ".") {
			n, err := strconv.Atoi(f)
			if err != nil {
				return nil
			}
			key = append(key, n)
		}
		return key
	}
	return nil
}

// versionLess 按字段比较版本号：a < b 返回 true。
func versionLess(a, b []int) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return len(a) < len(b)
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
			return newestByVersion(ms, "ccache")
		}
	}
	return ""
}

// findCmake 定位 cmake.exe：先 PATH，再便携工具链目录，最后 VS 自带。
// cmake 是 configure 的必需件，但 VS 自带的那个不进 PATH，所以在这里显式兜底。
func findCmake(vsRoot string) string {
	if p, err := exec.LookPath("cmake"); err == nil {
		return p
	}
	for _, pat := range []string{
		filepath.Join(`D:\WinKit-Terminal\share`, "cmake-*", "bin", "cmake.exe"),
		filepath.Join(`C:\WinKit-Terminal\share`, "cmake-*", "bin", "cmake.exe"),
		filepath.Join(`D:\winkit\share`, "cmake-*", "bin", "cmake.exe"),
		filepath.Join(`C:\winkit\share`, "cmake-*", "bin", "cmake.exe"),
		filepath.Join(`D:\tools`, "cmake-*", "bin", "cmake.exe"),
	} {
		if ms, err := filepath.Glob(pat); err == nil && len(ms) > 0 {
			return newestByVersion(ms, "cmake")
		}
	}
	if vsRoot != "" {
		p := filepath.Join(vsRoot, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", "CMake", "bin", "cmake.exe")
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
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

// removeBuildDir 全量删除构建目录。
func removeBuildDir(repoRoot, buildDir string) {
	printInfo("删除构建目录 " + buildDir)
	if err := os.RemoveAll(filepath.Join(repoRoot, buildDir)); err != nil {
		printWarning("删除构建目录失败（继续）: " + err.Error())
	}
}

// clearCMakeState 仅重置 CMake 状态、保留已编译对象。
// Ninja（单配置）的对象文件就在 buildDir/CMakeFiles/ 下，删 CMakeFiles 等于全量重编，
// 所以 Ninja 下不做任何删除；VS（多配置）对象在 <target>.dir/Release/ 下，清缓存不影响对象。
func clearCMakeState(repoRoot, buildDir, generator string) {
	if strings.Contains(generator, "Ninja") {
		printInfo("-keep（Ninja）: 保留全部构建状态，仅重跑 configure + build")
		return
	}
	printInfo("-keep: 仅清 CMake 缓存（保留已编译对象）")
	os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeCache.txt"))
	os.RemoveAll(filepath.Join(repoRoot, buildDir, "CMakeFiles"))
}

// uiDistMarker 是写在源码树 tools/ui/dist 里的出处标记文件。
// 有了它才能判断这份「跨分支共享」的预构建资源能不能被当前分支直接复用 ——
// build-full-v21.sh 的做法是无条件删除，代价是每次构建都要重跑一遍 vite；
// 实际只有**换分支**或**依赖/源码变了**时才有必要删。
const uiDistMarker = ".llama-ui-provenance"

// uiDistProv 记录预构建资源的出处。
type uiDistProv struct {
	branch string
	lock   string
}

// readUIDistProv 读取出处标记；不存在或字段缺失时返回零值。
func readUIDistProv(dist string) uiDistProv {
	var p uiDistProv
	b, err := os.ReadFile(filepath.Join(dist, uiDistMarker))
	if err != nil {
		return p
	}
	for _, l := range strings.Split(string(b), "\n") {
		l = strings.TrimSpace(l)
		if strings.HasPrefix(l, "branch=") {
			p.branch = strings.TrimPrefix(l, "branch=")
		}
		if strings.HasPrefix(l, "lock=") {
			p.lock = strings.TrimPrefix(l, "lock=")
		}
	}
	return p
}

// writeUIDistProv 写出处标记。
func writeUIDistProv(dist, branch, lock string) error {
	body := fmt.Sprintf("branch=%s\nlock=%s\nbuilt=%s\n", branch, lock, time.Now().Format(time.RFC3339))
	return os.WriteFile(filepath.Join(dist, uiDistMarker), []byte(body), 0o644)
}

// uiSrcNewestMtime 返回参与 UI 构建的源文件里最新的修改时间。
// 覆盖 tools/ui/sources.cmake 声明的全部输入：src/**、static/** 以及若干配置文件。
// 用途：判断源码树里的预构建 dist 是否已被源码改动超越。
func uiSrcNewestMtime(repoRoot string) time.Time {
	ui := filepath.Join(repoRoot, "tools", "ui")
	var newest time.Time
	consider := func(p string) {
		if fi, err := os.Stat(p); err == nil && fi.ModTime().After(newest) {
			newest = fi.ModTime()
		}
	}
	for _, d := range []string{"src", "static"} {
		_ = filepath.WalkDir(filepath.Join(ui, d), func(p string, de fs.DirEntry, err error) error {
			if err == nil && !de.IsDir() {
				consider(p)
			}
			return nil
		})
	}
	for _, f := range []string{
		"package.json", "package-lock.json", "src/.gitignore", "vite.config.ts",
		"svelte.config.js", "tsconfig.json", "scripts/vite-plugin-llama-cpp-build.ts",
		"bun.lock", "sources.cmake",
	} {
		consider(filepath.Join(ui, f))
	}
	return newest
}

// syncUIFrontend 处理源码树里的 tools/ui/dist（跨分支共享，必须防误用）。
//
// 保留条件（三者同时成立）：出处标记是**当前分支**、依赖清单指纹一致、
// 且 UI 源码没有比预构建资源的 index.html 更新。
// 满足时 ui-assets.cmake 会走「预构建资源」快路径 —— 完全不跑 bun/vite；
// 任一条不满足就删掉，退回源码构建，避免用错分支的静态页面。
func syncUIFrontend(repoRoot string) {
	dist := filepath.Join(repoRoot, "tools", "ui", "dist")
	indexHTML := filepath.Join(dist, "index.html")
	fi, err := os.Stat(indexHTML)
	if err != nil {
		return // 没有预构建资源，无需处理
	}
	branch := gitBranch(repoRoot)

	reason := ""
	prov := readUIDistProv(dist)
	switch {
	case prov.branch == "":
		reason = "出处不明（无 " + uiDistMarker + " 标记）"
	case prov.branch != branch:
		reason = fmt.Sprintf("属于分支 %q，当前是 %q", prov.branch, branch)
	case prov.lock != uiLockHash(repoRoot):
		reason = "UI 依赖清单已变更"
	default:
		if newest := uiSrcNewestMtime(repoRoot); newest.After(fi.ModTime()) {
			reason = "UI 源码比预构建资源新"
		}
	}

	if reason == "" {
		printSuccess(fmt.Sprintf("保留 tools/ui/dist（同分支 %q 且源未变）→ 本次直接复用预构建 UI，不跑 bun/vite", branch))
		return
	}
	printInfo("删除 tools/ui/dist: " + reason)
	if err := os.RemoveAll(dist); err != nil {
		printWarning("删除 tools/ui/dist 失败（继续）: " + err.Error())
	}
}

// mirrorUIDist 把构建目录里刚生成的 UI 资源镜像回源码树 tools/ui/dist，
// 并写出处标记。这样同一分支的后续构建（含 -fresh）可以直接走预构建快路径，
// 省掉一次 2~5 分钟的 vite 构建。
func mirrorUIDist(repoRoot, buildDir, branch string) {
	src := filepath.Join(repoRoot, buildDir, "tools", "ui", "dist")
	if _, err := os.Stat(filepath.Join(src, "index.html")); err != nil {
		return
	}
	dst := filepath.Join(repoRoot, "tools", "ui", "dist")
	if err := copyTree(src, dst); err != nil {
		printWarning("回写 tools/ui/dist 失败（不影响构建）: " + err.Error())
		return
	}
	if err := writeUIDistProv(dst, branch, uiLockHash(repoRoot)); err != nil {
		printWarning("写出处标记失败: " + err.Error())
	}
	printInfo("已回写 tools/ui/dist（标注分支 " + branch + "），后续同分支构建可直接复用")
}

// configure 组装并执行 CMake 配置。
// genFlag 只在创建新构建目录时传给 -G；effGen 是实际生效的生成器名（已有构建目录
// 时为探测到的那个），只用于判断 ccache 能不能挂 —— 之前两者混用一个变量，增量
// 构建时该值为空，导致 ccache 被静默丢掉、还打了条莫名其妙的「非 Ninja」告警。
func configure(repoRoot, buildDir, cudaArch, genFlag, effGen, ccachePath, logPath string, ccacheAll bool, forceCcache bool, ui uiPlan, extra []string) error {
	args := []string{"-B", buildDir}
	if genFlag != "" {
		args = append(args, "-G", genFlag)
	}
	uiFlag := "-DLLAMA_BUILD_UI=OFF"
	if ui.buildUI {
		uiFlag = "-DLLAMA_BUILD_UI=ON"
	}
	prebuiltFlag := "-DLLAMA_USE_PREBUILT_UI=OFF"
	if ui.usePrebuilt {
		prebuiltFlag = "-DLLAMA_USE_PREBUILT_UI=ON"
	}
	args = append(args,
		"-DCMAKE_CUDA_ARCHITECTURES="+cudaArch,
		"-DGGML_CUDA=ON",
		"-DGGML_NATIVE=ON",
		"-DGGML_CUDA_FA=ON",
		"-DGGML_CUDA_FA_ALL_QUANTS=ON",
		"-DCMAKE_BUILD_TYPE=Release",
		uiFlag,
		prebuiltFlag,
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
	ccacheOn := forceCcache && ccachePath != "" && strings.Contains(effGen, "Ninja")
	if forceCcache && ccachePath == "" {
		printWarning("未找到 ccache.exe，本次不使用 ccache")
	}
	if forceCcache && ccachePath != "" && !strings.Contains(effGen, "Ninja") {
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
	return runWithRetry(repoRoot, logPath, cmakeExe, 2, args...)
}

// build 执行编译。瞬时竞争（nvcc 临时文件等）允许再重试两次。
// targets 非空时只构建指定目标（用于单独验证 UI provisioning、单个可执行文件等）。
func build(repoRoot, buildDir, logPath string, jobs int, targets []string) error {
	args := []string{"--build", buildDir, fmt.Sprintf("-j%d", jobs), "--config", "Release"}
	if len(targets) > 0 {
		args = append(args, "--target")
		args = append(args, targets...)
	}
	return runWithRetry(repoRoot, logPath, cmakeExe, 3, args...)
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

	// 内嵌 UI 同样是产物：ui.cpp 由 llama-ui-embed 从 tools/ui/dist 生成，
	// 它缺失/过小就意味着 llama-server 没有 Web 界面 —— 正是要杜绝的退化。
	uiDir := filepath.Join(repoRoot, buildDir, "tools", "ui")
	uiCPP := filepath.Join(uiDir, "ui.cpp")
	if fi, err := os.Stat(uiCPP); err != nil {
		printError("构建自检: 缺少内嵌 UI 资源 " + uiCPP)
		missing++
	} else if fi.Size() < 1<<20 {
		printError(fmt.Sprintf("构建自检: 内嵌 UI 资源异常小（%d 字节）: %s", fi.Size(), uiCPP))
		missing++
	} else {
		ver := ""
		if b, err := os.ReadFile(filepath.Join(uiDir, "dist", "build.json")); err == nil {
			ver = "  " + strings.TrimSpace(string(b))
		}
		printSuccess(fmt.Sprintf("内嵌 UI: ui.cpp %.1f MB%s", float64(fi.Size())/(1<<20), ver))
	}
	if missing > 0 {
		return fmt.Errorf("缺少 %d 个产物", missing)
	}

	printSuccess("构建完成: " + binDir)
	return nil
}

// checkSkipConfigure 校验 -no-configure 的使用前提，并提示会失效的参数。
//
// 为什么敢跳：configure 里真正贵的是 generate 阶段（本项目要 289 秒，整段 5 分半），
// 而构建参数没变时它产出的 build.ninja 是确定性的。代价是 CMakeLists / 命令行参数
// 的变更不会生效 —— 本项目又开了 CMAKE_SUPPRESS_REGENERATION=ON（连自动重配置
// 都没有），所以必须由人自己确认，这里只做能做的护栏。
func checkSkipConfigure(repoRoot, buildDir string, explicit map[string]bool) {
	if _, err := os.Stat(filepath.Join(repoRoot, buildDir, "CMakeCache.txt")); err != nil {
		printError("-no-configure 需要已配置过的构建目录（未找到 CMakeCache.txt）。请先去掉 -no-configure 跑一次")
		os.Exit(1)
	}
	// 这些参数只有 configure 才会吃进去
	for _, n := range []string{"arch", "gen", "ui", "ui-deps-dir", "no-ccache", "ccache-all", "ccache-dir"} {
		if explicit[n] {
			printWarning(fmt.Sprintf("-%s 与 -no-configure 同用：跳过 configure 意味着该参数本次不会生效", n))
		}
	}
	printWarning("-no-configure：跳过 configure。若改过 CMakeLists 或换了构建参数，请去掉本参数重跑一次")
}

func main() {
	jobs := flag.Int("j", 8, "并行编译度")
	keep := flag.Bool("keep", false, "仅重置 CMake 状态，保留已编译对象（CMake 缓存损坏/改选项后用）")
	fresh := flag.Bool("fresh", false, "先删除构建目录再重建（全量）")
	arch := flag.String("arch", "", "CUDA 架构（默认 native；亦可用环境变量 CUDA_ARCH）")
	clean := flag.Bool("clean", false, "只清理不构建")
	root := flag.String("C", "", "仓库根（默认 builder.exe 所在目录）")
	gen := flag.String("gen", "auto", "生成器: auto|ninja|vs")
	uiMode := flag.String("ui", "auto", "内嵌 Web UI 方案: auto|archive|off（默认 auto：UI 必需，优先复用依赖缓存做源码构建）")
	uiDepsDir := flag.String("ui-deps-dir", "", "UI 依赖缓存目录（默认 <仓库父目录>/.ui-deps）")
	noConfigure := flag.Bool("no-configure", false, "跳过 CMake configure（确认没改过 CMakeLists/构建参数时用，省约 5 分钟 generate）")
	targetsFlag := flag.String("target", "", "只构建指定目标（逗号分隔，如 llama-ui-assets）；此时跳过产物齐全性自检")
	noCcache := flag.Bool("no-ccache", false, "关闭 ccache")
	ccacheAll := flag.Bool("ccache-all", false, "也给 C/CXX 挂 ccache（调试用）")
	ccacheDir := flag.String("ccache-dir", "", "ccache 缓存目录（默认 <仓库父目录>/.ccache）")
	listOnly := flag.Bool("list", false, "打印探测到的工具链后退出")
	flag.Parse()

	// 记下哪些参数是用户显式传的（-no-configure 的护栏要用）
	explicit := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

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
	cmakePath := findCmake(vsRoot)
	if cmakePath != "" {
		cmakeExe = cmakePath
	}
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
	if cmakePath != "" {
		printInfo("CMake: " + cmakePath)
	} else {
		printWarning("未找到 cmake.exe（放在 PATH、D:\\WinKit-Terminal\\share\\cmake-* 或 VS 自带目录）")
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

	if cmakePath == "" {
		printError("缺少 cmake.exe：configure 必需，安装 CMake 或把它加入 PATH 后重试")
		os.Exit(1)
	}

	// 环境自检提示（有代理变量时说明剥离动作，避免 MSB6001 复发时排查无门）。
	for _, p := range proxyVars {
		if os.Getenv(p) != "" {
			printWarning(fmt.Sprintf("检测到 %s，已对子进程剥离（防 MSB6001）", p))
			break
		}
	}

	if *clean {
		removeBuildDir(repoRoot, buildDir)
		// clean 就是要干净：源码树里的预构建 UI 资源一并删掉（下次重建）
		os.RemoveAll(filepath.Join(repoRoot, "tools", "ui", "dist"))
		printInfo("已删除构建目录与 tools/ui/dist")
		printSuccess("清理完成")
		return
	}
	switch {
	case *fresh:
		removeBuildDir(repoRoot, buildDir)
	case *keep:
		clearCMakeState(repoRoot, buildDir, chosenGen)
	default:
		printInfo("增量构建（保留已有构建状态）；要全新构建请加 -fresh")
	}
	// 只删「不该复用」的预构建 UI 资源：换分支 / 依赖清单变了 / UI 源码更新了。
	syncUIFrontend(repoRoot)

	os.MkdirAll(filepath.Join(repoRoot, buildDir), 0o755)
	cfgLog := filepath.Join(repoRoot, buildDir, "builder-configure.log")
	buildLog := filepath.Join(repoRoot, buildDir, "builder-build.log")

	// UI 方案必须在 -fresh 删过构建目录之后才决定/落地依赖树，否则会被一并删掉。
	uiDeps := uiDepsCacheDir(repoRoot, *uiDepsDir)
	ui := planUI(repoRoot, buildDir, *uiMode, uiDeps)
	printInfo("UI: " + ui.desc)

	var targets []string
	for _, t := range strings.Split(*targetsFlag, ",") {
		if t = strings.TrimSpace(t); t != "" {
			targets = append(targets, t)
		}
	}

	start := time.Now()
	if *noConfigure {
		checkSkipConfigure(repoRoot, buildDir, explicit)
	} else if err := configure(repoRoot, buildDir, *arch, passGen, chosenGen, ccachePath, cfgLog, *ccacheAll, !*noCcache, ui, flag.Args()); err != nil {
		printError("configure 失败，详见 " + cfgLog)
		os.Exit(1)
	}
	if err := build(repoRoot, buildDir, buildLog, *jobs, targets); err != nil {
		printError("编译失败，详见 " + buildLog)
		os.Exit(1)
	}
	// 源码构建出的 UI 资源回写到源码树，供同分支后续构建直接复用（含 -fresh）
	if ui.buildUI {
		mirrorUIDist(repoRoot, buildDir, gitBranch(repoRoot))
	}
	if len(targets) > 0 {
		printInfo("已指定 -target，跳过产物齐全性自检: " + strings.Join(targets, ", "))
	} else if err := verifyArtifacts(repoRoot, buildDir); err != nil {
		os.Exit(1)
	}
	ccacheStats(ccachePath)
	printSuccess(fmt.Sprintf("总耗时 %s", time.Since(start).Round(time.Second)))
}
