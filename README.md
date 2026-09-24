# laamaafung

由於 LlamaCpp 項目所有發布的版本序號單純就係提交次數累加卻非真正的生產就緒，故而，此分支的唯一目標是成為一個可驅動智能代理正常工作的穩定版本。

致力於從推理引擎側修復影響模型無法驅動智能代理勝任長程任務的所有問題：模型遞歸生成式的死循環及各種因選項組合未被邏輯正確處理導致的無故停止工作。

優質輸出 = （優質模型 + 優質模板 + 優質代理 + 優質引擎）* 正確參數

---

### 克隆指南

推薦優先克隆穩定分支 `v17`，適合穩定使用。

如須測試最新功能，可以克隆測試分支或開發分支 `master`。

- **克隆穩定分支（推薦稀疏模型使用）**：
  ```sh
  git clone -b v17 https://github.com/naamfung/laamaafung.git
  ```

- **回退版本（v12）**：
  如須使用舊版穩定分支 `v12`，可執行：
  ```sh
  git clone -b v12 https://github.com/naamfung/laamaafung.git
  ```

- **克隆開發分支**：
  ```sh
  git clone -b master https://github.com/naamfung/laamaafung.git
  ```

- **克隆 KVMem 分支（v21，尚未長期驗證穩定）**：
  KVMem（分層/稀疏 KV 記憶，見下文「KVMem」一節）目前只在 `v21` 分支上，適合需要在**有限顯存**下驅動超長上下文的場景：
  ```sh
  git clone -b v21 https://github.com/naamfung/laamaafung.git
  ```

- **克隆 KVMem + Prism 三值量化分支（v22，尚未長期驗證穩定）**：
  在 `v21` 的 KVMem 之上加入 **Prism 三值量化**支援（`PTQ1_0`／`PQ2_0` 權重；GGUF 帶 `prism.hadamard.*` 折疊元數據，推理時激活自動做同款旋轉、無需額外參數，見下文「啟動示例」的「Prism 三值量化」與「KVMem + Prism 三值量化」）。需要跑 Prism 三值模型時用此分支：
  ```sh
  git clone -b v22 https://github.com/naamfung/laamaafung.git
  ```

- **克隆 KVMem + Prism 三值量化 + TCQ/turbo1.5 KV 量化分支（v23，尚未長期驗證穩定）**：
  在 `v22` 之上加入 **TCQ（Trellis-Coded Quantization）KV 量化**：`turbo3_tcq`（3.25 bpv）／`turbo2_tcq`（2.25 bpv），以及 **turbo1.5 三值 KV 量化**：`turbo1.5`（2.25 bpv 有效載荷），見下文「啟動示例」的「TCQ KV 量化」與「turbo1.5 KV 量化」。需要更小的 KV cache 佔用時用此分支：
  ```sh
  git clone -b v23 https://github.com/naamfung/laamaafung.git
  ```

---

### 編譯指南（builder —— 標準生產構建流程）

構建統一走倉庫根目錄的 **builder**（Go 實現，源碼 `builder.go` 已分發至各分支；跨分支構建可用 `-C` 指定倉庫根）。builder 內置了舊腳本踩過的全部坑的處理：代理變量自動剝離（防 MSB6001）、MSVC 開發環境自建（INCLUDE/LIB/PATH 手工拼裝，無需 cmd.exe/vcvars）、CUDA 專屬 ccache 加速、Web UI 依賴兜底、產物齊全性自檢。

**工具鏈依賴**：

| 組件 | 用途 | 說明 |
|---|---|---|
| Go 編譯器 1.21+ | 編譯 builder 本身 | `go build -o builder.exe builder.go`。**builder.exe 為本地構建產物，不入倉庫**（.gitignore 已忽略），倉庫只跟蹤源碼 builder.go |
| Visual Studio 2022 | MSVC C/C++ 編譯器 + Windows SDK 10 | builder 自動探測安裝路徑並自建編譯環境；`-list` 可查看探測結果 |
| CUDA Toolkit 12.x | GPU 後端（nvcc） | 默認 `-DCMAKE_CUDA_ARCHITECTURES=native`，可用 `-arch` 覆蓋 |
| Ninja | 構建生成器 | builder 默認使用；換回 VS 生成器用 `-gen vs`（ccache 自動停用） |
| ccache 4.13+（可選） | CUDA 編譯緩存 | 僅包裝 nvcc（緩存目錄 `<倉庫父目錄>/.ccache`）；**必須保持 `-DGGML_CCACHE=OFF`**——包裝本機本地化 MSVC 的 cl.exe 會崩潰 |
| bun（可選） | Web UI 源碼構建 | 缺失時自動回退預構建 UI 資源，不會產出無 UI 的 llama-server |

**基本用法**（在倉庫根目錄執行）：

```sh
builder.exe              # 增量構建（默認；無構建目錄時即全新構建）
builder.exe -fresh       # 先刪構建目錄再全量重建
builder.exe -j 12        # 指定並行度
builder.exe -list        # 列出探測到的工具鏈/環境後退出
builder.exe clean        # 僅清理構建目錄與 ui/dist
```

**常用參數**：`-target T1,T2`（只構建指定目標）、`-keep`（僅重置 CMake 狀態）、`-arch 86`（覆蓋 CUDA 架構）、`-gen ninja|vs`（強制生成器）、`-no-ccache` / `-ccache-all`、`-ui auto|archive|off`（UI 三級兜底，絕不靜默產出無 UI 的二進制）、`-no-configure`、`-C <dir>`（指定倉庫根，用於 worktree 或跨分支構建，如 `./builder.exe -C ../wt-v23`）。

**構建目錄與產物**：構建目錄按分支命名（如 v26 分支 → `build-v26`）；產物在構建目錄下的 `bin` 目錄（llama-server、llama-cli、llama-bench、llama-perplexity、llama-quantize、llama-kvmem-server 等），每次構建結束自檢產物齊全性與內嵌 UI 體積。編譯日誌為構建目錄下的 `builder-build.log` 與 `builder-configure.log`；瞬時競爭錯誤（nvcc C1083 / MSB8066 / MSB6001）會自動重跑。

---

### 推薦模型

unsloth/Qwen-AgentWorld-35B-A3B 二零二六年六月廿五 / 原版 / 推薦IQ4及以上質量：
https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF/tree/main

mudler/Qwen-AgentWorld-35B-A3B-APEX / 原版 / 建議 APEX-I-Compact 或 APEX-Compact 及以上質量：
https://huggingface.co/mudler/Qwen-AgentWorld-35B-A3B-APEX-GGUF/tree/main

---

### 推薦模板

千問 3.5/3.6/AgentWorld 及以其為基座的衍生模型，建議使用「tmpl」目錄內的「Qwen-Agentic-EN / Qwen-Agentic-HON(S/T) 」模板。

---

### 啟動示例

Q80 + Q80：

```sh
GGML_CUDA_REGISTER_HOST=1 D:/Programs/llama-cpp-repos/laamaafung-v17/bin/Release/llama-server.exe \
--model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" \
--ctx-size 131072 --flash-attn on \
--reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" \
--reasoning-format deepseek --reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-repeat-penalty 1.1 --reasoning-repeat-last-n 256 \
--fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified \
--cache-type-k q8_0 --cache-type-v q8_0 \
--host 0.0.0.0 --port 8008 \
--batch-size auto --ubatch-size auto --load-mode mlock-ram --no-mmproj \
--cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 \
--temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 \
--jinja --spec-type draft-mtp --spec-draft-n-max 4 \
--cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action boost --cycle-boost-factor 0.5 \
--chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

Q80 + TURBO4：

```sh
GGML_CUDA_REGISTER_HOST=1 D:/Programs/llama-cpp-repos/laamaafung-v17/bin/Release/llama-server.exe \
--model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" \
--ctx-size 131072 --flash-attn on \
--reasoning on --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" \
--reasoning-format deepseek \
--fit 1 -ngl all -ngld all --n-cpu-moe 33 --threads 18 --threads-http 2 --parallel 1 --kv-unified \
--cache-type-k q8_0 --cache-type-v turbo4 \
--host 0.0.0.0 --port 8008 \
--batch-size auto --ubatch-size auto --ctx-checkpoints 42 --load-mode mlock-ram --no-mmproj \
--cache-prompt --cache-ram 8192 \
--temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 \
--reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-presence-penalty 1.07 \
--jinja --spec-type draft-mtp --spec-draft-n-max 4 \
--chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

TURBO4 + TURBO3：

```sh
GGML_CUDA_REGISTER_HOST=1 D:/Programs/llama-cpp-repos/laamaafung-v17/bin/Release/llama-server.exe \
--model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" \
--ctx-size 131072 --flash-attn on \
--reasoning on --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" \
--reasoning-format deepseek \
--fit on -ngl all -ngld all --n-cpu-moe 33 --threads 10 --threads-http 2 --parallel 1 --kv-unified \
--cache-type-k turbo4 --cache-type-v turbo3 \
--host 0.0.0.0 --port 8008 \
--batch-size auto --ubatch-size auto --ctx-checkpoints 42 --load-mode mlock-ram --no-mmproj \
--cache-prompt --cache-ram 8192 \
--temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 \
--reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-presence-penalty 1.07 \
--jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose \
--chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

KVMem（分層/稀疏 KV 記憶，`llama-server` 本體直接驅動，須 v21 分支）：

官版 `llama-kvmem-server` 的參數集原樣搬到 `llama-server`（差別只在 **ctx 與生成上限分開設**：`--ctx-size` 管上下文、`-n` 管每請求預設生成上限；**`--kv-dtype` 拆成 `-ctk` / `-ctv`**）：

```sh
llamaServer="D:/Programs/llama-cpp-repos/laamaafung-v21/bin/Release/llama-server.exe"
model="D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf"
mmproj="D:/models/Mudler/mmproj-Qwen-AgentWorld-35B-A3B-BF16.gguf"
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
$llamaServer --model $model --host 0.0.0.0 --port 8008 \
-c 262144 -n 32768 -ub 128 -b 512 -ngl 99 --n-cpu-moe 33 --parallel 1 \
--cuda-register-host --sched-prefetch-experts 1 \
--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 8192 --kvmem-gpu-ratio 0.9 --kvmem-block-tokens 128 \
-ctk q8_0 -ctv turbo4 \
--reasoning on --reasoning-budget 2048 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我响应。" \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 \
--mmproj $mmproj --no-mmproj-offload --image-min-tokens 1024 \
--chat-template-file $template --alias Agentic-Turbo-Coder
```

> 上例的 `-c 262144` / `-n 32768` / `-ub 128` / `-b 512` / `--kvmem-*` 全部沿用官版 KVMem 伺服器的生產值，只把 `--kv-dtype q8_0` 拆成 `-ctk q8_0 -ctv turbo4`、`--enable-thinking` 換成 `--reasoning on`。
> **MoE 模型必須顯式指定 `--n-cpu-moe`**（如上例的 33）：實測省略它時，啟動即報 `failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort`——`--fit` 的自動適配需要足夠的自由度（專家層卸載）才能把 35B 塞進 8GB 顯存，鎖死 `-ngl` 又沒有 `-ncmoe` 就無解。補上後實測 8 秒完成載入、推理正常（Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP，RTX 3060 Ti）。
> `--kvmem-budget`（工作集）與 `--kvmem-gen-reserve`（單輪生成頭寸）是唯二需要按自己機器／單輪生成長度調整的；其餘 KVMem 參數預設即生產值。
> 帶圖像時**整張圖必須整體駐留**：`--kvmem-budget` 要 ≥ 單張圖的行數（由 `--image-min-tokens` 決定）＋ sink ＋ 查詢。放不下時**預設會先在 token 化之前把圖縮小**（見下文「圖像自動縮放」）；只有連縮放下限都放不下、或關閉了自動縮放，才會明確拒絕（HTTP 400，訊息 `image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget`，與官版 `llama-kvmem-server` 相同）。服務器**不會**因這種請求崩潰，後續請求照常服務（`--kvmem-budget 32768` 足以容納多張全解析度圖）。
> **多圖**：同一條訊息裡**相鄰且尺寸相同**的圖片，會被 Qwen-VL 按「視頻幀合併」語義兩兩拼成一張畫布（`clip_model_n_temporal_merge`），模型看到的是合併後的圖；要讓每張圖各自獨立，請在兩張圖之間插一個文本 part（哪怕只是一個換行，見下文「啟用條件」的多圖說明）。
> 注意 KVMem 下**不要**加 `--context-shift` / `--prompt-truncate` / `--cache-reuse`：前兩者會被拒絕或自動禁用，後者依賴 K-shift 亦會自動禁用（見下文「KVMem」一節）。
> 純 CPU 或小顯存試跑可把 `-ngl 99` 換成 `-ngl 0`，並把 `-c` / `--kvmem-budget` 按比例調小。

Prism 三值量化（Ternary-Bonsai，須 v22 分支，無 KVMem）：

`PTQ1_0`（三值，1.75 bpw）示例；要跑 `PQ2_0`（2.06 bpw）把 `model` 換成對應 gguf 即可，其餘參數不變。權重是 Hadamard 折疊後的（GGUF 帶 `prism.hadamard.*` 元數據），推理時激活自動做同款旋轉，**無需任何額外參數**；與下例的差別只在於不帶任何 `--kvmem*` 參數：

```sh
llamaServer="G:/Agents/kvmem-works/laamaafung/build-v22/bin/Release/llama-server.exe"
############################################################
model="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
# model="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
############################################################
mmproj="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-mmproj-BF16.gguf"
############################################################
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
############################################################
$llamaServer --model $model --host 0.0.0.0 --port 8008 \
-c 262144 -n 32768 -ub 128 -b 512 -ngl 99 --parallel 1 \
--cuda-register-host --sched-prefetch-experts 1 \
-ctk q8_0 -ctv turbo4 \
--reasoning on --reasoning-budget 2048 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我响应。" \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 \
--mmproj $mmproj --no-mmproj-offload --image-min-tokens 1024 \
--chat-template-file $template --alias Agentic-Turbo-Coder
```

> 注意：`-c 262144` 的全量量化 KV cache 在 8GB 級顯存（RTX 3060 Ti）上裝不下，會溢出到主存走 PCIe，實測 decode 約 8 t/s（輸出正確，只是慢）；顯存充裕、或把 `-c` 調小到裝得下，即不受此限。要在有限顯存下跑長上下文，請用下例的 KVMem 版本（同一模型、同一 `-c` 實測 decode 約 33 t/s）。

KVMem + Prism 三值量化（Ternary-Bonsai，須 v22 分支）：

`PTQ1_0`（三值，1.75 bpw）示例；要跑 `PQ2_0`（2.06 bpw）把 `model` 換成對應 gguf 即可，其餘參數不變。權重是 Hadamard 折疊後的（GGUF 帶 `prism.hadamard.*` 元數據），推理時激活自動做同款旋轉，**無需任何額外參數**：

```sh
llamaServer="G:/Agents/kvmem-works/laamaafung/build-v22/bin/Release/llama-server.exe"
############################################################
model="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
# model="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
############################################################
mmproj="C:/WorkModels/Qwen3.8-27B/Ternary-Bonsai-2-27B-mmproj-BF16.gguf"
############################################################
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
############################################################
$llamaServer --model $model --host 0.0.0.0 --port 8008 \
-c 262144 -n 32768 -ub 128 -b 512 -ngl 99 --parallel 1 \
--cuda-register-host --sched-prefetch-experts 1 \
--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 16384 --kvmem-gpu-ratio 0.9 --kvmem-block-tokens 128 \
-ctk q8_0 -ctv turbo4 \
--reasoning on --reasoning-budget 2048 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我响应。" \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 \
--mmproj $mmproj --no-mmproj-offload --image-min-tokens 1024 \
--chat-template-file $template --alias Agentic-Turbo-Coder
```

> `--kvmem-gen-reserve 16384` 按單輪生成長度自定，並無固定區間或上限（此處取 16384 是本例長回合生成的取值）；三值模型在 8GB 級顯存（RTX 3060 Ti）實測 decode 約 33 t/s（PTQ1_0，預熱後的正常速度）／10 t/s（PQ2_0）。

TCQ KV 量化（turbo3_tcq / turbo2_tcq，須 v23 分支）：

TCQ（Trellis-Coded Quantization）是 turbo 系列的格狀編碼版本：`turbo3_tcq` 3.25 bpv（9-bit 狀態 → 512 碼本）、`turbo2_tcq` 2.25 bpv（8-bit 狀態 → 256 碼本），位率與 `turbo3`／`turbo2` 基本持平，靠 trellis 成形增益與在旋轉激活上訓練的碼本換取**更低的量化誤差**；代價是 decode 慢約 1~5%。**K 與 V 必須同為 TCQ 系列**（`turbo3_tcq` 與 `turbo2_tcq` 可交叉搭配，但不得與 `q8_0`／`turbo4` 等非 TCQ 類型混搭，否則沒有可用的注意力內核），且 head_dim 須為 128 或 256。TCQ 走專用的 VEC 注意力內核（碼本常駐共享記憶體），不參與上文 TurboQuant 的 MMA 融合路徑。要跑 `turbo2_tcq` 把 `-ctk` / `-ctv` 換掉即可，其餘參數不變：

```sh
llamaServer="G:/Agents/kvmem-works/laamaafung/build-v23/bin/Release/llama-server.exe"
############################################################
model="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix"
############################################################
mmproj="G:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-GGUF-ORIGINAL/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
############################################################
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
############################################################
$llamaServer --model $model --host 0.0.0.0 --port 8008 \
-c 262144 -n 32768 -ub 128 -b 512 -ngl 99 --parallel 1 \
--cuda-register-host --sched-prefetch-experts 1 \
--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 8192 --kvmem-gpu-ratio 0.9 --kvmem-block-tokens 128 \
-ctk turbo3_tcq -ctv turbo3_tcq \
--reasoning on --reasoning-budget 2048 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我响应。" \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 \
--mmproj $mmproj --no-mmproj-offload --image-min-tokens 1024 \
--chat-template-file $template --alias Agentic-Turbo-Coder
```

> 不帶 `--kvmem*` 即為無 KVMem 版本，其餘參數不變。實測（Qwen3.5-9B IQ4_XS，RTX 3060 Ti，`-ngl 99 -c 8192`，`llama-kvmem-cli`）：`turbo3_tcq` 約 80 t/s、`turbo2_tcq` 約 83 t/s，對照 `turbo3` 84.6／`turbo4` 84.8（trellis 遍歷的常規開銷）；KVMem + `turbo3_tcq` 組合同樣通過。輸出正確性已驗證（連貫中文長文與 17×23=391 多方法計算）。

turbo1.5 KV 量化（`turbo1.5`，須 v23 分支）：

turbo1.5 是 turbo 系列的**三值**版本：每 32 值一塊、5 trit/byte 打包（`packed = Σ(trit_i+1)×3^i`），理論載荷 2.25 bpv、存儲口徑 4.0 bpv（16 字節/32 值，含 7 字節對齊填充）。與 `turbo2/3/4` 一樣採用 128-group 修正範數 + Walsh–Hadamard 旋轉（旋轉激活上做三值量化，K/V 寫入與 Q 變換都在 graph 層自動配對，點積數學等價），量化誤差低於同位率的標量三值。與 TCQ 不同，**turbo1.5 可與 `turbo2/3/4`／`q8_0`／`f16` 交叉混搭**（例如 `-ctk q8_0 -ctv turbo1.5`），也可 K/V 同型走專用 VEC 注意力內核（與 TCQ 一樣不參與 MMA 融合路徑）；head_dim 須為 128 的倍數（不足時緩存自動零填充到 128 對齊）。要跑 `turbo1.5` 把 `-ctk` / `-ctv` 換掉即可，其餘參數不變：

```sh
llamaServer="G:/Agents/kvmem-works/laamaafung/build-v23/bin/Release/llama-server.exe"
############################################################
model="C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-IQ4_XS-UP-4.888G-genesis-imatrix"
############################################################
mmproj="G:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-GGUF-ORIGINAL/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
############################################################
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
############################################################
$llamaServer --model $model --host 0.0.0.0 --port 8008 \
-c 262144 -n 32768 -ub 128 -b 512 -ngl 99 --parallel 1 \
--cuda-register-host --sched-prefetch-experts 1 \
--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 8192 --kvmem-gpu-ratio 0.9 --kvmem-block-tokens 128 \
-ctk turbo1.5 -ctv turbo1.5 \
--reasoning on --reasoning-budget 2048 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我响应。" \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 \
--mmproj $mmproj --no-mmproj-offload --image-min-tokens 1024 \
--chat-template-file $template --alias Agentic-Turbo-Coder
```

> 不帶 `--kvmem*` 即為無 KVMem 版本，其餘參數不變。CLI 亦接受 `turbo1_5` 別名（`llama-bench` 同）。實測（Qwen3.5-9B IQ4_XS，RTX 3060 Ti，`-ngl 99 -c 8192`，`llama-kvmem-cli`）：`turbo1.5` 約 84 t/s，對照 `turbo3` 84.6／`turbo4` 84.8（同條件，LUT 查表的 unpack 開銷與三值解碼大體相抵）；純 `llama-cli` 無 KVMem 實測 80.9 t/s。輸出正確性已驗證（連貫中文長文與 17×23=391 多方法計算；KVMem 組合與 `llama-server -ctk turbo1.5` 亦同樣通過）。

文本 + 視覺：

```sh
llamaServer="D:/Programs/llama-cpp-repos/laamaafung-v17/bin/Release/llama-server.exe"
############################################################
model="C:/WorkModels/Qwen3.5-9B/Ornith-1.5-9B-IQ4_XS.gguf"
mmpj="C:/WorkModels/Qwen3.5-9B/mmproj-Ornith-1.5-9B-BF16.gguf"
############################################################
template="D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja"
$llamaServer -m $model \
--chat-template-file $template \
--reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-top-k 64 --reasoning-presence-penalty 1.2 \
--n-cpu-moe 0 -ngl all --alias "Agentic-Turbo-Coder" --ctx-size $((128*1024))  --threads 10 -lv 4 \
--cuda-register-host \
--parallel 1 -fa on -dev cuda0 --no-warmup --kv-unified --ctx-checkpoints 32 --cache-prompt  \
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
--batch-size 4096 -ub 256 \
--host 192.168.124.197 --port 8008 --threads-http 2 \
--reasoning on \
--reasoning-preserve \
--reasoning-format deepseek \
--reasoning-budget $((8*1024)) \
--reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" \
--load-mode mlock-ram \
--cont-batching \
--repeat_penalty 1.0 \
--repeat-last-n 64 \
--presence_penalty 0.0 \
--frequency_penalty 0.0 \
--jinja \
--cache-type-k q8_0 --cache-type-v turbo4 \
--cache-reuse 256 \
--cache-ram 8192 \
--cache-idle-slots \
--checkpoint-min-step 8192 \
--mmproj $mmpj --no-mmproj-offload --image-min-tokens 1024
```
---

#### 啟動參數與工作原理說明

以下係關鍵參數組及其工作原理，方便用户根據實際需求進行選擇：

| 參數組 | 說明 | 適用場景 |
| --- | --- | --- |
| `--cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64` | 啟用提示緩存（KV 緩存重用）機制。當多個請求有相同或相似的 prompt 前綴時，系統會重用之前計算的 KV 狀態，避免重複計算。`--cache-ram 8192` 設定緩存大小為 8GB，`--checkpoint-min-step 512` 設定創建 checkpoint 的最小步長，`--ctx-checkpoints 64` 設定保留的 checkpoint 數量。 | 適合有大量重複前綴請求、長對話歷史或需要加速響應的場景。 |
| `--cache-reuse N` | 啟用中間 chunk 位移重用。與 `--cache-prompt` 的前綴匹配不同，此功能透過 K-shift 將已緩存的 KV chunk 旋轉到新位置實現重用（非傳統意義的緩存重用）。**依賴 `llama_memory_can_shift()`**，當模型不支援 K-shift 時會自動禁用並警告 `cache_reuse is not supported by this context`。不支援的模型包括：使用 M-RoPE/IM-RoPE 位置編碼的模型（`n_pos_per_embd > 1`，如 Qwen3.5/3.6 系列、Qwen3VL 等多模態模型）、以及未啟用 `--swa-full` 的 SWA 模型。Qwen3.5/3.6 系列同時採用混合注意力架構（門控 DeltaNet 線性注意力 + 門控注意力），IM-RoPE 與混合注意力任一已足以令 K-shift 失效。**啟用 `--kvmem` 時一律自動禁用**（同因：KVMem 的索引無法跟隨位置重映射）。 | 僅對支援 K-shift 的標準 RoPE 模型（如 Qwen2.5、Llama 等）有效。不支援的模型移除此參數即可消除警告，`--cache-prompt` 已提供前綴緩存重用。 |
| `--context-shift` | 啟用生成階段的運行時 K-shift（KV cache 動態位移）。要求 `llama_memory_can_shift()` 回傳 true，否則會在 context 用盡時優雅停止（`STOP_TYPE_LIMIT`）。K-shift 不可用時自動禁用並警告，初始 prompt 截斷不受影響。隱含啟用 `--prompt-truncate`。**啟用 `--kvmem` 時一律拒絕**（KVMem 的分層索引按舊位置編號建立，無法跟隨重映射），需要更長上下文請加大 `--ctx-size`；此時「隱含啟用截斷」亦一併失效。 | 適合需要生成階段動態遷移 KV cache 的長程代理任務。使用 `--kvmem` 時請直接移除此參數。 |
| `--prompt-truncate` | 啟用初始 prompt 截斷（當請求 tokens 超過 `--ctx-size` 時自動截斷中間部分並保留頭尾）。對所有模型架構均生效，無需 KV cache 位移支援；但**啟用 `--kvmem` 時會被自動禁用並警告**（截斷會挖掉 prompt 中段並重寫 token 數組，令檢索查詢區間與 `--ctx-checkpoints` 的訊息分界全部錯位），超長提示改為返回明確的 400 `exceed_context_size`。由 `--context-shift` 隱含啟用，亦可單獨使用。 | 適合處理超長 prompt 提交、對話歷史較長的場景，避免 HTTP 400 錯誤。使用 `--kvmem` 時請改為加大 `--ctx-size` 或在客户端裁剪提示。 |
| `--kvmem` 及其子參數 | 啟用 KVMem 分層/稀疏 KV 記憶，讓 `llama-server` 本體在有限顯存下驅動超長上下文（須 v21 分支的 `LLAMA_KVMEM` 構建）。**強制 `--parallel 1`**；純線性注意力（recurrent）與 SWA 模型不支援，不滿足時靜默回退標準 KV。子參數：`--kvmem-budget`、`--kvmem-gpu-ratio`、`--kvmem-block-tokens`、`--kvmem-gen-reserve`、`--kvmem-query-last`、`--kvmem-query-max`、`--kvmem-method`、`--kvmem-harvest-v`、`--no-kvmem-image-autoscale`。 | 顯存不足以容納目標上下文的 KV cache 時（例如 8GB 顯存跑 128K 上下文）。詳見下文「KVMem（分層/稀疏 KV 記憶）」一節。 |
| `--swa-full` | 使用與 base cache 等大的全尺寸 SWA cache。僅對 GGUF 模型頭中明確聲明滑動窗口注意力（SWA）且窗口大小固定的模型有效（如 Gemma2/3、Cohere2、Exaone 等）。預設關閉時 SWA cache 僅為 `min(size_base, n_swa + n_ubatch)`，會導致 `llama_kv_cache_iswa::get_can_shift()` 回傳 false，使 `--context-shift` 的運行時 K-shift 失效（初始截斷不受影響）。啟用後 SWA 與 base 等大，K-shift 完全可用。 | 真正採用 SWA 架構的模型需要 `--context-shift` 完整功能（含生成階段運行時 K-shift）時必須配合使用。 |
| `--threads N` / `--threads-batch N` | 設置生成和 batch/prompt 處理的線程數。當 N <= 0（如 -1 或 0）時，系統會使用 `common_cpu_get_num_math()`（即物理數學核心數），而非 `hardware_concurrency()`（所有邏輯核心），以避免在 SMT（超線程）或混合架構 CPU 上過度訂閲導致的性能下降。 | 適合在具有 SMT（超線程）或混合架構（如 Apple M1）的 CPU 上優化 token 生成吞吐量。 |
| `LLAMA_THREADS_RATIO` (環境變數) | 當 `--threads` 為 auto 模式（N <= 0）時，按此比例縮放線程數（範圍 0.1 - 1.0，默認 1.0 即不縮放）。**適用於 GPU + CPU 混合推理場景**（如 MoE 專家層透過 `-ncmoe` 卸載到 CPU），留出部分 CPU 核心給 CUDA driver/sync 工作，可顯著提升 decode 吞吐量。見下方案例。 | GPU + CPU 混合推理（`-ncmoe > 0` 且 `-ngl > 0`）場景。純 CPU 推理或純 GPU 推理無需設置。 |

#### KVMem（分層/稀疏 KV 記憶）

KVMem 把 KV 緩存切成固定大小的**塊**（預設 128 tokens/塊），只把當前工作集常駐顯存，其餘分層存放主機內存並按需換回；檢索模式下以「本回合最後一條 user 訊息」的 token 區間作查詢，從歷史塊中挑回相關內容。目的：在**有限顯存**下驅動遠超顯存容量的上下文（例如 8GB 顯存 × 128K 上下文）。

與獨立進程 `llama-kvmem-server` 不同，本分支把 KVMem 接進了 **`llama-server` 本體**，因此 v21 的其餘特性（`--reasoning-*`、`--spec-*`、`--ctx-checkpoints`、`--cache-*`、turbo 系列 **KV** 量化）全部保留。可直接套用的啟動命令見上文「啟動示例」的 KVMem 一節。

**由官版 `llama-kvmem-server` 遷移（參數對照）**

| 官版 `llama-kvmem-server` | `llama-server`（本分支） | 說明 |
| --- | --- | --- |
| `-c 262144` | `--ctx-size 262144`（或 `-c`） | 上下文大小。 |
| `-n 32768` | `-n 32768`（`--n-predict`） | **語義不同、且在 llama-server 是分開設的**：官版 `-n` 是伺服器端的生成上限，llama-server 的 `-n` 是「每請求預設生成上限」，上下文由 `--ctx-size` 單獨管。 |
| `-ub 128 -b 512` | `--ubatch-size 128 --batch-size 512`（或 `-ub` / `-b`） | 同名同義。 |
| `-ngl 99` | `-ngl 99` | 同名同義。 |
| `--kvmem-budget` / `--kvmem-gen-reserve` / `--kvmem-gpu-ratio` / `--kvmem-block-tokens` | 同名 | KVMem 參數一一對應（預設值見下表）。 |
| `--no-kvmem-image-autoscale` | 同名 | **兩邊都已支援**：行數超預算的圖像會在 token 化**之前**自動縮小（見「啟用條件」）。關掉後兩邊都回同一條訊息 `image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget`（HTTP 400）。 |
| `--kv-dtype q8_0` | `-ctk q8_0 -ctv turbo4` | llama-server **沒有** `--kv-dtype`：K 與 V 分開設，但 `-ctk` 與 `-ctv` **接受完全相同的取值清單** —— `f32 / f16 / bf16 / q8_0 / q4_0 / q4_1 / iq4_nl / q5_0 / q5_1 / turbo2 / turbo3 / turbo4 / turbo3_tcq / turbo2_tcq / turbo1.5`（同一份 `get_all_kv_cache_types()`、同一個 `kv_cache_type_from_str()` 解析器；TCQ 兩型與 `turbo1.5` 須 v23 分支）。**turbo2/3/4 對 K 與 V 都可用，沒有「只能用於 V」的限制**（TCQ 兩型同樣 K/V 皆可，但 **K 與 V 必須同為 TCQ 系列**，不得與非 TCQ 類型混搭，見「TCQ KV 量化」；`turbo1.5` 同樣 K/V 皆可，且**可與非 TCQ 類型交叉混搭**，見「turbo1.5 KV 量化」）；上例只是 K 取精度、V 取壓縮的常見搭配。KVMem 直接從 `llama_memory_params.type_k/type_v` 取類型，兩者各自生效。唯一要注意的是別處的加速條件：CUDA 的 fused turbo MMA 路徑要求 **K 與 V 同型**，所以 `-ctk turbo4 -ctv turbo3` 這類混搭不會走上融合路徑 —— 但**仍然在 GPU 上執行**，只是回退到一般注意力派發（依架構走 `MMA_F16` 或 `VEC`，詳見下文「TurboQuant」一節），並非沒有加速。 |
| `--enable-thinking`、`--reasoning-effort`、`--reasoning-budget` | `--reasoning on`、`--reasoning-budget`（＋ `--reasoning-format/-preserve/-temp/-top-p/...`） | llama-server 用 `--reasoning [on\|off\|auto]` 開關；「思考強度」由 `--reasoning-budget` 與一整套 `--reasoning-*` 採樣覆蓋表達，沒有 `effort` 這個名字。 |
| `--mmproj` / `--no-mmproj-offload` / `--image-min-tokens` | 同名 | 同名同義；mmproj × KVMem 已支持（見下方「啟用條件」的 `--kvmem-budget` 要求）。 |
| `--chat-template-file`、`--temp/--top-p/--top-k/--min-p/--*-penalty` | 同名 | 同名同義。 |

**啟用條件**（不滿足時靜默回退標準 KV，並在日誌給出原因）

- 需要 `LLAMA_KVMEM` 構建（v21 分支的構建已默認開啟）；
- `--parallel` 會**強制為 1**（KVMem 要求 `n_seq_max == 1`）：顯式非 1 會打警告 `KVMem requires n_parallel = 1, but N was requested - forcing n_parallel = 1`，`auto` 則打提示；
- 純線性注意力（recurrent）架構與 SWA 模型不支援（日誌分別為 `KVMem skips purely recurrent arch ...` 與 `KVMem skips SWA models`）；混合注意力模型（如 Qwen3.5/3.6 的門控 DeltaNet + 門控注意力）可用；
- 多模態（mmproj）：**已支持**（官版參數集帶 `--mmproj`，示例照搬）。圖像塊佔用連續的 cache 行、但攜帶 2-D（M-RoPE）模型位置，因此 `llama-server` 會為嵌入批設置 `llama_batch::logical_pos`（相鄰行號）並把媒體行的範圍告知 KVMem（`set_media_ranges`），KVMem 才能按行尋址且**整張圖整體保留**。
  - **預算**：`--kvmem-budget` 必須 ≥ 單張圖像的行數（由 `--image-min-tokens` 決定）＋ sink ＋ 查詢。圖像組是強制的、不能拆塊，所以這一項不滿足時**預設會先自動縮小圖像**（見下），只有連縮放下限都放不下（或關閉了自動縮放）才會拒絕該請求。拒絕時回 **HTTP 400**（`invalid_request_error`）而非 500 —— 這是請求本身放不下，不是服務器故障；訊息 `image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget` 直接點出兩個解法，與官版 `llama-kvmem-server` 逐字相同。判定在**前填之前**做（`llama_kvmem_check_fit()`），所以失敗是瞬時的，不會先跑完一次前填才報錯。高細節大圖的行數會隨原生解像度上升：實測 224×224 純色圖 ≈ 49 行、1024×1024 ≈ 1024 行、2048×2048 ≈ 4096 行（`KVMEM_TRACE=1` 的 `KVMem media rows: N chunk(s) ... [start,end)` 可核對）。
  - **圖像自動縮放（預設開）**：行數超預算時，在 **token 化之前**按比例把圖縮小，而不是讓請求失敗。KVMem 先算出「一張圖可用的 token 數」＝ `(budget − (sink + 4 個塊)) ÷ 本次請求的圖片數`，交給 vision 層把該模型的 image-token 上限壓低；Qwen-VL 的動態解析度預處理器隨即在 bitmap 階段把圖縮到預算內，所以縮放發生在**像素層**、比「先 token 化再拒絕」早一步。三個性質值得記住：**只降上限、從不放大**（小圖保持原解析度）；**以模型的原始上限為基準逐請求重算**，不會跨請求累積；**下限為 2 個塊**（預設 256 tokens）。
  - 這與 **DeepSeek Harness 的圖像歸一化標準一致**：DSH 放行大圖、再按比例縮到 `normalizedImageMaxPixels = 2048×2048`；而本模型預設的 image-token 上限（4096 tokens × `patch_area` 1024 px）= 4,194,304 px，**恰好也是 2048×2048**。也就是説**預算充裕時（≳5K 行）行為與 DSH 完全一致、不做任何額外降質**，只有預算不足時才進一步縮小。日誌會打 `KVMem budget caps one image at N tokens (model limit M) for K media file(s)`；想完全關掉用 `--no-kvmem-image-autoscale`（回復為直接拒絕）。
  - **`llama-kvmem-server` 同樣支援**（同名開關）：那裡的「拒絕」原本是它自己的守衞，報 `image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget`（HTTP 400）；接上自動縮放後會先縮圖、請求正常完成。**注意它是獨立編譯的目標，需要 `tools/CMakeLists.txt` 給它 `LLAMA_KVMEM=1`**，否則 `server-common.cpp` 裡的 `#if defined(LLAMA_KVMEM)` 鈎子會被靜默編掉（本分支已補）。
  - **多圖**：同一條訊息裡**相鄰（中間無文本）且尺寸相同**的圖片會被 Qwen-VL 視為**視頻幀**兩兩合併成一張畫布（`mtmd.cpp` 的 `n_merge_frames = clip_model_n_temporal_merge()`，每組上限 2 張、逐對 `(1,2)(3,4)…`）。合併後模型看到的是拼合圖（例如兩張純色圖會被描述成「左右兩半」），且整組只佔一張畫布的 token 數。**要讓每張圖各自獨立，請在兩張圖之間插一個文本 part**（例如 `\n`）；尺寸不同則不會合併。日誌的 `KVMem media rows: N chunk(s)` 是判斷實際分塊數的可靠依據。
  - **錯誤恢復**：放不下的請求隻影響該次請求（HTTP 400），不會拖垮服務器；即便異常是在前填深處才浮現，也會被歸類成同一條 400 訊息（`llama_kvmem_classify_fit_error()`），不會退回 500。

**與其他功能的交互**

| 功能 | KVMem 下的行為 |
| --- | --- |
| `--context-shift` | **一律拒絕**並對該請求返回錯誤：KVMem 的分層索引（塊原始位置 + 逐行元數據）按舊位置編號建立，無法跟隨位置重映射。context 用盡時以 `STOP_TYPE_LIMIT` 優雅停止，要更長上下文請加大 `--ctx-size`。 |
| `--prompt-truncate` | **自動禁用並警告**：截斷會挖掉 prompt 中段並重寫 token 數組，令檢索查詢區間與 `--ctx-checkpoints` 的訊息分界全部錯位。超長提示改為返回明確的 400 `exceed_context_size`。 |
| `--cache-reuse` | 自動禁用並警告（依賴 K-shift）。 |
| `-ctk` / `-ctv`（含 `turbo2/3/4`） | 直接生效（KVMem 從 `llama_memory_params.type_k/type_v` 取 KV 類型）。 |
| `--cache-prompt` / `--cache-ram` / `--ctx-checkpoints` | 可用：走 KVMem 已實現的 `seq_rm` 與 `state_write/state_read` 路徑（實測多輪對話與 checkpoint 回滾恢復正常）。`--cache-idle-slots` 等同類開關共用這條路徑，未單獨驗證。 |
| `--spec-*`（含 `draft-mtp`） | 可用；投機批期間的 decode mean 不落盤（上游限制），僅影響投機期間生成行的檢索打分。 |

**參數表**

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--kvmem` | 關閉 | 總開關（環境變數 `LLAMA_ARG_KVMEM`） |
| `--kvmem-budget N` | `0` = `--ctx-size` | 顯存工作集 token 數，**最關鍵的容量參數**，見下方配置建議 |
| `--kvmem-gpu-ratio R` | `0.50` | KVMem 顯存池上限佔 **VRAM 總量**的比例：`cap_blocks = (VRAM × R) / 塊大小`。與官版預設一致（`0.50`），取偏保守的值，以免個別用户設錯參數時把顯存推爆。它是**上限**而非目標值：若 `budget + gen_reserve` 未超出 `cap_blocks` 就完全不起作用，實際工作集仍由 `--kvmem-budget` 決定；配方確實需要更大池時才調大（官版生產配方用 `0.85`）。核對方法：`KVMEM_TRACE=1` 的 `KVMEM_KV_BYTES` 行看 `cap_blocks` 與實際 `budget`/`pool`。 |
| `--kvmem-block-tokens N` | `128` | 塊大小。越大檢索粒度越粗、元數據越省 |
| `--kvmem-gen-reserve N` | `8192` | 生成階段保留在工作集內的 slack token 數（decode 頭寸）。每輪生成先寫進這段頭寸，用完才會觸發重選／換出，所以設太小會在**生成途中把剛檢索回來的內容擠掉**，長回合直接失去召回。**`256` 之類只是測試刻度**（連一段程式碼都不夠寫），官版生產值為 **8192**，上下限由用户按「單輪可能生成多長」自定。顯存成本 ≈ 該 token 數 × `block_bytes / block_tokens`。 |
| `--kvmem-query-last N` | `64` | prompt 無 chat 訊息分界（raw `/completion`）時的查詢長度 = 最後 N tokens；有分界時僅作兜底 |
| `--kvmem-query-max N` | `512` | 檢索查詢長度上限，超出時保留尾部（`0` = 不限制） |
| `--kvmem-method M` | `retrieval` | `retrieval`（按查詢檢索歷史塊）或 `recency`（只保留最近內容，等於純壓縮） |
| `--kvmem-harvest-v` | 關閉 | 前填時同時把 V 搬到主機（增加前填開銷，換取更完整的檢索載入） |
| `--no-kvmem-image-autoscale` | 自動縮放**開啟** | 關閉圖像自動縮放（環境變數 `LLAMA_ARG_NO_KVMEM_IMAGE_AUTOSCALE`）。預設開啟時，超出行數預算的圖像會在 **token 化之前**按比例縮小，而不是讓請求失敗；關閉後回到直接拒絕（HTTP 400，訊息見下文）。見「啟用條件」的多模態說明。 |

**配置建議**：`--kvmem-budget` 需 ≥「單次前填最大提示所佔的塊數 + `--kvmem-gen-reserve`」。若預算小於提示塊數，日誌會出現 `prepare_working_set: incoming block N was not placed on GPU`，KVMem 會回滾該次 append 並由上游拆小批次重試（可恢復、輸出正確，但前填反覆重試會變慢）——這屬**預算配置問題，不是缺陷**。`--kvmem-gen-reserve` 單獨看是「**單輪生成頭寸**」：要 ≥ 你預期最長的一輪生成（寫碼／長推理動輒數千 token），官版生產值取 8192、上下限由用户自定；只給 256 之類的測試刻度，KVMem 的召回會在生成途中被自己的換出邏輯吃掉。`--kvmem-gpu-ratio` 是**上限**而非目標值：它按 VRAM 總量換算可容納的塊數，若 `budget + gen_reserve` 超出就會把池壓小（見上表）。**預設 `0.50`（與官版一致）**，取值保守以免設錯參數時把顯存推爆；只要 `budget + gen_reserve` 未觸及 `cap_blocks`，它就不起作用，實際工作集完全由 `--kvmem-budget` 決定。生產組合 `-c 131072 --kvmem-budget 32768 --kvmem-gen-reserve 8192` 已在 8GB 顯存 + `-n-cpu-moe 36` 上實測正常（多輪對話、含 checkpoint 回滾，0 報錯）；該組合的 `budget + gen_reserve = 40960` tokens，在 8GB + `-ctk turbo4`（量化 KV）下仍低於 `cap_blocks`，故未顯式指定 ratio。**注意**：若改用 F16 KV，8GB 的 `cap_blocks` 僅約 43K tokens，與 40960 已相當接近——此時 `--kvmem-gpu-ratio 0.50` 會開始成為實際約束，需要更大工作集時請一併調大該值。

**日誌排查（`KVMEM_TRACE=1`）**：`KVMEM_KV_BYTES cells=.. budget=.. pool=..`（是否啟用與池大小）、`KVMem query span = [a, b) of N prompt tokens (last user turn|fallback: last tokens)`（本回合查詢區間及來源）、`KVMEM_CAPTURE tag=q/k`（捕獲是否生效）、`KVMEM_TRACE harvest n=.. q=.. k=..`、`KVMEM_TRACE retrieval stage_in=.. skip=.. window=..`、`KVMEM_DECODE_MEAN flush block=.. n=..`。若這些行全部缺失或計數為 0，代表 KVMem 並未真正參與，請先檢查是否被 `--parallel`、SWA 或 recurrent 條件擋下。

---

#### GPU + CPU 混合推理線程調優案例

當 MoE 專家層透過 `-ncmoe` 卸載到 CPU 時，CPU 與 GPU 之間每層都有同步開銷。若 CPU 使用全部物理核心做 MoE 計算，會與 CUDA driver 的 sync 調度競爭，反而降低 decode 吞吐量。設置 `LLAMA_THREADS_RATIO` 留出部分核心可顯著提升性能。

以下為 **Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP**（Qwen35MOE 架構）在 **Xeon E5-2696 v3（18 核/18 線程）+ RTX 3060 Ti 8GB** 上的實測數據（`-ngl 99 -ncmoe 33 -ctk turbo4 -ctv turbo3 -ub 1024`）：

| threads | ratio | tg128 (tok/s) | 變化 |
| :---: | :---: | :---: | :---: |
| 4  | 0.22 | 26.80 | -5% |
| 6  | 0.33 | 31.60 | +11% |
| 8  | 0.44 | 31.83 | +12% |
| **10** | **0.56** | **33.47** | **+18%** |
| 12 | 0.67 | 33.29 | +17% |
| 18 (預設) | 1.00 | 28.34 | baseline |
| 24 | -    | 29.26 | +3% |
| 32 | -    | 28.23 | -0.4% |

**最佳配置**：`LLAMA_THREADS_RATIO=0.56`（即 `-t 10`），比預設全核快 **+18%**。

> **注意：** 最優比例取決於 CPU 架構、GPU 算力、MoE 卸載比例、模型大小等多個因素，上表數據僅供參考。建議用户通過 `llama-bench` 實測自身硬件的最優值。設置方式：
>
> ```sh
> # Linux/macOS
> export LLAMA_THREADS_RATIO=0.56
>
> # Windows PowerShell
> $env:LLAMA_THREADS_RATIO=0.56
>
> # Windows CMD
> set LLAMA_THREADS_RATIO=0.56
> ```

| `--load-mode MODE` | 模型載入模式（默認 `mmap`）。取代舊參數 `--mmap`/`--no-mmap`/`--mlock`/`--direct-io`，五者互斥，僅能選一個 mode。 | 控制模型載入時的記憶體映射與駐留策略，見下表。 |

#### `--load-mode` 可選值一覽

| `--load-mode` 值 | 等效舊參數 | 含義 |
| --- | --- | --- |
| `none` | `--no-mmap` | 不使用 mmap（慢載入，但可減少 page-out） |
| `mmap` | `--mmap` | memory-map 模型（默認值） |
| `mlock` | `--mmap --mlock` | mmap + 強制系統將模型駐留 RAM，禁止 swap/壓縮 |
| `mlock-ram` | `--no-mmap --mlock` | 直接讀取模型到 RAM + mlock（不用 mmap）；避免推理時 mmap page-fault 導致的性能下降 |
| `dio` | `--direct-io` | 使用 DirectIO 載入（若可用） |

> **注意：** 切勿將 `--load-mode` 與舊參數 `--mlock`/`--mmap`/`--no-mmap`/`--direct-io`/`--no-direct-io` 混用，否則會觸發警告，且僅命令行最後一個 flag 生效。舊參數僅為向後兼容保留，後續版本可能移除。

> **舊組合 `--no-mmap --mlock` 遷移說明：** 舊版 `use_mmap` 與 `use_mlock` 是兩個獨立布爾字段，允許「不用 mmap + 鎖定記憶體」的組合（eager read 載入 CPU buffer 後再 mlock）。上游新版 `--load-mode` 合併為單枚舉，原本不再支持此組合。現 Laamaafung 已新增 `--load-mode mlock-ram` 恢復此行為：直接讀取模型到 RAM 後 mlock，不經過 mmap，避免推理時 mmap page-fault 導致的性能下降。建議根據自身設備的實際參數性能表現選用 `mlock-ram` 或者 `mlock`。

---

### 自動 Batch Size 調優（Auto Batch Size Tuning）

引入了對 `--batch-size` 和 `--ubatch-size` 參數的自動調優支持。此功能為本分支獨有，上游官方分支尚未支援。透過自動調優，程序可在啟動時根據 `n_ctx`（上下文大小）與硬件特徵（如 NUMA 架構狀態）自動計算並選擇最佳的邏輯 batch size (`n_batch`) 與物理 batch size (`n_ubatch`)，以充分發揮硬件並行計算能力並避免內存/Cache 瓶頸。

| 參數 | 說明 |
| --- | --- |
| `--batch-size auto` 或 `--batch-size -1` | 啟用邏輯 batch size (`n_batch`) 自動調優。程序會根據 `n_ctx` 和硬件特徵自動計算最佳值，最大上限為 8192。若系統為 NUMA 架構，則上限降低至 4096。確保最小值 `>= 32`（BLAS 要求）。 |
| `--ubatch-size auto` 或 `--ubatch-size -1` | 啟用物理 batch size (`n_ubatch`) 自動調優。程序會根據 `n_ctx` 和硬件特徵自動計算最佳值，最大上限為 4096。若系統為 NUMA 架構，則上限降低至 2048。確保最小值 `>= 64`（以觸發 Tiled Flash Attention 優化，對應 `Q_TILE_SZ` 閾值）。 |
| `--cuda-register-host` | **兩邊都已支援**。固定（pin）GPU 後端的主機緩衝（`cudaHostRegister`），讓傳輸可省下一次中轉拷貝；等同環境變量 `GGML_CUDA_REGISTER_HOST=1`（MUSA/HIP 後端亦適用），在無法設定環境變數的環境（受限 shell、服務單元）用它代替。註冊屬 best-effort，失敗會靜默略過。 |
| `--sched-prefetch-experts N` | **兩邊都已支援**。MoE 專家權重預取：等同環境變量 `GGML_SCHED_PREFETCH_EXPERTS=N`。`N=1` 用默認 3 個槽位（一層 MoE 的 gate/up/down），更大的 `N` 直接指定槽數，`N=0` 關閉。槽位越多，上傳越能跑在計算之前，代價是每槽一份「最大專家張量」的裝置記憶體。非數字會被拒絕（ggml 用 `atoi()` 讀取，否則會被靜默當成 0）。 |

**調優邏輯說明：**
- **基於 Context Length 的動態縮放**：自動計算時，會根據 `n_ctx` 進行縮放，避免過大的 batch 導致 KV cache 溢出或 intermediate tensors 過大。
- **NUMA 架構適應**：若檢測到系統為 NUMA 架構（多 CPU 插槽），則降低 `n_batch` 與 `n_ubatch` 的上限，以避免跨 NUMA 節點的內存訪問延遲增加和 L3 cache miss 率飆升。
- **觸發 Tiled Flash Attention 優化**：確保 `n_ubatch >= 64`，以滿足 `neq1 >= Q_TILE_SZ` 的條件，從而觸發 `ggml_compute_forward_flash_attn_ext_tiled` 中的 SIMD/GEMM tile 並行優化，避免回退到效率較低的 `one_chunk` 路徑。

**使用示例：**

```sh
# 啟用邏輯與物理 batch size 自動調優
./llama-server --model models/llama3.gguf --batch-size auto --ubatch-size auto ...

# 或使用 -1 值觸發自動調優
./llama-server --model models/llama3.gguf --batch-size -1 --ubatch-size -1 ...
```

當設置為 `auto` 或 `-1` 時，程序會在啟動時根據上述規則自動計算並輸出選擇的 `n_batch` 和 `n_ubatch` 值，例如：
```
llama_context::from_params: n_batch set to auto, selected value: 4096 based on n_ctx=131072 and hardware
llama_context::from_params: n_ubatch set to auto, selected value: 4096 based on n_ctx=131072 and hardware
```

---

#### TurboQuant 鍵值緩存 與 MMA 融合路徑

透過 `--cache-type-k` / `--cache-type-v` 指定 TurboQuant 量化類型（`turbo4` / `turbo3` / `turbo2`，v23 分支另支援 TCQ 兩型 `turbo3_tcq` / `turbo2_tcq` 與三值型 `turbo1.5`）可壓縮 KV 緩存佔用。在 CUDA 後端上，只要 GPU 架構為 Turing 及以上（Turing / Ampere / Ada Lovelace / Hopper / Blackwell 等，即 SM 7.5+），系統會自動啟用 MMA 融合注意力路徑（fused turbo MMA）以加速解碼；條件不滿足時（Volta 及更早、或 K/V 不同型等）會自動回退到一般注意力派發，仍在 GPU 上執行（詳見本節末尾）。

| 環境變數 | 預設值 | 描述 |
| --- | --- | --- |
| `GGML_TURBO_MMA_FUSED` | `1`（開啟） | 控制 CUDA fused turbo MMA 路徑。設為 `0` 可關閉融合，回退到一般注意力派發（依架構走 `MMA_F16` 或 `VEC`；功能完整，僅失去內聯反量化與 GQA 打包的額外收益）。 |

MMA 融合路徑生效條件：**K 與 V 同型**且為 `turbo4`/`turbo3`/`turbo2`（另需 Turing 及以上的 tensor core、`V->ne[0] == Q->ne[0]`，且 `Q->ne[0]` 為 128 或 256；`turbo2` + head_dim 256 刻意不融合，見 `ggml/src/ggml-cuda/fattn.cu` 的註解）。**TCQ 兩型（`turbo3_tcq` / `turbo2_tcq`）與三值型 `turbo1.5` 不參與 MMA 融合**：它們始終走專用的 VEC 注意力內核（TCQ 是碼本常駐共享記憶體的逐元素點積路徑、`turbo1.5` 是 q8_1 Q 路徑的逐元素點積路徑），要求 head_dim 為 128 或 256。TCQ 要求 K 與 V 同為 TCQ 系列（可交叉搭配）；`turbo1.5` 則可與 `turbo2/3/4`／`q8_0`／`f16` 自由混搭（11 種 K/V 組合都有實例化內核），含 `turbo1.5` 的混搭同樣走 VEC 路徑。

條件不滿足時**不是失去 GPU 加速**，而是回退到一般注意力派發（仍在 CUDA 上跑、KV 壓縮效果照舊），實際走哪一條視架構與 batch 而定：
- **Volta 及更早**：無 tensor core → TILE / VEC。
- **Turing / Ampere**：量化 KV → `MMA_F16`，**仍然是 tensor core**，但會先把 K/V 轉成 f16（多一份臨時顯存與頻寬開銷）。
- **Ada Lovelace 及以上**：解碼（`Q->ne[1] <= 2`）→ `VEC`（核心內反量化、無臨時緩衝）；其餘 → `MMA_F16`。

所以 `-ctk turbo4 -ctv turbo3` 這類 **K/V 不同型**的混搭只是拿不到「融合」那一檔優化（內聯反量化 + GQA 打包、省掉 f16 臨時緩衝），並非退回 CPU 或沒有加速；要吃滿融合路徑就把 K 與 V 設成同一個 turbo 類型。全過程自動，無需手動幹預。

---

#### MoE 模型性能優化（計算與數據上傳重疊及 CPU 權重內存固定）

針對 MoE（混合專家）模型，我們實現了以下性能優化：

1. **計算與數據上傳重疊**：
   在較大 batch size 下，幾乎所有專家都會被使用，因此無需等待路由 ID 的讀取回傳。系統會通過第二個 backend 實例（在同一設備上，使用自己的 stream）上傳完整的專家 tensors，並使用兩個 event-ordered staging slots，使得 N+1 張量的上傳與 N 張量的計算重疊。
   
   可通過環境變量 `GGML_SCHED_PREFETCH_EXPERTS=1` 啟用此優化；若不便（或無權）設定環境變量，改用啟動參數 `--sched-prefetch-experts N`（`N=1` 等同默認 3 個槽位，更大的 `N` 直接指定槽數，`N=0` 關閉）。

2. **CPU 權重內存固定優化**：
   在模型加載完成後，對保留在系統內存中的權重內存頁進行固定（pin mmap-backed CPU weights），以實現更快的主機到設備（H2D）傳輸。這對於 MoE 專家權重在 prefill 階段動態加載到 GPU 時特別有效。
   
   可通過環境變量 `GGML_CUDA_REGISTER_HOST=1`（針對 CUDA 後端）啟用此優化；若不便（或無權）設定環境變量，改用啟動參數 `--cuda-register-host`（MUSA/HIP 後端同樣適用）。

上述兩個啟動參數與對應的環境變量完全等價（參數只是在解析時把變量寫進行程環境，ggml 仍按原方式讀取），二選一即可；環境變量已設定時，參數優先。
注意兩點：`--sched-prefetch-experts 1` 是「用默認 3 個槽位」，等同 `GGML_SCHED_PREFETCH_EXPERTS=1`，**不是** 1 個槽位（要 1 個槽位請直接寫 `--sched-prefetch-experts 1` 之外的明確值）。另外若仍選擇用環境變量，**必須 `export`**（或寫成 `VAR=1 命令 …` 的前置形式）——單獨一行 `GGML_CUDA_REGISTER_HOST=1` 只會設成 shell 變數，子行程收不到。`llama-server` 與 `llama-kvmem-server` **都支援**這兩個參數；`llama-kvmem-server` 啟動時會打一行 `KVMEM_STARTUP ggml_env cuda_register_host=… prefetch_experts=…`（兩者皆為 off 時不打），可用來核對實際生效值。這些優化可顯著提升 MoE 模型的性能，例如在 Qwen3.6-35B-A3B 模型上，預取優化可將吞吐量從 1383 提升到 1663 t/s（在 RTX 3060 上，-ncmoe 26, ub 2048），而 CPU 權重內存固定優化可將吞吐量從 1144 提升到 1385 t/s。

---

#### 啟用上下文容量管理的啟動示例

如果須要處理可能超過上下文限制的請求，可以加入 `--prompt-truncate`（初始截斷）或 `--context-shift`（運行時 K-shift，隱含啟用初始截斷）。對於真正採用 SWA 架構的模型，若需要生成階段的運行時 K-shift 完整可用，須同時加入 `--swa-full`（**以上兩者在使用 `--kvmem` 時皆不適用**，見下文「KVMem」一節）：

```sh
./laamaafung/build/bin/Release/llama-server.exe \
--model /path/to/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf \
--ctx-size 131072 --flash-attn on \
--reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" \
--reasoning-format deepseek \
--fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified \
--cache-type-k q8_0 --cache-type-v q8_0 \
--host 0.0.0.0 --port 8008 \
-b 16384 -ub 256 --load-mode mlock --no-mmproj \
--cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --context-shift \
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 \
--jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose --verbosity 5 \
--chat-template-file /path/to/iStartModel/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

> **注意：** Qwen3.5/3.6 系列模型（MoE 與稠密變體）採用混合注意力機制（門控 DeltaNet 線性注意力 + 門控注意力），並非標準的滑動窗口注意力架構，GGUF 模型頭中 `n_swa = 0`。因此 `--swa-full` 對這些模型無效，載入時會自動檢測並禁用同時彈出警告 `swa_full is not supported by this model, it will be disabled`，此為正確行為，llama.cpp 已自動安全降級。`--context-shift` 會因 K-shift 不可用而自動禁用並警告，但 `--prompt-truncate` 不受影響，初始 prompt 截斷仍然生效。生成階段到達 context 上限時會優雅停止（`STOP_TYPE_LIMIT`）。**惟啟用 `--kvmem` 時例外：`--context-shift` 一律拒絕，`--prompt-truncate` 亦會被自動禁用並警告，超長提示返回 400 `exceed_context_size`——請改為加大 `--ctx-size`，詳見下文「KVMem」一節。**Qwen3.5/3.6 系列本身支援長上下文（如 256K/512K），無需依賴 SWA 即可高效處理長序列。若想消除日誌噪音，請直接移除 `--swa-full`。`--swa-full` 僅對 GGUF 文件頭中明確聲明滑動窗口注意力且窗口大小固定的模型有效（如 Gemma2/3、Cohere2、Exaone 等）。

---

#### 段級重複循環檢測參數說明

| 參數 | 類型 | 默認值 | 描述 |
| --- | --- | --- | --- |
| `--repeat-line-window` | 整數 | 0（已禁用） | 要跟蹤的歷史片段數量 |
| `--repeat-line-min-length` | 整數 | 20 | 最小片段長度（避免因短語而產生的誤報） |
| `--repeat-line-delimiters` | 字符串 | `"\n.!?:。！？："` | 結束一個片段的字符 |
| `--repeat-line-temp-boost` | 浮點數 | 0.5 | 檢測到迴路時温度升高 |


示例（啟用 repeat_line 採樣器以防止無限循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --repeat-line-window 10 --repeat-line-min-length 20 --repeat-line-delimiters "\n.!?:。！？：" --repeat-line-temp-boost 0.5
```

---

#### 連續 Token 重複失控檢測（內建，無須配置）

當模型陷入同一 token 反覆生成的死循環（例如 `</</</...`），系統會自動偵測並以分級升温打破循環，無需手動啟用任何參數。此機制與段級重複檢測（`--repeat-line-*`）互補：前者針對行/段級語義重複，本機制針對 token 級的硬性失控。

| 連續次數 | 動作 | 效果（以 base temp = 0.6 為例） |
| --- | --- | --- |
| 8 次 | `temp_boost = 2.0`，即 logit 乘以 1/(1+2) | 等效 temp = 1.8，温和升温，嘗試打破循環 |
| 16 次 | `temp_boost = 3.0`，即 logit 乘以 1/(1+3) | 等效 temp = 2.4，強力升温 |
| 64 次 | `STOP_TYPE_LIMIT` | 升温無效，強制停止作為最終安全網 |

升温原理與 `--repeat-line-temp-boost` 相同：對所有候選 token 的 logit 乘以 `1/(1+boost)`，等效於臨時提高採樣温度。一旦生成的 token 不再重複，boost 立即歸零，恢復正常採樣。

**與 `--repeat_penalty` / `--presence_penalty` 的區別：** 這兩個參數對已出現過的 token 施加持續性懲罰（降低其 logit），但對同一 token 連續出現的硬性失控無效。原因是：當模型對某 token（如 `</`）的 logit 遠高於所有其他候選 token 時，即使施加 1.5x 或 2.0x 的懲罰，此 token 仍然具有最高概率，模型會繼續選擇它，形成死循環。本機制不行"懲罰重複 token"的路線，而是通過升温（壓縮所有 logit 差距）令低概率 token 獲得被選中的機會，從根本上打破循環。

---

#### 週期性 Token 循環檢測機制（Periodic Token Loop Detection）

當模型生成呈現週期性或交替性的 token 死循環模式（例如「ababab...」、「abcabc...」等，其中 a、b、c 代表不同的獨立 token），內建的連續 token 失控檢測（針對連續相同 token，如 `aaaaaaaa...`）無法有效偵測此類模式。本機制提供專門的週期性重複檢測 sampler，用於檢查最近 token 序列中的循環/交替模式，並在檢測到時應用懲罰或升温以打破死循環。

| 參數 | 類型 | 預設值 | 描述 |
| --- | --- | --- | --- |
| `--cycle-detect-last-n N` | 整數 | 64 | 要檢查循環模式的最近 token 數量（0 = 停用） |
| `--cycle-detect-min-period N` | 整數 | 2 | 要檢測的最小週期長度 |
| `--cycle-detect-max-period N` | 整數 | 8 | 要檢測的最大週期長度 |
| `--cycle-detect-action TYPE` | 字符串 | `"boost"` | 檢測到循環模式時的動作：`"boost"`（温度升温，預設）或 `"penalty"`（重複懲罰） |
| `--cycle-boost-factor F` | 浮點數 | 0.5 | 檢測到循環模式時的升温因子（僅當 action = `"boost"` 時生效。等效於臨時將採樣温度提高，壓縮所有 logit 差距） |
| `--cycle-penalty-repeat F` | 浮點數 | 1.00 | 檢測到循環模式時的重複懲罰因子（僅當 action = `"penalty"` 時生效。1.0 = 停用懲罰） |

示例（啟用週期性重複採樣器並採用升温機制打破死循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action boost --cycle-boost-factor 0.5
```

示例（啟用週期性重複採樣器並採用懲罰機制打破死循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action penalty --cycle-penalty-repeat 1.5
```

**升温原理：** 當選擇 `"boost"` 動作時，對所有候選 token 的 logit 乘以 `1/(1+boost_factor)`，等效於臨時提高採樣温度。一旦生成的 token 不再呈現週期循環，檢測機制會重置，恢復正常採樣。

**與連續 Token 失控檢測的區別：** 內建的連續 token 失控檢測僅針對「連續相同 token」的硬性失控（如 `aaaaaaaa...` 或 `</</</...`）。而本機制專門針對 token-level 的交替/週期性循環模式（如 `ababab...`、`abcabc...`），透過週期檢測演算法（基於字串最小週期匹配屬性）識別並打破此類死循環。

---

#### 早停檢測與 EOG 抑制（Early-Stop Detection & EOG Suppression）

當模型在思考完成後未產生任何可見輸出就自動停止（例如思考結束但無正文回答，或工具調用被截斷），客户端會收到空響應且無明顯錯誤。流式模式下此問題尤其嚴重：用户可能完全不會察覺本回合已丟失。本機制通過雙層設計驅動模型持續生成直至出現真實內容，並對觸發此機制的 slot 開啟 5 回合高強度監控。

| 參數 | 類型 | 預設值 | 描述 |
| --- | --- | --- | --- |
| `--eog-retry-max N` | 整數 | 3 | 早停檢測觸發後的最大重試次數（0 = 禁用）。同時控制 slot 層 EOG 抑制次數和 HTTP 層非流式重試次數。 |

**雙層架構設計：**

1. **Slot 層 EOG 攔截（流式透明）** - 在 `process_token()` 中，當採樣到 EOG token 時檢查 `slot.generated_text` 判斷是否為缺陷性早停。若是，則不設置 `STOP_TYPE_EOS`，而是啟用 EOG 抑制（`common_sampler_set_suppress_eog`），將所有 EOG token 的 logit 強制設為 `-INFINITY`，令 sampler 返回次優 token 以繼續生成。由於抑制發生在 sampler 內部，流式客户端看到的是不中斷的 token 流，無需倒帶或重啟，且正常路徑與投機採樣路徑（共用 `common_sampler_sample()`）均被覆蓋。

2. **HTTP 層重試（非流式）** - 對非流式請求，`handle_completions_impl()` 將任務創建封裝為 `create_tasks` lambda 並運行重試循環：若最終結果的 `oaicompat_msg.content` 與 `tool_calls` 均為空，則重新提交任務。從第 2 次重試起 bump 採樣種子（`seed += http_retry`）以避免重複採樣同一死衚衕。循環受 `--eog-retry-max` 約束。

**早停檢測條件（任一命中即判定）：**

| 條件 | 場景說明 |
| --- | --- |
| `n_sent_text == 0` | 完全無任何輸出 |
| `<think>` 已開啟但無 `</think>` 且無 `<tool_call>` | 思考未閉合（`tagged_thinking_tools` 模板允許 `<tool_call>` 作為思考結束標籤，故有 `<tool_call>` 即視為思考已閉合） |
| `</think>` 存在其後僅空白 | 思考結束但無正文內容 |
| `<tool_call>` 已開啟但未閉合（`#<tool_call> > #</tool_call>`） | 工具調用被截斷（模型決定調用工具，輸出部分 JSON 後早停） |

**EOG 抑制清除條件：**

每個 token 採樣後（不僅 EOG），若 `slot.suppress_eog` 已設置則檢查是否已出現真實內容並清除標誌：
- 有 `<tool_call>`：要求所有 `<tool_call>` 標籤均已閉合
- 僅有 `<think>`：要求 `</think>` 後有非空白內容
- 純文本（無 think/tool 標籤）：`n_sent_text > 0`

**Per-slot 5 回合監控：**

觸發早停檢測時 `slot.monitoring_turns = 5`。此字段不被 `slot.reset()` 重置，而是在 `reset()` 中遞減，故跨同一對話的後續回合持久（`slot.id` 為天然 per-conversation 隔離鍵，免費支援並發）。若監控期間再次觸發早停，計數器重置為 5 並記錄警告。這為運維者提供了 slot 行為異常的可見信號。

**隱藏自檢 turn（監控增強）：**

單純的早停條件檢查無法區分"模型已正常完成簡短回覆"與"模型異常截斷"兩種場景。為此監控機制在觸發早停時不直接抑制 EOG，而是先注入一個對客户端透明的隱藏 turn，讓模型自判回覆是否完整，再根據判定結果決定是否抑制 EOG 並繼續生成。自檢使用內部協議標記 `<complete>` / `<incomplete>`，不依賴任何廠商特定標籤（如 `<think>`、`<tool_call>`），故通用於 Qwen、Gemma 等所有模型。

自檢狀態機（per-slot，`SELF_CHECK_*` 階段）：

```
NONE -> PREFILL -> GENERATING -> {NONE | ROLLBACK} -> NONE
```

1. **NONE -> PREFILL**：`process_token()` 中 `early_stop_no_output` 為真且 `cached_messages` 非空時觸發。`build_self_check_prompt()` 通過 `common_chat_format_single()` 構建增量 prompt（`past_msg = cached_messages + assistant_msg(generated_text)`，`new_msg = user "Review completeness, output <complete>/<incomplete>"`），tokenize 後入隊 `self_check_prefill`。
2. **PREFILL -> GENERATING**：`handle_last_sampled_token()` 將觸發 EOS 以 `output=false` 加入 batch，隨後追加 prefill tokens（末 token `output=true`）。保存回滾狀態（KV 位置、prompt 大小、EOS token id），sampler 同步接受每個 prefill token 以保持懲罰/repeat 狀態一致。
3. **GENERATING**：`process_token()` 將自檢回覆 token 路由至 `handle_self_check_token()`，累積文本至 `self_check_text`（清空 `text_to_send` 對客户端隱藏），監測 `<complete>`/`<incomplete>` 標記。安全閥：回覆超過 64 字符仍無標記時保守判定為 incomplete。
4. **完整判定**：`<complete>` -> `STOP_TYPE_EOS`，自檢 turn 留在 cache 中由下個請求覆蓋；`<incomplete>` -> 進入 `ROLLBACK`。
5. **ROLLBACK -> NONE**：`handle_last_sampled_token()` 通過 `common_context_seq_rm()` 從 KV cache 截斷整個自檢 turn，`keep_first()` 截斷 `prompt.tokens`，重新以 `output=true` 評估觸發 EOS，arm EOG 抑制後模型在原回覆上繼續生成。

自檢全過程所有 token 的 `text_to_send` 均被清空，流式與非流式客户端均無感知。`generated_text`、stop-word 偵測、partial response、EOG 早停邏輯在 GENERATING 階段全部旁路。

**重試預算：**
- `--eog-retry-max` 同時控制 slot 層抑制次數與 HTTP 層非流式重試次數
- `slot.eog_retry_count` 由 `slot.reset()` 重置，故預算按請求計算（`monitoring_turns` 為 per-conversation 信號）

---

#### 上下文容量管理的標籤邊界保護

啟用 `--context-shift` 時，截斷操作會檢查截斷邊界是否切斷了多 token 組成的特殊標籤（如 `</function>`、`<function=...>`），並自動調整邊界避免割裂標籤，防止模型因看到殘缺標籤而產生異常輸出。

---

#### DRY 採樣防重複參數說明

DRY (Don't Repeat Yourself) 是一種防止模型生成重複內容的採樣機制。

| 參數 | 默認值 | 描述 |
| --- | --- | --- |
| `--dry-multiplier N` | 0.00 | 設置 DRY 採樣乘數（0.0 = 禁用） |
| `--dry-base N` | 1.75 | 設置 DRY 採樣基礎值 |
| `--dry-allowed-length N` | 2 | 設置 DRY 採樣的允許長度 |
| `--dry-penalty-last-n N` | -1 | 設置 DRY 對最後 n 個 token 的懲罰（0 = 禁用，-1 = 上下文大小） |
| `--dry-sequence-breaker STRING` | - | 為 DRY 採樣添加序列中斷符，同時清除默認中斷符 ('\n', ':', '"', '*')；使用 "none" 表示不使用任何序列中斷符 |


示例（啟用 DRY 採樣以防止重複內容）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --dry-multiplier 1.5 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 2048 --dry-sequence-breaker "none"
```

---

#### 推理塊採樣參數覆蓋（Reasoning Sampling Overrides）

當模型生成進入推理塊（如 `<think>...</think>`）時，可為其獨立配置一套採樣參數，與正文（content）部分分開。此機制構建第二條 sampler chain（`chain_think`），在 reasoning budget sampler 偵測到進入推理塊時自動切換，離開推理塊後回到基礎 chain。

所有 `--reasoning-*` 參數均為可選覆蓋項；未覆蓋的參數沿用基礎採樣設定（inherit）。

| 參數 | 預設值 | 描述 |
| --- | --- | --- |
| `--reasoning-temp N` | 0.80 | 推理塊內的温度 |
| `--reasoning-top-k N` | 40 | 推理塊內的 top-k |
| `--reasoning-top-p N` | 0.95 | 推理塊內的 top-p |
| `--reasoning-min-p N` | 0.05 | 推理塊內的 min-p |
| `--reasoning-top-n-sigma N` | -1.00 | 推理塊內的 top-n-sigma |
| `--reasoning-xtc-probability N` | 0.00 | 推理塊內的 XTC 概率 |
| `--reasoning-xtc-threshold N` | 0.10 | 推理塊內的 XTC 閾值 |
| `--reasoning-typical-p N` | 1.00 | 推理塊內的 typical-p |
| `--reasoning-dynatemp-range N` | 0.00 | 推理塊內的動態温度範圍 |
| `--reasoning-dynatemp-exp N` | 1.00 | 推理塊內的動態温度指數（別名：`--reasoning-dynatemp-exponent`） |
| `--reasoning-repeat-last-n N` | 64 | 推理塊內的重複懲罰歷史長度 |
| `--reasoning-repeat-penalty N` | 1.00 | 推理塊內的重複懲罰倍數 |
| `--reasoning-presence-penalty N` | 0.00 | 推理塊內的存在懲罰 |
| `--reasoning-frequency-penalty N` | 0.00 | 推理塊內的頻率懲罰 |
| `--reasoning-dry-multiplier N` | 0.00 | 推理塊內的 DRY 乘數 |
| `--reasoning-dry-base N` | 1.75 | 推理塊內的 DRY 基礎值 |
| `--reasoning-dry-allowed-length N` | 2 | 推理塊內的 DRY 允許長度 |
| `--reasoning-dry-penalty-last-n N` | -1 | 推理塊內的 DRY 歷史範圍 |
| `--reasoning-mirostat N` | 0 | 推理塊內的 Mirostat 模式（0/1/2） |
| `--reasoning-mirostat-ent N` | 5.00 | 推理塊內的 Mirostat 目標熵（別名：`--reasoning-mirostat-tau`） |
| `--reasoning-mirostat-lr N` | 0.10 | 推理塊內的 Mirostat 學習率（別名：`--reasoning-mirostat-eta`） |
| `--reasoning-adaptive-target N` | -1.00 | 推理塊內的自適應採樣目標 |
| `--reasoning-adaptive-decay N` | 0.90 | 推理塊內的自適應衰減 |
| `--reasoning-min-keep N` | 0 | 推理塊內的最小候選數 |
| `--reasoning-seed SEED` | 隨機 | 推理塊內的 RNG 種子 |

**與連續 Token 重複失控檢測的互動：** 當推理塊內啟用 `chain_think` 時，runaway detection 的 `temp_boost` 仍會作用於所有候選 token 的 logit（在 `chain_think` apply 之前），因此推理塊內外的失控循環都能被打破。

示例（推理塊用較高温度 + 較大 top-p，正文用較低温度）：

```sh
./laamaafung/build/bin/Release/llama-server.exe \
  --model /path/to/model.gguf \
  --temp 0.6 --top-p 0.85 \
  --reasoning-temp 1.0 --reasoning-top-p 0.95 \
  --reasoning-repeat-penalty 1.1 --reasoning-repeat-last-n 256
```

也可在 server 啟動後，透過 per-request 欄位動態覆蓋推理塊採樣。OpenAI 相容接口（`/v1/chat/completions`）與 Anthropic 相容接口（`/v1/messages`）均支援以下欄位：

`reasoning_temp`（別名 `reasoning_temperature`）、`reasoning_top_k`、`reasoning_top_p`、`reasoning_min_p`、`reasoning_top_n_sigma`、`reasoning_xtc_probability`、`reasoning_xtc_threshold`、`reasoning_typical_p`、`reasoning_dynatemp_range`、`reasoning_dynatemp_exp`（別名 `reasoning_dynatemp_exponent`）、`reasoning_repeat_last_n`、`reasoning_repeat_penalty`、`reasoning_presence_penalty`、`reasoning_frequency_penalty`、`reasoning_dry_multiplier`、`reasoning_dry_base`、`reasoning_dry_allowed_length`、`reasoning_dry_penalty_last_n`、`reasoning_mirostat`、`reasoning_mirostat_tau`（別名 `reasoning_mirostat_ent`）、`reasoning_mirostat_eta`（別名 `reasoning_mirostat_lr`）、`reasoning_adaptive_target`、`reasoning_adaptive_decay`、`reasoning_min_keep`、`reasoning_seed`。

Anthropic 客户端範例（`/v1/messages`）：

```json
{
  "model": "Agentic-Turbo-Coder",
  "max_tokens": 8192,
  "messages": [{"role": "user", "content": "hello"}],
  "reasoning_temp": 1.0,
  "reasoning_top_p": 0.95,
  "reasoning_repeat_penalty": 1.1
}
```

---

## 編程代理

### Klaude Code

此乃適配本地模型服務的 Klaude Code 版本：https://github.com/naamfung/klaude/releases

默認設置上下文長度為 128k 容量，可使用 `ANTHROPIC_MODEL="Agentic-Turbo-Coder[256k]"` 等方式設置為你本地模型服務開啓的容量上限。


### 簡單配置

可以將以上下載的預編譯版本放入 `$HOME/.local/bin` 或你喜歡的路徑：

```bash
export PATH="$HOME/.local/bin":$PATH
```

### 配置 "Open" Claude / Klaude 環境變量

```bash
export ANTHROPIC_BASE_URL="http://192.168.124.197:8008"
export ANTHROPIC_AUTH_TOKEN="sk-888888"
export ANTHROPIC_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Agentic-Turbo-Coder"
```

### Glash

Glash 是基於我對 crush 的本地化適配，提供終端環境下的編程代理能力：https://github.com/naamfung/glash


### Rex

從響應速度而言，我推薦使用 Rex ，其基於我對 Reasonix 的本地化適配，提供終端環境下的編程代理能力：https://github.com/naamfung/rex

克隆之後用「make build」編譯，將得到的二進制程序放到你係統環境變量可搜索到的路徑，啓動終端運行「rex setup」選 ANTHROPIC 兼容協議配置好 laamaafung server 運行的端口。再次啓動「rex」即可暢享本地模型支持下的編程樂趣。

由於 Reasonix 主線發佈到其生產路徑下的版本根本就不穩定，時神時鬼，不推薦直接使用 Reasonix，建議鎖定一版自認穩定的版本或使用我維護的 Rex 版。


### Dsc

推薦在本地模型環境中使用 Dsc 作為編程代理：https://github.com/naamfung/dsc

---

## llama.cpp

`llama.cpp` is a C/C++ library for LLM inference, designed to enable efficient model inference with minimal setup on a wide range of hardware (Apple Silicon, x86/ARM CPUs, NVIDIA/AMD GPUs, Vulkan, WebGPU, etc.). 

This `laamaafung` fork is based on the `llama.cpp` upstream codebase, focusing on fixing inference engine issues that prevent models from successfully driving agentic long-horizon tasks.