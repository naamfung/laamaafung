# laamaafung

由於 LlamaCpp 項目所有發布的版本序號單純就係提交次數累加卻非真正的生產就緒，故而，此分支的唯一目標是成為一個可驅動智能代理正常工作的穩定版本。

致力於從推理引擎側修復影響模型無法驅動智能代理勝任長程任務的所有問題：模型遞歸生成式的死循環及各種因選項組合未被邏輯正確處理導致的無故停止工作。

优质输出 = （优质模型 + 优质模板 + 优质代理 + 优质引擎）* 正确参数

---

### 克隆指南

推薦優先克隆穩定分支 `v12` 或 `v7`，適合穩定使用。

如需體驗最新功能，推薦克隆最新分支 `v15`；如須測試開發中的功能，可以克隆開發分支 `master`。

- **克隆穩定分支（推薦）**：
  ```sh
  git clone -b v12 https://github.com/naamfung/laamaafung.git
  ```
  或
  ```sh
  git clone -b v7 https://github.com/naamfung/laamaafung.git
  ```

- **克隆最新分支（v15）**：
  如需體驗 v15 最新功能，可執行：
  ```sh
  git clone -b v15 https://github.com/naamfung/laamaafung.git
  ```

- **回退版本（v6）**：
  如須使用舊版穩定分支 `v6`，可執行：
  ```sh
  git clone -b v6 https://github.com/naamfung/laamaafung.git
  ```

- **克隆開發分支**：
  ```sh
  git clone -b master https://github.com/naamfung/laamaafung.git
  ```

---

### 推荐模型

unsloth/Qwen-AgentWorld-35B-A3B 二零二六年六月廿五 / 原版 / 推荐IQ4及以上质量：
https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF/tree/main

mudler/Qwen-AgentWorld-35B-A3B-APEX / 原版 / 建议 APEX-I-Compact 或 APEX-Compact 及以上质量：
https://huggingface.co/mudler/Qwen-AgentWorld-35B-A3B-APEX-GGUF/tree/main

---

### 推荐模板

千问 3.5/3.6/AgentWorld 及以其为基座的衍生模型，建议使用「tmpl」目录内的「Qwen-Agentic-EN / Qwen-Agentic-HON(S/T) 」模板。

---

### 啟動示例

Q80 + Q80：

```sh
D:/Programs/llama-cpp-repos/laamaafung/build-v11/bin/Release/llama-server.exe --model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" --ctx-size 131072 --flash-attn on --reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" --reasoning-format deepseek --reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-repeat-penalty 1.1 --reasoning-repeat-last-n 256 --fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8008 --batch-size auto --ubatch-size auto --load-mode mlock --no-mmproj --cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action boost --cycle-boost-factor 0.5 --chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

Q80 + TURBO4：

```sh
D:/Programs/llama-cpp-repos/laamaafung/build-v11/bin/Release/llama-server.exe --model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" --ctx-size 131072 --flash-attn on --reasoning on --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" --reasoning-format deepseek --fit 1 -ngl all -ngld all --n-cpu-moe 33 --threads 18 --threads-http 2 --parallel 1 --kv-unified --cache-type-k q8_0 --cache-type-v turbo4 --host 0.0.0.0 --port 8008 --batch-size auto --ubatch-size auto --ctx-checkpoints 42 --load-mode mlock-ram --no-mmproj --cache-prompt --cache-ram 8192 --temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-presence-penalty 1.07 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

TURBO4 + TURBO3：

```sh
D:/Programs/llama-cpp-repos/laamaafung/build-v12/bin/Release/llama-server.exe --model "D:/models/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf" --ctx-size 131072 --flash-attn on --reasoning on --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" --reasoning-format deepseek --fit on -ngl all -ngld all --n-cpu-moe 32 --threads 10 --threads-http 2 --parallel 1 --kv-unified --cache-type-k turbo4 --cache-type-v turbo3 --host 0.0.0.0 --port 8008 --batch-size auto --ubatch-size auto --ctx-checkpoints 42 --load-mode mlock-ram --no-mmproj --cache-prompt --cache-ram 8192 --temp 0.6 --top-p 0.85 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --reasoning-temp 1.0 --reasoning-top-p 0.95 --reasoning-presence-penalty 1.07 --jinja --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 24 --spec-ngram-mod-n-max 86 --verbose --chat-template-file D:/Programs/llama-cpp-repos/laamaafung/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

---

#### 啟動參數與工作原理說明

以下係關鍵參數組及其工作原理，方便用戶根據實際需求進行選擇：

| 參數組 | 說明 | 適用場景 |
| --- | --- | --- |
| `--cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64` | 啟用提示緩存（KV 緩存重用）機制。當多個請求有相同或相似的 prompt 前綴時，系統會重用之前計算的 KV 狀態，避免重複計算。`--cache-ram 8192` 設定緩存大小為 8GB，`--checkpoint-min-step 512` 設定創建 checkpoint 的最小步長，`--ctx-checkpoints 64` 設定保留的 checkpoint 數量。 | 適合有大量重複前綴請求、長對話歷史或需要加速響應的場景。 |
| `--cache-reuse N` | 啟用跨 slot 前綴緩存重用。與 `--cache-prompt` 的同 slot 前綴匹配不同，此功能以 256-token 區塊為單位計算鏈式 XXH64 雜湊，當不同 slot 的 prompt 前綴匹配時，透過 `seq_cp` 零拷貝共享 KV cell。N > 0 時自動啟用 `--kv-unified`。多模態模型自動禁用。 | 適合多 slot 並發且共享長前綴的場景（如系統提示詞複用）。N 建議 >= 256（單個區塊大小）。`--cache-prompt` 的同 slot 前綴重用獨立運作，不受此參數影響。 |
| `--context-shift` | 啟用生成階段的運行時 K-shift（KV cache 動態位移）。要求 `llama_memory_can_shift()` 回傳 true，否則會在 context 用盡時優雅停止（`STOP_TYPE_LIMIT`）。K-shift 不可用時自動禁用並警告，初始 prompt 截斷不受影響。隱含啟用 `--prompt-truncate`。 | 適合需要生成階段動態遷移 KV cache 的長程代理任務。 |
| `--prompt-truncate` | 啟用初始 prompt 截斷（當請求 tokens 超過 `--ctx-size` 時自動截斷中間部分並保留頭尾）。對所有模型架構均生效，無需 KV cache 位移支援。由 `--context-shift` 隱含啟用，亦可單獨使用。 | 適合處理超長 prompt 提交、對話歷史較長的場景，避免 HTTP 400 錯誤。 |
| `--swa-full` | 使用與 base cache 等大的全尺寸 SWA cache。僅對 GGUF 模型頭中明確聲明滑動窗口注意力（SWA）且窗口大小固定的模型有效（如 Gemma2/3、Cohere2、Exaone 等）。預設關閉時 SWA cache 僅為 `min(size_base, n_swa + n_ubatch)`，會導致 `llama_kv_cache_iswa::get_can_shift()` 回傳 false，使 `--context-shift` 的運行時 K-shift 失效（初始截斷不受影響）。啟用後 SWA 與 base 等大，K-shift 完全可用。 | 真正採用 SWA 架構的模型需要 `--context-shift` 完整功能（含生成階段運行時 K-shift）時必須配合使用。 |
| `--threads N` / `--threads-batch N` | 設置生成和 batch/prompt 處理的線程數。當 N <= 0（如 -1 或 0）時，系統會使用 `common_cpu_get_num_math()`（即物理數學核心數），而非 `hardware_concurrency()`（所有邏輯核心），以避免在 SMT（超線程）或混合架構 CPU 上過度訂閱導致的性能下降。 | 適合在具有 SMT（超線程）或混合架構（如 Apple M1）的 CPU 上優化 token 生成吞吐量。 |
| `LLAMA_THREADS_RATIO` (環境變數) | 當 `--threads` 為 auto 模式（N <= 0）時，按此比例縮放線程數（範圍 0.1 - 1.0，默認 1.0 即不縮放）。**適用於 GPU + CPU 混合推理場景**（如 MoE 專家層透過 `-ncmoe` 卸載到 CPU），留出部分 CPU 核心給 CUDA driver/sync 工作，可顯著提升 decode 吞吐量。見下方案例。 | GPU + CPU 混合推理（`-ncmoe > 0` 且 `-ngl > 0`）場景。純 CPU 推理或純 GPU 推理無需設置。 |

#### GPU + CPU 混合推理線程調優案例

當 MoE 專家層透過 `-ncmoe` 卸載到 CPU 時，CPU 與 GPU 之間每層都有同步開銷。若 CPU 使用全部物理核心做 MoE 計算，會與 CUDA driver 的 sync 調度競爭，反而降低 decode 吞吐量。設置 `LLAMA_THREADS_RATIO` 留出部分核心可顯著提升性能。

以下為 **Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP**（Qwen35MOE 架構）在 **Xeon E5-2696 v3（18 核/18 線程）+ RTX 3060 Ti 8GB** 上的實測數據（`-ngl 99 -ncmoe 33 -ctk turbo4 -ctv turbo4 -ub 1024`）：

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

> **注意：** 最優比例取決於 CPU 架構、GPU 算力、MoE 卸載比例、模型大小等多個因素，上表數據僅供參考。建議用戶通過 `llama-bench` 實測自身硬件的最優值。設置方式：
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

> **舊組合 `--no-mmap --mlock` 遷移說明：** 舊版 `use_mmap` 與 `use_mlock` 是兩個獨立布爾字段，允許「不用 mmap + 鎖定記憶體」的組合（eager read 載入 CPU buffer 後再 mlock）。上游新版 `--load-mode` 合併為單枚舉，原本不再支持此組合。現 Laamaafung 已新增 `--load-mode mlock-ram` 恢復此行為：直接讀取模型到 RAM 後 mlock，不經過 mmap，避免推理時 mmap page-fault 導致的性能下降。建議根据自身设备的实际参数性能表现选用 `mlock-ram` 或者 `mlock`。

---

### 自動 Batch Size 調優（Auto Batch Size Tuning）

引入了對 `--batch-size` 和 `--ubatch-size` 參數的自動調優支持。此功能為本分支獨有，上游官方分支尚未支援。透過自動調優，程序可在啟動時根據 `n_ctx`（上下文大小）與硬件特徵（如 NUMA 架構狀態）自動計算並選擇最佳的邏輯 batch size (`n_batch`) 與物理 batch size (`n_ubatch`)，以充分發揮硬件並行計算能力並避免內存/Cache 瓶頸。

| 參數 | 說明 |
| --- | --- |
| `--batch-size auto` 或 `--batch-size -1` | 啟用邏輯 batch size (`n_batch`) 自動調優。程序會根據 `n_ctx` 和硬件特徵自動計算最佳值，最大上限為 8192。若系統為 NUMA 架構，則上限降低至 4096。確保最小值 `>= 32`（BLAS 要求）。 |
| `--ubatch-size auto` 或 `--ubatch-size -1` | 啟用物理 batch size (`n_ubatch`) 自動調優。程序會根據 `n_ctx` 和硬件特徵自動計算最佳值，最大上限為 4096。若系統為 NUMA 架構，則上限降低至 2048。確保最小值 `>= 64`（以觸發 Tiled Flash Attention 優化，對應 `Q_TILE_SZ` 閾值）。 |

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

#### TurboQuant 键值缓存 與 MMA 融合路徑

透過 `--cache-type-k` / `--cache-type-v` 指定 TurboQuant 量化類型（`turbo4` / `turbo3` / `turbo2`）可壓縮 KV 缓存佔用。在 CUDA 後端上，只要 GPU 架構為 Turing 及以上（Turing / Ampere / Ada Lovelace / Hopper / Blackwell 等，即 SM 7.5+），系統會自動啟用 MMA 融合注意力路徑（fused turbo MMA）以加速解碼；Volta 及更早架構會自動回退到 VEC 路徑。

| 環境變數 | 預設值 | 描述 |
| --- | --- | --- |
| `GGML_TURBO_MMA_FUSED` | `1`（開啟） | 控制 CUDA fused turbo MMA 路徑。設為 `0` 可關閉，回退到 VEC 路徑（功能完整，僅失去 tensor core 加速）。 |

MMA 融合路徑生效條件：K 與 V 同型且為 `turbo4`/`turbo3`/`turbo2`、`Q->ne[1] <= 4`（解碼場景）、`Q->ne[0]` 為 128 或 256。條件不滿足時自動回退到 VEC 路徑，無需手動干預。

---

#### 啟用上下文容量管理的啟動示例

如果須要處理可能超過上下文限制的請求，可以加入 `--prompt-truncate`（初始截斷）或 `--context-shift`（運行時 K-shift，隱含啟用初始截斷）。對於真正採用 SWA 架構的模型，若需要生成階段的運行時 K-shift 完整可用，須同時加入 `--swa-full`：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf --ctx-size 131072 --flash-attn on --reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "…… 很好，推理经已足矣，现在等我回答。" --reasoning-format deepseek --fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8008 -b 16384 -ub 256 --load-mode mlock --no-mmproj --cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --context-shift --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose --verbosity 5 --chat-template-file /path/to/iStartModel/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

> **注意：** Qwen3.5/3.6 系列模型（MoE 與稠密變體）採用混合注意力機制（門控 DeltaNet 線性注意力 + 門控注意力），並非標準的滑動窗口注意力架構，GGUF 模型頭中 `n_swa = 0`。因此 `--swa-full` 對這些模型無效，載入時會自動檢測並禁用同時彈出警告 `swa_full is not supported by this model, it will be disabled`，此為正確行為，llama.cpp 已自動安全降級。`--context-shift` 會因 K-shift 不可用而自動禁用並警告，但 `--prompt-truncate` 不受影響，初始 prompt 截斷仍然生效。生成階段到達 context 上限時會優雅停止（`STOP_TYPE_LIMIT`）。Qwen3.5/3.6 系列本身支援長上下文（如 256K/512K），無需依賴 SWA 即可高效處理長序列。若想消除日誌噪音，請直接移除 `--swa-full`。`--swa-full` 僅對 GGUF 文件頭中明確聲明滑動窗口注意力且窗口大小固定的模型有效（如 Gemma2/3、Cohere2、Exaone 等）。

---

#### 段級重複循環檢測參數說明

| 參數 | 類型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `--repeat-line-window` | 整数 | 0（已禁用） | 要跟踪的历史片段数量 |
| `--repeat-line-min-length` | 整数 | 20 | 最小片段长度（避免因短语而产生的误报） |
| `--repeat-line-delimiters` | 字符串 | `"\n.!?:。！？："` | 结束一个片段的字符 |
| `--repeat-line-temp-boost` | 浮点数 | 0.5 | 检测到回路时温度升高 |


示例（啟用 repeat_line 采樣器以防止無限循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --repeat-line-window 10 --repeat-line-min-length 20 --repeat-line-delimiters "\n.!?:。！？：" --repeat-line-temp-boost 0.5
```

---

#### 連續 Token 重複失控檢測（內建，無須配置）

當模型陷入同一 token 反覆生成的死循環（例如 `</</</...`），系統會自動偵測並以分級升溫打破循環，無需手動啟用任何參數。此機制與段級重複檢測（`--repeat-line-*`）互補：前者針對行/段級語義重複，本機制針對 token 級的硬性失控。

| 連續次數 | 動作 | 效果（以 base temp = 0.6 為例） |
| --- | --- | --- |
| 8 次 | `temp_boost = 2.0`，即 logit 乘以 1/(1+2) | 等效 temp = 1.8，溫和升溫，嘗試打破循環 |
| 16 次 | `temp_boost = 3.0`，即 logit 乘以 1/(1+3) | 等效 temp = 2.4，強力升溫 |
| 64 次 | `STOP_TYPE_LIMIT` | 升溫無效，強制停止作為最終安全網 |

升溫原理與 `--repeat-line-temp-boost` 相同：對所有候選 token 的 logit 乘以 `1/(1+boost)`，等效於臨時提高採樣溫度。一旦生成的 token 不再重複，boost 立即歸零，恢復正常採樣。

**與 `--repeat_penalty` / `--presence_penalty` 的區別：** 這兩個參數對已出現過的 token 施加持續性懲罰（降低其 logit），但對同一 token 連續出現的硬性失控無效。原因是：當模型對某 token（如 `</`）的 logit 遠高於所有其他候選 token 時，即使施加 1.5x 或 2.0x 的懲罰，此 token 仍然具有最高概率，模型會繼續選擇它，形成死循環。本機制不行"懲罰重複 token"的路線，而是通過升溫（壓縮所有 logit 差距）令低概率 token 獲得被選中的機會，從根本上打破循環。

---

#### 週期性 Token 循環檢測機制（Periodic Token Loop Detection）

當模型生成呈現週期性或交替性的 token 死循環模式（例如「ababab...」、「abcabc...」等，其中 a、b、c 代表不同的獨立 token），內建的連續 token 失控檢測（針對連續相同 token，如 `aaaaaaaa...`）無法有效偵測此類模式。本機制提供專門的週期性重複檢測 sampler，用於檢查最近 token 序列中的循環/交替模式，並在檢測到時應用懲罰或升溫以打破死循環。

| 參數 | 類型 | 預設值 | 描述 |
| --- | --- | --- | --- |
| `--cycle-detect-last-n N` | 整數 | 64 | 要檢查循環模式的最近 token 數量（0 = 停用） |
| `--cycle-detect-min-period N` | 整數 | 2 | 要檢測的最小週期長度 |
| `--cycle-detect-max-period N` | 整數 | 8 | 要檢測的最大週期長度 |
| `--cycle-detect-action TYPE` | 字符串 | `"boost"` | 檢測到循環模式時的動作：`"boost"`（溫度升溫，預設）或 `"penalty"`（重複懲罰） |
| `--cycle-boost-factor F` | 浮點數 | 0.5 | 檢測到循環模式時的升溫因子（僅當 action = `"boost"` 時生效。等效於臨時將採樣溫度提高，壓縮所有 logit 差距） |
| `--cycle-penalty-repeat F` | 浮點數 | 1.00 | 檢測到循環模式時的重複懲罰因子（僅當 action = `"penalty"` 時生效。1.0 = 停用懲罰） |

示例（啟用週期性重複採樣器並採用升溫機制打破死循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action boost --cycle-boost-factor 0.5
```

示例（啟用週期性重複採樣器並採用懲罰機制打破死循環）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --cycle-detect-last-n 64 --cycle-detect-min-period 2 --cycle-detect-max-period 8 --cycle-detect-action penalty --cycle-penalty-repeat 1.5
```

**升溫原理：** 當選擇 `"boost"` 動作時，對所有候選 token 的 logit 乘以 `1/(1+boost_factor)`，等效於臨時提高採樣溫度。一旦生成的 token 不再呈現週期循環，檢測機制會重置，恢復正常採樣。

**與連續 Token 失控檢測的區別：** 內建的連續 token 失控檢測僅針對「連續相同 token」的硬性失控（如 `aaaaaaaa...` 或 `</</</...`）。而本機制專門針對 token-level 的交替/週期性循環模式（如 `ababab...`、`abcabc...`），透過週期檢測演算法（基於字串最小週期匹配屬性）識別並打破此類死循環。

---

#### 早停檢測與 EOG 抑制（Early-Stop Detection & EOG Suppression）

當模型在思考完成後未產生任何可見輸出就自動停止（例如思考結束但無正文回答，或工具調用被截斷），客戶端會收到空響應且無明顯錯誤。流式模式下此問題尤其嚴重：用戶可能完全不會察覺本回合已丟失。本機制通過雙層設計驅動模型持續生成直至出現真實內容，並對觸發此機制的 slot 開啟 5 回合高強度監控。

| 參數 | 類型 | 預設值 | 描述 |
| --- | --- | --- | --- |
| `--eog-retry-max N` | 整數 | 3 | 早停檢測觸發後的最大重試次數（0 = 禁用）。同時控制 slot 層 EOG 抑制次數和 HTTP 層非流式重試次數。 |

**雙層架構設計：**

1. **Slot 層 EOG 攔截（流式透明）** - 在 `process_token()` 中，當採樣到 EOG token 時檢查 `slot.generated_text` 判斷是否為缺陷性早停。若是，則不設置 `STOP_TYPE_EOS`，而是啟用 EOG 抑制（`common_sampler_set_suppress_eog`），將所有 EOG token 的 logit 強制設為 `-INFINITY`，令 sampler 返回次優 token 以繼續生成。由於抑制發生在 sampler 內部，流式客戶端看到的是不中斷的 token 流，無需倒帶或重啟，且正常路徑與投機採樣路徑（共用 `common_sampler_sample()`）均被覆蓋。

2. **HTTP 層重試（非流式）** - 對非流式請求，`handle_completions_impl()` 將任務創建封裝為 `create_tasks` lambda 並運行重試循環：若最終結果的 `oaicompat_msg.content` 與 `tool_calls` 均為空，則重新提交任務。從第 2 次重試起 bump 採樣種子（`seed += http_retry`）以避免重複採樣同一死胡同。循環受 `--eog-retry-max` 約束。

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

單純的早停條件檢查無法區分"模型已正常完成簡短回覆"與"模型異常截斷"兩種場景。為此監控機制在觸發早停時不直接抑制 EOG，而是先注入一個對客戶端透明的隱藏 turn，讓模型自判回覆是否完整，再根據判定結果決定是否抑制 EOG 並繼續生成。自檢使用內部協議標記 `<complete>` / `<incomplete>`，不依賴任何廠商特定標籤（如 `<think>`、`<tool_call>`），故通用於 Qwen、Gemma 等所有模型。

自檢狀態機（per-slot，`SELF_CHECK_*` 階段）：

```
NONE -> PREFILL -> GENERATING -> {NONE | ROLLBACK} -> NONE
```

1. **NONE -> PREFILL**：`process_token()` 中 `early_stop_no_output` 為真且 `cached_messages` 非空時觸發。`build_self_check_prompt()` 通過 `common_chat_format_single()` 構建增量 prompt（`past_msg = cached_messages + assistant_msg(generated_text)`，`new_msg = user "Review completeness, output <complete>/<incomplete>"`），tokenize 後入隊 `self_check_prefill`。
2. **PREFILL -> GENERATING**：`handle_last_sampled_token()` 將觸發 EOS 以 `output=false` 加入 batch，隨後追加 prefill tokens（末 token `output=true`）。保存回滾狀態（KV 位置、prompt 大小、EOS token id），sampler 同步接受每個 prefill token 以保持懲罰/repeat 狀態一致。
3. **GENERATING**：`process_token()` 將自檢回覆 token 路由至 `handle_self_check_token()`，累積文本至 `self_check_text`（清空 `text_to_send` 對客戶端隱藏），監測 `<complete>`/`<incomplete>` 標記。安全閥：回覆超過 64 字符仍無標記時保守判定為 incomplete。
4. **完整判定**：`<complete>` -> `STOP_TYPE_EOS`，自檢 turn 留在 cache 中由下個請求覆蓋；`<incomplete>` -> 進入 `ROLLBACK`。
5. **ROLLBACK -> NONE**：`handle_last_sampled_token()` 通過 `common_context_seq_rm()` 從 KV cache 截斷整個自檢 turn，`keep_first()` 截斷 `prompt.tokens`，重新以 `output=true` 評估觸發 EOS，arm EOG 抑制後模型在原回覆上繼續生成。

自檢全過程所有 token 的 `text_to_send` 均被清空，流式與非流式客戶端均無感知。`generated_text`、stop-word 偵測、partial response、EOG 早停邏輯在 GENERATING 階段全部旁路。

**重試預算：**
- `--eog-retry-max` 同時控制 slot 層抑制次數與 HTTP 層非流式重試次數
- `slot.eog_retry_count` 由 `slot.reset()` 重置，故預算按請求計算（`monitoring_turns` 為 per-conversation 信號）

---

#### 上下文容量管理的標籤邊界保護

啟用 `--context-shift` 時，截斷操作會檢查截斷邊界是否切斷了多 token 組成的特殊標籤（如 `</function>`、`<function=...>`），並自動調整邊界避免割裂標籤，防止模型因看到殘缺標籤而產生異常輸出。

---

#### DRY 采样防重复参数说明

DRY (Don't Repeat Yourself) 是一种防止模型生成重复内容的采样机制。

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| `--dry-multiplier N` | 0.00 | 设置 DRY 采样乘数（0.0 = 禁用） |
| `--dry-base N` | 1.75 | 设置 DRY 采样基础值 |
| `--dry-allowed-length N` | 2 | 设置 DRY 采样的允许长度 |
| `--dry-penalty-last-n N` | -1 | 设置 DRY 对最后 n 个 token 的惩罚（0 = 禁用，-1 = 上下文大小） |
| `--dry-sequence-breaker STRING` | - | 为 DRY 采样添加序列中断符，同时清除默认中断符 ('\n', ':', '"', '*')；使用 "none" 表示不使用任何序列中断符 |


示例（啟用 DRY 采樣以防止重複內容）：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/model.gguf --dry-multiplier 1.5 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 2048 --dry-sequence-breaker "none"
```

---

#### 推理塊採樣參數覆蓋（Reasoning Sampling Overrides）

當模型生成進入推理塊（如 `<think>...</think>`）時，可為其獨立配置一套採樣參數，與正文（content）部分分開。此機制構建第二條 sampler chain（`chain_think`），在 reasoning budget sampler 偵測到進入推理塊時自動切換，離開推理塊後回到基礎 chain。

所有 `--reasoning-*` 參數均為可選覆蓋項；未覆蓋的參數沿用基礎採樣設定（inherit）。

| 參數 | 預設值 | 描述 |
| --- | --- | --- |
| `--reasoning-temp N` | 0.80 | 推理塊內的溫度 |
| `--reasoning-top-k N` | 40 | 推理塊內的 top-k |
| `--reasoning-top-p N` | 0.95 | 推理塊內的 top-p |
| `--reasoning-min-p N` | 0.05 | 推理塊內的 min-p |
| `--reasoning-top-n-sigma N` | -1.00 | 推理塊內的 top-n-sigma |
| `--reasoning-xtc-probability N` | 0.00 | 推理塊內的 XTC 概率 |
| `--reasoning-xtc-threshold N` | 0.10 | 推理塊內的 XTC 閾值 |
| `--reasoning-typical-p N` | 1.00 | 推理塊內的 typical-p |
| `--reasoning-dynatemp-range N` | 0.00 | 推理塊內的動態溫度範圍 |
| `--reasoning-dynatemp-exp N` | 1.00 | 推理塊內的動態溫度指數（別名：`--reasoning-dynatemp-exponent`） |
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

示例（推理塊用較高溫度 + 較大 top-p，正文用較低溫度）：

```sh
./laamaafung/build/bin/Release/llama-server.exe \
  --model /path/to/model.gguf \
  --temp 0.6 --top-p 0.85 \
  --reasoning-temp 1.0 --reasoning-top-p 0.95 \
  --reasoning-repeat-penalty 1.1 --reasoning-repeat-last-n 256
```

也可在 server 啟動後，透過 per-request 欄位動態覆蓋推理塊採樣。OpenAI 相容接口（`/v1/chat/completions`）與 Anthropic 相容接口（`/v1/messages`）均支援以下欄位：

`reasoning_temp`（別名 `reasoning_temperature`）、`reasoning_top_k`、`reasoning_top_p`、`reasoning_min_p`、`reasoning_top_n_sigma`、`reasoning_xtc_probability`、`reasoning_xtc_threshold`、`reasoning_typical_p`、`reasoning_dynatemp_range`、`reasoning_dynatemp_exp`（別名 `reasoning_dynatemp_exponent`）、`reasoning_repeat_last_n`、`reasoning_repeat_penalty`、`reasoning_presence_penalty`、`reasoning_frequency_penalty`、`reasoning_dry_multiplier`、`reasoning_dry_base`、`reasoning_dry_allowed_length`、`reasoning_dry_penalty_last_n`、`reasoning_mirostat`、`reasoning_mirostat_tau`（別名 `reasoning_mirostat_ent`）、`reasoning_mirostat_eta`（別名 `reasoning_mirostat_lr`）、`reasoning_adaptive_target`、`reasoning_adaptive_decay`、`reasoning_min_keep`、`reasoning_seed`。

Anthropic 客戶端範例（`/v1/messages`）：

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

## Paged KV Cache（vLLM 风格分页 KV 缓存）

本分支默认启用 vLLM 风格的分页（paged）KV cache：K/V 按固定大小 block 分配，支持跨请求前缀共享、LRU 驱逐、swap 抢占、KV 量化与 CUDA Graph。可用 `LLAMA_KV_LEGACY=1` 环境变量切回旧式连续缓存。

### 核心能力

| 能力 | 说明 |
|---|---|
| 前缀共享（APC） | hash 链匹配，跨请求/跨 slot 复用相同前缀的 KV 块（含不满尾块）；配合 `--cache-prompt` |
| 跨重启持久化 | `--cache-dir <dir>`：退出时自动保存、启动时自动加载（模型指纹校验）；或手动 `POST /cache/save` + `POST /cache/load` |
| Preemption | 池满时按（优先级, LRU）选 victim：swap（存 CPU 内存，`LLAMA_KV_SWAP_COMPRESS=1` 启用压缩）或驱逐重算 |
| 超池降级 | 并发长 prompt 超出池容量时串行排队执行（不整批 500、不崩溃，生成中请求不中断） |
| KV 量化 | `--cache-type-k/--cache-type-v`（f16/bf16/turbo2/3/4/q8_0/q4_0 等），约省 50% 显存 |
| CUDA Graph | 默认兼容开启；`LLAMA_KV_PAGED_DISABLE_GRAPHS=1` 禁用 |
| 可观测性 | `/metrics` 默认启用：`kv_blocks_total/free/used/cached`、`kv_cache_usage`、`kv_swapped_tokens`、`kv_preempt_count`、`kv_swap_out/in_count` |

### 正确使用

```sh
# 推荐：开启缓存复用 + 量化 + metrics
./llama-server -m model.gguf -c 32768 -np 4 \
  --cache-prompt --metrics \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on    # 量化 V 必须开 flash-attn
```

- 池容量 = `n_ctx`（每 slot 分配 `n_ctx/n_parallel`）；并发 prompt 总需求超过池时自动降级为串行执行。
- KV 量化：量化 V（q8_0/q4_0/turbo 等）必须 `--flash-attn on`；浮点（f16/bf16）无此限制。
- 跨重启复用（推荐：指定 `--cache-dir`，退出自动保存、启动自动加载，无需 HTTP）：

```sh
./llama-server -m model.gguf -c 32768 -np 4 --cache-prompt --cache-dir ./cache-files ...
```

- 手动控制（默认关闭，需 `--cache-endpoint` 或环境变量 `LLAMA_ARG_ENDPOINT_CACHE` 开启；空闲时执行，load 会清空当前池，有请求处理中时自动排队等待）：

```sh
# 启动时开启端点
LLAMA_ARG_ENDPOINT_CACHE=1 ./llama-server ...   # 或 --cache-endpoint

curl -X POST localhost:8080/cache/save -d '{"path":"/path/prefix-cache.bin"}'
# 重启 server 后
curl -X POST localhost:8080/cache/load -d '{"path":"/path/prefix-cache.bin"}'
```

- 回归验证：`./test-kv-paged.exe -fa 0`（dense mock + 共享/swap/量化/持久化场景全过，约 40 项）。

### 环境变量

| 变量 | 作用 |
|---|---|
| `LLAMA_KV_LEGACY=1` | 切回旧式连续 KV 缓存 |
| `LLAMA_KV_BLOCK_SIZE` | 覆盖自适应 block size（默认按 n_embd_k 自动选择） |
| `LLAMA_KV_PAGED_DISABLE_GRAPHS=1` | 禁用 CUDA Graph 路径 |
| `LLAMA_KV_SWAP_COMPRESS=1` | swap 时无损压缩 K/V（默认不压缩，对齐 vLLM） |
| `LLAMA_KV_RS_SNAPSHOTS` | hybrid 模型 recurrent 快照数（默认 32） |

---

## 编程代理

### Klaude Code

此乃适配本地模型服务的 Klaude Code 版本：https://github.com/naamfung/klaude/releases

默认设置上下文长度为 128k 容量，可使用 `ANTHROPIC_MODEL="Agentic-Turbo-Coder[256k]"` 等方式设置为你本地模型服务开启的容量上限。


### 简单配置

可以将以上下载的预编译版本放入 `$HOME/.local/bin` 或你喜欢的路径：

```bash
export PATH="$HOME/.local/bin":$PATH
```

### 配置 "Open" Claude / Klaude 环境变量

```bash
export ANTHROPIC_BASE_URL="http://192.168.124.197:8008"
export ANTHROPIC_AUTH_TOKEN="sk-888888"
export ANTHROPIC_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Agentic-Turbo-Coder"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Agentic-Turbo-Coder"
```

### Glash

Glash 是基于我对 crush 的本地化适配，提供终端環境下的编程代理能力：https://github.com/naamfung/glash


### Inx

从响应速度而言，我推荐使用 Inx ，其基于我对 Reasonix 的本地化适配，提供终端環境下的编程代理能力：https://github.com/naamfung/inx

克隆之后用「make build」编译，将得到的二进制程序放到你系统环境变量可搜索到的路径，启动终端运行「inx setup」选 ANTHROPIC 兼容协议配置好 laamaafung server 运行的端口。再次启动「inx」即可畅享本地模型支持下的编程乐趣。


### Dsc

推荐在本地模型环境中使用 Dsc 作为编程代理：https://github.com/naamfung/dsc

---

## llama.cpp

`llama.cpp` is a C/C++ library for LLM inference, designed to enable efficient model inference with minimal setup on a wide range of hardware (Apple Silicon, x86/ARM CPUs, NVIDIA/AMD GPUs, Vulkan, WebGPU, etc.). 

This `laamaafung` fork is based on the `llama.cpp` upstream codebase, focusing on fixing inference engine issues that prevent models from successfully driving agentic long-horizon tasks.