# laamaafung

由於 LlamaCpp 項目所有發布的版本序號單純就係提交次數累加卻非真正的生產就緒，故而，此分支的唯一目標是成為一個可驅動智能代理正常工作的穩定版本。

致力於從推理引擎側修復影響模型無法驅動智能代理勝任長程任務的所有問題：模型遞歸生成式的死循環及各種因選項組合未被邏輯正確處理導致的無故停止工作。

### 克隆指南

推薦優先克隆穩定分支，当前为`v1`，此分支係遠端默認分支，適合穩定使用。

如须測試最新功能，可以克隆開發分支`master`。

- **克隆穩定分支（推薦）**：
  ```sh
  git clone https://github.com/naamfung/laamaafung.git
  ```

  或

  ```sh
  git clone -b v1 https://github.com/naamfung/laamaafung.git
  ```

- **克隆開發分支**：
  ```sh
  git clone -b master https://github.com/naamfung/laamaafung.git
  ```

---

啟動示例：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf --ctx-size 131072 --flash-attn on --reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "...enough. Need to give the final output now!" --reasoning-format deepseek --fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8008 -b 16384 -ub 256 --no-mmap --mlock --no-mmproj --cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose --verbosity 5 --chat-template-file /path/to/iStartModel/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

#### 啟動參數與工作原理說明

以下係關鍵參數組及其工作原理，方便用戶根據實際需求進行選擇：

| 參數組 | 說明 | 適用場景 |
| --- | --- | --- |
| `--cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64` | 啟用提示緩存（KV 緩存重用）機制。當多個請求有相同或相似的 prompt 前綴時，系統會重用之前計算的 KV 狀態，避免重複計算。`--cache-ram 8192` 設定緩存大小為 8GB，`--checkpoint-min-step 512` 設定創建 checkpoint 的最小步長，`--ctx-checkpoints 64` 設定保留的 checkpoint 數量。 | 適合有大量重複前綴請求、長對話歷史或需要加速響應的場景。 |
| `--context-shift` | 啟用上下文遷移功能。當請求的 tokens 超過 `--ctx-size` 限制時，系統會自動截斷 prompt 的中間部分並保留前 `n_keep` 個 token，或進行上下文遷移，確保任務繼續執行而唔會返回 HTTP 400 錯誤。 | 適合處理超長上下文、對話歷史較長或容易觸發上下文上限的長程代理任務。 |

#### 啟用上下文遷移的啟動示例

如果須要處理可能超過上下文限制的請求，可以加入 `--context-shift` 參數：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf --ctx-size 131072 --flash-attn on --reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "...enough. Need to give the final output now!" --reasoning-format deepseek --fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 1 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8008 -b 16384 -ub 256 --no-mmap --mlock --no-mmproj --cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --context-shift --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose --verbosity 5 --chat-template-file /path/to/iStartModel/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
```

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

#### 連續 Token 重複失控檢測（內建，無須配置）

當模型陷入同一 token 反覆生成的死循環（例如 `</</</...`），系統會自動偵測並以分級升溫打破循環，無需手動啟用任何參數。此機制與段級重複檢測（`--repeat-line-*`）互補：前者針對行/段級語義重複，本機制針對 token 級的硬性失控。

| 連續次數 | 動作 | 效果（以 base temp = 0.6 為例） |
| --- | --- | --- |
| 8 次 | `temp_boost = 2.0`，即 logit 乘以 1/(1+2) | 等效 temp = 1.8，溫和升溫，嘗試打破循環 |
| 16 次 | `temp_boost = 3.0`，即 logit 乘以 1/(1+3) | 等效 temp = 2.4，強力升溫 |
| 64 次 | `STOP_TYPE_LIMIT` | 升溫無效，強制停止作為最終安全網 |

升溫原理與 `--repeat-line-temp-boost` 相同：對所有候選 token 的 logit 乘以 `1/(1+boost)`，等效於臨時提高採樣溫度。一旦生成的 token 不再重複，boost 立即歸零，恢復正常採樣。

#### 上下文遷移的標籤邊界保護

啟用 `--context-shift` 時，截斷操作會檢查截斷邊界是否切斷了多 token 組成的特殊標籤（如 `</function>`、`<function=...>`），並自動調整邊界避免割裂標籤，防止模型因看到殘缺標籤而產生異常輸出。

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


# llama.cpp

`llama.cpp` is a C/C++ library for LLM inference, designed to enable efficient model inference with minimal setup on a wide range of hardware (Apple Silicon, x86/ARM CPUs, NVIDIA/AMD GPUs, Vulkan, WebGPU, etc.). 

This `laamaafung` fork is based on the `llama.cpp` upstream codebase, focusing on fixing inference engine issues that prevent models from successfully driving agentic long-horizon tasks.