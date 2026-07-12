# laamaafung

由於 LlamaCpp 項目所有發布的版本序號單純就係提交次數累加卻非真正的生產就緒，故而，此分支的唯一目標是成為一個可驅動智能代理正常工作的穩定版本。

致力於從推理引擎側修復影響模型無法驅動智能代理勝任長程任務的所有問題：模型遞歸生成式的死循環及各種因選項組合未被邏輯正確處理導致的無故停止工作。


啟動示例：

```sh
./laamaafung/build/bin/Release/llama-server.exe --model /path/to/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf --ctx-size 131072 --flash-attn on --reasoning on --reasoning-preserve --reasoning-budget 8192 --reasoning-budget-message "...enough. Need to give the final output now!" --reasoning-format deepseek --fit 1 -ngl all --n-cpu-moe 34 --threads 18 --threads-http 2 --parallel 2 --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8008 -b 16384 -ub 256 --no-mmap --mlock --no-mmproj --cache-prompt --cache-ram 8192 --checkpoint-min-step 512 --ctx-checkpoints 64 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat_penalty 1.0 --presence_penalty 0.0 --jinja --spec-type draft-mtp --spec-draft-n-max 4 --verbose --verbosity 5 --chat-template-file /path/to/iStartModel/tmpl/Qwen-Agentic-HONT.jinja --alias Agentic-Turbo-Coder
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