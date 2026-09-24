| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
=== [v26-9B-base] start 17:18:40 ===
| v26-9B-base | gen 59.0 t/s (tg_3s峰值 59.93) | prefill 60 t/s |  | 5358 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0` |
=== [v26-9B-f16] start 17:19:48 ===
| v26-9B-f16 | gen 59.4 t/s (tg_3s峰值 60.11) | prefill 63 t/s |  | 5478 MB | `-ngl 99 -c 8192 -ctk f16 -ctv f16` |
=== [v26-9B-turbo4] start 17:20:51 ===
| v26-9B-turbo4 | gen 58.7 t/s (tg_3s峰值 59.11) | prefill 62 t/s |  | 5332 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v26-9B-turbo3] start 17:21:18 ===
| v26-9B-turbo3 | gen 57.8 t/s (tg_3s峰值 58.76) | prefill 59 t/s |  | 5322 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo3` |
=== [v26-9B-turbo2] start 17:22:23 ===
| v26-9B-turbo2 | gen 58.3 t/s (tg_3s峰值 59.07) | prefill 60 t/s |  | 5314 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo2` |
=== [v26-9B-tq3] start 17:23:27 ===
[v26-9B-tq3] STARTUP FAILED
0.00.080.708 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.080.708 W srv  llama_server: -----------------
0.00.087.994 I srv    load_model: loading model 'C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf'
0.03.774.835 I cmn          init: llama threadpool init, n_threads = 18
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v26-9B-tq2] start 17:28:32 ===
[v26-9B-tq2] STARTUP FAILED
0.00.077.600 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.077.601 W srv  llama_server: -----------------
0.00.085.718 I srv    load_model: loading model 'C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf'
0.03.786.928 I cmn          init: llama threadpool init, n_threads = 18
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v26-9B-turbo1.5] start 17:33:36 ===
| v26-9B-turbo1.5 | gen 59.2 t/s (tg_3s峰值 59.93) | prefill 60 t/s |  | 5328 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5` |
=== [v26-9B-kvarn4] start 17:34:40 ===
| v26-9B-kvarn4 | gen 55.7 t/s (tg_3s峰值 57.36) | prefill 575 t/s |  | 5380 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4` |
=== [v26-9B-kvarn8-3] start 17:35:49 ===
| v26-9B-kvarn8-3 | gen 55.6 t/s (tg_3s峰值 56.95) | prefill 573 t/s |  | 5372 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn3` |
=== [v26-9B-kvarn4-3] start 17:36:56 ===
| v26-9B-kvarn4-3 | gen 55.5 t/s (tg_3s峰值 56.79) | prefill 536 t/s |  | 5340 MB | `-ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn3` |
=== [v26-9B-kvarn3-2] start 17:38:06 ===
| v26-9B-kvarn3-2 | gen 55.5 t/s (tg_3s峰值 56.91) | prefill 518 t/s |  | 5324 MB | `-ngl 99 -c 8192 -ctk kvarn3 -ctv kvarn2` |
=== [v26-9B-kvarn44] start 17:39:16 ===
| v26-9B-kvarn44 | gen 55.4 t/s (tg_3s峰值 56.87) | prefill 561 t/s |  | 5348 MB | `-ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn4` |
=== [v26-9B-turbo4-3] start 17:40:26 ===
| v26-9B-turbo4-3 | gen 57.6 t/s (tg_3s峰值 58.28) | prefill 58 t/s |  | 5280 MB | `-ngl 99 -c 8192 -ctk turbo4 -ctv turbo3` |
=== [v26-9B-turbo3-2] start 17:41:31 ===
| v26-9B-turbo3-2 | gen 59.4 t/s (tg_3s峰值 60.04) | prefill 66 t/s |  | 5271 MB | `-ngl 99 -c 8192 -ctk turbo3 -ctv turbo2` |
=== [v26-9B-kvmem] start 17:42:35 ===
| v26-9B-kvmem | gen 58.1 t/s (tg_3s峰值 59.08) | prefill 60 t/s |  | 5195 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-9B-kvmem-turbo4] start 17:43:40 ===
| v26-9B-kvmem-turbo4 | gen 58.5 t/s (tg_3s峰值 59.05) | prefill 61 t/s |  | 5181 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-9B-kvmem-kvarn4] start 17:44:05 ===
| v26-9B-kvmem-kvarn4 | gen 58.2 t/s (tg_3s峰值 60.86) | prefill 618 t/s |  | 5171 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-9B-vision] start 17:45:12 ===
| v26-9B-vision | gen 58.1 t/s (tg_3s峰值 59.16) | prefill 62 t/s |  | 5353 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v25-9B-base] start 17:45:54 ===
| v25-9B-base | gen 59.7 t/s (tg_3s峰值 61.7) | prefill 64 t/s |  | 5398 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0` |
=== [v25-9B-f16] start 17:46:58 ===
| v25-9B-f16 | gen 59.8 t/s (tg_3s峰值 61.84) | prefill 67 t/s |  | 5498 MB | `-ngl 99 -c 8192 -ctk f16 -ctv f16` |
=== [v25-9B-turbo4] start 17:48:01 ===
| v25-9B-turbo4 | gen 59.0 t/s (tg_3s峰值 59.78) | prefill 61 t/s |  | 5358 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v25-9B-turbo3] start 17:49:05 ===
| v25-9B-turbo3 | gen 58.4 t/s (tg_3s峰值 59.46) | prefill 61 t/s |  | 5345 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo3` |
=== [v25-9B-turbo2] start 17:50:09 ===
| v25-9B-turbo2 | gen 59.0 t/s (tg_3s峰值 59.46) | prefill 66 t/s |  | 5347 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo2` |
=== [v25-9B-tq3] start 17:51:14 ===
[v25-9B-tq3] STARTUP FAILED
0.00.216.419 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.216.419 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.216.419 W srv  llama_server: -----------------
0.00.226.824 I srv    load_model: loading model 'C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf'
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v25-9B-tq2] start 17:56:20 ===
[v25-9B-tq2] STARTUP FAILED
0.00.232.127 W srv  llama_server: this can be a security risk (cross-origin attacks)
0.00.232.127 W srv  llama_server: more info: https://github.com/ggml-org/llama.cpp/pull/25655
0.00.232.127 W srv  llama_server: -----------------
0.00.241.752 I srv    load_model: loading model 'C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix/Qwen3.5-9B-Uncensored-Genesis-FITKIT-Q3_K_L-DOWN-4.88G-genesis-imatrix.gguf'
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v25-9B-turbo1.5] start 18:01:24 ===
| v25-9B-turbo1.5 | gen 59.8 t/s (tg_3s峰值 60.7) | prefill 62 t/s |  | 5334 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5` |
=== [v25-9B-kvmem] start 18:02:27 ===
| v25-9B-kvmem | gen 59.8 t/s (tg_3s峰值 60.39) | prefill 633 t/s |  | 5184 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v25-9B-kvmem-turbo4] start 18:03:31 ===
| v25-9B-kvmem-turbo4 | gen 59.5 t/s (tg_3s峰值 60.04) | prefill 628 t/s |  | 5171 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v25-9B-vision] start 18:04:35 ===
| v25-9B-vision | gen 59.6 t/s (tg_3s峰值 60.09) | prefill 62 t/s |  | 5338 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v26-35B-base] start 18:05:14 ===
| v26-35B-base | gen 24.8 t/s (tg_3s峰值 25.73) | prefill 25 t/s |  | 4175 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v26-35B-mtp] start 18:07:43 ===
| v26-35B-mtp | gen 26.9 t/s (tg_3s峰值 30.7) | prefill 28 t/s | 0.50769/2.52 | 5566 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
=== [v26-35B-mtp-nmax2] start 18:09:57 ===
| v26-35B-mtp-nmax2 | gen 29.6 t/s (tg_3s峰值 32.16) | prefill 25 t/s | 0.65559/2.31 | 5324 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2` |
=== [v26-35B-mtp-turbo3] start 18:12:03 ===
| v26-35B-mtp-turbo3 | gen 31.4 t/s (tg_3s峰值 34.48) | prefill 28 t/s | 0.61538/2.84 | 5565 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo3 --spec-type draft-mtp` |
=== [v26-35B-mtp-turbo2] start 18:14:05 ===
| v26-35B-mtp-turbo2 | gen 31.0 t/s (tg_3s峰值 34.76) | prefill 29 t/s | 0.58513/2.76 | 5558 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo2 --spec-type draft-mtp` |
=== [v26-35B-mtp-tq3] start 18:16:05 ===
[v26-35B-mtp-tq3] STARTUP FAILED
0.00.082.972 W srv  llama_server: -----------------
0.00.099.741 I srv    load_model: loading model 'C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf'
0.02.386.021 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
0.05.775.113 I cmn          init: llama threadpool init, n_threads = 18
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v26-35B-mtp-tq3-nmax2] start 18:21:10 ===
[v26-35B-mtp-tq3-nmax2] STARTUP FAILED
0.00.066.551 W srv  llama_server: -----------------
0.00.069.268 I srv    load_model: loading model 'C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf'
0.02.386.891 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
0.05.873.314 I cmn          init: llama threadpool init, n_threads = 18
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v26-35B-mtp-kvarn4] start 18:26:15 ===
| v26-35B-mtp-kvarn4 | gen 31.5 t/s (tg_3s峰值 34.89) | prefill 83 t/s | 0.57619/2.73 | 5606 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp` |
=== [v26-35B-mtp-kvarn4-nmax2] start 18:28:16 ===
| v26-35B-mtp-kvarn4-nmax2 | gen 30.4 t/s (tg_3s峰值 33.76) | prefill 81 t/s | 0.68944/2.38 | 5402 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp --spec-draft-n-max 2` |
=== [v26-35B-mtp-kvarn8-3] start 18:30:20 ===
| v26-35B-mtp-kvarn8-3 | gen 31.6 t/s (tg_3s峰值 34.33) | prefill 81 t/s | 0.59322/2.78 | 5630 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn3 --spec-type draft-mtp` |
=== [v26-35B-mtp-kvarn3-2] start 18:32:21 ===
| v26-35B-mtp-kvarn3-2 | gen 29.3 t/s (tg_3s峰值 33.85) | prefill 74 t/s | 0.69375/2.39 | 5328 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn3 -ctv kvarn2 --spec-type draft-mtp --spec-draft-n-max 2` |
=== [v26-35B-mtp-turbo3-2] start 18:34:22 ===
| v26-35B-mtp-turbo3-2 | gen 30.3 t/s (tg_3s峰值 32.62) | prefill 25 t/s | 0.58795/2.76 | 5537 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk turbo3 -ctv turbo2 --spec-type draft-mtp` |
=== [v26-35B-mtp-kvmem] start 18:36:33 ===
| v26-35B-mtp-kvmem | gen 25.6 t/s (tg_3s峰值 30.37) | prefill 26 t/s | 0.52125/2.55 | 4829 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-35B-mtp-kvmem-nmax2] start 18:38:47 ===
| v26-35B-mtp-kvmem-nmax2 | gen 26.4 t/s (tg_3s峰值 32.57) | prefill 25 t/s | 0.62647/2.25 | 4751 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-35B-mtp-kvmem-tq3] start 18:40:50 ===
[v26-35B-mtp-kvmem-tq3] STARTUP FAILED
0.00.088.513 I srv    load_model: loading model 'C:/WorkModels/Qwen3.6-35B-A3B/Mudler/Qwen-AgentWorld-35B-A3B-APEX-I-Compact-MTP.gguf'
0.00.088.529 W KVMem is enabled: the KV working set is managed by the KVMem pool and cannot be statically fitted; forcing -fit off
0.02.376.822 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance
0.05.751.649 I cmn          init: llama threadpool init, n_threads = 18
G:\Agents\kvmem-works\wt-v25\ggml\src\ggml-cuda\fattn.cu:538: fatal error
=== [v26-35B-vision-mtp] start 18:45:55 ===
| v26-35B-vision-mtp | gen 30.9 t/s (tg_3s峰值 31.95) | prefill 28 t/s | 0.58173/2.73 | 5568 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
=== [v25-35B-base] start 18:47:13 ===
| v25-35B-base | gen 23.8 t/s (tg_3s峰值 25.31) | prefill 27 t/s |  | 4677 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4` |
=== [v25-35B-mtp] start 18:49:37 ===
| v25-35B-mtp | gen 29.2 t/s (tg_3s峰值 32.8) | prefill 27 t/s | 0.54005/2.62 | 5593 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
=== [v25-35B-mtp-nmax2] start 18:51:45 ===
| v25-35B-mtp-nmax2 | gen 34.3 t/s (tg_3s峰值 36.31) | prefill 29 t/s | 0.74593/2.49 | 5341 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2` |
=== [v25-35B-mtp-turbo3] start 18:53:36 ===
