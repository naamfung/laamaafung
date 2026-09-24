
## 9B Q4_K_M 生产基准矩阵（config 复刻，72.38 锚点）
| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
=== [v26-9B-replica] start 19:59:19 ===
| v26-9B-replica | gen 73.5 t/s (tg_3s峰值 74.36) | prefill 70 t/s |  | 峰值7689MiB(余324MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072` |
=== [v25-9B-replica] start 20:00:29 ===
| v25-9B-replica | gen 75.8 t/s (tg_3s峰值 76.19) | prefill 66 t/s |  | 峰值7693MiB(余328MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072` |
=== [v26-9B-q8q8] start 20:01:35 ===
| v26-9B-q8q8 | gen 65.3 t/s (tg_3s峰值 66.32) | prefill 66 t/s |  | 峰值7849MiB(余164MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctv q8_0` |
=== [v26-9B-t4t3] start 20:02:43 ===
| v26-9B-t4t3 | gen 72.2 t/s (tg_3s峰值 72.93) | prefill 62 t/s |  | 峰值6974MiB(余1038MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk turbo4 -ctv turbo3` |
=== [v26-9B-t4t2] start 20:03:49 ===
| v26-9B-t4t2 | gen 72.1 t/s (tg_3s峰值 73.06) | prefill 68 t/s |  | 峰值6847MiB(余1150MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk turbo4 -ctv turbo2` |
=== [v26-9B-f16] start 20:04:56 ===
| v26-9B-f16 | gen 39.0 t/s (tg_3s峰值 39.88) | prefill 42 t/s |  | 峰值7836MiB(余156MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk f16 -ctv f16` |
=== [v26-9B-t12] start 20:06:20 ===
| v26-9B-t12 | gen 72.1 t/s (tg_3s峰值 73.04) | prefill 55 t/s |  | 峰值7747MiB(余273MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 --threads 12` |
=== [v26-9B-ub1024] start 20:07:27 ===
| v26-9B-ub1024 | gen 56.3 t/s (tg_3s峰值 59.17) | prefill 61 t/s |  | 峰值7851MiB(余134MiB) MB | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ub 1024` |
=== [v26-9B-vision] start 20:08:39 ===
[v26-9B-vision] STARTUP FAILED
error: invalid argument: C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf
=== 9B Q4KM MATRIX DONE 20:16:34 ===
