## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 384`、temperature 0）

---

## 9B Q4_K_M 生产基准矩阵（config 复刻，72.38 锚点）

| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
| v26-9B-replica | gen 73.5 t/s (tg_3s峰值 74.36) | prefill 70 t/s |  | 峰值7689MiB(余324MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072` |
| v25-9B-replica | gen 75.8 t/s (tg_3s峰值 76.19) | prefill 66 t/s |  | 峰值7693MiB(余328MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072` |
| v26-9B-q8q8 | gen 65.3 t/s (tg_3s峰值 66.32) | prefill 66 t/s |  | 峰值7849MiB(余164MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctv q8_0` |
| v26-9B-t4t3 | gen 72.2 t/s (tg_3s峰值 72.93) | prefill 62 t/s |  | 峰值6974MiB(余1038MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk turbo4 -ctv turbo3` |
| v26-9B-t4t2 | gen 72.1 t/s (tg_3s峰值 73.06) | prefill 68 t/s |  | 峰值6847MiB(余1150MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk turbo4 -ctv turbo2` |
| v26-9B-f16 | gen 39.0 t/s (tg_3s峰值 39.88) | prefill 42 t/s |  | 峰值7836MiB(余156MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ctk f16 -ctv f16` |
| v26-9B-t12 | gen 72.1 t/s (tg_3s峰值 73.04) | prefill 55 t/s |  | 峰值7747MiB(余273MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 --threads 12` |
| v26-9B-ub1024 | gen 56.3 t/s (tg_3s峰值 59.17) | prefill 61 t/s |  | 峰值7851MiB(余134MiB) | `-ngl all -ngld all --n-cpu-moe 0 --threads 18 --threads-http 2 --parallel 1 --kv-unified -ctk q8_0 -ctv turbo4 -b 16384 -ub 256 --ctx-checkpoints 42 --load-mode mlock-ram --cache-prompt --cache-ram 8192 --fit on -c 131072 -ub 1024` |

[v26-9B-vision] STARTUP FAILED
error: invalid argument: C:/WorkModels/Qwen3.5-9B/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf
