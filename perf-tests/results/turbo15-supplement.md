## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26（2026-09-24 builder.exe 全量產物，含 TCQ×q8_0 混合對支持）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 4096`、temperature 0.6）

---

## turbo1.5 KV 补测（n_predict 4096 生产口径；9B 曾测过 q8_0×turbo1.5=59.2 不重复）

| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
| v26-9B-t15-t15 | gen 57.6 t/s (tg_3s峰值 58.29) | prefill 158 t/s |  | 峰值5403MiB(余2622

MiB) | `-ngl 99 -c 8192 -ctk turbo1.5 -ctv turbo1.5` |

| v26-9B-t15-q8 | gen 58.6 t/s (tg_3s峰值 58.99) | prefill 179 t/s |  | 峰值5443MiB(余2582

MiB) | `-ngl 99 -c 8192 -ctk turbo1.5 -ctv q8_0` |

| v26-27B-t15 | gen 30.4 t/s (tg_3s峰值 30.9) | prefill 36 t/s |  | 峰值7147MiB(余878

MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo1.5` |

| v26-27B-t15-t15 | gen 30.0 t/s (tg_3s峰值 30.63) | prefill 26 t/s |  | 峰值7144MiB(余881

MiB) | `-ngl 99 -c 4096 -fit off -ctk turbo1.5 -ctv turbo1.5` |

| v26-35B-mtp-t15 | gen 29.1 t/s (tg_3s峰值 38.27) | prefill 7 t/s | 0.67416/3.02 | 峰值5718MiB(余2307

MiB) | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo1.5 --spec-type draft-mtp` |
