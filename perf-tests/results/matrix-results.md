## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求。**本檔表格為舊短輸出口徑（`n_predict 384`、temperature 0）**，tg_3s 峰值仍具參考性、整段均值偏低；新口徑見 matrix-27b-supplement.md（n_predict 4096、temp 0.6）

---

| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
| v26-9B-base | gen 59.0 t/s (tg_3s峰值 59.93) | prefill 60 t/s |  | 5358 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0` |
| v26-9B-f16 | gen 59.4 t/s (tg_3s峰值 60.11) | prefill 63 t/s |  | 5478 MB | `-ngl 99 -c 8192 -ctk f16 -ctv f16` |
| v26-9B-turbo4 | gen 58.7 t/s (tg_3s峰值 59.11) | prefill 62 t/s |  | 5332 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
| v26-9B-turbo3 | gen 57.8 t/s (tg_3s峰值 58.76) | prefill 59 t/s |  | 5322 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo3` |
| v26-9B-turbo2 | gen 58.3 t/s (tg_3s峰值 59.07) | prefill 60 t/s |  | 5314 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo2` |

| v26-9B-turbo1.5 | gen 59.2 t/s (tg_3s峰值 59.93) | prefill 60 t/s |  | 5328 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5` |
| v26-9B-kvarn4 | gen 55.7 t/s (tg_3s峰值 57.36) | prefill 575 t/s |  | 5380 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4` |
| v26-9B-kvarn8-3 | gen 55.6 t/s (tg_3s峰值 56.95) | prefill 573 t/s |  | 5372 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn3` |
| v26-9B-kvarn4-3 | gen 55.5 t/s (tg_3s峰值 56.79) | prefill 536 t/s |  | 5340 MB | `-ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn3` |
| v26-9B-kvarn3-2 | gen 55.5 t/s (tg_3s峰值 56.91) | prefill 518 t/s |  | 5324 MB | `-ngl 99 -c 8192 -ctk kvarn3 -ctv kvarn2` |
| v26-9B-kvarn44 | gen 55.4 t/s (tg_3s峰值 56.87) | prefill 561 t/s |  | 5348 MB | `-ngl 99 -c 8192 -ctk kvarn4 -ctv kvarn4` |
| v26-9B-turbo4-3 | gen 57.6 t/s (tg_3s峰值 58.28) | prefill 58 t/s |  | 5280 MB | `-ngl 99 -c 8192 -ctk turbo4 -ctv turbo3` |
| v26-9B-turbo3-2 | gen 59.4 t/s (tg_3s峰值 60.04) | prefill 66 t/s |  | 5271 MB | `-ngl 99 -c 8192 -ctk turbo3 -ctv turbo2` |
| v26-9B-tcq3-tcq3 | gen 57.1 t/s (tg_3s峰值 57.27) | prefill 33 t/s |  | - | `-ngl 99 -c 8192 -ctk turbo3_tcq -ctv turbo3_tcq` |
| v26-9B-tcq3-tcq2 | gen 57.6 t/s (tg_3s峰值 59.0) | prefill 67 t/s |  | - | `-ngl 99 -c 8192 -ctk turbo3_tcq -ctv turbo2_tcq` |
| v26-9B-q8-tcq3 | gen 57.4 t/s (tg_3s峰值 59.26) | prefill 62 t/s |  | - | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo3_tcq` |
| v26-9B-tcq3-q8 | gen 56.6 t/s (tg_3s峰值 57.61) | prefill 67 t/s |  | - | `-ngl 99 -c 8192 -ctk turbo3_tcq -ctv q8_0` |
| v26-9B-kvmem | gen 58.1 t/s (tg_3s峰值 59.08) | prefill 60 t/s |  | 5195 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-9B-kvmem-turbo4 | gen 58.5 t/s (tg_3s峰值 59.05) | prefill 61 t/s |  | 5181 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-9B-kvmem-kvarn4 | gen 58.2 t/s (tg_3s峰值 60.86) | prefill 618 t/s |  | 5171 MB | `-ngl 99 -c 8192 -ctk kvarn8 -ctv kvarn4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-9B-vision | gen 58.1 t/s (tg_3s峰值 59.16) | prefill 62 t/s |  | 5353 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
| v25-9B-base | gen 59.7 t/s (tg_3s峰值 61.7) | prefill 64 t/s |  | 5398 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0` |
| v25-9B-f16 | gen 59.8 t/s (tg_3s峰值 61.84) | prefill 67 t/s |  | 5498 MB | `-ngl 99 -c 8192 -ctk f16 -ctv f16` |
| v25-9B-turbo4 | gen 59.0 t/s (tg_3s峰值 59.78) | prefill 61 t/s |  | 5358 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
| v25-9B-turbo3 | gen 58.4 t/s (tg_3s峰值 59.46) | prefill 61 t/s |  | 5345 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo3` |
| v25-9B-turbo2 | gen 59.0 t/s (tg_3s峰值 59.46) | prefill 66 t/s |  | 5347 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo2` |

| v25-9B-turbo1.5 | gen 59.8 t/s (tg_3s峰值 60.7) | prefill 62 t/s |  | 5334 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo1.5` |
| v25-9B-kvmem | gen 59.8 t/s (tg_3s峰值 60.39) | prefill 633 t/s |  | 5184 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v25-9B-kvmem-turbo4 | gen 59.5 t/s (tg_3s峰值 60.04) | prefill 628 t/s |  | 5171 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v25-9B-vision | gen 59.6 t/s (tg_3s峰值 60.09) | prefill 62 t/s |  | 5338 MB | `-ngl 99 -c 8192 -ctk q8_0 -ctv turbo4` |
| v26-35B-base | gen 24.8 t/s (tg_3s峰值 25.73) | prefill 25 t/s |  | 4175 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4` |
| v26-35B-mtp | gen 26.9 t/s (tg_3s峰值 30.7) | prefill 28 t/s | 0.50769/2.52 | 5566 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
| v26-35B-mtp-nmax2 | gen 29.6 t/s (tg_3s峰值 32.16) | prefill 25 t/s | 0.65559/2.31 | 5324 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2` |
| v26-35B-mtp-turbo3 | gen 31.4 t/s (tg_3s峰值 34.48) | prefill 28 t/s | 0.61538/2.84 | 5565 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo3 --spec-type draft-mtp` |
| v26-35B-mtp-turbo2 | gen 31.0 t/s (tg_3s峰值 34.76) | prefill 29 t/s | 0.58513/2.76 | 5558 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo2 --spec-type draft-mtp` |

| v26-35B-mtp-kvarn4 | gen 31.5 t/s (tg_3s峰值 34.89) | prefill 83 t/s | 0.57619/2.73 | 5606 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp` |
| v26-35B-mtp-kvarn4-nmax2 | gen 30.4 t/s (tg_3s峰值 33.76) | prefill 81 t/s | 0.68944/2.38 | 5402 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn4 --spec-type draft-mtp --spec-draft-n-max 2` |
| v26-35B-mtp-kvarn8-3 | gen 31.6 t/s (tg_3s峰值 34.33) | prefill 81 t/s | 0.59322/2.78 | 5630 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn8 -ctv kvarn3 --spec-type draft-mtp` |
| v26-35B-mtp-kvarn3-2 | gen 29.3 t/s (tg_3s峰值 33.85) | prefill 74 t/s | 0.69375/2.39 | 5328 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk kvarn3 -ctv kvarn2 --spec-type draft-mtp --spec-draft-n-max 2` |
| v26-35B-mtp-turbo3-2 | gen 30.3 t/s (tg_3s峰值 32.62) | prefill 25 t/s | 0.58795/2.76 | 5537 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk turbo3 -ctv turbo2 --spec-type draft-mtp` |
| v26-35B-mtp-kvmem | gen 25.6 t/s (tg_3s峰值 30.37) | prefill 26 t/s | 0.52125/2.55 | 4829 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-35B-mtp-kvmem-nmax2 | gen 26.4 t/s (tg_3s峰值 32.57) | prefill 25 t/s | 0.62647/2.25 | 4751 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |

| v26-35B-vision-mtp | gen 30.9 t/s (tg_3s峰值 31.95) | prefill 28 t/s | 0.58173/2.73 | 5568 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
| v25-35B-base | gen 23.8 t/s (tg_3s峰值 25.31) | prefill 27 t/s |  | 4677 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4` |
| v25-35B-mtp | gen 29.2 t/s (tg_3s峰值 32.8) | prefill 27 t/s | 0.54005/2.62 | 5593 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp` |
| v25-35B-mtp-nmax2 | gen 34.3 t/s (tg_3s峰值 36.31) | prefill 29 t/s | 0.74593/2.49 | 5341 MB | `-ngl 99 --n-cpu-moe 36 -c 8192 -ctk q8_0 -ctv turbo4 --spec-type draft-mtp --spec-draft-n-max 2` |
