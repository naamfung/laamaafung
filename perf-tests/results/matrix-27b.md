## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 384`、temperature 0）

---

## 27B 三值 PTQ1_0 段（-c 4096 -fit off）

| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
| v26-27B-base | gen 30.5 t/s (tg_3s峰值 31.21) | prefill 17 t/s |  | 峰值6859MiB(余903MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0` |
| v26-27B-turbo4 | gen 30.7 t/s (tg_3s峰值 31.13) | prefill 39 t/s |  | 峰值6826MiB(余1007MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4` |
| v26-27B-turbo3 | gen 30.2 t/s (tg_3s峰值 30.65) | prefill 39 t/s |  | 峰值6818MiB(余1015MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo3` |
| v26-27B-turbo2 | gen 29.1 t/s (tg_3s峰值 30.99) | prefill 39 t/s |  | 峰值6810MiB(余1023MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo2` |
| v26-27B-turbo3-2 | gen 29.5 t/s (tg_3s峰值 30.71) | prefill 39 t/s |  | 峰值6770MiB(余1063MiB) | `-ngl 99 -c 4096 -fit off -ctk turbo3 -ctv turbo2` |
| v26-27B-kvarn4 | gen 29.1 t/s (tg_3s峰值 29.87) | prefill 14 t/s |  | 峰值6930MiB(余903MiB) | `-ngl 99 -c 4096 -fit off -ctk kvarn8 -ctv kvarn4` |
| v26-27B-kvarn3-2 | gen 29.1 t/s (tg_3s峰值 30.03) | prefill 53 t/s |  | 峰值6874MiB(余959MiB) | `-ngl 99 -c 4096 -fit off -ctk kvarn3 -ctv kvarn2` |
| v26-27B-kvmem | gen 30.3 t/s (tg_3s峰值 30.49) | prefill 37 t/s |  | 峰值6446MiB(余1387MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-27B-kvmem-turbo4 | gen 30.2 t/s (tg_3s峰值 30.58) | prefill 39 t/s |  | 峰值6414MiB(余1419MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v25-27B-base | gen 31.1 t/s (tg_3s峰值 31.19) | prefill 44 t/s |  | 峰值6882MiB(余969MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0` |
| v25-27B-turbo4 | gen 31.1 t/s (tg_3s峰值 31.23) | prefill 41 t/s |  | 峰值6846MiB(余1005MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4` |
| v25-27B-kvmem | gen 30.2 t/s (tg_3s峰值 31.11) | prefill 56 t/s |  | 峰值6466MiB(余1379MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
