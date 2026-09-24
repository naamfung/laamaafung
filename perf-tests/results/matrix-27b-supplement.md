## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 4096`、temperature 0.6）

---

## 27B 三值补充段（PQ2_0 入组 + 256K KVMem；n_predict 4096 生产口径）

| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
| v26-27B-PQ2_0-kvmem4K | gen 35.4 t/s (tg_3s峰值 36.43) | prefill 21 t/s |  | 峰值7801MiB(余224MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-27B-PQ2_0-kvmem4K-turbo4 | gen 38.6 t/s (tg_3s峰值 39.25) | prefill 36 t/s |  | 峰值7833MiB(余192MiB) | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
| v26-27B-PQ2_0-kvmem-256K | gen 29.9 t/s (tg_3s峰值 32.45) | prefill 29 t/s |  | 峰值7883MiB(余142MiB) | `-ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048` |
| v26-27B-PTQ1_0-kvmem-256K | gen 30.3 t/s (tg_3s峰值 30.74) | prefill 28 t/s |  | 峰值6956MiB(余1069MiB) | `-ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 8192 --kvmem-gen-reserve 2048` |
| v26-27B-PTQ1_0-plain-256K | gen 9.2 t/s (tg_3s峰值 9.63) | prefill 2 t/s |  | 峰值7876MiB(余149MiB) | `-ngl 99 -c 262144 -fit off -ctk q8_0 -ctv turbo4` |
