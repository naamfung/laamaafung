
## 27B 三值 PTQ1_0 段（-c 4096 -fit off）
| 组合 | 生成速度 | Prefill | MTP接受/均长 | 显存 | 参数 |
|---|---|---|---|---|---|
=== [v26-27B-base] start 20:20:34 ===
| v26-27B-base | gen 30.5 t/s (tg_3s峰值 31.21) | prefill 17 t/s |  | 峰值6859MiB(余903MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0` |
=== [v26-27B-turbo4] start 20:21:02 ===
| v26-27B-turbo4 | gen 30.7 t/s (tg_3s峰值 31.13) | prefill 39 t/s |  | 峰值6826MiB(余1007MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4` |
=== [v26-27B-turbo3] start 20:21:28 ===
| v26-27B-turbo3 | gen 30.2 t/s (tg_3s峰值 30.65) | prefill 39 t/s |  | 峰值6818MiB(余1015MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo3` |
=== [v26-27B-turbo2] start 20:21:52 ===
| v26-27B-turbo2 | gen 29.1 t/s (tg_3s峰值 30.99) | prefill 39 t/s |  | 峰值6810MiB(余1023MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo2` |
=== [v26-27B-turbo3-2] start 20:22:17 ===
| v26-27B-turbo3-2 | gen 29.5 t/s (tg_3s峰值 30.71) | prefill 39 t/s |  | 峰值6770MiB(余1063MiB) MB | `-ngl 99 -c 4096 -fit off -ctk turbo3 -ctv turbo2` |
=== [v26-27B-kvarn4] start 20:22:43 ===
| v26-27B-kvarn4 | gen 29.1 t/s (tg_3s峰值 29.87) | prefill 14 t/s |  | 峰值6930MiB(余903MiB) MB | `-ngl 99 -c 4096 -fit off -ctk kvarn8 -ctv kvarn4` |
=== [v26-27B-kvarn3-2] start 20:23:10 ===
| v26-27B-kvarn3-2 | gen 29.1 t/s (tg_3s峰值 30.03) | prefill 53 t/s |  | 峰值6874MiB(余959MiB) MB | `-ngl 99 -c 4096 -fit off -ctk kvarn3 -ctv kvarn2` |
=== [v26-27B-kvmem] start 20:23:35 ===
| v26-27B-kvmem | gen 30.3 t/s (tg_3s峰值 30.49) | prefill 37 t/s |  | 峰值6446MiB(余1387MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v26-27B-kvmem-turbo4] start 20:24:00 ===
| v26-27B-kvmem-turbo4 | gen 30.2 t/s (tg_3s峰值 30.58) | prefill 39 t/s |  | 峰值6414MiB(余1419MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== [v25-27B-base] start 20:24:25 ===
| v25-27B-base | gen 31.1 t/s (tg_3s峰值 31.19) | prefill 44 t/s |  | 峰值6882MiB(余969MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0` |
=== [v25-27B-turbo4] start 20:24:50 ===
| v25-27B-turbo4 | gen 31.1 t/s (tg_3s峰值 31.23) | prefill 41 t/s |  | 峰值6846MiB(余1005MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv turbo4` |
=== [v25-27B-kvmem] start 20:25:14 ===
| v25-27B-kvmem | gen 30.2 t/s (tg_3s峰值 31.11) | prefill 56 t/s |  | 峰值6466MiB(余1379MiB) MB | `-ngl 99 -c 4096 -fit off -ctk q8_0 -ctv q8_0 --kvmem --kvmem-budget 4096 --kvmem-gen-reserve 1024` |
=== 27B SEGMENT DONE 20:25:38 ===
