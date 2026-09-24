# perf-tests

2026-09-24 大型性能矩陣的腳本與原始結果。環境：**NVIDIA GeForce RTX 3060 Ti 8GB**（驅動 616.92）、Intel Xeon E5-2696 v3（`--threads 18`）、31.8 GB 記憶體、Windows；各結果檔頭部均有完整環境標註。

## scripts/

| 腳本 | 用途 |
|---|---|
| `matrix-test.sh` | 9B/35B 主矩陣（MTP × KV 量化 × KVMem × 視覺） |
| `matrix-35b-coder.sh` | 35B CODER 128K 生產口徑 16 組（ncm 下探、nmax 掃描、ub 交叉） |
| `matrix-9b-prod.sh` | 9B Q4_K_M config 復刻 10 組 |
| `matrix-27b.sh` | 27B 三值 PTQ1_0 12 組（v25/v26 對照，含 CUDA 三值修復驗證） |
| `matrix-9b-iq4xs.sh` | 9B IQ4_XS CLI 錨點復現 |
| `ncm-scan.sh` | 35B `--n-cpu-moe` × KVMem 池寬交叉（ctx 8192） |
| `ncm-scan-prod.sh` | 生產口徑極限掃描（128K plain / 256K KVMem） |
| `vram-check.sh` | 顯存峰值/餘量雙值抓取輔助 |
| `mtx-extract.py` / `ncm-extract.py` | 響應 JSON + 服務端日誌 → 結果表行 |

腳本啟動判定只認日誌 `listening` / 進程退出；顯存同時抓峰值與最小餘量。跑之前 `unset` 代理變量、`curl --noproxy '*'`。
**請求口徑（用戶定）**：`n_predict` 生產底線 **4096**（上限可到 16384），temperature 測試 0.6 / 寫作 1.0，禁用 temp 0 與短輸出預算（384 之類不具生產參考價值）。`results/` 中標註 `n_predict 384` 的舊數據為短輸出口徑，tg_3s 峰值仍具參考性，整段均值偏低。

## results/

與腳本一一對應。`ncm-prod-results3.md` 為生產口徑極限掃描的最終結論（128K plain 極限 ncm36、
256K 必須配 KVMem），並附 256K 首測「空輸出」排障的最終定性：`n_predict 384` 連模型思考
（~1500–2200 token）都截不完，與溫度、模板無關 —— 由此確立請求口徑準則（4K/16K、temp 0.6/1.0）。
已移除各檔中的無效啟動失敗段（`turbo*_tcq` KV 不被 FA vec 支持，見主 README 已知限制）。
