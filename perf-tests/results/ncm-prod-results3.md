## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 4096`、temperature 0.6 / 寫作 1.0）

---

## 生產口徑 n-cpu-moe 極限掃描（35B + MTP，`-ctk q8_0 -ctv turbo4`）

### [128K-plain]（ctx=131072，無 KVMem）

| n-cpu-moe | 顯存峰值(餘量) | gen t/s (tg_3s 峰值) | 狀態 |
|---|---|---|---|
| 36 | 6432MiB(餘1509MiB) | 25.6 (29.6) | **plain 極限** |
| 34 | - | - | ❌ 放不下（failed to fit: n_gpu_layers already set by user to 99, abort） |

### [256K-kvmem32K]（ctx=262144，`--kvmem --kvmem-budget 32768 --kvmem-gen-reserve 16384`）

| n-cpu-moe | n_predict | 顯存峰值(餘量) | gen t/s | MTP 接受/均長 | 狀態 |
|---|---|---|---|---|---|
| 36 | 4096 | 6101MiB(餘1924MiB) | **26.51** | 0.592/2.78 | 正常（content 779 字，finish=stop） |
| 36 | 16384 | 6207MiB(餘1818MiB) | **30.60** | 0.683/3.05 | 正常（content 554 字，finish=stop） |
| 34 | - | - | - | - | ❌ 放不下 |

結論：**256K 必須配 KVMem**；ncm36 是 8GB 卡上兩種檔位的唯一可行層數，本組即 35B 生產 256K 配置的最終參考數據。

---

## 排障結論（256K 首測「請求失敗」的根因鏈，最終定性）

首測（`n_predict 384`、temp 0）content 為空觸發服務器 `empty output` 防禦重試，提取工具解析失敗。
經同參數多輪重測逐層排除：temp 0 無模板 24.91 → temp 0 帶 HONT 模板 26.20（模板確認生效）→
temp 1 26.69 t/s，均仍空輸出；最終以生產預算（4096/16384）復測，content 正常、EOS 自然結束。

**根因**：模型思考即需 ~1500–2200 token，`n_predict 384` 連思考都截不完 —— 與溫度、模板均無關。

**測試準則（用戶定，隨之修訂）**：

- `n_predict` 生產底線 **4096**、上限 16384，禁用短輸出預算；
- temperature 測試 **0.6**、寫作 **1.0**，禁用 temp 0；
- 長輸出口徑下 MTP 接受率顯著回升（0.59–0.68 vs 短輸出 0.52–0.56），tg_3s 穩態才是真實生產力。
