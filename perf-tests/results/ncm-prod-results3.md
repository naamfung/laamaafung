## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求（`n_predict 384`、temperature 0）

---

## 生产口径扫描 [128K-plain]（ctx=131072, MTP + turbo4 + q8_0 K）

| n-cpu-moe | 显存峰值(余量) | gen t/s (tg_3s峰值) | 状态 |
|---|---|---|---|
| 36 | 峰值6432MiB(余1509MiB) | 25.6 (tg_3s峰值 29.6) |  |
| 34 | - | - | ❌ 放不下（failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort） |

[128K-plain] n-cpu-moe=34 到达极限，扫描结束

## 生产口径扫描 [256K-kvmem32K]（ctx=262144, MTP + turbo4 + q8_0 K）

| n-cpu-moe | 显存峰值(余量) | gen t/s (tg_3s峰值) | 状态 |
|---|---|---|---|
| 36 | 峰值5982MiB(余1991MiB) | 请求失败: Expecting value: line 1 column 1 (char 0) |  |
| 34 | - | - | ❌ 放不下（failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort） |

[256K-kvmem32K] n-cpu-moe=34 到达极限，扫描结束

## 异常核查闭环（256K-kvmem32K @ncm36 "请求失败"）
- 服务器实际生成正常：task 0 生成 384 tokens，eval 23.75 t/s；带模板复测 29.40 t/s、draft acceptance 0.521。
- 失败根因：content_len=0 / reasoning_len=574 / finish=length —— 模型在 temperature 0 下把全部 384 token
  预算花在 <think>（reasoning）中未出正文，服务器按防御逻辑 `empty output, retrying (1/3..3/3)`，
  三次同样结果；curl 侧收到空 content 响应导致 extract 解析失败。
- 判定：模型行为（FP 布局差异影响思考长度），非引擎缺陷。128K plain 同提示能出正文属同源随机性。
- 修正数据：| 36 | 峰值5982MiB(余1991MiB) | 23.75~29.4 t/s（服务端日志口径） | 正常（reasoning-only 输出） |

## 同参数手工重测（2026-09-24，与扫描完全相同参数，无模板）

| n-cpu-moe | 显存峰值(余量) | gen t/s | MTP 接受/均长 | 状态 |
|---|---|---|---|---|
| 36 | 峰值约5982–6059MiB(余约1990MiB) | **24.91**（384 tokens / 15377 ms，tg_3s 口径 24.9） | 0.55581/2.66 | HTTP 200；content 空（reasoning-only），服务器自动重试 3 次后正常返回 |

- 重测确认：引擎与请求链路健康，24.91 t/s 落在首次扫描服务端日志口径（23.75）与带模板复测（29.40）之间。
- 显存采样受 nvidia-smi 千分位逗号输出干扰（部分样本被脚本丢弃），峰值取可信区间；与首次扫描 5982MiB 一致。
- 结论维持：该组数据可用于 256K 生产口径参考；content 为空是模型 reasoning-only 行为，非引擎缺陷。
