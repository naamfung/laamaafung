## 測試環境

- GPU：NVIDIA GeForce RTX 3060 Ti 8192 MiB（驅動 616.92）
- CPU：Intel Xeon E5-2696 v3 @ 2.30GHz（`--threads 18`）
- 記憶體：31.8 GB；作業系統：Windows
- 引擎：laamaafung build-v26 / build-v25（2026-09-24 builder.exe 全量產物）
- 口徑：啟動判定只認日誌 `listening` 或進程退出；顯存同時抓峰值與最小餘量；生成速度為非流式單請求。**本檔表格為舊短輸出口徑（`n_predict 384`、temperature 0）**，tg_3s 峰值仍具參考性、整段均值偏低；新口徑見 matrix-27b-supplement.md（n_predict 4096、temp 0.6）

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

## 同参数 + 指定模板手工重测（2026-09-24，`--chat-template-file Qwen-Agentic-HONT.jinja`）

| n-cpu-moe | 显存峰值(余量) | gen t/s | tg_3s 峰值 | MTP 接受/均长 | 状态 |
|---|---|---|---|---|---|
| 36 | 峰值6019MiB(余2006MiB) | **26.20**（384 tokens / 14616 ms） | 28.54 | 0.52125/2.55 | HTTP 200；content 空（reasoning-only），重试 3 次后返回 |

- 模板生效确认：reasoning 内容呈 HONT 模板的繁体风格（「思考過程：1. **分析請求**…」），无模板时为英文 "Thinking Process"。
- content_len=0 / reasoning_len=574 / finish=length 与无模板重测完全一致 —— **空输出与模板无关**，是模型在该提示 + temperature 0 下把全部生成预算花在 `<think>` 的固有行为。
- 三次实测速度排序（同一参数，仅模板差异）：23.75 / 24.91 / 26.20 t/s —— 波动属正常区间。

## temperature 1 重测（2026-09-24，写作提示按用户准则应用 temp 1）

| n-cpu-moe | 显存峰值(余量) | gen t/s | MTP 接受/均长 | 状态 |
|---|---|---|---|---|
| 36 | 峰值6016MiB(余2009MiB) | **26.69**（384 tokens / 14348 ms） | 0.52703/2.58 | HTTP 200；content 仍空（reasoning-only） |

- **最终定性：空输出与温度无关** —— temp 0 / 0.6 口径外的 temp 1 下 reasoning 结构与空 content 完全复现。
  该 Agentic 模型对短提示的思考长度即超过 `n_predict 384`，正文来不及输出；生产使用把 `n_predict`
  放大（如 1024+）即可正常出正文。
- 温度准则（用户定）：编程/通用测试 0.6，写作等发散型创造 1.0；性能矩阵脚本已统一改为 0.6。
  此前 temp 0 为沿用脚本旧口径的错误选择。

## 生产口径重测（2026-09-24，n_predict 4096 / 16384，temp 1 + HONT 模板）

> **根因最终修正**：此前所有「空输出」的真因是 `n_predict 384` 预算不足 —— 模型思考即需
> ~1500–2200 token，384 连思考都截不完，content 恒空。与温度、模板均无关。
> 给足预算后一切正常，前两节的「模型固有行为」说法据此修正。

| n_predict | 实际生成 | gen t/s | MTP 接受/均长 | finish | content | 显存峰值(余量) |
|---|---|---|---|---|---|---|
| 4096 | 2078 tokens（含思考 2197 字） | **26.51** | 0.592/2.78 | stop（EOS） | 779 字正文 | 6101MiB(余1924MiB) |
| 16384 | 1696 tokens（含思考 1802 字） | **30.60** | 0.683/3.05 | stop（EOS） | 554 字正文 | 6207MiB(余1818MiB) |

- 正文正常输出（京杭大运河历史三段论），零空输出、零截断，EOS 自然结束。
- 长输出口径下 MTP 接受率显著回升（0.59–0.68 vs 384 口径的 0.52–0.56），思考结束进入正文后草稿命中率高。
- 速度 26.5–30.6 t/s 较 384 口径不降反升 —— 短输出均值被小批次尾巴拖累，tg_3s 稳态才是真实生产力。
- **测试准则（用户定）**：`n_predict` 生产底线 4096、上限 16384；脚本已全部改为 4096。
