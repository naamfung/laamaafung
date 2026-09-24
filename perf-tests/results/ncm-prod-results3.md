
## 生产口径扫描 [128K-plain]（ctx=131072, MTP + turbo4 + q8_0 K）
| n-cpu-moe | 显存峰值(余量) | gen t/s (tg_3s峰值) | 状态 |
|---|---|---|---|
| 36 | 峰值6432MiB(余1509MiB) | 25.6 (tg_3s峰值 29.6) |  |
| 34 | - | - | ❌ 放不下（failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort） |
[128K-plain] n-cpu-moe=34 到达极限，扫描结束

## 生产口径扫描 [256K-kvmem32K]（ctx=262144, MTP + turbo4 + q8_0 K）
| n-cpu-moe | 显存峰值(余量) | gen t/s (tg_3s峰值) | 状态 |
|---|---|---|---|
| 36 | 峰值5982MiB(余1991MiB) | 请求失败: Expecting value: line 1 column 1 (char 0) |  |
| 34 | - | - | ❌ 放不下（failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort） |
[256K-kvmem32K] n-cpu-moe=34 到达极限，扫描结束
=== PROD SCAN DONE 19:16:25 ===

## 异常核查闭环（256K-kvmem32K @ncm36 "请求失败"）
- 服务器实际生成正常：task 0 生成 384 tokens，eval 23.75 t/s；带模板复测 29.40 t/s、draft acceptance 0.521。
- 失败根因：content_len=0 / reasoning_len=574 / finish=length —— 模型在 temperature 0 下把全部 384 token
  预算花在 <think>（reasoning）中未出正文，服务器按防御逻辑 `empty output, retrying (1/3..3/3)`，
  三次同样结果；curl 侧收到空 content 响应导致 extract 解析失败。
- 判定：模型行为（FP 布局差异影响思考长度），非引擎缺陷。128K plain 同提示能出正文属同源随机性。
- 修正数据：| 36 | 峰值5982MiB(余1991MiB) | 23.75~29.4 t/s（服务端日志口径） | 正常（reasoning-only 输出） |
