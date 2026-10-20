# SVRDD SCSA+HSFPN 后续消融（2026-10-20）

当前 `hsfpn-scsa` 已在 AutoDL 训练。以下四组固定 ResNet18、SVRDD 7 类、数据划分、随机初始化、seed 0、640 输入、batch 8、300 epoch、AdamW 与其余 `train.py` 参数；仅替换所列网络结构。验证集 `mAP50-95` 是主判定指标，另记录 mAP50、逐类 AP、参数、计算量、峰值显存和推理延迟。测试集不参与模型选择。

| `train.py --experiment` | HSFPN 上采样 | P2 细节支路 | 解码器输入 |
| --- | --- | --- | --- |
| `hsfpn-scsa` | 两处转置卷积 | 无 | P3/P4/P5 |
| `hsfpn-scsa-dysample` | 两处 DySample | 无 | P3/P4/P5 |
| `hsfpn-scsa-p2` | 两处转置卷积；P3→P2 最近邻上采样 | 有，P2 回流 P3 | P3/P4/P5 |
| `hsfpn-scsa-dysample-p2` | 三处 DySample | 有，P2 回流 P3 | P3/P4/P5 |

P2 支路不是新增第四个 decoder 尺度：它融合 backbone P2 与 neck P3，再下采样回注 P3，因此 decoder 仍是三尺度。`P2` 单独组用无参数的最近邻上采样作为新支路的基准；联合组在同一位置使用 DySample。这样可分别比较 DySample、P2 支路，以及两者是否有交互。它是借鉴与组合已有结构的工程研究，不宣称原创 DySample 或 P2 思想。

每次训练前 `train.py --check` 检查模块数量和连线；`on_train_start` 再检查实际训练模型并保存 YAML、代码及训练参数快照。运行目录分别位于同一个项目的 `runs/` 下。服务器上先完成正在跑的对照组，再运行 `bash scripts/run_svrdd_hsfpn_ablation_queue.sh <当前训练主 PID>`；队列要求上一组完成 300 轮且存在 `last.pt`，否则停止，不静默续训或覆盖已有结果。

本轮只有单种子筛选结果。若联合组提高主指标，应追加 seed 1/2，对照组与联合组做成对复核；若增益很小，应优先查看逐类 AP、目标尺寸和置信区间，不能仅凭一轮宣称稳定改善。早期基线、SCSA 单项、HSFPN 单项若使用不同配置或训练协议，不直接填入同一张严格消融表。
