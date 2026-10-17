# ResNet18 基线完成结果（2026-10-17）

采集时间：2026-10-17T10:21:52+08:00。原始运行目录：`/autodl-fs/data/RT-DETR/runs/rtdetr-resnet18`。
源代码分支：`main`，提交：`4a09acf1720b649abafcdfd26f7c9bb19e6eef90`；服务器 `train.py` 有未提交的续训配置，已单独归档到本目录，未修改仓库训练入口。

## 完成情况与验证指标

- CSV 连续记录第 1–300 轮，无缺轮、重复轮或非有限数值，达到原定 300 轮目标。
- 本次在 CPU 上读取 `best.pt` 和 `last.pt`，两者均为 ResNet18 基线，实际 SCSA 模块数为 0。
- 未重新运行训练、推理或验证；下表是原 `results.csv` 的验证集记录，不是独立测试集成绩。

| 记录 | Epoch | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| 最后一轮 | 300 | 86.433% | 81.595% | 87.778% | 61.977% |
| CSV 中 mAP50–95 最高轮 | 300 | 86.433% | 81.595% | 87.778% | 61.977% |

`best.pt` 的保存指标另外原样记录在 `summary.json` 中；不把 CSV 选行当作独立重测结果。

## 实验配置与续训记录

- 数据集：SVRDD，7 类；输入 640，batch=8，AdamW，初始学习率 0.0001，seed=0，AMP 关闭。
- 这次基线从完成第 135 轮的 `last.pt` 恢复，继续第 136–300 轮；训练器恢复优化器和 EMA。
- 保存了当前 `args.yaml`、基线模型 YAML、数据集 YAML、续训入口，以及续训前入口备份。原始参数不作改写。
- `args.yaml` 中 `model` 指向续训检查点；`resume: false` 字段不能单独用于否认续训，已保留实际入口与分段 TensorBoard 日志供核查。
- 完成后的权重已由项目结束流程剥离训练状态（具体 epoch/optimizer 字段见 `summary.json`），不能把这些最终权重当作保留优化器的中断检查点。

## 归档内容与同步边界

- `rtdetr-resnet18/`：CSV、原始配置、结果曲线、训练入口快照及两个阶段的 TensorBoard 事件文件。
- `summary.json`：汇总、文件 SHA256、两个权重的服务器路径与 SHA256、权重内保存的指标。
- 模型权重和数据集继续保存在服务器共享存储，本次 Git 提交不包含权重二进制或数据集。
- 旧的 `2026-10-14` 结果快照保留，不能再将其中未完成的基线记录当作最终成绩。
- 正在运行的启用 SCSA 实验属于另一条实验记录，本归档不包含它的数据或代码修改。

查看训练曲线，可在已安装 TensorBoard 的环境中运行：

```bash
tensorboard --logdir experiment_reports/2026-10-17/rtdetr-resnet18/tensorboard
```
