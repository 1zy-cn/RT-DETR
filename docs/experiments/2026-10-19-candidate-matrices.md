# RT-DETR 道路缺陷候选改进实验

本记录把两个三因素候选方案拆成可复现的消融矩阵。模型均以仓库的 ResNet18 RT-DETR 为共同基线，SCSA 沿用已验证会实际执行的 `BasicBlock_SCSA`。优先比较正式 SVRDD 验证集的 mAP50-95，同时记录 mAP50、各类别 AP、参数量、FLOPs、显存和推理耗时。

## 方案①：SCSA + P3 FreqFusion + 尺度偏置解码器

P3 FreqFusion 沿用仓库现有的 P3 融合配置，替换原 P4→P3 融合步骤。尺度偏置解码器把三个特征层每个查询的采样点数由 `[4,4,4]` 调整为 `[6,4,2]`，总数仍为 12；对应 RT-DETRv2 提出的按尺度调整采样点思路，但这里的具体分配是本项目待验证的假设，并非已证实的增益。默认整数参数仍走原解码路径。

| 组别 | `--experiment` | SCSA | FreqFusion | 尺度偏置解码器 |
| --- | --- | ---: | ---: | ---: |
| A0 | `baseline` | 0 | 0 | 0 |
| A1 | `scsa` | 1 | 0 | 0 |
| A2 | `freqfusion` | 0 | 1 | 0 |
| A3 | `scale-decoder` | 0 | 0 | 1 |
| A4 | `scsa-freqfusion` | 1 | 1 | 0 |
| A5 | `scsa-scale-decoder` | 1 | 0 | 1 |
| A6 | `freqfusion-scale-decoder` | 0 | 1 | 1 |
| A7 | `scsa-freqfusion-scale-decoder` | 1 | 1 | 1 |

## 方案②：SCSA + 双层 DySample + C4 局部 FDConv

DySample 替换颈部 P5→P4 和 P4→P3 的两处上采样。FDConv 只替换 ResNet C4 阶段**第二个残差块**里的 `branch2b` 3×3 卷积，保留其余网络结构和 SCSA 设置。FDConv 来自 [官方实现](https://github.com/Linwei-Chen/FDConv)，保留了其 MIT 许可；为适配该仓库，仅去掉 MMCV 注册及研究脚本依赖，并修复验证阶段的半精度输出类型。

| 组别 | `--experiment` | SCSA | DySample | FDConv C4 |
| --- | --- | ---: | ---: | ---: |
| B0 | `baseline` | 0 | 0 | 0 |
| B1 | `scsa` | 1 | 0 | 0 |
| B2 | `dysample` | 0 | 1 | 0 |
| B3 | `fdconv-c4` | 0 | 0 | 1 |
| B4 | `scsa-dysample` | 1 | 1 | 0 |
| B5 | `scsa-fdconv-c4` | 1 | 0 | 1 |
| B6 | `dysample-fdconv-c4` | 0 | 1 | 1 |
| B7 | `scsa-dysample-fdconv-c4` | 1 | 1 | 1 |

两个矩阵共用 A0/B0、A1/B1，因此为 **14 个独立配置**。新增 YAML 由 `python checks/generate_candidate_configs.py` 生成，已有基线、SCSA、FreqFusion、DySample 配置保持原样。`python train.py --experiment <名称> --check` 会构建模型并核验模块数量、位置和解码采样点；它不执行训练。

`--check` 首次打印的检测头类别数可能仍是 YAML 的 `nc: 80`。正式训练会按数据配置重新构建为实际类别数，并在 `on_train_start` 再次核验结构；不可把首次打印的 80 类误认为真实训练类别数。

## 本地真实图片流程检查

仓库没有正式 SVRDD 数据集。本地仅用 [SVRDD_YOLO 数据](https://huggingface.co/datasets/ShuoZheLi/SVRDD_YOLO)中 12 张训练、4 张验证道路图片和真实目标框做两轮训练流程检查。下载脚本将来源、修订版本、许可和图片哈希写入 `datasets/svrdd-smoke-hf/manifest.json`。该样本的类别映射与正式 `SVRDD.yaml` 不完全相同，因此其 mAP 和训练损失**不能用于论文性能比较**。

```powershell
python checks/download_svrdd_smoke.py
python checks/run_candidate_smoke_matrix.py --resume
```

矩阵脚本逐个用 `batch=2, imgsz=256, epochs=2, workers=0, warmup=0` 训练和验证，检查两条 `results.csv` 记录、`last.pt` 和源码快照。日志与 `summary.json` 在 `runs/smoke-matrix-logs/`，权重在 `runs/smoke-matrix/`；这些大文件不纳入 Git。FDConv 官方模块中的 1×1 全局路径含 BatchNorm，训练批量需至少为 2。小样本检查只证明能完成真实前向、反向、优化、验证和保存。

2026-10-19 本地执行结果：**14/14 组通过，合计 28 条 epoch 记录**；每组两轮训练和验证损失均为有限值，生成了 `last.pt`。使用的是 RTX 3060 Laptop 6 GB GPU。两轮随机初始化的检测 AP 基本为零，属于预期的小样本现象，不代表任何改进点的有效性。

## 正式实验要求

在有完整 SVRDD 数据和 GPU 的机器上，先运行 `--check`，再去掉 `--check`。例如：

```bash
python train.py --experiment scsa-freqfusion-scale-decoder --check
python train.py --experiment scsa-freqfusion-scale-decoder
python train.py --experiment scsa-dysample-fdconv-c4 --check
python train.py --experiment scsa-dysample-fdconv-c4
```

默认训练协议仍为 seed 0、640 输入、batch 8、300 epoch、AdamW 及仓库既有学习率设置；模型从 YAML 随机初始化，不加载旧权重。各组必须使用同一正式数据划分、训练协议和评估版本。先筛选单项，再比较双项和三项；最终候选需要至少增加种子复验，并给出各类别及小目标表现。不要用这里两轮小样本的 AP 决定模型优劣。

尺度偏置思路参考：[RT-DETRv2 论文](https://arxiv.org/abs/2407.17140)。
