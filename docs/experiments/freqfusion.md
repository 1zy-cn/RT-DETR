# ResNet18：FreqFusion 与 SCSA＋FreqFusion 对照实验

> 历史候选方案。当前正式对照为 `svrdd-baseline`、`svrdd-hsfpn-scsa` 和 `svrdd-hsfpn-scsa-dysample`；参见仓库根目录 `RESEARCH_SYNC.md`。

两组都只在生成 P3 的位置使用一次 FreqFusion，保留原 RT-DETR 的 P5→P4 融合、自底向上路径和解码器。两份 YAML 的融合网络完全相同，主干差异仅为是否使用 8 个已启用的 SCSA。

| 命令选择 | 模型 YAML（位于 ultralytics/cfg/models/rt-detr） | SCSA | FreqFusion | 七类别参数量 |
|---|---|---:|---:|---:|
| `freqfusion` | `rtdetr-resnet18-FreqFusion-P3.yaml` | 0 | 1 | 20,200,094 |
| `scsa-freqfusion` | `rtdetr-resnet18-SCSA-FreqFusion-P3.yaml` | 8 | 1 | 20,230,814 |

## 训练入口

必须显式指定 `--experiment`；不带选择时只显示参数错误，不启动训练。先在已配置项目依赖的环境中检查：

```bash
python train.py --experiment freqfusion --check
python train.py --experiment scsa-freqfusion --check
```

正式训练二选一：

```bash
python train.py --experiment freqfusion
python train.py --experiment scsa-freqfusion
```

共同设置：从 YAML 随机初始化、不加载旧权重、resume=False、300 轮、imgsz=640、batch=8、workers=4、seed=0、AdamW、lr0=0.0001、lrf=1、weight_decay=0.0001、AMP=False、close_mosaic=10。本分支 `warmup_epochs=2000` 按迭代次数解释。其余使用同一份项目默认参数，初始 mosaic=1.0。两组不因模型名称或 Git 分支改变训练超参数。

当前默认数据配置为仓库根目录 `SVRDD.yaml`，其中数据目录指向共享盘 `/autodl-fs/data/RT-DETR/datasets/SVRDD-official-split`。临时目录中的代码也读取这份数据，不需要复制数据集。`--data /absolute/path/dataset.yaml` 可覆盖配置，但两组必须使用同一数据划分。旧重叠划分结果不能与新划分直接比较。

默认 GPU 为 0；`--device 1` 可选择另一个 GPU。仅有一张 GPU 时建议依次训练；Git 分支和独立目录不会隔离显存。`--project /absolute/shared/output` 可覆盖结果父目录。

默认输出为：

```text
/autodl-fs/data/RT-DETR/runs/rtdetr-resnet18-FreqFusion-P3-s0
/autodl-fs/data/RT-DETR/runs/rtdetr-resnet18-SCSA-FreqFusion-P3-s0
```

旧 FreqFusion 运行文件仍保留在仓库根目录的 `RT-DETR-FreqFusion-runs/`，不随 Git 同步。

`exist_ok=False`，已有同名目录时框架自动追加编号；不要将自动追加编号误认为从旧权重续训。旧组合可显式使用 `python train.py --experiment hsfpn-scsa`，原 HSFPN YAML 和历史结果保留。

## 主分支与 autodl-tmp 子分支分别运行

以下示例是在主检出目录跑 FreqFusion，临时检出目录跑 SCSA＋FreqFusion；两组选项可互换。先确保本次提交已经同步到 GitHub。主目录更新操作以工作区干净、该目录没有训练正在运行作为前提；如果有手改 `train.py` 等文件，先保留修改，不能强制覆盖。

```bash
cd /root/autodl-fs/RT-DETR
git status --short
git switch main
git pull --ff-only
git rev-parse HEAD
python train.py --experiment freqfusion --check
python train.py --experiment freqfusion
```

在另一个终端或服务器创建独立检出目录（路径应尚不存在）：

```bash
git clone --branch main https://github.com/1zy-cn/RT-DETR.git /root/autodl-tmp/RT-DETR-FF-SCSA
cd /root/autodl-tmp/RT-DETR-FF-SCSA
# 如果 main 已有后续变动，先 git switch --detach <第一组记录的完整提交号>。
git switch -c exp/scsa-freqfusion-p3
git rev-parse HEAD
python train.py --experiment scsa-freqfusion --check
python train.py --experiment scsa-freqfusion
```

`exp/scsa-freqfusion-p3` 是示例本地子分支名，不是必须预先存在的远程分支。两个检出目录应从相同提交开始；无需在主目录反复切换分支。运行期间保持各自代码目录不变。临时目录可能随容器释放消失，所以权重与日志默认仍写共享盘。

## 模型连接与实现设置

P4 融合输出为第 15 层，C3 经第 16 层 1×1 投影后成为高分辨率输入；第 17 层 `FreqFusionConcat([16,15])` 同时处理两者，按 `[上采样低分辨率特征, 增强高分辨率特征]` 顺序拼接。第 18 层 RepC3 产生 P3，最终解码器使用 `[18,21,24]`，分别对应 P3/P4/P5。

本实验固定：压缩通道 64、低通核 5×5、高通核 3×3、Hamming 窗、残差高频增强、4 组局部相似度引导偏移采样，**ALPF、AHPF、偏移重采样全部启用**。官方独立类的 `feature_resample` 默认 False，本实验明确选择 True；因此不是直接套用其默认参数。

实现参考 Chen 等人的 [Frequency-aware Feature Fusion for Dense Image Prediction, TPAMI 2024](https://arxiv.org/abs/2408.12879) 与 [官方代码固定版本](https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py)。原始文件 SHA-256 为 `5852efce406681ce6f3f7763ff8e034cab44f93cb0fdda0114130bc4c8c97e42`。核心方法归原作者，本仓库的改动是针对 RT-DETR 的接线与实现适配，不声称提出了 FreqFusion。采集时上游独立仓库没有明确 LICENSE 声明；这里保留来源说明，不为上游补造授权条款。

本地实现固定使用纯 PyTorch、reflection padding 的重组路径，与上游纯 PyTorch fallback 对照。它不会因为服务器安装了 MMCV 而自动切换算子。重组使用张量收缩避免显式复制上采样后的邻域张量；训练阶段使用非重入 activation checkpointing 减少中间激活保留。使用 PyTorch 2.0 或更新版本及原项目依赖，无新增 MMCV/CUDA 扩展安装要求。

不能把本实现与上游 MMCV CUDA 算子的结果或速度视为相同。纯 PyTorch 实现的 GPU 显存/耗时需要服务器实测；本地 CPU 检查不保证 batch=8 在任意显卡上都能运行。偏移分支使用 grid_sample，CUDA 反向可能有非确定性；固定 seed 和 deterministic 设置不保证不同硬件逐位复现。

## 防止实验选错与保存来源

入口在模型创建及训练器构建实际七类别模型后，检查 SCSA/FreqFusion 数量、SCSA 开关、完整频率分支、P3 输入和解码器连接，拒绝错选模型。每次训练开始自动写入 `source_snapshot/`：入口、选中的 YAML、实际数据 YAML、FreqFusion/SCSA/ResNet 实现、解析器和训练器，以及 JSON 中的实际模型结构、运行参数、Git 提交及状态、源码哈希和环境信息。

`--check` 只构建模板模型并检查结构、打印选择及参数，不读取训练集、不启动训练、不验证 GPU。实际类别数在训练器读取数据 YAML 后确定；不要将 `--check` 的 80 类模板参数量当作表中七类别参数量。

## 本地验证

```bash
python -m pytest checks/test_scsa.py checks/test_freqfusion.py checks/test_train_entrypoint.py -q -o addopts=
```

检查真实模型解析、640×640 输出、256×256 合成目标下真实 RT-DETR 检测损失反向传播和一次参数更新；确认低通、高通、偏移分支和组合模型中的 SCSA 有有限、非零梯度。另用逐邻域显式参考计算验证重组输出及输入/掩码梯度。

可选：将上述固定版本官方文件下载到仓库以外，再进行独立同权重数值对照：

```bash
python checks/compare_freqfusion_reference.py /path/to/upstream/FreqFusion.py
```

审计脚本先验证文件哈希，再将上游设为 `feature_resample=True`、纯 PyTorch fallback，比较三个输出和输入/参数梯度。本次 double 精度 CPU 对照最大输出绝对误差约 1.32e-15，22 个梯度张量最大绝对误差约 2.22e-16。核验记录见 `docs/experiments/freqfusion-validation.json`。这证明对应设置的数值实现一致，不构成真实数据精度提升的证据。
