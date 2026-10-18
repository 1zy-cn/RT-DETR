# Road defect detection research

Research code snapshot for the master thesis 基于深度学习的道路缺陷检测系统.

See [inspection report](experiment_reports/2026-10-14/CHECK_REPORT.md) for experiment metrics, limitations, and sync scope.

The completed 300-epoch SVRDD baseline is archived in the [2026-10-17 baseline report](experiment_reports/2026-10-17/BASELINE_REPORT.md), including CSV metrics, configurations, curves, TensorBoard logs, and checkpoint hashes. The 2026-10-14 baseline snapshot is incomplete and retained as historical evidence.

The completed, enabled SCSA experiment is archived in the [SCSA comparison report](experiment_reports/2026-10-17-scsa-enabled/SCSA_REPORT.md). Its checkpoints contain eight active SCSA modules. Compared with the baseline, validation mAP50 increased by 1.496 percentage points and mAP50–95 decreased by 0.804 points; this is a mixed result, not an across-the-board improvement. The SCSA configuration fixes and CPU checks are included on `main`; the baseline configurations remain available.

The completed HSFPN + enabled SCSA run is archived in the [2026-10-18 combination report](experiment_reports/2026-10-18-hsfpn-scsa/COMBO_REPORT.md). Its 300-epoch validation mAP50/mAP50–95 are 87.464%/58.987%. Both checkpoints contain eight active SCSA modules and the HSFPN neck. **This run used batch 16 while the three controls used batch 8, so it is not a controlled architecture-only ablation.** The archived server entrypoint preserves batch 16; the root `train.py` below uses batch 8 for a future matched rerun. No rerun was started during archiving.

The original upstream README and LICENSE are preserved. Datasets, checkpoints and live runs remain on the AutoDL server.

## 下一次对齐实验：ResNet18 + HSFPN + 有效 SCSA（batch=8）

`train.py` 默认使用 `ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml`，四个阶段共 8 个 SCSA。采用 YAML 随机初始化，不加载已有检查点，`resume=False`。本轮保持原 HSFPN 融合结构，训练设置为 300 轮、640 输入、batch 8、workers 4、GPU 0、seed 0、AdamW、lr0=0.0001、lrf=1、weight_decay=0.0001、AMP=False、close_mosaic=10，其余沿用项目默认设置。

已配置环境、仓库没有本地未提交修改的 AutoDL 服务器执行：

```bash
cd /root/autodl-fs/RT-DETR
git switch main
git pull --ff-only
python train.py --check
python train.py
```

如果旧目录有未提交修改，先保留这些修改，或 clone 到新的独立目录再执行上述 Python 命令；不要直接覆盖旧入口。本次采集服务器的 `train.py` 仍是手动修改的 batch=16 脚本，归档工作没有替服务器执行 pull。

如果导入时报缺少 `torchsummary`，补装：`python -m pip install torchsummary==1.5.1`。其他环境依赖仍需满足项目及 `checks/README.md` 的说明；`--check` 能检查导入和模型构建，不验证 GPU 或数据文件。

`SVRDD.yaml` 指向 `/autodl-fs/data/RT-DETR/datasets/SVRDD`，应包含 `train/images`、`valid/images` 和对应标签。训练结果写入共享盘：

```text
/root/autodl-fs/RT-DETR-HSFPN-SCSA-runs/rtdetr-resnet18-HSFPN-SCSA-enabled-s0
```

同名目录已存在时，框架会自动追加编号创建新目录，不覆盖旧实验。训练入口在构建模型及训练开始时检查 8 个有效 SCSA、5 个 HSFPN 通道注意力和 2 个转置卷积；训练开始后自动保存 `source_snapshot/`，包含本次入口、配置、实际模型结构、训练参数、Git 提交号及运行环境。实际使用的七类别模型在训练器读取数据 YAML 后构建，`--check` 只构建模板模型。

## 多设备共享同一份代码

仓库：https://github.com/1zy-cn/RT-DETR （公开）。克隆和拉取不需要 GitHub 登录；推送仍需有写权限的 GitHub 认证。更换服务器后，旧容器的 SSH 密钥和环境依赖不一定保留；没有原密钥时可将远程地址改为上述 HTTPS 地址用于拉取，并为需要推送的设备单独配置认证。

新电脑首次使用：

```bash
git clone https://github.com/1zy-cn/RT-DETR.git
cd RT-DETR
```

每次开始工作前：

```bash
git pull --ff-only
```

完成修改并检查提交内容后：

```bash
git status
git add <修改的文件>
git commit -m "说明本次修改"
git push
```

Git 不会自动上传未提交的修改。多台设备尽量避免同时修改同一文件；出现分叉时先合并，避免强制推送。数据集、模型权重和原始 runs 不在 Git 同步范围内；experiment_reports 保存按日期归档的结果快照。继承的 GitHub Actions 已禁用。
