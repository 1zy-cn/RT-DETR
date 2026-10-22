# Road defect detection research

Research code snapshot for the master thesis 基于深度学习的道路缺陷检测系统.

See [inspection report](experiment_reports/2026-10-14/CHECK_REPORT.md) for experiment metrics, limitations, and sync scope.

The completed 300-epoch SVRDD baseline is archived in the [2026-10-17 baseline report](experiment_reports/2026-10-17/BASELINE_REPORT.md), including CSV metrics, configurations, curves, TensorBoard logs, and checkpoint hashes. The 2026-10-14 baseline snapshot is incomplete and retained as historical evidence.

The completed, enabled SCSA experiment is archived in the [SCSA comparison report](experiment_reports/2026-10-17-scsa-enabled/SCSA_REPORT.md). Its checkpoints contain eight active SCSA modules. Compared with the baseline, validation mAP50 increased by 1.496 percentage points and mAP50–95 decreased by 0.804 points; this is a mixed result, not an across-the-board improvement. The SCSA configuration fixes and CPU checks are included on `main`; the baseline configurations remain available.

The completed HSFPN + enabled SCSA run is archived in the [2026-10-18 combination report](experiment_reports/2026-10-18-hsfpn-scsa/COMBO_REPORT.md). Its 300-epoch validation mAP50/mAP50–95 are 87.464%/58.987%. Both checkpoints contain eight active SCSA modules and the HSFPN neck. **This run used batch 16 while the three controls used batch 8, so it is not a controlled architecture-only ablation.** The archived server entrypoint preserves batch 16; the root `train.py` retains an explicit `hsfpn-scsa` batch-8 option for a future matched rerun. No rerun was started during archiving.

The original upstream README and LICENSE are preserved. Datasets, checkpoints and live runs remain on the AutoDL server.

后续选题与实验量规划见 [论文候选改进点备忘录](docs/experiments/2026-10-19-thesis-candidate-roadmap.md)。其中的候选方案均待验证，不能当作已完成的实验结论。

## 当前正式实验：SVRDD 官方划分与修正 IoU

基线与 HSFPN+SCSA 已结束并归档在 [2026-10-22 同协议结果报告](experiment_reports/2026-10-22-officialsplit-fixediou/REPORT.md)：seed 0 验证集 mAP50:95 分别为 45.816% 与 44.640%，组合没有涨点。权重仍在服务器，Git 保存完整 CSV、配置、源码快照、图表及权重哈希。

`SVRDD.yaml` 指向去重后的 `datasets/SVRDD-official-split`。正式对照只使用以下三组，均为 batch 8、640 输入、300 轮、seed 0，并从 YAML 随机初始化；`train.py` 会核对官方划分、IoU 实现和模型模块。

```bash
python train.py --experiment svrdd-baseline
python train.py --experiment svrdd-hsfpn-scsa
python train.py --experiment svrdd-hsfpn-scsa-dysample
```

本轮基线结果目录为 `runs/svrdd-r18-baseline-fixediou-officialsplit-b8-s0/`。旧的重叠划分结果仅作探索记录，不能与此轮正式结果直接比较。本轮的 `runs/`、数据集、权重及所有历史候选结果均保留在服务器，不纳入 Git。

## 历史候选实验：FreqFusion 与 SCSA＋FreqFusion

两份配置均只改生成 P3 的融合位置，保留原 RT-DETR 双向融合路径；SCSA 组合含 8 个有效注意力模块。统一使用 batch=8、300轮、640输入、seed=0，从 YAML 随机初始化。

```bash
python train.py --experiment freqfusion --check
python train.py --experiment scsa-freqfusion --check
# 如需复现历史候选实验，移除 --check：
python train.py --experiment freqfusion
python train.py --experiment scsa-freqfusion
```

历史结果保留在仓库根目录的 `RT-DETR-FreqFusion-runs/`；当前 `train.py` 默认将新运行写入 `runs/`。完整模型配置、主分支与 `autodl-tmp` 独立子分支运行步骤、源码来源和验证范围见 [FreqFusion 实验指南](docs/experiments/freqfusion.md)。

入口必须显式指定实验；旧 HSFPN＋SCSA 复验可用 `python train.py --experiment hsfpn-scsa`。训练前后会检查模块启用状态，并在结果目录自动保存 `source_snapshot/`。如果旧服务器有手动修改的入口，更新仓库前先保留修改；本地准备和同步不会替你在服务器启动训练或改写旧结果。

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
