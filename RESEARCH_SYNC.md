# Road defect detection research

Research code snapshot for the master thesis 基于深度学习的道路缺陷检测系统.

See [inspection report](experiment_reports/2026-10-14/CHECK_REPORT.md) for experiment metrics, limitations, and sync scope.

The completed 300-epoch SVRDD baseline is archived in the [2026-10-17 baseline report](experiment_reports/2026-10-17/BASELINE_REPORT.md), including CSV metrics, configurations, curves, TensorBoard logs, and checkpoint hashes. The 2026-10-14 baseline snapshot is incomplete and retained as historical evidence.

The completed, enabled SCSA experiment is archived in the [SCSA comparison report](experiment_reports/2026-10-17-scsa-enabled/SCSA_REPORT.md). Its checkpoints contain eight active SCSA modules. Compared with the baseline, validation mAP50 increased by 1.496 percentage points and mAP50–95 decreased by 0.804 points; this is a mixed result, not an across-the-board improvement. The SCSA configuration fixes and CPU checks are included on `main`; the baseline configurations remain available.

The original upstream README and LICENSE are preserved. Datasets, checkpoints and live runs remain on the AutoDL server.

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
