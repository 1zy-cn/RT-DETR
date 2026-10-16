# Road defect detection research

Research code snapshot for the master thesis 基于深度学习的道路缺陷检测系统.

See [inspection report](experiment_reports/2026-10-14/CHECK_REPORT.md) for experiment metrics, limitations, and sync scope.

The original upstream README and LICENSE are preserved. Datasets, checkpoints and live runs remain on the AutoDL server.

## 多设备共享同一份代码

仓库：https://github.com/1zy-cn/RT-DETR （私有）。服务器已配置免输密码的仓库专用 SSH 认证；其他电脑使用自己的 GitHub 登录即可。不要把服务器私钥复制到其他电脑。

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
