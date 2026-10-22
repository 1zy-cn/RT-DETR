# SVRDD 官方划分、修正 IoU：基线与 HSFPN+SCSA 完成实验

这两组是同一正式协议下的 **单种子验证集** 结果。主指标采用训练保存最优权重所用的 fitness：`0.1 × mAP50 + 0.9 × mAP50:95`。表中 P、R 和两个 AP 均取各组 fitness 最佳的同一轮，单位为百分比。

| 结构 | 训练结束轮次 | fitness 最佳轮次 | P | R | mAP50 | mAP50:95 | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 RT-DETR 基线 | 232 | 182 | 74.990 | 68.350 | 73.697 | 45.816 | 20,095,404 |
| HSFPN+SCSA | 270 | 220 | 77.930 | 65.835 | 73.007 | 44.640 | 19,281,196 |
| 组合相对基线（百分点；参数量为绝对差） | — | — | +2.940 | −2.515 | −0.690 | −1.176 | −814,208 |

**结论：**在这次受控的 seed 0 对照中，HSFPN+SCSA 没有涨点。精确率提高但召回率下降，mAP50 与主指标 mAP50:95 均下降。组合参数量减少约 4.05%，但目前不能据此声称检测效果改善。不能用旧的交叉重叠数据划分下约 87%/59% 的结果与这里直接比较。

## 协议与核验

- 两组均读取 `datasets/SVRDD-official-split`，7 类；该划分按照官方 `train.txt`/`val.txt`/`test.txt` 的 6,000/1,000/1,000 张图像重建，见 [划分修复记录](../../docs/experiments/svrdd-official-split-repair.md)。本次没有把测试集指标混入验证集表格。
- 两组 `args.yaml` 除模型路径、运行名和保存路径外一致：640 输入、batch 8、seed 0、AdamW、初始学习率 0.0001、AMP 关闭、最大 300 轮、patience 50。`args.yaml` 中的 `pretrained: true` 不表示加载了权重；两份启动 provenance 均明确记录 `random_from_yaml; no checkpoint loaded`。启动快照中的实际 `nbs` 均为 8。
- `train.py`、数据 YAML、评估/损失/训练器和其他共用源码的 SHA-256 在两组快照中相同。启动时模型检查显示：基线 0 个 SCSA、0 个 HSFPN-CA；组合 8 个 SCSA、5 个 HSFPN-CA、2 个转置卷积；两组 decoder points 均为 `[4,4,4]`。两个来源 Git 提交号不同，但上述可比的训练代码文件内容相同。
- 两份 CSV 分别含第 1–232、1–270 轮，轮号连续、指标数值有限；结束轮与最佳轮差值恰为 patience 50，符合早停。`mAP50` 单项峰值分别是基线第 215 轮的 73.834%、组合第 184 轮的 73.301%；不可把这些不同轮次的峰值拼成“best.pt”同轮指标。checkpoint 在训练完成后被清理了 `epoch` 元数据（均为 `-1`），因此最佳轮依据 CSV 的 fitness 重算，而非从 checkpoint 元数据读取。
- 本归档的 56 个实验文件已与 [manifest.json](manifest.json) 中的大小和 SHA-256 对照。`best.pt`、`last.pt` 各约 39–40 MB，仍在服务器原始 `runs/` 路径；manifest 记录其 SHA-256，权重本体没有进入 Git。源快照、配置、完整 CSV、曲线和验证可视化均在本目录。

## 解释边界与下一步

这是同一协议下的一次结构筛选，不是多随机种子显著性结论，也不是独立测试集结果。当前 CSV 仅有总体指标，不能据此断言下降由哪一类道路缺陷、哪类尺度或 HSFPN/SCSA 中哪个模块造成。下一步先用服务器保留的两份 `best.pt` 在同一验证集导出逐类 AP、漏检与定位误差；再补单独 HSFPN、单独 SCSA 的官方划分实验，形成 2×2 消融。若要作为论文最终结论，需对预选候选补独立随机种子，并在锁定结构后仅一次使用测试集。

原始运行目录：

```text
runs/svrdd-r18-baseline-fixediou-officialsplit-b8-s0
runs/svrdd-r18-hsfpn-scsa-fixediou-officialsplit-b8-s0
```
