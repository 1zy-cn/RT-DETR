# SVRDD 第三章候选：CDRB 主干与 Conv3XC 融合层配置审计

日期：2026-10-23。当前阶段只完成代码配置、结构与合成数据验证；没有完成 SVRDD 正式训练，也没有任何 mAP 结论。

后续服务器审计已增加全量真实数据检查、真实图片短训练及协议 v2，最新启动说明和完整风险判断见 [服务器审计与启动指南](svrdd-server-readiness-2026-10-23.md)。本文保留模型配置阶段的验证记录。

术语必须分清：**CDRB = Composite Dilated Residual Block（复合膨胀残差块）**，是替换 BasicBlock 的完整模块；**DRB = Dilated Reparameterization Block（膨胀重参数化块）**，只是 CDRB 内部的一个组成部分。论文没有把完整 CDRB 命名为“膨胀重参数化残差块”。第三章建议写“引入复合膨胀残差块（CDRB）”，结构细节再解释其中的 DRB。三因素模型应称 SCSA + CDRB + Conv3XC。

## 文献到本项目的对应关系

- 于天河等，《基于改进 RT-DETR 的有遮挡交通标志检测算法》，《工程科学学报》2026，48(2):393–408，DOI:10.13374/j.issn2095-9389.2025.06.11.001。论文在 iRMB 的多头自注意力之前加入 DRB，以 CDRB 替换 ResNet-18 BasicBlock。本项目用已有的 `DilatedReparamBlock` 和 `iRMB` 构造 `BasicBlock_CDRB`，属于结构借鉴；论文未提供足够可核对的训练源码，因此不可宣称逐行复现。
- *Pavement-DETR: A High-Precision Real-Time Detection Transformer for Pavement Defect Detection*, *Sensors* 2025, 25(8), 2426。文中将 Conv3XC 用于特征融合。Conv3XC 的原始公开实现见 SPAN 仓库 `basicsr/archs/span_arch.py`。本项目采用同样的三个串联卷积和跳接结构，但仅替换 RT-DETR 的 P3 自顶向下融合块第 19 层的三个内部 RepConv；这是针对 SVRDD 的局部适配，不等于整篇 Pavement-DETR 的网络。

## 模型结构与配置

主干四个 stage 共 8 个 BasicBlock 替换为 `BasicBlock_CDRB`。每块保留 ResNet 外残差与尺寸；当 stride 或通道改变时先用 1×1 投影，然后经过训练期多分支 9×9 DRB，再经过包含扩张窗口多头注意力及 3×3 深度卷积的 iRMB。窗口为49个间隔采样点，不是连续局部7×7；iRMB 内残差关闭，仅保留 CDRB 外残差。iRMB 的通道扩展比为 2，具备先扩展再压缩的结构；这是项目适配参数。SCSA 选项置于主分支末端、外残差加法之前。DRB 可在评估模式合并为一个深度卷积。

颈部第 19 层保持 RepC3 的两条外部分支、输入输出通道和解码器连接，仅将该 RepC3 内三个 RepConv 改为三个 Conv3XC。每个 Conv3XC 为 1×1 → 3×3 → 1×1 路径加 1×1 跳接，通道扩展系数为 2。原 RepConv 在相加之后有 SiLU；严格评审时发现首版 Conv3XC 缺少这一非线性，现已在每个 Conv3XC 的输出端补回 SiLU。部署时线性卷积部分可显式融合为一个 3×3 卷积，SiLU 保留。其他 P4/P5 融合块不变。

## 严格结构评审结论

| 检查项 | 结论 |
| --- | --- |
| CDRB 核心语义 | 通过：多分支膨胀深度卷积、iRMB 的扩张窗口 MHSA/深度卷积、通道扩展/压缩以及外残差都确实执行。本项目将首版扩展比从1调整为2，不能据此断言原 iRMB 扩展比1本身存在错误。 |
| SCSA 组合语义 | 通过：8 个 CDRB 中各有一个 SCSA，位于主分支末端，实际前向执行 8 次；无 SCSA 组则为 0 次。它与 iRMB 的 MHSA 串联，可能冗余或增加显存，是否有收益仍需实验。 |
| Conv3XC 融合语义 | 通过，但限于 P3：第 19 层三个内部 RepConv 被 Conv3XC 替换；输入输出形状与 P3/P4/P5 路由保留。首版遗漏 RepConv 的 SiLU 已修正。 |
| 与参考论文的一致性 | 只满足拓扑层面的结构借鉴。当前 DRB 使用 UniRepLKNet 的 9×9 主分支及 5/5/3/3 分支，膨胀率为 1/2/3/4；参考论文的表 9 将 `(9,5,5,3,3)` 表述为膨胀率组合，未给出可核对源码，不能认定完全复现。Conv3XC 仅改 P3，也不是 Pavement-DETR 的完整融合网络。 |
| 部署重参数化 | 显式调用 `reparameterize_cdrb_conv3xc(model.eval())` 后可融合 DRB/Conv3XC 线性分支，连续特征层融合前后数值接近。随机未训练解码器的 top-k 候选可能因极小数值扰动改变顺序，所以不把逐行最终预测完全相等作为部署保证。 |

8 组消融：`svrdd-baseline`、`svrdd-scsa`、`svrdd-cdrb`、`svrdd-conv3xc`、`svrdd-scsa-cdrb`、`svrdd-scsa-conv3xc`、`svrdd-cdrb-conv3xc`、`svrdd-scsa-cdrb-conv3xc`。后三个因素可单独分析主效应与组合效应。新增 7 个 YAML 均从七类官方基线生成，保留 P3/P4/P5、300 queries、3 个解码器特征层及原有分类数。`train.py --experiment <组名> --check` 可核对模块数和路由，不启动训练。

## 已完成的验证

1. 单模块梯度与推理重参数化等价：CDRB 改变通道/分辨率后尺寸正确；DRB 在非初始 BatchNorm 统计量下融合前后数值接近，Conv3XC 在有/无 SiLU 两种情况下融合前后数值接近。
2. 8 个 YAML 均能构建，指定的 SCSA、CDRB、Conv3XC 数量分别为 0/8、0/8、0/3，P3/P4/P5 解码路由一致。
3. 三因素组合在 256×256 合成目标批次上可计算有限检测损失；DRB 与 Conv3XC 参数有非零梯度。前向钩子核对了 8 个 CDRB、8 个 SCSA、3 个 Conv3XC 都被执行，P3/P4/P5 形状正确。整网融合前后连续特征图数值接近。
4. `train.py --experiment svrdd-scsa-cdrb-conv3xc --check` 通过。当前本地官方划分 YAML 与清单匹配：训练 6000、验证 1000、测试 1000。相关完整回归测试 44 项通过。
5. 参数量：基线 20,095,404；CDRB 13,800,556；Conv3XC 21,619,884；三因素组合 15,355,756。CLI 估计 FLOPs：基线 58.3、CDRB 44.2、Conv3XC 77.8、组合 63.7 GFLOPs。组合比基线参数少约 23.6%，但估算计算量多约 9.3%；不能声称组合同时降参数与降计算。工具可能未完整计入注意力和其他算子，实际延时仍需实测。
6. 修正后的本机 RTX 3060 Laptop 6 GB 专用显存探测：640×640、batch=1、FP32 合成批次单次反传峰值 allocated 约 1.35 GiB；batch=8、AMP 合成批次单次反传虽完成，但峰值 allocated 约 7.49 GiB、reserved 约 9.19 GiB，超过专用显存，可能使用 Windows 共享显存。合成批次未包括完整优化器、数据加载和持续训练。

## 风险与后续判定

| 事项 | 现状与判定 |
| --- | --- |
| 本机正式训练显存 | `train.py` 正式协议为 FP32、batch=8、640。6 GB 本机不能由现有单步探测证明可持续训练；推荐在显存充足的服务器按同一协议做正式对照。若改用本机 AMP/小批量，应把整套 8 组统一改为相同协议，结果单列。 |
| 训练入口 | `SVRDD.yaml` 默认数据路径是 Linux 服务器路径。Windows 本地需指定 `--data C:\Users\86186\Documents\Codex\2026-10-20\new-chat\outputs\SVRDD-local.yaml` 及可写的 `--project`。现有 `train_local_svrdd.py` 固定为旧的 HSFPN+SCSA+DySample 实验，不能直接用于本候选。 |
| 速度/部署 | 训练图含多分支 DRB 和 Conv3XC；未调用融合方法时推理也会走多分支。使用 `reparameterize_cdrb_conv3xc(model.eval())` 可合并自定义分支，但 ONNX/TensorRT 导出和实机 FPS 尚未验证，论文中不能预先写“速度提升”。 |
| 权重初始化 | 新主干不能直接完整继承原 ResNet-18 的 BasicBlock 权重。当前与基线一样按 YAML 随机初始化；如改用预训练，必须记录层匹配比例并重新做公平对照。 |
| 学术归因 | CDRB 是论文启发的本项目实现，Conv3XC 仅作用于 P3 的局部融合。论文应准确写“借鉴/引入”，不可说与原论文网络完全一致。 |
| 检测收益 | 交通标志与 SVRDD 裂缝/坑槽分布不同；模块通过不代表涨点。先看单因素验证集 AP50、AP50:95、各类别 AP/Recall、误检、参数/FPS，再决定是否保留组合。 |
| 复现一致性 | 8 组均应固定官方清单、IoU 实现、输入尺寸、seed、优化器、训练轮次、精度及评估脚本。若改协议或重跑失败，不能把不同协议的结果直接放进同一消融表。 |

## 可复现入口

```powershell
D:\Users\86186\Miniconda3\envs\rtdetr\python.exe checks/generate_cdrb_conv3xc_configs.py
D:\Users\86186\Miniconda3\envs\rtdetr\python.exe train.py --experiment svrdd-cdrb --check
D:\Users\86186\Miniconda3\envs\rtdetr\python.exe train.py --experiment svrdd-conv3xc --check
D:\Users\86186\Miniconda3\envs\rtdetr\python.exe train.py --experiment svrdd-scsa-cdrb-conv3xc --check
```

正式训练前先确认数据路径、输出目录、GPU 和成组训练协议；`--check` 本身不检查数据或运行训练。

参考链接：

- https://cje.ustb.edu.cn/cn/article/pdf/preview/10.13374/j.issn2095-9389.2025.06.11.001.pdf
- https://www.mdpi.com/1424-8220/25/8/2426
- https://github.com/hongyuanyu/SPAN/blob/main/basicsr/archs/span_arch.py
