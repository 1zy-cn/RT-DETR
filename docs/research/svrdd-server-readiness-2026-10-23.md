# SVRDD 第三章：服务器实验前审计与启动指南

审计日期：2026-10-23。对象：RT-DETR-R18 + SCSA + CDRB + Conv3XC，七类 SVRDD。

## 1. 结论与证据边界

候选代码具备开展对照实验的条件；**尚无本组合在 SVRDD 上涨点的实验证据**。本轮完成代码/配置审查、独立只读评审、全量本地数据完整性核验、数值回归及真实图片短流程训练。没有登录服务器，没有验证服务器 640/batch8 的持续显存，也没有启动正式长训练。

建议先筛选单因素，再决定组合。三个模块分别对应特征选择、主干表示、P3 融合参数化，逻辑可以成立，但其效果不能相加。CDRB 是风险最高的因素；SCSA 也仍需在修正后的划分和 IoU 下独立验证。论文应称“引入/组合已有改进方法”，不宣称原创模块或完整复现参考论文。

## 2. 八组配置与实际执行语义

| 训练入口 `--experiment` | SCSA 数 | CDRB 数 | Conv3XC 数 | 训练图参数量 |
| --- | ---: | ---: | ---: | ---: |
| svrdd-baseline | 0 | 0 | 0 | 20,095,404 |
| svrdd-scsa | 8 | 0 | 0 | 20,126,124 |
| svrdd-cdrb | 0 | 8 | 0 | 13,800,556 |
| svrdd-conv3xc | 0 | 0 | 3 | 21,619,884 |
| svrdd-scsa-cdrb | 8 | 8 | 0 | 13,831,276 |
| svrdd-scsa-conv3xc | 8 | 0 | 3 | 21,650,604 |
| svrdd-cdrb-conv3xc | 0 | 8 | 3 | 15,325,036 |
| svrdd-scsa-cdrb-conv3xc | 8 | 8 | 3 | 15,355,756 |

已有七类基线加七个新 YAML 构成完整 2³ 因子设计。独立评审核对：排除预期的 block 类型、SCSA 开关和 P3 内部模块替换后，七份配置与基线相同。全部保留七类输出、P3/P4/P5（第 19/22/25 层）、300 queries、256 隐藏通道、3 层 decoder、每尺度 4 个采样点；没有混入 HSFPN、DySample、额外 P2 或新的损失函数。

- **SCSA**：每个主干块主分支末端、与 shortcut 相加之前执行；无 SCSA 组不实例化它。组合组位置相同。
- **CDRB = Composite Dilated Residual Block，复合膨胀残差块**：四个 stage 全部 8 个 BasicBlock 被替换。通道/步长转换 → DRB → iRMB → 可选 SCSA → 外残差 → ReLU。
- **DRB = Dilated Reparameterization Block，膨胀重参数化块**：只是 CDRB 的组成部分。主分支为 9×9 depthwise；辅助核 5/5/3/3、膨胀率 1/2/3/4，训练分支分别有 BN，部署可融合。
- **iRMB**：扩展比 2、dim_head=8、3×3 depthwise。其 7×7 是每组 49 个间隔采样点的**扩张窗口**，不是连续局部窗口；与官方 EMO 分组一致。`has_skip=False` 同时关闭 iRMB 局部与末端内残差，仅保留 CDRB 外残差。这是本项目适配，不是原 EMO 原样复现。
- **Conv3XC**：只替换第 19 层 P3 RepC3 内的三个 RepConv；1×1→3×3→1×1，通道扩展系数 2，并联 1×1 跳接，相加后保留 SiLU。其他融合层及外部分支不变。线性卷积链之间没有激活，才可合并为单个 3×3；不能把其收益解释成扩大空间感受野。

组合的估算训练 FLOPs 为 63.7G，基线 58.3G；参数减少约 23.6%，估算计算量增加约 9.3%。单 Conv3XC 估算 77.8G。THOP 可能遗漏注意力算子，不是实测耗时。框架普通融合还会合并部分 Conv/BN/RepConv，因此加载验证日志的参数量可能低于训练图；自定义 DRB/Conv3XC 的部署融合需要显式调用，不能混用部署/训练图统计。

## 3. 数据核查

核查本地实际 `SVRDD-official-split`，不是仅阅读历史报告。结果文件：`docs/research/svrdd-data-preflight-2026-10-23.json`。

- 官方清单精确匹配：训练 6000、验证 1000、测试 1000；文件名互斥，图像/标注一一对应。
- 全部图像实际解码成功；没有发现字节完全相同的同划分或跨划分图片。**这不能排除相邻街景、同一路段的视觉近重复**；官方划分也不等于跨城市泛化测试。
- 七类顺序固定为：0 纵向裂缝，1 横向裂缝，2 网状裂缝，3 坑槽，4 井盖，5 纵向修补，6 横向修补。纳入了用户已修正的 `SVRDD.yaml` 类别名称；旧版本 4/5/6 名称不正确。
- 标签均为合法五列 YOLO 检测框，整数类别 0–6、有限归一化坐标、正宽高；空标签和越界框均为 0。
- 训练集有 1 条完全重复的井盖框。保留官方原文件，框架按既有逻辑在内存去重。审计同时记录原始数与有效数，不能把原始框总数直接当作训练使用框数。

| 类别 | train 有效框 | val 框 | test 框 |
| --- | ---: | ---: | ---: |
| 纵向裂缝 | 3461 | 588 | 616 |
| 横向裂缝 | 2550 | 418 | 436 |
| 网状裂缝 | 1324 | 222 | 182 |
| 坑槽 | 679 | 115 | 124 |
| 井盖 | 2516 | 403 | 419 |
| 纵向修补 | 3109 | 519 | 500 |
| 横向修补 | 1980 | 347 | 295 |

全数据指纹（有序路径+原始标签+原始图像）：`6c39bc64da1fbf7f8db636feea2ed8cbf66dd525a8ae631c257337a097b1cd79`。服务器重新运行深检后应比较此指纹；改变标签换行也会改变指纹，应调查差异，不能直接忽略。报告中的清单指纹采用 LF 规范化，不能与原档案 CRLF 的字节 SHA 混为一谈。

正式入口会再次深检，并只重建派生的 `labels.cache`，防止框架按文件大小判断缓存有效而读到旧标签。原始图像和标签不改动。训练和选 best 使用 val；test 只在最终方案和协议锁定后评估。

## 4. 训练与评估协议 v2

修复三个容易被参数表掩盖的问题：默认 `patience=50` 导致各组提前结束；验证代码无条件转换 FP16；保存和最终精简权重无条件转换 FP16。现在显式尊重 `half=False`，并以 `v2-formal` / `v2-screen80` 区分新目录。旧固定 IoU 三组入口也统一 v2，保持其共同基线可比。

| 项目 | 正式协议 |
| --- | --- |
| 初始化 | 从 YAML 随机初始化，pretrained=False，resume=False |
| 输入与批量 | 640×640，batch=8，固定分辨率；训练 FP32、验证 FP32、保存 FP32 |
| 预算 | epochs=300，patience=0（关闭早停） |
| 优化器 | AdamW，lr=0.0001，beta1=0.9，beta2=0.999，weight_decay=0.0001 |
| 学习率 | lrf=1，cos_lr=False；预热结束后恒定学习率，不是余弦退火 |
| 预热 | 本 fork 的 warmup_epochs=2000 **实际表示迭代数**，约 2.67 个训练 epoch；bias 从 0 开始 |
| 梯度 | 本 fork 强制 nbs=batch，accumulate=1；梯度裁剪 max_norm=0.1；不能误写有效 batch64 |
| 数据 | 完整 train，val 每轮评估；不使用 test 选模型 |
| 增强 | mosaic=1，最后10轮关闭；mixup/copy_paste=0；hsv=.015/.7/.4；translate=.1，scale=.5，fliplr=.5；无旋转、竖直翻转或多尺度训练 |
| best 选择 | 框架 fitness=0.1×mAP50+0.9×mAP50–95，采用同一 checkpoint 的全部指标 |
| 重复 | 默认 seed0，支持 --seed；deterministic=True 不保证不同 GPU/软件版本逐位相同 |

可选 albumentations 的安装状态会改变增强。当前验证环境未安装；应固定服务器环境并保留每次快照中的包版本，不能中途升级或切换有/无 albumentations 的环境。原库自动验证 batch 为训练的两倍，正式为16，服务器 smoke 使用小验证子集，不能代替全量验证显存监控。

### IoU 四种含义必须分开

1. 回归：L1 权重5，标准 GIoU 权重2；分类使用 VFL，类别损失权重1；decoder辅助和去噪分支损失也参与反向传播。
2. Hungarian 匹配：类别/L1/GIoU 权重2/5/2。分类质量目标也使用标准 IoU。没有偷偷换成 Inner-IoU、PIoU、NWD 或 DS-IoU。
3. AP 匹配：mAP50 阈值0.5；mAP50–95采用0.50到0.95、步长0.05。这是评估定义。
4. 参数 `iou=0.7` 是通用 NMS 配置，当前 RTDETRValidator **不执行 NMS**，也不按 conf=0.001 截掉预测；保留300 queries、每query最高分的一个类别，再按分数排序。修改 iou 不会把本模型的 mAP 门槛改成0.7。混淆矩阵另用 confidence0.25/IoU0.45，不能与 AP 阈值混用。

解析值回归验证了标准 IoU、xywh/xyxy 两种 GIoU，包括不相交框 GIoU=-1/3；已有框回归测试继续通过。当前是本 fork 的评估协议，不应声称与所有官方 RT-DETR 实现完全相同。

## 5. 能否涨点：证据、机制与风险

### SCSA：保留为候选，但未在当前协议证实独立有效

原论文为 *Neurocomputing* 2025（预印本2024），不是“CVPR2024 SCSA”。SMSA/PCSA分别处理空间与通道关系，可用于突出裂缝方向和抑制路面背景。[官方 SCSA 仓库](https://github.com/HZAI-ZJNU/SCSA)

本地旧划分中 SCSA 相对旧基线：mAP50 +1.496pp，但 mAP50–95 -0.804pp；旧划分和旧 IoU 存在问题，不能据此认定当前场景涨点。当前 HSFPN+SCSA 为73.007/44.640，基线73.697/45.816，且这不是 SCSA 单因素实验。应先跑新 SCSA-only。若只提高 AP50 而降低 AP50–95，论文需说明定位精度的取舍。

### Conv3XC：有同类任务依据，值得优先小规模验证

*Pavement-DETR* 表4在 UAV-PDD2023 中：baseline79.4/50.9，+CSS81.3/54.8，+CSS+Conv3XC85.2/58.4；即在 CSS 条件下增加3.9/3.6pp。**没有 Conv3XC-only 行，也不是 SVRDD**。本项目只替换 P3，不能照搬增益。[论文全文及表4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031190/)

它改变训练参数化、通道变换和优化过程，可能有利于 P3 的细节融合；部署等价于3×3不意味着训练期免费。本实现替换了原 RepConv 内的 BN，保留外部 Conv/BN和输出SiLU；训练稳定性及收敛速度需观察。单模块估算训练计算明显增加，若收益很小应结合耗时决定是否保留。

### CDRB：理论动机成立，迁移风险最高

交通标志论文以 DRB+iRMB 替换 BasicBlock，报告本域增益；交通标志与低对比、细长的道路裂缝存在形态和背景差异，不能迁移其数字。[中文原论文](https://cje.ustb.edu.cn/cn/article/doi/10.13374/j.issn2095-9389.2025.06.11.001)

多尺度膨胀卷积和扩张窗口可补充长距离背景与缺陷上下文，参数也更少。但目前是全8块替换：三个降采样块从3×3 stride2变成1×1 stride2后再DRB，可能先损失细裂缝；C2/C3高分辨率阶段也开启MHSA，注意力开销不小。iRMB内残差关闭、exp_ratio2等是本项目适配选择，未被参考论文在SVRDD中验证。既不能把效果全归因于DRB，也不能把更少参数解释成更高精度。

[官方 EMO iRMB](https://github.com/zhangzjn/EMO/blob/main/model/emo.py) 和本仓采用相同扩张窗口重排；这是预期结构而不是索引错误。若全替换失败，后续可以另立方案只换C4/C5或保留原3×3降采样，但须重新命名并重新做相应消融，不能在当前8组中途改掉。

### 三者组合：重点检查负交互

SCSA 与 CDRB 的 MHSA，以及原有 AIFI 都进行信息选择/聚合，可能互补，也可能过度抑制弱纹理。CDRB省参数和Conv3XC增训练容量，不能自动相互抵消性能损失。检查 `AP(A+B)-AP(A)-AP(B)+AP(base)` 等组合差值，先描述交互趋势；单seed不能把差值称为统计显著。绝对提升百分比与百分点必须区分。

## 6. 控制算力支出的顺序

1. 先完成服务器深检和三模块 smoke，记录耗时/显存，确认运行的是本仓源码。
2. 用统一 `--phase screen` 跑 baseline、SCSA、CDRB、Conv3XC 四组（各80轮、seed0）。screen不关闭mosaic，以保持正式训练前80轮的增强安排；学习率也是相同的恒定计划。比较同一80轮预算、AP50与AP50–95、每类AP、后段趋势、耗时和显存。
3. 优先验证 SCSA+Conv3XC，再根据 CDRB-only 的趋势决定是否继续 CDRB 组合。80轮可能尚未收敛，不能仅凭它证明300轮必然无效；若仍在追赶，可给基线与候选共同延长预算的独立方案。当前入口正式300轮与screen80轮分别锁定，不能混表。
4. 候选确认后完成八组正式300轮消融。最终基线和完整模型优先补seed1、2：八组seed0加这4次重复，共12次正式运行。经费不足时如实报告单seed，不把微小差值当作可靠结论。
5. 最终选型后用同一评估入口在test评估，不根据test重选模块。所有负结果留档；若三个因素不能共同产生有意义收益，应换候选，不为凑三个“改进点”强行下结论。

历史基线73.697/45.816只作为参考。它有早停和旧精度行为；本轮v2必须重跑baseline，不能把它直接放入新消融主表。

## 7. 服务器使用

在已有项目目录执行（先保留服务器尚未提交的本地改动，`--ff-only`遇到分叉会停止）：

```bash
git pull --ff-only origin main
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出必须位于当前仓库。已有服务器环境优先复用，不要执行 `pip install -U ultralytics` 覆盖分支代码。本机验证环境：Python3.10.22，torch2.0.1+cu118，torchvision0.15.2+cu118，numpy1.26.4，scipy1.15.3，timm0.9.12，einops0.8.1，mmengine0.10.7，opencv-python4.10.0.84，PyWavelets1.8.0。三个新模块不要求额外编译CUDA扩展；仓库原有的模块注册会导入timm/einops/mmengine等，`pyproject.toml`没有完整声明这些扩展依赖。CUDA/PyTorch版本应适配实际服务器，不要仅为匹配本机随意降级。

确认 `SVRDD.yaml` 的 `path` 是服务器实际目录（默认 `/autodl-fs/data/RT-DETR/datasets/SVRDD-official-split`；某些实例挂载点不同）。也可以复制为 `SVRDD-server.yaml` 后通过 `--data` 指定，保留七类顺序和三个子目录。以下以默认文件为例：

```bash
# 全量检查数据，不训练；结果应与仓库的data-preflight报告指纹一致
python -m scripts.audit_svrdd --data SVRDD.yaml --output runs/server-data-audit.json

# 快速构建检查；本步不检查数据和GPU
python train.py --experiment svrdd-scsa-cdrb-conv3xc --check

# 也可一次完成所选模型和全量数据检查
python train.py --experiment svrdd-scsa-cdrb-conv3xc --data SVRDD.yaml --preflight

# 实际640/b8两批训练、验证和权重重新加载，输出在runs/server-smoke
python -m checks.smoke_ch3_training --experiment svrdd-scsa-cdrb-conv3xc --data SVRDD.yaml --device 0

# 四组筛选可逐条运行；每条均有独立输出目录
python train.py --experiment svrdd-baseline --phase screen --data SVRDD.yaml
python train.py --experiment svrdd-scsa --phase screen --data SVRDD.yaml
python train.py --experiment svrdd-conv3xc --phase screen --data SVRDD.yaml
python train.py --experiment svrdd-cdrb --phase screen --data SVRDD.yaml

# 正式实验示例（300轮；不要用smoke结果或旧baseline替代）
python train.py --experiment svrdd-baseline --data SVRDD.yaml
python train.py --experiment svrdd-scsa-cdrb-conv3xc --data SVRDD.yaml
```

单条命令只训练所选模型，不会自动排满8组。两批 smoke 只证明短流程可执行，且处于预热阶段；不证明全量训练显存/长期数值稳定。正式运行监控至少首轮训练及全量验证；若OOM，先统一修改整套协议，禁止只给某个模型缩batch或开启AMP后继续混比。`train_local_svrdd.py` 是旧HSFPN实验的本机脚本，不是本轮入口。

最终测试示例，路径替换成真实best.pt（方案锁定后才执行）：

```python
from ultralytics import RTDETR
model = RTDETR('runs/<正式运行目录>/weights/best.pt')
model.val(data='SVRDD.yaml', split='test', imgsz=640, batch=8,
          device='0', half=False, conf=0.001, iou=0.7, max_det=300)
```

## 8. 本轮验证记录

- 提交前合并运行全部相关检查：**78 passed、10 subtests passed**，覆盖下列模型、数据、IoU、协议、精度和来源记录检查；7条现有依赖提示，无失败。未运行依赖外部服务或下载预训练权重的整个上游测试库。
- 核心服务端审计回归：49 passed、10 subtests passed，覆盖模型结构/梯度/融合、训练入口、标准IoU/GIoU、官方划分和来源快照；仅有现有依赖弃用提示。
- 前一轮包含其他候选与SCSA回归：62 passed；与上项有重合，不能相加当作不同测试总数。
- 末次数据/协议/精度与smoke设备回归：14项全部通过，已包含于提交前合并检查中。
- 本地CPU真实图片 smoke：4张train、2张val、256输入、batch2；两批反传/优化、EMA验证、best/last保存、最终best重新验证和FP32重载均通过。子集AP无研究意义。
- 未执行服务器CUDA、正式300轮、ONNX/TensorRT和部署FPS验证。用于毕业论文的有效性结论仍需来自后续正式实验。
