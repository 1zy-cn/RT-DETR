# 第四章服务器：先训练基线

## 直接使用 train.py

在服务器已有RT-DETR仓库根目录执行：

```bash
git pull --ff-only
cd road_adapt_20261031
```

编辑本目录`train.py`顶部参数区，只先填写这两项服务器实际路径：

```python
DATA_ROOT = Path("天气数据集SVRDD-weather-v1的实际目录")
ORIGINAL_WEIGHTS = Path("已有原图基础RT-DETR-R18的best.pt实际路径")
```

保留`CONFIGURATION = "baseline"`、`BATCH = 8`、`AMP = True`、`SEED = 0`。然后：

```bash
python train.py --check
python -u train.py
```

第一条核对输入并预览，不开始训练；第二条直接开始。它们都会检查原图初始化权重SHA256、天气manifest版本、七类名称和全部图像/标签路径，再生成本服务器的18份本地数据配置。只按manifest核对文件存在，不重算44k图像像素哈希，也不替代目标可辨性核验。脚本不会改原始图像或标签，不使用数据根里带旧Windows路径的训练清单。

`DATA_ROOT`需要已有完整44,000张天气数据及标签、`manifest.jsonl`和`data_weather_train.yaml`。Git只同步代码/模板/生成器，不携带天气数据或权重；仅有原图8,000张的目录不满足这个入口。此处的自动配置重建路径清单，不重新生成天气图像。数据和权重保持与本地已核准版本相同。

原图权重SHA256应为`faf76f76820986b8affaa2de9d91c0c7c3a76c105de3814430d87735d1bcd69b`，文件名相同不代表模型相同。原图教师已训练完成，本次不需要再训练它。

## 数据与论文逻辑

当前本地manifest及关键清单摘要与上次全量审计一致；论文的数据协议无需为服务器另建一套：

| 用途 | 图像数 | 组成 |
| --- | ---: | --- |
| 训练 | 24,000 | 6,000原图＋雾/雨/低照各6,000；每天气轻/中/重各2,000 |
| 固定选模验证 | 4,000 | 1,000原图＋每天气轻/中各500；不含重度 |
| 主测试 | 6,000 | 每天气轻/中各1,000，最后合并计算AP |
| 独立补充测试 | 1,000原图、3,000重度 | 不用于选模 |

原图与全部退化版本继承来源ID并保留在同一split，训练/验证/测试来源分别为6,000/1,000/1,000，交集为0；七类和标注几何保持一致。主测试6,000张来自1,000个原始来源，不能当作6,000个独立街景来源。

本次`baseline`指基础RT-DETR在上述24k混合天气数据上适配，关闭H/F/C。它不是只用6,000张原图的旧基线重训。后续所有组仍从同一个`ORIGINAL_WEIGHTS`开始，含C时也用它作冻结教师；不要把新训练得到的天气baseline best.pt替换到`ORIGINAL_WEIGHTS`里。

当前协议保留重度图检测监督，C仅作用于轻/中度配对，与论文当前表述一致。此前重度低照细裂缝的人工可辨性仍待核验；如决定修改重度样本排除/忽略规则，应对所有方法统一，已经训练的baseline也需要按新规则重训。当前结论限定为受控合成天气。

## 统一参数与结果目录

`train.py`服务器预设：640、batch8、AMP、workers0、AdamW、lr0/weight_decay均1e-4、最多300轮、patience20、seed0；`nbs=batch`，梯度累积为1。预热为3轮对应的小批数（24k/b8时为9,000）；固定几何/HSV增强关闭，AP阈值0.001且最多300预测，不加NMS。论文旧稿关于预热仅索引0–3以及不执行0.001过滤的表述仍需同步。

服务器预设与旧高级入口`train_weather_hfc.py`的本地默认batch2/FP32有意区分。8组正式比较统一使用最终选定的batch和精度，不能各组随意修改；调整模型时通常只改`CONFIGURATION`。本地RTX3060的batch8/AMP单步检查不等于服务器长程显存/收敛已实测，服务器环境与实际AMP状态会写入运行记录。

默认基线目录为`runs/weather_hfc/baseline_seed0_b8_amp/`。该运行自己的`weights/best.pt`、`last.pt`、`results.csv`及`weather_protocol.json`保存在此。已有同名运行目录拒绝覆盖；后续改变组名会自动产生独立目录。

## 消融与续训

后续把顶部`CONFIGURATION`改为以下一个值即可，八组完整配置均已存在：

```python
CONFIGURATION = "h"    # baseline / h / f / c / hf / hc / fc / hfc
```

每组保持同一数据和`ORIGINAL_WEIGHTS`，切换组后重新运行`python train.py --check`及`python -u train.py`。重复实验仅另改`SEED`。

如需恢复中断的基线，把顶部设为：

```python
RESUME_CHECKPOINT = Path("原基线运行目录/weights/last.pt")
```

随后仍运行`python -u train.py`。续训严格沿用原运行记录，不应用顶部新配置，也不重建数据清单；保持代码、输入和路径不变。开始下一消融组前，把`RESUME_CHECKPOINT`恢复为`None`。

## 环境

使用原来能训练本RT-DETR项目的Python环境。已验证本地Python3.10、PyTorch2.0.1+cu118、torchvision0.15.2+cu118；自定义模块导入还需要timm、einops、mmengine、PyWavelets。若原环境缺少这些模块，可执行`python -m pip install -r requirements-weather-extra.txt`补齐。不要在导入前安装另一个官方ultralytics版本替代本目录源码。

新版PyTorch完整模型检查点需要显式的可信模型加载语义，相关入口已保留用户原图权重/自有训练权重的完整模型加载。[PyTorch官方序列化说明](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true)。
