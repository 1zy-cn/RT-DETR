# SCSA 启用验证

适用配置：`rtdetr-resnet18-SCSA.yaml` 和 `resnet18-HSFPN+SCSA.yaml`。
两者的四个 ResNet18 阶段均设置 `att=True`，每阶段两个残差块，共 8 个 SCSA。
`rtdetr-resnet18.yaml` 和 `rtdetr-resnet18-HSFPN.yaml` 是不启用 SCSA 的对照组。

## 执行检查

从仓库根目录运行，使用已安装本项目依赖的 Python 环境：

```bash
python -m pytest checks/test_scsa.py -v -s -o addopts=
```

可生成包含参数量、合成损失和分支梯度的机器可读记录：

```bash
python -m pytest checks/test_scsa.py -v -s -o addopts= -o junit_family=legacy --junitxml=scsa-validation.xml
```

测试强制在 CPU 上运行，限制为两个计算线程，种子为 0，类别数为 7。
无需本地数据集或预训练权重；输入及两个目标框均为合成数据。
没有调用训练器的训练循环，但会执行真实检测损失的反向传播和一次 SGD 参数更新。
导入原项目时仍可能运行其内置联网状态探测；测试不依赖联网成功。
设置 `YOLO_AUTOINSTALL=false` 可以关闭项目自动安装依赖的行为。
如果需要隔离项目设置，可将 `YOLO_CONFIG_DIR` 指向临时目录。

验证内容：

1. 四组模型由真实解析器构建，SCSA 数量分别为 0、0、8、8。
2. SCSA 通道数为 64、64、128、128、256、256、512、512。
3. 对两个启用配置挂接前向钩子，验证每个 SCSA 均实际被调用。
4. 在相同权重和输入下关闭注意力，验证输出变化，且关闭期间没有调用 SCSA。
5. 使用真实 RT-DETR 检测损失反向传播，检查每个 SCSA 参数有有限梯度，空间和通道分支梯度均非零。
6. 一次优化器更新后，每个 SCSA 的空间卷积权重均发生变化。
7. 两个配置均完成 640×640 推理，输出为 `[1, 300, 11]` 且没有 NaN/Inf。

反向检查使用 256×256 输入以减少 CPU 开销。当前 SCSA 使用 7×7 池化，
测试不能随意缩小至 128×128，因为最深层特征图会小于池化窗口。
独立构建模型后，测试设置 `model.nc=7`，对应正式训练器的
`DetectionTrainer.set_model_attributes()` 行为。

## 已验证的 CPU 环境

Windows、Python 3.11.8、PyTorch 2.0.1+cpu、torchvision 0.15.2+cpu、NumPy 1.26.4、
timm 1.0.29、mmengine 0.10.7、einops 0.8.2、pytest 8.3.5。
除 `pyproject.toml` 中的依赖，还需项目自定义模块导入使用的
`timm`、`mmengine`、`einops`、`PyWavelets`、`torchsummary`。
验证环境使用独立虚拟环境；CPU 依赖版本不能直接替代服务器的 CUDA 环境。

## 后续正式实验的加载方式

使用修正后的 YAML 新建模型，并使用新的实验目录名。
直接加载旧 `.pt` 文件或恢复旧训练，会沿用检查点里的模型结构及配置；
修改磁盘上的 YAML 不会自动给旧模型增加 SCSA。

```python
from ultralytics import RTDETR
from ultralytics.nn.Addmodules.SCSA import BasicBlock_SCSA, SCSA

model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18-SCSA.yaml')
# 组合实验改用 ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml
assert sum(isinstance(m, SCSA) for m in model.model.modules()) == 8
blocks = [m for m in model.model.modules() if isinstance(m, BasicBlock_SCSA)]
assert len(blocks) == 8 and all(m.att and hasattr(m, 'se') for m in blocks)
```

如需迁移已有权重，应先按新 YAML 构建模型，再用 `model.load(...)` 加载形状匹配的参数，
随后重新核验 SCSA；新增参数需要训练。各对照组必须采用一致、明确记录的初始化策略。
当前 `train.py` 默认选择 `resnet18-HSFPN+SCSA.yaml`，用于重新训练有效的组合实验。
`python train.py --check` 只检查模型结构和显示训练参数，不开始训练，也不验证数据集或 GPU。
训练启动后还会检查训练器实际构建的模型，并将入口、模型 YAML、数据 YAML 和 Git/环境信息保存在实验目录的 `source_snapshot/`。

验证入口拒绝错误模型及保存配置快照：

```bash
python -m pytest checks/test_scsa.py checks/test_train_entrypoint.py -v -o addopts=
```

此次检查证明启用和梯度路径有效，不衡量 mAP，不代替真实数据训练、GPU/AMP 稳定性测试或消融实验。
历史目录名包含 SCSA 不能单独证明当时启用了注意力；旧结果先保留，待核对对应权重和训练时配置。
