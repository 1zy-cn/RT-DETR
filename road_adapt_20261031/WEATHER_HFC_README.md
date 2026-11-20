# 第四章 HFC 模型与数据配置

服务器先训练基线：进入本子目录，修改`train.py`顶部的`DATA_ROOT`和`ORIGINAL_WEIGHTS`，保持`CONFIGURATION = "baseline"`，执行`python train.py --check`，通过后`python -u train.py`。后续仅修改`CONFIGURATION`即可切换八组消融。完整步骤及数据逻辑见[服务器基线指南](docs/experiments/chapter4-server-baseline.md)。

`train.py`是服务器预设（batch8、AMP）；下文`train_weather_hfc.py`与PowerShell高级入口保留本地默认（batch2、FP32）。同一正式比较需固定统一参数。

2026-11-19依据最新第四章完成本地复核。H/F/C及八组模型YAML在11月12日已经实现，本次复用并补齐数据、训练、评价与续训入口。配置检查不等于正式实验结果。

2026-11-20再次独立审查，当前准备范围为下列8组核心配置及12次训练计划；外部8种比较模型尚未就绪。当前本机默认协议是640、batch2、FP32，不能与论文旧稿的batch8、AMP混写。新的审查证据及修复记录见[本轮审查](docs/research/chapter4-readiness-2026-11-20.md)。

## 模型与参数

配置名：`baseline/h/f/c/hf/hc/fc/hfc`，文件在`ultralytics/cfg/models/rt-detr/weather-hfc/`。
H为P2→P3首个残差块的共享Haar下采样，F为P4投影后的FFT频带选择，C为冻结原图基础教师与学生融合后P3的GT框3×3区域余弦约束，λ=0.1。C只用于轻中度配对；原图、重度仍接受检测监督。推理仅保留学生。

默认640、batch2、workers0、FP32、AdamW、lr0/weight_decay=1e-4、lrf0.1、线性日程、最多300轮、patience20。Weather预热按`round(3*len(train_loader))`小批迭代；其他既有trainer的预热默认行为不变。几何/HSV/Mosaic/MixUp增强关闭。AP保留分数≥0.001且最多300个预测，不向DETR增加NMS。

默认解释器：`D:/Users/86186/Miniconda3/envs/rtdetr/python.exe`。
共同学生初始化及固定教师：
`D:/Users/86186/ResearchData/SVRDD-local-runs/svrdd-r18-baseline-officialsplit-local-amp-b8-p20-s0/weights/best.pt`。
该权重为七类、640、原图官方划分、epoch168的基础RT-DETR，不使用第三章完整改进模型当教师。

## 数据配置

图像与标签仍在`D:/Users/86186/ResearchData/SVRDD-weather-v1`，项目只增加清单/YAML，不复制或修改这些输入。

| 用途 | 项目配置或清单 | 图像数 |
| --- | --- | ---: |
| 训练 | configs/chapter4/train_mixed.txt | 24000 |
| 选模 | configs/chapter4/valid_selection_v2.txt | 4000 |
| 主测试 | configs/chapter4/test_primary.yaml | 6000 |
| 原图 | configs/chapter4/test_clear.yaml | 1000 |
| 雾/雨/低照 | configs/chapter4/test_fog.yaml、test_rain.yaml、test_lowlight.yaml | 各2000 |
| 重度压力 | configs/chapter4/test_severe.yaml | 3000 |
| 计时输入 | configs/chapter4/timing_valid_200.txt | 200 |

默认训练数据`configs/chapter4/dataset.yaml`：train24k、val-v2 4k、test6k。v2含原图1000，每类天气轻/中各500，不含重度；JSONL SHA256为`172f75da98920067a499265b88da78fe6af03c956b35598d0b339ba7ac4193a2`。各测试YAML的val始终是同一选模集，正式评价需要`split=test`，新评价入口已固定。

来源、计数与文件SHA在`configs/chapter4/provenance.json`。如需重建，在本项目目录执行：

```powershell
& 'D:\Users\86186\Miniconda3\envs\rtdetr\python.exe' .\scripts\configure_chapter4_data.py
```

脚本支持`--data-root/--output`，默认拒绝覆盖内容不同的文件，审查差异后才使用`--force`；禁止输出到原始数据根。

## 新训练与续训

在本项目目录打开PowerShell。默认只预览，不创建训练运行目录：

```powershell
.\run_weather_hfc_experiment.ps1 -Configuration hfc -Seed 0
```

明确开始新训练才添加`-Train`；已有同名目录拒绝覆盖：

```powershell
.\run_weather_hfc_experiment.ps1 -Configuration hfc -Seed 0 -Train
```

八组各seed0，baseline/hfc另补seed1、2，共12次核心训练。服务器`train.py`计划在`checks/chapter4_run_plan_20261119.csv`，运行名包含`_b8_amp`。模型出现在多张结果表中时复用其权重，不重复训练。

八组应采用同一batch、精度和训练预算。当前本地就绪验证只覆盖batch2、FP32；切换batch8、AMP属于另一套运行参数，需要先验证显存与数值稳定，再统一应用于所有组并同步论文。

续训使用原运行目录的`last.pt`，默认同样只预览：

```powershell
.\run_weather_hfc_experiment.ps1 -Resume '.\runs\weather_hfc\hfc_seed0\weights\last.pt'
.\run_weather_hfc_experiment.ps1 -Resume '.\runs\weather_hfc\hfc_seed0\weights\last.pt' -Train
```

续训不与新configuration/data/teacher/seed等参数混用。`weather_protocol.json`冻结源码、配置、manifest、train/val/test列表及原始学生/教师权重指纹，变化时拒绝续训。仅支持本版保存了优化器及运行状态的中断检查点，旧短程权重、已strip权重、已完成预算或早停运行不可作为安全续训状态。

恢复raw学生与对应optimizer，EMA单独恢复，同时恢复scheduler、scaler、早停及随机状态。支持epoch边界状态延续，不宣称多worker队列及非确定算子的逐位复现。教师不进入学生/EMA。默认plots=False时，Weather数据检查不下载字体或缺失数据；其他trainer的默认检查行为保留。

Weather训练采用有限epoch DataLoader，默认workers0下已检查中断恢复后连续两轮样本顺序与不中断运行一致；普通RT-DETR仍使用原有InfiniteDataLoader。不要在此固定几何协议中启用Mosaic或close_mosaic。

## 正式评价

默认只预览参数和清单，不读取测试像素：

```powershell
.\run_weather_hfc_evaluation.ps1 -Weights '.\runs\weather_hfc\hfc_seed0\weights\best.pt' -Group primary
```

明确评价才添加`-Evaluate`：

```powershell
.\run_weather_hfc_evaluation.ps1 -Weights '.\runs\weather_hfc\hfc_seed0\weights\best.pt' -Group primary -Evaluate
```

Group可选`primary/clear/severe/fog/rain/lowlight`。评价读取学生EMA（如有），核对七类顺序，固定640、FP32、split=test和统一AP门槛；保存`evaluation_protocol.json`及逐图预测JSON，已有输出目录拒绝覆盖。主结果在6000张轻中度图上合并计算AP，不能用三个天气AP的算术平均替代。

独立评价入口显式关闭几何增强，避免合并库默认的Mosaic等参数后被配对数据加载器拒绝。

## 从Git恢复配置

本子目录包含HFC所需的本地Ultralytics源码，运行时应先进入本目录，避免导入仓库根目录的旧实现。图像、标签、权重以及包含逐图路径的生成清单、provenance和本机数据YAML均不入Git；Git只保存模型配置、`dataset.example.yaml`结构模板和生成器。当前本机已生成的配置仍保留；其他设备需要复制同一版本的数据和原图基线权重，并重建路径配置：

```bash
cd road_adapt_20261031
python scripts/configure_chapter4_data.py --data-root /data/SVRDD-weather-v1 --force
python train_weather_hfc.py --configuration hfc --manifest /data/SVRDD-weather-v1/manifest.jsonl --weights /data/original-baseline-best.pt --teacher /data/original-baseline-best.pt
```

上述最后一行仍为预览，确认路径后添加`--train`开始。生成器只重写本项目的清单/YAML，不改数据图像或标签；会核对manifest及选模清单。原图基线权重SHA256为`faf76f76820986b8affaa2de9d91c0c7c3a76c105de3814430d87735d1bcd69b`。环境版本见`checks/weather_environment_20261120.json`，该文件记录已验证本机环境，不是跨平台安装锁。

本子目录`.gitattributes`保留审计文件的原始字节，避免Git换行转换破坏已核准清单的SHA256。正式运行记录冻结代码及输入文件，续训期间保持代码、数据、教师和运行目录不变。

服务器也可以直接用`train.py`，由它自动校验并生成上述本地清单，不必手工执行生成器。`python train.py --check`不训练，但会重建本机项目内的派生配置；生成清单、数据和权重继续不入Git。

## 验证与尚待工作

使用`checks/road_pytest.ini`与显式Weather测试文件；不要直接运行带其他清理行为的通用`tests/conftest.py`。`checks/verify_weather_hfc.py`默认包含真实短程训练，不能当只读审计命令。

本次未启动正式训练、正式test或计时。计时清单已准备，实际检测时延仍待正式权重测量。八个外部比较架构的实现和对应SVRDD原图权重尚未逐一核验，不把论文表里的模型名当作已部署实现。

训练前仍应抽检困难目标及重度退化配对的可辨性，再统一决定重度检测监督规则。文件/标签一致不等于目标仍可辨。论文mock数据仍需真实结果替换。本轮审查与验证记录位于Codex论文任务目录`work/chapter4_project_setup`，历史11月12日的101项检查单独保留。
