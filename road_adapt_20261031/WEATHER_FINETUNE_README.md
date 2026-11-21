# 第四章基线：有预算的微调候选

入口为`train_weather_finetune.py`，profile为`baseline_finetune_v2`，目前仅支持H/F/C全部关闭的基线。原`train.py`和`train_weather_hfc.py`仍保留原协议。

**状态说明（2026-11-21）：本地先运行1个完整epoch，再从raw last续训1个完整epoch，短预算与两轮延续预算均通过预先设定的本地门槛；服务器尚未启动本候选。** 共6000次优化尝试全部成功，AMP跳步为0；4k验证集最佳mAP50–95为46.157%，两轮末为45.837%。这只支持已检查的短预算，不代表长期收敛、所有八组消融已就绪或6k主测试结论。实际状态以每次运行的`weather_protocol.json`及验收报告为准。

## 实际训练设置

| 项目 | 本候选 |
|---|---|
| 共同初始化 | 已批准的SVRDD七类原图基础RT-DETR-R18权重 |
| 训练数据 | 原24k天气列表；原图/雾/雨/低照，原数据和标注不变 |
| 输入、训练batch | 640；默认真实batch 8 |
| 其他默认运行参数 | seed=0，device=0，workers=0 |
| 在线增强 | 颜色、几何、Mosaic、MixUp等均保持关闭 |
| 优化器 | AdamW，LR=1e-5，weight decay=1e-4，momentum=0.9，梯度裁剪0.1 |
| LR日程 | `lrf=1.0`，预热后恒定；两轮本地预算已检查，长期收益仍待验证 |
| 预热 | policy中`warmup_updates=64`优先于占位的`warmup_epochs=0`；原生索引ni=0…64，首批LR为0 |
| 精度 | 示例带`--amp`启用训练AMP，初始scale=64；验证始终FP32 |
| 验证batch | 固定4，与训练batch独立 |
| 验证频率 | 训练前、每1000次优化器更新尝试、每个完整epoch末；边界去重 |
| 选模 | 原生fitness=0.1×mAP50+0.9×mAP50–95；同时记录分天气及合并退化组 |
| 上限与早停 | 默认总上限300个epoch；patience=20仍按完整epoch计算 |
| 本次调用预算 | 默认最多1个完整epoch，可用`--budget-epochs`指定 |

24k图、batch8时每个完整epoch有3000次更新尝试：在0/1000/2000/3000处完整验证4000张v2图，3000处只做一次epoch末验证。日志分别记录尝试、成功更新和AMP跳步次数；EMA仅随成功更新推进。中途验证保护随机状态与训练模式，不推进按epoch计算的早停器。

`paused_budget`表示本次调用处理完预算内的完整epoch并保留状态；它与早停、训练完成、科学验收通过分别独立。预算1不会把总上限改为1，也不会把patience20改成20次中途验证。

## 权重与续训

- `weights/last.pt`：完整epoch边界的raw学生、EMA、优化器、scaler、scheduler、RNG、加载器状态与更新计数；这是本入口的续训入口。
- `weights/best.pt`：最佳验证模型。初始或中途保存时是明确的推理文件（`epoch=-1, optimizer=None`）；epoch末同分时按原生规则允许较新的完整checkpoint覆盖。统一使用last续训。
- `weights/best_inference.pt`：真正完成后单独导出的推理副本；不会strip原last/best。
- `finetune_validation.jsonl`：initial/update/epoch/final验证及分条件成绩。
- `weather_protocol.json`：配置、来源摘要、`paused_budget`/`failed`/`completed`状态及续训历史。

首个完整epoch结束前不承诺存在last.pt；已有的中途best仍只能用于推理。

同一路径、同一入口、同一代码与数据版本才支持原运行续训；模型和优化器设置从原记录恢复。`--resume`时仅允许另带`--budget-epochs`及`--train`，AMP等设置自动恢复。真正早停或达到总上限的运行不作为原协议的续训任务。

本地资格验证与服务器正式新实验分别从同一原图权重开始。当前不提供把本地绝对路径记录直接搬到服务器的精确续训迁移。

## 服务器示例：仅在本地验收通过后使用

先确认新入口及其两个依赖已集成到项目，当前目录为：

```bash
cd /root/autodl-fs/RT-DETR/road_adapt_20261031
```

新运行先检查数据/原始权重并生成服务器绝对路径清单；此步骤不改原始数据。续训时沿用原记录，不重新生成配置。

```bash
/root/autodl-tmp/rtdetr-ch4-venv/bin/python - <<'PY'
from pathlib import Path
from chapter4_server import prepare_inputs
print(prepare_inputs(
    Path('/root/autodl-tmp/SVRDD-weather-v1'),
    Path('/root/autodl-tmp/chapter4_setup_20261120/original_baseline_best.pt'),
    Path('configs/chapter4')))
PY
```

不带`--train`只预览实际配置：

```bash
/root/autodl-tmp/rtdetr-ch4-venv/bin/python train_weather_finetune.py \
  --data configs/chapter4/dataset.yaml \
  --manifest /root/autodl-tmp/SVRDD-weather-v1/manifest.jsonl \
  --weights /root/autodl-tmp/chapter4_setup_20261120/original_baseline_best.pt \
  --amp --batch 8 --seed 0 --budget-epochs 1 \
  --name baseline_finetune_v2_s0_b8_amp
```

预览和输入验收通过后启动一个完整epoch的阶段；同名运行目录已存在时入口会拒绝覆盖。日志使用追加方式保留历史：

```bash
mkdir -p logs
nohup /root/autodl-tmp/rtdetr-ch4-venv/bin/python -u train_weather_finetune.py \
  --data configs/chapter4/dataset.yaml \
  --manifest /root/autodl-tmp/SVRDD-weather-v1/manifest.jsonl \
  --weights /root/autodl-tmp/chapter4_setup_20261120/original_baseline_best.pt \
  --amp --batch 8 --seed 0 --budget-epochs 1 \
  --name baseline_finetune_v2_s0_b8_amp --train \
  >> logs/baseline_finetune_v2_s0_b8_amp.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -n 80 -f logs/baseline_finetune_v2_s0_b8_amp.log
```

确认状态为`paused_budget`、检查完整验证趋势后，使用同一入口再续一个完整epoch；此时不附加`--amp`、路径或新训练参数。下面只展示恢复方式，追加预算应限定在已经本地检查的范围内；两轮通过不代表第三轮或长期训练已验证：

```bash
nohup /root/autodl-tmp/rtdetr-ch4-venv/bin/python -u train_weather_finetune.py \
  --resume runs/weather_finetune/baseline_finetune_v2_s0_b8_amp/weights/last.pt \
  --budget-epochs 1 --train \
  >> logs/baseline_finetune_v2_s0_b8_amp.log 2>&1 < /dev/null &
```

**训练进程预算暂停或正常退出不会自动关闭付费实例。** 本入口不操作服务器开关机；实例运行状态与费用需要另行确认。

## 论文与验收边界

资格验证要覆盖完整24k训练流、完整4k选模验证和可恢复的epoch边界状态。512步筛查、CPU测试通过或一次预算暂停均不能代替效果验收。H/F/C正式因素实验仍需共同训练规则下的实测；当前成绩不直接填入论文6k主测试表。数据与角色口径见[数据逻辑说明](docs/baseline-data-logic-notes.md)。
