# 第四章基线的入口与预算

2026-11-24。`train.py`的新运行默认转入已本地验收的`baseline_finetune_v2`，不再静默使用旧八组入口的LR1e-4与3epoch预热。模型、损失、数据和教师没有在本次修正中改变。

## 运行方式

仍在`train.py`顶部修改服务器实际的`DATA_ROOT`和`ORIGINAL_WEIGHTS`；原图初始化必须是SVRDD七类基础RT-DETR-R18权重。新基线初始适配的默认LR为1e-5、lrf1、真batch8、AMP、patience20、总上限30epoch，每次最多执行1个完整epoch，并在每1000次更新完整验证。

```bash
python train.py --check
python -u train.py
```

`--check`检查输入并预览实际协议，不创建训练目录或加载模型权重。正常运行到单次预算时出现`BUDGET_PAUSED`，这是主动预算暂停，不是训练失败或早停。先查看该阶段完整验证与best，再决定是否继续；不要直接启动一整晚来证明轮数足够。

把`RESUME_CHECKPOINT`改为原运行的`weights/last.pt`可按完整状态继续。入口从原`weather_protocol.json`自动选择配方，忽略顶部的新LR、模型组和其他新运行参数；v2只允许`BUDGET_EPOCHS`调整本次执行预算。底层仍会检查冻结源码、数据和运行状态，旧源码训练的checkpoint应在原冻结版本续训，不应绕过SHA检查。

## 已知结果与使用边界

本地2026-11-22运行由原图初始化的4k验证41.777%提高到第2000次更新46.171%；第21轮末41.635%不能替代best。使用保存的best是正常选模，不需要通过增加patience追求每轮上涨。这些都是验证值，不是正式test成绩。

后峰值的LR1e-6实验是从weatherbest重新建立优化器的局部诊断，不支持把B0初始适配默认改为1e-6。诊断endpoint无优化器状态，不能当原生resume；只比继续训练的控制组好也不等于超过原best。

`baseline_finetune_v2`目前只验收B组。`CONFIGURATION`改成其他组会明确报错，避免自动混用不同训练协议。旧`baseline/h/f/c/hf/hc/fc/hfc`配方仍可通过显式`TRAINING_PROFILE="legacy_hfc"`使用，保留其原超参数及`LEGACY_EPOCHS=300`调度上限；此模式不使用v2的EPOCHS/LR0/LRF/COS_LR/BUDGET_EPOCHS。切换组和配方需共同固定后再做公平消融，不能将不同训练配方的差值作为HFC结构贡献。

## 回归验证

新回归先在旧入口观察到6项行为失败、2项通过；修正后相关59项CPU检查通过，覆盖真实协议预览、默认路由、预算、学习率、续训、输入和预算暂停。专用pytest配置避开旧项目全量doctest设置：

```bash
python -m pytest -c checks/pytest_chapter4.ini -q checks/test_bounded_server_entry.py checks/test_finetune_entrypoint.py checks/test_finetune_lr_options.py checks/test_weather_finetune_budget.py checks/test_chapter4_server_inputs.py
```
