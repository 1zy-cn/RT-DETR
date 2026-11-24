"""第四章服务器训练入口：先修改下方路径，再运行 python train.py。

DATA_ROOT 和 ORIGINAL_WEIGHTS 是示例，必须改成服务器上的实际路径。
DATA_ROOT 应包含完整天气数据和 manifest.jsonl；ORIGINAL_WEIGHTS 必须是
已有的 SVRDD 七类原图基础 RT-DETR-R18 best.pt，所有消融组共用这一权重。

python train.py          检查输入、重建路径清单，然后开始训练。
python train.py --check  检查输入并预览参数，不开始训练或创建正式运行目录。

默认使用已本地验收的 baseline_finetune_v2：单次最多一个完整epoch，
每1000次更新完整验证并保留best；预算暂停不是早停，也不是新配方效果验收。
LR0=1e-5用于原图权重的初始天气适配，后峰值低LR诊断不改变这一默认。

baseline_finetune_v2目前仅支持baseline。旧八组配方需显式选择legacy_hfc，
其原学习率/预热不同，不能与v2混用后直接归因于结构；统一消融协议尚待验收。
legacy_hfc含C组自动以ORIGINAL_WEIGHTS为冻结教师。
RESUME_CHECKPOINT 非空时，沿用原 weather_protocol.json；顶部新训练参数
不会覆盖原记录，也不会重建数据清单。续训时请保持原数据、代码和路径。
"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# ===== 服务器参数编辑区：先修改这两个示例路径 =====
CONFIGURATION = "baseline"  # baseline / h / f / c / hf / hc / fc / hfc
TRAINING_PROFILE = "baseline_finetune_v2"  # baseline-only; legacy_hfc须显式选择
DATA_ROOT = Path("/root/autodl-tmp/SVRDD-weather-v1")
ORIGINAL_WEIGHTS = Path("/root/autodl-tmp/SVRDD-baseline/weights/best.pt")
SEED = 0
BATCH = 8
AMP = True
WORKERS = 0
DEVICE = "0"
EPOCHS = 30  # 总上限；v2每次执行还受BUDGET_EPOCHS约束
LEGACY_EPOCHS = 300  # 仅旧legacy_hfc，保留其原调度上限
BUDGET_EPOCHS = 1
LR0 = 1e-5  # 仅v2新运行；legacy_hfc沿用其原始配方
LRF = 1.0
COS_LR = False
IMGSZ = 640
PROJECT = ROOT / "runs/weather_hfc"
RESUME_CHECKPOINT = None  # 续训时改成 Path("原运行目录/weights/last.pt")
# ===== 编辑区结束 =====


def _saved_profile(checkpoint):
    record_path = Path(checkpoint).resolve().parent.parent / "weather_protocol.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    protocol = record.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Resume record must contain its original protocol")
    profile = protocol.get("training_profile")
    if profile is None:
        return "legacy_hfc"
    if profile != "baseline_finetune_v2":
        raise ValueError(f"Unsupported saved training profile: {profile}")
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查输入和预览参数，不训练")
    args = parser.parse_args(argv)

    if RESUME_CHECKPOINT is not None:
        profile = _saved_profile(RESUME_CHECKPOINT)
        print("续训沿用原 weather_protocol.json；保持原数据、代码和路径，忽略顶部新训练参数。")
        weather_args = ["--resume", str(RESUME_CHECKPOINT)]
        if profile == "baseline_finetune_v2":
            weather_args += ["--budget-epochs", str(BUDGET_EPOCHS)]
    else:
        profile = TRAINING_PROFILE
        if profile not in {"baseline_finetune_v2", "legacy_hfc"}:
            raise ValueError(f"Unknown TRAINING_PROFILE: {profile}")
        if profile == "baseline_finetune_v2" and CONFIGURATION != "baseline":
            raise ValueError("baseline_finetune_v2目前只验收baseline；其他组需先统一并验收训练协议，"
                             "或显式选择原legacy_hfc配方，不能混用后作结构比较。")
        # Keep imports lazy: merely importing this editable file never touches data.
        from chapter4_server import prepare_inputs

        dataset = prepare_inputs(DATA_ROOT, ORIGINAL_WEIGHTS, ROOT / "configs/chapter4")
        precision = "amp" if AMP else "fp32"
        weather_args = [
            "--data", str(dataset),
            "--manifest", str(DATA_ROOT / "manifest.jsonl"),
            "--weights", str(ORIGINAL_WEIGHTS),
            "--seed", str(SEED),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--device", str(DEVICE),
            "--epochs", str(EPOCHS if profile == "baseline_finetune_v2" else LEGACY_EPOCHS),
            "--imgsz", str(IMGSZ),
            "--project", str(PROJECT),
            "--name", f"{CONFIGURATION}_{profile}_seed{SEED}_b{BATCH}_{precision}",
        ]
        if profile == "baseline_finetune_v2":
            weather_args += ["--lr0", str(LR0), "--lrf", str(LRF),
                             "--budget-epochs", str(BUDGET_EPOCHS)]
            if COS_LR:
                weather_args.append("--cos-lr")
        else:
            print("使用旧legacy_hfc配方；v2的LR0/LRF/COS_LR/BUDGET_EPOCHS不适用于此模式。")
            weather_args += ["--configuration", CONFIGURATION]
        if AMP:
            weather_args.append("--amp")

    if not args.check:
        weather_args.append("--train")

    if profile == "baseline_finetune_v2":
        from train_weather_finetune import main as train_weather
    else:
        from train_weather_hfc import main as train_weather

    return train_weather(weather_args)


if __name__ == "__main__":
    main()
