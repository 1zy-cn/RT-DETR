"""第四章服务器训练入口：先修改下方路径，再运行 python train.py。

DATA_ROOT 和 ORIGINAL_WEIGHTS 是示例，必须改成服务器上的实际路径。
DATA_ROOT 应包含完整天气数据和 manifest.jsonl；ORIGINAL_WEIGHTS 必须是
已有的 SVRDD 七类原图基础 RT-DETR-R18 best.pt，所有消融组共用这一权重。

python train.py          检查输入、重建路径清单，然后开始训练。
python train.py --check  检查输入并预览参数，不开始训练或创建正式运行目录。

后续仅修改 CONFIGURATION 即可切换 baseline/h/f/c/hf/hc/fc/hfc。
含 C 的组自动以 ORIGINAL_WEIGHTS 为冻结教师，不需要重新训练原图教师。
RESUME_CHECKPOINT 非空时，沿用原 weather_protocol.json；顶部新训练参数
不会覆盖原记录，也不会重建数据清单。续训时请保持原数据、代码和路径。
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# ===== 服务器参数编辑区：先修改这两个示例路径 =====
CONFIGURATION = "baseline"  # baseline / h / f / c / hf / hc / fc / hfc
DATA_ROOT = Path("/root/autodl-tmp/SVRDD-weather-v1")
ORIGINAL_WEIGHTS = Path("/root/autodl-tmp/SVRDD-baseline/weights/best.pt")
SEED = 0
BATCH = 8
AMP = True
WORKERS = 0
DEVICE = "0"
EPOCHS = 300
IMGSZ = 640
PROJECT = ROOT / "runs/weather_hfc"
RESUME_CHECKPOINT = None  # 续训时改成 Path("原运行目录/weights/last.pt")
# ===== 编辑区结束 =====


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查输入和预览参数，不训练")
    args = parser.parse_args(argv)

    if RESUME_CHECKPOINT is not None:
        print("续训沿用原 weather_protocol.json；保持原数据、代码和路径，忽略顶部新训练参数。")
        weather_args = ["--resume", str(RESUME_CHECKPOINT)]
    else:
        # Keep imports lazy: merely importing this editable file never touches data.
        from chapter4_server import prepare_inputs

        dataset = prepare_inputs(DATA_ROOT, ORIGINAL_WEIGHTS, ROOT / "configs/chapter4")
        precision = "amp" if AMP else "fp32"
        weather_args = [
            "--configuration", CONFIGURATION,
            "--data", str(dataset),
            "--manifest", str(DATA_ROOT / "manifest.jsonl"),
            "--weights", str(ORIGINAL_WEIGHTS),
            "--seed", str(SEED),
            "--batch", str(BATCH),
            "--workers", str(WORKERS),
            "--device", str(DEVICE),
            "--epochs", str(EPOCHS),
            "--imgsz", str(IMGSZ),
            "--project", str(PROJECT),
            "--name", f"{CONFIGURATION}_seed{SEED}_b{BATCH}_{precision}",
        ]
        if AMP:
            weather_args.append("--amp")

    if not args.check:
        weather_args.append("--train")

    from train_weather_hfc import main as train_weather

    return train_weather(weather_args)


if __name__ == "__main__":
    main()
