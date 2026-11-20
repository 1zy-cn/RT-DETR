"""Protocol checks independent of model construction and GPU."""
import importlib.util
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_runner():
    path = ROOT / "train_weather_hfc.py"
    assert path.is_file(), "weather training runner is not implemented"
    spec = importlib.util.spec_from_file_location("weather_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def arguments(module, tmp_path, configuration="hfc"):
    for name in ("data.yaml", "manifest.jsonl", "weights.pt"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(dict(nc=7,names=[str(i) for i in range(7)])),encoding="utf-8")
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir(exist_ok=True)
    flags = dict(haar="h" in configuration, freq="f" in configuration, consistency="c" in configuration)
    (cfgdir / f"{configuration}.yaml").write_text(yaml.safe_dump(dict(nc=7, weather_hfc=flags)), encoding="utf-8")
    args = module.make_parser().parse_args([
        "--configuration", configuration, "--data", str(tmp_path / "data.yaml"),
        "--manifest", str(tmp_path / "manifest.jsonl"), "--weights", str(tmp_path / "weights.pt"),
        "--project", str(tmp_path / "runs"), "--name", "test",
    ])
    return args, cfgdir

def test_protocol_keeps_geometry_identical(tmp_path):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path)
    p = m.build_protocol(args, config_dir=cfgdir)
    for key in ("degrees","translate","scale","shear","perspective","flipud","fliplr","mosaic","mixup","copy_paste"):
        assert p["train_args"][key] == 0
    assert p["train_args"]["pretrained"] == str(tmp_path / "weights.pt")
    assert p["teacher_checkpoint"] == str(tmp_path / "weights.pt")
    assert p["train_args"]["amp"] is False
    assert p["configuration"] == "hfc"

def test_baseline_needs_no_teacher(tmp_path):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path, "baseline")
    p = m.build_protocol(args, config_dir=cfgdir)
    assert p["teacher_checkpoint"] is None

@pytest.mark.parametrize("field,value", [("batch",0),("epochs",0),("workers",-1),("consistency_weight",-0.1),("imgsz",639),("imgsz",64),("imgsz",96)])
def test_invalid_protocol_rejected(tmp_path, field, value):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        m.build_protocol(args, config_dir=cfgdir)

def test_missing_checkpoint_rejected(tmp_path):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path)
    args.weights = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        m.build_protocol(args, config_dir=cfgdir)

def test_existing_run_is_not_overwritten(tmp_path):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path)
    (tmp_path / "runs" / "test").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        m.build_protocol(args, config_dir=cfgdir)

def test_mismatched_factor_metadata_rejected(tmp_path):
    m = load_runner()
    args, cfgdir = arguments(m, tmp_path)
    (cfgdir / "hfc.yaml").write_text(yaml.safe_dump(dict(nc=7,weather_hfc=dict(haar=False,freq=True,consistency=True))))
    with pytest.raises(ValueError):
        m.build_protocol(args, config_dir=cfgdir)

@pytest.mark.parametrize("data",[dict(nc=1,names=["other"]),dict(nc=7,names=["other"]),dict(nc=7,names={i+1:str(i) for i in range(7)})])
def test_wrong_data_class_metadata_rejected(tmp_path,data):
    m=load_runner()
    args,cfgdir=arguments(m,tmp_path)
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data),encoding="utf-8")
    with pytest.raises(ValueError):
        m.build_protocol(args,config_dir=cfgdir)

