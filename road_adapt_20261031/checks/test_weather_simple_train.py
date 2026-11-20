"""Exercise the editable server entrypoint without training or dataset pixels."""
import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint(monkeypatch):
    """Importing the entrypoint must not load training code or inspect storage."""
    original_import = builtins.__import__
    deferred = {"chapter4_server", "train_weather_hfc", "torch", "ultralytics", "yaml"}

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in deferred, "training imports must be lazy"
        return original_import(name, *args, **kwargs)

    def no_storage(*args, **kwargs):
        raise AssertionError("entrypoint import must not inspect data or create runs")

    spec = importlib.util.spec_from_file_location("weather_simple_train_under_test", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        for method in ("open", "read_text", "read_bytes", "exists", "is_file", "mkdir"):
            guard.setattr(Path, method, no_storage)
        spec.loader.exec_module(module)
    return module


def install_dependencies(monkeypatch, tmp_path, *, fail_prepare=False):
    """Only the slow data audit and real training dispatch are replaced."""
    events = []
    dataset = tmp_path / "generated" / "dataset.yaml"

    def prepare(data_root, weights, output):
        events.append(("prepare", data_root, weights, output))
        if fail_prepare:
            raise ValueError("input check failed")
        return dataset

    def delegate(argv):
        events.append(("delegate", argv))
        return "delegated"

    monkeypatch.setitem(sys.modules, "chapter4_server", SimpleNamespace(prepare_inputs=prepare))
    monkeypatch.setitem(sys.modules, "train_weather_hfc", SimpleNamespace(main=delegate))
    return events, dataset


def options(argv):
    return {key: argv[i + 1] for i, key in enumerate(argv)
            if key.startswith("--") and key not in ("--amp", "--train")}


def test_import_is_inert(monkeypatch):
    load_entrypoint(monkeypatch)


def test_plain_python_routes_formal_baseline_after_preparation(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    events, dataset = install_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(module, "PROJECT", tmp_path / "formal-runs")

    assert module.main([]) == "delegated"

    assert events[0] == (
        "prepare", Path("/root/autodl-tmp/SVRDD-weather-v1"),
        Path("/root/autodl-tmp/SVRDD-baseline/weights/best.pt"), ROOT / "configs/chapter4",
    )
    assert [event[0] for event in events] == ["prepare", "delegate"]
    argv = events[1][1]
    assert options(argv) == {
        "--configuration": "baseline", "--data": str(dataset),
        "--manifest": str(Path("/root/autodl-tmp/SVRDD-weather-v1/manifest.jsonl")),
        "--weights": str(Path("/root/autodl-tmp/SVRDD-baseline/weights/best.pt")),
        "--seed": "0", "--batch": "8", "--workers": "0", "--device": "0",
        "--epochs": "300", "--imgsz": "640", "--project": str(tmp_path / "formal-runs"),
        "--name": "baseline_seed0_b8_amp",
    }
    assert argv.count("--amp") == 1
    assert argv.count("--train") == 1


@pytest.mark.parametrize("configuration", ["baseline", "h", "f", "c", "hf", "hc", "fc", "hfc"])
def test_every_ablation_uses_original_initialization(monkeypatch, tmp_path, configuration):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    original = tmp_path / "seven-class-original" / "best.pt"
    monkeypatch.setattr(module, "CONFIGURATION", configuration)
    monkeypatch.setattr(module, "ORIGINAL_WEIGHTS", original)
    module.main([])

    argv = events[-1][1]
    parsed = options(argv)
    assert parsed["--configuration"] == configuration
    assert parsed["--weights"] == str(original)
    assert parsed["--name"] == f"{configuration}_seed0_b8_amp"
    # C derives its fixed teacher from these same weights in train_weather_hfc.
    assert "--teacher" not in argv


def test_check_prepares_and_previews_without_starting_training(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    project = tmp_path / "no-formal-runs"
    monkeypatch.setattr(module, "PROJECT", project)
    module.main(["--check"])

    assert [event[0] for event in events] == ["prepare", "delegate"]
    assert "--train" not in events[-1][1]
    assert "--check" not in events[-1][1]
    assert not project.exists()


def test_editable_paths_and_settings_reach_the_weather_runner(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    values = dict(DATA_ROOT=tmp_path / "weather elsewhere", ORIGINAL_WEIGHTS=tmp_path / "clear.pt",
                  PROJECT=tmp_path / "results", SEED=2, BATCH=2, AMP=False, WORKERS=4,
                  DEVICE="1", EPOCHS=120, IMGSZ=768)
    for key, value in values.items():
        monkeypatch.setattr(module, key, value)
    module.main(["--check"])

    assert events[0][1:3] == (values["DATA_ROOT"], values["ORIGINAL_WEIGHTS"])
    argv = events[-1][1]
    parsed = options(argv)
    assert parsed["--manifest"] == str(values["DATA_ROOT"] / "manifest.jsonl")
    assert parsed["--weights"] == str(values["ORIGINAL_WEIGHTS"])
    assert parsed["--project"] == str(values["PROJECT"])
    assert {key: parsed[key] for key in ("--seed", "--batch", "--workers", "--device", "--epochs", "--imgsz", "--name")} == {
        "--seed": "2", "--batch": "2", "--workers": "4", "--device": "1",
        "--epochs": "120", "--imgsz": "768", "--name": "baseline_seed2_b2_fp32",
    }
    assert "--amp" not in argv


@pytest.mark.parametrize("check", [False, True])
def test_resume_keeps_original_record_and_skips_regeneration(monkeypatch, tmp_path, capsys, check):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    checkpoint = tmp_path / "existing" / "last.pt"
    monkeypatch.setattr(module, "RESUME_CHECKPOINT", checkpoint)
    # These edits must never override the resumed protocol.
    monkeypatch.setattr(module, "CONFIGURATION", "hfc")
    monkeypatch.setattr(module, "BATCH", 999)
    module.main(["--check"] if check else [])

    assert events == [("delegate", ["--resume", str(checkpoint)] + ([] if check else ["--train"]))]
    assert "weather_protocol.json" in capsys.readouterr().out


def test_failed_input_check_never_dispatches_training(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path, fail_prepare=True)
    with pytest.raises(ValueError, match="input check failed"):
        module.main([])
    assert [event[0] for event in events] == ["prepare"]


def test_only_check_is_accepted_on_the_command_line(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as error:
        module.main(["--configuration", "hfc"])
    assert error.value.code == 2
    assert events == []


def test_generated_arguments_match_the_real_weather_cli(monkeypatch, tmp_path):
    module = load_entrypoint(monkeypatch)
    # Keep the real parser to catch unsupported forwarding options.
    spec = importlib.util.spec_from_file_location("weather_cli_contract", ROOT / "train_weather_hfc.py")
    weather = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(ROOT))
    spec.loader.exec_module(weather)
    events, _ = install_dependencies(monkeypatch, tmp_path)
    module.main([])
    parsed = weather.make_parser().parse_args(events[-1][1])
    assert parsed.configuration == "baseline"
    assert parsed.batch == 8 and parsed.amp and parsed.train
