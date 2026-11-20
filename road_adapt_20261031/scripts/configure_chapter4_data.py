"""Build the frozen Chapter 4 input lists/configs without changing dataset files.

Uses only the Python standard library. Re-running is idempotent. Existing files
with different bytes are rejected before any write unless --force is supplied.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("D:/Users/86186/ResearchData/SVRDD-weather-v1")
MANIFEST_SHA256 = "748b8851e80b87195866b19d80c26f5d2789430b74711e0abc5405c9afa8951f"
SELECTION_SHA256 = "172f75da98920067a499265b88da78fe6af03c956b35598d0b339ba7ac4193a2"
SELECTION_SEED = "20261118-validation-v2"
WEATHER = ("fog", "rain", "lowlight")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_names(path):
    """Read the known source YAML schema, preserving names and integer IDs."""
    text = path.read_text(encoding="utf-8-sig")
    if not re.search(r"^nc:\s*7\s*$", text, re.MULTILINE):
        raise ValueError("Source data YAML must declare nc: 7")
    names = {}
    in_names = False
    for line in text.splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names or not line.strip():
            continue
        match = re.fullmatch(r"\s+(\d+):\s*(.+?)\s*", line)
        if not match:
            raise ValueError("Unsupported source names schema; expected IDs 0..6")
        key, value = int(match[1]), match[2]
        if key in names:
            raise ValueError("Duplicate source class ID")
        if value.startswith('"'):
            value = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        names[key] = value
    if set(names) != set(range(7)):
        raise ValueError("Source data YAML must define exactly classes 0..6")
    return names


def load_manifest(root):
    raw = (root / "manifest.jsonl").read_bytes()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError("Manifest differs from the approved Chapter 4 dataset")
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
    index = {}
    sources = defaultdict(set)
    source_hashes = defaultdict(set)
    outputs = set()
    counts = Counter()
    for row in rows:
        split, oid = row["split"], row["original_id"]
        key = (split, oid, row["condition"], row["severity"])
        if key in index or row["output_image"] in outputs:
            raise ValueError("Duplicate manifest key or image path")
        for field in ("output_image", "output_label"):
            relative = Path(row[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest output paths must stay within data root")
        index[key] = row
        outputs.add(row["output_image"])
        sources[split].add(oid)
        source_hashes[split].add(row["source_sha256"])
        counts[(split, row["condition"], row["severity"])] += 1
    if {s: len(v) for s, v in sources.items()} != {"train": 6000, "valid": 1000, "test": 1000}:
        raise ValueError("Unexpected original-source split counts")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if sources[left] & sources[right] or source_hashes[left] & source_hashes[right]:
            raise ValueError("Cross-split source leakage")
    expected = {}
    for split, n in (("train", 6000), ("valid", 1000), ("test", 1000)):
        expected[(split, "clear", 0)] = n
        for condition in WEATHER:
            for severity in (1, 2, 3):
                expected[(split, condition, severity)] = 2000 if split == "train" else 1000
    if counts != Counter(expected) or len(rows) != 44000:
        raise ValueError("Unexpected condition/severity counts")
    return rows, index, sources


def make_lists(rows, index, sources):
    lists = {"train_mixed.txt": [r for r in rows if r["split"] == "train"]}
    order = sorted(sources["valid"], key=lambda oid: digest((SELECTION_SEED + "|" + oid).encode()))
    selected = []
    for position, oid in enumerate(order):
        levels = {"clear": 0, "fog": 1 if position < 500 else 2,
                  "rain": 2 if position < 500 else 1,
                  "lowlight": 1 if position < 500 else 2}
        selected.extend(index[("valid", oid, condition, levels[condition])]
                        for condition in ("clear",) + WEATHER)
    lists["valid_selection_v2.txt"] = selected
    # Keep component order explicit: fog s1/s2, rain s1/s2, lowlight s1/s2.
    for condition in WEATHER:
        lists[f"test_{condition}_2000.txt"] = [
            r for severity in (1, 2) for r in rows
            if (r["split"], r["condition"], r["severity"]) == ("test", condition, severity)]
    lists["test_primary_6000.txt"] = [r for condition in WEATHER
                                            for r in lists[f"test_{condition}_2000.txt"]]
    lists["test_clear_1000.txt"] = [r for r in rows if r["split"] == "test" and r["condition"] == "clear"]
    lists["test_severe_3000.txt"] = [r for condition in WEATHER for r in rows
                                     if (r["split"], r["condition"], r["severity"]) == ("test", condition, 3)]
    lists["timing_valid_200.txt"] = [r for position in list(range(25)) + list(range(500, 525))
                                    for r in selected[position * 4:position * 4 + 4]]
    return lists


def yaml_bytes(root, output, names, test):
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    fields = {"path": root, "train": output / "train_mixed.txt",
              "val": output / "valid_selection_v2.txt", "test": output / test}
    text = "\n".join(f"{key}: {quote(value)}" for key, value in fields.items())
    text += "\nnc: 7\nnames:\n" + "\n".join(f"  {key}: {quote(names[key])}" for key in range(7)) + "\n"
    return text.encode("utf-8")


def build_artifacts(root, output):
    rows, index, sources = load_manifest(root)
    source_yaml = root / "data_weather_train.yaml"
    names = read_names(source_yaml)
    lists = make_lists(rows, index, sources)
    artifacts, records = {}, {}
    for name, selected in lists.items():
        # CRLF is intentional: preserves the already approved external v2 bytes.
        content = ("\r\n".join(str(root / r["output_image"]) for r in selected) + "\r\n").encode("utf-8")
        artifacts[name] = content
        counts = Counter(f"{r['condition']}_s{r['severity']}" for r in selected)
        records[name] = dict(path=str(output / name), images=len(selected),
                             source_ids=len({r["original_id"] for r in selected}),
                             counts=dict(sorted(counts.items())), sha256=digest(content))
    selected = lists["valid_selection_v2.txt"]
    content = ("\r\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in selected) + "\r\n").encode("utf-8")
    if digest(content) != SELECTION_SHA256:
        raise ValueError("Rebuilt v2 JSONL differs from approved selection")
    artifacts["valid_selection_v2.jsonl"] = content
    records["valid_selection_v2.jsonl"] = dict(records["valid_selection_v2.txt"],
                                              path=str(output / "valid_selection_v2.jsonl"), sha256=digest(content))
    evaluations = {"dataset.yaml": "test_primary_6000.txt",
                   "test_primary.yaml": "test_primary_6000.txt",
                   "test_clear.yaml": "test_clear_1000.txt",
                   "test_severe.yaml": "test_severe_3000.txt",
                   **{f"test_{c}.yaml": f"test_{c}_2000.txt" for c in WEATHER}}
    for name, test in evaluations.items():
        content = yaml_bytes(root, output, names, test)
        artifacts[name] = content
        records[name] = dict(path=str(output / name), sha256=digest(content),
                             train="train_mixed.txt", val="valid_selection_v2.txt", test=test)
    provenance = dict(protocol="chapter4-svrdd-weather-v2", status="configured_not_trained",
                      data_root=str(root), output_root=str(output),
                      manifest=dict(path=str(root / "manifest.jsonl"), sha256=MANIFEST_SHA256, records=44000),
                      source_data_yaml=dict(path=str(source_yaml), sha256=digest(source_yaml.read_bytes())),
                      generator_sha256=digest(Path(__file__).read_bytes()),
                      selection=dict(seed=SELECTION_SEED, jsonl_sha256=SELECTION_SHA256,
                                     validation_only=True, no_performance_selection=True,
                                     rule="SHA256(seed|source_id) order; first/last 500 complementary mild/moderate assignments"),
                      names=names, files=records,
                      notes=["Image and label files remain in the read-only source dataset.",
                             "Primary AP is evaluated jointly over 6000 mild/moderate images, not averaged APs.",
                             "Test groups share 1000 source images; variants are not independent sources.",
                             "Use split=test for test YAMLs; their val remains the fixed selection set.",
                             "No model training, timing or performance evaluation is performed by this script."])
    artifacts["provenance.json"] = (json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return artifacts, provenance


def configure(root, output, force=False):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the read-only dataset root")
    artifacts, provenance = build_artifacts(root, output)
    different = [name for name, content in artifacts.items()
                 if (output / name).exists() and (not (output / name).is_file()
                                                  or (output / name).read_bytes() != content)]
    if different and not force:
        raise FileExistsError("Refusing to overwrite different files; inspect or use --force: " + ", ".join(different))
    if any((output / name).exists() and not (output / name).is_file() for name in artifacts):
        raise ValueError("An output file path is occupied by a directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = output / name
        if not path.is_file() or path.read_bytes() != content:
            path.write_bytes(content)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "configs/chapter4")
    parser.add_argument("--force", action="store_true", help="Replace differing generated files after review")
    args = parser.parse_args(argv)
    try:
        result = configure(args.data_root, args.output, args.force)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Configuration failed: {exc}\n")
    print(json.dumps(dict(dataset=str(args.output.resolve() / "dataset.yaml"),
                          images={name: info["images"] for name, info in result["files"].items() if "images" in info},
                          validation_jsonl_sha256=result["selection"]["jsonl_sha256"],
                          status=result["status"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
