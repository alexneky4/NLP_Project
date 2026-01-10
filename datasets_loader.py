import json
from pathlib import Path
from typing import List, Dict


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_train_data_a(
        base_dir: str = ".",
        use_synthetic: bool = False,
) -> List[Dict]:
    base = Path(base_dir)

    paths = [
        base / "development_data" / "dev_track_a.jsonl",
        base / "sample_data" / "sample_track_a.jsonl",
    ]

    data = []

    for path in paths:
        for item in _load_jsonl(path):
            data.append({
                "anchor_text": item["anchor_text"],
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "text_a_is_closer": item["text_a_is_closer"],
            })

    if use_synthetic:
        synth_path = (
                base
                / "synthetic_data_classification"
                / "synthetic_data_for_classification.jsonl"
        )

        for item in _load_jsonl(synth_path):
            # explicitly ignore `model_name`
            data.append({
                "anchor_text": item["anchor_text"],
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "text_a_is_closer": item["text_a_is_closer"],
            })

    return data


def get_train_data_b(
        base_dir: str = ".",
) -> List[Dict]:
    base = Path(base_dir)

    paths = [
        base / "development_data" / "dev_track_b.jsonl",
        base / "sample_data" / "sample_track_b.jsonl",
    ]

    data = []

    for path in paths:
        for item in _load_jsonl(path):
            data.append({
                "text": item["text"]
            })

    return data

def get_test_data_a(
        base_dir: str = ".",
) -> List[Dict]:
    base = Path(base_dir)

    path = base / "test_data" / "test_track_a.jsonl"

    data = []

    for item in _load_jsonl(path):
        data.append({
            "anchor_text": item["anchor_text"],
            "text_a": item["text_a"],
            "text_b": item["text_b"],
            "text_a_is_closer": item["text_a_is_closer"],
        })

    return data


def get_test_data_b(
        base_dir: str = ".",
) -> List[Dict]:
    base = Path(base_dir)

    path = base / "test_data" / "test_track_b.jsonl"

    data = []

    for item in _load_jsonl(path):
        data.append({
            "text": item["text"]
        })

    return data

def get_dev_data_a(
        base_dir: str = ".",
) -> List[Dict]:
    base = Path(base_dir)
    path = base / "development_data" / "dev_track_a.jsonl"

    data = []

    for item in _load_jsonl(path):
        data.append({
            "anchor_text": item["anchor_text"],
            "text_a": item["text_a"],
            "text_b": item["text_b"],
            "text_a_is_closer": item["text_a_is_closer"],
        })

    return data

