import json
from pathlib import Path
from typing import List, Dict, Literal


class NarrativeSimilarityDataset:
    def __init__(self, data_path: str, track: Literal["A", "B"]):
        self.data_path = Path(data_path)
        self.track = track.upper()

        if self.track not in {"A", "B"}:
            raise ValueError("track must be 'A' or 'B'")

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        if self.data_path.suffix == ".jsonl":
            return self._load_jsonl()
        elif self.data_path.suffix == ".json":
            return self._load_json()
        else:
            raise ValueError("Unsupported file format. Use .json or .jsonl")

    def _load_jsonl(self) -> List[Dict]:
        data = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_json(self) -> List[Dict]:
        with self.data_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def get_triplet(self, idx: int):
        if self.track != "A":
            raise RuntimeError("get_triplet is only valid for Track A")

        item = self.data[idx]
        return {
            "anchor": item["anchor_text"],
            "text_a": item["text_a"],
            "text_b": item["text_b"],
            "label": item["text_a_is_closer"]
        }

    def get_text(self, idx: int) -> str:
        if self.track != "B":
            raise RuntimeError("get_text is only valid for Track B")

        return self.data[idx]["text"]
