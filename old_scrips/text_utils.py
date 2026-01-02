import re
import numpy as np


def split_sentences(text: str):
    # simple, robust sentence split (no extra deps)
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


def outcome_text(text: str, ratio: float = 0.25):
    words = text.split()
    if len(words) == 0:
        return text
    start = int(len(words) * (1 - ratio))
    return " ".join(words[start:])


def mean_pool(embeddings: np.ndarray):
    return embeddings.mean(axis=0)
