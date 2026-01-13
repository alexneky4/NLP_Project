import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

import torch


def plot_test_result_matrix(
    y_true: List[bool],
    y_pred: List[bool],
    output_path: str = "track_a_confusion_matrix.png",
):
    assert len(y_true) == len(y_pred)

    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))

    total = len(y_true)

    matrix = np.array([
        [tp, fn],
        [fp, tn]
    ], dtype=float)

    matrix_perc = 100 * matrix / total

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix_perc, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred: A closer", "Pred: B closer"])
    ax.set_yticklabels(["True: A closer", "True: B closer"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{matrix_perc[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=12
            )

    ax.set_title("Track A – Test Result Matrix (%)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("[INFO] Confusion matrix saved to:", output_path)
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


def inspect_near_miss_errors(
    records: List[Dict],
    max_examples: int = 5,
    diff_threshold: float = 0.05,
):
    near_misses = [
        r for r in records
        if r["label"] != r["pred"]
        and abs(r["score_diff"]) <= diff_threshold
    ]

    if not near_misses:
        print("[INFO] No near-miss errors found.")
        return

    print(
        f"[INFO] Showing up to {max_examples} near-miss errors "
        f"( |score_diff| ≤ {diff_threshold} )\n"
    )

    for i, r in enumerate(near_misses[:max_examples], 1):
        print("=" * 80)
        print(f"Example {i}")
        print(f"Score difference (A - B): {r['score_diff']:.4f}")
        print(f"Ground truth: {'A closer' if r['label'] else 'B closer'}")
        print(f"Prediction : {'A closer' if r['pred'] else 'B closer'}")
        print("\nANCHOR:\n", r["anchor"])
        print("\nTEXT A:\n", r["text_a"])
        print("\nTEXT B:\n", r["text_b"])
        print()


def write_track_a_submission(
    data: List[Dict],
    predictions: List[bool],
    output_path: str = "track_a.jsonl",
):
    assert len(data) == len(predictions), (
        "Data and predictions must have the same length"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for item, pred in zip(data, predictions):
            out = {
                "anchor_text": item["anchor_text"],
                "text_a": item["text_a"],
                "text_b": item["text_b"],
                "text_a_is_closer": bool(pred),
            }
            f.write(json.dumps(out) + "\n")

    print(f"[INFO] Track A submission file written to: {output_path}")
