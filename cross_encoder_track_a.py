import random
import torch
from pathlib import Path
import json
from typing import List

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from datasets_loader import get_train_data_a
from utils import plot_test_result_matrix, inspect_near_miss_errors

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR = Path("models/cross_encoder_track_a")

BATCH_SIZE = 16
EPOCHS = 4
TRAIN_SPLIT = 0.8
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_pairwise_samples(data):
    samples = []

    skipped = 0

    for item in data:
        anchor = item.get("anchor_text")
        text_a = item.get("text_a")
        text_b = item.get("text_b")
        label = item.get("text_a_is_closer")

        if not all(isinstance(x, str) and x.strip() for x in [anchor, text_a, text_b]):
            skipped += 1
            continue

        anchor = anchor.strip()
        text_a = text_a.strip()
        text_b = text_b.strip()

        if label:
            samples.append(InputExample(texts=[anchor, text_a], label=1.0))
            samples.append(InputExample(texts=[anchor, text_b], label=0.0))
        else:
            samples.append(InputExample(texts=[anchor, text_a], label=0.0))
            samples.append(InputExample(texts=[anchor, text_b], label=1.0))

    if skipped > 0:
        print(f"[WARNING] Skipped {skipped} malformed training examples")

    return samples


def fine_tune_cross_encoder(
    train_data,
    model_name: str,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    device: str,
):
    train_samples = build_pairwise_samples(train_data)

    train_loader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size,
    )

    model = CrossEncoder(
        model_name,
        num_labels=1,
        device=device,
    )
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=int(0.1 * len(train_loader)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    print(f"[INFO] Model saved to: {output_dir}")
    return model


def evaluate_cross_encoder(model, test_data):
    """
    Evaluate a Cross-Encoder on Track A test data,
    using confusion matrix and near-miss analysis.
    """
    y_true = []
    y_pred = []
    records = []

    for item in test_data:
        anchor = item["anchor_text"]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["text_a_is_closer"]

        scores = model.predict([
            [anchor, text_a],
            [anchor, text_b],
        ])

        sim_a, sim_b = scores[0], scores[1]
        pred = sim_a > sim_b

        y_true.append(label)
        y_pred.append(pred)

        records.append({
            "anchor": anchor,
            "text_a": text_a,
            "text_b": text_b,
            "label": label,
            "pred": pred,
            "score_diff": sim_a - sim_b,
        })

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"[RESULT] Test accuracy: {accuracy:.4f}")

    plot_test_result_matrix(
        y_true,
        y_pred,
        output_path="cross_encoder_track_a_confusion_matrix.png",
    )

    inspect_near_miss_errors(
        records,
        max_examples=5,
        diff_threshold=0.03,
    )

def write_dev_track_a_jsonl(
    model,
    dev_data,
    output_path: str = "track_a.jsonl",
):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dev_data:
            anchor = item["anchor_text"]
            text_a = item["text_a"]
            text_b = item["text_b"]

            scores = model.predict([
                [anchor, text_a],
                [anchor, text_b],
            ])

            pred = bool(scores[0] > scores[1])

            out = {
                "anchor_text": anchor,
                "text_a": text_a,
                "text_b": text_b,
                "text_a_is_closer": pred,
            }

            f.write(json.dumps(out) + "\n")

    print(f"[INFO] Dev Track A JSONL written to: {output_path}")



if __name__ == "__main__":
    print("[INFO] Using device:", DEVICE)
    random.seed(SEED)

    print("[INFO] Loading Track A data...")
    data = get_train_data_a(use_synthetic=True)
    random.shuffle(data)

    split_idx = int(TRAIN_SPLIT * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"[INFO] Train samples: {len(train_data)}")
    print(f"[INFO] Test samples : {len(test_data)}")

    model = fine_tune_cross_encoder(
        train_data=train_data,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
    )

    print("[INFO] Evaluating model...")
    evaluate_cross_encoder(model, test_data)

    # print("[INFO] Loading trained Cross-Encoder...")
    # model = CrossEncoder(str(OUTPUT_DIR), device=DEVICE)
    #
    # print("[INFO] Loading dev Track A data (order preserved)...")
    # from datasets_loader import get_dev_data_a
    # dev_data = get_dev_data_a()
    #
    # print("[INFO] Writing dev Track A JSONL file...")
    # write_dev_track_a_jsonl(
    #     model=model,
    #     dev_data=dev_data,
    #     output_path="track_a.jsonl",
    # )

