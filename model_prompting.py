import os
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets_loader import get_train_data_a
from utils import plot_test_result_matrix, inspect_near_miss_errors


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_DIR = "models/phi_3_mini"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFUSION_MATRIX_PATH = "phi3_weighted_track_a_confusion_matrix.png"

W_THEME = 0.4
W_ACTION = 0.3
W_OUTCOME = 0.3


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    model.eval()
    return tokenizer, model


def score_candidate(tokenizer, model, anchor: str, candidate: str):
    prompt = f"""
You are evaluating narrative similarity.

Reference story:
{anchor}

Candidate story:
{candidate}

Score the similarity from 1 to 10 (1 = very different, 10 = very similar) for:

Theme: underlying ideas and motives
Action: sequence of main events
Outcome: how the story concludes

Return ONLY the following format:

Theme: <number>
Action: <number>
Outcome: <number>
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )


def parse_weighted_score(output: str):
    def extract(key):
        match = re.search(fr"{key}:\s*(\d+)", output, re.IGNORECASE)
        return float(match.group(1)) if match else None

    theme = extract("Theme")
    action = extract("Action")
    outcome = extract("Outcome")

    if any(v is None for v in [theme, action, outcome]):
        return None

    return (
        theme * W_THEME +
        action * W_ACTION +
        outcome * W_OUTCOME
    )


if __name__ == "__main__":
    print("[INFO] Loading Phi-3 model...")
    tokenizer, model = load_model()

    print("[INFO] Loading Track A data...")
    data = get_train_data_a(use_synthetic=False)

    y_true, y_pred, records = [], [], []

    for item in tqdm(data):
        anchor = item["anchor_text"]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["text_a_is_closer"]

        out_a = score_candidate(tokenizer, model, anchor, text_a)
        print(out_a)
        score_a = parse_weighted_score(out_a)

        out_b = score_candidate(tokenizer, model, anchor, text_b)
        score_b = parse_weighted_score(out_b)

        if score_a is None or score_b is None:
            continue

        pred = score_a > score_b
        diff = score_a - score_b

        y_true.append(label)
        y_pred.append(pred)

        records.append({
            "anchor": anchor,
            "text_a": text_a,
            "text_b": text_b,
            "label": label,
            "pred": pred,
            "score_diff": diff,
        })

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"[RESULT] Accuracy: {accuracy:.4f}")

    plot_test_result_matrix(y_true, y_pred, output_path=CONFUSION_MATRIX_PATH)
    inspect_near_miss_errors(records, max_examples=5)
