import os
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Custom imports
from datasets_loader import get_train_data_a
from utils import plot_test_result_matrix, inspect_near_miss_errors

# ----------------------------
# MODEL CONFIG
# ----------------------------

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_DIR = "models/mistral_7b_instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFUSION_MATRIX_PATH = "mistral_track_a_confusion_matrix.png"

USE_WEIGHTED_APPROACH = True

W_THEME = 0.4
W_ACTION = 0.3
W_OUTCOME = 0.3


# ----------------------------
# MODEL LOADING
# ----------------------------

def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR
    )

    # Mistral does not define a pad token by default
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        device_map="auto" if DEVICE == "cuda" else None,
        quantization_config=bnb_config if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    model.eval()
    return tokenizer, model


# ----------------------------
# INFERENCE
# ----------------------------

def get_mistral_prediction(tokenizer, model, prompt: str, max_new_tokens: int = 100):
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()


# ----------------------------
# PARSERS
# ----------------------------

def parse_simple_prediction(output: str) -> bool | None:
    output = output.upper()
    if re.search(r"\bA\b", output):
        return True
    if re.search(r"\bB\b", output):
        return False
    return None


def parse_weighted_prediction(output: str):
    def extract(key, text):
        match = re.search(fr"{key}:\s*(\d+)", text, re.IGNORECASE)
        return float(match.group(1)) if match else 5.0

    a_scores = [
        extract("A_Theme", output),
        extract("A_Action", output),
        extract("A_Outcome", output),
    ]
    b_scores = [
        extract("B_Theme", output),
        extract("B_Action", output),
        extract("B_Outcome", output),
    ]

    score_a = (
        a_scores[0] * W_THEME
        + a_scores[1] * W_ACTION
        + a_scores[2] * W_OUTCOME
    )
    score_b = (
        b_scores[0] * W_THEME
        + b_scores[1] * W_ACTION
        + b_scores[2] * W_OUTCOME
    )

    return (score_a >= score_b), (score_a - score_b)


# ----------------------------
# MAIN LOOP
# ----------------------------

if __name__ == "__main__":
    print(f"[INFO] Using Approach: {'WEIGHTED' if USE_WEIGHTED_APPROACH else 'SIMPLE'}")

    data = get_train_data_a(use_synthetic=False)
    tokenizer, model = load_model()

    y_true, y_pred, records = [], [], []

    for item in tqdm(data):
        anchor = item["anchor_text"]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["text_a_is_closer"]

        if USE_WEIGHTED_APPROACH:
            prompt = f"""
Compare stories A and B to the Reference. Score 1-10 (1=opposite, 10=identical) for:
1. Abstract Theme: Ideas/motives.
2. Course of Action: Plot sequence.
3. Outcomes: Results.

Reference: {anchor}
Story A: {text_a}
Story B: {text_b}

Format:
A_Theme: [score]
A_Action: [score]
A_Outcome: [score]
B_Theme: [score]
B_Action: [score]
B_Outcome: [score]
"""
            output = get_mistral_prediction(tokenizer, model, prompt, max_new_tokens=100)
            pred, diff = parse_weighted_prediction(output)
        else:
            prompt = f"""
Analyze these three story summaries.

Anchor: {anchor}
Choice A: {text_a}
Choice B: {text_b}

Which choice is more narratively similar to the Anchor?
Answer with ONLY 'A' or 'B'.
"""
            output = get_mistral_prediction(tokenizer, model, prompt, max_new_tokens=10)
            pred = parse_simple_prediction(output)
            diff = 0.0

        if pred is None:
            continue

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
