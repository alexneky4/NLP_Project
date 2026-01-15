import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datasets_loader import get_train_data_a
from utils import plot_test_result_matrix, inspect_near_miss_errors

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
USE_SYNTHETIC = False
OUTPUT_CONFUSION_MATRIX = "sbert_track_a_confusion_matrix.png"


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]


if __name__ == "__main__":
    print("[INFO] Loading Track A data...")
    data = get_train_data_a(use_synthetic=USE_SYNTHETIC)
    print(f"[INFO] Loaded {len(data)} triplets")

    print("[INFO] Loading SBERT model...")
    model = SentenceTransformer(MODEL_NAME)

    y_true = []
    y_pred = []
    records = []

    print("[INFO] Running SBERT baseline evaluation...")

    for item in tqdm(data):
        anchor = item["anchor_text"]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["text_a_is_closer"]

        emb_anchor = model.encode(anchor, convert_to_numpy=True)
        emb_a = model.encode(text_a, convert_to_numpy=True)
        emb_b = model.encode(text_b, convert_to_numpy=True)

        sim_a = cos_sim(emb_anchor, emb_a)
        sim_b = cos_sim(emb_anchor, emb_b)

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
    print(f"[RESULT] SBERT accuracy: {accuracy:.4f}")

    plot_test_result_matrix(
        y_true,
        y_pred,
        output_path=OUTPUT_CONFUSION_MATRIX
    )

    inspect_near_miss_errors(
        records,
        max_examples=5,
        diff_threshold=0.03
    )
