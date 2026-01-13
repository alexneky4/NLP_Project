"""
Track B: Full pipeline for generating story embeddings.
1. Fine-tunes SBERT on Track A triplets (if model doesn't exist)
2. Generates embeddings for Track B stories
3. Visualizes embeddings with UMAP
"""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import umap

import sys
sys.path.append("..")
from datasets_loader import get_train_data_a, get_train_data_b, get_test_data_b

# Config
BASE_MODEL = "BAAI/bge-base-en-v1.5"
FINETUNED_MODEL_DIR = Path("models/bge_finetuned")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")

BATCH_SIZE = 6  # Slightly increased, still safe for 6GB GPU
EPOCHS = 10
SEED = 42


# === Fine-tuning ===

def build_triplet_samples_with_hard_negatives(data: list[dict]) -> list[InputExample]:
    """Convert Track A data into (anchor, positive, hard_negative) triplets."""
    samples = []
    for item in data:
        anchor = (item.get("anchor_text") or "").strip()
        text_a = (item.get("text_a") or "").strip()
        text_b = (item.get("text_b") or "").strip()
        a_is_closer = item.get("text_a_is_closer")

        if not all([anchor, text_a, text_b]):
            continue

        # (anchor, positive, hard_negative)
        if a_is_closer:
            samples.append(InputExample(texts=[anchor, text_a, text_b]))
        else:
            samples.append(InputExample(texts=[anchor, text_b, text_a]))

    return samples


def fine_tune_model() -> SentenceTransformer:
    """Fine-tune SBERT with MultipleNegativesRankingLoss + hard negatives."""
    print("[INFO] Loading Track A data for fine-tuning...")
    data = get_train_data_a(base_dir="..", use_synthetic=True)
    random.shuffle(data)

    train_samples = build_triplet_samples_with_hard_negatives(data)
    print(f"[INFO] Created {len(train_samples)} triplet samples (with hard negatives)")

    train_loader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # MNRL with hard negatives: (anchor, positive, negative) format
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(0.1 * len(train_loader) * EPOCHS)

    print(f"[INFO] Fine-tuning for {EPOCHS} epochs (batch={BATCH_SIZE}, loss=MNRL+hard_negatives)...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        optimizer_params={"lr": 2e-5},
        use_amp=True,  # Mixed precision to save memory
    )

    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINETUNED_MODEL_DIR))
    print(f"[INFO] Model saved to: {FINETUNED_MODEL_DIR}")

    return model


def load_or_train_model() -> SentenceTransformer:
    """Load fine-tuned model if exists, otherwise train it."""
    if FINETUNED_MODEL_DIR.exists():
        print(f"[INFO] Loading existing model from {FINETUNED_MODEL_DIR}")
        return SentenceTransformer(str(FINETUNED_MODEL_DIR))
    else:
        print("[INFO] No fine-tuned model found, training...")
        return fine_tune_model()


# === Embedding generation ===

def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode texts into embeddings."""
    return model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)


def save_embeddings(texts: list[str], embeddings: np.ndarray, name: str):
    """Save embeddings as JSONL and NPY."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL for submission
    jsonl_path = OUTPUT_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for text, emb in zip(texts, embeddings):
            f.write(json.dumps({"text": text, "embedding": emb.tolist()}) + "\n")
    print(f"[INFO] Saved {jsonl_path}")

    # NPY for analysis
    npy_path = OUTPUT_DIR / f"{name}.npy"
    np.save(npy_path, embeddings)
    print(f"[INFO] Saved {npy_path}")


# === Visualization ===

def visualize_embeddings(embeddings: np.ndarray, name: str):
    """Create UMAP scatter plot of embeddings."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running UMAP on {name} embeddings...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=30, c="steelblue")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Track B {name.title()} Embeddings (UMAP)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"{name}_embeddings_umap.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot: {plot_path}")


# === Evaluation on Track A triplets ===

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_on_track_a(model: SentenceTransformer, data: list[dict], output_path: str):
    """
    Evaluate Track B embeddings using Track A triplets.
    For each triplet, check if cosine similarity correctly predicts which text is closer.
    """
    y_true = []
    y_pred = []
    
    print("[INFO] Evaluating embeddings on Track A triplets...")
    
    for item in data:
        anchor = (item.get("anchor_text") or "").strip()
        text_a = (item.get("text_a") or "").strip()
        text_b = (item.get("text_b") or "").strip()
        label = item.get("text_a_is_closer")
        
        if not all([anchor, text_a, text_b]) or label is None:
            continue
        
        # Encode independently (as required by Track B rules)
        emb_anchor = model.encode(anchor, convert_to_numpy=True)
        emb_a = model.encode(text_a, convert_to_numpy=True)
        emb_b = model.encode(text_b, convert_to_numpy=True)
        
        # Predict based on cosine similarity
        sim_a = cos_sim(emb_anchor, emb_a)
        sim_b = cos_sim(emb_anchor, emb_b)
        pred = sim_a > sim_b
        
        y_true.append(label)
        y_pred.append(pred)
    
    # Compute accuracy
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"[RESULT] Track B embeddings accuracy on Track A: {accuracy:.4f}")
    
    # Plot confusion matrix
    plot_track_b_confusion_matrix(y_true, y_pred, output_path)
    
    return accuracy


def plot_track_b_confusion_matrix(y_true: list, y_pred: list, output_path: str):
    """Plot confusion matrix for Track B evaluation."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    
    total = len(y_true)
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    matrix_perc = 100 * matrix / total
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix_perc, cmap="Blues")
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: A closer", "Pred: B closer"])
    ax.set_yticklabels(["True: A closer", "True: B closer"])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix_perc[i, j]:.1f}%",
                    ha="center", va="center", color="black", fontsize=12)
    
    ax.set_title("Track B Embeddings – Evaluation on Track A (%)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to: {output_path}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


# === Main ===

if __name__ == "__main__":
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Step 1: Load or train model
    model = load_or_train_model()

    # Step 2: Load Track B data
    print("[INFO] Loading Track B data...")
    train_data = get_train_data_b(base_dir="..")
    test_data = get_test_data_b(base_dir="..")

    train_texts = [item["text"] for item in train_data]
    test_texts = [item["text"] for item in test_data]

    print(f"[INFO] Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Step 3: Generate embeddings
    print("[INFO] Generating embeddings...")
    train_emb = generate_embeddings(model, train_texts)
    test_emb = generate_embeddings(model, test_texts)

    print(f"[INFO] Embedding dim: {train_emb.shape[1]}")

    # Step 4: Save
    save_embeddings(train_texts, train_emb, "train")
    save_embeddings(test_texts, test_emb, "track_b")

    # Step 5: Visualize
    visualize_embeddings(train_emb, "train")
    visualize_embeddings(test_emb, "test")

    # Step 6: Evaluate on Track A dev data (test data has no labels)
    print("[INFO] Evaluating on Track A dev triplets...")
    dev_data_a = get_train_data_a(base_dir="..", use_synthetic=False)
    evaluate_on_track_a(model, dev_data_a, str(PLOTS_DIR / "track_b_confusion_matrix.png"))

    print("[INFO] Done!")
