import re
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from datasets_loader import get_train_data_a, get_train_data_b


def count_tokens(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    # simple, robust sentence split
    return len([s for s in re.split(r"[.!?]\s+", text) if s.strip()])


def plot_text_lengths(texts, output_prefix="all_texts"):
    token_counts = [count_tokens(t) for t in texts]
    sentence_counts = [count_sentences(t) for t in texts]

    # Tokens
    plt.figure(figsize=(6, 4))
    plt.hist(token_counts, bins=50)
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.title("Token count distribution")
    plt.tight_layout()
    plt.savefig(f"plots/{output_prefix}_token_distribution.png", dpi=300)
    plt.close()

    # Sentences
    plt.figure(figsize=(6, 4))
    plt.hist(sentence_counts, bins=30)
    plt.xlabel("Number of sentences")
    plt.ylabel("Frequency")
    plt.title("Sentence count distribution")
    plt.tight_layout()
    plt.savefig(f"plots/{output_prefix}_sentence_distribution.png", dpi=300)
    plt.close()

    print(f"[INFO] Saved length plots with prefix '{output_prefix}'")


def plot_track_a_label_distribution(data_a):
    labels = [item["text_a_is_closer"] for item in data_a]
    counts = Counter(labels)

    total = sum(counts.values())
    perc_true = 100 * counts.get(True, 0) / total
    perc_false = 100 * counts.get(False, 0) / total

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Text A closer", "Text B closer"],
        [perc_true, perc_false]
    )
    plt.ylabel("Percentage (%)")
    plt.title("Track A label distribution")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("plots/track_a_label_distribution.png", dpi=300)
    plt.close()

    print("[INFO] Saved Track A label distribution plot")
    print(f"Text A closer: {perc_true:.2f}%")
    print(f"Text B closer: {perc_false:.2f}%")


if __name__ == "__main__":
    data_a = get_train_data_a(use_synthetic=True)
    data_b = get_train_data_b()

    all_texts = []

    for item in data_a:
        for key in ("anchor_text", "text_a", "text_b"):
            text = item.get(key)
            if isinstance(text, str) and text.strip():
                all_texts.append(text)

    for item in data_b:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            all_texts.append(text)

    print(f"[INFO] Total number of texts: {len(all_texts)}")

    plot_text_lengths(all_texts, output_prefix="all_texts")

    plot_track_a_label_distribution(data_a)
