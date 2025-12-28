import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Initial Setup
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


def load_jsonl(file_path):
    """Loads a JSONL file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame()
    return pd.read_json(file_path, lines=True)


def get_jaccard(str1, str2):
    """Calculates lexical overlap (Jaccard Similarity)."""
    s1 = set(str(str1).lower().split())
    s2 = set(str(str2).lower().split())
    if not s1 or not s2: return 0
    return len(s1 & s2) / len(s1 | s2)


def run_narrative_analysis(track_a_path, track_b_path):
    # 1. Load Data
    df_a = load_jsonl(track_a_path)
    df_b = load_jsonl(track_b_path)

    # --- TRACK A ANALYSIS ---
    if not df_a.empty:
        # Calculate word counts
        for col in ['anchor_text', 'text_a', 'text_b']:
            df_a[f'{col}_len'] = df_a[col].apply(lambda x: len(str(x).split()))

        # Calculate Jaccard Similarity
        df_a['sim_a'] = df_a.apply(lambda r: get_jaccard(r['anchor_text'], r['text_a']), axis=1)
        df_a['sim_b'] = df_a.apply(lambda r: get_jaccard(r['anchor_text'], r['text_b']), axis=1)

        # Distinguish between 'Closer' and 'Farther' text for plotting
        df_a['closer_sim'] = df_a.apply(lambda r: r['sim_a'] if r['text_a_is_closer'] else r['sim_b'], axis=1)
        df_a['farther_sim'] = df_a.apply(lambda r: r['sim_b'] if r['text_a_is_closer'] else r['sim_a'], axis=1)
        df_a['closer_len'] = df_a.apply(lambda r: r['text_a_len'] if r['text_a_is_closer'] else r['text_b_len'], axis=1)
        df_a['farther_len'] = df_a.apply(lambda r: r['text_b_len'] if r['text_a_is_closer'] else r['text_a_len'],
                                         axis=1)

    # --- TRACK B ANALYSIS ---
    if not df_b.empty:
        df_b['text_len'] = df_b['text'].apply(lambda x: len(str(x).split()))

    # --- VISUALIZATION DASHBOARD ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.style.use('seaborn-v0_8-muted')

    # PLOT 1: Track A - Lexical Overlap Comparison
    # Purpose: Does word-overlap explain the labels? (If overlap is the same, task is hard!)
    melted_sim = df_a[['closer_sim', 'farther_sim']].melt(var_name='Type', value_name='Similarity')
    sns.boxplot(data=melted_sim, x='Type', y='Similarity', ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Track A: Lexical Overlap (Jaccard Score)\nCloser vs Farther")
    axes[0, 0].set_xticklabels(['Ground Truth: Closer', 'Ground Truth: Farther'])

    # PLOT 2: Track A - Length Bias Check
    # Purpose: Check if your model might just "cheat" by picking the longer text.
    sns.kdeplot(df_a['closer_len'], ax=axes[0, 1], label='Closer Text', fill=True, color='blue')
    sns.kdeplot(df_a['farther_len'], ax=axes[0, 1], label='Farther Text', fill=True, color='orange')
    axes[0, 1].set_title("Track A: Length Distribution Comparison")
    axes[0, 1].set_xlabel("Number of Words")
    axes[0, 1].legend()

    # PLOT 3: Track B - Narrative Lengths
    # Purpose: Visualizing the scale of Wikipedia plot summaries.
    sns.histplot(df_b['text_len'], bins=30, ax=axes[1, 0], kde=True, color='green')
    axes[1, 0].set_title("Track B: Story Length Distribution")
    axes[1, 0].set_xlabel("Word Count")

    # PLOT 4: Track B - Common Narrative Bigrams
    # Purpose: See the "Actions" of the stories (e.g., 'finds love', 'kills man').
    all_text = " ".join(df_b['text'].astype(str)).lower().split()
    # Filter stopwords and non-alpha
    clean_words = [w for w in all_text if w.isalpha() and w not in STOPWORDS]
    bigrams = list(zip(clean_words, clean_words[1:]))
    bigram_counts = Counter(bigrams).most_common(15)

    bg_labels = [f"{b[0]} {b[1]}" for b, c in bigram_counts]
    bg_values = [c for b, c in bigram_counts]

    sns.barplot(x=bg_values, y=bg_labels, ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("Track B: Top 15 Narrative Bigrams\n(Frequent Story Beats)")

    plt.tight_layout()
    plt.show()

# To use this, just put the filenames of your JSONL files here:
run_narrative_analysis('sample_data/sample_track_a.jsonl', 'sample_data/sample_track_b.jsonl')