import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import split_sentences, outcome_text, mean_pool


class SBERTModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        weights=(0.4, 0.4, 0.2),  # theme, action, outcome
    ):
        self.model = SentenceTransformer(model_name)
        self.w_theme, self.w_action, self.w_outcome = weights

    def _embed_theme(self, text: str):
        return self.model.encode(text, convert_to_numpy=True)

    def _embed_action(self, text: str):
        sentences = split_sentences(text)
        if not sentences:
            return self._embed_theme(text)
        sent_embs = self.model.encode(sentences, convert_to_numpy=True)
        return mean_pool(sent_embs)

    def _embed_outcome(self, text: str):
        return self.model.encode(outcome_text(text), convert_to_numpy=True)

    def _similarity(self, a: np.ndarray, b: np.ndarray):
        return cosine_similarity([a], [b])[0][0]

    def predict(self, anchor: str, text_a: str, text_b: str) -> bool:
        # Theme
        a_theme = self._embed_theme(anchor)
        ta_theme = self._embed_theme(text_a)
        tb_theme = self._embed_theme(text_b)

        # Course of action
        a_action = self._embed_action(anchor)
        ta_action = self._embed_action(text_a)
        tb_action = self._embed_action(text_b)

        # Outcome
        a_outcome = self._embed_outcome(anchor)
        ta_outcome = self._embed_outcome(text_a)
        tb_outcome = self._embed_outcome(text_b)

        sim_a = (
            self.w_theme * self._similarity(a_theme, ta_theme)
            + self.w_action * self._similarity(a_action, ta_action)
            + self.w_outcome * self._similarity(a_outcome, ta_outcome)
        )

        sim_b = (
            self.w_theme * self._similarity(a_theme, tb_theme)
            + self.w_action * self._similarity(a_action, tb_action)
            + self.w_outcome * self._similarity(a_outcome, tb_outcome)
        )

        return sim_a > sim_b
