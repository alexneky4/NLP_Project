from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimCSEModel:
    def __init__(self, model_name="princeton-nlp/sup-simcse-roberta-base"):
        self.model = SentenceTransformer(model_name)

    def predict(self, anchor, text_a, text_b):
        emb = self.model.encode(
            [anchor, text_a, text_b],
            convert_to_numpy=True
        )

        sim_a = cosine_similarity([emb[0]], [emb[1]])[0][0]
        sim_b = cosine_similarity([emb[0]], [emb[2]])[0][0]

        return sim_a > sim_b

