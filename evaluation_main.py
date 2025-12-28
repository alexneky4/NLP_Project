from tqdm import tqdm

from narative_similarity_dataset import NarrativeSimilarityDataset
from sentence_bert_class import SBERTModel
from simcse_class import SimCSEModel


def evaluate_task_a_per_model(dataset, models):
    results = {}

    for model in models:
        correct = 0

        for i in tqdm(range(len(dataset)), desc=model.__class__.__name__):
            triplet = dataset.get_triplet(i)

            pred = model.predict(
                triplet["anchor"],
                triplet["text_a"],
                triplet["text_b"]
            )

            label = triplet["label"]
            correct += int(pred == label)

        accuracy = correct / len(dataset)
        results[model.__class__.__name__] = accuracy

    return results


dataset = NarrativeSimilarityDataset(
    data_path="sample_data/sample_track_a.jsonl",
    track="A"
)

models = [
    SBERTModel(),
]

accuracies = evaluate_task_a_per_model(dataset, models)

for model_name, acc in accuracies.items():
    print(f"{model_name} accuracy: {acc:.4f}")
