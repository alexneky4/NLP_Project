from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from typing import List
import torch
from pathlib import Path

from narative_similarity_dataset import NarrativeSimilarityDataset


class NarrativeCrossEncoder:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        epochs: int = 1,
        device: str | None = None,
        output_dir: str = "saved_cross_encoder",
    ):
        """
        Cross-Encoder for Task A narrative similarity.

        Args:
            model_name: HF model name or local path
            batch_size: training batch size
            epochs: number of fine-tuning epochs
            device: "cuda" or "cpu" (auto-detect if None)
            output_dir: where to save the trained model
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)

        self.model = CrossEncoder(
            model_name,
            num_labels=1,
            device=self.device,
        )

        self.batch_size = batch_size
        self.epochs = epochs

        print(f"[INFO] Using device: {self.device}")

    # -----------------------------
    # Data preparation
    # -----------------------------

    def build_training_samples(
        self, dataset: NarrativeSimilarityDataset
    ) -> List[InputExample]:
        """
        Convert Track A triplets into pairwise labeled examples.
        """
        samples = []

        for i in range(len(dataset)):
            item = dataset.get_triplet(i)

            anchor = item["anchor"]
            text_a = item["text_a"]
            text_b = item["text_b"]
            a_is_closer = item["label"]

            if a_is_closer:
                samples.append(InputExample(texts=[anchor, text_a], label=1.0))
                samples.append(InputExample(texts=[anchor, text_b], label=0.0))
            else:
                samples.append(InputExample(texts=[anchor, text_a], label=0.0))
                samples.append(InputExample(texts=[anchor, text_b], label=1.0))

        return samples

    # -----------------------------
    # Fine-tuning + Saving
    # -----------------------------

    def fit(self, dataset: NarrativeSimilarityDataset, save: bool = True):
        """
        Fine-tune the cross-encoder on Track A and optionally save it.
        """
        train_samples = self.build_training_samples(dataset)

        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=self.batch_size,
        )

        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=self.epochs,
            warmup_steps=10,
            output_path=str(self.output_dir) if save else None,
        )

        if save:
            self.save()

    # -----------------------------
    # Save / Load
    # -----------------------------

    def save(self):
        """
        Save the trained model to disk.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.output_dir))
        print(f"[INFO] Model saved to: {self.output_dir}")

    @classmethod
    def load(cls, model_path: str):
        """
        Load a saved cross-encoder.
        """
        return CrossEncoder(model_path)

    # -----------------------------
    # Prediction (Task A inference)
    # -----------------------------

    def predict(self, anchor: str, text_a: str, text_b: str) -> bool:
        """
        Return True if text_a is closer than text_b.
        """
        scores = self.model.predict(
            [
                [anchor, text_a],
                [anchor, text_b],
            ]
        )

        return scores[0] > scores[1]
