"""
src/transformer_models.py
─────────────────────────────────────────────────────────────────────────────
Fine-tuning et inférence d'un modèle Transformer (DistilBERT) avec Hugging Face.

Architecture :
  - Tokenisation via AutoTokenizer (DistilBERT)
  - Dataset PyTorch personnalisé
  - Fine-tuning avec la boucle d'entraînement standard PyTorch
  - Évaluation et inférence

Note : DistilBERT est utilisé car c'est un modèle léger (~67M params, 40% plus
rapide que BERT-base) tout en conservant 97% des performances.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration par défaut
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
WARMUP_RATIO = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """Dataset PyTorch pour la classification de sentiment."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_len: int = DEFAULT_MAX_LEN,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale du modèle Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SentimentTransformer:
    """
    Wrapper pour le fine-tuning et l'inférence d'un Transformer de classification.

    Args:
        model_name: Identifiant Hugging Face du modèle (ex: 'distilbert-base-uncased').
        num_labels: Nombre de classes de sortie.
        max_len: Longueur maximale de séquence en tokens.
        device: Dispositif PyTorch ('cuda', 'mps' ou 'cpu').
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        max_len: int = DEFAULT_MAX_LEN,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_len = max_len

        # Sélection automatique du device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Device utilisé : {self.device}")
        logger.info(f"Chargement du tokenizer : {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def _init_model(self) -> None:
        """Initialise le modèle de classification pré-entraîné."""
        logger.info(f"Chargement du modèle : {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.model = self.model.to(self.device)

    def _create_data_loader(
        self, texts: List[str], labels: List[int], batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Crée un DataLoader pour un split."""
        dataset = SentimentDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    # ─────────────────────────────────────────────────────────────────────
    # Boucle d'entraînement
    # ─────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LR,
    ) -> float:
        """
        Fine-tune le modèle sur le jeu d'entraînement.

        Args:
            train_texts: Textes d'entraînement.
            train_labels: Labels d'entraînement.
            val_texts: Textes de validation.
            val_labels: Labels de validation.
            epochs: Nombre d'époques.
            batch_size: Taille des batches.
            learning_rate: Taux d'apprentissage initial.

        Returns:
            Temps total d'entraînement en secondes.
        """
        self._init_model()

        train_loader = self._create_data_loader(train_texts, train_labels, batch_size, shuffle=True)
        val_loader = self._create_data_loader(val_texts, val_labels, batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"  FINE-TUNING : {self.model_name}")
        logger.info(f"  Epochs={epochs} | Batch={batch_size} | LR={learning_rate}")
        logger.info(f"  Steps totaux={total_steps} | Warmup={warmup_steps}")
        logger.info(f"{'='*60}")

        total_start = time.time()

        for epoch in range(1, epochs + 1):
            # --- Phase d'entraînement ---
            self.model.train()
            train_loss_sum = 0.0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                train_loss_sum += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                if (step + 1) % max(1, len(train_loader) // 4) == 0:
                    avg_loss = train_loss_sum / (step + 1)
                    logger.info(
                        f"  Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f}"
                    )

            avg_train_loss = train_loss_sum / len(train_loader)

            # --- Phase d'évaluation ---
            val_loss, val_acc = self._evaluate_loop(val_loader)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            print(
                f"\nEpoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_acc*100:.2f}%"
            )

        total_time = time.time() - total_start
        logger.info(f"\nEntraînement terminé en {total_time:.2f}s ({total_time/60:.1f} min)")
        return total_time

    def _evaluate_loop(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Boucle d'évaluation interne (loss + accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(data_loader), correct / total

    # ─────────────────────────────────────────────────────────────────────
    # Prédictions
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectue les prédictions sur un ensemble de textes.

        Args:
            texts: Liste de textes nettoyés.
            batch_size: Taille des batches pour l'inférence.

        Returns:
            (y_pred, y_proba) — labels prédits et probabilités de la classe positive.
        """
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas encore entraîné. Appelez .train() d'abord.")

        dummy_labels = [0] * len(texts)
        data_loader = self._create_data_loader(texts, dummy_labels, batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_probas = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probas = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probas.extend(probas[:, 1].cpu().numpy())

        return np.array(all_preds), np.array(all_probas)

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde / Chargement
    # ─────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str = "models/distilbert_finetuned") -> None:
        """Sauvegarde le modèle et le tokenizer dans output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Modèle Transformer sauvegardé : {output_dir}")

    @classmethod
    def load(cls, model_dir: str) -> "SentimentTransformer":
        """Charge un modèle fine-tuné depuis un répertoire."""
        instance = cls.__new__(cls)
        instance.model_name = model_dir
        instance.num_labels = 2
        instance.max_len = DEFAULT_MAX_LEN
        instance.history = {}
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.model = instance.model.to(instance.device)
        logger.info(f"Modèle Transformer chargé depuis : {model_dir}")
        return instance

    def plot_training_history(self, output_dir: str = "reports/figures") -> str:
        """Sauvegarde les courbes d'entraînement (loss + accuracy)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Historique d'entraînement – {self.model_name.split('/')[-1]}",
                     fontsize=13, fontweight="bold")

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "b-o", label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"], "r-o", label="Val Loss")
        axes[0].set_title("Loss par époque")
        axes[0].set_xlabel("Époque")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, [a * 100 for a in self.history["val_accuracy"]],
                     "g-o", label="Val Accuracy")
        axes[1].set_title("Accuracy de validation par époque")
        axes[1].set_xlabel("Époque")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 105)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(output_dir, "transformer_training_history.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Historique d'entraînement sauvegardé : {filepath}")
        return filepath
