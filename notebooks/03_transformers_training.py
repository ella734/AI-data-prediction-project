"""
notebooks/03_transformers_training.py
─────────────────────────────────────────────────────────────────────────────
ÉTAPE 3 – Fine-tuning d'un Transformer (DistilBERT)

Ce script effectue :
  1. Chargement des données préparées
  2. Fine-tuning de DistilBERT sur le jeu d'entraînement
  3. Évaluation sur le jeu de test
  4. Sauvegarde du modèle fine-tuné
  5. Visualisation des courbes d'entraînement

IMPORTANT :
  - L'exécution sur CPU est plus lente (~5-15 min selon la machine).
  - Pour accélérer : utilisez un GPU (NVIDIA CUDA ou Apple MPS).
  - Le modèle DistilBERT sera téléchargé depuis Hugging Face Hub (~270 MB).

Exécution :
    python notebooks/03_transformers_training.py

Prérequis :
    Avoir exécuté 01_eda_and_preprocessing.py au préalable.

Sorties :
    - models/distilbert_finetuned/   (modèle complet)
    - reports/figures/transformer_training_history.png
    - data/processed/transformer_results.pkl
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils import setup_logging, set_seed, print_header, check_gpu_availability
from src.transformer_models import SentimentTransformer
from src.evaluate import compute_metrics, plot_confusion_matrix

# ─── Configuration ─────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SAVE_DIR = "models/distilbert_finetuned"

# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(SEED)

    print_header("ÉTAPE 3 – FINE-TUNING DISTILBERT (TRANSFORMER)")

    # Vérification GPU
    gpu_info = check_gpu_availability()
    print(f"  Dispositif : {gpu_info}\n")

    # 1. Chargement des données
    if not os.path.exists("data/processed/train.csv"):
        print("ERREUR : Données non trouvées. Exécutez d'abord :")
        print("  python notebooks/01_eda_and_preprocessing.py")
        sys.exit(1)

    df_train = pd.read_csv("data/processed/train.csv")
    df_val   = pd.read_csv("data/processed/val.csv")
    df_test  = pd.read_csv("data/processed/test.csv")

    # DistilBERT fonctionne mieux avec les textes originaux (pas sur-nettoyés)
    train_texts = df_train["text_clean"].tolist()
    val_texts   = df_val["text_clean"].tolist()
    test_texts  = df_test["text_clean"].tolist()

    y_train = df_train["label"].tolist()
    y_val   = df_val["label"].tolist()
    y_test  = df_test["label"].values

    print(f"  Train : {len(train_texts)} | Val : {len(val_texts)} | Test : {len(test_texts)}")
    print(f"\n  Modèle    : {MODEL_NAME}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Max Len   : {MAX_LEN} tokens")
    print(f"  LR        : {LEARNING_RATE}")

    # 2. Initialisation du modèle Transformer
    print_header("Téléchargement et initialisation du modèle")
    transformer = SentimentTransformer(
        model_name=MODEL_NAME,
        num_labels=2,
        max_len=MAX_LEN,
    )

    # 3. Fine-tuning
    print_header(f"Fine-tuning ({EPOCHS} époques)")
    print("  Le modèle DistilBERT (~270 MB) va être téléchargé si nécessaire...")
    print("  Cela peut prendre quelques minutes selon votre connexion et votre CPU.\n")

    train_time = transformer.train(
        train_texts=train_texts,
        train_labels=y_train,
        val_texts=val_texts,
        val_labels=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    print(f"\n  Entraînement terminé en {train_time:.1f}s ({train_time/60:.1f} min)")

    # 4. Visualisation des courbes d'entraînement
    history_fig = transformer.plot_training_history(output_dir="reports/figures")
    print(f"  Courbes d'entraînement : {history_fig}")

    # 5. Évaluation sur le jeu de test
    print_header("Évaluation sur le jeu de test")
    start_inf = time.time()
    y_pred, y_proba = transformer.predict(test_texts, batch_size=BATCH_SIZE)
    inference_time = time.time() - start_inf

    metrics = compute_metrics(y_test, y_pred, y_proba, model_name="DistilBERT (fine-tuné)")
    metrics["train_time_s"] = train_time
    metrics["inference_time_ms"] = (inference_time / len(test_texts)) * 1000

    print(f"\n  Temps d'inférence moyen : {metrics['inference_time_ms']:.3f} ms/exemple")

    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model_name="DistilBERT",
                          output_dir="reports/figures")

    # 6. Sauvegarde du modèle
    print_header("Sauvegarde du modèle fine-tuné")
    transformer.save(output_dir=SAVE_DIR)
    print(f"  Modèle sauvegardé : {SAVE_DIR}/")

    # 7. Sauvegarde des résultats pour le benchmark (étape 4)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/transformer_results.pkl", "wb") as f:
        pickle.dump({
            "metrics": metrics,
            "roc_data": {
                "model_name": "DistilBERT (fine-tuné)",
                "y_true": y_test,
                "y_proba": y_proba,
            }
        }, f)
    logger.info("Résultats Transformer sauvegardés : data/processed/transformer_results.pkl")

    print("\n✓ ÉTAPE 3 TERMINÉE AVEC SUCCÈS")
    print("  → Modèle fine-tuné dans models/distilbert_finetuned/")
    print("  → Courbes d'entraînement dans reports/figures/")
    print("\nProchaine étape : python notebooks/04_model_comparison.py")
