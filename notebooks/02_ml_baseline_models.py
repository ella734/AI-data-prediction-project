"""
notebooks/02_ml_baseline_models.py
─────────────────────────────────────────────────────────────────────────────
ÉTAPE 2 – Entraînement des modèles de Machine Learning classiques

Modèles entraînés (Pipeline TF-IDF + Classifier) :
  - Régression Logistique
  - SVM (LinearSVC calibré)
  - Random Forest

Optimisation des hyperparamètres via RandomizedSearchCV (3-fold CV).

Exécution :
    python notebooks/02_ml_baseline_models.py

Prérequis :
    Avoir exécuté 01_eda_and_preprocessing.py au préalable.

Sorties :
    - reports/figures/confusion_matrix_*.png
    - reports/figures/roc_curves_comparison.png  (partiel — complet en étape 4)
    - Résultats affichés dans la console
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils import setup_logging, set_seed, print_header
from src.ml_models import train_all_models, predict, save_model
from src.evaluate import compute_metrics, plot_confusion_matrix, print_benchmark_table

# ─── Configuration ─────────────────────────────────────────────────────────
SEED = 42
N_ITER_SEARCH = 8    # Nombre d'itérations RandomizedSearchCV

# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(SEED)

    print_header("ÉTAPE 2 – MODÈLES DE MACHINE LEARNING CLASSIQUES")

    # 1. Chargement des données préparées
    if not os.path.exists("data/processed/train.csv"):
        print("ERREUR : Données non trouvées. Exécutez d'abord :")
        print("  python notebooks/01_eda_and_preprocessing.py")
        sys.exit(1)

    df_train = pd.read_csv("data/processed/train.csv")
    df_val   = pd.read_csv("data/processed/val.csv")
    df_test  = pd.read_csv("data/processed/test.csv")

    train_texts = df_train["text_clean"].tolist()
    val_texts   = df_val["text_clean"].tolist()
    test_texts  = df_test["text_clean"].tolist()

    y_train = df_train["label"].values
    y_val   = df_val["label"].values
    y_test  = df_test["label"].values

    print(f"\n  Train : {len(train_texts)} | Val : {len(val_texts)} | Test : {len(test_texts)}")

    # 2. Entraînement de tous les modèles
    print_header("Entraînement des modèles (RandomizedSearchCV)")
    trained_models = train_all_models(train_texts, y_train, n_iter=N_ITER_SEARCH)

    # 3. Évaluation sur le jeu de test
    print_header("Évaluation sur le jeu de test")
    all_metrics = []
    all_roc_data = []

    os.makedirs("reports/figures", exist_ok=True)

    for model_name, (model, train_time) in trained_models.items():
        # Prédictions
        start_inf = time.time()
        y_pred, y_proba = predict(model, test_texts)
        inference_time = time.time() - start_inf

        # Métriques
        metrics = compute_metrics(y_test, y_pred, y_proba, model_name=model_name)
        metrics["train_time_s"] = train_time
        metrics["inference_time_ms"] = (inference_time / len(test_texts)) * 1000
        all_metrics.append(metrics)

        # Matrice de confusion
        plot_confusion_matrix(y_test, y_pred, model_name=model_name,
                              output_dir="reports/figures")

        # Données pour ROC (sauvegardées pour l'étape 4)
        if y_proba is not None:
            all_roc_data.append({
                "model_name": model_name,
                "y_true": y_test,
                "y_proba": y_proba,
            })

        # Sauvegarde du modèle
        save_model(model, model_name)

        print(f"\n  Temps d'inférence moyen : {metrics['inference_time_ms']:.3f} ms/exemple")

    # 4. Tableau comparatif
    print_header("Tableau de Benchmark – Modèles ML Classiques")
    print_benchmark_table(all_metrics)

    # 5. Sauvegarde des résultats pour l'étape 4
    import pickle
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/ml_results.pkl", "wb") as f:
        pickle.dump({"metrics": all_metrics, "roc_data": all_roc_data}, f)
    logger.info("Résultats ML sauvegardés : data/processed/ml_results.pkl")

    print("\n✓ ÉTAPE 2 TERMINÉE AVEC SUCCÈS")
    print("  → Modèles sauvegardés dans models/")
    print("  → Matrices de confusion dans reports/figures/")
    print("\nProchaine étape : python notebooks/03_transformers_training.py")
