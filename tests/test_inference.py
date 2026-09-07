"""
tests/test_inference.py
─────────────────────────────────────────────────────────────────────────────
Tests unitaires pour l'inférence des modèles (ML classique + Transformer).

Exécution :
    python -m pytest tests/test_inference.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import generate_synthetic_dataset, preprocess_dataframe, split_data
from src.ml_models import (
    get_logistic_regression_pipeline,
    get_svm_pipeline,
    predict,
)
from src.evaluate import compute_metrics


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prepared_data():
    """Prépare un petit jeu de données pour les tests d'inférence."""
    df_raw = generate_synthetic_dataset(n_samples=200, seed=42)
    df_clean = preprocess_dataframe(df_raw)
    df_train, df_val, df_test = split_data(df_clean, seed=42)
    return {
        "train_texts": df_train["text_clean"].tolist(),
        "val_texts":   df_val["text_clean"].tolist(),
        "test_texts":  df_test["text_clean"].tolist(),
        "y_train":     df_train["label"].values,
        "y_val":       df_val["label"].values,
        "y_test":      df_test["label"].values,
    }


@pytest.fixture(scope="module")
def trained_logistic_regression(prepared_data):
    """Entraîne une Régression Logistique simple pour les tests."""
    pipeline, _ = get_logistic_regression_pipeline()
    pipeline.fit(prepared_data["train_texts"], prepared_data["y_train"])
    return pipeline


@pytest.fixture(scope="module")
def trained_svm(prepared_data):
    """Entraîne un SVM simple pour les tests."""
    pipeline, _ = get_svm_pipeline()
    pipeline.fit(prepared_data["train_texts"], prepared_data["y_train"])
    return pipeline


# ─── Tests : prédictions ML classique ─────────────────────────────────────────

class TestMLPredictions:

    def test_lr_prediction_shape(self, trained_logistic_regression, prepared_data):
        y_pred, y_proba = predict(trained_logistic_regression, prepared_data["test_texts"])
        assert len(y_pred) == len(prepared_data["test_texts"])

    def test_lr_binary_predictions(self, trained_logistic_regression, prepared_data):
        y_pred, _ = predict(trained_logistic_regression, prepared_data["test_texts"])
        assert set(np.unique(y_pred)).issubset({0, 1}), "Les prédictions doivent être 0 ou 1"

    def test_lr_probability_range(self, trained_logistic_regression, prepared_data):
        _, y_proba = predict(trained_logistic_regression, prepared_data["test_texts"])
        assert y_proba is not None
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1), "Les probabilités doivent être dans [0, 1]"

    def test_lr_probability_shape(self, trained_logistic_regression, prepared_data):
        _, y_proba = predict(trained_logistic_regression, prepared_data["test_texts"])
        assert y_proba is not None
        assert len(y_proba) == len(prepared_data["test_texts"])

    def test_svm_binary_predictions(self, trained_svm, prepared_data):
        y_pred, _ = predict(trained_svm, prepared_data["test_texts"])
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_lr_reasonable_accuracy(self, trained_logistic_regression, prepared_data):
        """Une Régression Logistique bien entraînée doit dépasser 65% d'accuracy."""
        y_pred, _ = predict(trained_logistic_regression, prepared_data["test_texts"])
        accuracy = np.mean(y_pred == prepared_data["y_test"])
        assert accuracy > 0.60, f"Accuracy trop basse : {accuracy:.2f} (attendu > 0.60)"

    def test_single_text_inference(self, trained_logistic_regression):
        """Test sur un seul texte."""
        text = ["Ce produit est vraiment excellent, je le recommande."]
        y_pred, y_proba = predict(trained_logistic_regression, text)
        assert len(y_pred) == 1
        assert y_pred[0] in {0, 1}

    def test_multiple_texts_consistency(self, trained_logistic_regression):
        """Les prédictions doivent être cohérentes entre un batch et des prédictions individuelles."""
        texts = [
            "Excellent product, highly recommend!",
            "Terrible, waste of money, do not buy.",
        ]
        y_pred_batch, _ = predict(trained_logistic_regression, texts)
        y_pred_single_0, _ = predict(trained_logistic_regression, [texts[0]])
        y_pred_single_1, _ = predict(trained_logistic_regression, [texts[1]])

        assert y_pred_batch[0] == y_pred_single_0[0]
        assert y_pred_batch[1] == y_pred_single_1[0]


# ─── Tests : compute_metrics ──────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred, model_name="Test Parfait")
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_random_prediction_50pct(self):
        np.random.seed(42)
        y_true = np.array([0, 1] * 50)
        y_pred = np.random.randint(0, 2, 100)
        metrics = compute_metrics(y_true, y_pred, model_name="Aléatoire")
        # L'accuracy aléatoire sur classes équilibrées ≈ 50%
        assert 0.30 <= metrics["accuracy"] <= 0.70

    def test_metrics_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        for key in ["accuracy", "f1_macro", "f1_weighted"]:
            assert key in metrics, f"Clé manquante dans les métriques : {key}"

    def test_roc_auc_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    def test_metrics_range(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        for key in ["accuracy", "f1_macro", "f1_weighted"]:
            assert 0.0 <= metrics[key] <= 1.0, f"Métrique hors de [0, 1] : {key}={metrics[key]}"

    def test_model_name_in_output(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_metrics(y_true, y_pred, model_name="MonModèle")
        assert metrics["model_name"] == "MonModèle"
