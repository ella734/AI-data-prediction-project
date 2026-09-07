"""
src/ml_models.py
─────────────────────────────────────────────────────────────────────────────
Entraînement et optimisation des modèles de Machine Learning classiques.

Modèles inclus :
  - Régression Logistique (baseline)
  - SVM (LinearSVC + calibration)
  - Random Forest
  - (XGBoost si installé)

Chaque modèle est entraîné avec TF-IDF + un pipeline Scikit-Learn.
L'optimisation des hyperparamètres est gérée via RandomizedSearchCV.
"""

import os
import time
import logging
import joblib
from typing import Dict, Tuple, Optional, Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Répertoire de sauvegarde des modèles entraînés
MODELS_DIR = "models"


# ─────────────────────────────────────────────────────────────────────────────
# Définitions des pipelines et hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────

def get_logistic_regression_pipeline() -> Tuple[Pipeline, Dict]:
    """Retourne le pipeline de Régression Logistique et sa grille d'hyperparamètres."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                    solver="lbfgs", class_weight="balanced")),
    ])
    param_dist = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "saga"],
    }
    return pipeline, param_dist


def get_svm_pipeline() -> Tuple[Pipeline, Dict]:
    """Retourne le pipeline SVM (LinearSVC calibré) et sa grille d'hyperparamètres."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                   min_df=2, sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(
            max_iter=5000, random_state=42, class_weight="balanced"
        ))),
    ])
    param_dist = {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0],
    }
    return pipeline, param_dist


def get_random_forest_pipeline() -> Tuple[Pipeline, Dict]:
    """Retourne le pipeline Random Forest et sa grille d'hyperparamètres."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                   min_df=2, sublinear_tf=True)),
        ("clf", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )),
    ])
    param_dist = {
        "tfidf__max_features": [3000, 5000],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 20, 50],
        "clf__min_samples_split": [2, 5],
    }
    return pipeline, param_dist


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement avec optimisation d'hyperparamètres
# ─────────────────────────────────────────────────────────────────────────────

def train_with_search(
    pipeline: Pipeline,
    param_dist: Dict,
    X_train,
    y_train,
    model_name: str,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1_macro",
) -> Tuple[Pipeline, float]:
    """
    Entraîne un pipeline avec RandomizedSearchCV.

    Args:
        pipeline: Pipeline Scikit-Learn à optimiser.
        param_dist: Grille d'hyperparamètres à explorer.
        X_train: Features d'entraînement (textes bruts ou matrice TF-IDF).
        y_train: Labels d'entraînement.
        model_name: Nom du modèle (pour les logs).
        n_iter: Nombre de combinaisons à tester.
        cv: Nombre de plis de validation croisée.
        scoring: Métrique d'optimisation.

    Returns:
        (best_pipeline, train_time_in_seconds)
    """
    logger.info(f"\n--- Entraînement : {model_name} ---")
    logger.info(f"RandomizedSearchCV : n_iter={n_iter}, cv={cv}, scoring={scoring}")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(f"Meilleurs hyperparamètres : {search.best_params_}")
    logger.info(f"Meilleur score CV ({scoring}) : {search.best_score_:.4f}")
    logger.info(f"Temps d'entraînement : {elapsed:.2f}s")

    return search.best_estimator_, elapsed


def train_all_models(
    train_texts,
    y_train,
    n_iter: int = 8,
) -> Dict[str, Tuple[Pipeline, float]]:
    """
    Entraîne tous les modèles ML classiques et retourne les résultats.

    Args:
        train_texts: Textes nettoyés d'entraînement.
        y_train: Labels d'entraînement.
        n_iter: Nombre d'itérations pour RandomizedSearchCV.

    Returns:
        Dict {model_name: (fitted_pipeline, train_time)}
    """
    model_configs = [
        ("Logistic Regression", get_logistic_regression_pipeline),
        ("SVM (LinearSVC)",     get_svm_pipeline),
        ("Random Forest",       get_random_forest_pipeline),
    ]

    trained_models = {}
    for name, get_fn in model_configs:
        pipeline, param_dist = get_fn()
        best_model, train_time = train_with_search(
            pipeline, param_dist,
            train_texts, y_train,
            model_name=name,
            n_iter=n_iter,
        )
        trained_models[name] = (best_model, train_time)

    return trained_models


# ─────────────────────────────────────────────────────────────────────────────
# Inférence
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: Pipeline,
    texts,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Effectue une prédiction avec un pipeline entraîné.

    Args:
        model: Pipeline Scikit-Learn ajusté.
        texts: Textes nettoyés (itérable).

    Returns:
        (y_pred, y_proba) — y_proba peut être None si le modèle ne supporte pas predict_proba.
    """
    y_pred = model.predict(texts)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(texts)[:, 1]
        except Exception:
            pass
    return y_pred, y_proba


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde / Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, model_name: str, output_dir: str = MODELS_DIR) -> str:
    """Sérialise un pipeline scikit-learn avec joblib."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filepath = os.path.join(output_dir, f"{safe_name}.joblib")
    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé : {filepath}")
    return filepath


def load_model(filepath: str) -> Pipeline:
    """Charge un pipeline sérialisé depuis un fichier joblib."""
    model = joblib.load(filepath)
    logger.info(f"Modèle chargé : {filepath}")
    return model
