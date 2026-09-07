"""
tests/test_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Tests unitaires pour le module src/data_preprocessing.py

Exécution :
    python -m pytest tests/test_preprocessing.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    generate_synthetic_dataset,
    clean_text,
    preprocess_dataframe,
    split_data,
    create_tfidf_features,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Crée un petit DataFrame de test."""
    return pd.DataFrame({
        "review_id": [1, 2, 3, 4, 5, 6],
        "text": [
            "Ce produit est excellent !",
            "Very disappointed, waste of money.",
            "Super qualité, livraison rapide.",
            "Horrible product, broke immediately.",
            "Très satisfait de mon achat.",
            "   ",   # Texte vide / espaces
        ],
        "label": [1, 0, 1, 0, 1, 0],
        "label_name": ["positif", "négatif", "positif", "négatif", "positif", "négatif"],
    })


# ─── Tests : generate_synthetic_dataset ───────────────────────────────────────

class TestGenerateSyntheticDataset:

    def test_output_length(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert len(df) == 100, "Le nombre d'exemples doit correspondre à n_samples"

    def test_required_columns(self):
        df = generate_synthetic_dataset(n_samples=50)
        for col in ["review_id", "text", "label", "label_name"]:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_binary_labels(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert set(df["label"].unique()).issubset({0, 1}), "Les labels doivent être binaires (0 ou 1)"

    def test_class_balance(self):
        df = generate_synthetic_dataset(n_samples=200)
        # Équilibre attendu ≈ 50% chaque classe (tolérance ±10%)
        pos_ratio = df["label"].mean()
        assert 0.40 <= pos_ratio <= 0.60, f"Déséquilibre des classes détecté : {pos_ratio:.2f}"

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=99)
        df2 = generate_synthetic_dataset(n_samples=50, seed=99)
        assert df1["text"].tolist() == df2["text"].tolist(), "La génération doit être déterministe avec le même seed"

    def test_no_empty_texts(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert df["text"].str.len().min() > 0, "Aucun texte vide ne devrait être généré"


# ─── Tests : clean_text ───────────────────────────────────────────────────────

class TestCleanText:

    def test_lowercasing(self):
        assert clean_text("Hello WORLD") == "hello world"

    def test_url_removal(self):
        result = clean_text("Visitez https://example.com pour plus d'infos")
        assert "http" not in result
        assert "example" not in result

    def test_special_chars_removal(self):
        result = clean_text("Produit! Super@$# Qualité?")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_accent_preservation(self):
        result = clean_text("Très bonne qualité, résultat exceptionnel")
        # Les accents doivent être préservés
        assert "è" in result or "tres" in result  # lowercased

    def test_extra_spaces_normalization(self):
        result = clean_text("  hello    world  ")
        assert result == "hello world"

    def test_empty_string_input(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_numeric_input(self):
        result = clean_text(12345)
        assert isinstance(result, str)


# ─── Tests : preprocess_dataframe ─────────────────────────────────────────────

class TestPreprocessDataframe:

    def test_adds_text_clean_column(self, sample_df):
        df_clean = preprocess_dataframe(sample_df)
        assert "text_clean" in df_clean.columns

    def test_adds_text_length_column(self, sample_df):
        df_clean = preprocess_dataframe(sample_df)
        assert "text_length" in df_clean.columns

    def test_removes_empty_rows(self, sample_df):
        # Le DataFrame de test contient 1 ligne avec texte vide
        df_clean = preprocess_dataframe(sample_df)
        assert len(df_clean) < len(sample_df), "Les lignes avec texte vide doivent être supprimées"
        assert df_clean["text_clean"].str.len().min() > 0

    def test_output_is_copy(self, sample_df):
        original_len = len(sample_df)
        _ = preprocess_dataframe(sample_df)
        assert len(sample_df) == original_len, "Le DataFrame original ne doit pas être modifié"

    def test_text_length_is_correct(self, sample_df):
        df_clean = preprocess_dataframe(sample_df)
        # Vérification que text_length correspond bien au nombre de mots
        for _, row in df_clean.iterrows():
            expected_len = len(row["text_clean"].split())
            assert row["text_length"] == expected_len


# ─── Tests : split_data ───────────────────────────────────────────────────────

class TestSplitData:

    @pytest.fixture
    def clean_df(self):
        df = generate_synthetic_dataset(n_samples=200, seed=42)
        from src.data_preprocessing import preprocess_dataframe
        return preprocess_dataframe(df)

    def test_total_samples_preserved(self, clean_df):
        df_train, df_val, df_test = split_data(clean_df, seed=42)
        assert len(df_train) + len(df_val) + len(df_test) == len(clean_df)

    def test_splits_have_no_overlap(self, clean_df):
        df_train, df_val, df_test = split_data(clean_df, seed=42)
        ids_train = set(df_train["review_id"])
        ids_val = set(df_val["review_id"])
        ids_test = set(df_test["review_id"])
        assert ids_train.isdisjoint(ids_val)
        assert ids_train.isdisjoint(ids_test)
        assert ids_val.isdisjoint(ids_test)

    def test_approximate_split_ratios(self, clean_df):
        df_train, df_val, df_test = split_data(clean_df, test_size=0.15, val_size=0.15, seed=42)
        n = len(clean_df)
        # Test : ≈15% ± 5%
        assert 0.10 <= len(df_test) / n <= 0.20
        assert 0.10 <= len(df_val) / n <= 0.20

    def test_stratified_class_distribution(self, clean_df):
        df_train, df_val, df_test = split_data(clean_df, seed=42)
        for split in [df_train, df_val, df_test]:
            pos_ratio = split["label"].mean()
            assert 0.35 <= pos_ratio <= 0.65, f"Répartition des classes déséquilibrée : {pos_ratio}"


# ─── Tests : create_tfidf_features ────────────────────────────────────────────

class TestTfidfFeatures:

    def test_output_shapes(self):
        train_texts = ["bon produit qualité", "mauvais produit déçu", "excellent achat",
                       "nul livraison retard", "super qualité excellent"]
        val_texts = ["très bien", "pas bien"]
        test_texts = ["ok", "mauvais"]

        from src.data_preprocessing import create_tfidf_features
        vec, X_train, X_val, X_test = create_tfidf_features(
            train_texts, val_texts, test_texts, max_features=100
        )
        assert X_train.shape[0] == len(train_texts)
        assert X_val.shape[0] == len(val_texts)
        assert X_test.shape[0] == len(test_texts)
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]

    def test_vocabulary_size(self):
        train_texts = ["bonjour monde", "monde entier", "bonjour tout le monde"]
        val_texts = ["bonjour"]
        test_texts = ["monde"]

        from src.data_preprocessing import create_tfidf_features
        vec, X_train, X_val, X_test = create_tfidf_features(
            train_texts, val_texts, test_texts, max_features=50
        )
        assert len(vec.vocabulary_) > 0
        assert len(vec.vocabulary_) <= 50
