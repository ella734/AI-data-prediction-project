"""
src/data_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de prétraitement et d'ingénierie des features.

Étapes couvertes :
  1. Génération / chargement des données brutes
  2. Nettoyage du texte (lowercasing, suppression de ponctuation, etc.)
  3. Analyse exploratoire de base (EDA)
  4. Vectorisation TF-IDF (pour ML classique)
  5. Tokenisation Hugging Face (pour Transformers)
  6. Split stratifié Train / Validation / Test
  7. Sauvegarde des artefacts de données
"""

import os
import re
import logging
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Génération d'un jeu de données synthétique réaliste
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_TEMPLATES = [
    "Ce produit est vraiment excellent, je le recommande vivement.",
    "Très satisfait de mon achat, qualité au rendez-vous.",
    "Service impeccable et livraison ultra-rapide, merci !",
    "Parfait sous tous les angles, je suis ravi.",
    "Produit conforme à la description, fonctionne très bien.",
    "Excellente qualité, je rachèterai sans hésiter.",
    "Très bon rapport qualité-prix, je suis pleinement satisfait.",
    "Super produit, livraison rapide et emballage soigné.",
    "Je suis agréablement surpris par la qualité de cet article.",
    "Rien à redire, produit parfait et service client top.",
    "This movie was absolutely fantastic, I loved every minute.",
    "Great product, exactly as described and fast delivery.",
    "Outstanding quality, highly recommend to everyone.",
    "Really happy with this purchase, exceeded my expectations.",
    "Brilliant film with excellent acting and gripping story.",
]

NEGATIVE_TEMPLATES = [
    "Très déçu par ce produit, qualité bien en dessous de mes attentes.",
    "Livraison en retard et produit endommagé à la réception.",
    "Service client inexistant, problème non résolu après plusieurs relances.",
    "Produit défectueux dès le premier jour d'utilisation.",
    "Ne correspond pas du tout à la description, arnaque totale.",
    "Mauvaise qualité, se casse après 2 jours, à éviter absolument.",
    "Déçu par ce film, scénario creux et jeu d'acteur médiocre.",
    "Perte d'argent pure, produit inutilisable.",
    "Ne fonctionne pas comme prévu, très insatisfait de cet achat.",
    "Qualité déplorable, je ne recommande pas du tout ce vendeur.",
    "Terrible product, broke after one week of use.",
    "Very disappointed, nothing like the description, waste of money.",
    "Horrible movie, no plot, poor acting, complete waste of time.",
    "Worst purchase I have made, do not recommend at all.",
    "Product arrived damaged and customer support was useless.",
]


def generate_synthetic_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Génère un DataFrame synthétique de reviews avec labels binaires.

    Args:
        n_samples: Nombre total d'exemples à générer.
        seed: Graine pour la reproductibilité.

    Returns:
        DataFrame avec colonnes ['review_id', 'text', 'label', 'label_name'].
    """
    random.seed(seed)
    np.random.seed(seed)

    texts, labels = [], []
    for i in range(n_samples):
        # Équilibrage des classes 50/50
        if i % 2 == 0:
            # Positif : concaténation de 1 à 3 templates
            n_t = random.randint(1, 3)
            text = " ".join(random.choices(POSITIVE_TEMPLATES, k=n_t))
            labels.append(1)
        else:
            n_t = random.randint(1, 3)
            text = " ".join(random.choices(NEGATIVE_TEMPLATES, k=n_t))
            labels.append(0)
        texts.append(text)

    df = pd.DataFrame({
        "review_id": range(1, n_samples + 1),
        "text": texts,
        "label": labels,
        "label_name": ["positif" if l == 1 else "négatif" for l in labels],
    })
    logger.info(f"Jeu de données synthétique généré : {len(df)} exemples.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chargement depuis un fichier CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Charge un CSV brut et retourne un DataFrame pandas."""
    df = pd.read_csv(filepath)
    logger.info(f"Données chargées depuis {filepath} : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Nettoyage du texte
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Nettoie un texte brut :
      - Minuscules
      - Suppression des URLs
      - Suppression des caractères spéciaux (non alphanumériques hors accents)
      - Normalisation des espaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"[^a-zàâçéèêëîïôùûüœæ\s]", " ", text) # Caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()              # Espaces multiples
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Applique le nettoyage sur la colonne texte et supprime les lignes vides.

    Args:
        df: DataFrame d'entrée.
        text_col: Nom de la colonne contenant le texte brut.

    Returns:
        DataFrame nettoyé avec une colonne 'text_clean'.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].apply(clean_text)
    df["text_length"] = df["text_clean"].str.split().str.len()

    # Suppression des textes vides après nettoyage
    initial_len = len(df)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    removed = initial_len - len(df)
    if removed > 0:
        logger.warning(f"{removed} lignes supprimées (texte vide après nettoyage).")

    logger.info(f"Prétraitement terminé : {len(df)} exemples valides.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Analyse Exploratoire des Données (EDA)
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Génère et sauvegarde les visualisations EDA clés :
      - Répartition des classes
      - Distribution de la longueur des textes
      - Boîtes à moustaches longueur par classe
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Distribution des classes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Analyse Exploratoire des Données (EDA)", fontsize=14, fontweight="bold")

    # Graphique 1 : Répartition des classes
    class_counts = df["label_name"].value_counts()
    axes[0].bar(class_counts.index, class_counts.values,
                color=["#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[0].set_title("Répartition des classes")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Nombre d'exemples")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Graphique 2 : Distribution des longueurs de texte
    axes[1].hist(df["text_length"], bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    axes[1].set_title("Distribution de la longueur des textes")
    axes[1].set_xlabel("Nombre de mots")
    axes[1].set_ylabel("Fréquence")
    axes[1].axvline(df["text_length"].mean(), color="red", linestyle="--",
                    label=f"Moyenne: {df['text_length'].mean():.1f}")
    axes[1].legend()

    # Graphique 3 : Longueur par classe (boxplot)
    df.boxplot(column="text_length", by="label_name", ax=axes[2],
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color="red", linewidth=2))
    axes[2].set_title("Longueur du texte par classe")
    axes[2].set_xlabel("Sentiment")
    axes[2].set_ylabel("Nombre de mots")
    plt.suptitle("")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "eda_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique EDA sauvegardé : {fig_path}")

    # Statistiques descriptives
    print("\n" + "=" * 60)
    print("  STATISTIQUES DESCRIPTIVES DU JEU DE DONNÉES")
    print("=" * 60)
    print(f"  Nombre total d'exemples : {len(df)}")
    print(f"  Répartition des classes :\n{class_counts.to_string()}")
    print(f"\n  Longueur des textes (mots) :")
    print(df["text_length"].describe().to_string())
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Vectorisation TF-IDF (pour ML Classique)
# ─────────────────────────────────────────────────────────────────────────────

def create_tfidf_features(
    train_texts,
    val_texts,
    test_texts,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple:
    """
    Crée les features TF-IDF à partir des textes nettoyés.

    Args:
        train_texts: Textes d'entraînement.
        val_texts: Textes de validation.
        test_texts: Textes de test.
        max_features: Nombre maximum de tokens dans le vocabulaire.
        ngram_range: Plage de n-grams (unigrammes + bigrammes par défaut).

    Returns:
        (vectorizer, X_train, X_val, X_test) : matrices TF-IDF sparses.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        sublinear_tf=True,  # Lissage logarithmique des TF
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    logger.info(
        f"TF-IDF vectorisé : vocabulaire={len(vectorizer.vocabulary_)}, "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )
    return vectorizer, X_train, X_val, X_test


# ─────────────────────────────────────────────────────────────────────────────
# 6. Split Stratifié Train / Validation / Test
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié : 70% train / 15% val / 15% test.

    Args:
        df: DataFrame prétraité.
        label_col: Colonne de labels.
        test_size: Fraction pour le test.
        val_size: Fraction pour la validation.
        seed: Graine aléatoire.

    Returns:
        (df_train, df_val, df_test)
    """
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_ratio, stratify=df_train_val[label_col],
        random_state=seed
    )

    logger.info(
        f"Split : train={len(df_train)} | val={len(df_val)} | test={len(df_test)}"
    )
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split_df[label_col].value_counts(normalize=True).round(3)
        logger.info(f"  {name} distribution : {dist.to_dict()}")

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sauvegarde des données préparées
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    """Sauvegarde les splits CSV dans le dossier de données préparées."""
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    logger.info(f"Données sauvegardées dans {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée de test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    logger.info("=== Lancement du pipeline de prétraitement ===")

    # 1. Génération des données
    df_raw = generate_synthetic_dataset(n_samples=1000)
    os.makedirs("data/raw", exist_ok=True)
    df_raw.to_csv("data/raw/reviews_raw.csv", index=False)
    logger.info("Données brutes sauvegardées : data/raw/reviews_raw.csv")

    # 2. Nettoyage
    df_clean = preprocess_dataframe(df_raw, text_col="text")

    # 3. EDA
    run_eda(df_clean, output_dir="reports/figures")

    # 4. Split
    df_train, df_val, df_test = split_data(df_clean)

    # 5. Sauvegarde
    save_processed_data(df_train, df_val, df_test)

    logger.info("=== Pipeline de prétraitement terminé avec succès ===")
