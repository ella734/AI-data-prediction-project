"""
notebooks/01_eda_and_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
ÉTAPE 1 – Prétraitement et Analyse Exploratoire des Données (EDA)

Ce script effectue :
  1. Génération du jeu de données synthétique
  2. Nettoyage du texte
  3. Analyse exploratoire (statistiques, visualisations)
  4. Split stratifié Train / Val / Test
  5. Sauvegarde des données préparées

Exécution :
    python notebooks/01_eda_and_preprocessing.py

Sorties :
    - data/raw/reviews_raw.csv
    - data/processed/train.csv, val.csv, test.csv
    - reports/figures/eda_analysis.png
"""

import sys
import os
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Ajoute la racine du projet au chemin Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, set_seed, print_header
from src.data_preprocessing import (
    generate_synthetic_dataset,
    preprocess_dataframe,
    run_eda,
    split_data,
    save_processed_data,
)

# ─── Configuration ────────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 1000
RAW_DATA_PATH = "data/raw/reviews_raw.csv"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = Path("data/raw/aclImdb_v1.tar.gz")


def load_imdb_dataset():
    """Télécharge l'archive IMDb officielle et retourne les avis labellisés."""
    if not IMDB_ARCHIVE.exists():
        logger.info("Téléchargement de l'archive IMDb officielle...")
        IMDB_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMDB_URL, IMDB_ARCHIVE)

    extract_dir = Path("data/raw/aclImdb")
    if not extract_dir.exists():
        logger.info("Extraction de l'archive IMDb...")
        with tarfile.open(IMDB_ARCHIVE, "r:gz") as archive:
            archive.extractall("data/raw")

    rows = []
    for split in ("train", "test"):
        for label_name, label in (("neg", 0), ("pos", 1)):
            folder = extract_dir / split / label_name
            for review_path in sorted(folder.glob("*.txt")):
                rows.append({
                    "text": review_path.read_text(encoding="utf-8"),
                    "label": label,
                    "label_name": "negatif" if label == 0 else "positif",
                    "split": split,
                })

    dataframe = pd.DataFrame(rows)
    dataframe.insert(0, "review_id", range(1, len(dataframe) + 1))
    return dataframe

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(SEED)

    print_header("ÉTAPE 1 – PRÉTRAITEMENT ET ANALYSE EXPLORATOIRE DES DONNÉES")

    # 1. Chargement du jeu de données réel IMDb depuis Stanford
    logger.info("Chargement du jeu de données IMDb...")
    df_raw = load_imdb_dataset()
    df_train = df_raw[df_raw["split"] == "train"].drop(columns="split")
    df_test = df_raw[df_raw["split"] == "test"].drop(columns="split")

    os.makedirs("data/raw", exist_ok=True)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Données IMDb sauvegardées : {RAW_DATA_PATH}")
    print(f"\nAperçu des données IMDb ({len(df_train) + len(df_test)} lignes) :")
    print(df_train.head(5).to_string())

    # 2. Nettoyage du texte
    print_header("Nettoyage du texte")
    df_train = preprocess_dataframe(df_train, text_col="text")
    df_test = preprocess_dataframe(df_test, text_col="text")

    # Exemple de nettoyage
    print("\nExemple AVANT / APRÈS nettoyage :")
    sample = df_train.iloc[0]
    print(f"  AVANT : {sample['text'][:100]}...")
    print(f"  APRÈS : {sample['text_clean'][:100]}...")

    # 3. Analyse exploratoire
    print_header("Analyse Exploratoire des Données (EDA)")
    run_eda(df_train, output_dir="reports/figures")
    print("Graphique EDA généré : reports/figures/eda_analysis.png")

    # 4. Validation issue du train IMDb; le test officiel reste intact
    print_header("Split Train / Validation / Test")
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.15,
        stratify=df_train["label"],
        random_state=SEED,
    )
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"\n  Train : {len(df_train)} exemples")
    print(f"  Val   : {len(df_val)} exemples")
    print(f"  Test  : {len(df_test)} exemples")

    # Vérification de l'équilibre des classes
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split["label"].value_counts(normalize=True).round(3)
        print(f"  {name} >> Positif: {dist.get(1, 0)*100:.1f}% | Negatif: {dist.get(0, 0)*100:.1f}%")

    # 5. Sauvegarde
    print_header("Sauvegarde des données prétraitées")
    save_processed_data(df_train, df_val, df_test)

    print("\n✓ ÉTAPE 1 TERMINÉE AVEC SUCCÈS")
    print("  → Données disponibles dans data/processed/")
    print("  → Graphiques dans reports/figures/")
    print("\nProchaine étape : python notebooks/02_ml_baseline_models.py")
