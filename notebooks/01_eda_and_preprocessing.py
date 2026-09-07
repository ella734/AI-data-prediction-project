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

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(SEED)

    print_header("ÉTAPE 1 – PRÉTRAITEMENT ET ANALYSE EXPLORATOIRE DES DONNÉES")

    # 1. Génération des données brutes
    logger.info(f"Génération de {N_SAMPLES} avis synthétiques...")
    df_raw = generate_synthetic_dataset(n_samples=N_SAMPLES, seed=SEED)

    os.makedirs("data/raw", exist_ok=True)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Données brutes sauvegardées : {RAW_DATA_PATH}")
    print(f"\nAperçu des données brutes ({df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes) :")
    print(df_raw.head(5).to_string())

    # 2. Nettoyage du texte
    print_header("Nettoyage du texte")
    df_clean = preprocess_dataframe(df_raw, text_col="text")

    # Exemple de nettoyage
    print("\nExemple AVANT / APRÈS nettoyage :")
    sample = df_raw.iloc[0]
    print(f"  AVANT : {sample['text'][:100]}...")
    print(f"  APRÈS : {df_clean.iloc[0]['text_clean'][:100]}...")

    # 3. Analyse exploratoire
    print_header("Analyse Exploratoire des Données (EDA)")
    run_eda(df_clean, output_dir="reports/figures")
    print("Graphique EDA généré : reports/figures/eda_analysis.png")

    # 4. Split stratifié
    print_header("Split Train / Validation / Test")
    df_train, df_val, df_test = split_data(df_clean, seed=SEED)

    print(f"\n  Train : {len(df_train)} exemples")
    print(f"  Val   : {len(df_val)} exemples")
    print(f"  Test  : {len(df_test)} exemples")

    # Vérification de l'équilibre des classes
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        dist = split["label"].value_counts(normalize=True).round(3)
        print(f"  {name} → Positif: {dist.get(1, 0)*100:.1f}% | Négatif: {dist.get(0, 0)*100:.1f}%")

    # 5. Sauvegarde
    print_header("Sauvegarde des données prétraitées")
    save_processed_data(df_train, df_val, df_test)

    print("\n✓ ÉTAPE 1 TERMINÉE AVEC SUCCÈS")
    print("  → Données disponibles dans data/processed/")
    print("  → Graphiques dans reports/figures/")
    print("\nProchaine étape : python notebooks/02_ml_baseline_models.py")
