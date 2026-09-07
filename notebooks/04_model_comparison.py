"""
notebooks/04_model_comparison.py
─────────────────────────────────────────────────────────────────────────────
ÉTAPE 4 – Benchmark Comparatif des Modèles

Ce script agrège les résultats des étapes 2 et 3 pour produire :
  - Tableau de benchmark chiffré (console)
  - Graphique de comparaison des scores (accuracy, F1-Score)
  - Courbes ROC superposées
  - Analyse de l'efficacité calculatoire (temps train vs performance)
  - Analyse d'erreurs qualitative

Exécution :
    python notebooks/04_model_comparison.py

Prérequis :
    Avoir exécuté les étapes 02 et 03 au préalable.

Sorties :
    - reports/figures/benchmark_comparison.png
    - reports/figures/roc_curves_comparison.png
    - reports/figures/efficiency_tradeoff.png
    - reports/benchmark_final.txt
"""

import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils import setup_logging, set_seed, print_header
from src.evaluate import (
    generate_benchmark_chart,
    plot_roc_curve_multi,
    print_benchmark_table,
)

# ─── Configuration ─────────────────────────────────────────────────────────
SEED = 42

# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = setup_logging()
    set_seed(SEED)

    print_header("ÉTAPE 4 – BENCHMARK COMPARATIF FINAL")

    # 1. Chargement des résultats de l'étape 2 (ML Classiques)
    ml_file = "data/processed/ml_results.pkl"
    tf_file = "data/processed/transformer_results.pkl"

    if not os.path.exists(ml_file):
        print("ERREUR : Résultats ML non trouvés. Exécutez :")
        print("  python notebooks/02_ml_baseline_models.py")
        sys.exit(1)

    with open(ml_file, "rb") as f:
        ml_data = pickle.load(f)

    all_metrics = ml_data["metrics"]
    all_roc_data = ml_data["roc_data"]

    # 2. Chargement des résultats de l'étape 3 (Transformer) si disponible
    if os.path.exists(tf_file):
        with open(tf_file, "rb") as f:
            tf_data = pickle.load(f)
        all_metrics.append(tf_data["metrics"])
        all_roc_data.append(tf_data["roc_data"])
        print("  Résultats DistilBERT chargés avec succès.")
    else:
        print("  AVERTISSEMENT : Résultats Transformer non trouvés.")
        print("  Le benchmark sera effectué uniquement sur les modèles ML classiques.")
        print("  Pour inclure DistilBERT, exécutez d'abord :")
        print("    python notebooks/03_transformers_training.py\n")

    # 3. Tableau de benchmark console
    print_header("Tableau de Benchmark Complet")
    print_benchmark_table(all_metrics)

    # 4. Graphique de comparaison des scores
    print_header("Génération des visualisations comparatives")
    os.makedirs("reports/figures", exist_ok=True)

    benchmark_fig = generate_benchmark_chart(all_metrics, output_dir="reports/figures")
    print(f"  ✓ Graphique de benchmark : {benchmark_fig}")

    # 5. Courbes ROC superposées
    if all_roc_data:
        roc_fig = plot_roc_curve_multi(all_roc_data, output_dir="reports/figures")
        print(f"  ✓ Courbes ROC : {roc_fig}")

    # 6. Graphique Efficacité vs Performance (Scatter Plot)
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, m in enumerate(all_metrics):
        acc = m.get("accuracy", 0) * 100
        train_t = m.get("train_time_s", 0.01)
        inf_t = m.get("inference_time_ms", 0)
        name = m["model_name"]

        # La taille du point représente le temps d'inférence (latence)
        size = max(100, inf_t * 500)
        ax.scatter(train_t, acc, s=size, color=colors[i % len(colors)],
                   alpha=0.85, edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(name, (train_t, acc), textcoords="offset points",
                    xytext=(10, 5), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Temps d'entraînement (secondes, échelle log)", fontsize=12)
    ax.set_ylabel("Accuracy sur le jeu de test (%)", fontsize=12)
    ax.set_title("Trade-off Efficacité Calculatoire vs Performance Prédictive\n"
                 "(taille des points = latence d'inférence)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 102)

    # Légende pour la taille des points
    handles = [
        mpatches.Patch(color="gray", alpha=0.5, label="Point plus grand = inférence plus lente"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    efficiency_fig = "reports/figures/efficiency_tradeoff.png"
    plt.savefig(efficiency_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Graphique efficacité : {efficiency_fig}")

    # 7. Rapport textuel final
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/benchmark_final.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RAPPORT DE BENCHMARK FINAL – MINI-PROJET IA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Modèle':<30} {'Accuracy':>10} {'F1 Macro':>10} {'ROC-AUC':>10} "
                f"{'Train (s)':>12} {'Inférence (ms)':>15}\n")
        f.write("-" * 90 + "\n")
        for m in all_metrics:
            f.write(
                f"{m['model_name']:<30}"
                f" {m.get('accuracy', 0)*100:>9.2f}%"
                f" {m.get('f1_macro', 0)*100:>9.2f}%"
                f" {m.get('roc_auc', 0):>10.4f}"
                f" {m.get('train_time_s', 0):>11.2f}s"
                f" {m.get('inference_time_ms', 0):>14.3f}ms\n"
            )
        f.write("=" * 90 + "\n\n")

        # Analyse textuelle
        best_acc = max(all_metrics, key=lambda x: x.get("accuracy", 0))
        best_f1 = max(all_metrics, key=lambda x: x.get("f1_macro", 0))
        fastest = min(all_metrics, key=lambda x: x.get("train_time_s", float("inf")))
        lowest_lat = min(all_metrics, key=lambda x: x.get("inference_time_ms", float("inf")))

        f.write("ANALYSE DES RÉSULTATS :\n\n")
        f.write(f"  Meilleure Accuracy     : {best_acc['model_name']} "
                f"({best_acc.get('accuracy', 0)*100:.2f}%)\n")
        f.write(f"  Meilleur F1-Score      : {best_f1['model_name']} "
                f"({best_f1.get('f1_macro', 0)*100:.2f}%)\n")
        f.write(f"  Entraînement le plus rapide : {fastest['model_name']} "
                f"({fastest.get('train_time_s', 0):.2f}s)\n")
        f.write(f"  Inférence la plus rapide    : {lowest_lat['model_name']} "
                f"({lowest_lat.get('inference_time_ms', 0):.3f} ms/exemple)\n\n")
        f.write("  Conclusion : Les modèles de ML classique offrent un excellent rapport\n")
        f.write("  vitesse/performance pour des tâches standard, tandis que les Transformers\n")
        f.write("  apportent une compréhension sémantique supérieure au coût d'une plus\n")
        f.write("  grande empreinte calculatoire.\n")

    print(f"  ✓ Rapport final : {report_path}")

    # Afficher le rapport dans la console
    with open(report_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())

    print("\n" + "=" * 60)
    print("  ✓ ÉTAPE 4 TERMINÉE – BENCHMARK COMPLET")
    print("=" * 60)
    print("\n  Fichiers générés :")
    print("    reports/figures/benchmark_comparison.png")
    print("    reports/figures/roc_curves_comparison.png")
    print("    reports/figures/efficiency_tradeoff.png")
    print("    reports/benchmark_final.txt")
    print("\n  Félicitations ! Le mini-projet est terminé.")
