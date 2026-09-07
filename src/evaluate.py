"""
src/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Module d'évaluation des modèles et de génération de visualisations.

Fonctions principales :
  - compute_metrics()      : Calcule toutes les métriques (Accuracy, F1, ROC-AUC)
  - plot_confusion_matrix(): Sauvegarde la matrice de confusion
  - plot_roc_curve()       : Sauvegarde la courbe ROC
  - generate_benchmark()   : Génère le rapport comparatif HTML/texte
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour la génération de fichiers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Calcul des métriques
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Modèle",
) -> Dict[str, float]:
    """
    Calcule les métriques de classification.

    Args:
        y_true: Étiquettes réelles.
        y_pred: Prédictions du modèle.
        y_proba: Probabilités de prédiction (pour ROC-AUC et Log Loss).
        model_name: Nom du modèle (pour l'affichage).

    Returns:
        Dictionnaire de métriques {accuracy, f1_macro, f1_weighted, roc_auc, log_loss}.
    """
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Impossible de calculer ROC-AUC / Log Loss : {e}")

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS – {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score Macro : {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted    : {metrics['f1_weighted']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    if "log_loss" in metrics:
        print(f"  Log Loss       : {metrics['log_loss']:.4f}")
    print(f"\n  Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=["Négatif", "Positif"]))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Matrice de confusion
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures",
    class_names: List[str] = None,
) -> str:
    """Génère et sauvegarde la matrice de confusion."""
    if class_names is None:
        class_names = ["Négatif", "Positif"]

    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Réalité", fontsize=12)
    ax.set_title(f"Matrice de Confusion – {model_name}", fontsize=13, fontweight="bold")

    # Annotations TP / TN / FP / FN
    ax.text(0.5, -0.15, f"TP={cm[1,1]} | TN={cm[0,0]} | FP={cm[0,1]} | FN={cm[1,0]}",
            ha="center", transform=ax.transAxes, fontsize=9, color="gray")

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    filepath = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée : {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# 3. Courbe ROC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve_multi(
    results: List[Dict],
    output_dir: str = "reports/figures",
) -> str:
    """
    Trace les courbes ROC de plusieurs modèles sur un même graphique.

    Args:
        results: Liste de dicts avec clés 'model_name', 'y_true', 'y_proba'.
        output_dir: Répertoire de sauvegarde.

    Returns:
        Chemin vers le fichier PNG sauvegardé.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, res in enumerate(results):
        if "y_proba" not in res or res["y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        auc = roc_auc_score(res["y_true"], res["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{res['model_name']} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Aléatoire (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
    ax.set_title("Comparaison des Courbes ROC", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Courbes ROC sauvegardées : {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tableau de Benchmark Comparatif
# ─────────────────────────────────────────────────────────────────────────────

def generate_benchmark_chart(
    all_metrics: List[Dict],
    output_dir: str = "reports/figures",
) -> str:
    """
    Génère un graphique à barres groupées comparant les modèles sur les métriques clés.

    Args:
        all_metrics: Liste de dicts de métriques (sortie de compute_metrics + timing).
        output_dir: Répertoire de sauvegarde.

    Returns:
        Chemin vers le fichier PNG sauvegardé.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = [m["model_name"] for m in all_metrics]
    accuracy = [m.get("accuracy", 0) * 100 for m in all_metrics]
    f1_macro = [m.get("f1_macro", 0) * 100 for m in all_metrics]
    f1_weighted = [m.get("f1_weighted", 0) * 100 for m in all_metrics]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy (%)",
                   color="#3498db", edgecolor="white", linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x, f1_macro, width, label="F1-Score Macro (%)",
                   color="#e74c3c", edgecolor="white", linewidth=0.5, alpha=0.9)
    bars3 = ax.bar(x + width, f1_weighted, width, label="F1-Score Weighted (%)",
                   color="#2ecc71", edgecolor="white", linewidth=0.5, alpha=0.9)

    # Annotations de valeurs
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Modèles", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Benchmark Comparatif des Modèles de Classification",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graphique de benchmark sauvegardé : {filepath}")
    return filepath


def print_benchmark_table(all_metrics: List[Dict]) -> None:
    """Affiche un tableau récapitulatif formaté des métriques de tous les modèles."""
    print("\n" + "=" * 80)
    print("  TABLEAU DE BENCHMARK – COMPARAISON DES MODÈLES")
    print("=" * 80)
    header = f"{'Modèle':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weight':>10} {'ROC-AUC':>10} {'Train (s)':>10}"
    print(header)
    print("-" * 80)
    for m in all_metrics:
        row = (
            f"{m['model_name']:<25}"
            f" {m.get('accuracy', 0)*100:>9.2f}%"
            f" {m.get('f1_macro', 0)*100:>9.2f}%"
            f" {m.get('f1_weighted', 0)*100:>9.2f}%"
            f" {m.get('roc_auc', 0):>10.4f}"
            f" {m.get('train_time_s', 0):>9.2f}s"
        )
        print(row)
    print("=" * 80 + "\n")
