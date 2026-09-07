# Mini-projet – Analyse et Prédiction de Données par Intelligence Artificielle

> Prétraitement Pandas · Machine Learning classique · Transformers · Git/GitHub · Azure DevOps

---

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Jeu de données](#jeu-de-données)
6. [Modèles implémentés](#modèles-implémentés)
7. [Résultats et comparaison](#résultats-et-comparaison)
8. [Gestion de version & CI/CD](#gestion-de-version--cicd)
9. [Azure DevOps](#azure-devops)

---

## Vue d'ensemble

Ce projet illustre le cycle de vie complet d'un projet de Machine Learning professionnel :

| Phase | Technologies |
|-------|-------------|
| Prétraitement & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| ML Classique | `scikit-learn` (Logistic Regression, SVM, Random Forest) |
| Transformers | `torch`, `transformers` (DistilBERT / BERT fine-tuning) |
| Évaluation | Accuracy, F1-Score (macro/weighted), Confusion Matrix, ROC |
| CI/CD | GitHub Actions, Azure Pipelines |
| Gestion de projet | Git, GitHub, Azure DevOps Boards |

**Tâche cible :** Classification binaire de sentiment sur des avis textuels (positif / négatif).

---

## Structure du projet

```text
mini_projet_ia/
├── .github/workflows/python-ci.yml   # CI GitHub Actions
├── azure-devops/
│   ├── azure-pipelines.yml            # Pipeline Azure DevOps
│   └── backlog_user_stories.md       # Backlog & User Stories
├── data/
│   ├── raw/                           # Données brutes (non versionnées)
│   └── processed/                     # Données préparées (non versionnées)
├── notebooks/
│   ├── 01_eda_and_preprocessing.py   # EDA & nettoyage
│   ├── 02_ml_baseline_models.py      # Modèles ML classiques
│   ├── 03_transformers_training.py   # Fine-tuning Transformer
│   └── 04_model_comparison.py        # Benchmark comparatif
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Pipeline de données
│   ├── ml_models.py                  # Entraînement ML classiques
│   ├── transformer_models.py         # Fine-tuning & inférence Transformers
│   ├── evaluate.py                   # Métriques & visualisations
│   └── utils.py                      # Logger & utilitaires
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
├── reports/figures/                   # Graphiques générés
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/mini_projet_ia.git
cd mini_projet_ia

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 – Prétraitement des données et EDA
```bash
python notebooks/01_eda_and_preprocessing.py
```

### Étape 2 – Entraînement des modèles ML classiques
```bash
python notebooks/02_ml_baseline_models.py
```

### Étape 3 – Fine-tuning du Transformer (DistilBERT)
```bash
python notebooks/03_transformers_training.py
```

### Étape 4 – Benchmark et comparaison des performances
```bash
python notebooks/04_model_comparison.py
```

### Lancer les tests unitaires
```bash
python -m pytest tests/ -v --tb=short
```

---

## Jeu de données

Le projet utilise un **jeu de données de classification de sentiment** synthétique mais représentatif (1 000 avis). En production, remplacez-le par :
- **IMDb Large Movie Reviews Dataset** (50K exemples, Stanford)
- **Amazon Customer Reviews** (Multi-domaines)
- **CamemBERT Corpus** (textes français)

---

## Modèles implémentés

| Modèle | Bibliothèque | Type |
|--------|-------------|------|
| Régression Logistique | scikit-learn | ML Classique |
| SVM (LinearSVC) | scikit-learn | ML Classique |
| Random Forest | scikit-learn | ML Classique |
| DistilBERT (fine-tuné) | Hugging Face | Transformer |

---

## Résultats et comparaison

Les résultats complets sont générés dans `reports/figures/` après exécution de `04_model_comparison.py`.

Exemple de tableau de benchmark typique :

| Modèle | Accuracy | F1-Score | Temps Train | Temps Inférence |
|--------|----------|----------|-------------|-----------------|
| Logistic Regression | ~85% | ~0.84 | < 1s | Très rapide |
| SVM | ~87% | ~0.86 | ~2s | Rapide |
| Random Forest | ~84% | ~0.83 | ~5s | Rapide |
| **DistilBERT** | **~93%** | **~0.93** | ~5 min | Moyen |

---

## Gestion de version & CI/CD

### Stratégie de branches Git

```text
main          ← branche stable, prête pour la production
  └─ develop  ← intégration continue
       ├─ feature/eda
       ├─ feature/ml-models
       └─ feature/transformers
```

### GitHub Actions CI

Le fichier `.github/workflows/python-ci.yml` déclenche automatiquement :
- Installation des dépendances
- Linting du code (`flake8`)
- Exécution de la suite de tests unitaires (`pytest`)

---

## Azure DevOps

Consultez [`azure-devops/backlog_user_stories.md`](azure-devops/backlog_user_stories.md) pour le découpage complet en **Epics / Features / User Stories** et sprints.

Le fichier [`azure-devops/azure-pipelines.yml`](azure-devops/azure-pipelines.yml) est le pipeline CI/CD prêt à l'emploi pour **Azure Pipelines**.

---

## Auteur

Projet réalisé dans le cadre d'un mini-projet de formation en Intelligence Artificielle.

**Technologies :** Python 3.x · pandas · scikit-learn · PyTorch · Hugging Face Transformers · Git · GitHub Actions · Azure DevOps
