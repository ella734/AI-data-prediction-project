# Backlog Azure DevOps – Mini-Projet d'Analyse et Prédiction de Données par IA

> **Projet :** Mini-Projet IA  
> **Organisation :** [Votre Organisation]  
> **Outil :** Azure Boards (Scrum)  
> **Dates :** Sprint 1 → Sprint 4

---

## Structure Organisationnelle

```
Epic
  └─ Feature
       └─ User Story
            └─ Task
```

---

## EPIC 1 – Préparation des Données et EDA

### Feature 1.1 – Acquisition et Exploration des Données

#### User Story 1.1.1
**En tant que** Data Scientist,  
**Je veux** charger et explorer le jeu de données de reviews,  
**Afin de** comprendre sa structure, ses distributions et ses valeurs manquantes.

**Critères d'acceptation :**
- [ ] Le jeu de données est chargé avec pandas
- [ ] Le rapport de valeurs manquantes est généré
- [ ] Les statistiques descriptives sont affichées
- [ ] Les distributions sont visualisées (matplotlib/seaborn)

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-001 | Écrire la fonction `generate_synthetic_dataset()` | 2h | Dev 1 | À faire |
| T-002 | Implémenter l'analyse EDA dans `run_eda()` | 3h | Dev 1 | À faire |
| T-003 | Créer les visualisations (bar chart, histogramme, boxplot) | 2h | Dev 1 | À faire |

---

#### User Story 1.1.2
**En tant que** Data Scientist,  
**Je veux** nettoyer et prétraiter le texte brut,  
**Afin d'** obtenir des données de qualité pour l'entraînement des modèles.

**Critères d'acceptation :**
- [ ] La fonction `clean_text()` supprime URLs, caractères spéciaux, espaces en surplus
- [ ] La fonction `preprocess_dataframe()` s'applique sur tout le DataFrame
- [ ] Les lignes vides post-nettoyage sont supprimées
- [ ] Un test unitaire valide chaque cas de nettoyage

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-004 | Implémenter `clean_text()` avec regex | 2h | Dev 1 | À faire |
| T-005 | Écrire `preprocess_dataframe()` | 1h | Dev 1 | À faire |
| T-006 | Écrire les tests unitaires `TestCleanText` | 2h | Dev 2 | À faire |

---

### Feature 1.2 – Split et Sauvegarde des Données

#### User Story 1.2.1
**En tant que** Data Scientist,  
**Je veux** diviser les données de manière stratifiée,  
**Afin de** garantir une représentation équilibrée des classes dans chaque split.

**Critères d'acceptation :**
- [ ] Split 70% Train / 15% Val / 15% Test
- [ ] Distribution des classes ≈ identique dans chaque split (tolérance ±5%)
- [ ] Les splits sont sauvegardés en CSV dans `data/processed/`

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-007 | Implémenter `split_data()` avec stratification | 1h | Dev 1 | À faire |
| T-008 | Implémenter `save_processed_data()` | 30min | Dev 1 | À faire |
| T-009 | Tests unitaires du split | 1h | Dev 2 | À faire |

---

## EPIC 2 – Modèles de Machine Learning Classiques

### Feature 2.1 – Entraînement des Modèles Baseline

#### User Story 2.1.1
**En tant que** Data Scientist,  
**Je veux** entraîner plusieurs modèles de ML classique avec TF-IDF,  
**Afin d'** établir des benchmarks de référence pour la comparaison.

**Critères d'acceptation :**
- [ ] 3 modèles entraînés : Logistic Regression, SVM, Random Forest
- [ ] Pipeline scikit-learn (TF-IDF + Classifier) pour chaque modèle
- [ ] Hyperparamètres optimisés via `RandomizedSearchCV` (n_iter=8, cv=3)
- [ ] Modèles sérialisés avec `joblib`

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-010 | Implémenter `get_logistic_regression_pipeline()` | 1h | Dev 1 | À faire |
| T-011 | Implémenter `get_svm_pipeline()` | 1h | Dev 1 | À faire |
| T-012 | Implémenter `get_random_forest_pipeline()` | 1h | Dev 1 | À faire |
| T-013 | Implémenter `train_with_search()` avec RandomizedSearchCV | 2h | Dev 1 | À faire |
| T-014 | Implémenter `save_model()` et `load_model()` | 30min | Dev 1 | À faire |

---

### Feature 2.2 – Évaluation des Modèles

#### User Story 2.2.1
**En tant que** Data Scientist,  
**Je veux** évaluer chaque modèle avec des métriques standardisées,  
**Afin de** comparer objectivement les performances.

**Critères d'acceptation :**
- [ ] Calcul d'Accuracy, F1-Score (macro + weighted), ROC-AUC
- [ ] Matrice de confusion générée pour chaque modèle
- [ ] Tableau récapitulatif affiché dans la console

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-015 | Implémenter `compute_metrics()` | 1.5h | Dev 2 | À faire |
| T-016 | Implémenter `plot_confusion_matrix()` | 1h | Dev 2 | À faire |
| T-017 | Implémenter `print_benchmark_table()` | 30min | Dev 2 | À faire |
| T-018 | Tests unitaires `TestComputeMetrics` | 1.5h | Dev 2 | À faire |

---

## EPIC 3 – Expérimentation avec les Transformers

### Feature 3.1 – Fine-tuning DistilBERT

#### User Story 3.1.1
**En tant que** Data Scientist,  
**Je veux** fine-tuner un modèle Transformer pré-entraîné sur notre jeu de données,  
**Afin d'** obtenir des prédictions basées sur la compréhension contextuelle du texte.

**Critères d'acceptation :**
- [ ] `SentimentDataset` (PyTorch Dataset) correctement implémenté
- [ ] Fine-tuning de `distilbert-base-uncased` sur 3 époques
- [ ] Loss et accuracy enregistrés par époque
- [ ] Modèle fine-tuné sauvegardé avec `save_pretrained()`

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-019 | Implémenter `SentimentDataset` | 2h | Dev 1 | À faire |
| T-020 | Implémenter la classe `SentimentTransformer` | 4h | Dev 1 | À faire |
| T-021 | Implémenter la boucle d'entraînement avec warmup scheduler | 3h | Dev 1 | À faire |
| T-022 | Implémenter `plot_training_history()` | 1h | Dev 1 | À faire |
| T-023 | Tester le fine-tuning sur jeu réduit (50 exemples) | 1h | Dev 1 | À faire |

---

## EPIC 4 – Benchmark et Comparaison des Performances

### Feature 4.1 – Visualisation Comparative

#### User Story 4.1.1
**En tant que** Data Scientist,  
**Je veux** comparer tous les modèles sur des critères homogènes,  
**Afin de** formuler des recommandations d'usage basées sur les faits.

**Critères d'acceptation :**
- [ ] Graphique à barres groupées (Accuracy, F1 Macro, F1 Weighted)
- [ ] Courbes ROC superposées sur un même graphique
- [ ] Scatter plot Trade-off Temps d'Entraînement vs Performance
- [ ] Rapport texte complet sauvegardé dans `reports/benchmark_final.txt`

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-024 | Implémenter `generate_benchmark_chart()` | 2h | Dev 2 | À faire |
| T-025 | Implémenter `plot_roc_curve_multi()` | 1.5h | Dev 2 | À faire |
| T-026 | Implémenter le scatter plot Efficacité vs Performance | 1.5h | Dev 2 | À faire |
| T-027 | Générer le rapport texte final | 1h | Dev 2 | À faire |

---

## EPIC 5 – DevOps et Gestion de Version

### Feature 5.1 – Configuration Git/GitHub

#### User Story 5.1.1
**En tant que** développeur,  
**Je veux** une structure de branches Git bien organisée,  
**Afin de** travailler en parallèle sans conflits et maintenir une branche `main` stable.

**Critères d'acceptation :**
- [ ] Stratégie de branches documentée (main, develop, feature/*)
- [ ] `.gitignore` configuré pour Python/Data Science
- [ ] Politique de Pull Requests activée (minimum 1 reviewer)
- [ ] Protection de la branche `main` (pas de push direct)

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-028 | Initialiser le dépôt Git et créer les branches | 30min | Lead Dev | À faire |
| T-029 | Configurer le `.gitignore` | 30min | Lead Dev | À faire |
| T-030 | Configurer les Branch Policies sur GitHub/Azure Repos | 1h | Lead Dev | À faire |

---

### Feature 5.2 – CI/CD

#### User Story 5.2.1
**En tant que** Tech Lead,  
**Je veux** un pipeline d'intégration continue automatique,  
**Afin de** détecter les régressions dès chaque commit.

**Critères d'acceptation :**
- [ ] Le pipeline se déclenche sur chaque PR vers `main` et `develop`
- [ ] Le linting vérifie les erreurs critiques (flake8)
- [ ] Les tests unitaires passent (couverture ≥ 70%)
- [ ] Le rapport de couverture est publié

**Tasks :**
| ID | Description | Estimation | Assigné à | Statut |
|----|-------------|------------|-----------|--------|
| T-031 | Écrire `python-ci.yml` (GitHub Actions) | 2h | DevOps | À faire |
| T-032 | Écrire `azure-pipelines.yml` (Azure DevOps) | 3h | DevOps | À faire |
| T-033 | Valider le pipeline sur un Push de test | 1h | DevOps | À faire |

---

## Planning des Sprints (2 semaines chacun)

| Sprint | Objectif principal | User Stories incluses |
|--------|-------------------|-----------------------|
| Sprint 1 | Data & EDA | 1.1.1, 1.1.2, 1.2.1 |
| Sprint 2 | ML Classiques | 2.1.1, 2.2.1 |
| Sprint 3 | Transformers | 3.1.1 |
| Sprint 4 | Benchmark & DevOps | 4.1.1, 5.1.1, 5.2.1 |

---

## Definition of Done (DoD)

Une User Story est considérée comme **Terminée** si :
- ✅ Le code est implémenté et fonctionne correctement
- ✅ Les tests unitaires sont écrits et passent (pytest)
- ✅ La couverture de code est ≥ 70% pour le module concerné
- ✅ Le code a été revu par au moins 1 autre développeur (Pull Request)
- ✅ Le pipeline CI passe sur la branche develop
- ✅ Le code est mergé dans develop

---

## Équipe du projet

| Rôle | Responsabilités |
|------|----------------|
| Data Scientist (Dev 1) | EDA, Feature Engineering, Modèles ML, Fine-tuning |
| ML Engineer (Dev 2) | Évaluation, Métriques, Visualisations, Tests |
| DevOps (Lead Dev) | Git, CI/CD, Azure Pipelines, Reviews |
