# Analyse des Facteurs de Lisibilité du Néerlandais Langue Étrangère (L2)

Ce dépôt contient le code et les ressources de recherche pour un mémoire de Master en linguistique et 
Traitement Automatique du Langage (TAL), réalisé à l'UCLouvain. L'objectif principal de ce projet est d'identifier les 
facteurs linguistiques qui déterminent la difficulté d'un texte en néerlandais pour les apprenants non-natifs, 
en utilisant des techniques de TAL et une analyse par régression linéaire multiple.

## Description du projet

La lisibilité d'un texte est un enjeu majeur pour l'enseignement des langues étrangères. Ce projet s'attaque à cette 
problématique pour le néerlandais en tant que langue étrangère (L2), un domaine encore peu exploré.

L'étude repose sur une approche multidimensionnelle et empirique :
1. **Constitution d'un corpus** : Un corpus de 178 textes a été assemblé, classé selon les niveaux du Cadre Européen Commun de Référence pour les Langues (CECR), de A1 à C1-C2.
2. **Extraction de variables** : 18 variables linguistiques, couvrant des aspects lexicaux et syntaxiques, ont été extraites automatiquement de chaque texte à l'aide de scripts Python et de la bibliothèque `spaCy`.
3. **Analyse statistique** : Des analyses descriptives et de la variance (ANOVA) ont été menées pour identifier les variables les plus discriminantes entre les niveaux CECR.
4. **Modélisation prédictive** : Un modèle de régression linéaire multiple a été entraîné pour évaluer le poids de chaque facteur linguistique et prédire le niveau de difficulté d'un texte.

Le modèle final explique **65%** de la variance dans la difficulté des textes, démontrant la pertinence des variables sélectionnées.

## Structure du dépôt

```
.
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── CODE/                        # Scripts d'analyse
│   ├── FINAL_CODE_ALL.py        # Script principal pour l'extraction des variables linguistiques
│   └── analyse_regression.py    # Script pour l'analyse de régression et la visualisation
├── DATA/                        # Données d'entrée
│   ├── data_clean.csv           # Liste de fréquence lexicale 
│   ├── reference_list.txt       # Liste de référence pour les variables SUBTLEX
│   └── concreteness_brysbaert.xlsx # Scores de concrétude (Brysbaert et al.)
└── RESULTS/                     # Résultats détaillés par variable
    ├── LEXICAL/                 # Analyses lexicales
    │   ├── ABSENT_LEX/          # Proportion de mots absents des listes de fréquence
    │   ├── CONCR/               # Scores de concrétude
    │   ├── FREQ_LEX/            # Fréquence lexicale
    │   └── MTLD/                # Diversité lexicale (MTLD)
    └── SYNTAXIC/                # Analyses syntaxiques
        ├── CAUS/                # Connecteurs causaux
        ├── CONTR/               # Connecteurs contrastifs
        ├── MEAN_SENT/           # Longueur moyenne des phrases
        ├── NB_SUBORD/           # Nombre de subordonnées
        ├── PROPSEPV/            # Proportion de verbes à particule séparable
        └── SOV/                 # Proportion de structures SOV
```

## Fonctionnalités principales des scripts

### `CODE/FINAL_CODE_ALL.py`
Script principal responsable de l'extraction de 18 variables linguistiques à partir d'un corpus de textes.

**Analyses lexicales** :
- Longueur moyenne des phrases (`MeanSentL`)
- Proportion de mots absents de listes de fréquence (`PA_SUBTLEX`)
- Diversité lexicale (`MTLD`)
- Score de concrétude moyen (`Meanconc`)
- Fréquence lexicale moyenne

**Analyses syntaxiques** :
- Nombre de subordonnées (`NbSubord`)
- Proportion de verbes à particule séparable (`PropSepVerb`)
- Proportion de structures de phrase SOV (`PropSOV`)
- Ratios de connecteurs logiques (causaux et contrastifs)

### `CODE/analyse_regression.py`
Script d'analyse statistique utilisant les données extraites pour effectuer l'analyse de régression.

**Fonctionnalités** :
- **Préparation des données** : Chargement, exploration et encodage des variables
- **Analyse de régression** : Entraînement d'un modèle de régression linéaire multiple pour prédire le niveau CECR
- **Évaluation du modèle** : Calcul du R² et du RMSE sur les ensembles d'entraînement et de test

## Comment utiliser ce projet

### Prérequis
- Python 3.12
- Les bibliothèques listées dans `requirements.txt`

### Installation
1. Clonez le dépôt :
   ```bash
   git clone [URL_DE_VOTRE_DEPOT]
   cd [NOM_DU_DEPOT]
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez le modèle spaCy pour le néerlandais :
   ```bash
   python -m spacy download nl_core_news_sm
   ```

### Exécution
#### Extraction des variables linguistiques
```bash
python CODE/FINAL_CODE_ALL.py
```
Ce script traite le corpus et génère les fichiers avec toutes les métriques extraites, ces fichiers se placent dans "RESULTATS".

#### Analyse de régression
```bash
python CODE/analyse_regression.py
```
Ce script charge les données depuis `RESULTS\all_metrics_cleaned.csv` et effectue l'analyse de régression, générant des visualisations et des rapports dans le dossier `RESULTS/`.


## Données utilisées

### Corpus principal
- **178 textes** classés par niveau CECR (A1 à C1-C2)

### Ressources lexicales
- **SUBTLEX-NL** : Fréquences lexicales du néerlandais
- **Brysbaert et al.** : Scores de concrétude pour 30,000 mots néerlandais
- **Listes de référence** : Vocabulaire de base 

## ✒Auteur

**Victorine Colin** - *Les apports du Traitement Automatique du Langage (TAL) à l’évaluation des facteurs de
 lisibilité du néerlandais en tant que langue étrangère*  
UCLouvain - 2025

