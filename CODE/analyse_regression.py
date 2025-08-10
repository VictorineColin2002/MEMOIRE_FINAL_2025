#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse de régression linéaire multiple pour l'évaluation des facteurs de lisibilité
du néerlandais en tant que langue étrangère
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration pour l'affichage
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data(filepath):
    """Charger et explorer les données"""
    print("=== CHARGEMENT ET EXPLORATION DES DONNÉES ===")
    
    # Charger les données
    df = pd.read_csv(filepath)
    print(f"Nombre de textes analysés : {len(df)}")
    print(f"Nombre de variables : {len(df.columns) - 2}")  # -2 pour fichier et niveau
    
    # Distribution des niveaux
    print("\nDistribution des niveaux CECR :")
    niveau_counts = df['niveau'].value_counts().sort_index()
    print(niveau_counts)
    
    # Statistiques descriptives des variables
    print("\nStatistiques descriptives des variables linguistiques :")
    variables = df.columns[2:]  # Exclure 'fichier' et 'niveau'
    print(df[variables].describe())
    
    return df, variables

def prepare_data_for_regression(df, variables):
    """Préparer les données pour la régression"""
    print("\n=== PRÉPARATION DES DONNÉES POUR LA RÉGRESSION ===")
    
    # Encoder les niveaux CECR en valeurs numériques
    le = LabelEncoder()
    df['niveau_numeric'] = le.fit_transform(df['niveau'])
    
    # Mapping des niveaux
    niveau_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Encodage des niveaux CECR :")
    for niveau, code in niveau_mapping.items():
        print(f"  {niveau} -> {code}")
    
    # Variables prédictives (X) et variable cible (y)
    X = df[variables]
    y = df['niveau_numeric']
    
    # Vérifier les valeurs manquantes
    missing_values = X.isnull().sum()
    if missing_values.any():
        print(f"\nValeurs manquantes détectées :")
        print(missing_values[missing_values > 0])
    else:
        print("\nAucune valeur manquante détectée.")
    
    return X, y, le, niveau_mapping

def analyze_correlations(X, variables):
    """Analyser les corrélations entre variables"""
    print("\n=== ANALYSE DES CORRÉLATIONS ===")
    
    # Matrice de corrélation
    corr_matrix = X.corr()
    
    # Identifier les corrélations élevées (> 0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("Corrélations élevées détectées (|r| > 0.8) :")
        for var1, var2, corr in high_corr_pairs:
            print(f"  {var1} <-> {var2}: r = {corr:.3f}")
    else:
        print("Aucune corrélation élevée (|r| > 0.8) détectée.")
    
    return corr_matrix, high_corr_pairs

def perform_regression_analysis(X, y, variables):
    """Effectuer l'analyse de régression linéaire multiple"""
    print("\n=== ANALYSE DE RÉGRESSION LINÉAIRE MULTIPLE ===")
    
    # Diviser les données (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créer et entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métriques de performance
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Performance sur l'ensemble d'entraînement :")
    print(f"  R² = {r2_train:.4f}")
    print(f"  RMSE = {rmse_train:.4f}")
    
    print(f"\nPerformance sur l'ensemble de test :")
    print(f"  R² = {r2_test:.4f}")
    print(f"  RMSE = {rmse_test:.4f}")
    
    # Coefficients du modèle
    coefficients = pd.DataFrame({
        'Variable': variables,
        'Coefficient': model.coef_,
        'Coefficient_abs': np.abs(model.coef_)
    }).sort_values('Coefficient_abs', ascending=False)
    
    print(f"\nCoefficients du modèle (ordonnés par importance) :")
    print(f"Constante (intercept) : {model.intercept_:.4f}")
    print("\nCoefficients des variables :")
    for _, row in coefficients.iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        print(f"  {row['Variable']:<25} : {row['Coefficient']:>8.4f} {direction}")
    
    return model, coefficients, r2_train, r2_test, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def interpret_results(coefficients, niveau_mapping):
    """Interpréter les résultats de la régression"""
    print("\n=== INTERPRÉTATION DES RÉSULTATS ===")
    
    print("Interprétation des coefficients :")
    print("(↑ = augmente la difficulté, ↓ = diminue la difficulté)\n")
    
    # Top 5 des variables les plus influentes
    top_variables = coefficients.head(5)
    
    for i, (_, row) in enumerate(top_variables.iterrows(), 1):
        var_name = row['Variable']
        coef = row['Coefficient']
        
        print(f"{i}. {var_name}")
        
        # Interprétations spécifiques selon la variable
        if 'PA_SUBTLEX' in var_name:
            if coef > 0:
                print(f"   → Plus il y a de mots absents de cette liste de fréquence, plus le texte est difficile")
            else:
                print(f"   → Plus il y a de mots absents de cette liste de fréquence, plus le texte est facile")
        
        elif var_name == 'MeanSentL':
            if coef > 0:
                print(f"   → Plus les phrases sont longues, plus le texte est difficile")
            else:
                print(f"   → Plus les phrases sont longues, plus le texte est facile")
        
        elif var_name == 'MTLD':
            if coef > 0:
                print(f"   → Plus la diversité lexicale est élevée, plus le texte est difficile")
            else:
                print(f"   → Plus la diversité lexicale est élevée, plus le texte est facile")
        
        elif var_name == 'Meanconc':
            if coef > 0:
                print(f"   → Plus la concrétude moyenne est élevée, plus le texte est difficile")
            else:
                print(f"   → Plus la concrétude moyenne est élevée, plus le texte est facile")
        
        elif 'Subord' in var_name or 'sub' in var_name:
            if coef > 0:
                print(f"   → Plus il y a de subordination, plus le texte est difficile")
            else:
                print(f"   → Plus il y a de subordination, plus le texte est facile")
        
        elif 'Conn' in var_name:
            if coef > 0:
                print(f"   → Plus il y a de connecteurs, plus le texte est difficile")
            else:
                print(f"   → Plus il y a de connecteurs, plus le texte est facile")
        
        print(f"   Coefficient : {coef:.4f}\n")

def create_visualizations(model, coefficients, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test):
    """Créer des visualisations"""
    print("\n=== CRÉATION DES VISUALISATIONS ===")
    
    # Figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse de Régression - Facteurs de Lisibilité du Néerlandais L2', fontsize=16, fontweight='bold')
    
    # 1. Importance des variables (coefficients)
    top_10 = coefficients.head(10)
    colors = ['red' if x < 0 else 'blue' for x in top_10['Coefficient']]
    
    axes[0,0].barh(range(len(top_10)), top_10['Coefficient'], color=colors, alpha=0.7)
    axes[0,0].set_yticks(range(len(top_10)))
    axes[0,0].set_yticklabels(top_10['Variable'], fontsize=10)
    axes[0,0].set_xlabel('Coefficient')
    axes[0,0].set_title('Top 10 des Variables les Plus Influentes')
    axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Prédictions vs Valeurs réelles (entraînement)
    axes[0,1].scatter(y_train, y_pred_train, alpha=0.6, color='blue')
    axes[0,1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Niveau réel')
    axes[0,1].set_ylabel('Niveau prédit')
    axes[0,1].set_title('Prédictions vs Réalité (Entraînement)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Prédictions vs Valeurs réelles (test)
    axes[1,0].scatter(y_test, y_pred_test, alpha=0.6, color='green')
    axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Niveau réel')
    axes[1,0].set_ylabel('Niveau prédit')
    axes[1,0].set_title('Prédictions vs Réalité (Test)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Résidus
    residuals_train = y_train - y_pred_train
    axes[1,1].scatter(y_pred_train, residuals_train, alpha=0.6, color='purple')
    axes[1,1].axhline(y=0, color='red', linestyle='--')
    axes[1,1].set_xlabel('Valeurs prédites')
    axes[1,1].set_ylabel('Résidus')
    axes[1,1].set_title('Analyse des Résidus')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/regression_analysis.png', dpi=300, bbox_inches='tight')
    print("Graphiques sauvegardés dans : /home/ubuntu/regression_analysis.png")
    
    return fig

def generate_regression_formula(model, coefficients, niveau_mapping):
    """Générer la formule de régression"""
    print("\n=== FORMULE DE RÉGRESSION ===")
    
    formula_parts = [f"{model.intercept_:.4f}"]
    
    for _, row in coefficients.iterrows():
        coef = row['Coefficient']
        var = row['Variable']
        sign = "+" if coef >= 0 else ""
        formula_parts.append(f"{sign}{coef:.4f} × {var}")
    
    formula = " ".join(formula_parts)
    
    print("Formule de régression :")
    print(f"Niveau_numérique = {formula}")
    
    print(f"\nOù :")
    for niveau, code in niveau_mapping.items():
        print(f"  {niveau} = {code}")
    
    return formula

def save_detailed_results(df, model, coefficients, r2_train, r2_test, formula, niveau_mapping):
    """Sauvegarder les résultats détaillés"""
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")
    
    # Créer un rapport détaillé
    report = f"""
ANALYSE DE RÉGRESSION LINÉAIRE MULTIPLE
Facteurs de Lisibilité du Néerlandais en tant que Langue Étrangère

=== DONNÉES ===
Nombre de textes analysés : {len(df)}
Distribution des niveaux :
{df['niveau'].value_counts().sort_index().to_string()}

=== PERFORMANCE DU MODÈLE ===
R² (entraînement) : {r2_train:.4f} ({r2_train*100:.2f}%)
R² (test) : {r2_test:.4f} ({r2_test*100:.2f}%)

=== FORMULE DE RÉGRESSION ===
{formula}

Encodage des niveaux :
{chr(10).join([f"{niveau} = {code}" for niveau, code in niveau_mapping.items()])}

=== COEFFICIENTS DES VARIABLES ===
(Ordonnés par importance absolue)

Constante : {model.intercept_:.4f}

"""
    
    for _, row in coefficients.iterrows():
        direction = "augmente" if row['Coefficient'] > 0 else "diminue"
        report += f"{row['Variable']:<25} : {row['Coefficient']:>8.4f} ({direction} la difficulté)\n"
    
    report += f"""

=== INTERPRÉTATION ===
Le modèle explique {r2_test*100:.1f}% de la variance dans la difficulté des textes.

Variables les plus influentes :
"""
    
    top_5 = coefficients.head(5)
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        direction = "augmente" if row['Coefficient'] > 0 else "diminue"
        report += f"{i}. {row['Variable']} ({direction} la difficulté)\n"
    
    # Sauvegarder le rapport
    with open('/home/ubuntu/rapport_regression.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Sauvegarder les coefficients en CSV
    coefficients.to_csv('/home/ubuntu/coefficients_regression.csv', index=False, encoding='utf-8')
    
    print("Rapport détaillé sauvegardé dans : /home/ubuntu/rapport_regression.txt")
    print("Coefficients sauvegardés dans : /home/ubuntu/coefficients_regression.csv")

def main():
    """Fonction principale"""
    print("ANALYSE DE RÉGRESSION LINÉAIRE MULTIPLE")
    print("Facteurs de Lisibilité du Néerlandais L2")
    print("=" * 50)
    
    # Charger et explorer les données
    df, variables = load_and_explore_data('/home/ubuntu/upload/all_metrics_filtered.csv')
    
    # Préparer les données
    X, y, le, niveau_mapping = prepare_data_for_regression(df, variables)
    
    # Analyser les corrélations
    corr_matrix, high_corr_pairs = analyze_correlations(X, variables)
    
    # Effectuer la régression
    model, coefficients, r2_train, r2_test, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = perform_regression_analysis(X, y, variables)
    
    # Interpréter les résultats
    interpret_results(coefficients, niveau_mapping)
    
    # Créer les visualisations
    fig = create_visualizations(model, coefficients, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test)
    
    # Générer la formule
    formula = generate_regression_formula(model, coefficients, niveau_mapping)
    
    # Sauvegarder les résultats
    save_detailed_results(df, model, coefficients, r2_train, r2_test, formula, niveau_mapping)
    
    print("\n" + "=" * 50)
    print("ANALYSE TERMINÉE")
    print("Consultez les fichiers générés pour les résultats détaillés.")

if __name__ == "__main__":
    main()

