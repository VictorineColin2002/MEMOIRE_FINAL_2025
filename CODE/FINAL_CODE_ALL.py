#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variables analysées:
- mean_fl, median_fl, p75_fl, p90_fl (statistiques de fréquence)
- MeanSentL (longueur moyenne des phrases)
- PA_SUBTLEX_1500, PA_SUBTLEX_3000, PA_SUBTLEX_MAX (proportions d'absents)
- PA_SUBTLEX_1500_U, PA_SUBTLEX_3000_U, PA_SUBTLEX_MAX_U (proportions d'absents uniques)
- MTLD (diversité lexicale)
- RatioConnContr (ratio connecteurs contrastifs)
- RatioConnCaus (ratio connecteurs causaux)
- NbSubord (nombre de subordonnées)
- PropSepVerb (proportion verbes séparables)
- PropSOV (proportion ordre SOV)
- Meanconc (concrétude moyenne)



Ce code a été réorganisé et commenté à l'aide de ChatGPT
"""

import os
import glob
import pandas as pd
import numpy as np
import spacy
from pathlib import Path
from scipy.stats import f_oneway
from lexical_diversity import lex_div as ld

# =============================================================================
# CONFIGURATION ET CONSTANTES
# =============================================================================

# Chemins (à modifier)
DOSSIER_TEXTES = r"C:\CODE_MEM\MES_TEXTES"
DOSSIER_RESULTATS = r"C:\CODE_MEM\RESULTATS"
FICHIER_FREQ_LIST = r"C:\CODE_MEM\reference_list.txt"
FICHIER_DATA_CLEAN = r"C:\CODE_MEM\data_clean.csv"
LEXICON_CONCRETENESS = r"C:\CODE_MEM\concreteness_brysbaert.xlsx"

# Modèle spaCy
MODELE_SPACY = "nl_core_news_sm"

# Connecteurs
CONNECTEURS_CONTRASTIFS = {"echter", "hoewel", "ook al", "desondanks"}
CONNECTEURS_CAUSAUX = {"doordat", "aangezien", "daardoor"}

# Dépendances syntaxiques
DEPS_SUBORDONNEES = {"mark", "advcl", "csubj", "ccomp"}
DEPS_CLAUSES = {"advcl", "ccomp", "xcomp", "acl", "relcl"}

# =============================================================================
# FONCTIONS UTILITAIRES COMMUNES
# =============================================================================


def extraire_niveau_cecr(nom_fichier):
    """
    Extrait le niveau CECR à partir du nom de fichier de manière standardisée.
    
    Args:
        nom_fichier (str): Nom du fichier
        
    Returns:
        str: Niveau CECR ("A1", "A2", "B1", "B2", "C1-C2", "Inconnu")
    """
    nom_upper = nom_fichier.upper()
    
    # Extraction du préfixe
    if "_" in nom_fichier:
        prefixe = nom_fichier.split("_")[0].upper()
    else:
        prefixe = nom_fichier[:2].upper()
    
    # Regroupement C1 et C2
    if prefixe in ["C1", "C2"] or nom_upper.startswith("C1") or nom_upper.startswith("C2"):
        return "C1-C2"
    elif prefixe in ["A1", "A2", "B1", "B2"]:
        return prefixe
    else:
        # Recherche dans le nom complet
        for niveau in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            if niveau in nom_upper:
                return "C1-C2" if niveau in ["C1", "C2"] else niveau
        return "Inconnu"



def calculer_statistiques_descriptives(df, variable, groupe_col="niveau_CECR"):
    """
    Calcule les statistiques descriptives pour une variable par groupe.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        variable (str): Nom de la variable
        groupe_col (str): Nom de la colonne de groupement
        
    Returns:
        pd.DataFrame: Statistiques descriptives par groupe
    """
    stats = df.groupby(groupe_col)[variable].agg([
        'count', 'mean', 'std', 'min', 'max', 'median',
        lambda x: x.quantile(0.25),  # Q25
        lambda x: x.quantile(0.75)   # Q75
    ]).round(4)
    
    stats.columns = ['Count', 'Moyenne', 'Ecart_type', 'Min', 'Max', 'Mediane', 'Q25', 'Q75']
    return stats.reset_index()

def effectuer_test_anova(df, variable, groupe_col="niveau_CECR"):
    """
    Effectue un test ANOVA pour une variable donnée.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        variable (str): Nom de la variable à tester
        groupe_col (str): Nom de la colonne des groupes
        
    Returns:
        dict: Résultats du test ANOVA
    """
    try:
        # Filtrage des données valides
        df_clean = df.dropna(subset=[variable])
        
        # Création des groupes pour ANOVA
        groupes = [
            groupe[variable].values 
            for nom, groupe in df_clean.groupby(groupe_col)
            if nom != "Inconnu" and len(groupe) > 1
        ]
        
        if len(groupes) < 2:
            return {
                "variable": variable,
                "F_statistic": "N/A",
                "p_value": "N/A",
                "message": "Pas assez de groupes pour ANOVA"
            }
        
        # Test ANOVA
        f_stat, p_val = f_oneway(*groupes)
        
        return {
            "variable": variable,
            "F_statistic": round(f_stat, 4),
            "p_value": f"{p_val:.4e}",
            "message": "ANOVA réalisée"
        }
        
    except Exception as e:
        return {
            "variable": variable,
            "F_statistic": "Erreur",
            "p_value": "Erreur",
            "message": f"Erreur: {str(e)}"
        }



def sauvegarder_resultats_anova(resultats_anova, nom_fichier):
    """
    Sauvegarde les résultats ANOVA dans un fichier.
    
    Args:
        resultats_anova (list): Liste des résultats ANOVA
        nom_fichier (str): Nom du fichier de sortie
    """
    chemin_anova = os.path.join(DOSSIER_RESULTATS, nom_fichier)
    with open(chemin_anova, "w", encoding="utf-8") as f:
        f.write("RÉSULTATS DES TESTS ANOVA\n")
        f.write("=" * 50 + "\n\n")
        
        for resultat in resultats_anova:
            f.write(f"Variable: {resultat['variable']}\n")
            f.write(f"F-statistic: {resultat['F_statistic']}\n")
            f.write(f"p-value: {resultat['p_value']}\n")
            f.write(f"Statut: {resultat['message']}\n")
            f.write("-" * 30 + "\n")



# =============================================================================
# ANALYSES SPÉCIFIQUES
# =============================================================================

def analyser_statistiques_frequence():
    """
    Analyse les statistiques de fréquence (mean_fl, median_fl, p75_fl, p90_fl).
    """
    print("Analyse des statistiques de fréquence...")
    
    # Lecture des données depuis data_clean.csv
    try:
        df_res = pd.read_csv(FICHIER_DATA_CLEAN)
        print(f"Fichier data_clean.csv chargé: {len(df_res)} lignes")
    except FileNotFoundError:
        print(f"Fichier non trouvé: {FICHIER_DATA_CLEAN}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement: {str(e)}")
        return
    

    # Variables à analyser
    variables = ['mean_fl', 'median_fl', 'p75_fl', 'p90_fl']
    
    # Calcul des statistiques transposées (comme dans le code original)
    stats_labels = {
        "mean": "moyenne",
        "std": "écart-type", 
        "min": "min",
        "max": "max",
        "median": "médiane",
        "q25": "1er quartile",
        "q75": "3e quartile"
    }
    
    niveaux = sorted(df_res['niveau'].unique())
    rows = []
    
    for var in variables:
        for stat_code, label in stats_labels.items():
            row = {"Variable": f"{var} - {label}"}
            for niveau in niveaux:
                subset = df_res[df_res["niveau"] == niveau][var]
                if stat_code == "mean":
                    value = subset.mean()
                elif stat_code == "std":
                    value = subset.std()
                elif stat_code == "min":
                    value = subset.min()
                elif stat_code == "max":
                    value = subset.max()
                elif stat_code == "median":
                    value = subset.median()
                elif stat_code == "q25":
                    value = subset.quantile(0.25)
                elif stat_code == "q75":
                    value = subset.quantile(0.75)
                
                row[niveau] = f"{value:.2e}"
            rows.append(row)
    
    df_transposed = pd.DataFrame(rows)
    df_transposed.to_csv(os.path.join(DOSSIER_RESULTATS, "tableau_transpose.csv"), index=False)
    
    # Tests ANOVA pour chaque variable
    resultats_anova = []
    for var in variables:
        resultat = effectuer_test_anova(df_res, var, "niveau")
        resultats_anova.append(resultat)
    
    sauvegarder_resultats_anova(resultats_anova, "anova_statistiques_frequence.txt")
    print("Statistiques de fréquence terminées")




def analyser_longueur_phrases():
    """
    Analyse la longueur moyenne des phrases.
    """
    print("Analyse de la longueur des phrases...")
    
    # Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY)
    
    # Traitement des textes
    resultats = []
    corpus_dir = Path(DOSSIER_TEXTES)
    
    for file_path in corpus_dir.glob("*.txt"):
        niveau = extraire_niveau_cecr(file_path.name)
        text = file_path.read_text(encoding="utf-8")
        doc = nlp(text)
        
        sentences = list(doc.sents)
        nb_sent = len(sentences)
        nb_words = len([t for t in doc if not t.is_punct])
        
        mean_sent_l = nb_words / nb_sent if nb_sent > 0 else 0
        
        resultats.append({
            "fichier": file_path.name,
            "niveau_CECR": niveau,
            "MeanSentL": mean_sent_l
        })
    
    # DataFrame des résultats
    df = pd.DataFrame(resultats)
    
    # Sauvegarde des résultats individuels
    df.to_csv(os.path.join(DOSSIER_RESULTATS, "mean_sent_length_par_texte.csv"), index=False, encoding="utf-8")
    
    # Statistiques par niveau
    stats_niveau = calculer_statistiques_descriptives(df, "MeanSentL")
    stats_niveau.to_csv(os.path.join(DOSSIER_RESULTATS, "mean_sent_length_par_niveau.csv"), index=False, encoding="utf-8")
    
    # Test ANOVA
    resultat_anova = effectuer_test_anova(df, "MeanSentL")
    sauvegarder_resultats_anova([resultat_anova], "anova_mean_sent_length.txt")
    
    print("Analyse longueur phrases terminée")



def analyser_subtlex():
    """
    Analyse les proportions d'absents SUBTLEX.
    """
    print("Analyse SUBTLEX")
    
    # Chargement du modèle spaCy et de la liste de fréquence
    nlp = spacy.load(MODELE_SPACY)
    
    with open(FICHIER_FREQ_LIST, encoding='utf-8') as f:
        freq_list = [line.strip() for line in f if line.strip()]
    
    sub_1500 = set(freq_list[:1500])
    sub_3000 = set(freq_list[:3000])
    sub_all = set(freq_list)
    
    # Fonction de calcul des proportions d'absents
    def calc_absents(tokens, unique_tokens, sub_1500, sub_3000, sub_all):
        total_tokens = len(tokens)
        total_types = len(unique_tokens)
        
        abs_1500 = sum(1 for t in tokens if t not in sub_1500)
        abs_3000 = sum(1 for t in tokens if t not in sub_3000)
        abs_all = sum(1 for t in tokens if t not in sub_all)
        
        abs_1500_u = sum(1 for t in unique_tokens if t not in sub_1500)
        abs_3000_u = sum(1 for t in unique_tokens if t not in sub_3000)
        abs_all_u = sum(1 for t in unique_tokens if t not in sub_all)
        
        return {
            'PA_SUBTLEX_1500': abs_1500 / total_tokens,
            'PA_SUBTLEX_3000': abs_3000 / total_tokens,
            'PA_SUBTLEX_MAX': abs_all / total_tokens,
            'PA_SUBTLEX_1500_U': abs_1500_u / total_types,
            'PA_SUBTLEX_3000_U': abs_3000_u / total_types,
            'PA_SUBTLEX_MAX_U': abs_all_u / total_types
        }
    
    # Traitement des fichiers
    liste_fichiers = glob.glob(os.path.join(DOSSIER_TEXTES, "*.txt"))
    resultats_corpus = []
    
    for fichier in liste_fichiers:
        with open(fichier, encoding='utf-8') as f:
            texte = f.read()
        
        doc = nlp(texte)
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        unique_tokens = set(tokens)
        
        resultats = calc_absents(tokens, unique_tokens, sub_1500, sub_3000, sub_all)
        
        nom_fichier = os.path.basename(fichier)
        niveau = extraire_niveau_cecr(nom_fichier)
        
        resultats['nom_fichier'] = nom_fichier
        resultats['niveau_CECR'] = niveau
        resultats_corpus.append(resultats)
    
    # DataFrame des résultats
    df_resultats = pd.DataFrame(resultats_corpus)
    df_resultats = df_resultats[
        ['nom_fichier', 'niveau_CECR',
         'PA_SUBTLEX_1500', 'PA_SUBTLEX_3000', 'PA_SUBTLEX_MAX',
         'PA_SUBTLEX_1500_U', 'PA_SUBTLEX_3000_U', 'PA_SUBTLEX_MAX_U']
    ]
    
    # Sauvegarde des résultats
    df_resultats.to_csv(os.path.join(DOSSIER_RESULTATS, "resultats_subtlex_par_texte.csv"), index=False)
    
    # Moyennes par niveau
    df_moyennes = df_resultats.groupby("niveau_CECR").mean(numeric_only=True).reset_index()
    df_moyennes.to_csv(os.path.join(DOSSIER_RESULTATS, "moyennes_subtlex_par_niveau.csv"), index=False)
    
    # Tests ANOVA pour toutes les variables SUBTLEX
    variables_subtlex = ['PA_SUBTLEX_1500', 'PA_SUBTLEX_3000', 'PA_SUBTLEX_MAX',
                        'PA_SUBTLEX_1500_U', 'PA_SUBTLEX_3000_U', 'PA_SUBTLEX_MAX_U']
    
    resultats_anova = []
    for var in variables_subtlex:
        resultat = effectuer_test_anova(df_resultats, var)
        resultats_anova.append(resultat)
    
    sauvegarder_resultats_anova(resultats_anova, "anova_subtlex.txt")
    print("Analyse SUBTLEX terminée")

def analyser_mtld():
    """
    Analyse la diversité lexicale (MTLD).
    """
    print("Analyse MTLD...")
    
    # Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY)
    
    # Répertoire contenant les fichiers textes
    txt_dir = Path(DOSSIER_TEXTES)
    
    rows = []
    for fp in txt_dir.glob("*.txt"):
        niveau = extraire_niveau_cecr(fp.name)
        
        # Lecture et traitement du texte
        text = fp.read_text(encoding="utf-8", errors="ignore")
        doc = nlp(text)
        
        # Prétraitement : lemmes minuscules, sans stop-words, uniquement alphabétiques
        lemmas = [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
        
        # Calcul du MTLD sur les lemmes
        mtld_val = ld.mtld(lemmas) if lemmas else float("nan")
        
        rows.append({
            "fichier": fp.stem,
            "niveau_CECR": niveau,
            "MTLD": mtld_val,
        })
    
    # Création du DataFrame
    df = pd.DataFrame(rows).sort_values("MTLD", ascending=False)
    
    # Sauvegarde
    df.to_csv(os.path.join(DOSSIER_RESULTATS, "mtld_par_texte.csv"), index=False, encoding="utf-8")
    
    # Statistiques par niveau
    stats_mtld = calculer_statistiques_descriptives(df, "MTLD")
    stats_mtld.to_csv(os.path.join(DOSSIER_RESULTATS, "mtld_par_niveau.csv"), index=False, encoding="utf-8")
    
    # Test ANOVA
    resultat_anova = effectuer_test_anova(df, "MTLD")
    sauvegarder_resultats_anova([resultat_anova], "anova_mtld.txt")
    
    print("Analyse MTLD terminée")

def analyser_connecteurs():
    """
    Analyse les ratios de connecteurs contrastifs et causaux.
    """
    print("Analyse des connecteurs")
    
    # Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY)
    
    # Analyse des connecteurs contrastifs
    data_contrastifs = []
    for nom_fichier in os.listdir(DOSSIER_TEXTES):
        if nom_fichier.endswith(".txt"):
            chemin = os.path.join(DOSSIER_TEXTES, nom_fichier)
            with open(chemin, "r", encoding="utf-8") as f:
                texte = f.read()
            
            doc = nlp(texte)
            lemmes = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
            
            nb_connecteurs = sum(1 for lemme in lemmes if lemme in CONNECTEURS_CONTRASTIFS)
            nb_total_lemmes = len(lemmes)
            ratio = nb_connecteurs / nb_total_lemmes if nb_total_lemmes > 0 else 0
            
            niveau = extraire_niveau_cecr(nom_fichier)
            
            data_contrastifs.append({
                "nom_fichier": nom_fichier,
                "niveau_CECR": niveau,
                "RatioConnContr": ratio,
                "NbConnecteurs": nb_connecteurs,
                "NbTotalLemmes": nb_total_lemmes
            })
    
    # Analyse des connecteurs causaux
    data_causaux = []
    for nom_fichier in os.listdir(DOSSIER_TEXTES):
        if nom_fichier.endswith(".txt"):
            chemin_fichier = os.path.join(DOSSIER_TEXTES, nom_fichier)
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                texte = f.read()
            
            doc = nlp(texte)
            lemmes = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
            total_mots = len(lemmes)
            total_connecteurs = sum(1 for lemme in lemmes if lemme in CONNECTEURS_CAUSAUX)
            ratio = total_connecteurs / total_mots if total_mots > 0 else 0
            
            niveau = extraire_niveau_cecr(nom_fichier)
            
            data_causaux.append({
                "nom_fichier": nom_fichier,
                "niveau_CECR": niveau,
                "RatioConnCaus": ratio,
                "nb_connecteurs": total_connecteurs,
                "nb_mots": total_mots
            })
    
    # DataFrames
    df_contrastifs = pd.DataFrame(data_contrastifs)
    df_causaux = pd.DataFrame(data_causaux)
    
    # Sauvegarde des résultats
    df_contrastifs.to_csv(os.path.join(DOSSIER_RESULTATS, "connecteurs_contrastifs_par_texte.csv"), index=False)
    df_causaux.to_csv(os.path.join(DOSSIER_RESULTATS, "connecteurs_causaux_par_texte.csv"), index=False)
    
    # Statistiques par niveau
    stats_contr = calculer_statistiques_descriptives(df_contrastifs, "RatioConnContr")
    stats_caus = calculer_statistiques_descriptives(df_causaux, "RatioConnCaus")
    
    stats_contr.to_csv(os.path.join(DOSSIER_RESULTATS, "connecteurs_contrastifs_par_niveau.csv"), index=False)
    stats_caus.to_csv(os.path.join(DOSSIER_RESULTATS, "connecteurs_causaux_par_niveau.csv"), index=False)
    
    # Tests ANOVA
    resultats_anova = []
    resultats_anova.append(effectuer_test_anova(df_contrastifs, "RatioConnContr"))
    resultats_anova.append(effectuer_test_anova(df_causaux, "RatioConnCaus"))
    
    sauvegarder_resultats_anova(resultats_anova, "anova_connecteurs.txt")
    print("Analyse connecteurs terminée")

def analyser_complexite_syntaxique():
    """
    Analyse la complexité syntaxique (subordonnées, verbes séparables, ordre SOV).
    """
    print("Analyse de la complexité syntaxique")
    
    # Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY, disable=["ner", "lemmatizer", "attribute_ruler"])
    
    corpus_dir = Path(DOSSIER_TEXTES)
    resultats = []
    
    # Fonctions utilitaires pour la complexité syntaxique
    def compter_subordonnees(text):
        doc = nlp(text)
        return sum(1 for token in doc if token.dep_ in DEPS_SUBORDONNEES)
    
    def compute_prop_sep_verb(text):
        doc = nlp(text)
        total_verbs = 0
        sep_verbs = 0
        
        for token in doc:
            if token.pos_ == "VERB":
                total_verbs += 1
                for child in token.children:
                    if child.dep_ == "compound:prt":
                        sep_verbs += 1
                        break
        
        return sep_verbs / total_verbs if total_verbs > 0 else 0
    
    def last_non_punct(tokens):
        for t in reversed(tokens):
            if not t.is_space and t.pos_ != "PUNCT":
                return t
        return None
    
    def compute_prop_sov(text):
        doc = nlp(text)
        total, sov = 0, 0
        
        for tok in doc:
            if tok.dep_ in DEPS_CLAUSES and tok.pos_ in {"VERB", "AUX"}:
                span = doc[tok.left_edge.i : tok.right_edge.i + 1]
                toks = [t for t in span if not t.is_space and t.pos_ != "PUNCT"]
                if len(toks) < 3:
                    continue
                
                total += 1
                
                last_tok = last_non_punct(toks)
                if last_tok is not None and last_tok.pos_ in {"VERB", "AUX"}:
                    sov += 1
                elif len(toks) >= 2 and toks[-2].pos_ in {"VERB", "AUX"}:
                    sov += 1
        
        return sov / total if total > 0 else 0.0
    
    # Analyse de tous les fichiers
    for file_path in corpus_dir.glob("*.txt"):
        niveau = extraire_niveau_cecr(file_path.name)
        if niveau == "Inconnu":
            continue
        
        text = file_path.read_text(encoding="utf-8")
        
        # Calcul des métriques
        nb_subord = compter_subordonnees(text)
        prop_sep_verb = compute_prop_sep_verb(text)
        prop_sov = compute_prop_sov(text)
        
        resultats.append({
            "fichier": file_path.name,
            "niveau_CECR": niveau,
            "NbSubord": nb_subord,
            "PropSepVerb": prop_sep_verb,
            "PropSOV": prop_sov
        })
    
    # DataFrame
    df = pd.DataFrame(resultats)
    
    # Sauvegarde des résultats individuels
    df.to_csv(os.path.join(DOSSIER_RESULTATS, "complexite_syntaxique_par_texte.csv"), index=False, encoding="utf-8")
    
    # Statistiques par niveau pour chaque variable
    variables_syntaxe = ["NbSubord", "PropSepVerb", "PropSOV"]
    
    for var in variables_syntaxe:
        stats = calculer_statistiques_descriptives(df, var)
        stats.to_csv(os.path.join(DOSSIER_RESULTATS, f"{var.lower()}_par_niveau.csv"), index=False, encoding="utf-8")
    
    # Tests ANOVA pour toutes les variables syntaxiques
    resultats_anova = []
    for var in variables_syntaxe:
        resultat = effectuer_test_anova(df, var)
        resultats_anova.append(resultat)
    
    sauvegarder_resultats_anova(resultats_anova, "anova_complexite_syntaxique.txt")
    print("Analyse complexité syntaxique terminée")

def analyser_concretude():
    """
    Analyse la concrétude moyenne des textes.
    """
    print("Analyse de la concrétude...")
    
    # Chargement du lexique de concrétude
    try:
        lex = pd.read_excel(LEXICON_CONCRETENESS)
        
        # Détection flexible des colonnes
        lemma_col = [c for c in lex.columns
                     if c.lower() in {"lemma", "word", "stimulus"}][0]
        
        score_col = [c for c in lex.columns
                     if "conc" in c.lower() or "concrete_m" in c.lower()][0]
        
        lex[lemma_col] = lex[lemma_col].str.lower()
        conc_dict = dict(zip(lex[lemma_col], lex[score_col]))
        
        print(f"Lexique de concrétude chargé: {len(conc_dict)} entrées")
        
    except FileNotFoundError:
        print(f"Fichier lexique non trouvé: {LEXICON_CONCRETENESS}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du lexique: {str(e)}")
        return
    
    # Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY)
    
    # Traitement des fichiers
    corpus_dir = Path(DOSSIER_TEXTES)
    resultats = []
    
    for file_path in corpus_dir.glob("*.txt"):
        niveau = extraire_niveau_cecr(file_path.name)
        
        # Lecture et traitement du texte
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        doc = nlp(text)
        
        # Extraction des lemmes alphabétiques
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        
        # Calcul des scores de concrétude
        scores = [conc_dict[lemme] for lemme in lemmas if lemme in conc_dict]
        
        # Calcul de la moyenne
        meanconc = np.mean(scores) if scores else np.nan
        
        resultats.append({
            "fichier": file_path.name,
            "niveau_CECR": niveau,
            "n_tokens": len(lemmas),
            "n_covered": len(scores),
            "Meanconc": round(meanconc, 4) if not np.isnan(meanconc) else np.nan
        })
    
    # DataFrame des résultats
    df = pd.DataFrame(resultats)
    
    # Sauvegarde des résultats individuels
    df.to_csv(os.path.join(DOSSIER_RESULTATS, "concretude_par_texte.csv"), index=False, encoding="utf-8")
    
    # Statistiques par niveau
    stats_concretude = calculer_statistiques_descriptives(df, "Meanconc")
    stats_concretude.to_csv(os.path.join(DOSSIER_RESULTATS, "concretude_par_niveau.csv"), index=False, encoding="utf-8")
    
    # Test ANOVA
    resultat_anova = effectuer_test_anova(df, "Meanconc")
    sauvegarder_resultats_anova([resultat_anova], "anova_concretude.txt")
    
    print("Analyse concrétude terminée")

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Fonction principale qui lance toutes les analyses.
    """
    print("Début de l'analyse linguistique complète")
    print("=" * 50)
    
    # Création du dossier de résultats
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    
    # Lancement de toutes les analyses
    try:
        # Note: analyser_statistiques_frequence() nécessite un fichier resultats_freq.csv existant
        # analyser_statistiques_frequence()
        
        analyser_longueur_phrases()
        analyser_subtlex()
        analyser_mtld()
        analyser_connecteurs()
        analyser_complexite_syntaxique()
        analyser_concretude()
        
        print("\n" + "=" * 50)
        print("Toutes les analyses sont terminées avec succès!")
        print(f"Résultats sauvegardés dans: {DOSSIER_RESULTATS}")
        
        # Résumé des fichiers ANOVA créés
        print("\nTests ANOVA effectués pour toutes les variables:")
        print("- anova_mean_sent_length.txt")
        print("- anova_subtlex.txt (6 variables)")
        print("- anova_mtld.txt")
        print("- anova_connecteurs.txt (2 variables)")
        print("- anova_complexite_syntaxique.txt (3 variables)")
        print("- anova_concretude.txt")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {str(e)}")
        raise

if __name__ == "__main__":
    main()

