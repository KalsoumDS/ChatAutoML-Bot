"""
Module d'interface conversationnelle pour le chatbot
"""
import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import re


def show_welcome_message():
    """Affiche le message de bienvenue avec le menu - style professionnel"""
    try:
        import config.config as cfg
        chatbot_name = cfg.CHATBOT_NAME
        chatbot_description = getattr(cfg, 'CHATBOT_DESCRIPTION', 'Votre assistant intelligent pour l\'AutoML sur donnees tabulaires')
    except (ImportError, AttributeError):
        chatbot_name = "ChatAutoML-Bot"
        chatbot_description = "Votre assistant intelligent pour l'AutoML sur donnees tabulaires"
    
    welcome_text = f"""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem; font-size: 2rem; font-weight: 800;">
        {chatbot_name}
    </h1>
    <p style="font-size: 1.1rem; color: #6b7280; margin: 0;">
        {chatbot_description}
    </p>
</div>

<div style="background: linear-gradient(135deg, rgba(37,99,235,0.12) 0%, rgba(124,58,237,0.08) 100%);
    padding: 2rem; border-radius: 14px; color: #1f2937; margin: 2rem 0;
    border: 1px solid rgba(107,114,128,0.15);">
    <h2 style="color: #1f2937; margin-bottom: 1.5rem; font-size: 1.25rem;
        letter-spacing: 0.06em; text-transform: uppercase; font-weight: 700;">
        Pipeline AutoML Complet
    </h2>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.2rem;">
        <div style="background: rgba(255,255,255,0.7); padding: 1.5rem; border-radius: 10px;
            border: 1px solid rgba(107,114,128,0.12);">
            <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase;
                color: #6b7280; font-weight: 600; margin-bottom: 0.5rem;">Etape 01</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem; color: #111827;">Analyse des donnees</strong>
            <small style="color: #6b7280; font-size: 0.9rem;">Exploration et statistiques descriptives</small>
        </div>
        <div style="background: rgba(255,255,255,0.7); padding: 1.5rem; border-radius: 10px;
            border: 1px solid rgba(107,114,128,0.12);">
            <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase;
                color: #6b7280; font-weight: 600; margin-bottom: 0.5rem;">Etape 02</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem; color: #111827;">Preprocessing</strong>
            <small style="color: #6b7280; font-size: 0.9rem;">Nettoyage, encodage et normalisation</small>
        </div>
        <div style="background: rgba(255,255,255,0.7); padding: 1.5rem; border-radius: 10px;
            border: 1px solid rgba(107,114,128,0.12);">
            <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase;
                color: #6b7280; font-weight: 600; margin-bottom: 0.5rem;">Etape 03</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem; color: #111827;">Gestion du desequilibre</strong>
            <small style="color: #6b7280; font-size: 0.9rem;">SMOTE, oversampling, undersampling</small>
        </div>
        <div style="background: rgba(255,255,255,0.7); padding: 1.5rem; border-radius: 10px;
            border: 1px solid rgba(107,114,128,0.12);">
            <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase;
                color: #6b7280; font-weight: 600; margin-bottom: 0.5rem;">Etape 04</div>
            <strong style="font-size: 1.1rem; display: block; margin-bottom: 0.3rem; color: #111827;">Recherche de modeles</strong>
            <small style="color: #6b7280; font-size: 0.9rem;">Optimisation d'hyperparametres</small>
        </div>
    </div>
</div>
"""
    return welcome_text


def detect_user_intent(message: str) -> Dict[str, Any]:
    """
    Detecte l'intention de l'utilisateur a partir de son message
    
    Returns:
        Dict avec 'intent', 'confidence', 'entities'
    """
    message_lower = message.lower().strip()
    message_lower = " ".join(message_lower.split())
    
    # Salutations
    if any(word in message_lower for word in ['bonjour', 'salut', 'hello', 'hi', 'coucou']):
        return {'intent': 'greeting', 'confidence': 0.9, 'entities': {}}

    # Demande de statut
    if any(phrase in message_lower for phrase in [
        "vous avez fini", "t'as fini", "tu as fini", "as-tu fini", "avez-vous fini",
        "vous avez termine", "tu as termine", "c'est fini", "c'est termine", "c est fini", "c est termine",
        "c'est bon", "c est bon", "ou en est on", "ou en es tu", "ou en est-on",
    ]):
        return {'intent': 'status_check', 'confidence': 0.9, 'entities': {}}

    # Revenir sur une etape / expliquer mieux
    if any(w in message_lower for w in [
        'explique', 'expliquer', 'reexplique', 'reexpliquer', 'detaille', 'detailler', 'plus de detail',
        'explain', 'clarify', 'details', 'detail', 'elaborate',
        'revenir', 'revient', 'reviens', 'revenez', 'retour', 'retourne', 'retournez', 'reprendre', 'reprends', 'reprenez',
    ]):
        entities: Dict[str, Any] = {}
        m = re.search(r"(etape|step)\s*(\d)", message_lower)
        if m:
            try:
                entities['step'] = int(m.group(2))
            except Exception:
                pass
        if any(w in message_lower for w in ['analyse', 'dataset', 'donnees', 'donnees']):
            entities['section'] = 'analysis'
        if any(w in message_lower for w in ['pretraitement', 'preprocessing', 'nettoyage', 'encodage', 'scaling']):
            entities['section'] = 'preprocessing'
        if any(w in message_lower for w in ['desequilibre', 'imbalance', 'smote', 'oversampling', 'undersampling']):
            entities['section'] = 'imbalance'
        if any(w in message_lower for w in ['comparaison', 'modeles', 'modele', 'tableau', 'cv', 'grid', 'random search', 'recherche']):
            entities['section'] = 'model_comparison'
        if any(w in message_lower for w in ['evaluation', 'train', 'test', 'overfitting', 'underfitting', 'performance']):
            entities['section'] = 'evaluation'
        if any(w in message_lower for w in ['importance', 'variables', 'features', 'feature importance']):
            entities['section'] = 'feature_importance'
        if any(w in message_lower for w in ['resume', 'rapport', 'conclusion', 'summary', 'global']):
            entities['section'] = 'final_summary'
        return {'intent': 'explain_step', 'confidence': 0.85, 'entities': entities}

    # Refaire / relancer une etape
    if any(w in message_lower for w in [
        'refais', 'refait', 'refaite', 'refaire', 'refaites', 'recommence', 'recommencer', 'recommencez',
        'relance', 'relancer', 'relancez', 'recalcule', 'recalculer', 'recalculez',
        're-execute', 'reexecute', 'reexécute', 'réexecute',
        'lance', 'lancer', 'exécute', 'execute', 'demarre', 'demarrer', 'démarre', 'démarrer',
    ]):
        entities: Dict[str, Any] = {}
        m = re.search(r"(etape|step)\s*(\d)", message_lower)
        if m:
            try:
                entities['step'] = int(m.group(2))
            except Exception:
                pass
        if any(w in message_lower for w in ['analyse', 'dataset', 'donnees', 'donnees']):
            entities['section'] = 'analysis'
        if any(w in message_lower for w in ['pretraitement', 'preprocessing']):
            entities['section'] = 'preprocessing'
        if any(w in message_lower for w in ['desequilibre', 'imbalance', 'smote', 'oversampling', 'undersampling']):
            entities['section'] = 'imbalance'
        if any(w in message_lower for w in ['comparaison', 'modeles', 'modele', 'recherche', 'cv']):
            entities['section'] = 'model_comparison'
        if any(w in message_lower for w in ['evaluation', 'train', 'test', 'overfitting', 'underfitting']):
            entities['section'] = 'evaluation'
        if any(w in message_lower for w in ['importance', 'variables', 'features']):
            entities['section'] = 'feature_importance'
        if any(w in message_lower for w in ['automl', 'pipeline', 'modele', 'ml']):
            entities['section'] = 'automl'
        if any(w in message_lower for w in ['resume', 'rapport', 'conclusion']):
            entities['section'] = 'final_summary'
        return {'intent': 'redo_step', 'confidence': 0.85, 'entities': entities}

    # Feedback / insatisfaction
    if any(phrase in message_lower for phrase in [
        "j'aime pas", "j aime pas", "je n'aime pas", "j'aime pas ca", "c'est nul", "c'est pas bien",
        "je n'aime pas", "je suis pas content", "je suis mecontent", "ca marche pas", "ca ne marche pas",
        "bug", "probleme", "casse", "erreur", "error", "ne repond pas", "ne marche pas",
    ]):
        return {'intent': 'feedback_negative', 'confidence': 0.85, 'entities': {}}

    # Reset / refaire / recommencer
    if any(phrase in message_lower for phrase in [
        "refais tout", "refaire tout", "recommencer tout", "reset", "reinitialiser", "réinitialiser",
        "nouvelle session", "tout supprimer", "remettre a zero", "remettre à zéro", "vider",
    ]):
        entities = {}
        if any(w in message_lower for w in ["pipeline", "automl", "modele", "resultats", "resultats", "analyse"]):
            entities['scope'] = 'pipeline'
            return {'intent': 'reset_pipeline', 'confidence': 0.9, 'entities': entities}
        entities['scope'] = 'session'
        return {'intent': 'reset_session', 'confidence': 0.9, 'entities': entities}

    # Changer le theme
    if any(word in message_lower for word in ["theme", "sombre", "clair", "dark", "light", "apparence"]):
        entities = {}
        if any(w in message_lower for w in ["sombre", "dark", "sombre"]):
            entities['theme'] = 'sombre'
        elif any(w in message_lower for w in ["clair", "light"]):
            entities['theme'] = 'clair'
        return {'intent': 'change_theme', 'confidence': 0.75, 'entities': entities}
    
    # Charger un dataset
    if any(word in message_lower for word in ['charger', 'upload', 'importer', 'televerser', 'fichier', 'uploader']) or (
        'dataset' in message_lower and any(v in message_lower for v in ['charger', 'upload', 'importer'])
    ):
        return {'intent': 'load_data', 'confidence': 0.9, 'entities': {}}
    
    # Preprocessing
    if any(word in message_lower for word in ['preprocessing', 'pretraitement', 'nettoyer', 'preparer', 'traiter', 'normalisation', 'encodage', 'imputation', 'scaling']):
        return {'intent': 'preprocessing', 'confidence': 0.9, 'entities': {}}
    
    # Lancer AutoML
    if (
        'automl' in message_lower
        or any(word in message_lower for word in ['lancer automl', 'lance automl', 'commencer automl', 'demarrer automl', 'demarrer automl', 'lancer ml'])
        or any(word in message_lower for word in ['entrainer', 'entraîner', 'train']) and any(word in message_lower for word in ['modele', 'pipeline'])
    ):
        return {'intent': 'run_automl', 'confidence': 0.85, 'entities': {}}
    
    # Questions sur les donnees
    if any(word in message_lower for word in ['donnees', 'dataset', 'colonnes', 'lignes', 'statistiques', 'stats', 'donnees', 'shape', 'taille', 'variables', 'nan', 'valeurs manquantes', 'manquant', 'missing']):
        return {'intent': 'ask_about_data', 'confidence': 0.8, 'entities': {}}
    
    # Questions sur les modeles
    if any(word in message_lower for word in ['modele', 'model', 'performance', 'score', 'resultat', 'resultat', 'meilleur', 'best', 'xgboost', 'random forest', 'regression', 'svm', 'knn', 'arbre', 'decision tree', 'logistic']):
        return {'intent': 'ask_about_models', 'confidence': 0.8, 'entities': {}}
    
    # Questions concepts ML generaux
    if any(word in message_lower for word in [
        'overfitting', 'underfitting', 'surapprentissage', 'sous-apprentissage', 'sous apprentissage', 'generalisation',
        'validation croisee', 'cross validation', 'cv', 'k-fold', 'kfold',
        'hyperparametre', 'hyperparamètre', 'hyperparameter', 'grid search', 'random search', 'bayesian',
        'smote', 'oversampling', 'undersampling', 'deséquilibre', 'desequilibre', 'imbalance', 'classe majoritaire', 'classe minoritaire',
        'normalisation', 'standardisation', 'standardscaler', 'minmaxscaler', 'scaling',
        'encodage', 'onehot', 'one-hot', 'one hot', 'label encoding', 'ordinal',
        'imputation', 'median', 'mean', 'moyenne', 'mediane',
        'split', 'train test', 'train/test', 'stratification', 'holdout',
        'accuracy', 'precision', 'rappel', 'recall', 'f1', 'f1-score', 'auc', 'roc', 'matrice de confusion', 'confusion',
        'r2', 'r-squared', 'r carre', 'rmse', 'mae', 'mse', 'mape',
        'feature importance', 'importance des variables', 'variables importantes',
        'classification', 'regression', 'type de tache', 'tache',
        'outlier', 'valeur aberrante', 'aberrante',
    ]):
        return {'intent': 'question', 'confidence': 0.75, 'entities': {}}
    
    # Aide
    if any(word in message_lower for word in ['aide', 'help', 'commande', 'menu', 'que puis', 'comment ca marche', 'comment utiliser', 'instruction', 'guide']):
        return {'intent': 'help', 'confidence': 0.9, 'entities': {}}
    
    # Question generale (avec ?)
    if '?' in message or any(word in message_lower for word in [
        'quoi', 'comment', 'pourquoi', 'quand', 'ou', 'qui', 'est-ce que', 'est ce que',
    ]):
        return {'intent': 'question', 'confidence': 0.7, 'entities': {}}
    
    # Par defaut
    return {'intent': 'general', 'confidence': 0.5, 'entities': {}}


def _get_concept_explanation(question: str) -> Optional[str]:
    """Retourne une explication detaillee pour un concept ML si detecte."""
    q = question.lower()
    
    # Overfitting
    if any(w in q for w in ['overfitting', 'surapprentissage', 'sur apprentissage']):
        return (
            "## Overfitting (Sur-apprentissage)\n\n"
            "### Definition\n"
            "L'overfitting se produit lorsque le modele apprend **trop bien** les donnees d'entrainement, "
            "au point de memoriser le bruit plutot que d'apprendre le pattern general.\n\n"
            "### Causes principales\n"
            "- Modele trop complexe par rapport au volume de donnees\n"
            "- Trop d'iterations d'entrainement\n"
            "- Donnees bruitees ou aberrantes\n"
            "- Nombre de caracteristiques trop eleve par rapport aux echantillons\n\n"
            "### Indicateurs\n"
            "- Performance tres elevee sur train mais faible sur test\n"
            "- Ecart important entre scores train et test\n"
            "- Modele ne generalise pas sur nouvelles donnees\n\n"
            "### Solutions\n"
            "- Reduire la complexite du modele\n"
            "- Augmenter la regularisation (L1, L2, dropout)\n"
            "- Collecter plus de donnees d'entrainement\n"
            "- Utiliser le early stopping\n"
            "- Faire du feature selection\n\n"
            "Dans ChatAutoML-Bot, l'overfitting est detecte automatiquement lors de l'evaluation finale."
        )
    
    # Underfitting
    if any(w in q for w in ['underfitting', 'sous-apprentissage', 'sous apprentissage', 'sousajustement']):
        return (
            "## Underfitting (Sous-apprentissage)\n\n"
            "### Definition\n"
            "L'underfitting se produit lorsque le modele est **trop simple** pour capturer la structure "
            "sous-jacente des donnees. Il performe mal sur train ET sur test.\n\n"
            "### Causes principales\n"
            "- Modele trop simple (ex: regression lineaire pour relation non-lineaire)\n"
            "- Donnees insuffisantes pour apprendre les patterns\n"
            "- Trop de regularisation\n"
            "- Temps d'entrainement insuffisant\n"
            "- Feature engineering inadquat\n\n"
            "### Indicateurs\n"
            "- Performance faible sur le jeu d'entrainement\n"
            "- Performance egalement faible sur le jeu de test\n"
            "- Scores train et test proches mais bas\n\n"
            "### Solutions\n"
            "- Utiliser un modele plus complexe\n"
            "- Creer de nouvelles variables pertinentes\n"
            "- Reduire la regularisation\n"
            "- Augmenter le temps d'entrainement\n"
            "- Ameliorer la qualite des donnees\n\n"
            "Dans ChatAutoML-Bot, l'underfitting est detecte automatiquement lors de l'evaluation finale."
        )
    
    # Validation croisee
    if any(w in q for w in ['validation croisee', 'cross validation', 'k-fold', 'kfold', 'cv ']):
        return (
            "## Validation Croisee (Cross-Validation)\n\n"
            "### Definition\n"
            "La validation croisee est une technique d'evaluation qui permet d'estimer la performance "
            "d'un modele de maniere plus robuste qu'un simple split train/test.\n\n"
            "### Principe K-Fold\n"
            "1. Division du dataset en K parties (folds) egales (ex: K=5)\n"
            "2. Pour chaque iteration :\n"
            "   - Entrainement sur K-1 folds\n"
            "   - Validation sur le fold restant\n"
            "3. Moyenne des K scores de validation\n\n"
            "### Avantages\n"
            "- Utilise toutes les donnees pour l'entrainement et la validation\n"
            "- Reduit le risque de sur-optimisation sur un split particulier\n"
            "- Donne une estimation plus fiable de la generalisation\n"
            "- Permet de detecter l'instabilite du modele\n\n"
            "### Dans ChatAutoML-Bot\n"
            "- 5-fold par defaut (configurable)\n"
            "- Applique lors de la recherche de modeles\n"
            "- Garantit une evaluation plus representative des performances reelles"
        )
    
    # SMOTE / desequilibre
    if any(w in q for w in ['smote']):
        return (
            "## SMOTE - Synthetic Minority Oversampling Technique\n\n"
            "### Definition\n"
            "SMOTE est une methode de re-echantillonnage pour gerer le desequilibre des classes "
            "en generant de nouveaux exemples synthetiques.\n\n"
            "### Probleme de l'oversampling classique\n"
            "L'oversampling naive (duplication) cree des exemples identiques, ce qui :\n"
            "- Entraine le modele a memoriser les exemples minoritaires\n"
            "- Ne cree pas de diversite dans les donnees\n"
            "- Peut conduire a l'overfitting\n\n"
            "### Comment fonctionne SMOTE\n"
            "Au lieu de dupliquer, SMOTE **genere de nouveaux exemples synthetiques** :\n"
            "1. Pour chaque point de la classe minoritaire\n"
            "2. Trouver ses K plus proches voisins (generalement K=5)\n"
            "3. Prendre un point aleatoire sur le segment reliant chaque paire\n"
            "4. Ajouter ce nouveau point synthetique au dataset\n\n"
            "### Variantes dans ChatAutoML-Bot\n"
            "- **SMOTE-Tomek** : SMOTE + suppression des liens Tomek (nettoie les frontieres)\n"
            "- **SMOTE** : version standard\n"
            "- **Random Oversampling** : duplications aleatoires\n"
            "- **Random Undersampling** : suppression de points de la classe majoritaire\n\n"
            "### Quand utiliser SMOTE\n"
            "- Desequilibre des classes (ex: 95 % vs 5 %)\n"
            "- Classe minoritaire importante (ex: detection de fraude)\n"
            "- Volume de donnees limite\n\n"
            "### Limitations\n"
            "- Peut creer du bruit si les classes se chevauchent\n"
            "- Augmente le temps d'entrainement\n"
            "- Moins efficace sur donnees de tres haute dimension"
        )
    
    # Desequilibre general
    if any(w in q for w in ['desequilibre', 'deséquilibre', 'imbalance', 'classe majoritaire', 'classe minoritaire']):
        return (
            "## Desequilibre des Classes\n\n"
            "### Definition\n"
            "Un dataset est desequilibre lorsqu'une classe est beaucoup plus representee que les autres.\n\n"
            "### Exemple typique\n"
            "Detection de fraude : 99.9 % non-fraude, 0.1 % fraude\n"
            "Diagnostic medical : 95 % sains, 5 % malades\n\n"
            "### Pourquoi c'est un probleme\n"
            "Un modele naif peut predire la classe majoritaire a chaque coup et obtenir une accuracy "
            "elevee sans jamais detecter la classe minoritaire (la plus importante !).\n\n"
            "### Strategies dans ChatAutoML-Bot\n"
            "- **SMOTE** : generation d'exemples synthetiques (classe minoritaire)\n"
            "- **SMOTE-Tomek** : SMOTE + nettoyage des frontieres\n"
            "- **Oversampling** : duplication aleatoire\n"
            "- **Undersampling** : suppression de points de la classe majoritaire\n"
            "- **Auto** : detection automatique et choix de la meilleure strategie\n\n"
            "### Metriques adequates\n"
            "Pour desequilibre, eviter l'accuracy et preferer :\n"
            "- F1-Score (equilibre precision/recall)\n"
            "- ROC AUC (capacite de discrimination)\n"
            "- Precision-Recall AUC\n\n"
            "### Detection automatique\n"
            "ChatAutoML-Bot detecte automatiquement le desequilibre (> 70 % classe majoritaire) "
            "et propose les strategies adequates."
        )
    
    # Normalisation / Scaling
    if any(w in q for w in ['normalisation', 'standardisation', 'standardscaler', 'minmaxscaler', 'scaling', 'normaliser', 'standardiser']):
        return (
            "## Normalisation et Standardisation des Donnees\n\n"
            "### Definition\n"
            "Ces techniques mettent les variables numeriques a la meme echelle pour garantir "
            "que certains modeles fonctionnent correctement.\n\n"
            "### Pourquoi c'est necessaire\n"
            "Sans mise a l'echelle, les variables avec des plages de valeurs elevees dominent "
            "les calculs de distance et ponderent trop les modeles.\n\n"
            "### StandardScaler (Standardisation)\n"
            "- Transforme pour avoir moyenne = 0 et ecart-type = 1\n"
            "- Formule : z = (x - moyenne) / ecart_type\n"
            "- Modeles beneficiaires : SVM, KNN, Linear Models, PCA, Reseaux de neurones\n"
            "- Robuste pour les donnees normalement distribuees\n\n"
            "### MinMaxScaler (Normalisation Min-Max)\n"
            "- Redimensionne dans l'intervalle [0, 1]\n"
            "- Formule : x_scaled = (x - min) / (max - min)\n"
            "- Moins sensible aux outliers que StandardScaler\n"
            "- Utile pour les reseaux de neurones\n\n"
            "### Quand ne PAS scaler\n"
            "- Modeles bases sur des arbres (Decision Tree, Random Forest, XGBoost, LightGBM)\n"
            "- Ces modeles sont invariants par transformation monotone des variables\n\n"
            "### Dans ChatAutoML-Bot\n"
            "StandardScaler est applique automatiquement aux variables numeriques lors du pretraitement."
        )
    
    # Encodage
    if any(w in q for w in ['encodage', 'onehot', 'one-hot', 'one hot', 'label encoding', 'ordinal', 'categorielle', 'categorical']):
        return (
            "## Encodage des Variables Categorielles\n\n"
            "### Definition\n"
            "Les modeles de Machine Learning ne travaillent qu'avec des nombres. Il faut convertir "
            "les variables qualitatives (texte, categories) en variables numeriques.\n\n"
            "### One-Hot Encoding (utilise dans ChatAutoML-Bot)\n"
            "- Cree une nouvelle colonne binaire (0/1) par categorie\n"
            "- Exemple : Couleur = ['Rouge', 'Bleu', 'Vert'] devient 3 colonnes :\n"
            "  - Couleur_Rouge : 1 si Rouge sinon 0\n"
            "  - Couleur_Bleu : 1 si Bleu sinon 0\n"
            "  - Couleur_Vert : 1 si Vert sinon 0\n"
            "- Avantage : pas d'ordre artificiel entre categories\n"
            "- Desavantage : explosion du nombre de variables (cardinalite elevee)\n\n"
            "### Label Encoding\n"
            "- Assigne un entier unique a chaque categorie\n"
            "- Exemple : Rouge=0, Bleu=1, Vert=2\n"
            "- Attention : cree un ordre artificiel (Bleu > Rouge ?)\n"
            "- A utiliser seulement pour variables ordinales (ex: petit=0, moyen=1, grand=2)\n\n"
            "### Dans ChatAutoML-Bot\n"
            "One-Hot Encoding est applique automatiquement aux variables categorielles lors du pretraitement. "
            "Les nouvelles categories rencontrees lors de la prediction sont ignorees (handle_unknown='ignore')."
        )
    
    # Imputation
    if any(w in q for w in ['imputation', 'median', 'mean', 'moyenne', 'mediane', 'valeur manquante', 'nan']):
        return (
            "## Imputation des Valeurs Manquantes\n\n"
            "### Definition\n"
            "La plupart des modeles de Machine Learning ne gerent pas les valeurs manquantes (NaN). "
            "Il faut les remplacer par des valeurs plausibles.\n\n"
            "### Strategies utilisees dans ChatAutoML-Bot\n\n"
            "**Variables numeriques : Median**\n"
            "- Utilise la valeur centrale (50e percentile)\n"
            "- Robuste aux valeurs aberrantes (outliers)\n"
            "- Preferable a la moyenne pour les distributions asymetriques\n\n"
            "**Variables categorielles / texte : 'Unknown'**\n"
            "- Cree une categorie dediee pour les valeurs manquantes\n"
            "- Preserve l'information que la donnee etait manquante\n"
            "- Permet au modele d'apprendre des patterns sur donnees manquantes\n\n"
            "**Variables booleennes : Mode**\n"
            "- Utilise la valeur la plus frequente (True ou False)\n"
            "- Simple et efficace pour variables binaires\n\n"
            "### Autres methodes (non utilisees par defaut)\n"
            "- KNN Imputer : impute par les plus proches voisins\n"
            "- Iterative Imputer : regression entre variables\n"
            "- Forward fill / Back fill (series temporelles)\n"
            "- Suppression (si tres peu de donnees manquantes)\n\n"
            "### Pourquoi pas la moyenne ?\n"
            "La moyenne est sensible aux outliers et peut biaiser l'imputation pour les distributions asymetriques."
        )
    
    # Split train/test
    if any(w in q for w in ['split', 'train test', 'train/test', 'stratification', 'holdout', 'echantillon']):
        return (
            "## Split Train / Test\n\n"
            "### Definition\n"
            "Separer le dataset en deux parties independantes pour evaluer la capacite de generalisation du modele.\n\n"
            "### Principe\n"
            "- **Train set (generalement 70-80 %)** : pour entrainer le modele\n"
            "- **Test set (generalement 20-30 %)** : pour evaluer la performance finale\n"
            "- Le test set ne doit **jamais** etre utilise pendant l'entrainement ou l'optimisation\n\n"
            "### Parametres dans ChatAutoML-Bot\n"
            "- Taille de test : 20 % par defaut (configurable)\n"
            "- `random_state = 42` : pour la reproductibilite des resultats\n"
            "- **Stratification** (classification) : preserve la proportion de chaque classe dans train ET test\n"
            "- **Shuffle** : les donnees sont melangees avant le split (sauf series temporelles)\n\n"
            "### Pourquoi stratifier ?\n"
            "Sans stratification, on risque d'avoir 0 exemple de la classe minoritaire dans le test set, "
            "ce qui fausse l'evaluation finale et la metrique.\n\n"
            "### Importance de la separation\n"
            "- Evite l'overfitting sur les donnees d'entrainement\n"
            "- Simule la performance sur de nouvelles donnees\n"
            "- Permet de detecter le memorisation (overfitting) vs apprentissage reel"
        )
    
    # Metriques classification
    if any(w in q for w in ['accuracy', 'precision', 'rappel', 'recall', 'f1', 'f1-score', 'auc', 'roc', 'matrice de confusion', 'confusion']):
        return (
            "## Metriques d'Evaluation - Classification\n\n"
            "### Matrice de Confusion\n"
            "Tableau qui compare les predictions aux valeurs reelles :\n"
            "- **VP (Vrais Positifs)** : predit positif et reellement positif\n"
            "- **VN (Vrais Negatifs)** : predit negatif et reellement negatif\n"
            "- **FP (Faux Positifs)** : predit positif mais reellement negatif (erreur type I)\n"
            "- **FN (Faux Negatifs)** : predit negatif mais reellement positif (erreur type II)\n\n"
            "### Metriques Principales\n\n"
            "**Accuracy (Exactitude)**\n"
            "- Formule : (VP + VN) / Total\n"
            "- Pourcentage de bonnes reponses\n"
            "- Attention : trompeur sur datasets desequilibres\n"
            "- Exemple : 99.9 % accuracy sur detection de fraude mais ne detecte aucune fraude\n\n"
            "**Precision**\n"
            "- Formule : VP / (VP + FP)\n"
            "- Parmi les predits positifs, combien sont vraiment positifs ?\n"
            "- Importe quand un FP est couteux (ex: classer un email important comme spam)\n\n"
            "**Recall / Rappel (Sensibilite)**\n"
            "- Formule : VP / (VP + FN)\n"
            "- Parmi les vrais positifs, combien en a-t-on detecte ?\n"
            "- Importe quand un FN est critique (ex: manquer une detection de fraude ou de cancer)\n\n"
            "**F1-Score**\n"
            "- Formule : 2 * (Precision * Recall) / (Precision + Recall)\n"
            "- Moyenne harmonique de Precision et Recall\n"
            "- Bon equilibre entre les deux metriques\n"
            "- F1 macro : moyenne simple sur toutes les classes\n"
            "- F1 weighted : moyenne ponderee par le nombre d'exemples par classe\n\n"
            "**ROC AUC**\n"
            "- Surface sous la courbe ROC\n"
            "- Mesure la capacite du modele a distinguer les classes\n"
            "- 0.5 = modele aleatoire, 1.0 = modele parfait\n"
            "- Robuste au desequilibre des classes\n\n"
            "### Dans ChatAutoML-Bot\n"
            "Le systeme calcule automatiquement toutes ces metriques lors de l'evaluation finale "
            "et selectionne la meilleure selon la tache (F1 macro pour classification, R² pour regression)."
        )
    
    # Metriques regression
    if any(w in q for w in ['r2', 'r-squared', 'r carre', 'rmse', 'mae', 'mse', 'mape', 'regression']):
        return (
            "## Metriques d'Evaluation - Regression\n\n"
            "### Definition\n"
            "Pour evaluer un modele de regression (prediction d'une valeur continue comme un prix, une temperature, etc.).\n\n"
            "### MAE - Mean Absolute Error\n"
            "- Formule : moyenne de |y_pred - y_true|\n"
            "- Robuste aux outliers (valeurs aberrantes)\n"
            "- Interpretable : dans l'unite de la variable cible\n"
            "- Exemple : MAE = 5€ signifie erreur moyenne de 5€\n\n"
            "### MSE - Mean Squared Error\n"
            "- Formule : moyenne de (y_pred - y_true)²\n"
            "- Penalise fortement les grandes erreurs (car carre)\n"
            "- Unite : carre de l'unite de la cible (difficile a interpreter)\n"
            "- Utile pour l'optimisation mathematique\n\n"
            "### RMSE - Root Mean Squared Error\n"
            "- Formule : racine carree du MSE\n"
            "- Retourne dans l'unite de la cible (interpretable)\n"
            "- Penalise les grandes erreurs (plus sensible aux outliers que MAE)\n"
            "- Preferred quand les grandes erreurs sont critiques\n\n"
            "### R² - Coefficient de Determination\n"
            "- Proportion de variance expliquee par le modele\n"
            "- Interpretation :\n"
            "  - R² = 1 : modele parfait (100 % de variance expliquee)\n"
            "  - R² = 0 : aussi bon que predire la moyenne\n"
            "  - R² < 0 : pire que predire la moyenne (probleme)\n"
            "- Attention : R² augmente mecaniquement avec le nombre de variables\n"
            "- Pour comparer modeles : utiliser R² ajuste ou RMSE\n\n"
            "### Dans ChatAutoML-Bot\n"
            "Le systeme calcule R², RMSE et MAE automatiquement lors de l'evaluation finale. "
            "Le meilleur modele est selectionne selon R² pour la regression."
        )
    
    # Feature importance
    if any(w in q for w in ['feature importance', 'importance des variables', 'variables importantes', 'features']):
        return (
            "## Importance des Variables (Feature Importance)\n\n"
            "### Definition\n"
            "L'importance des variables quantifie la contribution de chaque caracteristique (feature) "
            "aux predictions du modele.\n\n"
            "### Comment ca fonctionne (modeles a base d'arbres)\n"
            "Chaque fois qu'un arbre effectue une separation sur une variable, on mesure la reduction "
            "d'impurete (Gini, Entropy, MSE) apportee. On somme ces reductions sur tous les arbres et tous les splits.\n\n"
            "### Interpretation\n"
            "- **Haute importance** : fort impact sur la prediction\n"
            "- **Importance nulle/quasi-nulle** : variable n'apporte rien\n"
            "- **Possibilite** : supprimer les variables inutiles pour simplifier le modele\n\n"
            "### Methodes alternatives\n"
            "- **SVM lineaire** : utiliser les coefficients\n"
            "- **Permutation Importance** : mesurer l'impact de la permutation aleatoire\n"
            "- **SHAP values** : explication locale et globale des predictions\n\n"
            "### Limitations\n"
            "- Variables correlees peuvent etre sous-estimees\n"
            "- Importance depend du type de modele\n"
            "- Ne montre pas le DIRECTION de l'impact (positif/negatif)\n\n"
            "### Dans ChatAutoML-Bot\n"
            "L'importance est calculee automatiquement pour Random Forest, XGBoost, Gradient Boosting, "
            "Decision Tree et Logistic Regression (coefficients). Un graphique est affiche lors de l'evaluation finale."
        )
    
    # Grid Search / Random Search
    if any(w in q for w in ['grid search', 'random search', 'hyperparametre', 'hyperparamètre', 'hyperparameter', 'optimisation d hyperparametres']):
        return (
            "## Optimisation des Hyperparametres\n\n"
            "### Definition\n"
            "Les hyperparametres sont les parametres du modele fixes AVANT l'entrainement "
            "(ex: profondeur max d'un arbre, taux d'apprentissage, nombre d'arbres...).\n"
            "Ils different des parametres appris (ex: poids d'un reseau) qui sont ajustes pendant l'entrainement.\n\n"
            "### Grid Search (utilise par defaut dans ChatAutoML-Bot)\n"
            "- Teste **TOUTES** les combinaisons d'une grille pre-definie\n"
            "- Avantage : exhaustif, trouve l'optimum global dans la grille\n"
            "- Desavantage : tres lent quand il y a beaucoup d'hyperparametres (explosion combinatoire)\n"
            "- Example : 3 valeurs × 4 hyperparametres = 3^4 = 81 combinaisons\n\n"
            "### Random Search\n"
            "- Teste des combinaisons aleatoires pendant N iterations\n"
            "- Avantage : beaucoup plus rapide en grande dimension\n"
            "- Trouve generalement une bonne solution (pas forcement l'optimum global)\n"
            "- Dans ChatAutoML-Bot : N_ITER = 20 par defaut\n"
            "- Efficace quand certains hyperparametres sont plus importants que d'autres\n\n"
            "### Autres methodes (non implementees)\n"
            "- **Bayesian Optimization** (Optuna, Hyperopt) : utilise un modele de substitution\n"
            "- **Genetic Algorithms** : evolution d'une population de solutions\n"
            "- **Halving Grid Search** : elimine rapidement les mauvaises combinaisons\n\n"
            "### Dans ChatAutoML-Bot\n"
            "Grid Search est utilise par defaut avec validation croisee 5-fold. "
            "Random Search est disponible via la configuration pour des datasets plus volumineux."
        )
    
    # Outliers
    if any(w in q for w in ['outlier', 'valeur aberrante', 'aberrante', 'extreme value']):
        return (
            "## Outliers - Valeurs Aberrantes\n\n"
            "### Definition\n"
            "Un outlier est une observation qui differe significativement des autres points de donnees.\n\n"
            "### Methodes de Detection\n\n"
            "**IQR Method (utilise dans ChatAutoML-Bot)**\n"
            "- Calculer Q1 (25e percentile) et Q3 (75e percentile)\n"
            "- IQR = Q3 - Q1\n"
            "- Outliers si < Q1 - 1.5*IQR ou > Q3 + 1.5*IQR\n"
            "- Robuste et ne necessite pas de distribution normale\n\n"
            "**Z-score**\n"
            "- Valeur eloignee de plus de 3 ecarts-types de la moyenne\n"
            "- Necessite une distribution normale\n"
            "- Sensible aux extremes eux-memes\n\n"
            "**Methodes avancees**\n"
            "- Isolation Forest : methode basee sur les arbres\n"
            "- DBSCAN : methode de clustering\n"
            "- Local Outlier Factor (LOF)\n\n"
            "### Comment Gerer les Outliers\n\n"
            "**Conserver**\n"
            "- Si legitimes (ex: chiffre d'affaires exceptionnel)\n"
            "- Representent des phenomenes reels et importants\n\n"
            "**Traiter**\n"
            "- Winsorization : remplacer par le percentile le plus proche (1er ou 99e)\n"
            "- Log transformation : pour distributions tres asymetriques\n"
            "- Capping : limiter a une valeur maximale raisonnable\n\n"
            "**Supprimer**\n"
            "- Seulement si erreur de saisie evidente\n"
            "- Si tres peu d'outliers (< 1 %)\n\n"
            "**Modeles robustes**\n"
            "- Arbres de decision (insensibles aux outliers)\n"
            "- Huber Regressor, Quantile Regression\n"
            "- Random Forest, XGBoost (robustes)\n\n"
            "### Dans ChatAutoML-Bot\n"
            "Les outliers sont detectes automatiquement lors de l'analyse initiale via la methode IQR. "
            "Les resultats sont affiches dans les statistiques descriptives."
        )
    
    return None


def generate_response(intent: Dict[str, Any], context: Dict[str, Any], 
                     llm_explainer, message: str = "") -> str:
    """Genere une reponse basee sur l'intention detectee"""
    
    intent_type = intent.get('intent', 'general')
    
    try:
        import config.config as cfg
        chatbot_name = cfg.CHATBOT_NAME
    except (ImportError, AttributeError):
        chatbot_name = "ChatAutoML-Bot"
    
    # D'abord, verifier si c'est une question conceptuelle ML
    concept_expl = _get_concept_explanation(message or "")
    if concept_expl is not None and intent_type in ['question', 'general']:
        return concept_expl
    
    if intent_type == 'greeting':
        dataset = context.get('dataset')
        target_column = context.get('target_column')
        automl_done = bool(context.get('automl_done', False))
        selection_result = context.get('selection_result')
        
        if dataset is not None and target_column is not None and automl_done:
            best_name = None
            try:
                best_name = (selection_result or {}).get('best_model_name')
            except Exception:
                best_name = None
            
            n_rows = None
            n_cols = None
            try:
                n_rows = int(dataset.shape[0])
                n_cols = int(dataset.shape[1])
            except Exception:
                pass
            
            test_acc = None
            try:
                eval_ctx = context.get('evaluation') or {}
                test_m = eval_ctx.get('test_metrics') or {}
                if 'accuracy' in test_m:
                    test_acc = float(test_m['accuracy'])
                elif 'r2' in test_m:
                    test_acc = float(test_m['r2'])
            except Exception:
                pass
            
            return f"""Bonjour.

**Etat actuel : Pipeline AutoML termine**
- Dataset charge : Oui ({n_rows} lignes x {n_cols} colonnes)
- Colonne cible : {target_column}
- Meilleur modele selectionne : {best_name or 'Indisponible'}
""" + (f"- Performance test : {test_acc:.4f}" if test_acc is not None else "") + f"""

**Pour explorer les resultats, vous pouvez demander :**
- "Explique la comparaison des modeles"
- "Detaille l'evaluation finale"
- "Quelles sont les variables les plus importantes ?"
- "Refais le resume global"
- "Comment puis-je ameliorer ces resultats ?"
"""

        if dataset is not None and target_column is not None:
            task_type = context.get('task_type', 'inconnu')
            return f"""Bonjour.

**Etat actuel :**
- Dataset : Charge
- Colonne cible : {target_column}
- Type de tache detecte : {task_type}

**Prochaine etape :**
Cliquez sur le bouton "Lancer AutoML" dans le menu lateral, ou dites directement "lancer automl".

Le pipeline executera automatiquement :
1. Analyse descriptive du dataset
2. Pretraitement (nettoyage, encodage, scaling)
3. Gestion eventuelle du desequilibre des classes
4. Recherche de modeles + optimisation d'hyperparametres
5. Selection du meilleur modele
6. Evaluation detailed + resume
"""
        
        elif dataset is not None:
            return f"""Bonjour.

**Etat actuel :** Dataset charge avec succes.

**Prochaine etape obligatoire :**
Selectionnez la **colonne cible** dans le menu lateral (liste deroulante "Choisir la colonne cible").

La colonne cible est la variable que vous voulez predire (ex: "churn", "prix", "diagnostic", etc.).
Une fois selectionnee, je detecterai automatiquement s'il s'agit d'une classification ou d'une regression.
"""
        
        else:
            return f"""Bonjour. Je suis {chatbot_name}, votre assistant pour les pipelines d'AutoML sur donnees tabulaires.

**Pour commencer :**
1. Utilisez le menu lateral
2. Cliquez sur "Charger un Dataset"
3. Uploadez votre fichier (formats supportes : CSV, Excel .xlsx/.xls, Parquet, JSON)
4. Selectionnez la colonne cible a predire
5. Laissez-vous guider par le pipeline automatique

A tout moment, vous pouvez me poser des questions sur les concepts de Machine Learning :
- Overfitting, underfitting, validation croisee
- SMOTE, normalisation, encodage
- Metriques (accuracy, F1, R², RMSE...)
- Feature importance, hyperparametres
"""
    
    elif intent_type == 'status_check':
        dataset = context.get('dataset')
        target_column = context.get('target_column')
        analysis_done = bool(context.get('analysis_done', False))
        preprocessing_done = bool(context.get('preprocessing_done', False))
        automl_done = bool(context.get('automl_done', False))

        check = lambda v: "Effectue" if v else "A faire"
        lines = []
        lines.append(f"- Dataset charge : {check(dataset is not None)}")
        lines.append(f"- Colonne cible selectionnee : {check(target_column is not None)}")
        if target_column:
            lines.append(f"    -> Colonne : {target_column}")
        lines.append(f"- Analyse initiale : {check(analysis_done)}")
        lines.append(f"- Pretraitement : {check(preprocessing_done)}")
        lines.append(f"- AutoML (recherche + selection + evaluation) : {check(automl_done)}")

        if automl_done:
            best_name = None
            try:
                best_name = (context.get('selection_result') or {}).get('best_model_name')
            except Exception:
                pass
            next_step = (
                "Oui, le pipeline est termine. Le meilleur modele selectionne est "
                + (best_name or "indisponible")
                + ". Vous pouvez consulter le Resume global en bas de page, ou me demander d'expliquer une partie specifique."
            )
        elif preprocessing_done:
            next_step = "Pipeline en cours d'execution. Prochaine etape : lancez l'AutoML (bouton dans le menu lateral ou dites-moi \"lancer automl\")."
        elif analysis_done:
            next_step = "Analyse terminee. Prochaine etape : lancez le pretraitement (bouton \"Lancer le pretraitement\" ou dites \"lance le pretraitement\")."
        elif dataset is not None and target_column:
            next_step = "Dataset et cible prets. Prochaine etape : cliquez sur \"Analyser le Dataset\" ou dites \"lance l'analyse\"."
        elif dataset is not None:
            next_step = "Dataset charge. Il faut maintenant selectionner la colonne cible dans le menu lateral."
        else:
            next_step = "Aucune etape demarree. Commencez par charger un dataset (menu lateral)."

        return "**Etat d'avancement du pipeline AutoML :**\n\n" + "\n".join(lines) + "\n\n" + next_step

    elif intent_type == 'explain_step':
        entities = intent.get('entities') or {}
        step = entities.get('step')
        section = entities.get('section')
        if step:
            return f"Demande enregistree. Je vais reafficher l'etape {step}/6 avec l'ensemble des tableaux, graphiques et commentaires associes."
        if section:
            labels = {
                'analysis': 'Analyse initiale du dataset',
                'preprocessing': 'Pretraitement des donnees',
                'imbalance': 'Gestion du desequilibre des classes',
                'model_comparison': 'Comparaison des modeles (validation croisee)',
                'evaluation': 'Evaluation finale (train vs test)',
                'feature_importance': 'Importance des variables',
                'final_summary': 'Resume global du pipeline',
            }
            label = labels.get(section, section)
            return f"Demande enregistree. Je vais reafficher la section : **{label}** avec l'ensemble des elements associes (tableaux, graphiques, commentaires)."
        return "Demande prise en compte. Pour affiner : quelle section voulez-vous revoir ? Analyse, pretraitement, desequilibre, comparaison de modeles, evaluation finale, importance des variables, ou resume global."

    elif intent_type == 'redo_step':
        entities = intent.get('entities') or {}
        section = entities.get('section')
        step = entities.get('step')
        
        if step:
            return f"OK. Je vais relancer l'etape {step}/6 du pipeline depuis le debut, ce qui effacera les resultats ulterieurs."
        
        if section:
            labels = {
                'analysis': 'l\'analyse du dataset',
                'preprocessing': 'le pretraitement',
                'imbalance': 'la gestion du desequilibre',
                'model_comparison': 'la recherche et la comparaison des modeles',
                'evaluation': 'l\'evaluation finale',
                'automl': 'tout le pipeline AutoML',
                'final_summary': 'le resume global',
            }
            label = labels.get(section, section)
            return f"OK. Je vais relancer {label} depuis le debut. Les resultats des etapes suivantes seront effaces et recalcules."
        return "OK. Quelle partie voulez-vous relancer ? Tapez par exemple \"refais l'analyse\" ou \"relance l'AutoML\"."
    
    elif intent_type == 'load_data':
        return """**Comment charger un dataset :**

1. Allez dans le **menu lateral** a gauche de l'ecran
2. Developpez la section **"Charger un Dataset"**
3. Cliquez sur **"Browse files"** ou glissez-déposez votre fichier
4. Formats supportes :
   - CSV (.csv) - detection automatique du separateur (virgule, point-virgule, tabulation)
   - Excel (.xlsx, .xls)
   - Parquet (.parquet)
   - JSON (.json)
5. Taille maximale : 100 Mo par fichier

Apres le chargement, vous devrez selectionner la **colonne cible** dans la liste deroulante qui apparaitra en dessous du fichier charge.
"""
    
    elif intent_type == 'preprocessing':
        if context.get('dataset') is None:
            return """**Pretraitement (nettoyage + preparation des donnees)**

Le pretraitement sera execute automatiquement lors du lancement de l'AutoML, apres avoir charge un dataset et selectionne la colonne cible.

Etapes effectuees :
1. Detection automatique des types de colonnes (numerique, categorielle, booleenne)
2. Imputation des valeurs manquantes :
   - Numeriques : mediane (robuste aux outliers)
   - Categorielles : categorie dediee "Unknown"
   - Booleennes : mode (valeur la plus frequente)
3. Encodage des variables categorielles : One-Hot Encoding
4. Standardisation des variables numeriques : StandardScaler (moyenne=0, ecart-type=1)
5. Split train/test (80 % / 20 %) avec stratification si classification
"""
        else:
            return """**Pretraitement - pipeline applique :**

1. **Types detectes** : Numerique, categorielle, booleenne
2. **Valeurs manquantes** :
   - Numeriques -> mediane
   - Categorielles -> "Unknown"
   - Booleennes -> mode
3. **Encodage** : One-Hot Encoding pour toutes les variables categorielles
4. **Scaling** : StandardScaler sur les variables numeriques
5. **Split** : train 80 % / test 20 %, random_state=42, stratification (classification)

Pour lancer concretement le pretraitement : utilisez le bouton "Lancer le pretraitement" dans le menu lateral, ou dites "lance le pretraitement".
"""
    
    elif intent_type == 'run_automl':
        if context.get('dataset') is None:
            return """**Lancement de l'AutoML - prerequis manquants**

Avant de lancer l'AutoML, il faut :
1. **Charger un dataset** (CSV, Excel, Parquet, JSON) via le menu lateral
2. **Selectionner la colonne cible** a predire

Faites d'abord ces deux etapes, puis revenez me voir ou cliquez sur "Lancer AutoML".
"""
        elif context.get('target_column') is None:
            return """**Lancement de l'AutoML - colonne cible manquante**

Le dataset est bien charge, mais il faut specifier **quelle colonne vous voulez predire**.

Dans le menu lateral :
1. Developpez "Selectionner la colonne cible"
2. Choisissez dans la liste deroulante la variable a predire (ex: "churn", "prix", "diagnostic")

Une fois selectionnee, le type de tache (classification ou regression) est detecte automatiquement, et vous pourrez lancer l'AutoML.
"""
        else:
            return """**Lancement de l'AutoML**

Pour demarrer le pipeline complet : cliquez sur le bouton **"Lancer AutoML"** dans le menu lateral, ou dites "lance automl".

Le pipeline va executer successivement :

**1. Analyse du dataset** (etape 1)
   - Shape, types de colonnes
   - Valeurs manquantes, doublons
   - Statistiques descriptives
   - Distribution de la variable cible

**2. Pretraitement** (etape 2)
   - Nettoyage, imputation, encodage, scaling
   - Split train/test stratifie

**3. Gestion du desequilibre** (classification uniquement, si besoin)
   - Detection automatique du seuil (> 70 % classe majoritaire)
   - SMOTE-Tomek par defaut (ou strategie choisie)

**4. Recherche de modeles** (etape 3)
   - Test de plusieurs algorithmes
   - Grid Search ou Random Search sur les hyperparametres
   - Validation croisee 5-fold

**5. Selection du meilleur modele** (etape 4)
   - Classement selon la metrique principale (F1 macro / R²)
   - Tableau comparatif + graphique

**6. Evaluation finale + explications** (etapes 5-6)
   - Metriques train vs test
   - Diagnostic overfitting / underfitting
   - Matrice de confusion / graphique
   - Feature importance
   - Resume global + suggestions d'amelioration
"""
    
    elif intent_type == 'reset_pipeline':
        return (
            "Demande de reinitialisation enregistree. Je peux :\n\n"
            "- **Option A** : Garder le dataset charge, mais re-executer tout le pipeline "
            "(analyse, pretraitement, AutoML, resultats, chat partiellement garde)\n"
            "- **Option B** : Reinitialiser completement la session "
            "(dataset supprime, cible perdue, resultats effaces, chat vide)\n\n"
            "Pour confirmer, repondez simplement par la lettre : `A` ou `B`."
        )

    elif intent_type == 'reset_session':
        return (
            "Demande de reinitialisation complete de la session enregistree.\n\n"
            "Etes-vous sur de vouloir TOUT effacer ?\n"
            "- Dataset charge -> PERDU\n"
            "- Colonne cible selectionnee -> PERDUE\n"
            "- Resultats d'analyse, pretraitement, AutoML -> PERDUS\n"
            "- Historique du chat -> VIDE\n\n"
            "Confirmez par : `oui` (tout reinitialiser) ou `non` (garder le dataset, seulement refaire le pipeline)."
        )

    elif intent_type == 'change_theme':
        theme = (intent.get('entities') or {}).get('theme')
        if theme in ('sombre', 'clair'):
            return f"Changement de theme enregistre : passage au theme **{theme}**."
        return "Quel theme voulez-vous appliquer ? **Sombre** (dark) ou **Clair** (light) ?"

    elif intent_type == 'feedback_negative':
        return (
            "Message recu. Merci pour ce retour. Pour vous aider au mieux, "
            "pouvez-vous preciser quel element ne vous convient pas ?\n\n"
            "Vous pouvez repondre directement avec une description, ou choisir un chiffre :\n"
            "1. L'execution est trop lente\n"
            "2. Probleme pour charger le dataset ou selectionner la cible\n"
            "3. Les resultats de l'AutoML ne sont pas clairs\n"
            "4. L'interface ou le style vous semble inadequat\n"
            "5. Le chatbot ne comprend pas vos demandes\n"
            "6. Autre probleme (a decrire en quelques mots)"
        )

    elif intent_type == 'help':
        return """**Aide - Guide d'utilisation de ChatAutoML-Bot**

=== Flux de travail recommande ===

1. **Charger un dataset** : menu lateral -> bouton Browse files
   - Formats : CSV, Excel (.xlsx/.xls), Parquet, JSON
   - Max 100 Mo

2. **Choisir la colonne cible** : liste deroulante dans le menu lateral
   - Variable que vous voulez predire

3. **Lancer l'analyse** -> bouton "Analyser le Dataset"
   - Affiche shape, types, valeurs manquantes, stats, distribution cible

4. **Lancer le pretraitement** -> bouton "Lancer le pretraitement"
   - Nettoyage, encodage, scaling, split train/test

5. **Lancer l'AutoML** -> bouton "Lancer AutoML"
   - Recherche de modeles, selection, evaluation, explications

=== Commandes conversationnelles disponibles ===

Salutations et statut :
- "Bonjour", "Salut", "Hello"
- "Ou en est-on ?", "C'est termine ?", "Tu as fini ?"

Relancer / revoir une partie :
- "Explique l'analyse" / "Reviens sur l'evaluation" / "Detaille la comparaison"
- "Refais le pretraitement" / "Relance l'AutoML" / "Recommence depuis l'analyse"
- "Reaffiche le resume global"

Questions techniques (reponse detaillee systematique) :
- Overfitting, underfitting, generalisation
- Validation croisee, K-Fold, CV
- SMOTE, oversampling, undersampling, desequilibre
- Normalisation, StandardScaler, MinMaxScaler, scaling
- Encodage, One-Hot, Label Encoding
- Imputation, valeurs manquantes, NaN
- Split train/test, stratification
- Metriques : accuracy, precision, recall, F1, ROC AUC, R², RMSE, MAE
- Feature importance, variables importantes
- Grid search, random search, hyperparametres
- Outliers, valeurs aberrantes

Reset :
- "Reset la session"
- "Reinitialise le pipeline"
- "Tout supprimer"

Aide :
- "Aide", "Help", "Comment ca marche ?", "Que peux-tu faire ?"
"""
    
    elif intent_type == 'ask_about_data':
        analysis_info = context.get('analysis_info')
        if analysis_info:
            rows = analysis_info.get('shape', {}).get('rows', '?')
            cols = analysis_info.get('shape', {}).get('columns', '?')
            target = analysis_info.get('target_info', {}).get('column', None)
            miss_vals = analysis_info.get('missing_values', {})
            cols_with_miss = [k for k, v in miss_vals.items() if v.get('count', 0) > 0]
            
            n_num = 0
            n_cat = 0
            for c, info in (analysis_info.get('columns') or {}).items():
                t = info.get('type')
                if t == 'numeric':
                    n_num += 1
                elif t == 'categorical':
                    n_cat += 1
            
            resp = f"**Resume du dataset (analyse effectuee) :**\n\n"
            resp += f"- Dimensions : {rows} lignes x {cols} colonnes\n"
            resp += f"- Colonnes numeriques : {n_num}\n"
            resp += f"- Colonnes categorielles / texte : {n_cat}\n"
            if target:
                resp += f"- Colonne cible : {target}\n"
            
            if cols_with_miss:
                top = sorted(cols_with_miss, key=lambda k: -miss_vals[k].get('count', 0))[:5]
                top_str = ", ".join([f"{k} ({miss_vals[k].get('count', 0)} v., {miss_vals[k].get('percentage', 0):.1f} %)" for k in top])
                resp += f"- Colonnes avec valeurs manquantes : {len(cols_with_miss)} au total. Plus impactees : {top_str}\n"
            else:
                resp += "- Valeurs manquantes : aucune detectee\n"
            
            resp += "\nPour une exploration plus fine, vous pouvez demander : \"montre-moi les valeurs manquantes\" ou \"statistiques descriptives\"."
            return resp
        elif context.get('dataset') is not None:
            df = context.get('dataset')
            try:
                return (
                    f"Dataset charge mais analyse non encore lancee.\n"
                    f"Shape detectee : {int(df.shape[0])} lignes x {int(df.shape[1])} colonnes.\n"
                    f"Lancez l'analyse (bouton dans le menu lateral ou \"lance l'analyse\") pour obtenir les statistiques detaillees."
                )
            except Exception:
                pass
        return """**Questions sur les donnees**

Pour obtenir des informations sur le dataset :
1. Commencez par charger un fichier via le menu lateral
2. Lancez l'analyse (bouton "Analyser le Dataset" ou dites "lance l'analyse")

Vous obtiendrez alors : dimensions, types de colonnes, valeurs manquantes, doublons, statistiques descriptives, distribution de la cible.
"""
    
    elif intent_type == 'ask_about_models':
        selection_result = context.get('selection_result')
        if selection_result:
            best_name = selection_result.get('best_model_name', 'Inconnu')
            best_score = selection_result.get('best_score', None)
            comp_df = selection_result.get('comparison_summary')
            
            resp = f"**Resultats de la recherche de modeles :**\n\n"
            resp += f"Modele selectionne : **{best_name}**\n"
            if best_score is not None:
                try:
                    resp += f"Score CV (validation croisee) : {float(best_score):.4f}\n"
                except Exception:
                    resp += f"Score CV : {best_score}\n"
            
            if comp_df is not None and hasattr(comp_df, 'head'):
                try:
                    n_models = int(comp_df.shape[0])
                    resp += f"Nombre total de modeles testes : {n_models}\n"
                except Exception:
                    pass
            
            test_m = (context.get('evaluation') or {}).get('test_metrics') or {}
            if test_m:
                resp += "\n**Performance sur le test set :**\n"
                for k, v in list(test_m.items())[:8]:
                    try:
                        fv = float(v)
                        resp += f"- {k} : {fv:.4f}\n"
                    except Exception:
                        resp += f"- {k} : {v}\n"
            
            resp += "\nPour une vue plus detaillee, vous pouvez demander :\n"
            resp += "- \"Montre-moi la comparaison complete des modeles\"\n"
            resp += "- \"Explique-moi l'evaluation train vs test\"\n"
            resp += "- \"Quelles sont les variables les plus importantes ?\"\n"
            resp += "- \"Comment ameliorer le modele ?\""
            return resp
        else:
            return """**Questions sur les modeles**

Les resultats de modelisation ne sont pas encore disponibles.

**Ordre des etapes :**
1. Charger dataset + colonne cible
2. (Optionnel) Lancer l'analyse
3. Lancer le pretraitement
4. **Lancer l'AutoML** (bouton dans le menu lateral)

Une fois l'AutoML termine, vous pourrez consulter :
- Le meilleur modele et son score CV
- Le tableau comparatif de tous les modeles testes
- Les performances train vs test
- L'eventuel overfitting / underfitting
- L'importance des variables
"""
    
    elif intent_type == 'question':
        # Concept deja traite en haut si concerne
        # Sinon repondre avec l'explainer pour les questions contextuelles
        try:
            if llm_explainer is not None:
                llm_resp = llm_explainer.answer_question(message, context)
                if isinstance(llm_resp, str) and llm_resp.strip() and "Posez-moi une question plus specifique" not in llm_resp:
                    return llm_resp
        except Exception:
            pass
        
        # Fallback selon l'etat du pipeline
        dataset = context.get('dataset')
        target = context.get('target_column')
        automl_done = bool(context.get('automl_done', False))
        
        if automl_done:
            return (
                "Question recue. Si elle concerne un concept de Machine Learning specifique "
                "(SMOTE, overfitting, metrics...), reformulez en utilisant les mots-cles. "
                "Sinon, pour explorer les resultats existants, essayez :\n"
                "- \"Explique la comparaison des modeles\"\n"
                "- \"Montre-moi l'evaluation\"\n"
                "- \"Quelles sont les variables les plus importantes ?\"\n"
                "- \"Resume global\"\n"
                "- \"Comment ameliorer ?\""
            )
        elif dataset and target:
            return (
                "Question recue. Dataset et cible sont charges. Prochaines actions possibles :\n"
                "- \"Lance l'analyse\" -> pour explorer les donnees\n"
                "- \"Lance le pretraitement\" -> nettoyer et preparer\n"
                "- \"Lance automl\" -> executer tout le pipeline"
            )
        elif dataset:
            return (
                "Question recue. Dataset charge. Prochaine etape obligatoire : "
                "selectionner la **colonne cible** dans le menu lateral (variable a predire)."
            )
        else:
            return (
                "Question recue. Aucun dataset n'est actuellement charge. "
                "Commencez par uploader un fichier CSV/Excel/Parquet/JSON via le menu lateral. "
                "Tapez \"Aide\" pour la liste complete des commandes."
            )
    
    else:
        # Fallback intelligent : concept ML d'abord
        concept = _get_concept_explanation(message or "")
        if concept is not None:
            return concept
        
        try:
            if llm_explainer is not None and (message or '').strip():
                natural = llm_explainer.answer_question(message, context)
                if isinstance(natural, str) and natural.strip():
                    # Ne pas retourner la reponse generique "posez une question specifique"
                    if "Posez-moi une question plus specifique" not in natural and "Je peux vous aider" not in natural:
                        return natural
        except Exception:
            pass

        dataset = context.get('dataset')
        target = context.get('target_column')
        automl_done = bool(context.get('automl_done', False))
        
        if automl_done:
            return (
                "Je n'ai pas identifie de commande ou de concept precis. "
                "Le pipeline AutoML est termine. Essayez par exemple :\n"
                "- \"Explique la comparaison des modeles\"\n"
                "- \"Montre-moi l'evaluation finale\"\n"
                "- \"Importance des variables\"\n"
                "- \"Resume global\"\n"
                "- Ou tapez \"Aide\" pour la liste complete."
            )
        if dataset and target:
            return (
                "Je n'ai pas identifie de commande ou de concept precis. "
                "Dataset et colonne cible sont prets.\n"
                "Prochaines etapes suggerees : \"lance l'analyse\", \"lance le pretraitement\" ou directement \"lance automl\"."
            )
        if dataset:
            return (
                "Je n'ai pas identifie de commande ou de concept precis. "
                "Dataset charge. Choisissez maintenant la **colonne cible** dans le menu lateral."
            )
        
        return (
            "Je n'ai pas identifie de commande ou de concept precis. "
            "Pour demarrer : chargez un dataset via le menu lateral, ou tapez \"Aide\" pour la liste complete des commandes."
        )
