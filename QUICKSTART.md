# 🚀 QUICKSTART — ChatAutoML-Bot

> **Lancez votre première analyse ML en moins de 5 minutes, sans écrire une ligne de code.**

---

## Option 1 — Démo live (aucune installation)

👉 **[Accéder à l'application](https://lbx6ryyhzigbsh3d5uwiyg.streamlit.app/)**

Uploadez `demo_data.csv` (inclus dans ce repo), sélectionnez `quitte_entreprise` comme colonne cible, et cliquez **Lancer AutoML**.

---

## Option 2 — Lancement local (Python)

```bash
# 1. Cloner le repo
git clone https://github.com/KalsoumDS/ChatAutoML-Bot.git
cd ChatAutoML-Bot

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run streamlit_app.py
```

Ouvre ton navigateur sur : **http://localhost:8501**

---

## Option 3 — Lancement avec Docker (recommandé en production)

```bash
# Build
docker build -t chatautoml .

# Run
docker run -p 8501:8501 chatautoml
```

Ouvre ton navigateur sur : **http://localhost:8501**

---

## Utiliser l'application en 4 étapes

1. **Upload** → Charge ton fichier CSV (ou utilise `demo_data.csv` fourni)
2. **Cible** → Sélectionne la colonne à prédire dans le menu latéral
3. **Analyse** → Lance le pipeline : preprocessing → benchmark 8 modèles → optimisation Optuna
4. **Export** → Télécharge le rapport complet (métriques, feature importance, recommandations)

---

## Fichier de démonstration inclus

`demo_data.csv` — Dataset RH fictif (200 lignes, 9 colonnes) :

| Colonne | Type | Description |
|---|---|---|
| `age` | Numérique | Âge de l'employé |
| `salaire_mensuel` | Numérique | Salaire en € |
| `anciennete_mois` | Numérique | Ancienneté en mois |
| `niveau_education` | Catégoriel | Bac / Bac+2 / Bac+3 / Bac+5 |
| `secteur` | Catégoriel | Commerce / Tech / Industrie... |
| `nb_formations` | Numérique | Nombre de formations suivies |
| `score_performance` | Numérique | Score d'évaluation (2.0 → 5.0) |
| `teletravail` | Catégoriel | Oui / Non |
| `quitte_entreprise` | **Cible** | 0 = reste, 1 = part |

**Colonne cible recommandée :** `quitte_entreprise` (classification binaire)

---

## Structure du projet

```
ChatAutoML-Bot/
├── streamlit_app.py        # Interface Streamlit principale
├── app/                    # Modules UI (chat, dashboard, visualisations)
├── automl/                 # Pipeline ML (preprocessing, search, evaluation)
├── llm/                    # Intégration LLM pour explications
├── config/                 # Configuration
├── datasets/               # Datasets d'exemple
├── demo_data.csv           # Données de démonstration (inclus)
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Déploiement containerisé
└── QUICKSTART.md           # Ce guide
```

---

## Technologies

| Composant | Librairie |
|---|---|
| Interface | Streamlit |
| Optimisation | Optuna (TPE Sampler) |
| ML | Scikit-learn, XGBoost, LightGBM |
| Explainabilité | SHAP |
| Data | pandas, NumPy |
| Containerisation | Docker |

---

## Auteur

**Oumou Kaltoum Sall** — Data Scientist & ML Engineer  
[Portfolio](https://portfolio-oumou-kaltoum.vercel.app) · [GitHub](https://github.com/kalsoumds) · [Gumroad](https://kalsia.gumroad.com)
