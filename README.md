# Prêt à dépenser — API de credit scoring (FastAPI) + CI/CD

Ce dépôt fournit une **API FastAPI** de scoring crédit, avec :
- chargement du modèle via **MLflow** (URI `models:/...@champion`) ou **fallback** sur un modèle packagé localement,
- application du **seuil métier** (coût FN/FP) pour décider *APPROVED* vs *REFUSED*,
- **tests unitaires** (pytest),
- **CI** GitHub Actions,
- **CD** (déclenchement d’un déploiement via **Render Deploy Hook**).

> Important : je ne connais pas tes versions exactes de packages.  
> Après installation locale, exécute `pip freeze > requirements.txt` pour figer les versions.

---

## 1) Structure

```
api/
  main.py
  config.py
  model_loader.py
  schemas.py
tests/
  test_api.py
scripts/
  export_champion_to_local.py
.github/workflows/
  ci.yml
  deploy_render.yml
data/
  clients_sample.csv
outputs/
  evidently_drift_report.html
```

---

## 2) Configuration

Copie `.env.example` vers `.env` (local uniquement) et adapte :
- `MODEL_URI=models:/home_credit_default_model@champion` (si ton MLflow registry est accessible)
- `LOCAL_MODEL_PATH=artifacts/model` (fallback)
- `THRESHOLD=0.527`
- `FN_COST=10`
- `FP_COST=1`
- `CLIENTS_CSV_PATH=data/clients_sample.csv`

---

## 3) Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 4) Déploiement sur Render (CD via GitHub Actions)

### A) Côté Render
1. Crée un **Web Service** Render depuis ce repo (ou connecte-le).
2. Build : `pip install -r requirements.txt`
3. Start : `uvicorn api.main:app --host 0.0.0.0 --port 10000`
4. Crée un **Deploy Hook** Render.

Docs Render :
- Deploy Hooks : https://render.com/docs/deploy-hooks
- Connect GitHub : https://render.com/docs/github

### B) Côté GitHub
Ajoute un secret `RENDER_DEPLOY_HOOK_URL` (Settings → Secrets and variables → Actions).

---

## 5) Notes MLflow / Model Registry

MLflow permet de référencer un modèle par alias via un URI `models:/MyModel@champion`.
Voir : https://mlflow.org/docs/latest/ml/model-registry/
# trigger CI
# cd proof
