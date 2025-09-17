# FinSight

**FinSight** is a Flask-based web application for loan scoring: it provides **loan approval** and **loan default** predictions backed by trained machine-learning models (Random Forest, XGBoost, etc.). The project includes the web front-end (HTML/CSS/JS), model artifacts, and Jupyter notebooks used during model development.

**Live demo:** https://finsight-cv2z.onrender.com/

---

## Table of contents
- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Included model & data files](#included-model--data-files)
- [Quick start (run locally)](#quick-start-run-locally)
- [Deploy to Render (production)](#deploy-to-render-production)
- [Debugging & tips](#debugging--tips)
- [Development & contributing](#development--contributing)
- [Credits](#credits)

---

## Features
- Web UI for:
  - Loan approval prediction
  - Loan default (credit risk) prediction
- Pre-trained ML models (Random Forest, XGBoost) shipped in `app/models/` for immediate inference
- Debug endpoint to inspect model/scaler feature names
- Example datasets and notebooks for reproducibility


## Tech stack
- Python (Flask)
- scikit-learn, XGBoost
- NumPy, pandas, SciPy
- joblib
- HTML / CSS / JavaScript (frontend)
- Gunicorn (production WSGI server)


## Project structure (important files)
```
FinSight/
├─ app/                       # Flask application package
│  ├─ __init__.py             # create_app() factory
│  ├─ routes.py               # HTTP routes and blueprint
│  ├─ utils.py                # model loading + inference helpers
│  ├─ models/                 # trained model artifacts (pkl, joblib)
│  ├─ templates/              # Jinja2 HTML templates
│  └─ static/                 # css/js/static assets
├─ data/                      # sample/train/test CSVs used in notebooks
├─ notebooks/                 # model development notebooks
├─ run.py                     # application entrypoint (app = create_app())
├─ Procfile                   # for Render / Heroku: `web: gunicorn run:app`
├─ runtime.txt                # pinned Python version (e.g. python-3.11.9)
├─ requirements.txt           # pinned dependencies
└─ FinSight.zip (you uploaded)
```

## Included model & data files (from the uploaded repo)
**Model artifacts (app/models):**
- `loan_approval_best_rf.pkl`
- `loan_approval_scaler.pkl`
- `scaler.pkl`
- `scaler_gmsc.joblib`
- `xgboost_model.pkl`
- `xgboost_model_tuned.pkl`

**Data (data/):** several CSVs used for training/inspection; notable files:
- `GiveMeSomeCredit-training.csv`
- `LoanApprovalPredictionDataset.csv`
- preprocessed train/test CSVs like `X_train_scaled.csv`, `y_train_smote.csv`, etc.

**Notebooks:**
- `notebooks/loanApprovalPrediction.ipynb`
- `notebooks/loanDefaultPrediction.ipynb`


## Quick start (run locally)
> Tested with Python 3.11 (see `runtime.txt`).

1. **Clone repo**
```bash
git clone <your-repo-url>
cd FinSight
```

2. **Create & activate a virtual environment**
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```bash
pip install -U pip wheel
pip install -r requirements.txt
```

> If you get binary wheel issues for `xgboost` or `scipy`, try upgrading `pip` and installing the problematic package separately (e.g. `pip install xgboost`).

4. **Run the app (development)**
```bash
python run.py
```
Open your browser at: `http://localhost:5000`

5. **Run in production mode (locally)**
```bash
# use gunicorn as in Procfile
pip install gunicorn
gunicorn run:app
```


## Deploy to Render (step-by-step)
Render works well for long-running Flask services and is a good choice for an ML-backed app.

1. Push your repository to GitHub (or GitLab/Bitbucket).
2. Ensure the repo root contains `run.py`, `Procfile`, `runtime.txt` and `requirements.txt`.
   - `Procfile` should contain:
     ```text
     web: gunicorn run:app
     ```
   - `runtime.txt` sample:
     ```text
     python-3.11.9
     ```
3. Log in to https://render.com and **Create → Web Service**.
4. Connect your Git provider and select the repo & branch.
5. Configure the service:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn run:app`
6. Create the service and let Render build & deploy. Logs are available in the Render dashboard.

**Notes / tips for Render**
- If your model artifacts are large (>50–100MB) you might hit repository size or build-time limits. Consider storing large model files in cloud storage (S3, Google Cloud Storage, or Hugging Face Hub) and download them at first startup.
- For private repos, connect Render to your account and authorize the repo.
- To view logs or restart: use the Render web dashboard.


## Debugging & common issues
- **Model file not found / mismatch**: The app includes a debug endpoint `/debug-models` which returns information about saved scaler/model `feature_names` to help diagnose feature-order mismatches.
  - Example: `GET https://<your-deploy>/debug-models`
- **Static assets 404**: Ensure templates reference static files via `{{ url_for('static', filename='css/styles.css') }}`. This repo already uses `url_for`.
- **Large model loads cause memory errors**: Move models to a remote object store and lazy-load them.
- **Missing binaries on build**: Some scientific packages need compilation. Upgrading pip, using prebuilt wheels, or using a Render instance with the appropriate buildpacks can help.


## Development & contributing
- The Flask app uses an application factory `create_app()` in `app/__init__.py` — this makes testing and configuration easier.
- Jupyter notebooks in `notebooks/` show how the models were trained and preprocessed. Use them to retrain or inspect the pipeline.
- To add/change models: place new artifacts in `app/models/` with a supported extension: `.pkl`, `.joblib`, `.sav`, `.pki`. The utilities in `app/utils.py` will try to locate them automatically.


## Useful commands
- Create `requirements.txt` from your current environment:
```bash
pip freeze > requirements.txt
```

- Create `Procfile` (if you need to re-create it):
```bash
echo "web: gunicorn run:app" > Procfile
```

- Create `runtime.txt` (pin Python):
```bash
echo "python-3.11.9" > runtime.txt
```


## Credits
Built by **Vedant Jadhav**.

---

If you'd like, I can:
- Produce a `README.md` with screenshots (send 1–3 images or tell me which UI screens to capture),
- Create a cleaned `requirements.txt` and a small `deploy.md` with screenshots for Render, or
- Convert the README into a short `README-deploy.md` that only contains production deployment instructions.

Tell me which one you'd prefer and I will update the document.

