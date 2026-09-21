# AI Operations Request Router & Dashboard

Applied NLP prototype for routing internal service requests into operational categories, exposing predictions through a REST endpoint, and reviewing model outputs in a polished Streamlit dashboard.

## Dataset
The uploaded dataset contains **47,837 labeled requests** across **8 categories**: Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, and Administrative rights.

Data validation found no missing text/labels and no duplicate documents.

## What I built
- Used an **80/20 stratified train/test split** with `random_state=42`.
- Converted request text to unigram/bigram TF-IDF features (`min_df=2`, sublinear TF, max 60,000 features).
- Compared class-weighted Logistic Regression and Linear SVM baselines.
- Evaluated accuracy, macro F1, weighted F1, per-class precision/recall/F1, and misclassified requests.
- Added model persistence, a FastAPI prediction endpoint, and a recruiter-friendly Streamlit routing interface.

## Verified results
**Linear SVM** was the stronger baseline on the held-out test set:
- Accuracy: **86.34%**
- Macro F1: **86.15%**
- Weighted F1: **86.35%**

Logistic Regression macro F1 was **85.71%**.

The largest error patterns were between semantically overlapping categories such as Hardware, HR Support, and Miscellaneous.

## Interactive demo
The Streamlit interface is prepared for deployment from this repository.

**Streamlit entry point:** `app/dashboard.py`

The interface includes:
- example internal-service requests
- free-text request input
- predicted routing category
- benchmark metrics and supported categories
- project scope, limitations, and a direct GitHub link

### One deployment requirement
The live classifier expects the trained artifact at:

`models/request_router.joblib`

Generate it before publishing the live demo:

```bash
python src/train.py data/all_tickets_processed_improved_v3.csv
```

The dashboard now handles a missing model artifact gracefully instead of crashing, but predictions stay disabled until that file exists.

### Deploy on Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose this repository: `mahaqam/ai-ops-request-router`.
3. Select branch `main`.
4. Set the app file to `app/dashboard.py`.
5. Confirm `models/request_router.joblib` is present in the repository.
6. Click **Deploy**.

## Run locally
```bash
pip install -r requirements.txt
python src/train.py data/all_tickets_processed_improved_v3.csv
uvicorn api.app:app --reload
streamlit run app/dashboard.py
```

## Repository structure
- `src/train.py` — validation, split, training, evaluation, model save
- `src/predict.py` — local inference
- `api/app.py` — REST prediction endpoint
- `app/dashboard.py` — interactive request-routing UI
- `.streamlit/config.toml` — Streamlit theme/deployment configuration
- `results/metrics.json` — verified benchmark results
- `results/class_metrics.csv` — held-out per-class performance
- `results/top_confusions.csv` — largest error pairs

## Limitations
- This is an in-dataset benchmark, not production performance.
- Real internal requests may contain new categories, multilingual inputs, abbreviations, sensitive information, and domain shift.
- A deployed system should add confidence thresholds, abstention/human routing, monitoring, and retraining.
