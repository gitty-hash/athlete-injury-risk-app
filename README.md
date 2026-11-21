# Athlete Injury Risk Predictor (MIS 545 final project, Lab 12 Task 3)

A Streamlit app that serves our best final-project model. It predicts whether an athlete's
**next** session will be labelled high risk (`injury_occurred == 2`) from the current session's
readings and recent history.

## Files

| File | What it is |
|---|---|
| `streamlit_app.py` | The app: sliders in, probability, risk label and the top factors out |
| `model/injury_risk_model.joblib` | Saved best model: the full scikit-learn pipeline (imputer, scaler, one-hot encoder, logistic regression) |
| `model/model_info.json` | Feature list, slider ranges, test scores and two example sessions |
| `train_and_save_model.py` | Rebuilds the model exactly as in `models(updated).ipynb` and saves the two files above |
| `requirements.txt` | Packages Streamlit Cloud installs |

## Why logistic regression

| Model | Test F1 | Test ROC-AUC |
|---|---|---|
| dummy | 0.153 | 0.502 |
| **logistic regression** | **0.499** | 0.859 |
| random forest | 0.486 | 0.846 |
| XGBoost | 0.492 | 0.862 |

(Scores from the notebook run in Colab.) The notebook selects on F1, and logistic regression has
the best test F1. XGBoost's ROC-AUC is only 0.003 higher. Logistic regression is also a 3 KB file,
passed every leakage check, and its coefficients explain each prediction.

The saved model was retrained locally with scikit-learn 1.9.1, using the same settings
(C = 0.1, L1 penalty). Its test F1 is 0.489 instead of 0.499 because newer scikit-learn assigns
athletes to the train and test sets slightly differently (12,139/3,125 rows instead of 12,134/3,130).

## Deploy (same steps as Lab 12)

1. Create a new **blank app** on Streamlit Community Cloud and open its Codespace.
2. Upload `streamlit_app.py`, `requirements.txt` and the whole `model/` folder (both files).
3. In the Codespace terminal, run:
   `uv add streamlit scikit-learn==1.9.1 pandas numpy joblib altair`
   Pin scikit-learn to 1.9.1, the version that saved the model.
4. Commit and push, then open the published URL.
5. Pick a preset in the sidebar or move the sliders. The prediction updates instantly.

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To rebuild the model (downloads the Kaggle dataset; no login needed), first run
`pip install kagglehub`, then run `python train_and_save_model.py`.
