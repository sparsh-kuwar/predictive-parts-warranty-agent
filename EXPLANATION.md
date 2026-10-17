# Predictive Parts & Warranty Agent: interview guide

This guide is for explaining and defending the project in an interview. Each section names the file it describes.

## 1. The 30-second pitch

> A predictive-maintenance prototype in two layers. A Random Forest trained on equipment sensor and usage data estimates the probability that a pump fails in the next 30 days and maps it to LOW / MEDIUM / HIGH risk. An LLM then turns that number into a technician-friendly plan: what to inspect, which parts, and a warranty or service action. The two layers have strictly separate jobs: **the ML model predicts, the LLM explains.** The LLM is never asked whether the machine will fail.

## 2. The 2-minute walkthrough

1. **Data** (`src/generate_data.py`). 1,000 synthetic pump records with 9 features. The label comes from a hand-written risk formula plus random noise (section 4). Result: 870 non-failures, 130 failures (13%).
2. **Training** (`src/train_model.py`). 80/20 **stratified** split, `RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)`, metrics printed, model saved with `joblib` to `models/failure_prediction_model.pkl`.
3. **Prediction** (`src/predict.py`). Loads the saved model once, builds a one-row DataFrame **with the same column order as training**, and returns `failure_predicted`, `failure_probability` (`predict_proba`) and `risk_level` (≥0.70 HIGH, ≥0.40 MEDIUM, else LOW).
4. **Recommendation** (`src/warranty_agent.py`). LangChain `ChatPromptTemplate | ChatOpenAI(temperature=0)` gets the equipment data and the prediction, and returns Risk Assessment / Likely Failure Concern / Recommended Parts / Warranty-Service Recommendation / Priority. The prompt says: *"The ML model predicts risk, not confirmed failure."*
5. **Demo** (`src/main.py`). Runs one example pump end to end in the terminal.

## 3. The numbers (rerun these yourself before the interview)

Retraining with the exact settings in `train_model.py` on the committed CSV gives this test-set result (200 rows: 174 non-failures, 26 failures):

| Metric | Value |
|---|---|
| Accuracy | **84.5%** |
| Failure-class precision | **43.6%** |
| Failure-class recall | **65.4%** |
| Failure-class F1 | 0.52 |
| ROC-AUC | 0.85 |

Confusion matrix: 17 failures caught, 9 missed, 22 false alarms, 152 correct non-failures.

**The key point to make yourself:** a model that *always* predicts "no failure" gets **87% accuracy**, higher than this model. Accuracy is the wrong metric with 13% positives. What matters is **recall** (failures caught: 65%) and precision (alarms that are real: 44%). This is also why `class_weight="balanced"` is used: it trades some accuracy for catching more failures.

Feature importance (impurity-based): temperature 0.24, vibration 0.20, operating hours 0.16, days since maintenance 0.11, part age 0.08, equipment age 0.07, pressure 0.06, previous failures 0.04, maintenance count 0.04.

**A sharp observation to have ready:** `pressure_bar` and `maintenance_count` are **not in the formula** that generated the labels, yet they get about 5–6% importance each. That is the model fitting noise, and impurity-based importance is known to give some credit to irrelevant continuous features. Permutation importance on the test set would show it more honestly.

## 4. The synthetic data (be upfront about this)

```text
risk = 0.04·age + 0.00012·hours + 0.035·(temp − 70) + 0.25·vibration
     + 0.00025·part_age + 0.12·previous_failures + 0.0015·days_since_maintenance
     + Normal(0, 0.8) noise
label = 1 if sigmoid(risk − 5) > 0.5, i.e. if risk > 5
```

How to say it: *"I didn't have access to real failure records, so I generated data with domain-inspired relationships (heat, vibration and overdue maintenance raise risk) and added noise so the problem isn't trivial. The model is learning back a relationship I designed, so the metrics show the pipeline works, not real-world accuracy. With real data, the first job would be checking which of these relationships actually hold."*

## 5. Key design decisions and why

| Decision | Why |
|---|---|
| Random Forest | Strong default for tabular data, handles nonlinear effects and interactions, no scaling needed, gives probabilities, little tuning |
| `max_depth=8` | Limits overfitting on 800 training rows |
| `class_weight="balanced"` | Only 13% failures. Without it the model leans toward "no failure". |
| Stratified split | Keeps 13% failures in both train and test. A random split could leave the test set with very few failures. |
| Save with joblib | Train once, predict many times. No retraining per request. |
| Probability + risk band | Business users act on "HIGH", engineers want the number. Both are returned. |
| ML predicts, LLM explains | LLMs are bad at numeric risk estimation and can't be validated like a classifier. A classifier can't write a service plan. Each does what it is good at. |
| `temperature=0` | Repeatable recommendations for the same input |
| "risk, not confirmed failure" in the prompt | Stops the LLM turning "70% risk" into "the bearing has failed" |

## 6. Weaknesses to state openly

Fix the first three before the interview if you can. They are the ones an interviewer will spot.

1. **The README describes `src/api.py` (a Flask `/predict` API) and a `tests/` folder that aren't in the repo.** Either add them or remove them from the README. Don't claim a REST API that isn't there.
2. **`requirements.txt` is missing `pandas`, `scikit-learn`, `joblib` and `numpy`**, so a fresh install can't train or predict.
3. **`warranty_agent.py` line 2 has a stray `load_dotenv`** (no parentheses, does nothing). Also confirm the configured model name works with your OpenAI account.
4. **Synthetic data**: see section 4.
5. **Thresholds 0.70 / 0.40 are arbitrary.** They should come from business costs: a missed failure (downtime, emergency repair) vs. a false alarm (unneeded inspection).
6. **The decision threshold for `failure_predicted` is the default 0.5.** With imbalanced data, a lower threshold would raise recall.
7. **One train/test split, no cross-validation, no hyperparameter search.**
8. **No time dimension.** Real failure prediction uses sensor *trends* (vibration rising over a week), not one snapshot. Real data would also need a time-based split to avoid leakage.
9. **The LLM output is free text**, not schema-validated JSON, so other systems can't consume it reliably.
10. **No input validation**: missing or out-of-range features aren't checked before `predict`.

## 7. Likely interview questions

**Q: Your accuracy is 84.5%. Is that good?**
No, and accuracy is the wrong lens. Always predicting "no failure" scores 87%. I look at recall on the failure class (65%: we catch about 2 in 3 failures) and precision (44%: a bit under half of alarms are real). For maintenance, missing a failure usually costs more than an extra inspection, so I would tune for recall.

**Q: How would you increase recall?**
Lower the decision threshold below 0.5 (pick it from the precision-recall curve), keep class weighting, try gradient boosting (XGBoost/LightGBM), and above all get better features, such as sensor trends.

**Q: Why Random Forest over logistic regression or XGBoost?**
Logistic regression assumes linear effects on the log-odds. RF captures thresholds and interactions (high temperature *and* high vibration) with no feature scaling. XGBoost would likely do slightly better. RF was a robust, explainable first baseline. With more time I would compare all three with cross-validation.

**Q: What is `predict_proba` actually giving you?**
The fraction of the 200 trees that vote "failure" (averaged leaf probabilities). It is a ranking score, not a calibrated probability. For "0.87 = 87% chance" to be true, I would calibrate with `CalibratedClassifierCV` and check a reliability curve.

**Q: Why not just ask the LLM if the machine will fail?**
It can't be evaluated or reproduced like a classifier, it is poor at numeric reasoning over many features, and it can't learn from your historical data without fine-tuning. The classifier is measurable. The LLM adds the explanation layer.

**Q: How do you stop the LLM recommending the wrong part?**
At the moment only through the prompt ("risk, not confirmed failure", temperature 0). Better: give it a **parts catalogue and failure-mode table to choose from** (retrieval), require JSON output with a fixed set of part ids, and validate against the catalogue, the same grounding approach as in my field-service and claims projects.

**Q: How does this link to warranty?**
High risk on a unit still under warranty → schedule proactive service and pre-order the part, which is cheaper than an emergency warranty repair. High risk near the end of warranty → inspect before it expires. Out of warranty → offer an extended-warranty or service contract. Production would read warranty status from a registry, not leave it to the LLM.

**Q: How would you deploy and monitor it?**
A Flask/FastAPI `/predict` endpoint loading the model at startup, input validation with Pydantic, model versioning, and monitoring of **data drift** (feature distributions) and **outcome tracking** (did the predicted failures happen?) to trigger retraining.

**Q: What is data leakage, and could it happen here?**
Leakage means the model learns from information it won't have at prediction time. With real data, features like `previous_failures` must be counted *before* the prediction date, and the split must be by time, otherwise the model "sees the future".

## 8. What I would build next

1. Add the missing Flask/FastAPI API, `requirements.txt` entries and tests (section 6).
2. Choose the threshold from the PR curve, calibrate probabilities, use permutation importance.
3. Compare models (LogReg / RF / XGBoost) with cross-validation.
4. Structured LLM output grounded in a parts catalogue.
5. SHAP values per prediction, so the technician sees *why* (for example "vibration 6.5 mm/s pushed risk up most").
6. Time-series features on real or public data (for example NASA turbofan, CWRU bearing datasets).
