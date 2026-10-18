# Predictive Parts & Warranty Failure Agent

An AI-assisted predictive-maintenance system combining a **Random Forest failure-prediction model** with an **LLM-based parts, maintenance, and warranty recommendation layer**.

> **Dataset note:** This portfolio project uses a synthetic, domain-inspired equipment dataset. It is not trained on proprietary or real manufacturer failure records.

## 1. Project Overview

The system estimates whether equipment is likely to fail within the next 30 days, converts the prediction into a risk level, and uses an LLM to turn the ML output into an actionable maintenance/service recommendation.

```text
Equipment Data
      |
      v
Data Preparation
      |
      v
Random Forest Classifier
      |
      v
Failure Probability
      |
      v
Risk Classification
      |
      v
LLM Warranty / Parts Agent
      |
      +-------------------+
      |                   |
      v                   v
Parts / Failure       Warranty / Service
Concern               Recommendation
      |
      v
Flask REST API
```

## 2. What the Model Predicts

Target:

```text
failure_next_30_days
```

where:

```text
0 = not predicted to fail within 30 days
1 = predicted to fail within 30 days
```

The model also returns a probability using `predict_proba()`.

Example:

```text
Failure probability = 0.87
Risk = HIGH
```

Prototype risk thresholds:

```text
>= 0.70  HIGH
0.40-0.69 MEDIUM
< 0.40   LOW
```

These thresholds should be calibrated against real domain data before production use.

## 3. Input Features

| Feature | Meaning |
|---|---|
| `equipment_age_years` | Equipment age |
| `operating_hours` | Total operating hours |
| `temperature_c` | Operating temperature |
| `vibration_mm_s` | Vibration level |
| `pressure_bar` | Operating pressure |
| `maintenance_count` | Number of maintenance events |
| `days_since_maintenance` | Days since last maintenance |
| `part_age_hours` | Age of relevant component |
| `previous_failures` | Historical failure count |

## 4. Dataset Generation

Real industrial failure data is often proprietary, so this project generates a synthetic dataset with 1,000 equipment records.

The generator creates operational and maintenance features and then uses domain-inspired relationships to create a simulated failure probability and binary failure label.

```text
Synthetic equipment characteristics
              |
              v
Operational + maintenance signals
              |
              v
Domain-inspired risk calculation
              |
              v
Failure probability
              |
              v
Binary target
```

Output:

```text
data/equipment_failure_data.csv
```

The generated example contains 870 non-failure and 130 failure records.

This should be described honestly as **synthetic training data** in interviews.

## 5. Machine Learning Workflow

Implemented in `src/train_model.py`:

```text
CSV
 |
 v
Select features
 |
 v
80/20 train-test split
 |
 v
Random Forest
 |
 v
Classification metrics
 |
 v
Save model
```

The split uses `stratify=y` so the class distribution is approximately preserved between training and testing data.

## 6. Why Random Forest?

The project uses:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
```

Random Forest combines many decision trees.

```text
             Input
               |
      +--------+--------+
      |        |        |
    Tree 1   Tree 2   Tree 3 ...
      |        |        |
      +--------+--------+
               |
               v
        Final prediction
```

It was chosen because it works well with tabular data, can model nonlinear relationships, provides class probabilities, and is straightforward to explain.

`class_weight="balanced"` gives additional weight to the minority failure class.

## 7. Model Evaluation

The training script reports:

- Accuracy
- Precision
- Recall
- F1-score

**Precision:** Of the cases predicted as failures, how many were actually failures.

**Recall:** Of the actual failures, how many the model detected.

**F1:** Harmonic mean of precision and recall.

For predictive maintenance, recall can be important because missing a genuine failure-risk case can lead to an unexpected equipment failure.

## 8. Saved Model

After training:

```text
models/failure_prediction_model.pkl
```

is created using Joblib.

This means the application does not retrain the model for every prediction.

```text
TRAINING
Dataset -> Train model -> Save .pkl

RUNTIME
New equipment data -> Load .pkl -> Predict
```

## 9. Prediction Engine

`src/predict.py` loads the saved model and accepts new equipment measurements.

It returns:

```json
{
  "failure_predicted": true,
  "failure_probability": 0.87,
  "risk_level": "HIGH"
}
```

The key distinction is:

```text
Prediction = binary outcome
Probability = estimated model confidence for the positive class
Risk = business-friendly category
```

## 10. Why Add an LLM?

The ML model answers:

> How likely is the equipment to fail?

It does not naturally produce a technician-friendly service plan.

The LLM receives the equipment information and ML prediction and generates:

- Risk assessment
- Likely failure concern
- Recommended parts/components to inspect
- Warranty/service recommendation
- Priority

The architecture deliberately separates responsibilities:

```text
ML
 |
 +--> Prediction
 +--> Probability
 +--> Risk

LLM
 |
 +--> Explanation
 +--> Parts recommendation
 +--> Warranty/service recommendation
```

The LLM is **not** used to predict failure.

## 11. Warranty / Parts Agent

Implemented in:

```text
src/warranty_agent.py
```

It receives:

```text
Equipment data
+
ML prediction
```

and generates a structured natural-language recommendation.

The prompt also tells the model:

```text
The ML model predicts risk, not confirmed failure.
```

This prevents the recommendation from presenting a probability prediction as proof that a physical component has failed.

## 12. Complete Application Flow

```text
Equipment measurements
        |
        v
Feature preparation
        |
        v
Random Forest
        |
        v
Failure probability
        |
        v
Risk level
        |
        v
LLM recommendation
        |
   +----+----+----+
   |         |    |
   v         v    v
Risk      Parts  Warranty/
analysis  concern service action
   |         |    |
   +---------+----+
             |
             v
       Final response
```

## 13. REST API

Implemented in:

```text
src/api.py
```

Endpoints:

```text
GET  /health
POST /predict
```

### Health

```bash
curl http://127.0.0.1:5000/health
```

Expected:

```json
{
  "status": "healthy"
}
```

### Prediction

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "equipment_age_years": 9.5,
  "operating_hours": 10500,
  "temperature_c": 91,
  "vibration_mm_s": 6.5,
  "pressure_bar": 3.1,
  "maintenance_count": 2,
  "days_since_maintenance": 280,
  "part_age_hours": 4200,
  "previous_failures": 3
}'
```

The API validates the input, runs the ML model, generates the LLM recommendation, and returns JSON.

Validation rules: all 9 features are required and must be numbers; `maintenance_count`, `days_since_maintenance` and `previous_failures` must be whole numbers; every feature except `temperature_c` must be zero or more. Invalid input returns `422` with a list of problems, and a missing JSON body returns `400`.

Add `?recommend=false` to get only the ML prediction without calling the LLM. If the LLM call fails (no API key, network error, rate limit), the response still contains the prediction, with `"recommendation": null` and a `recommendation_error` message.

## 14. Project Structure

```text
predictive-parts-warranty-agent/
|
├── data/
|   └── equipment_failure_data.csv
|
├── models/
|   └── failure_prediction_model.pkl
|
├── src/
|   ├── generate_data.py
|   ├── train_model.py
|   ├── predict.py
|   ├── warranty_agent.py
|   ├── main.py
|   └── api.py
|
├── tests/
|   ├── conftest.py
|   ├── test_predict.py
|   └── test_api.py
├── README.md
├── EXPLANATION.md          (interview guide)
├── DETAILED_EXPLANATION.md (file-by-file and Random Forest deep dive)
├── requirements.txt
├── render.yaml             (Render deployment blueprint)
├── .python-version
├── .gitignore
└── .env              (you create this; not committed)
```

### File responsibilities

| File | Responsibility |
|---|---|
| `generate_data.py` | Generates synthetic equipment data |
| `train_model.py` | Trains/evaluates Random Forest |
| `predict.py` | Loads model and predicts failure risk |
| `warranty_agent.py` | Generates AI maintenance/parts/warranty recommendation |
| `main.py` | Runs ML + LLM workflow |
| `api.py` | Flask REST API |
| `data/` | Training dataset |
| `models/` | Serialized model |
| `tests/` | pytest suite for the model and the API (the LLM is mocked) |

For a line-by-line explanation of every file and of how the Random Forest reaches its answer, see [DETAILED_EXPLANATION.md](DETAILED_EXPLANATION.md).

## 15. Setup

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd predictive-parts-warranty-agent

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.11 or newer (pandas 3).

If installing manually:

```bash
pip install pandas numpy scikit-learn joblib flask python-dotenv
pip install langchain langchain-openai
```

Create `.env`:

```text
OPENAI_API_KEY=your_api_key_here
# optional: override the chat model used for recommendations
OPENAI_MODEL=gpt-5.6-luna
```

Never commit `.env`.

Recommended `.gitignore`:

```text
.env
venv/
__pycache__/
.DS_Store
```

## 16. Running the Project

Generate the dataset:

```bash
python src/generate_data.py
```

Train the model:

```bash
python src/train_model.py
```

Run a single prediction:

```bash
python src/predict.py
```

Run the complete ML + LLM workflow:

```bash
python src/main.py
```

Start the API:

```bash
python src/api.py
```

Run the tests (no API key needed; the LLM is mocked):

```bash
python -m pytest -q
```

### Deploy to Render

The repo includes a Render Blueprint (`render.yaml`). It runs the API with Gunicorn on Python 3.13 and uses `/health` as the health check.

1. In the Render dashboard, choose **New → Blueprint** and select this repository.
2. When asked, enter your `OPENAI_API_KEY`. It is stored as a secret in Render and is never committed.
3. Click **Apply**. When the build finishes, open the service URL. `GET /` lists the endpoints.

```bash
curl -X POST "https://<your-service>.onrender.com/predict?recommend=false" \
  -H "Content-Type: application/json" \
  -d '{"equipment_age_years":9.5,"operating_hours":10500,"temperature_c":91,"vibration_mm_s":6.5,"pressure_bar":3.1,"maintenance_count":2,"days_since_maintenance":280,"part_age_hours":4200,"previous_failures":3}'
```

On the free plan the service sleeps after 15 minutes without traffic, so the first request after that takes 30–60 seconds.

## 17. Example Scenario

Example equipment:

```text
Age: 9.5 years
Operating hours: 10,500
Temperature: 91 C
Vibration: 6.5 mm/s
Pressure: 3.1 bar
Maintenance count: 2
Days since maintenance: 280
Part age: 4,200 hours
Previous failures: 3
```

The workflow is:

```text
Measurements
    |
    v
Random Forest
    |
    v
Failure probability
    |
    v
HIGH risk
    |
    v
LLM
    |
    v
Maintenance + parts + warranty recommendation
```

## 18. Key Interview Concepts

### Supervised learning

The model learns:

```text
Features -> Known failure label
```

### Binary classification

There are two target classes:

```text
0 = no predicted failure
1 = predicted failure
```

### Random Forest

An ensemble of decision trees whose predictions are combined into a final result.

### Probability

`predict_proba()` provides class probabilities, allowing risk-based decisions instead of only True/False.

### Train/test split

Training data teaches the model; unseen test data evaluates generalization.

### LLM augmentation

The LLM interprets the ML result and creates a useful service recommendation.

### REST API

Other applications can send equipment measurements to `/predict` and receive the ML + LLM result as JSON.

## 19. Limitations

This is a portfolio prototype.

- Training data is synthetic.
- Failure-generation logic is simulated.
- The model is not validated against real industrial equipment.
- Probability calibration has not been performed.
- Risk thresholds are prototype thresholds.
- No real warranty database is connected.
- No real parts inventory is connected.
- The LLM recommendation is not a substitute for qualified engineering judgment.
- Production authentication, monitoring, and deployment are not included.

## 20. Future Improvements

A production-oriented version could add:

1. Real historical failure data.
2. XGBoost/LightGBM comparison.
3. Cross-validation and hyperparameter tuning.
4. Probability calibration.
5. SHAP model explanations.
6. Real parts inventory integration.
7. Real warranty database integration.
8. Automated maintenance scheduling.
9. Alerts and notifications.
10. Model drift monitoring.
11. Docker and CI/CD.
12. Cloud deployment.
13. Technician dashboard.
14. Time-series sensor models.
15. Remaining Useful Life prediction.

## 21. Interview Explanation

> I built a predictive-maintenance system that combines a Random Forest classifier with an LLM-based recommendation layer. I generated a synthetic equipment dataset containing operational and maintenance features such as temperature, vibration, operating hours, component age, and maintenance history. The Random Forest predicts whether equipment is likely to fail within 30 days and provides a failure probability. I then classify the risk and pass the equipment context and prediction to an LLM agent, which generates maintenance, parts, and warranty/service recommendations. Finally, I exposed the workflow through a Flask REST API so another application can submit equipment measurements and receive the prediction and recommendation as JSON.

## 22. Resume Description

**Predictive Parts & Warranty Failure Agent** — Built a Random Forest-based predictive-maintenance system that estimates 30-day equipment failure probability from operational and maintenance signals, then uses an LLM agent to generate risk explanations, parts recommendations, and warranty/service actions through a Flask REST API.

## Author

**Sparsh Kuwar**

Computer Science Engineering  
JSPM Rajarshi Shahu College of Engineering
