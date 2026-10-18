# Predictive Parts & Warranty Agent: detailed explanation

This document explains how the project works from the inside: what every file does, how the data is made, how the Random Forest reaches its answer, and how the LLM and the API sit on top of it.

[EXPLANATION.md](EXPLANATION.md) is the short interview guide. This one is the deep version to read when you want to understand each step.

Every number below was produced by running the committed code on the committed data and model, so you can reproduce it yourself.

---

## 1. The big picture

```text
 generate_data.py         train_model.py              predict.py               warranty_agent.py
┌───────────────┐      ┌──────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│ 1,000 fake    │ CSV  │ Random Forest    │ .pkl │ 9 numbers in      │ dict │ LLM writes the    │
│ pump records  │ ───► │ learns failure   │ ───► │ → probability     │ ───► │ maintenance and   │
│ + label       │      │ patterns         │      │ → risk band       │      │ warranty plan     │
└───────────────┘      └──────────────────┘      └───────────────────┘      └───────────────────┘
   (run once)              (run once)                  (every request)            (every request)
                                                            ▲
                                       main.py (terminal demo) and api.py (HTTP) call these two
```

There are two phases:

- **Offline, done once:** create the dataset, then train and save the model.
- **Online, for every pump you check:** load the saved model, predict, then ask the LLM to explain.

The central rule of the design: **the Random Forest decides the risk, the LLM only explains it.** The LLM never decides whether the pump will fail.

---

## 2. Every file, one by one

```text
predictive-parts-warranty-agent/
├── data/equipment_failure_data.csv       the training data (generated)
├── models/failure_prediction_model.pkl   the trained Random Forest (generated)
├── src/
│   ├── generate_data.py   step 1: make the dataset
│   ├── train_model.py     step 2: train, evaluate, save the model
│   ├── predict.py         step 3: load the model, predict one pump
│   ├── warranty_agent.py  step 4: LLM recommendation
│   ├── main.py            terminal demo of steps 3 + 4
│   └── api.py             Flask REST API around steps 3 + 4
├── tests/
│   ├── conftest.py        shared test setup and two sample pumps
│   ├── test_predict.py    tests for the model wrapper
│   └── test_api.py        tests for the API (LLM mocked)
├── requirements.txt       pinned library versions
├── .gitignore             keeps .env, venv and caches out of git
├── README.md              project overview and how to run
└── EXPLANATION.md         interview guide
```

### `src/generate_data.py`: creates the dataset

**Why it exists:** there was no access to real pump failure records, so the data is simulated.

**What it does, step by step:**

1. `np.random.seed(42)` fixes the random generator, so every run produces exactly the same 1,000 rows.
2. It creates 1,000 pumps (`PUMP-0001` … `PUMP-1000`) and draws 9 features at random:

   | Feature | Range | Meaning |
   |---|---|---|
   | `equipment_age_years` | 0.5 – 12 | How old the pump is |
   | `operating_hours` | 500 – 12,000 | Total hours it has run |
   | `temperature_c` | 55 – 100 | Operating temperature |
   | `vibration_mm_s` | 1 – 8 | Vibration speed (a classic sign of bearing wear) |
   | `pressure_bar` | 2 – 5.5 | Outlet pressure |
   | `maintenance_count` | 0 – 14 | How many services it has had |
   | `days_since_maintenance` | 1 – 499 | Days since the last service |
   | `part_age_hours` | 100 – 5,000 | Hours on the current wear part |
   | `previous_failures` | 0 – 4 | Past breakdowns |

   Decimal features use `uniform` (any value in the range is equally likely). Counts use `randint` (whole numbers only).

3. It computes a **hidden risk score** for each pump with a formula that is meant to resemble the real world:

   ```text
   risk = 0.04    × equipment_age_years
        + 0.00012 × operating_hours
        + 0.035   × (temperature_c − 70)      ← hotter than 70 °C adds risk, cooler removes it
        + 0.25    × vibration_mm_s            ← the strongest single term
        + 0.00025 × part_age_hours
        + 0.12    × previous_failures
        + 0.0015  × days_since_maintenance
        + random noise from Normal(mean 0, std 0.8)
   ```

4. It turns the score into a probability with the **sigmoid** function, `1 / (1 + e^-(risk − 5))`. The sigmoid squashes any number into the range 0–1, and the "− 5" moves its midpoint to a risk of 5.
5. The label `failure_next_30_days` is 1 if that probability is above 0.5, which is the same as saying **risk > 5**. Otherwise it is 0.
6. It saves the table to `data/equipment_failure_data.csv` and prints the class counts.

**The result:** 870 pumps that don't fail (87%) and 130 that do (13%). Failures are rare, which is realistic and matters a lot later (section 6).

**Two things to notice:**

- `pressure_bar` and `maintenance_count` are generated but **not used in the formula**. They are deliberate noise: a good model should learn to mostly ignore them.
- The noise (std 0.8) means two pumps with identical features can get different labels. That keeps the problem from being trivially learnable, as in real life.

**Worked example:** the demo pump (age 9.5, 10,500 h, 91 °C, 6.5 mm/s, 280 days, part age 4,200 h, 3 past failures) has a formula risk of **5.83** before noise. That's above 5, so it would usually be labelled a failure. The healthy test pump scores **0.12**, far below 5.

### `src/train_model.py`: trains, evaluates and saves the model

1. **Loads** the CSV with pandas.
2. **Chooses the inputs (`X`) and the answer (`y`).** `X` is the 9 feature columns. `y` is `failure_next_30_days`. `equipment_id` is left out because a pump's name says nothing about failure.
3. **Splits the data 80/20** with `train_test_split(test_size=0.2, random_state=42, stratify=y)`:
   - 800 rows for training (696 OK, 104 failures) and 200 for testing (174 OK, 26 failures).
   - `stratify=y` keeps the 13% failure rate in both parts. Without it, a random split could put very few failures in the test set and make the scores meaningless.
   - The test set is never shown to the model during training. It plays the role of "future pumps".
4. **Creates the model:** `RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")`. Section 3 explains each setting.
5. **Trains** it with `model.fit(X_train, y_train)`.
6. **Evaluates** it on the test set and prints accuracy plus a classification report (precision, recall and F1 per class). Section 6 explains these.
7. **Saves** the trained model with `joblib.dump` to `models/failure_prediction_model.pkl`. This saves the 200 trained trees, so prediction never needs to retrain.

It must be run from the project root (`python src/train_model.py`), because the CSV and model paths are relative.

### `src/predict.py`: turns 9 numbers into a risk answer

1. **Finds the model file** relative to its own location (`Path(__file__).resolve().parent.parent / "models" / ...`), so it works from any folder.
2. **Loads the model once, when the module is imported** (`model = joblib.load(MODEL_PATH)`). Loading takes time, so the API does it at startup and not on every request.
3. **`FEATURES`** lists the 9 columns in the exact order used in training. The trained model remembers its column names, and one of the tests checks that this list matches them.
4. **`predict_failure(equipment)`:**
   - `pd.DataFrame([equipment], columns=FEATURES)` builds a one-row table. Passing `columns=FEATURES` puts the values in the right order no matter what order the dict keys arrive in.
   - `model.predict(data)[0]` returns 0 or 1. That becomes `failure_predicted`.
   - `model.predict_proba(data)[0][1]` returns the failure probability. `[0]` is the first (only) row, and `[1]` is the column for class 1 ("failure").
   - The probability is mapped to a band: **≥ 0.70 → HIGH**, **≥ 0.40 → MEDIUM**, otherwise **LOW**.
   - It returns `{"failure_predicted": bool, "failure_probability": 3-decimal float, "risk_level": str}`.
5. Running the file directly (`python src/predict.py`) predicts the demo pump.

### `src/warranty_agent.py`: the LLM explanation layer

1. `load_dotenv()` reads `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) from a `.env` file.
2. `MODEL_NAME` is the chat model to use. The default can be overridden with `OPENAI_MODEL`.
3. **`get_llm()` creates the LLM client lazily**, on first use. That way, importing this file (for example in tests or when the API starts) doesn't require an API key. `temperature=0` makes the answer as repeatable as possible.
4. **`prompt`** is a LangChain `ChatPromptTemplate` with two blanks, `{equipment}` and `{prediction}`. It asks for five fixed sections: Risk Assessment, Likely Failure Concern, Recommended Parts, Warranty / Service Recommendation and Priority. It ends with the guard rail *"Do not claim that a specific component has definitely failed. The ML model predicts risk, not confirmed failure."*
5. **`generate_recommendation(equipment, prediction)`**:
   - `chain = prompt | get_llm()` uses LangChain's pipe syntax (LCEL): the filled-in prompt flows into the model.
   - `chain.invoke({...})` fills the blanks and sends one request to OpenAI.
   - It returns `response.content`, the text answer.

The LLM receives the raw measurements **and** the model's probability and band. So it can say things like "vibration of 6.5 mm/s is high, inspect the bearings", but the risk level itself comes from the Random Forest.

### `src/main.py`: terminal demo

It hard-codes the demo pump, calls `predict_failure`, prints the probability and band, calls `generate_recommendation` and prints the LLM text. It's the quickest way to see the whole flow. It needs an API key.

### `src/api.py`: the REST API

- **`validate_equipment(data)`** checks the incoming JSON before it reaches the model:
  - The body must be a JSON object.
  - All 9 features must be present. Missing ones are listed by name.
  - Each value must be a real number. `True`/`False` are rejected explicitly, because in Python `bool` is a kind of `int` and would otherwise slip through as 1 or 0.
  - The value must be finite (no infinity or NaN).
  - `maintenance_count`, `days_since_maintenance` and `previous_failures` must be whole numbers.
  - Nothing may be negative except `temperature_c`.
  - Extra fields such as `equipment_id` are dropped, so only the 9 features reach the model.
  - It collects **all** the problems at once, so the caller can fix everything in one go.
- **`create_app()`** builds the Flask app (a "factory" function, which lets the tests build fresh copies):
  - `GET /health` → `{"status": "healthy"}`, for uptime checks.
  - `POST /predict`:
    1. Not JSON → **400**.
    2. Validation errors → **422** with a `details` list.
    3. Otherwise it runs `predict_failure`.
    4. `?recommend=false` → returns the prediction only, without calling the LLM (fast and free).
    5. Otherwise it calls the LLM inside `try/except`. **If the LLM fails** (no key, network error, rate limit), the response still contains the prediction, with `recommendation: null` and a `recommendation_error` message. An LLM outage must never hide a HIGH-risk prediction.
- `app.json.sort_keys = False` keeps the JSON fields in a readable order.

### `tests/conftest.py`

- Adds `src/` to Python's import path, so tests can `import predict` the same way the scripts import each other.
- Defines two fixtures (reusable test inputs): `high_risk_equipment` (the demo pump) and `low_risk_equipment` (a new, cool, quiet, recently serviced pump).

### `tests/test_predict.py` (6 tests)

| Test | What it proves |
|---|---|
| high-risk pump is flagged | The demo pump gets `True` and `HIGH` |
| low-risk pump is not flagged | The healthy pump gets `False` and `LOW` |
| probability is between 0 and 1 | The output is a valid probability |
| risk level matches thresholds | The 0.70/0.40 banding logic is applied correctly |
| key order doesn't matter | Shuffled dict keys give the same answer (the `columns=FEATURES` protection) |
| feature list matches the model | `FEATURES` equals the column names the model was trained on |

### `tests/test_api.py` (14 tests)

The LLM is replaced with a fake function (`monkeypatch`), so the tests never call OpenAI and need no key. They cover `/health`, a normal prediction, `?recommend=false` (asserting the LLM is **not** called), LLM failure still returning the prediction, extra fields being ignored, a non-JSON body (400), a non-object body (422), missing fields being listed, five kinds of invalid value (text, boolean, null, negative, fractional count) and a negative temperature being allowed.

### `data/equipment_failure_data.csv` and `models/failure_prediction_model.pkl`

Both are generated files, committed so the project works right after cloning. The `.pkl` is a pickled scikit-learn object. Pickles are tied to the library version, so the model only loads reliably with the pinned `scikit-learn==1.9.1`. After upgrading scikit-learn, retrain.

### `requirements.txt` and `.gitignore`

`requirements.txt` pins every package version so the project installs the same way everywhere. The main ones are pandas, numpy, scikit-learn, joblib, Flask, langchain-core, langchain-openai, python-dotenv and pytest. The rest (for example langgraph and langsmith) are dependencies of those packages, not something the code imports directly. `.gitignore` keeps `.env` (your API key), `venv/` and caches out of git.

---

## 3. How the Random Forest works

### 3.1 Start with one decision tree

A decision tree is a flowchart of yes/no questions about the features, learned from data. For example:

```text
                 temperature_c ≤ 75.2 ?
                 /                    \
              yes                      no
               │                        │
     vibration_mm_s ≤ 5.1 ?     days_since_maintenance ≤ 272 ?
        /         \                 /                \
      ...         ...             ...                ...
                                                      │
                                              leaf: 97% failure
```

You follow the questions from the top (the **root**) until you reach an end point (a **leaf**). The leaf holds the share of failures among the training pumps that ended up there, and that share is the tree's answer.

**How a tree picks each question.** At each step it tries possible features and thresholds and keeps the split that best separates failures from non-failures. "Best" is measured with **Gini impurity**:

```text
Gini = 1 − p(OK)² − p(fail)²
```

- A group that is all one class has Gini 0 (perfectly pure).
- A 50/50 mix has Gini 0.5 (as mixed as possible).

The tree chooses the split that lowers the (size-weighted) Gini of the two child groups the most, then repeats inside each child. In this model it stops at depth 8 (`max_depth=8`).

**The problem with one tree:** a deep tree memorises its training data, including the noise. Small changes in the data give a very different tree. That's called **high variance**.

### 3.2 A forest: many different trees, averaged

A Random Forest trains many trees (here **200**, `n_estimators=200`) and makes each one different in two ways:

1. **Bootstrap sampling (`bootstrap=True`, the default).** Each tree trains on 800 rows drawn *with replacement* from the 800 training rows. Some rows appear twice, and about 37% don't appear at all. So every tree sees a slightly different dataset.
2. **Random feature choice (`max_features="sqrt"`, the default).** At every split, a tree may only look at a random √9 = **3** of the 9 features. That stops all trees from asking the same questions and forces them to find different patterns.

Each tree is individually a bit wrong, in different ways. Averaging many of them cancels out much of that error. This is called **bagging**, and it's why a forest is much more stable than a single tree.

You can see the variety in the trained model. The first question (root split) of the 200 trees is:

| Root feature | Number of trees |
|---|---|
| temperature_c | 73 |
| vibration_mm_s | 45 |
| operating_hours | 39 |
| days_since_maintenance | 18 |
| part_age_hours | 12 |
| previous_failures | 9 |
| equipment_age_years | 3 |
| maintenance_count | 1 |

Temperature and vibration come first most often because they are the strongest signals in the formula. Every tree reaches the full depth of 8, with 38–75 leaves (56 on average).

### 3.3 The settings in `train_model.py`

| Setting | Value | What it does |
|---|---|---|
| `n_estimators` | 200 | Number of trees. More trees give a more stable average; beyond a few hundred the gain is tiny. |
| `max_depth` | 8 | Each tree asks at most 8 questions in a row. This limits memorisation of noise on just 800 rows. |
| `class_weight` | `"balanced"` | Makes the rare failure class count more (see 3.4). |
| `random_state` | 42 | Fixes the randomness, so retraining gives exactly the same model. |
| `max_features` | `"sqrt"` (default) | 3 random features are considered at each split. |
| `bootstrap` | `True` (default) | Each tree gets its own resampled data. |

### 3.4 `class_weight="balanced"`: why the rare class needs help

The training set has 696 OK pumps and 104 failures. A tree that always says "OK" is already right 87% of the time, so the trees are tempted to ignore failures.

`"balanced"` gives each class a weight of `total rows / (2 × rows in that class)`:

```text
OK:      800 / (2 × 696) = 0.575
Failure: 800 / (2 × 104) = 3.846
```

So **one failure counts as much as about 6.7 OK pumps** when the trees measure Gini. In effect, both classes count equally in total. You can see it in tree 0: its root node holds a **weighted** mix of about 51% OK and 49% failure, with a Gini of 0.4997, even though the raw rows are 87/13.

The trade-off: the model raises more alarms. It catches more real failures (higher recall) but some alarms are false (lower precision). For maintenance that's usually the right trade, because a missed failure costs far more than an extra inspection.

### 3.5 How the forest gets the answer for one pump

Take the demo pump from `main.py`:

```json
{"equipment_age_years": 9.5, "operating_hours": 10500, "temperature_c": 91,
 "vibration_mm_s": 6.5, "pressure_bar": 3.1, "maintenance_count": 2,
 "days_since_maintenance": 280, "part_age_hours": 4200, "previous_failures": 3}
```

**Step 1: every tree routes the pump to one leaf.** This is the real path the pump takes through tree 0:

```text
node 0    temperature_c          91     > 75.16    → right
node 30   days_since_maintenance 280    > 272      → right
node 76   vibration_mm_s         6.5    > 4.60     → right
node 98   previous_failures      3      > 0.5      → right
node 104  equipment_age_years    9.5    ≤ 11.54    → left
node 105  pressure_bar           3.1    > 2.17     → right
node 107  operating_hours        10500  > 4375     → right
node 111  maintenance_count      2      ≤ 13.5     → left
leaf 112  39 training rows, weighted 96.7% failure  → this tree says 0.967
```

Read the path as a story: hot, overdue for service, vibrating, with a failure history and many hours. Tree 0 learned that pumps like that almost always failed.

**Step 2: the forest averages the 200 tree answers.** Each tree gives its leaf's failure share (between 0 and 1). `predict_proba` is the **average of those 200 numbers**:

```text
failure_probability = (0.967 + tree 2 + … + tree 200) / 200 = 0.899
```

For this pump, **189 of the 200 trees** lean towards failure (their leaf share is above 0.5).

Note: `predict_proba` averages the trees' probabilities. It isn't a strict vote count. 189/200 = 0.945, but the average probability is 0.899, because some trees that lean towards failure are only, say, 70% sure.

**Step 3: `predict` turns the probability into yes/no.** For a Random Forest, `model.predict` picks the class with the higher average probability. With two classes, that means **failure if the probability is above 0.5**. 0.899 > 0.5 → `failure_predicted: true`.

**Step 4: `predict.py` adds the risk band.** 0.899 ≥ 0.70 → `HIGH`.

**Final answer:**

```json
{"failure_predicted": true, "failure_probability": 0.899, "risk_level": "HIGH"}
```

For the healthy pump (1 year old, 58 °C, 1.2 mm/s, serviced 20 days ago) only **2 of 200** trees lean towards failure, and the result is `{"failure_predicted": false, "failure_probability": 0.01, "risk_level": "LOW"}`.

### 3.6 Two outputs, two thresholds: a subtle point

`failure_predicted` uses a cut-off of **0.5**, while the risk bands use **0.40 and 0.70**. So a probability between 0.40 and 0.50 gives:

```json
{"failure_predicted": false, "risk_level": "MEDIUM"}
```

That's intended: "not predicted to fail, but worth watching." You should be ready to explain it if an interviewer spots it.

### 3.7 What-if experiments on the demo pump

Change one feature at a time and re-predict:

| Change | Probability | Band | `failure_predicted` |
|---|---|---|---|
| None (demo pump) | 0.899 | HIGH | true |
| Vibration 6.5 → 2.0 mm/s | 0.511 | MEDIUM | true |
| Temperature 91 → 65 °C | 0.319 | LOW | false |
| Days since maintenance 280 → 10 | 0.415 | MEDIUM | false |

This shows what the model has learned: cooling the pump has the biggest effect, then servicing it, then reducing vibration. It's also a nice demo of the "maintenance lowers risk" story. The last row is an example of the subtle case from 3.6.

---

## 4. How the LLM gets its answer

1. The API (or `main.py`) passes two dicts to `generate_recommendation`: the 9 measurements and the prediction (`0.899`, `HIGH`).
2. LangChain fills them into the prompt template as text.
3. One request goes to the OpenAI chat model at temperature 0.
4. The model writes the five sections. For the demo pump it would typically point to bearings or seals (high vibration and temperature), suggest inspecting the wear part (4,200 hours), recommend scheduling service and checking warranty coverage, and give a priority of HIGH or URGENT.
5. The text is returned as-is.

What the LLM **doesn't** do: decide the probability, see the training data, or look up a real parts catalogue or warranty registry. Its parts and warranty suggestions are general knowledge guided by the prompt. Grounding it in a real catalogue is listed as a next step in EXPLANATION.md.

---

## 5. One full API request, start to finish

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"equipment_id":"PUMP-0042","equipment_age_years":9.5,"operating_hours":10500,"temperature_c":91,"vibration_mm_s":6.5,"pressure_bar":3.1,"maintenance_count":2,"days_since_maintenance":280,"part_age_hours":4200,"previous_failures":3}'
```

1. Flask parses the JSON body.
2. `validate_equipment` checks the 9 features and drops `equipment_id`.
3. `predict_failure` builds the one-row table in training column order.
4. The pump runs through all 200 trees. Their leaf values are averaged to 0.899.
5. 0.899 > 0.5 → failure predicted. 0.899 ≥ 0.70 → HIGH.
6. `generate_recommendation` sends the measurements and prediction to the LLM.
7. The response is returned:

```json
{
  "equipment": { "...the 9 features..." },
  "prediction": {"failure_predicted": true, "failure_probability": 0.899, "risk_level": "HIGH"},
  "recommendation": "Risk Assessment: ...\nLikely Failure Concern: ...\n..."
}
```

If step 6 fails, you still get the full `prediction`, plus `"recommendation": null` and a `recommendation_error` message.

---

## 6. How good is the model? (evaluation explained)

### 6.1 The confusion matrix

On the 200 test pumps the model has never seen:

|  | Predicted OK | Predicted failure |
|---|---|---|
| **Actually OK (174)** | 152 ✔ (true negatives) | 22 ✘ (false alarms) |
| **Actually failed (26)** | 9 ✘ (missed failures) | 17 ✔ (caught failures) |

### 6.2 The metrics, from the table

| Metric | Formula | Value | In plain words |
|---|---|---|---|
| Accuracy | (152 + 17) / 200 | **84.5%** | Share of all predictions that are right |
| Precision (failure) | 17 / (17 + 22) | **43.6%** | When it raises an alarm, how often it's real |
| Recall (failure) | 17 / (17 + 9) | **65.4%** | Of the real failures, how many it catches |
| F1 (failure) | harmonic mean of the two | **0.52** | One number balancing precision and recall |
| ROC-AUC | — | **0.85** | How well the probabilities rank failing pumps above healthy ones (0.5 = random, 1.0 = perfect) |

**Why accuracy misleads here:** a "model" that always answers "OK" scores **87%** accuracy and catches **zero** failures. With rare failures, recall and precision are what matter.

### 6.3 Is it overfitting?

Accuracy is **96%** on the training data and **84.5%** on the test data. The gap shows the trees have partly memorised the training set, noise included. `max_depth=8` keeps the gap from being even larger. Cross-validation and tuning `max_depth` or `min_samples_leaf` would be the next step.

### 6.4 Choosing a different cut-off

`failure_predicted` uses 0.5, but you can choose any cut-off on the probability:

| Cut-off | Caught failures | False alarms | Precision | Recall |
|---|---|---|---|---|
| 0.5 (current) | 17 of 26 | 22 | 43.6% | 65.4% |
| 0.4 | 19 of 26 | 31 | 38.0% | 73.1% |
| 0.3 | 19 of 26 | 45 | 29.7% | 73.1% |

Lowering the cut-off to 0.4 catches 2 more failures at the cost of 9 more alarms. Going to 0.3 adds 14 more alarms and no extra catches. The right point depends on what a missed failure costs compared with an unneeded inspection.

On the test set, the risk bands split the pumps into 150 LOW, 34 MEDIUM and 16 HIGH.

### 6.5 Which features matter: two ways to measure

| Feature | Built-in importance (Gini) | Permutation importance (drop in AUC) | In the label formula? |
|---|---|---|---|
| temperature_c | 0.243 | 0.128 | yes |
| vibration_mm_s | 0.196 | 0.102 | yes (largest weight) |
| operating_hours | 0.162 | 0.045 | yes |
| days_since_maintenance | 0.111 | 0.019 | yes |
| part_age_hours | 0.077 | 0.015 | yes |
| equipment_age_years | 0.070 | 0.005 | yes |
| pressure_bar | 0.059 | **0.000** | **no** |
| previous_failures | 0.043 | 0.012 | yes |
| maintenance_count | 0.039 | **0.002** | **no** |

- **Built-in (Gini) importance:** how much each feature reduced Gini across all splits in all trees. It's free, but it's measured on training data and tends to give credit to continuous noise features.
- **Permutation importance:** shuffle one feature in the **test set** and see how much the AUC drops. If the model really relies on the feature, shuffling hurts.

The comparison is a strong interview point. Gini importance gives the two noise features, `pressure_bar` and `maintenance_count`, about 4–6% each. Permutation importance correctly shows they add almost nothing (0.000 and 0.002). The model learned the real signals, and permutation importance is the more honest measure.

---

## 7. How to run everything

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env        # only needed for the LLM part

python src/generate_data.py    # recreate data/equipment_failure_data.csv
python src/train_model.py      # retrain and save the model, print metrics
python src/predict.py          # predict the demo pump (no API key needed)
python src/main.py             # prediction + LLM recommendation
python src/api.py              # API on http://127.0.0.1:5000
python -m pytest -q            # 20 tests, no API key needed
```

Run the scripts from the project root, because `generate_data.py` and `train_model.py` use relative paths.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Feature | One input measurement, such as temperature |
| Label / target | The answer the model learns, here `failure_next_30_days` |
| Supervised learning | Learning from examples where the answer is known |
| Binary classification | Predicting one of two classes (fail / not fail) |
| Decision tree | A learned flowchart of yes/no questions |
| Leaf | An end point of a tree; holds the failure share of its training rows |
| Gini impurity | How mixed a group is: 0 = pure, 0.5 = 50/50 |
| Bootstrap | Sampling rows with replacement, so each tree sees different data |
| Bagging | Training many models on bootstrap samples and averaging them |
| `max_features` | How many random features a split may consider |
| Class imbalance | One class (failure) is much rarer than the other |
| Class weight | Making mistakes on the rare class cost more during training |
| Stratified split | A train/test split that keeps the class ratio the same |
| Overfitting | Memorising training data, so test performance is worse |
| Precision | Of the alarms raised, the share that are real |
| Recall | Of the real failures, the share that are caught |
| ROC-AUC | How well scores rank positives above negatives |
| Permutation importance | The drop in score when a feature is shuffled |
| Sigmoid | A function that squashes any number into 0–1 |
| Pickle / joblib | Saving a Python object (the model) to a file |
| LCEL (`prompt \| llm`) | LangChain's syntax for connecting steps into a chain |
| 422 status | "Your request was understood but the data is invalid" |
