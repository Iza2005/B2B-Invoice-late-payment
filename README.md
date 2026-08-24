# Invoice Late-Payment Prediction API

## 1. The Machine Learning problem being solved

The goal of this project is to predict, at the moment a B2B invoice is issued, whether it is at 
risk of being paid late (`paid_late` = 0/1), using information already known at that time : amount, 
payment terms granted, the customer's tenure and credit score, the customer's history of late payments, how
many invoices are currently open for that customer, the month/weekday of issuance, and the
sales channel.

This is a binary classification problem. The business goal lies in letting a credit-control team
prioritize follow-ups before the due date rather than reacting once an invoice is already
overdue. 

The dataset (`notebooks/invoices.csv`, 6,000 invoices) is the one provided for this exercise, and it 
was made exclusively for this project.
About 19% of invoices in it are late, so the classes are imbalanced.

## 2. Methodology

1. **Load the data** (`notebooks/train.ipynb`) from `invoices.csv`.


2. **Exploratory analysis** with `seaborn`/`matplotlib`: target class distribution
   (`paid_late`), numeric feature distributions, correlation heatmap (`numpy` used to sort
   correlations by absolute value).


3. **Preparation** : split features/target, train/test split (80/20, stratified).


4. **Preprocessing and model in a single scikit-learn `Pipeline`**:
   - `StandardScaler` on the numeric features,
   - `OneHotEncoder` on the categorical `sales_channel` feature,
   - `RandomForestClassifier` (300 trees, `class_weight="balanced"` to compensate for the
     class imbalance).
   Bundling preprocessing and the model into a single `Pipeline` guarantees that the exact
   transformations used during training are automatically re-applied at inference time.


5. **Evaluation** on the test set : accuracy, per-class precision/recall, ROC AUC, confusion
   matrix, feature importances.


6. **Persistence** : the trained pipeline plus a bit of metadata (metrics, training timestamp,
   feature list) is serialized with `joblib` to `api/model.pkl`.


7. **Serving** : a FastAPI app (`api/main.py`) loads `model.pkl` once at startup and exposes
   the prediction through `POST /predict`.

To see the full detail of each step and executed code, check `notebooks/train.ipynb`.

## 3. Project structure

```
invoice-late-payment/
├── api/
│   ├── main.py            # FastAPI app with endpoints : /health, /predict
│   └── model.pkl           # Trained scikit-learn pipeline + metadata
├── notebooks/
│   ├── invoices.csv        # Provided dataset
│   └── train.ipynb         # Training notebook (EDA, training, evaluation, persistence)
├── tests/
│   └── test_api.py         # 4 pytest tests for the API
├── Dockerfile
├── requirements.txt
└── README.md
```

## 4. Run it locally (without Docker)

Requires Python 3.11+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the tests
pytest tests/ -v

# 4. Start the API
uvicorn api.main:app --reload
```

The API is then available at `http://localhost:8000`, with interactive Swagger docs at
`http://localhost:8000/docs`.

## 5. Run it with Docker

```bash
# Build the image (from the project root, where the Dockerfile is)
docker build -t invoice-late-payment-api .

# Run the container, mapping port 8000
docker run -p 8000:8000 invoice-late-payment-api
```

The API is then available at `http://localhost:8000`, exactly as in the local run.

> `api/model.pkl` must already exist before building the image (produced by
> `notebooks/train.ipynb`): the Dockerfile copies the `api/` folder, model included, into
> the image — no training happens inside the container.

## 6. Using the API

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_amount": 4200.50,
    "payment_terms_days": 30,
    "customer_tenure_months": 18.5,
    "customer_credit_score": 690,
    "previous_late_payments_count": 2,
    "avg_past_delay_days": 7.0,
    "number_of_open_invoices": 3,
    "discount_offered_pct": 0,
    "invoice_month": 11,
    "is_recurring_customer": 1,
    "weekday_issued": 2,
    "sales_channel": "direct"
  }'
```

Here is an example response :

```json
{
  "prediction": "late",
  "probability_late": 0.5281,
  "model_metadata": {
    "model_type": "RandomForestClassifier",
    "trained_at": "2026-08-23T19:31:08.169321+00:00",
    "metrics": {"accuracy": 0.7075, "roc_auc": 0.6612}
  }
}
```

On a missing or invalid field, the API responds `HTTP 400` with the list of errors :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"invoice_amount": 100}'
```
```json
{"errors": ["Missing required field: 'payment_terms_days'.", "..."]}
```

## 7. Technical choices and why

- **FastAPI** : I chose this simple framework because it turns a scikit-learn model into a REST API with minimal boilerplate, while auto-generating interactive documentation from the code itself, there a no separate spec to maintain.


- **Manual validation for the request** : The request body is read as a raw JSON dict and checked field-by-field in validate_payload(). This keeps the entire request contract, every rule the API enforces, visible in one plain function, rather than split across schema classes.


- **RandomForestClassifier** : I chose this thype of Machine Learning classifier model because late-payment risk factors interact non-linearly, for example, a poor credit score matters far more combined with a history of late payments than either alone, and a tree ensemble captures these interactions automatically, without hand-built feature crosses, and handles numeric and categorical inputs cleanly once encoded.


- **class_weight="balanced"** : In the dataset, the target is imbalanced (~19% late). Missing a genuinely risky invoice costs the business more than a false positive, so the minority class is upweighted rather than optimizing for accuracy, which would just favor predicting "on_time" every time.


- **Single scikit-learn Pipeline** : Preprocessing (scaling, encoding) and the model are saved as one fitted object. This guarantees the exact transformations used at training time are automatically reapplied at inference. Eliminating train/serve skew as a source of bugs.



- **Model loaded once at startup** : I chose to keep the pipeline loaded into memory when the module starts, rather than reloading on every request. This keeps the /predict endpoint latency low and avoids repeated disk I/O on each call.


- **Model metadata in every response** : model_type, trained_at, and metrics are returned alongside each prediction. This lets any API consumer know which model version answered and how trustworthy it currently is, without needing a separate endpoint.


- **python:3.11-slim base image** : Chosen over a full Python image because it's smaller and builds faster, while remaining fully compatible with every dependency this project needs.

## 8. Known limitations

- The dataset is the one provided for the exercise, the data is generated. On real data, feature importance and
  performance would need to be re-validated.


- No model versioning or registry : For production use, this would move to something like MLflow
  plus a dedicated `/model-info` endpoint. And for that purpose, the performances of the model could be better.


- No authentication on the API : It would need to be added (API key or OAuth) before exposing
  this beyond a local environment, but we assume that in this project, authentification is not needed.


- The current ROC AUC (~0.66) reflects the fact that late payment is only partly predictable
  from these features alone, and other variables (industry sector, macroeconomic context...) would
  likely improve it.
