# Olist Customer Satisfaction Prediction API

A production ML API that predicts whether an Olist e-commerce order will receive a positive (4–5 stars) or negative (1–3 stars) customer review, based on order and delivery characteristics.

## Live URL

**https://hw4-mlops.onrender.com**

> Free tier services spin down after inactivity. The first request may take 30–60 seconds while the service wakes up.

## API Documentation

### GET /health

Health check endpoint confirming the API and model are operational.

**Response:**
```json
{"status": "healthy", "model": "loaded"}
```

### POST /predict

Single prediction. Send a JSON object with feature values.

**Request:**
```bash
curl -X POST https://hw4-mlops.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "delivery_days": 8,
    "delivery_vs_estimated": -8,
    "price": 29.99,
    "freight_value": 8.72,
    "seller_state": "SP",
    "payment_type": "credit_card",
    "payment_installments": 1,
    "payment_value": 38.71,
    "product_weight_g": 500,
    "product_length_cm": 19,
    "product_height_cm": 8,
    "product_width_cm": 13,
    "customer_state": "SP",
    "total_cost": 38.71,
    "log_price": 3.43,
    "is_late": 0
  }'
```

**Response:**
```json
{"prediction": 1, "probability": 0.87, "label": "positive"}
```

### POST /predict/batch

Batch prediction. Send a JSON array of up to 100 records.

**Request:** JSON array of objects (same schema as /predict)

**Response:**
```json
{"predictions": [{"prediction": 1, "probability": 0.87, "label": "positive"}, ...], "count": 5}
```

### Error Responses

All endpoints return informative 400 errors for invalid input:
```json
{"error": "Invalid input", "details": {"price": "must be a positive number"}}
```

## Input Schema

| Feature | Type | Valid Values/Range |
|---------|------|--------------------|
| delivery_days | float | >= 0 |
| delivery_vs_estimated | float | any (negative = early, positive = late) |
| price | float | >= 0 |
| freight_value | float | >= 0 |
| seller_state | string | Brazilian state codes (SP, RJ, MG, PR, etc.) |
| payment_type | string | credit_card, boleto, voucher, debit_card, not_defined |
| payment_installments | float | >= 0 |
| payment_value | float | >= 0 |
| product_weight_g | float | >= 0 |
| product_length_cm | float | >= 0 |
| product_height_cm | float | >= 0 |
| product_width_cm | float | >= 0 |
| customer_state | string | Brazilian state codes (SP, RJ, MG, PR, etc.) |
| total_cost | float | >= 0 (price + freight_value) |
| log_price | float | any (log1p of price) |
| is_late | int | 0 or 1 (1 if delivery_vs_estimated > 0) |

## Local Setup

### Without Docker

```bash
git clone https://github.com/sanjogkadayat-web/hw4-mlops.git
cd hw4-mlops
pip install -r requirements.txt
python app.py
# API runs on http://localhost:5000
```

### With Docker

```bash
git clone https://github.com/sanjogkadayat-web/hw4-mlops.git
cd hw4-mlops
docker build -t hw4-api .
docker run -p 5000:5000 hw4-api
# API runs on http://localhost:5000
```

### Testing

```bash
python test_api.py                                    # test local
python test_api.py https://hw4-mlops.onrender.com     # test deployed
```

## Model Information

- **Model:** Random Forest Classifier (scikit-learn Pipeline)
- **Hyperparameters:** n_estimators=50, max_depth=15, min_samples_split=5
- **Test Set Performance:** F1 ≈ 0.88, ROC-AUC ≈ 0.76
- **Training Data:** Brazilian E-Commerce (Olist) dataset — 94,516 training records
- **Features:** 16 features (13 numeric, 3 categorical) covering delivery timing, pricing, payment behavior, and product attributes
- **Top Predictors (SHAP):** delivery_days, delivery_vs_estimated, payment_value

### Known Limitations

- Model was trained on historical Olist data (2016–2018) and may not reflect current e-commerce patterns
- Free tier deployment has cold start latency (30–60 seconds after inactivity)
- Categorical features are validated against known Brazilian state codes — new states or payment types will be rejected
- Model file size was reduced for free tier deployment, resulting in slightly lower performance than the original HW2 model