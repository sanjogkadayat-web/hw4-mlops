import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# ============================================================
# Initialize app and load model (loads ONCE at startup)
# ============================================================
app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'model/model.pkl')
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"Model loaded: {type(model).__name__}")

# Feature definitions (from your HW2 pipeline)
NUMERIC_FEATURES = [
    "delivery_days", "delivery_vs_estimated", "price", "freight_value",
    "payment_installments", "payment_value", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
    "total_cost", "log_price", "is_late"
]

CATEGORICAL_FEATURES = ["seller_state", "payment_type", "customer_state"]

REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# Helper: Input Validation
# ============================================================
def validate_input(data):
    """Validate a single prediction input. Returns (errors_dict, None) or (None, df)."""
    errors = {}

    # Check missing fields
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        errors["missing_fields"] = missing
        return errors, None

    # Check numeric types and values
    for f in NUMERIC_FEATURES:
        val = data[f]
        if not isinstance(val, (int, float)):
            errors[f] = "must be a number"
            continue
        if f in ["price", "freight_value", "payment_value", "total_cost",
                  "product_weight_g", "product_length_cm",
                  "product_height_cm", "product_width_cm"] and val < 0:
            errors[f] = "must be a positive number"

    # Check categorical values
    valid_categories = {
        "payment_type": ["credit_card", "boleto", "voucher", "debit_card", "not_defined"],
        "seller_state": [
            "SP", "PR", "MG", "RJ", "SC", "RS", "BA", "GO", "ES", "PE",
            "CE", "MA", "MS", "MT", "DF", "RN", "PB", "PA", "PI", "RO",
            "AM", "SE", "AC", "AL", "AP", "RR", "TO"
        ],
        "customer_state": [
            "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "CE", "GO", "ES",
            "PE", "DF", "MA", "MT", "MS", "PB", "PA", "PI", "RN", "AL",
            "SE", "AM", "RO", "AC", "AP", "TO", "RR"
        ]
    }
    for f in CATEGORICAL_FEATURES:
        val = data[f]
        if not isinstance(val, str):
            errors[f] = "must be a string"
        elif f in valid_categories and val not in valid_categories[f]:
            errors[f] = f"invalid value '{val}'. Must be one of {valid_categories[f]}"

    if errors:
        return errors, None

    # Build DataFrame for prediction
    df = pd.DataFrame([{f: data[f] for f in REQUIRED_FEATURES}])
    return None, df


# ============================================================
# Endpoint 1: Health Check — GET /health
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "loaded"}), 200


# ============================================================
# Endpoint 2: Single Prediction — POST /predict
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        errors, df = validate_input(data)
        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 400

        prediction = int(model.predict(df)[0])
        probability = round(float(model.predict_proba(df)[0][1]), 4)
        label = "positive" if prediction == 1 else "negative"

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "label": label
        }), 200

    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


# ============================================================
# Endpoint 3: Batch Prediction — POST /predict/batch
# ============================================================
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if data is None or not isinstance(data, list):
            return jsonify({"error": "Request body must be a JSON array"}), 400

        if len(data) > 100:
            return jsonify({"error": "Batch size exceeds limit of 100 records"}), 400

        # Validate each record
        all_dfs = []
        for i, record in enumerate(data):
            errors, df = validate_input(record)
            if errors:
                return jsonify({
                    "error": f"Invalid input at index {i}",
                    "details": errors
                }), 400
            all_dfs.append(df)

        # Combine and predict
        batch_df = pd.concat(all_dfs, ignore_index=True)
        preds = model.predict(batch_df)
        probas = model.predict_proba(batch_df)[:, 1]

        results = [{
            "prediction": int(p),
            "probability": round(float(pr), 4),
            "label": "positive" if p == 1 else "negative"
        } for p, pr in zip(preds, probas)]

        return jsonify({
            "predictions": results,
            "count": len(results)
        }), 200

    except Exception as e:
        return jsonify({"error": "Batch prediction failed", "message": str(e)}), 500


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    