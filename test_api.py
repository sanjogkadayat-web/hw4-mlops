"""
HW4 API Test Script
Runs 5 test cases against the Flask API.
Usage:
    python test_api.py                              # test local
    python test_api.py https://your-app.onrender.com  # test deployed
"""

import sys
import requests
import json

# ============================================================
# Configuration
# ============================================================
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
PASSED = 0
FAILED = 0


def report(test_name, passed, detail=""):
    global PASSED, FAILED
    status = "PASS" if passed else "FAIL"
    if passed:
        PASSED += 1
    else:
        FAILED += 1
    print(f"  [{status}] {test_name}")
    if detail:
        print(f"         {detail}")


print(f"\nTesting API at: {BASE_URL}\n")
print("=" * 60)

# ============================================================
# Test 1: Health Check
# ============================================================
print("\nTest 1: GET /health")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=60)
    data = r.json()
    passed = (
        r.status_code == 200
        and data.get("status") == "healthy"
        and data.get("model") == "loaded"
    )
    report("Health check returns status and model fields", passed, f"Response: {data}")
except Exception as e:
    report("Health check returns status and model fields", False, str(e))

# ============================================================
# Test 2: Valid Single Prediction
# ============================================================
print("\nTest 2: POST /predict (valid input)")
valid_record = {
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
}
try:
    r = requests.post(f"{BASE_URL}/predict", json=valid_record, timeout=60)
    data = r.json()
    passed = (
        r.status_code == 200
        and "prediction" in data
        and "probability" in data
        and "label" in data
        and data["prediction"] in [0, 1]
        and 0 <= data["probability"] <= 1
        and data["label"] in ["positive", "negative"]
    )
    report("Single prediction returns prediction, probability, label", passed, f"Response: {data}")
except Exception as e:
    report("Single prediction returns prediction, probability, label", False, str(e))

# ============================================================
# Test 3: Valid Batch Prediction (5 records)
# ============================================================
print("\nTest 3: POST /predict/batch (5 valid records)")
batch_records = []
for i in range(5):
    record = valid_record.copy()
    record["delivery_days"] = 5 + i * 3          # vary delivery days
    record["price"] = 30.0 + i * 20              # vary price
    record["total_cost"] = record["price"] + record["freight_value"]
    batch_records.append(record)

try:
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch_records, timeout=60)
    data = r.json()
    passed = (
        r.status_code == 200
        and "predictions" in data
        and "count" in data
        and data["count"] == 5
        and len(data["predictions"]) == 5
    )
    report("Batch returns 5 predictions with count", passed, f"Count: {data.get('count')}")
except Exception as e:
    report("Batch returns 5 predictions with count", False, str(e))

# ============================================================
# Test 4: Missing Required Field → 400
# ============================================================
print("\nTest 4: POST /predict (missing field)")
incomplete_record = valid_record.copy()
del incomplete_record["delivery_days"]  # remove a required field

try:
    r = requests.post(f"{BASE_URL}/predict", json=incomplete_record, timeout=60)
    data = r.json()
    passed = (
        r.status_code == 400
        and "error" in data
        and "details" in data
    )
    report("Missing field returns 400 with error details", passed, f"Response: {data}")
except Exception as e:
    report("Missing field returns 400 with error details", False, str(e))

# ============================================================
# Test 5: Invalid Type → 400
# ============================================================
print("\nTest 5: POST /predict (invalid type — string for price)")
bad_type_record = valid_record.copy()
bad_type_record["price"] = "not_a_number"  # string where number expected

try:
    r = requests.post(f"{BASE_URL}/predict", json=bad_type_record, timeout=60)
    data = r.json()
    passed = (
        r.status_code == 400
        and "error" in data
        and "details" in data
    )
    report("Invalid type returns 400 with error details", passed, f"Response: {data}")
except Exception as e:
    report("Invalid type returns 400 with error details", False, str(e))

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"\nResults: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED} tests")
if FAILED == 0:
    print("All tests passed!")
else:
    print("Some tests failed. Review output above.")
print()
