"""Task 5 - Python client to test the FastAPI /health and /predict endpoints.

Start the server first:  uvicorn app:app
"""
import requests

BASE_URL = "http://127.0.0.1:8000"

valid = {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
         "Parch": 0, "Fare": 100.0, "Embarked": "S"}
invalid = {"Pclass": 7, "Sex": "unknown", "Age": -5, "SibSp": 0,
           "Parch": 0, "Fare": "abc"}

r = requests.get(f"{BASE_URL}/health")
print("GET /health            ->", r.status_code, r.json())

r = requests.post(f"{BASE_URL}/predict", json=valid)
print("POST /predict (valid)  ->", r.status_code, r.json())

r = requests.post(f"{BASE_URL}/predict", json=invalid)
print("POST /predict (invalid)->", r.status_code)
for err in r.json()["detail"]:
    print("    -", err["loc"][-1], ":", err["msg"])
