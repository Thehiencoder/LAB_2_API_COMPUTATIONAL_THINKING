"""
Automated Test Script for Sentiment Analysis API
Using the `requests` library to validate all required endpoints:
1. GET /        (System overview / Root)
2. GET /health  (System & model health check)
3. POST /predict (Valid positive sentence)
4. POST /predict (Valid negative sentence)
5. POST /predict (Validation error: whitespace-only string)
"""

import json
import sys
import requests

BASE_URL = "http://127.0.0.1:8000"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(endpoint: str, status_code: int, response_data: dict):
    print(f"[*] Target Endpoint : {endpoint}")
    print(f"[*] HTTP Status     : {status_code}")
    print("[*] Response JSON   :")
    print(json.dumps(response_data, indent=2, ensure_ascii=False))


def test_root():
    print_section("TEST 1: GET / (Root Information)")
    url = f"{BASE_URL}/"
    response = requests.get(url, timeout=10)
    print_result("GET /", response.status_code, response.json())
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "endpoints" in data, "Response should list endpoints"
    print("[+] Test 1 PASSED: Root endpoint returned valid system information.")


def test_health():
    print_section("TEST 2: GET /health (Health Check)")
    url = f"{BASE_URL}/health"
    response = requests.get(url, timeout=10)
    print_result("GET /health", response.status_code, response.json())
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy", f"Expected 'healthy', got {data.get('status')}"
    assert data["model_loaded"] is True, "Model should be reported as loaded"
    print("[+] Test 2 PASSED: Healthcheck confirms model is active and ready.")


def test_predict_positive():
    print_section("TEST 3: POST /predict (Positive Sentiment Input)")
    url = f"{BASE_URL}/predict"
    payload = {
        "text": "FastAPI combined with Hugging Face transformers makes deploying machine learning models an absolute delight!"
    }
    print(f"[*] Input Payload: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(url, json=payload, timeout=15)
    print_result("POST /predict", response.status_code, response.json())
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["label"] == "POSITIVE", f"Expected POSITIVE label, got {data['label']}"
    assert data["score"] > 0.5, "Confidence score should be > 0.5"
    print(f"[+] Test 3 PASSED: Correctly predicted '{data['label']}' with score {data['score']} ({data['inference_time_ms']} ms).")


def test_predict_negative():
    print_section("TEST 4: POST /predict (Negative Sentiment Input)")
    url = f"{BASE_URL}/predict"
    payload = {
        "text": "The application kept crashing on startup, the documentation was completely misleading and frustrating."
    }
    print(f"[*] Input Payload: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(url, json=payload, timeout=15)
    print_result("POST /predict", response.status_code, response.json())
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["label"] == "NEGATIVE", f"Expected NEGATIVE label, got {data['label']}"
    assert data["score"] > 0.5, "Confidence score should be > 0.5"
    print(f"[+] Test 4 PASSED: Correctly predicted '{data['label']}' with score {data['score']} ({data['inference_time_ms']} ms).")


def test_predict_validation_error():
    print_section("TEST 5: POST /predict (Input Validation - Empty/Whitespace Text)")
    url = f"{BASE_URL}/predict"
    payload = {
        "text": "    "  # Whitespace only to trigger validation error
    }
    print(f"[*] Input Payload: {json.dumps(payload, ensure_ascii=False)}")
    response = requests.post(url, json=payload, timeout=10)
    print_result("POST /predict", response.status_code, response.json())
    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, got {response.status_code}"
    print("[+] Test 5 PASSED: Validation logic properly rejected empty/whitespace input with HTTP 422.")


if __name__ == "__main__":
    print(f"Connecting to API server at: {BASE_URL}")
    try:
        # Quick check if server is reachable
        requests.get(BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"\n[!] ERROR: Could not connect to {BASE_URL}.")
        print("[!] Make sure the API server is running before executing this test script:")
        print("    uvicorn main:app --reload")
        print("    OR: python main.py\n")
        sys.exit(1)

    try:
        test_root()
        test_health()
        test_predict_positive()
        test_predict_negative()
        test_predict_validation_error()
        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED SUCCESSFULLY! ")
        print("=" * 70 + "\n")
    except AssertionError as ae:
        print(f"\n[X] TEST FAILED: {ae}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[X] UNEXPECTED ERROR: {exc}\n")
        sys.exit(1)
