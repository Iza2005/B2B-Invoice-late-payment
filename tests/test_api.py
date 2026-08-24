"""
4 pytest tests for the invoice late-payment API.

Uses only the stdlib `urllib` HTTP client against a real uvicorn server
started for the test session.
"""

import json
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError(f"Server at {url} did not start in time.")


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            HOST,
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://{HOST}:{port}"
    try:
        _wait_for_server(f"{url}/health")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=5)


VALID_PAYLOAD = {
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
    "sales_channel": "direct",
}

# The 4 tests to run for the invoice API that use the functions above :
def test_health_returns_ok(base_url):
    """This function tests the correct functionning of the API"""
    status, body = _get(f"{base_url}/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_with_valid_payload_returns_prediction(base_url):
    """This function tests the response of the API with a valid invoice"""
    status, body = _post_json(f"{base_url}/predict", VALID_PAYLOAD)
    assert status == 200
    assert body["prediction"] in {"on_time", "late"}
    assert 0.0 <= body["probability_late"] <= 1.0
    assert body["model_metadata"]["model_type"] == "RandomForestClassifier"


def test_predict_with_missing_field_returns_400(base_url):
    """This function tests the response of the API with invalid invoice (missing features)"""
    incomplete_payload = VALID_PAYLOAD.copy()
    del incomplete_payload["customer_credit_score"]

    status, body = _post_json(f"{base_url}/predict", incomplete_payload)
    assert status == 400
    assert any("customer_credit_score" in msg for msg in body["errors"])


def test_predict_with_invalid_sales_channel_returns_400(base_url):
    """This function tests the response of the API with invalid sales_channel values"""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["sales_channel"] = "carrier_pigeon"

    status, body = _post_json(f"{base_url}/predict", bad_payload)
    assert status == 400
    assert any("sales_channel" in msg for msg in body["errors"])
