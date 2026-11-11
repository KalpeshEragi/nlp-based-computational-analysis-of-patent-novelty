"""
API test against the real models and index (slow: loads ~2 GB of models).

    pytest tests/test_api.py -s
"""

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app

PATENT = "US9968239B2"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        for _ in range(180):                     # wait up to 6 minutes for the models
            status = c.get("/api/health").json()["status"]
            if status != "loading":
                break
            time.sleep(2)
        assert status == "ok", c.get("/api/health").json()
        yield c


def test_rejects_unsupported_file(client):
    r = client.post("/api/documents", files={"file": ("sheet.xlsx", b"not a patent", "application/octet-stream")})
    assert r.status_code == 400


def test_rejects_document_without_claims(client):
    r = client.post("/api/analyses", json={"document": {"title": "x", "claims": "This text has no numbered claims at all."}})
    assert r.status_code == 400


def test_lookup_suggest_and_analyse(client):
    r = client.post("/api/documents/lookup", json={"number": PATENT})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["n_claims"] == 13 and doc["priority_date"] == "20150129"

    body = {k: doc[k] for k in ("title", "abstract", "summary", "claims", "source", "priority_date")}
    sug = client.post("/api/cpc/suggest", json=body).json()["suggestions"]
    assert len(sug) == 5 and all(len(s["code"]) == 4 for s in sug)
    print("\nCPC suggestions:", [(s["code"], s["prob"]) for s in sug])

    job = client.post("/api/analyses", json={"document": body, "mode": "fast",
                                             "cpc_codes": [s["code"] for s in sug[:3]]}).json()
    t0, last = time.time(), None
    while time.time() - t0 < 900:
        st = client.get(f"/api/analyses/{job['id']}").json()
        if st["stage"] != last:
            print(f"  {st['percent']:3d}%  {st['stage']}")
            last = st["stage"]
        if st["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert st["status"] == "done", st.get("error")
    res = st["result"]
    assert res["patent"]["summary"] and len(res["claims"]) == 13
    assert res["cpc"]["filter_note"]
    print(f"done in {st['elapsed_s']}s | {res['patent']['summary']}")
    print("verdicts:", res["patent"]["counts"])
    print("timing:", res["timing"])
