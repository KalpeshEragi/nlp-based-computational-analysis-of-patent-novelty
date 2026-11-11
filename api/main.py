"""
Patent Analyser API.

    uvicorn api.main:app --port 8000          # docs at http://localhost:8000/docs

Endpoints
    GET  /api/health                 ok / loading / error
    POST /api/documents              upload PDF, DOCX or TXT -> title, abstract, summary, claims
    POST /api/documents/lookup       patent number -> the same fields (Google Patents)
    POST /api/cpc/suggest            document text -> top CPC subclasses
    POST /api/analyses               start an analysis job -> job id
    GET  /api/analyses/{job_id}      status, stage, progress, result

Models load once in a background thread at startup. Analyses run one at a
time in a worker thread (fine for a single-user demo); jobs live in memory
and are lost on restart.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analysis.elements import parse_claims
from analysis.extractor import ExtractedPatent, extract, fetch_patent

log = logging.getLogger("patent-analyser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_UPLOAD_MB = 20
MAX_JOBS_KEPT = 20
DEFAULT_MODE = os.environ.get("ANALYSIS_MODE", "fast")


# ----------------------------------------------------------------------------
# Engine: models loaded once, in the background
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.status, self.error, self.analyzer = "loading", None, None
        self.loaded_in_s = None

    def load(self):
        t = time.time()
        try:
            from analysis.analyzer import Analyzer
            a = Analyzer()
            a.search.source._load()      # index, vectors, text store
            _ = a.search.source.encoder  # PatentSBERTa
            _ = a.checker                # disclosure model
            _ = a.search.predictor       # CPC classifier
            self.analyzer = a
            self.status = "ok"
            self.loaded_in_s = round(time.time() - t, 1)
            log.info("models loaded in %.1fs", self.loaded_in_s)
        except Exception as e:  # keep the API up and report the failure
            self.status, self.error = "error", repr(e)
            log.exception("model loading failed")

    def require(self):
        if self.status == "loading":
            raise HTTPException(503, "models are still loading, try again in a minute")
        if self.status == "error":
            raise HTTPException(500, f"models failed to load: {self.error}")
        return self.analyzer


engine = Engine()


# ----------------------------------------------------------------------------
# Jobs: one worker, in-memory store
# ----------------------------------------------------------------------------
jobs: "OrderedDict[str, dict]" = OrderedDict()
job_queue: "queue.Queue[str]" = queue.Queue()


def _worker():
    while True:
        job_id = job_queue.get()
        job = jobs.get(job_id)
        if job is None:
            continue
        job.update(status="running", started=time.time())

        def progress(stage: str, pct: int):
            job.update(stage=stage, percent=pct)

        try:
            a = engine.analyzer
            doc = ExtractedPatent(**job["document"])
            result = a.analyze(doc, before_date=job["before_date"], cpc_codes=job["cpc_codes"],
                               mode=job["mode"], progress=progress)
            # round-trip through JSON so numpy / dataclass leftovers become plain values
            job["result"] = json.loads(json.dumps(result, default=str))
            job.update(status="done", stage="done", percent=100)
        except Exception as e:
            log.exception("analysis %s failed", job_id)
            job.update(status="error", error=str(e) or repr(e))
        finally:
            job["finished"] = time.time()
            try:
                engine.analyzer.reset_caches()
            except Exception:
                pass
            while len(jobs) > MAX_JOBS_KEPT:
                jobs.popitem(last=False)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="Patent Analyser API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    threading.Thread(target=engine.load, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()


class DocumentIn(BaseModel):
    title: str = ""
    abstract: str = ""
    summary: str = ""
    claims: str = Field(..., min_length=10)
    source: str = ""
    priority_date: str = ""


class LookupIn(BaseModel):
    number: str = Field(..., min_length=4, examples=["US9968239B2"])


class AnalysisIn(BaseModel):
    document: DocumentIn
    cpc_codes: list[str] = []
    before_date: str | None = Field(None, pattern=r"^\d{8}$",
                                    description="YYYYMMDD; defaults to the document's priority date")
    mode: str = Field(DEFAULT_MODE, pattern="^(fast|full)$")


@app.get("/api/health")
def health():
    return {"status": engine.status, "error": engine.error, "loaded_in_s": engine.loaded_in_s,
            "queued": job_queue.qsize(), "default_mode": DEFAULT_MODE}


def _document_response(doc: ExtractedPatent) -> dict:
    d = doc.to_dict()
    d["claims_parsed"] = [{"num": c.num, "parent": c.parent, "independent": c.independent,
                           "text": c.text} for c in parse_claims(doc.claims)]
    if not d["claims_parsed"]:
        d["warnings"] = d["warnings"] + ["no numbered claims found; check the document"]
    return d


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
    try:
        doc = extract(data, file.filename or "upload.txt")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(422, f"could not read the document: {e}")
    return _document_response(doc)


@app.post("/api/documents/lookup")
def lookup_document(body: LookupIn):
    try:
        doc = fetch_patent(body.number)
    except Exception as e:
        raise HTTPException(502, f"could not fetch {body.number} from Google Patents: {e}")
    if not doc.claims:
        raise HTTPException(404, f"no claims found for {body.number}")
    return _document_response(doc)


@app.post("/api/cpc/suggest")
def suggest_cpc(body: DocumentIn, k: int = 5):
    a = engine.require()
    from retrieval.sources import Query
    return {"suggestions": a.search.suggest_cpc(
        Query(body.title, body.abstract, body.summary, body.claims), k)}


@app.post("/api/analyses", status_code=202)
def start_analysis(body: AnalysisIn):
    engine.require()
    if not parse_claims(body.document.claims):
        raise HTTPException(400, "no numbered claims found in the document")
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id, "status": "queued", "stage": "queued", "percent": 0,
        "created": time.time(), "mode": body.mode,
        "cpc_codes": [c.strip().upper()[:4] for c in body.cpc_codes if c.strip()],
        "before_date": body.before_date or body.document.priority_date or time.strftime("%Y%m%d"),
        "document": body.document.model_dump(include={"title", "abstract", "summary", "claims",
                                                      "source", "priority_date"}),
        "result": None, "error": None,
    }
    job_queue.put(job_id)
    return {"id": job_id, "status": "queued", "position": job_queue.qsize()}


@app.get("/api/analyses/{job_id}")
def get_analysis(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown or expired job id")
    out = {k: v for k, v in job.items() if k != "document"}
    if job.get("started"):
        out["elapsed_s"] = round((job.get("finished") or time.time()) - job["started"], 1)
    return out


# the built React app (M3) is served from the same server when present
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
