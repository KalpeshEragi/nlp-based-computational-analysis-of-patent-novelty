"""
On-disk store for the prior-art corpus text (SQLite).

The search only needs IDs, titles, CPC codes and dates in memory; the long
texts (abstract, summary, claims) are read per patent when the analysis
needs them. This keeps the API's memory small (the full corpus text is ~3 GB
in pandas). Built once from corpus.parquet in batches, so building it is also
low-memory.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

TEXT_COLUMNS = ["patent_id", "title", "abstract", "summary", "claims",
                "publication_number", "filing_date"]


def build(index_dir: Path, batch_size: int = 2000) -> Path:
    import pyarrow.parquet as pq

    index_dir = Path(index_dir)
    db_path = index_dir / "texts.sqlite"
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute(f"CREATE TABLE docs ({', '.join(c + ' TEXT' for c in TEXT_COLUMNS)})")
    pf = pq.ParquetFile(index_dir / "corpus.parquet")
    for batch in pf.iter_batches(batch_size=batch_size, columns=TEXT_COLUMNS):
        cols = [batch.column(c).to_pylist() for c in TEXT_COLUMNS]
        con.executemany(f"INSERT INTO docs VALUES ({', '.join('?' * len(TEXT_COLUMNS))})",
                        zip(*cols))
    con.execute("CREATE UNIQUE INDEX idx_pid ON docs(patent_id)")
    con.commit()
    con.close()
    tmp.replace(db_path)
    return db_path


class TextStore:
    def __init__(self, index_dir: Path):
        db = Path(index_dir) / "texts.sqlite"
        if not db.exists():
            build(Path(index_dir))
        self._con = sqlite3.connect(db, check_same_thread=False)
        self._lock = threading.Lock()

    def get(self, patent_ids: list[str], columns: list[str]) -> dict[str, dict]:
        ids = list(dict.fromkeys(patent_ids))
        if not ids:
            return {}
        out = {}
        with self._lock:
            for i in range(0, len(ids), 500):  # SQLite parameter limit
                chunk = ids[i:i + 500]
                rows = self._con.execute(
                    f"SELECT patent_id, {', '.join(columns)} FROM docs "
                    f"WHERE patent_id IN ({', '.join('?' * len(chunk))})", chunk).fetchall()
                for r in rows:
                    out[r[0]] = {c: (v or "") for c, v in zip(columns, r[1:])}
        return out
