"""Optional associative recall: vector embeddings over gists.

Honors the no-model-required baseline (Design §6.3): the core never imports
this module's optional dependency. With nothing installed, FornixDB works
exactly as in P1 (keyword + time recall). With `model2vec` installed (a small
static-embedding model — CPU-only, no torch, runs anywhere), recall gains a
similarity axis: "the glitch where her eyes sparkled" finds the eye-twinkle
memory with zero keyword overlap.

Vectors are stored as float32 little-endian BLOBs in the `embedding` table.
Search is exact brute-force cosine — at the few-thousand-memory scale of a hot
store this is sub-millisecond and avoids any index dependency. An ANN index
(sqlite-vec) becomes worthwhile only far beyond that; revisit at P3 scale.

Embedder protocol: any object with `.name` (str) and `.embed(texts) ->
list[list[float]]` works, so other backends (ONNX, llama.cpp, a remote box on
the LAN) can plug in without touching this file.
"""

from __future__ import annotations

import math
import struct
from typing import Protocol

DEFAULT_MODEL = "minishlab/potion-base-8M"  # ~30MB, CPU, no torch
MODEL_ENV = "FORNIXDB_EMBED_MODEL"          # repo id OR local directory
LOCAL_MODEL_CACHE = "~/.cache/fornixdb-models"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _resolve_model_source(model_name: str = DEFAULT_MODEL) -> str:
    """Resolution order: $FORNIXDB_EMBED_MODEL → local cache dir → repo id.
    The local-dir paths make offline/air-gapped installs first-class: drop the
    model files in the cache dir and no network is ever attempted."""
    import os
    from pathlib import Path
    env = os.environ.get(MODEL_ENV)
    if env:
        return env
    cached = Path(LOCAL_MODEL_CACHE).expanduser() / model_name.split("/")[-1]
    if (cached / "model.safetensors").exists():
        return str(cached)
    return model_name


class Model2VecEmbedder:
    """Default embedder: model2vec static embeddings (tiny, fast, local)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from model2vec import StaticModel  # optional dep, import deferred
        self.name = f"model2vec:{model_name.split('/')[-1]}"
        self._model = StaticModel.from_pretrained(_resolve_model_source(model_name))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts)]


_default_embedder = "unset"


def get_default_embedder() -> Embedder | None:
    """Best available local embedder, or None — never raises, never required.
    Cached per process: callers (shims, adapters) invoke this per store/embed
    call, and reconstructing the model each time re-reads it from disk."""
    global _default_embedder
    if _default_embedder == "unset":
        try:
            _default_embedder = Model2VecEmbedder()
        except Exception:
            _default_embedder = None
    return _default_embedder


# ------------------------------------------------------------- blob helpers

def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def from_blob(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------- store ops

# A memory embeds as several chunks: chunk 0 = name + gist (the headline),
# chunks 1..n = detail windows. Recall scores a memory by its BEST chunk, so
# a paraphrase of a fact buried deep in a long detail still finds the memory
# (eval-found gap: a single gist+detail[:500] vector missed exactly that).
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 8  # headline + up to 7 detail windows; bounds store growth


def _chunk_texts(row) -> list[str]:
    name = (row["name"] or "").replace("-", " ").replace("_", " ")
    head = f"{name}\n{row['gist'] or ''}".strip()
    out = [head]
    detail = row["detail"] or ""
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for start in range(0, len(detail), step):
        if len(out) >= MAX_CHUNKS:
            break
        piece = detail[start:start + CHUNK_CHARS].strip()
        if piece:
            out.append(piece)
    return out


def _write_chunks(store, embedder: Embedder, rows) -> None:
    texts, keys = [], []
    for r in rows:
        for ci, t in enumerate(_chunk_texts(r)):
            texts.append(t)
            keys.append((r["id"], ci))
    vecs = embedder.embed(texts)
    ids = list({mid for mid, _ in keys})
    store.conn.executemany("DELETE FROM embedding WHERE memory_id = ?",
                           [(i,) for i in ids])  # no stale higher chunks
    store.conn.executemany(
        "INSERT INTO embedding(memory_id, chunk, model, dim, vector) VALUES (?,?,?,?,?)",
        [(mid, ci, embedder.name, len(v), to_blob(v))
         for (mid, ci), v in zip(keys, vecs)],
    )
    store.conn.commit()


def embed_memory(store, embedder: Embedder, memory_id: int) -> None:
    row = store.conn.execute(
        "SELECT id, name, gist, detail FROM memory WHERE id = ?",
        (memory_id,)).fetchone()
    if row is not None:
        _write_chunks(store, embedder, [row])


def backfill(store, embedder: Embedder, batch: int = 64) -> int:
    """Embed every memory that lacks vectors for this model. Idempotent."""
    rows = store.conn.execute(
        """SELECT m.id, m.name, m.gist, m.detail FROM memory m
           LEFT JOIN embedding e ON e.memory_id = m.id AND e.model = ?
           WHERE e.memory_id IS NULL""",
        (embedder.name,),
    ).fetchall()
    done = 0
    for i in range(0, len(rows), batch):
        _write_chunks(store, embedder, rows[i:i + batch])
        done += len(rows[i:i + batch])
    return done


def similar(store, embedder: Embedder, query: str, *, limit: int = 25,
            include_superseded: bool = False) -> list[tuple[int, float]]:
    """(memory_id, cosine) nearest the query, best first — a memory scores
    as its best-matching chunk."""
    qvec = embedder.embed([query])[0]
    where = "" if include_superseded else \
        "JOIN memory m ON m.id = e.memory_id AND m.superseded_time IS NULL"
    best: dict[int, float] = {}
    for r in store.conn.execute(
            f"SELECT e.memory_id, e.vector FROM embedding e {where} WHERE e.model = ?",
            (embedder.name,)):
        cos = cosine(qvec, from_blob(r["vector"]))
        if cos > best.get(r["memory_id"], -1.0):
            best[r["memory_id"]] = cos
    scored = sorted(best.items(), key=lambda t: t[1], reverse=True)
    return scored[:limit]
