"""Optional associative recall: vector embeddings over gists.

Honors the no-model-required baseline (Design §6.3): the core never imports
this module's optional dependency. With nothing installed, FornixDB works
exactly as in P1 (keyword + time recall). With `model2vec` installed (a small
static-embedding model — CPU-only, no torch, runs anywhere), recall gains a
similarity axis: "the glitch where her eyes sparkled" finds the eye-twinkle
memory with zero keyword overlap.

Vectors are stored as float32 little-endian BLOBs in the `embedding` table.
Search is exact brute-force cosine — every chunk scored, no index dependency,
no approximation. Scoring is one matmul per block where numpy is available and
the same arithmetic row by row where it is not; measured on a live store of
24.6k chunks, exact search costs ~21ms, of which ~17ms is reading the blobs out
of SQLite. An ANN index (sqlite-vec) buys nothing until well past that, and
costs the exactness; revisit only when the read itself stops being the floor.

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
        import os
        # cosmetic: without this the cached-model load prints a "Fetching N
        # files" progress bar to stderr on EVERY recall — noise on each
        # proactive-recall hook turn. setdefault so an explicit user value wins.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from model2vec import StaticModel  # optional dep, import deferred
        self.name = f"model2vec:{model_name.split('/')[-1]}"
        self._model = StaticModel.from_pretrained(_resolve_model_source(model_name))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.encode(texts)]


_default_embedder = "unset"
EMBEDDER_ENV = "FORNIXDB_EMBEDDER"  # "package.module:factory" — the warm hook


def set_default_embedder(embedder: "Embedder | None") -> None:
    """Inject the process-wide embedder, overriding the model2vec default.

    THE WARM-EMBEDDER HOOK. By design FornixDB cold-loads model2vec in a fresh
    process per recall (the proactive-recall hook is a new process each turn).
    Cold is the only behavior portable across the unknown hardware/OS FornixDB
    targets — microcontroller to server — so it is the default. But a *specific*
    deployment on *known* hardware that runs FornixDB inside one long-lived
    process — a humanoid robot, a resident agent — can keep the model warm in
    RAM and inject it here ONCE at startup; every recall path (CLI, MCP, the
    proactive hook via core, shims, consolidation) then reuses it with no
    per-turn load. See INTEGRATION.md "Warm embedding on dedicated hardware."

    Safety rule that keeps warm correct: a warm Embedder may cache only the
    immutable model. NEVER cache the DB row/vector set — recall re-reads the
    store every call, so memories written moments ago are always visible; a
    cached row set would go stale on the next write. Pass None to force the
    keyword-only baseline (no vectors)."""
    global _default_embedder
    _default_embedder = embedder


def _load_env_embedder() -> "Embedder | None":
    """$FORNIXDB_EMBEDDER='pkg.module:factory' → import and call it (no args).
    Lets a deployment swap in a warm/daemon/ONNX/remote backend with no fork
    and no code edit — the env-var twin of set_default_embedder(). Never raises:
    a missing/bad spec falls through to the bundled model2vec default."""
    import os
    spec = os.environ.get(EMBEDDER_ENV)
    if not spec:
        return None
    try:
        import importlib
        mod_name, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(mod_name), attr or "embedder")
        return obj() if callable(obj) else obj
    except Exception:
        return None


def get_default_embedder() -> Embedder | None:
    """Best available local embedder, or None — never raises, never required.
    Resolution, first hit wins: an injected embedder (set_default_embedder) →
    $FORNIXDB_EMBEDDER factory → the bundled model2vec → None (keyword-only).
    Cached per process; reconstructing model2vec each call would re-read it from
    disk, which is the cold cost a warm injection removes."""
    global _default_embedder
    if _default_embedder == "unset":
        emb = _load_env_embedder()
        if emb is None:
            try:
                emb = Model2VecEmbedder()
            except Exception:
                emb = None
        _default_embedder = emb
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


# ------------------------------------------------------- scoring many vectors

def _numpy():
    """numpy if it imports, else None — the fast path, never a requirement.
    numpy arrives with model2vec, so in practice it is present exactly when
    vectors are; but an injected embedder (set_default_embedder) can bring its
    own backend with no numpy at all, and a store built with vectors must still
    open on a machine without them. Probed per call: the import is cached by
    Python, and deciding once at module load would freeze the answer for
    processes that install numpy later."""
    try:
        import numpy
        return numpy
    except Exception:
        return None


# Rows are scored in blocks so peak memory stays bounded by the block, not by
# the store. The old loop held one vector at a time; reading every blob at once
# would be ~25MB at today's scale and ~700MB at the size the disk budget
# permits, which is not a trade worth making for a few milliseconds.
#
# Changing this constant perturbs the LAST BIT of a cosine (~3e-08, one float32
# ULP) without changing any ranking: BLAS chooses its accumulation strategy by
# matrix shape, so a different block size sums the same products in a different
# order. Verified on the live store — block 7 vs block 8192 gives byte-different
# scores on 18 of 25 rows and the identical row order. Worth knowing before
# treating a cosine as reproducible to full precision across machines.
_SCORE_BLOCK = 8192  # chunks per matmul (~8MB at 256 dims)


def _best_by_memory(qvec: list[float], rows) -> dict[int, float]:
    """memory_id → best cosine over its chunks, for rows of (memory_id, vector).

    A memory scores as its best-matching chunk, so this is where every
    vector-recall path converges. With numpy the per-block cosines are one
    matmul; without it, the identical arithmetic one row at a time. The two
    paths agree to ~6e-08 (float32 matmul vs float64 scalars) and return the
    same ranking — verified against the live store, top-10 identical.

    Grouping stays a plain dict: a memory has few chunks, and a numpy
    sort-and-segment measured SLOWER than the dict it would replace."""
    np = _numpy()
    best: dict[int, float] = {}
    block_ids: list[int] = []
    block_blobs: list[bytes] = []

    def flush() -> None:
        if not block_blobs:
            return
        dim = len(block_blobs[0]) // 4
        buf = b"".join(block_blobs)
        if not dim or len(buf) != len(block_blobs) * dim * 4:
            # ragged: one model reporting two dimensionalities. Shouldn't
            # happen (rows are filtered by model) but a reshape would raise,
            # and a scoring path must degrade rather than fail.
            for mid, blob in zip(block_ids, block_blobs):
                c = cosine(qvec, from_blob(blob))
                if c > best.get(mid, -1.0):
                    best[mid] = c
            return
        m = np.frombuffer(buf, dtype="<f4").reshape(len(block_blobs), dim)
        q = np.asarray(qvec, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1)
        norms[norms == 0] = 1.0  # a zero vector scores 0.0, as cosine() gives
        qn = float(np.linalg.norm(q)) or 1.0
        # .tolist() and not float(): callers compare and serialize these, and a
        # numpy scalar leaking into a result row reads as a float until it hits
        # json.dumps
        sims = ((m @ q) / (norms * qn)).tolist()
        for mid, c in zip(block_ids, sims):
            if c > best.get(mid, -1.0):
                best[mid] = c

    for r in rows:
        if np is None:
            c = cosine(qvec, from_blob(r["vector"]))
            mid = r["memory_id"]
            if c > best.get(mid, -1.0):
                best[mid] = c
            continue
        block_ids.append(r["memory_id"])
        block_blobs.append(r["vector"])
        if len(block_blobs) >= _SCORE_BLOCK:
            flush()
            block_ids, block_blobs = [], []
    flush()
    return best


# ---------------------------------------------------------------- store ops

# A memory embeds as several chunks: chunk 0 = name + gist (the headline),
# chunks 1..n = detail windows. Recall scores a memory by its BEST chunk, so
# a paraphrase of a fact buried deep in a long detail still finds the memory
# (eval-found gap: a single gist+detail[:500] vector missed exactly that).
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 8  # headline + up to 7 detail windows; bounds store growth


# A sense capture's gist is a bare caption ("acoustic guitar", "whistling") —
# 1-3 words whose vector structurally can't overlap a long query the way a
# multi-sentence gist does, so captions sat below the proactive floors on
# queries they were dead-on for (measured live on a second-consumer store, 2026-07-18:
# "acoustic guitar" cos 0.466 vs the 0.50 field floor on a guitar-demo query;
# "heard: acoustic guitar" 0.588). The row MEANS "I heard X" — embed that.
_SENSE_PREFIX = {"sight": "saw:", "sound": "heard:", "feel": "felt:"}


def _sense_prefix(row) -> str:
    try:
        source = row["source"] or ""
    except (IndexError, KeyError):
        return ""
    if not source.startswith("senses:"):
        return ""
    return _SENSE_PREFIX.get(source.split(":", 1)[1], "sensed:")


def _chunk_texts(row) -> list[str]:
    name = (row["name"] or "").replace("-", " ").replace("_", " ")
    prefix = _sense_prefix(row)
    gist = f"{prefix} {row['gist'] or ''}".strip() if prefix else (row["gist"] or "")
    head = f"{name}\n{gist}".strip()
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
        "SELECT id, name, gist, detail, source FROM memory WHERE id = ?",
        (memory_id,)).fetchone()
    if row is not None:
        _write_chunks(store, embedder, [row])


def backfill(store, embedder: Embedder, batch: int = 64) -> int:
    """Embed every memory that lacks vectors for this model. Idempotent."""
    rows = store.conn.execute(
        """SELECT m.id, m.name, m.gist, m.detail, m.source FROM memory m
           LEFT JOIN embedding e ON e.memory_id = m.id AND e.model = ?
           WHERE e.memory_id IS NULL""",
        (embedder.name,),
    ).fetchall()
    done = 0
    for i in range(0, len(rows), batch):
        _write_chunks(store, embedder, rows[i:i + batch])
        done += len(rows[i:i + batch])
    return done


def refresh_senses(store, embedder: Embedder, batch: int = 64) -> int:
    """Re-embed every sense-capture row so stores that predate the modality
    prefix pick it up (existing rows HAVE vectors, so `backfill` skips them).
    Idempotent — re-running rewrites the same chunks."""
    rows = store.conn.execute(
        """SELECT id, name, gist, detail, source FROM memory
           WHERE source LIKE 'senses:%' AND superseded_time IS NULL""",
    ).fetchall()
    done = 0
    for i in range(0, len(rows), batch):
        _write_chunks(store, embedder, rows[i:i + batch])
        done += len(rows[i:i + batch])
    return done


def cosines_for(store, embedder: Embedder, query: str,
                ids: list[int]) -> dict[int, float]:
    """Exact query↔memory cosine (best chunk) for SPECIFIC rows. The blend
    needs a real similarity for EVERY candidate, not just the ones that made
    the nearest-neighbor shortlist: a keyword-anchored row outside the
    shortlist used to read as cosine 0.0 — no vector term in its relevance,
    rankings that changed with `limit` (the shortlist scales with it), and
    the abstention gate mistaking "not shortlisted" for "nothing similar"
    (false-abstaining on rank-1 hits whose true cosine cleared the gate)."""
    if not ids:
        return {}
    qvec = embedder.embed([query])[0]
    ph = ",".join("?" * len(ids))
    return _best_by_memory(qvec, store.conn.execute(
        f"SELECT e.memory_id, e.vector FROM embedding e "
        f"WHERE e.model = ? AND e.memory_id IN ({ph})",
        (embedder.name, *ids)))


def similar(store, embedder: Embedder, query: str, *, limit: int = 25,
            include_superseded: bool = False) -> list[tuple[int, float]]:
    """(memory_id, cosine) nearest the query, best first — a memory scores
    as its best-matching chunk."""
    qvec = embedder.embed([query])[0]
    where = "" if include_superseded else \
        "JOIN memory m ON m.id = e.memory_id AND m.superseded_time IS NULL"
    best = _best_by_memory(qvec, store.conn.execute(
        f"SELECT e.memory_id, e.vector FROM embedding e {where} WHERE e.model = ?",
        (embedder.name,)))
    scored = sorted(best.items(), key=lambda t: t[1], reverse=True)
    return scored[:limit]
