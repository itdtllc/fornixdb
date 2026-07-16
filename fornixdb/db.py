"""SQLite schema and connection management for the fornixdb hot spine.

Schema notes:
- Bi-temporal: event_time (when it happened) vs recorded_time (when stored).
- Supersession is a tombstone, never a delete: superseded rows stay queryable.
- memory_fts is an external-content FTS5 index over gist+detail, kept in
  sync by triggers, providing the subject-recall axis until vectors land (P2).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 13  # v2: FTS gains name; chunked embeddings. v3: last_reinforced.
                    # v13: proactive_suppressed_at + justifying stats — a memory
                    #     chronically pushed (>=N) but never referenced is excluded
                    #     from the L3/L4/L5 PUSH channels (never from explicit
                    #     recall/show/timeline). The cosine floor provably can't
                    #     separate this population (useful vs noise cosines overlap);
                    #     per-memory push OUTCOME history can. Redeemable (see
                    #     core.clear_proactive_suppression).
                    # v11: prospective (reminders — new table only, IF NOT EXISTS)
                    # v12: prospective urgent/deliveries/last_delivery (nagging)
                    # v5: writer. v6: helpful_count/last_helpful (usefulness).
                    # v4: recall_feedback (negative feedback, new table only)
                    # v5: memory.writer (shared-tier writer provenance, B3)
                    # v7: surfaced_count/last_surfaced — proactive-PUSH impressions,
                    #     kept distinct from recall_count (explicit PULL) so the
                    #     usefulness loop can tell "kept getting pushed" from "used"
                    # v8: referenced_count/last_referenced — a PUSH that was actually
                    #     used in reasoning (cited downstream, honest transcript
                    #     signal). A pushed memory sits in context and is used without
                    #     a PULL, so recall_count can't see it; this closes that loop.
                    # v9: 'distinct' link relation — a reviewed pair the dream keeps
                    #     re-proposing (contradiction/merge/resolution) accepted as
                    #     legitimately distinct (the pair-level reality-ok/noise-ok).
                    #     memory_link's CHECK bakes the relation list, so old stores
                    #     get the table rebuilt in place.
                    # v10: modal_embedding (senses latent lane, new table only) —
                    #      a perceptual memory's modality vector (image/audio/sensor
                    #      model), one row per model, beside its ordinary caption
                    #      embedding in `embedding` (the cross-modal text lane)

DEFAULT_DB_ENV = "FORNIXDB_DB"
# FornixDB-branded so a default store is never mistaken for a host AI's memory
# file (e.g. a "memory.db"). Existing stores are referenced by explicit path, so
# this only names NEWLY created default-path stores. See decision #356.
DEFAULT_DB_PATH = "~/.fornixdb/fornix.db"

KINDS = ("episodic", "semantic", "feedback", "reference")
# Native memory taxonomies (e.g. Claude Code's user|feedback|project|reference)
# don't map 1:1 to ours, so a model naturally reaches for a kind we don't have.
# Accept those names as aliases so a write never bounces on a vocabulary
# mismatch: "project"/"user" facts are standing knowledge -> semantic.
KIND_ALIASES = {"project": "semantic", "user": "semantic"}
RELATIONS = ("refines", "supersedes", "relates", "distinct")
TIERS = ("hot", "consolidated", "cold")

_SCHEMA = f"""
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory (
    id              INTEGER PRIMARY KEY,
    name            TEXT UNIQUE,
    kind            TEXT NOT NULL CHECK (kind IN {KINDS!r}),
    event_time      TEXT NOT NULL,
    event_time_end  TEXT,
    recorded_time   TEXT NOT NULL,
    session_id      TEXT,
    project         TEXT,
    gist            TEXT NOT NULL,
    detail          TEXT,
    salience        REAL NOT NULL DEFAULT 0.5,
    retention_tier  TEXT NOT NULL DEFAULT 'hot' CHECK (retention_tier IN {TIERS!r}),
    source          TEXT,
    source_ref      TEXT,
    writer          TEXT,  -- which agent wrote it; stamped on shared-tier rows
                           -- so the weakest model on a machine can't launder
                           -- memories into every other agent's trust (B3)
    last_recalled   TEXT,
    last_reinforced TEXT,  -- detail engagement only — the staleness anchor;
                           -- passive listing (last_recalled) must not clear it
    recall_count    INTEGER NOT NULL DEFAULT 0,
    helpful_count   INTEGER NOT NULL DEFAULT 0,  -- v6: explicit "this helped"
                           -- endorsements — a durable, query-independent
                           -- usefulness signal (counterpart to recall_feedback's
                           -- query-conditional "irrelevant"); feeds ranking
    last_helpful    TEXT,  -- when the memory was last marked helpful
    surfaced_count  INTEGER NOT NULL DEFAULT 0,  -- v7: times PUSHED unsolicited
                           -- (proactive L3 / rhythmic L4 injection). An
                           -- impression, NOT a use: high surfaced_count with low
                           -- recall_count/helpful_count = "kept getting pushed
                           -- but never used" — the implicit noise signal the
                           -- usefulness loop raises the relevance floor against
    last_surfaced   TEXT,  -- when the memory was last pushed proactively
    referenced_count INTEGER NOT NULL DEFAULT 0,  -- v8: PUSH impressions that were
                           -- actually USED — cited downstream in reasoning (the
                           -- honest usefulness-scan signal). A pushed memory is
                           -- already in context, so it's used WITHOUT a pull;
                           -- recall_count never sees it. Folded into effective_floor
                           -- as a use-credit so a proven-useful push isn't treated
                           -- as ignored noise. Materialized by `usefulness-scan
                           -- --apply` from session transcripts (absolute set).
    last_referenced TEXT,  -- when a push of this memory was last used downstream
    proactive_suppressed_at TEXT,  -- v13: set when a memory is PROACTIVE-SUPPRESSED
                           -- — chronically pushed (surfaced) but never referenced,
                           -- so it is excluded from the L3/L4/L5 PUSH channels. It
                           -- is NEVER hidden from explicit recall/show/timeline
                           -- (invariant). NULL = eligible to push. Redeemable:
                           -- show/mark_helpful/supersede/set-gist clear it (see
                           -- core.clear_proactive_suppression). The floor cosine
                           -- provably can't separate this population (useful vs
                           -- noise cosines overlap); push OUTCOME history can.
    suppressed_pushed     INTEGER,  -- v13: the push count that justified suppression
    suppressed_referenced INTEGER,  -- v13: the referenced count at suppression time
                           -- (0 by the rule) — kept so `suppress --list` can show
                           -- WHY without re-scanning transcripts
    superseded_by   INTEGER REFERENCES memory(id),
    superseded_time TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_event_time ON memory(event_time);
CREATE INDEX IF NOT EXISTS idx_memory_project    ON memory(project);
CREATE INDEX IF NOT EXISTS idx_memory_kind       ON memory(kind);

CREATE TABLE IF NOT EXISTS topic (
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_topic (
    memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    topic_id  INTEGER NOT NULL REFERENCES topic(id)  ON DELETE CASCADE,
    PRIMARY KEY (memory_id, topic_id)
);

CREATE TABLE IF NOT EXISTS memory_link (
    memory_id  INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    related_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    relation   TEXT NOT NULL CHECK (relation IN {RELATIONS!r}),
    PRIMARY KEY (memory_id, related_id, relation)
);

CREATE TABLE IF NOT EXISTS session (
    id         TEXT PRIMARY KEY,
    project    TEXT,
    started    TEXT,
    ended      TEXT,
    source     TEXT,
    source_ref TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    name, gist, detail,
    content='memory', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, name, gist, detail)
    VALUES (new.id, coalesce(new.name, ''), new.gist, coalesce(new.detail, ''));
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, name, gist, detail)
    VALUES ('delete', old.id, coalesce(old.name, ''), old.gist, coalesce(old.detail, ''));
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE OF name, gist, detail ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, name, gist, detail)
    VALUES ('delete', old.id, coalesce(old.name, ''), old.gist, coalesce(old.detail, ''));
    INSERT INTO memory_fts(rowid, name, gist, detail)
    VALUES (new.id, coalesce(new.name, ''), new.gist, coalesce(new.detail, ''));
END;

-- v4: explicit negative feedback (mark_irrelevant), query-conditional. A row
-- says "memory X was irrelevant to query Q" — recall downweights X only for
-- queries similar to Q, never globally. Retraction is a tombstone, never a
-- delete. The query's embedding (when a model is available at mark time) makes
-- similarity associative. (This is the EXPLICIT negative path; the implicit
-- "pushed but never used" signal lives separately in memory.surfaced_count and
-- only nudges the proactive PUSH floor, never this query-conditional penalty.)
CREATE TABLE IF NOT EXISTS recall_feedback (
    id        INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    query     TEXT NOT NULL,
    model     TEXT,           -- embedder that produced vector (NULL = keyword-only)
    vector    BLOB,           -- query embedding at mark time
    created   TEXT NOT NULL,
    retracted TEXT,           -- tombstone: feedback is never deleted
    UNIQUE (memory_id, query)
);

-- P2: optional vector embeddings (float32 little-endian BLOB). The store is
-- fully functional without any rows here; associative recall is an upgrade.
-- chunk 0 = name+gist; chunks 1..n = detail windows, so paraphrases of facts
-- buried deep in detail are findable (recall scores a memory by its best chunk).
CREATE TABLE IF NOT EXISTS embedding (
    memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    chunk     INTEGER NOT NULL DEFAULT 0,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (memory_id, chunk)
);

-- v11: prospective memory — "remembering to remember" (Einstein & McDaniel).
-- A reminder is an ordinary memory row (kind=episodic, event_time = when it's
-- due, so the timeline shows it) plus one row here carrying the delivery
-- state. delivered_at is a tombstone, not a delete: once surfaced, the row
-- becomes a normal episodic memory of the intention ("I reminded him") and
-- decays like everything else. Cancelling a reminder = forget_memory on the
-- memory row (CASCADE cleans this side-table).
-- v12: urgent reminders NAG. urgent=0 (default): delivered_at is set the
-- moment the reminder is surfaced — fire exactly once, done. urgent=1:
-- delivery only increments deliveries/last_delivery; delivered_at means
-- ACKNOWLEDGED (the owner responded after a delivery, observed by the host
-- shell — never inferred by a model), and until then due() re-offers it every
-- nag interval up to the attempt cap. deliveries/last_delivery double as the
-- escalation state ("third time now:") and the give-up boundary.
CREATE TABLE IF NOT EXISTS prospective (
    memory_id     INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
    due           TEXT NOT NULL,
    delivered_at  TEXT,
    urgent        INTEGER NOT NULL DEFAULT 0,
    deliveries    INTEGER NOT NULL DEFAULT 0,
    last_delivery TEXT
);
CREATE INDEX IF NOT EXISTS idx_prospective_due ON prospective(due);

-- v10: the senses' latent lane. A perceptual memory keeps its caption gist
-- embedded in `embedding` like every other memory (the cross-modal text
-- lane), and its modality vector (image/audio/sensor model) here. Separate
-- table so the hot text path is untouched; one row per model, and similarity
-- is only ever scored between rows of the SAME model — spaces never mix.
CREATE TABLE IF NOT EXISTS modal_embedding (
    memory_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    PRIMARY KEY (memory_id, model)
);

-- Lower-friction capture (§15.2 #1): a cheap staging scratchpad. `jot` drops a
-- raw thought here mid-work (no title/kind/embedding cost); at a checkpoint the
-- AI reviews these and promotes the keepers into real memories, discarding the
-- rest. NOT memories — never recalled — until promoted; this table stays small.
CREATE TABLE IF NOT EXISTS candidate (
    id         INTEGER PRIMARY KEY,
    note       TEXT NOT NULL,
    session_id TEXT,
    created    TEXT NOT NULL,
    promoted   TEXT             -- set when turned into a memory (kept as a trace)
);
"""


def _migrate(conn: sqlite3.Connection) -> bool:
    """v1 → v2, before the schema script runs (IF NOT EXISTS would keep the
    old shapes). Returns True if the FTS index must be rebuilt.

    Runs INSIDE the caller's BEGIN IMMEDIATE transaction (see _setup): every
    ALTER here is non-idempotent, and the column probes below decide what to
    run — probing must happen under the write lock or two processes opening
    the store after a version bump both see the stale shape and both ALTER
    (the loser dies with 'duplicate column name'). No executescript in here:
    executescript implicitly COMMITs the open transaction."""
    try:
        fts_cols = [r[1] for r in conn.execute("PRAGMA table_info(memory_fts)")]
    except sqlite3.OperationalError:
        fts_cols = []
    rebuild = bool(fts_cols) and "name" not in fts_cols
    if rebuild:
        for stmt in (
                "DROP TRIGGER IF EXISTS memory_ai",
                "DROP TRIGGER IF EXISTS memory_ad",
                "DROP TRIGGER IF EXISTS memory_au",
                "DROP TABLE memory_fts"):
            conn.execute(stmt)
        # Persist the decision: the rebuild itself runs after the schema
        # script recreates memory_fts, outside this transaction. A second
        # process connecting in that window re-probes, sees no memory_fts,
        # and would otherwise conclude nothing needs rebuilding — the flag
        # makes whoever gets there next finish the job.
        conn.execute("INSERT OR REPLACE INTO meta(key, value) "
                     "VALUES ('fts_rebuild_pending', '1')")
    emb_cols = [r[1] for r in conn.execute("PRAGMA table_info(embedding)")]
    if emb_cols and "chunk" not in emb_cols:
        # derived data — drop and re-run `embed` (cheap) rather than migrate rows
        conn.execute("DROP TABLE embedding")
    mem_cols = [r[1] for r in conn.execute("PRAGMA table_info(memory)")]
    if mem_cols and "last_reinforced" not in mem_cols:  # v3
        conn.execute("ALTER TABLE memory ADD COLUMN last_reinforced TEXT")
    if mem_cols and "writer" not in mem_cols:  # v5
        conn.execute("ALTER TABLE memory ADD COLUMN writer TEXT")
    if mem_cols and "helpful_count" not in mem_cols:  # v6
        conn.execute("ALTER TABLE memory ADD COLUMN helpful_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE memory ADD COLUMN last_helpful TEXT")
    if mem_cols and "surfaced_count" not in mem_cols:  # v7
        conn.execute("ALTER TABLE memory ADD COLUMN surfaced_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE memory ADD COLUMN last_surfaced TEXT")
    if mem_cols and "referenced_count" not in mem_cols:  # v8
        conn.execute("ALTER TABLE memory ADD COLUMN referenced_count INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE memory ADD COLUMN last_referenced TEXT")
    if mem_cols and "proactive_suppressed_at" not in mem_cols:  # v13
        # nullable, no default: an existing store's rows are all push-eligible
        # (NULL) until the next suppress scan classifies them — no data rewrite
        conn.execute("ALTER TABLE memory ADD COLUMN proactive_suppressed_at TEXT")
        conn.execute("ALTER TABLE memory ADD COLUMN suppressed_pushed INTEGER")
        conn.execute("ALTER TABLE memory ADD COLUMN suppressed_referenced INTEGER")
    # v12: a v11 prospective table (0.8.5) predates the nag columns; SQLite
    # adds them in place, defaults matching the schema (all existing reminders
    # are non-urgent — exactly their 0.8.5 behavior)
    pros_cols = [r[1] for r in conn.execute("PRAGMA table_info(prospective)")]
    if pros_cols and "urgent" not in pros_cols:
        conn.execute("ALTER TABLE prospective ADD COLUMN urgent INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE prospective ADD COLUMN deliveries INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE prospective ADD COLUMN last_delivery TEXT")
    # v9: memory_link's CHECK bakes the relation list; a pre-'distinct' store
    # needs the table rebuilt in place (SQLite can't ALTER a CHECK). Rows are
    # copied verbatim; the child-table rebuild is safe under foreign_keys=ON.
    link_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_link'"
    ).fetchone()
    if link_row and link_row[0] and "'distinct'" not in link_row[0]:
        for stmt in (
                f"""CREATE TABLE memory_link_v9 (
                    memory_id  INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
                    related_id INTEGER NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
                    relation   TEXT NOT NULL CHECK (relation IN {RELATIONS!r}),
                    PRIMARY KEY (memory_id, related_id, relation)
                )""",
                "INSERT INTO memory_link_v9 SELECT * FROM memory_link",
                "DROP TABLE memory_link",
                "ALTER TABLE memory_link_v9 RENAME TO memory_link"):
            conn.execute(stmt)
    return rebuild


def default_db_path() -> Path:
    return Path(os.environ.get(DEFAULT_DB_ENV, "") or DEFAULT_DB_PATH).expanduser()


REGISTRY_ENV = "FORNIXDB_REGISTRY"
# FornixDB-branded so it's never confused with a host AI's own files (owner rule:
# every default on-disk name begins with "fornix"). Legacy "stores.json" is
# auto-migrated by _migrate_legacy_registry(). See decision 2026-06-21.
DEFAULT_REGISTRY = "~/.fornixdb/fornix-stores.json"

SHARED_ENV = "FORNIXDB_SHARED_DB"          # canonical here; multistore re-exports
DEFAULT_SHARED_PATH = "~/.fornixdb/fornix-shared.db"  # FornixDB-branded; see #357

# Install-time machine cap default (owner decision 2026-06-12, ceiling raised
# to 2 GB 2026-06-16): a fresh machine gets a cap of 20% of free disk, at most
# 2 GB — never silently: it is marked as a default and announced until the
# owner reviews it. The 2 GB is a CEILING; the 20%-of-free-disk rule still wins
# on constrained devices, so small machines stay protected.
DEFAULT_MACHINE_CAP_MAX_MB = 2000
DEFAULT_MACHINE_CAP_DISK_FRACTION = 0.20


def shared_db_path() -> Path:
    return Path(os.environ.get(SHARED_ENV, "") or DEFAULT_SHARED_PATH).expanduser()


def _maybe_default_machine_budget(conn: sqlite3.Connection, path: Path) -> None:
    """First creation of the SHARED tier = the machine-level install moment:
    default the machine-wide cap to min(20% of free disk, 2 GB) and mark it
    `machine_budget_defaulted` so every surface tells the owner to review it
    (the marker clears when they set or clear the cap themselves)."""
    import shutil
    import sys
    try:
        if path.resolve() != shared_db_path().resolve():
            return
        if conn.execute("SELECT 1 FROM meta WHERE key = 'machine_budget_mb'"
                        ).fetchone():
            return
        free_mb = shutil.disk_usage(path.parent).free / 1e6
        mb = min(DEFAULT_MACHINE_CAP_MAX_MB,
                 int(free_mb * DEFAULT_MACHINE_CAP_DISK_FRACTION))
        if mb <= 0:
            return
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('machine_budget_mb', ?)",
                     (str(mb),))
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('machine_budget_defaulted', '1')")
        conn.commit()
        print(f"FornixDB: machine-wide memory cap defaulted to {mb} MB "
              f"(20% of free disk, max {DEFAULT_MACHINE_CAP_MAX_MB} MB). Review it: "
              "`fornixdb config machine_budget_mb <MB> --shared` (or 'off').",
              file=sys.stderr)
    except Exception:
        pass  # a default must never block creating the store


def _migrate_legacy_registry(reg: Path) -> None:
    """One-time rename of the pre-2026-06-21 'stores.json' registry to the
    FornixDB-branded 'fornix-stores.json'. Acts only when reg is the default
    name and absent while a legacy 'stores.json' sits beside it — never
    overwrites an existing registry, never touches a custom ($FORNIXDB_REGISTRY)
    name. Best-effort: a registry migration must never block opening a store."""
    try:
        if reg.name != "fornix-stores.json" or reg.exists():
            return
        legacy = reg.with_name("stores.json")
        if legacy.exists():
            legacy.rename(reg)
    except Exception:
        pass


def registry_path() -> Path | None:
    raw = os.environ.get(REGISTRY_ENV, "") or DEFAULT_REGISTRY
    if raw == "off":
        return None
    reg = Path(raw).expanduser()
    _migrate_legacy_registry(reg)
    return reg


def _register_store(path: Path) -> None:
    """Record this store in the machine-level registry so 'how much space is
    FornixDB taking OVERALL' is answerable from any one AI (each agent's tools
    otherwise see only their own store). Paths only — no content. Skipped for
    in-memory and temp-dir stores (tests, smokes) and when $FORNIXDB_REGISTRY
    is 'off'. Best-effort: registration must never block opening a store."""
    import json
    import tempfile
    reg = registry_path()
    if reg is None or path.name == ":memory:":
        return
    try:
        p = str(path.resolve())
        if p.startswith(str(Path(tempfile.gettempdir()).resolve())):
            return
        new_reg_dir = not reg.parent.exists()
        reg.parent.mkdir(parents=True, exist_ok=True)
        if new_reg_dir:
            _restrict_to_owner_path(reg.parent, is_dir=True)
        stores = []
        if reg.exists():
            try:
                stores = json.loads(reg.read_text() or "[]")
            except ValueError:
                stores = []  # unreadable registry: start over — membership
                             # self-heals because every connect re-registers
        if p not in stores:
            stores.append(p)
            # Write-to-temp + atomic rename: two agents connecting at once can
            # still lose the OTHER's brand-new entry (last writer wins), which
            # the next connect repairs — but no reader ever sees a torn file,
            # which does NOT self-repair (json.loads fails → registry dark).
            tmp = reg.with_name(f"{reg.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(sorted(stores), indent=1))
            _restrict_to_owner_path(tmp)  # paths reveal which AIs keep stores where
            os.replace(tmp, reg)
    except Exception:
        pass


def append_log_line(path: str | os.PathLike, payload: str) -> None:
    """Append `payload` (one or more complete lines) to a side-log with a
    SINGLE O_APPEND write. Hook processes, host threads, and CLI runs all
    append to the same floor/field/suppress logs; a buffered text-mode
    f.write can split a large record across syscalls, tearing a line through
    the middle of another writer's. One os.write under O_APPEND keeps each
    payload contiguous. Created files are owner-only, like the store itself."""
    data = payload.encode("utf-8")
    if not data.endswith(b"\n"):
        data += b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _restrict_to_owner_path(path: Path, *, is_dir: bool = False) -> None:
    """Make one path readable only by the current user, cross-platform.

    POSIX: chmod 700 (dir) / 600 (file). Windows: os.chmod cannot express this
    (it only toggles the read-only bit, and 0o600 keeps write set — a no-op for
    access control), so reset ACL inheritance and grant the current user alone
    via the built-in `icacls` — no third-party dependency. Best-effort by
    design: a failure to harden must never block opening a store. Callers apply
    it only to paths the process just created, so a deliberate loosening by the
    owner on an existing file is respected."""
    try:
        if os.name == "nt":
            import getpass
            import subprocess
            user = os.environ.get("USERNAME") or getpass.getuser()
            # /inheritance:r drops inherited ACEs; /grant:r replaces the user's
            # entry with Full. (OI)(CI) lets a new dir's children inherit it.
            spec = f"{user}:(OI)(CI)(F)" if is_dir else f"{user}:(F)"
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", spec],
                check=False, capture_output=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.chmod(path, 0o700 if is_dir else 0o600)
    except Exception:
        pass


def _restrict_to_owner(path: Path) -> None:
    """Make a store file and its WAL/SHM siblings owner-only. Memories are
    personal data; a default 644 on a multi-user POSIX box leaves them readable
    by every other account (macOS group `staff` is all regular users), and on
    Windows a file under a world-traversable directory inherits broad ACEs.
    Applied only to stores THIS process creates."""
    for f in (path, path.with_name(path.name + "-wal"),
              path.with_name(path.name + "-shm")):
        if f.exists():
            _restrict_to_owner_path(f)


DEFAULT_BUSY_TIMEOUT_MS = 5000


def _stored_version(conn: sqlite3.Connection) -> int | None:
    """The store's recorded schema_version, or None (fresh store, pre-meta
    store, or anything unreadable — all of which mean 'run setup')."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else None
    except (sqlite3.OperationalError, ValueError):
        return None


def _configured_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    """`config busy_timeout_ms <ms>` — how long a connection waits on a locked
    store before erroring. The default suits interactive hosts; a store that
    coexists with long admin passes (dream, shrink/VACUUM) can raise it."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'busy_timeout_ms'").fetchone()
        if row:
            ms = int(str(row[0]).strip())
            if ms > 0:
                return ms
    except (sqlite3.OperationalError, ValueError):
        pass
    return DEFAULT_BUSY_TIMEOUT_MS


def _setup(conn: sqlite3.Connection) -> None:
    """Migrate and create the schema, serialized against concurrent openers.

    The stored schema_version is probed first WITHOUT any lock: a store that is
    already current — every connect after the first — skips migration entirely
    and the schema script below is pure no-ops (IF NOT EXISTS on existing
    objects writes nothing), so routine connects take no write lock at all and
    can open even while another process holds the store for a long admin op.

    On a version change, BEGIN IMMEDIATE serializes the migrators: exactly one
    process runs the ALTERs; the others block on the lock, re-probe inside
    _migrate, and find nothing left to do. Consequence of the version gate:
    ANY schema change must bump SCHEMA_VERSION, or existing stores never run
    the new statements."""
    if _stored_version(conn) != SCHEMA_VERSION:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _migrate(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    conn.executescript(_SCHEMA)
    try:
        pending = conn.execute(
            "SELECT 1 FROM meta WHERE key = 'fts_rebuild_pending'").fetchone()
        if pending:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
            conn.execute("DELETE FROM meta WHERE key = 'fts_rebuild_pending'")
            conn.commit()
    except sqlite3.OperationalError:
        # rebuild lost a lock race — the flag stays set for the next opener;
        # roll back so a half-done rebuild doesn't sit on an open transaction
        conn.rollback()
    if _stored_version(conn) != SCHEMA_VERSION:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()


def connect(db_path: str | os.PathLike | None = None, *,
            check_same_thread: bool = True) -> sqlite3.Connection:
    """Open (creating if needed) an fornixdb store and return the connection.

    check_same_thread=False is for callers that manage their own per-thread
    confinement (MemoryStore keeps one connection per thread and only needs
    the flag off so close() can run from a different thread); everyone else
    should keep the default and let sqlite3 catch cross-thread sharing."""
    path = Path(db_path).expanduser() if db_path else default_db_path()
    if not path.parent.exists():  # new dirs owner-only; existing dirs untouched
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _restrict_to_owner_path(path.parent, is_dir=True)  # Windows ACL too
    creating = path.name != ":memory:" and not path.exists()
    _register_store(path)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Wait for a lock instead of erroring instantly. WAL lets readers and a
    # writer coexist, but writers still serialize — and a BEGIN IMMEDIATE
    # (write_txn, migration) queues here rather than failing. Per-connection,
    # so set on every open; configurable per store via `config busy_timeout_ms`
    # (read after setup — the meta table may not exist yet on a fresh store).
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
    _setup(conn)
    ms = _configured_busy_timeout_ms(conn)
    if ms != DEFAULT_BUSY_TIMEOUT_MS:
        conn.execute(f"PRAGMA busy_timeout = {ms}")
    if creating:
        _restrict_to_owner(path)
        _announce_new_store(path)
        _maybe_default_machine_budget(conn, path)
    return conn


def _announce_new_store(path: Path) -> None:
    """Creating a store is loud, never silent: the default-path fallback is
    convenient for new users, but it makes a MISDIRECTED write (env var not
    set, typo'd --db) invisible — the row lands in a fresh store nobody meant
    to create. Temp-dir stores (tests, smokes) stay quiet."""
    import sys
    import tempfile
    try:
        if str(path.resolve()).startswith(
                str(Path(tempfile.gettempdir()).resolve())):
            return
        print(f"FornixDB: created NEW store at {path} — if you expected an "
              "existing store, check $FORNIXDB_DB / --db.", file=sys.stderr)
    except Exception:
        pass
