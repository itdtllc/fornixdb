"""Export memories to a directory of human-readable Markdown files (P4).

The inverse of ``markdown_import``: one memory -> one ``<name>.md`` file with
frontmatter the frontmatter-importer understands (``name`` / ``description`` ->
gist / ``metadata.type`` -> kind), the detail as the body, and a ``## Related``
footer of ``[[wikilinks]]``. So a store can be committed to git as readable
files and re-imported (``import-markdown --frontmatter``) to reconstruct it.

A ``MEMORY.md`` index (one line per memory) is written too, mirroring the
Claude-Code memory layout; the importer skips it on a round-trip.

Round-trip notes: the frontmatter importer maps ``metadata.type`` back to kind
(feedback/reference exact; everything else -> semantic), so an ``episodic`` row
re-imports as ``semantic``. ``relates`` is the only link relation a wikilink
re-creates, so ``refines``/``supersedes`` edges flatten to ``relates`` on
re-import (the footer still records the original relation for the reader).

Usage:
    python3 -m fornixdb.adapters.markdown_export <out-dir> [--db PATH]
        [--project NAME] [--kind KIND] [--include-superseded]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..core import MemoryStore

# kind -> the metadata.type the frontmatter importer reads back to that kind
KIND_TO_TYPE = {"feedback": "feedback", "reference": "reference",
                "semantic": "semantic", "episodic": "episodic"}
# Strip every character illegal in a Windows filename (`< > : " / \ | ? *` and
# control chars) plus `#` — not just the POSIX path separators. A name with a
# colon (e.g. "site phase 3: RE…") otherwise silently lands in an NTFS
# Alternate Data Stream on Windows: the visible file is 0 bytes and the content
# is lost to normal tooling.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f#]+')


def _safe_filename(name: str) -> str:
    return _UNSAFE.sub("__", name).strip("_") or "memory"


def _quote(value: str) -> str:
    """One-line, double-quoted scalar for the minimal frontmatter parser."""
    return '"' + value.replace("\n", " ").replace('"', "'") + '"'


def _render(row, topics, links) -> str:
    fm = [
        "---",
        f"name: {row['name']}" if row["name"] else f"name: mem-{row['id']}",
        f"description: {_quote(row['gist'])}",
        f"kind: {row['kind']}",
    ]
    if row["project"]:
        fm.append(f"project: {_quote(row['project'])}")
    if topics:
        fm.append("topics: [" + ", ".join(topics) + "]")
    fm.append(f"event_time: {row['event_time']}")
    fm.append(f"recorded_time: {row['recorded_time']}")
    fm.append(f"salience: {row['salience']}")
    if row["superseded_time"]:
        fm.append(f"superseded_time: {row['superseded_time']}")
    # metadata block LAST so its nested key doesn't capture later scalars
    fm += ["metadata:", f"  type: {KIND_TO_TYPE.get(row['kind'], 'semantic')}", "---", ""]

    body = (row["detail"] or "").strip()
    parts = ["\n".join(fm), body] if body else ["\n".join(fm)]
    if links:
        footer = ["", "## Related"] + [
            f"- {rel}: [[{target}]]" for rel, target in links
        ]
        parts.append("\n".join(footer))
    return "\n".join(parts).rstrip() + "\n"


def export_directory(
    store: MemoryStore,
    out_dir: str | Path,
    *,
    project: str | None = None,
    kind: str | None = None,
    include_superseded: bool = False,
    write_index: bool = True,
) -> dict:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    where, params = [], []
    if not include_superseded:
        where.append("superseded_time IS NULL")
    if project:
        where.append("project = ?"); params.append(project)
    if kind:
        where.append("kind = ?"); params.append(kind)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = store.conn.execute(
        f"SELECT * FROM memory {clause} ORDER BY recorded_time, id", params
    ).fetchall()

    exported, used_names = [], set()
    for row in rows:
        topics = [r["name"] for r in store.conn.execute(
            "SELECT t.name FROM topic t JOIN memory_topic mt ON mt.topic_id = t.id "
            "WHERE mt.memory_id = ? ORDER BY t.name", (row["id"],))]
        # outgoing links, referenced by the target's name (else #id)
        links = [
            (lr["relation"], lr["nm"] or f"#{lr['related_id']}")
            for lr in store.conn.execute(
                "SELECT ml.relation, ml.related_id, m2.name AS nm "
                "FROM memory_link ml JOIN memory m2 ON m2.id = ml.related_id "
                "WHERE ml.memory_id = ? ORDER BY ml.relation, ml.related_id",
                (row["id"],))]

        base = _safe_filename(row["name"] or f"mem-{row['id']}")
        fname = base
        n = 2
        while fname in used_names:  # distinct files if two names sanitize alike
            fname = f"{base}-{n}"; n += 1
        used_names.add(fname)

        (out / f"{fname}.md").write_text(_render(row, topics, links), encoding="utf-8")
        exported.append((fname, row["gist"]))

    if write_index:
        index = ["# Memory Index", ""] + [
            f"- [{gist}]({fname}.md)" for fname, gist in exported
        ]
        (out / "MEMORY.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    return {"exported": len(exported), "dir": str(out), "index": write_index}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--db")
    ap.add_argument("--project")
    ap.add_argument("--kind")
    ap.add_argument("--include-superseded", action="store_true")
    args = ap.parse_args(argv)
    result = export_directory(
        MemoryStore(db_path=args.db), args.out_dir,
        project=args.project, kind=args.kind,
        include_superseded=args.include_superseded)
    print(f"exported {result['exported']} memories to {result['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
