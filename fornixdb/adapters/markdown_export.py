"""Export memories to human-readable Markdown (P4).

Two shapes, same selection:

* ``export_directory`` — the inverse of ``markdown_import``: one memory -> one
  ``<name>.md`` file with frontmatter the frontmatter-importer understands
  (``name`` / ``description`` -> gist / ``metadata.type`` -> kind), the detail as
  the body, and a ``## Related`` footer of ``[[wikilinks]]``. So a store can be
  committed to git as readable files and re-imported (``import-markdown
  --frontmatter``) to reconstruct it. An index file (one line per memory) is
  written too; the importer skips it on a round-trip. The index is named
  ``FornixDB.md`` by default — deliberately NOT ``MEMORY.md``, so a person never
  confuses a FornixDB export for Claude Code's own memory index; override with
  ``index_name``.

* ``export_document`` — ONE consolidated, human-readable document (sections per
  memory, no machine frontmatter) for a person who maintains notes by hand. Not
  meant to round-trip back into the store.

Both honor the same selection filters: ``project`` / ``kind`` /
``include_superseded`` as before, plus subject + time selection that reuses the
recall machinery — ``query`` ("export the token-usage work"), ``when``
("yesterday", "last week", parsed by ``timeparse.parse_when``), and explicit
``since`` / ``until`` ISO bounds. Time filters match on ``event_time`` (when the
work happened), the same axis ``recall_timeline`` uses.

Round-trip notes: the frontmatter importer maps ``metadata.type`` back to kind
(feedback/reference exact; everything else -> semantic), so an ``episodic`` row
re-imports as ``semantic``. ``relates`` is the only link relation a wikilink
re-creates, so ``refines``/``supersedes`` edges flatten to ``relates`` on
re-import (the footer still records the original relation for the reader).

Usage:
    python3 -m fornixdb.adapters.markdown_export <out-dir> [--db PATH]
        [--project NAME] [--kind KIND] [--include-superseded]
        [--index-name NAME] [--query TEXT] [--when PHRASE]
        [--since ISO] [--until ISO] [--document [FILE]]
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
# the low/high sentinels keep timeline's half-open comparison total when only
# one bound is given
_MIN_TS, _MAX_TS = "0001-01-01T00:00:00", "9999-12-31T00:00:00"


def _safe_filename(name: str) -> str:
    return _UNSAFE.sub("__", name).strip("_") or "memory"


def _quote(value: str) -> str:
    """One-line, double-quoted scalar for the minimal frontmatter parser."""
    return '"' + value.replace("\n", " ").replace('"', "'") + '"'


# ----------------------------------------------------------------- selection

def _window(when: str | None, since: str | None, until: str | None
            ) -> tuple[str | None, str | None]:
    """Resolve (start, end) ISO bounds from a phrase and/or explicit dates.
    Explicit since/until override the phrase. ``parse_when`` may raise
    ValueError on an unreadable phrase — that propagates to the caller."""
    start = end = None
    if when:
        from ..timeparse import parse_when
        s, e = parse_when(when)
        start, end = s.isoformat(), e.isoformat()
    if since:
        start = since
    if until:
        end = until
    return start, end


def _select_ids(store: MemoryStore, *, query: str | None, start: str | None,
                end: str | None, kind: str | None, project: str | None,
                include_superseded: bool) -> list[int] | None:
    """IDs matching the subject/time filter, or None when neither is set (the
    caller then exports the whole store — the original behavior). Subject and
    time reuse the same recall/timeline retrieval the recall tools use, so
    "export the X work" and "export yesterday" select exactly what those tools
    would surface."""
    if query:
        rows = store.recall(query, limit=10000, kind=kind, project=project,
                            since=start, until=end,
                            include_superseded=include_superseded)
        return [r["id"] for r in rows]
    if start or end:
        rows = store.timeline(start or _MIN_TS, end or _MAX_TS,
                              kind=kind, project=project, limit=10000)
        if not include_superseded:
            rows = [r for r in rows if not r.get("superseded_time")]
        return [r["id"] for r in rows]
    return None


def _fetch_rows(store: MemoryStore, *, project: str | None, kind: str | None,
                include_superseded: bool, query: str | None, when: str | None,
                since: str | None, until: str | None) -> list:
    """The shared row set for both export shapes, ordered oldest-recorded
    first. Honors every filter."""
    start, end = _window(when, since, until)
    id_filter = _select_ids(store, query=query, start=start, end=end, kind=kind,
                            project=project, include_superseded=include_superseded)
    if id_filter is not None and not id_filter:
        return []                       # filter matched nothing — export nothing

    where, params = [], []
    if not include_superseded:
        where.append("superseded_time IS NULL")
    if project:
        where.append("project = ?"); params.append(project)
    if kind:
        where.append("kind = ?"); params.append(kind)
    if id_filter:
        where.append(f"id IN ({','.join('?' * len(id_filter))})")
        params += id_filter
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return store.conn.execute(
        f"SELECT * FROM memory {clause} ORDER BY recorded_time, id", params
    ).fetchall()


def _topics_and_links(store: MemoryStore, mem_id: int) -> tuple[list[str], list]:
    topics = [r["name"] for r in store.conn.execute(
        "SELECT t.name FROM topic t JOIN memory_topic mt ON mt.topic_id = t.id "
        "WHERE mt.memory_id = ? ORDER BY t.name", (mem_id,))]
    links = [
        (lr["relation"], lr["nm"] or f"#{lr['related_id']}")
        for lr in store.conn.execute(
            "SELECT ml.relation, ml.related_id, m2.name AS nm "
            "FROM memory_link ml JOIN memory m2 ON m2.id = ml.related_id "
            "WHERE ml.memory_id = ? ORDER BY ml.relation, ml.related_id",
            (mem_id,))]
    return topics, links


# --------------------------------------------------------- directory export

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
    index_name: str = "FornixDB.md",
    query: str | None = None,
    when: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = _fetch_rows(store, project=project, kind=kind,
                       include_superseded=include_superseded, query=query,
                       when=when, since=since, until=until)

    exported, used_names = [], set()
    for row in rows:
        topics, links = _topics_and_links(store, row["id"])
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
        (out / index_name).write_text("\n".join(index) + "\n", encoding="utf-8")

    return {"exported": len(exported), "dir": str(out), "index": write_index,
            "index_name": index_name if write_index else None}


# ---------------------------------------------------------- document export

def _filter_caption(project, kind, query, when, since, until) -> str | None:
    bits = []
    if query:
        bits.append(f'subject "{query}"')
    if when:
        bits.append(f'time "{when}"')
    if since or until:
        bits.append(f"range {since or '…'} → {until or '…'}")
    if project:
        bits.append(f"project {project}")
    if kind:
        bits.append(f"kind {kind}")
    return ", ".join(bits) if bits else None


def export_document(
    store: MemoryStore,
    out_file: str | Path,
    *,
    title: str | None = None,
    project: str | None = None,
    kind: str | None = None,
    include_superseded: bool = False,
    query: str | None = None,
    when: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> dict:
    """Write ONE consolidated, human-readable Markdown document: a title, an
    optional one-line note of what was selected, then a section per memory with
    a light italic ``kind · date`` line, the body prose, and a plain
    ``Related:`` line. No machine frontmatter — meant to be read and hand-edited,
    not re-imported."""
    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _fetch_rows(store, project=project, kind=kind,
                       include_superseded=include_superseded, query=query,
                       when=when, since=since, until=until)

    lines = [f"# {title or 'FornixDB Export'}", ""]
    caption = _filter_caption(project, kind, query, when, since, until)
    if caption:
        lines += [f"_Selected by {caption} — {len(rows)} memories._", ""]
    else:
        lines += [f"_{len(rows)} memories._", ""]

    for row in rows:
        topics, links = _topics_and_links(store, row["id"])
        date = (row["event_time"] or "")[:10]
        lines.append(f"## {row['name'] or f'memory #{row['id']}'}")
        meta = "*" + row["kind"] + (f" · {date}" if date else "") + "*"
        lines.append(meta)
        if topics:
            lines.append(f"_topics: {', '.join(topics)}_")
        lines.append("")
        body = (row["detail"] or row["gist"] or "").strip()
        if body:
            lines += [body, ""]
        if links:
            lines.append("Related: " + ", ".join(t for _, t in links))
            lines.append("")
        lines.append("---")
        lines.append("")

    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"exported": len(rows), "file": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--db")
    ap.add_argument("--project")
    ap.add_argument("--kind")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--index-name", default="FornixDB.md",
                    help="index filename (default FornixDB.md, never MEMORY.md)")
    ap.add_argument("--query", help="export only memories matching this subject")
    ap.add_argument("--when", help="time phrase, e.g. 'yesterday', 'last week'")
    ap.add_argument("--since", help="ISO lower bound on event_time")
    ap.add_argument("--until", help="ISO upper bound on event_time")
    ap.add_argument("--document", nargs="?", const="", metavar="FILE",
                    help="write ONE human-readable doc instead of a directory; "
                         "optional FILE path (else <out_dir>/FornixDB-export.md)")
    args = ap.parse_args(argv)

    store = MemoryStore(db_path=args.db)
    sel = dict(project=args.project, kind=args.kind,
               include_superseded=args.include_superseded, query=args.query,
               when=args.when, since=args.since, until=args.until)
    try:
        if args.document is not None:
            out_file = args.document or str(Path(args.out_dir) / "FornixDB-export.md")
            r = export_document(store, out_file, **sel)
            print(f"exported {r['exported']} memories to {r['file']}")
        else:
            r = export_directory(store, args.out_dir, index_name=args.index_name, **sel)
            print(f"exported {r['exported']} memories to {r['dir']}")
    except ValueError as e:  # e.g. an unreadable --when phrase
        print(f"couldn't export: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
