"""Import a directory of frontmatter Markdown memory files as semantic rows.

Source format (as used by Claude Code's auto-memory and similar systems):

    ---
    name: some-kebab-slug
    description: one-line summary
    metadata:
      type: user | feedback | project | reference
    ---
    body text, possibly containing [[wikilinks]] to other memories' names.

Mapping: description -> gist, body -> detail, name -> name, file mtime ->
event/recorded time, [[wikilinks]] -> 'relates' links (resolved by name after
all files are imported). Index files (MEMORY.md) are skipped. The source
directory is never modified.

Usage:
    python3 -m fornixdb.adapters.markdown_import <dir> [--db PATH] [--project NAME]
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from ..core import MemoryStore

WIKILINK = re.compile(r"\[\[([^\]\n]+)\]\]")

TYPE_TO_KIND = {
    "feedback": "feedback",
    "reference": "reference",
    "project": "semantic",
    "user": "semantic",
}
TYPE_SALIENCE = {"feedback": 0.7, "project": 0.6, "user": 0.6, "reference": 0.5}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal frontmatter parser (stdlib-only; full YAML not required for
    the simple key/value + one-level-nested shape these files use)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header, body = text[3:end], text[end + 4:]
    meta: dict = {}
    stack = meta
    for line in header.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"').strip("'")
        if indent == 0:
            if value:
                meta[key] = value
                stack = meta
            else:
                meta[key] = {}
                stack = meta[key]
        else:
            stack[key] = value
    return meta, body.lstrip("\n")


def import_directory(
    store: MemoryStore,
    directory: str | Path,
    *,
    project: str | None = None,
    source: str = "markdown-import",
    skip_names: tuple[str, ...] = ("MEMORY.md", "README.md"),
) -> dict:
    directory = Path(directory).expanduser()
    imported, skipped, links_made = [], 0, 0
    name_to_id: dict[str, int] = {}
    pending_links: list[tuple[int, str]] = []

    for path in sorted(directory.glob("*.md")):
        if path.name in skip_names:
            skipped += 1
            continue
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = meta.get("name") or path.stem
        if store.conn.execute("SELECT 1 FROM memory WHERE name = ?", (name,)).fetchone():
            skipped += 1  # already imported — idempotent re-runs
            continue
        mtype = (meta.get("metadata") or {}).get("type", "") if isinstance(meta.get("metadata"), dict) else ""
        kind = TYPE_TO_KIND.get(mtype, "semantic")
        gist = meta.get("description") or body.strip().splitlines()[0][:200] if body.strip() else name
        mtime = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)
        mem_id = store.store(
            gist,
            body.strip() or None,
            kind=kind,
            name=name,
            topics=[mtype] if mtype else [],
            project=project,
            event_time=mtime.isoformat(),
            recorded_time=mtime.isoformat(),
            salience=TYPE_SALIENCE.get(mtype, 0.5),
            source=source,
            source_ref=str(path),
        )
        name_to_id[name] = mem_id
        imported.append(name)
        for target in WIKILINK.findall(body):
            pending_links.append((mem_id, target.strip()))

    # second pass: resolve wikilinks among everything now in the store
    for mem_id, target in pending_links:
        row = store.conn.execute("SELECT id FROM memory WHERE name = ?", (target,)).fetchone()
        if row and row["id"] != mem_id:
            store.link(mem_id, row["id"], "relates")
            links_made += 1

    return {"imported": len(imported), "skipped": skipped, "links": links_made,
            "names": imported}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory")
    ap.add_argument("--db")
    ap.add_argument("--project")
    args = ap.parse_args(argv)
    result = import_directory(MemoryStore(db_path=args.db), args.directory,
                              project=args.project)
    print(f"imported {result['imported']}, skipped {result['skipped']}, "
          f"links {result['links']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
