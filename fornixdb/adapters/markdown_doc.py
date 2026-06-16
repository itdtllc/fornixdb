"""Import an *arbitrary* Markdown document by chunking it into memories.

This is the free-form counterpart to ``markdown_import`` (which ingests a
directory of our own frontmatter *memory files*, one file -> one row). Here a
single document — an Obsidian note, a design doc, a README — is split along its
heading structure into many gist+detail memories, so a vault of prose becomes
recallable at section granularity.

Mapping:
- Each ATX heading (``#`` .. ``######``) starts a section. gist = heading text,
  detail = the section's own body (the text up to the next heading of any
  level). A heading with no body becomes a gist-only container memory.
- Hierarchy is preserved as ``refines`` links: a subsection refines its parent
  section (resolved via a level stack), giving progressive disclosure.
- Text before the first heading becomes a doc-root memory whose gist is the
  frontmatter ``title``/``description`` or the file name.
- ``[[wikilinks]]`` in a section body become ``relates`` links, resolved first
  against the sections of this document, then against existing memory names.
- Memory names are ``<doc-slug>#<heading-slug>`` so re-imports are idempotent
  (an already-present name is skipped) and intra-doc links can resolve.
- Fenced code blocks (``` / ~~~) are tracked so ``#`` comment lines inside them
  are never mistaken for headings.

The source file is never modified.

Usage:
    python3 -m fornixdb.adapters.markdown_doc <file-or-dir> [--db PATH] [--project NAME]
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from ..core import MemoryStore
from .markdown_import import (
    TYPE_SALIENCE,
    TYPE_TO_KIND,
    WIKILINK,
    parse_frontmatter,
)

# ATX heading at the start of a line: 0-3 leading spaces, 1-6 #, a space, text.
# Trailing #'s (a closed ATX heading) are stripped by the caller.
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*$")
FENCE = re.compile(r"^ {0,3}(```+|~~~+)")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase kebab slug; empty -> 'section'."""
    s = _SLUG_STRIP.sub("-", text.strip().lower()).strip("-")
    return s or "section"


def _split_sections(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Return (preamble_body, [(level, heading_text, body), ...]).

    Splits on ATX headings while ignoring any inside fenced code blocks.
    """
    lines = text.splitlines()
    preamble: list[str] = []
    sections: list[tuple[int, str, list[str]]] = []
    current: list[str] | None = None  # body lines of the open section
    in_fence = False
    fence_marker = ""

    for line in lines:
        fm = FENCE.match(line)
        if fm:
            marker = fm.group(1)[:3]  # ``` or ~~~
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            (current if current is not None else preamble).append(line)
            continue

        m = None if in_fence else HEADING.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).rstrip("#").strip()  # drop optional closing #'s
            current = []
            sections.append((level, heading, current))
        else:
            (current if current is not None else preamble).append(line)

    return (
        "\n".join(preamble).strip(),
        [(lvl, head, "\n".join(body).strip()) for lvl, head, body in sections],
    )


def import_document(
    store: MemoryStore,
    path: str | Path,
    *,
    project: str | None = None,
    source: str = "markdown-doc",
) -> dict:
    """Chunk one Markdown file into section memories. Idempotent by name."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    preamble, sections = _split_sections(body)

    mtype = (meta.get("metadata") or {}).get("type", "") if isinstance(meta.get("metadata"), dict) else ""
    kind = TYPE_TO_KIND.get(mtype, "semantic")
    salience = TYPE_SALIENCE.get(mtype, 0.5)
    mtime = datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat()
    doc_slug = slugify(meta.get("title") or path.stem)

    imported, skipped, links_made = [], 0, 0
    slug_to_id: dict[str, int] = {}      # heading-slug -> mem_id (this doc)
    pending_links: list[tuple[int, str]] = []
    # stack of (level, mem_id) for resolving parent of each section
    stack: list[tuple[int, int]] = []

    def add(level: int, gist: str, detail: str, slug: str, name: str) -> None:
        nonlocal links_made
        if store.conn.execute("SELECT 1 FROM memory WHERE name = ?", (name,)).fetchone():
            nonlocal skipped
            skipped += 1
            return
        mem_id = store.store(
            gist,
            detail or None,
            kind=kind,
            name=name,
            topics=[mtype] if mtype else [],
            project=project,
            event_time=mtime,
            recorded_time=mtime,
            salience=salience,
            source=source,
            source_ref=str(path),
        )
        imported.append(name)
        slug_to_id[slug] = mem_id
        for target in WIKILINK.findall(detail):
            pending_links.append((mem_id, target.strip()))
        # link to the nearest shallower section already open
        while stack and stack[-1][0] >= level:
            stack.pop()
        if stack:
            store.link(mem_id, stack[-1][1], "refines")
            links_made += 1
        stack.append((level, mem_id))

    # doc-root from the preamble (or a bare title) at level 0; named with the
    # bare doc-slug so [[Title]] links and `show <doc-slug>` resolve to it.
    root_gist = meta.get("description") or meta.get("title") or path.stem
    if preamble or meta.get("description") or meta.get("title"):
        add(0, root_gist, preamble, slugify(meta.get("title") or path.stem), doc_slug)

    for level, heading, sec_body in sections:
        if not heading:
            continue
        slug = slugify(heading)
        add(level, heading, sec_body, slug, f"{doc_slug}#{slug}")

    # resolve wikilinks: prefer this doc's headings, then any existing name
    for mem_id, target in pending_links:
        related = slug_to_id.get(slugify(target))
        if related is None:
            for candidate in (target, f"{doc_slug}#{slugify(target)}"):
                row = store.conn.execute(
                    "SELECT id FROM memory WHERE name = ?", (candidate,)
                ).fetchone()
                if row:
                    related = row["id"]
                    break
        if related is not None and related != mem_id:
            store.link(mem_id, related, "relates")
            links_made += 1

    return {"imported": len(imported), "skipped": skipped, "links": links_made,
            "names": imported}


def import_path(
    store: MemoryStore,
    path: str | Path,
    *,
    project: str | None = None,
    source: str = "markdown-doc",
) -> dict:
    """Import a single .md file or every .md file in a directory."""
    path = Path(path).expanduser()
    files = sorted(path.glob("*.md")) if path.is_dir() else [path]
    total = {"imported": 0, "skipped": 0, "links": 0, "names": []}
    for f in files:
        r = import_document(store, f, project=project, source=source)
        for k in ("imported", "skipped", "links"):
            total[k] += r[k]
        total["names"].extend(r["names"])
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a .md file or a directory of .md files")
    ap.add_argument("--db")
    ap.add_argument("--project")
    args = ap.parse_args(argv)
    result = import_path(MemoryStore(db_path=args.db), args.path, project=args.project)
    print(f"imported {result['imported']}, skipped {result['skipped']}, "
          f"links {result['links']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
