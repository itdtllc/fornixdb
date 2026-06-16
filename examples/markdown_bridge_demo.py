"""A guided walkthrough of the Markdown bridge — run it and READ the output.

    python3 -m examples.markdown_bridge_demo

It uses examples/sample_docs/homelab_notes.md (a short, plain Markdown note like
the AI community keeps) and shows, end to end:

  1. INGEST  — the note is split by its headings into separate memories.
  2. RECALL  — a question pulls back just the relevant section, cheaply.
  3. BENEFIT — the same questions cost far more to answer from the whole note.
  4. EXPORT  — memories are written back out as Markdown you can read & edit.

No network, no API, nothing is written outside a temp folder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fornixdb.adapters.markdown_export import export_directory
from fornixdb.markdown_benefit import benefit_report, build_chunked, format_benefit

DOC = Path(__file__).parent / "sample_docs" / "homelab_notes.md"
RULE = "─" * 72


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    banner("0.  THE SOURCE NOTE  (plain Markdown — what you'd write or paste in)")
    print(f"file: {DOC}\n")
    print(DOC.read_text().rstrip())

    # 1. INGEST -------------------------------------------------------------
    banner("1.  INGEST  —  the note becomes one memory PER HEADING")
    store = build_chunked(DOC)
    rows = store.conn.execute(
        "SELECT name, gist, detail FROM memory ORDER BY id").fetchall()
    print(f"The single note split into {len(rows)} separate memories.\n")
    print(f"  {'memory name (handle)':<34} {'gist = the heading'}")
    for r in rows:
        print(f"  {r['name']:<34} {r['gist']}")
    print("\nEach memory keeps its section text as 'detail'. For example, the")
    print("'Backups' memory holds exactly — and only — its own section:\n")
    backups = store.show("home-lab-notes#backups", reinforce=False)
    print(f'    gist  : {backups["gist"]}')
    print(f'    detail: {backups["detail"]}')
    if backups.get("links"):
        print("    links :", ", ".join(
            f'{l["relation"]} -> {l["related_gist"][:24]}' for l in backups["links"]))

    # 2. RECALL -------------------------------------------------------------
    banner("2.  RECALL  —  a question pulls back JUST the right section")
    for q in ["nightly Time Machine offsite Backblaze backup",
              "Synology NAS web UI port",
              "ISP Sonic fiber internet provider"]:
        top = store.recall(q, limit=1)[0]
        print(f'  you ask : "{q}"')
        print(f'  you get : [{top["gist"]}]  {top["detail"][:90]}...\n')

    # 3. BENEFIT ------------------------------------------------------------
    banner("3.  BENEFIT  —  vs keeping the whole note as ONE memory (the 'before')")
    questions = [
        {"query": "nightly Time Machine offsite Backblaze backup", "answer_contains": "2:00 AM"},
        {"query": "Synology NAS web UI port", "answer_contains": "192.168.1.50"},
        {"query": "ISP Sonic fiber internet provider", "answer_contains": "Sonic"},
        {"query": "cameras retain footage Surveillance", "answer_contains": "14 days"},
    ]
    print(format_benefit(benefit_report(DOC, questions)))
    print("\n  Why this matters: an AI must RE-READ whatever recall returns on")
    print("  every turn. Returning one small section instead of the whole note")
    print("  is the difference in cost — and it keeps the irrelevant text out.")
    print("\n  Honest caveats:")
    print("  • This note is tiny, so the ratio is modest. It GROWS with document")
    print("    size — a real multi-page design doc or vault page is far larger,")
    print("    while a section stays small, so the saving is much bigger there.")
    print("  • The win assumes the answer lives in ONE section. A question that")
    print("    must combine several sections needs several chunks (still cheaper")
    print("    than the whole doc, but not one tidy hit).")
    print("  • Recall here is keyword-based; phrase queries with words the note")
    print("    actually uses (or install the optional vectors for fuzzy match).")

    # 4. EXPORT -------------------------------------------------------------
    banner("4.  EXPORT  —  memories written back out as readable Markdown")
    out = Path(tempfile.mkdtemp()) / "export"
    res = export_directory(store, out)
    print(f"exported {res['exported']} memories to {out}\n")
    print("INDEX FILE (MEMORY.md) — your table of contents:\n")
    print("    " + "\n    ".join((out / "MEMORY.md").read_text().rstrip().splitlines()))
    sample = out / "home-lab-notes__backups.md"
    print(f"\nONE EXPORTED MEMORY ({sample.name}):\n")
    print("    " + "\n    ".join(sample.read_text().rstrip().splitlines()))

    banner("HOW TO READ AN EXPORTED MARKDOWN MEMORY")
    print("""\
  • The block between the '---' lines is FRONTMATTER (metadata), key: value:
      name           the unique handle you can recall or link by
      description    the gist — the one-line summary recall shows first
      kind           semantic / feedback / reference / episodic
      topics, times, salience — bookkeeping the store tracks
      metadata.type  lets the file be re-imported to the same kind
  • Everything BELOW the second '---' is the memory's detail (the section text).
  • A '## Related' list shows links to other memories as [[handles]]; the
    word before it (refines / relates) is how they connect.
  • These are ordinary .md files: read them in any editor, commit them to git
    to see what changed over time, or edit one and re-import with
    `fornixdb import-markdown <dir> --frontmatter` to feed changes back in.""")
    store.close()
    print(f"\n(temp export left at {out} — delete it whenever; nothing else was touched.)")


if __name__ == "__main__":
    main()
