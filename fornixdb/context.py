"""Active-project / context detection for project-scoped pulse recall.

A "context" is a label the user works under (e.g. "fornixdb"). A memory belongs
to it if its project field OR any of its (non-structural) topics matches the
label or one of the label's aliases — the belongs test itself is set logic in
`core.effective_floor`. This module owns everything around it that needs config
or the store's vocabulary, and is host-neutral (pure Python + the store):

  - **Aliases** (`config project_aliases`): so fornixdb == engramdb == aimemory,
    bridging a project's messy historical names.
  - **Declarable labels**: the project values + alias labels a user could *name*
    when they say what they're working on. (Topics aren't declarable — many are
    structural like "reference"/"milestone" — but they DO count in the belongs
    test.)
  - **Prompt detection**: a cue phrase ("continue the X project", "working on X",
    "switch to X") naming a declarable label sets the SESSION's active context,
    sticky until the user declares a different one. Conservative on purpose — a
    passing mention mid-task must not flip context.

Precedence for the active context (resolved in `proactive.resolve_active_project`):
a pinned `config active_project` > this session's prompt-declared label > the
host-supplied cwd basename > none.
"""

from __future__ import annotations

import re

from .multistore import get_config, set_config

_SESSION_KEY = "active_project_session_"   # + session_id

# "I am declaring what I'm working on" cues. Tight by design: a bare mention of a
# project name without one of these does NOT change the active context.
_CUE = re.compile(
    r"\b(?:work(?:ing)?\s+(?:on|with)|continue|resume|switch(?:ing)?\s+to|"
    r"let'?s\s+(?:do|continue|work\s+on|pick\s+up)|pick\s+up|back\s+to|"
    r"start(?:ing)?\s+(?:on|with))\b",
    re.I)


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def alias_groups(store) -> list[set[str]]:
    """Parse `config project_aliases` into equivalence groups. Lenient format:
    groups separated by ';' or newlines; labels within a group by '=', ',', or
    whitespace. e.g. "fornixdb=engramdb,aimemory; videos=elira"."""
    raw = get_config(store, "project_aliases", "") or ""
    groups: list[set[str]] = []
    for chunk in re.split(r"[;\n]+", raw):
        labels = {_norm(x) for x in re.split(r"[=,\s]+", chunk) if x.strip()}
        labels.discard("")
        if len(labels) > 1:
            groups.append(labels)
    return groups


def aliases_for(store, label: str) -> set[str]:
    """Every label equivalent to `label` (its alias group), EXCLUDING `label`
    itself. Empty when it has no aliases."""
    l = _norm(label)
    out: set[str] = set()
    for g in alias_groups(store):
        if l in g:
            out |= g
    out.discard(l)
    return out


def declarable_labels(store) -> set[str]:
    """Labels a prompt can name to declare a project: the store's distinct
    project values plus every alias label. Topics are deliberately excluded — too
    many are structural words ("reference", "milestone") that would false-trigger
    — but a project's friendly name is reachable by adding it as an alias."""
    out: set[str] = set()
    for (p,) in store.conn.execute(
            "SELECT DISTINCT project FROM memory "
            "WHERE project IS NOT NULL AND project <> ''"):
        out.add(_norm(p))
    for g in alias_groups(store):
        out |= g
    out.discard("")
    return out


def detect_active_project(store, prompt: str) -> str | None:
    """The project a prompt declares, or None. Requires BOTH a declaration cue
    and a known declarable label so an incidental mention doesn't change context.
    When several labels appear, the earliest in the prompt wins ("switch to
    videos" → videos)."""
    if not prompt or not _CUE.search(prompt):
        return None
    labels = declarable_labels(store)
    if not labels:
        return None
    low = prompt.lower()
    best, best_pos = None, None
    for lab in labels:
        m = re.search(r"\b" + re.escape(lab) + r"\b", low)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = lab, m.start()
    return best


def session_active_project(store, session_id: str | None) -> str | None:
    """The sticky active context declared earlier this session, or None."""
    if not session_id:
        return None
    return (get_config(store, _SESSION_KEY + session_id, "") or "").strip() or None


def maybe_set_session_project(store, session_id: str | None,
                              prompt: str) -> str | None:
    """If `prompt` declares a project, persist it as this session's sticky active
    context and return it; else return None and leave any prior value in place.
    Best-effort — a read-only store just skips the write."""
    if not session_id:
        return None
    lab = detect_active_project(store, prompt)
    if lab:
        try:
            set_config(store, _SESSION_KEY + session_id, lab)
        except Exception:
            pass
    return lab
