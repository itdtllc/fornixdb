"""Active-project / context detection for project-scoped pulse recall.

A "context" is a label the user works under (e.g. "fornixdb"). A memory belongs
to it if its project field OR any of its (non-structural) topics matches the
label or one of the label's aliases — the belongs test itself is set logic in
`core.effective_floor`. This module owns everything around it that needs config
or the store's vocabulary, and is host-neutral (pure Python + the store):

  - **Aliases** (`config project_aliases`): so fornixdb == engramdb == aimemory,
    bridging a project's messy historical names. The FIRST label in a group is
    that project's canonical spelling.
  - **Canonical labels** (`canonical_project`): one project, one stored spelling.
    Labels equal under case-folding are the same project — the store's own
    dominant spelling wins — and an alias group overrides that. Applied on the
    way IN (so capture can't fragment a project by cwd casing) and to the label
    a caller filters by (so a query still finds rows written before the fold).
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


def ordered_alias_groups(store) -> list[list[str]]:
    """`config project_aliases` as ordered groups, ORIGINAL casing kept. Format:
    groups separated by ';' or newlines; labels within a group by '=' or ','.
    e.g. "fornixdb=engramdb,aimemory; videos=archive".

    Whitespace separates nothing — it is part of the label, because project names
    have spaces in them ("Site Notes"). Splitting on it silently tore that
    group's third member into two junk labels, so the real one never aliased.

    Order carries meaning: the first label is the group's canonical spelling, and
    it is the one `canonical_project` rewrites the others to. Duplicates within a
    group are dropped, first occurrence winning."""
    raw = get_config(store, "project_aliases", "") or ""
    groups: list[list[str]] = []
    for chunk in re.split(r"[;\n]+", raw):
        labels: list[str] = []
        seen: set[str] = set()
        for x in re.split(r"[=,]+", chunk):
            lab = x.strip()
            if lab and _norm(lab) not in seen:
                seen.add(_norm(lab))
                labels.append(lab)
        if len(labels) > 1:
            groups.append(labels)
    return groups


def alias_groups(store) -> list[set[str]]:
    """`ordered_alias_groups` as case-folded sets — the membership view used by
    the belongs test, where order and casing are irrelevant."""
    return [{_norm(x) for x in g} for g in ordered_alias_groups(store)]


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


def _stored_label_counts(store) -> list[tuple[str, int]]:
    """Every distinct non-empty project spelling in the store with its row count,
    ordered most-used first (ties broken alphabetically so the result is stable)."""
    try:
        rows = store.conn.execute(
            "SELECT project, COUNT(*) FROM memory "
            "WHERE project IS NOT NULL AND project <> '' "
            "GROUP BY project").fetchall()
    except Exception:
        return []
    return sorted(((p, n) for p, n in rows), key=lambda r: (-r[1], r[0]))


def project_canon_map(store) -> dict[str, str]:
    """Folded label -> the canonical spelling that label should be written as.

    Two sources, config winning: an alias group's FIRST label canonicalizes every
    other member of the group, and otherwise a project's dominant spelling in the
    store canonicalizes its own case variants. Only case-folding and owner-declared
    aliases merge anything — two genuinely different names never collapse on their
    own, because deciding they mean one project is the owner's call, not ours."""
    out: dict[str, str] = {}
    # Dominant stored spelling first, so config can overwrite it below.
    for label, _n in reversed(_stored_label_counts(store)):
        out[_norm(label)] = label.strip()
    for group in ordered_alias_groups(store):
        canon = group[0].strip()
        for member in group:
            out[_norm(member)] = canon
    return out


def canonical_project(store, label: str | None) -> str | None:
    """The spelling `label` should be stored and queried under. Unknown labels
    (a project's first memory) canonicalize to themselves, merely trimmed — a new
    project must not need config to be storable."""
    if label is None:
        return None
    lab = label.strip()
    if not lab:
        return lab
    return project_canon_map(store).get(_norm(lab), lab)


def project_equivalents(store, label: str | None) -> list[str]:
    """Every spelling PRESENT IN THIS STORE that means the same project as
    `label`, canonical first. This is what a project filter must match on: a store
    written before the fold — or a read-only peer that will never be rewritten —
    still holds the old spellings, and a query for one of them must find them all.
    Returns [label] when nothing else matches, so callers can filter unconditionally."""
    if label is None:
        return []
    lab = label.strip()
    if not lab:
        return [lab]
    canon = canonical_project(store, lab)
    cmap = project_canon_map(store)
    out = [canon]
    for stored, _n in _stored_label_counts(store):
        s = stored.strip()
        if cmap.get(_norm(s), s) == canon and s not in out:
            out.append(s)
    if lab not in out:          # the caller's own spelling, even if unstored
        out.append(lab)
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
