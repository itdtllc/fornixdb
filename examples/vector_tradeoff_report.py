"""Print the real vector tradeoff so the default can be set from data.

    python3 -m examples.vector_tradeoff_report     # needs model2vec for real numbers

Measures vector-on vs vector-off on one controlled corpus across recall
ability, db space, write/recall time, and prompt-token cost. Uses the real
default embedder (model2vec) when installed; without it, says so and stops —
the point is true magnitudes, not the fake one the unit test uses.
"""

from __future__ import annotations

from fornixdb.vector_tradeoff import format_report, measure
from fornixdb.vectors import get_default_embedder


def main() -> None:
    emb = get_default_embedder()
    if emb is None:
        print("No embedder installed — `pip install model2vec` (or "
              "`pip install fornixdb[vectors]`) to measure real numbers.\n"
              "Without it FornixDB runs keyword + time only, which is the "
              "'vector off' column this report would compare against.")
        return
    print(format_report(measure(embedder=emb, repeats=50)))
    print("\nReading it: vectors add db space + a little write/recall time and "
          "~no prompt tokens, and buy recall that finds memories by MEANING "
          "(the synonym row) that keyword search misses entirely.")


if __name__ == "__main__":
    main()
