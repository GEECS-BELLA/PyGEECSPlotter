# PyGEECSPlotter

Scan-analysis library for the BELLA laser–plasma accelerator (LBNL). It loads
a GEECS scan's scalar table (the "sfile"), filters/bins/correlates it, runs
per-shot diagnostic analyzers over camera images, spectra and traces, writes
new scalars back to the sfile, and makes scan-level plots.

- Changing or extending library code → use the `pygeecsplotter-dev` skill.
- Analysing or plotting a scan with the library → use the
  `pygeecsplotter-usage` skill.

## Rules (every session)

- No keyword-only `*` markers in function signatures, anywhere.
- Branch off and open PRs against `origin/sbir_post_analysis`, never `main`.
- Don't commit, push or open a PR unless asked.
- Never write to a real sfile while testing (`write_columns_to_sfile=False`).
- Never hardcode data paths (e.g. `N:\data`) in library code; take them as
  parameters.
- Scratch scripts go outside the repo, never committed.
- There is no `python` on PATH; the interpreter to use is in `CLAUDE.local.md`
  or given by the user.
