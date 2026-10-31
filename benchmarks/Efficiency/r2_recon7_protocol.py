"""RECON7 (the reconciliation of revision 7, KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE CELL GRANTS of the three measurement cells

  (a) r2_recon7_cell.py     the profiler-window cell of the best arms (L4PTFDGUDCD, L4PTFDGUDKWCD): r7pm, r7pa, r7pb
  (b) r2_recon7_streams.py  the concurrency screen (X1): r7s
  (c) r2_recon7_mem.py      the C 81 memory probe (R3 discriminator D3): r7mem

and their one step script r2_recon7_step.sh. The protocol of each cell is docs/NOTES_recon7.md.

DEFAULT DENY, ONE GRANT PER CELL (review 3, finding 1: one approval must not unlock the other cells). A cell runs only
when ITS line below reads exactly `GRANT_<cell> = "<stamp>"` with a `date -u` stamp of the form 2026-10-31T12:00:00Z
(lesson L33: from `date -u` in the same command). Every other value (None, a malformed stamp, a missing line) refuses.
The step script reads the same line of `git show <sha>:<this file>` with one fixed pattern per cell; each cell's main()
calls refusal(cell, module, args), which also checks that the cell is one of the module's cells and that the run's
arguments are the granted PROFILE of that cell (a grant of r7pm does not run r7pa's batch through the same module).
"""

STAMP_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
CELLS = ("r7pm", "r7pa", "r7pb", "r7s", "r7mem")
CELL_MODULE = dict(r7pm="r2_recon7_cell", r7pa="r2_recon7_cell", r7pb="r2_recon7_cell", r7s="r2_recon7_streams",
                   r7mem="r2_recon7_mem")

# THE GRANTS (one line each; the step script greps `^GRANT_<cell> = "<stamp>"$`).
GRANT_r7pm = None
GRANT_r7pa = None
GRANT_r7pb = None
GRANT_r7s = None
GRANT_r7mem = None

_ARMS_A = ["L4PTFD", "L4PTFDGUDC", "L4PTFDGUDCD", "L4PTFDGUDCDP"]
_ARMS_B = ["L4PTFD", "L4PTFDGUDKWC", "L4PTFDGUDKWCD", "L4PTFDGUDKWCDP"]
_ARMS_M = ["L4PTFD", "L4PTFDGUDC", "L4PTFDGUDKWC", "L4PTFDGUDCD", "L4PTFDGUDKWCD", "L4PTFDGUDCDP", "L4PTFDGUDKWCDP"]
# THE GRANTED PROFILES: the parsed arguments that a grant covers (= the step script's CARGS; a CPU test compares them).
PROFILES = dict(
    r7pm=dict(batch=8, steps=72, pool=10, arms=_ARMS_M, probe_every=4, profile_rounds=[8, 12], label_rounds=[12],
              count_rounds=[16, 20]),
    r7pa=dict(batch=216, steps=512, pool=65, arms=_ARMS_A, probe_every=16, profile_rounds=[96, 112],
              label_rounds=[112], count_rounds=[48, 64, 144]),
    r7pb=dict(batch=318, steps=512, pool=10, arms=_ARMS_B, probe_every=16, profile_rounds=[96, 112],
              label_rounds=[112], count_rounds=[48, 64, 144]),
    r7s=dict(shapes={128: 216, 73: 318}, new_blocks={128: 1.226, 73: 5.0}, union_pages=72, win_share=0.5, layers=32,
             reps=10, warmup=2, seed=7, max_shape_s=240.0, max_run_s=600.0),
    r7mem=dict(batches=[64, 192], steps=512, pool=18, probe_every=16, probe_rounds=49))


def stamp_ok(stamp) -> bool:
    """True when `stamp` is a `date -u` stamp string."""
    import re
    return isinstance(stamp, str) and re.fullmatch(STAMP_RE, stamp) is not None


def grant(cell):
    """The grant value of `cell` (None when the cell or its line is missing)."""
    return globals().get("GRANT_%s" % cell) if cell in CELLS else None


def refusal(cell, module: str, args=None):
    """The cells' main(): None when `cell` is granted, is one of `module`'s cells and `args` is its granted profile;
    else the refusal text (exit 20). Default deny."""
    if cell not in CELLS:
        return "REFUSED: %s: cell %r is not one of %s (--cell)" % (module, cell, CELLS)
    if CELL_MODULE[cell] != module:
        return "REFUSED: %s: cell %s belongs to %s" % (module, cell, CELL_MODULE[cell])
    if not stamp_ok(grant(cell)):
        return ("REFUSED: %s: GRANT_%s is %r (r2_recon7_protocol.py): cell %s is not granted (default deny)"
                % (module, cell, grant(cell), cell))
    if args is None:
        return "REFUSED: %s: no arguments to check against the granted profile of %s" % (module, cell)
    bad = [k for k, v in PROFILES[cell].items() if getattr(args, k, None) != v]
    if bad:
        return "REFUSED: %s: %s differs from the granted profile of %s: %s" % (
            module, bad, cell, {k: (getattr(args, k, None), PROFILES[cell][k]) for k in bad})
    return None
