"""RECON7 (the reconciliation of revision 7, KS/analysis/plan/HUMAN_RECON_REQUEST_2026-10-30.md; KS =
/mnt/beegfs/ojerbi/kstep-cpu-runs): THE PROTOCOL STAMP of the three measurement cells

  (a) r2_recon7_cell.py     the profiler-window cell of the best arms (L4PTFDGUDCD, L4PTFDGUDKWCD)
  (b) r2_recon7_streams.py  the two-stream concurrency micro-harness (R2 discriminator D1)
  (c) r2_recon7_mem.py      the C 81 memory probe (R3 discriminator D3)

and their one step script r2_recon7_step.sh (cells r7pm, r7pa, r7pb, r7s, r7mem). The protocol of each cell (what the
run decides, its expected duration, its gates, its falsifier) is docs/NOTES_recon7.md section 3.

PROTOCOL_STAMP_RECON7 is None until the HUMAN approves the GPU protocol. The step script STOPS EVERY CELL while the line
below reads exactly `PROTOCOL_STAMP_RECON7 = None` at the commit it runs (a grep of `git show <sha>:<this file>`), and
each cell's main() refuses (exit 20) while it is None. The approver sets it in ONE commit to a `date -u` stamp of the
form 2026-10-30T12:00:00Z (lesson L33: from `date -u` in the same command). A builder never sets it.
"""

PROTOCOL_STAMP_RECON7 = None
STAMP_RE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
_UNSET = object()


def stamp_ok(stamp=_UNSET) -> bool:
    """True when the stamp (default: PROTOCOL_STAMP_RECON7) is a `date -u` stamp (the approval)."""
    import re
    s = PROTOCOL_STAMP_RECON7 if stamp is _UNSET else stamp
    return isinstance(s, str) and re.fullmatch(STAMP_RE, s) is not None


def refusal(cell: str):
    """The cells' main(): None when the protocol is approved, else the refusal text (exit 20)."""
    if stamp_ok():
        return None
    return ("REFUSED: %s: PROTOCOL_STAMP_RECON7 is None (r2_recon7_protocol.py): the GPU protocol of "
            "docs/NOTES_recon7.md is not approved" % cell)
