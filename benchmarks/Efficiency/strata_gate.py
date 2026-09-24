"""The stage-N0 CORRECTNESS gate of the Strata / timeline experiment: the NOSI stages (S2, T) run only if SGLang's native
HiCache path was shown correct on the proxy. Pure python, CPU-tested (retroinfer-eval tests/test_strata_transfer.py).
A performance number never enters the gate.

    python strata_gate.py N0_JSON [--expect-pass-bytes N] [--grid TIMELINE_JSON]   -> exit 0 = PASS, 1 = FAIL, 2 = usage

PASS requires, from hisparse_repro.py's JSON of the hicache payload:
  1. fails == 0 (every transfer rep byte-exact against the source rows with the NaN-poisoned destination and the canary
     intact, every step's logits equal the mode's reference, NUMA / pointer gates)
  2. controls: can_use_host_pointer_for_registered_mem == 1; host_ptr_is_device_ptr == 1 for every host buffer;
     skip_io_detected.detected (HiSparse SkipIO); every neg_* control detected (omitted layer, swapped destination
     index, both kernels); payload_stats.canary_fail == 0; numa_local.ok when present
  3. at least one error-free, correct row of a strata* arm AND of a hicachejit* arm in every mode of the run
  4. meta.pass_bytes == --expect-pass-bytes when given (the registered 117,440,512 B)
  5. --grid: every rep of the nsys timeline that carries a side-kernel grid check has ok == true (the launched grid and
     block equal the upstream formulas); an absent file is reported and does not fail the gate (nsys may be refused)
"""
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def check(res: Dict, expect_pass_bytes: Optional[int] = None, grid: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    why = []
    meta = res.get("meta") or {}
    c = meta.get("controls") or {}
    if res.get("fails", 1) != 0:
        why.append("fails = %s" % res.get("fails"))
    if c.get("can_use_host_pointer_for_registered_mem") != 1:
        why.append("cudaDevAttrCanUseHostPointerForRegisteredMem = %r" % c.get("can_use_host_pointer_for_registered_mem"))
    ptr = c.get("host_ptr_is_device_ptr") or {}
    if not ptr or any(v != 1 for v in ptr.values()):
        why.append("host_ptr_is_device_ptr %r" % ptr)
    if not (c.get("skip_io_detected") or {}).get("detected"):
        why.append("HiSparse SkipIO control not detected: %r" % c.get("skip_io_detected"))
    negs = {k: v for k, v in c.items() if k.startswith("neg_")}
    if not negs:
        why.append("no negative control ran")
    for k, v in negs.items():
        if not isinstance(v, dict) or not v.get("detected"):
            why.append("%s not detected: %r" % (k, v))
    if (c.get("payload_stats") or {}).get("canary_fail", 1) != 0:
        why.append("canary: %r" % c.get("payload_stats"))
    if "numa_local" in c and not c["numa_local"].get("ok"):
        why.append("numa_local %r" % c["numa_local"])
    modes = (meta.get("config") or {}).get("modes") or []
    rows = res.get("rows") or []
    for fam in ("strata", "hicachejit"):
        for mode in modes:
            good = [r for r in rows if r.get("arm", "").startswith(fam) and r.get("mode") == mode and not r.get("error") and r.get("fails") == 0]
            if not good:
                why.append("no correct %s row in mode %s" % (fam, mode))
    errs = [r["arm"] for r in rows if r.get("error")]
    if errs:
        why.append("rows with errors: %s" % sorted(set(errs)))
    if expect_pass_bytes is not None and meta.get("pass_bytes") != expect_pass_bytes:
        why.append("pass_bytes %r != %d" % (meta.get("pass_bytes"), expect_pass_bytes))
    if grid is not None:
        reps = [r for r in grid.get("reps") or [] if r.get("grid")]
        bad = [r["label"] for r in reps if not r["grid"].get("ok")]
        if not reps:
            why.append("nsys grid check: no side kernel of a known family in the trace")
        if bad:
            why.append("nsys grid mismatch in %d reps: %s" % (len(bad), bad[:4]))
    return not why, why


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    res = json.load(open(argv[1]))
    epb = int(argv[argv.index("--expect-pass-bytes") + 1]) if "--expect-pass-bytes" in argv else None
    grid = None
    if "--grid" in argv:
        gp = argv[argv.index("--grid") + 1]
        if os.path.exists(gp):
            grid = json.load(open(gp))
        else:
            print("[gate] nsys grid file %s absent: grid check NOT done (BLOCKER, not a gate failure)" % gp, flush=True)
    ok, why = check(res, epb, grid)
    print("[gate] N0 %s%s" % ("PASS" if ok else "FAIL", "" if ok else ": " + "; ".join(why)), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
