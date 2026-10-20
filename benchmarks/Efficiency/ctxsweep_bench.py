"""CONTEXT SWEEP driver (paper 4.2): the rules, units and labels are in ctxsweep_core.py (its docstring); one arm is
eqmem_bench.run_arm (the 4.1 arm) with ctxsweep_core's S-aware plan.

  python ctxsweep_bench.py --mode plan   --budget-gb 75 --cells-file ctxsweep_cells.txt [--fit-rows-dir <run dir>]
                                         [--host-pow2-base-gib 18] --out <plan.json>
  python ctxsweep_bench.py --mode arm    --budget-gb 75 --ctx 65280 --cell cached:50 --batch 32 --mem-gib 180
                                         --model <NOSA-8B> --tokens <ctx65280_tokens.bin> --manifest <ctx65280_manifest.json>
                                         --out <new dir> [--fit-rows-dir <run dir>] [--rep 0] [--gpu-uuid <uuid>]
  python ctxsweep_bench.py --mode report --results <run dir> [--results <4.1 run dir>] --out <new dir>

PLAN: every entry of the cell list planned with the fits (the prior GPU fit and the '2x' host model, or the refit from
the fit_row.json files under --fit-rows-dir; --host-pow2-base-gib plans with the measured power-of-two host model
instead): B, bound_by, the largest admitted B and its bound, the predicted device use, the host peak and the --mem it
needs, the documents, the minutes; the minutes per tier. Exit 4 when an entry is refused (nothing written).
ARM: the plan of (--ctx, --cell, --batch, --mem-gib) refit from --fit-rows-dir (the step passes its run directory, so a
'max' cell uses every row measured before it), then eqmem_bench.run_arm. The manifest must be of --ctx.
REPORT: every result.json under the --results directories (4.1 results are 16128 rows) -> report.json / report.md: per
context the cells ('<cell>@<B>'), equal memory (the best VALID cell per family; candidates vs the best baseline), the
matched-batch verdicts (helps / hurts / inconclusive: ctxsweep_core.verdict).
EXIT: as eqmem_bench: 0 the arm ran and is VALID, 1 an error or OVER_BUDGET, 2 --out exists, 3 another GPU, 4 a refused
configuration, 6 a module from outside NOSI_REPO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import ctxsweep_core as CS  # noqa: E402
import eqmem_bench as EB  # noqa: E402
import eqmem_core as EC  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="context sweep (ctxsweep_core.py)")
    ap.add_argument("--mode", choices=("plan", "arm", "report"), required=True)
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--batch", default="max", help="an explicit batch or 'max'")
    ap.add_argument("--mem-gib", type=int, default=CS.MEM_MAX_GIB, help="the step's srun --mem (GiB)")
    ap.add_argument("--cells-file", default=None)
    ap.add_argument("--n-new", type=int, default=512, help="generated tokens per request (the first from the prefill)")
    ap.add_argument("--margin-gb", type=float, default=1.0)
    ap.add_argument("--b-cap", type=int, default=4096)
    ap.add_argument("--fit-rows-dir", action="append", default=[], help="a run directory of */fit_row.json (repeatable)")
    ap.add_argument("--host-pow2-base-gib", type=float, default=None, help="plan: the pow2 host model with this base")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--doc-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--gpu-uuid", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", action="append", default=[])
    return ap


def load_rows(dirs: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*/fit_row.json")):
            rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return rows


def fits_for(a):
    rows = load_rows(a.fit_rows_dir)
    sfit = CS.fit_ctx_from_rows(rows) if rows else CS.prior_fit()
    hfit = CS.fit_host_from_rows(rows)
    if a.host_pow2_base_gib is not None:
        hfit = CS.HostFit(base_gib=a.host_pow2_base_gib, mode="pow2", source="--host-pow2-base-gib %s" % a.host_pow2_base_gib)
    return sfit, hfit, len(rows)


def _batch(text: str):
    if text == "max":
        return "max"
    try:
        return int(text)
    except ValueError:
        raise EB.Refused("--batch %r: an integer or 'max'" % text)


def run_plan(a) -> int:
    if a.budget_gb is None or not a.cells_file:
        raise EB.Refused("plan mode needs --budget-gb and --cells-file")
    out = Path(a.out)
    if out.exists():
        print("ctxsweep: --out %s exists" % out, file=sys.stderr)
        return 2
    try:
        entries = CS.parse_cell_list(Path(a.cells_file).read_text())
    except ValueError as e:
        raise EB.Refused(str(e))
    sfit, hfit, nrows = fits_for(a)
    plans = CS.sweep_plan(entries, a.budget_gb, a.n_new, sfit, hfit, a.margin_gb, a.b_cap)
    bad = [p for p in plans if "error" in p]
    print("%-6s %-11s %5s %4s %-11s %8s %7s %8s %9s %6s %6s %s" % ("S", "cell", "batch", "B", "bound_by", "B_max", "used",
                                                               "host", "need/mem", "min", "tier", "docs"))
    for p in plans:
        e = p["entry"]
        if "error" in p:
            print("%-6d %-11s %5s REFUSED: %s" % (e["S"], e["cell"], e["batch"], p["error"]))
            continue
        q = p["plan"]
        d = q["docs"] or {}
        print("%-6d %-11s %5s %4d %-11s %4d %-3s %5.1fGB %6.1fGi %4d/%-4d %6.1f %6s %s" % (
            e["S"], e["cell"], e["batch"], q["B"], q["bound_by"], q["B_max"], q["B_max_bound_by"][:3],
            q["predicted_used_gb"], q["host"]["predicted_gib"], q["host"]["need_gib"], q["host"]["mem_gib"],
            q["est"]["minutes"], e["tier"], "%d distinct of %d%s" % (d.get("distinct", 0), d.get("n_docs", 0),
                                                                     ", REUSED" if d.get("reused") else "")))
    tiers = CS.tier_minutes(plans)
    print("minutes per tier: %s; fit rows %d; GPU fit: %s; host: %s" % (
        {k: round(v, 1) for k, v in sorted(tiers.items())}, nrows, sfit.source, hfit.source))
    if bad:
        print("ctxsweep: %d entries refused; nothing written" % len(bad), file=sys.stderr)
        return 4
    out.write_text(json.dumps(dict(budget_gb=a.budget_gb, n_new=a.n_new, margin_gb=a.margin_gb, fit_rows=nrows,
                                   sfit=sfit.as_dict(), hfit=hfit.as_dict(), tier_minutes=tiers, cells=plans),
                              indent=1, default=str) + "\n")
    return 0


def run_arm(a) -> int:
    for k in ("cell", "model", "tokens", "manifest", "ctx", "budget_gb"):
        if getattr(a, k) is None:
            raise EB.Refused("--%s is required in arm mode" % k.replace("_", "-"))
    try:
        cell = EC.parse_cell(a.cell)
    except ValueError as e:
        raise EB.Refused(str(e))
    sfit, hfit, nrows = fits_for(a)
    try:
        plan = CS.plan_ctx_cell(cell, a.ctx, a.budget_gb, a.n_new, sfit, hfit, a.margin_gb, a.mem_gib, _batch(a.batch),
                                a.b_cap)
    except ValueError as e:
        raise EB.Refused(str(e))
    plan["fit_rows"] = nrows
    ns = argparse.Namespace(cell=a.cell, out=a.out, manifest=a.manifest, tokens=a.tokens, doc_offset=a.doc_offset,
                            rep=a.rep, gpu_uuid=a.gpu_uuid, model=a.model, n_new=a.n_new, boot=a.boot, seed=a.seed,
                            fit=None, fit_rows=None, budget_gb=a.budget_gb, margin_gb=a.margin_gb)
    return EB.run_arm(ns, plan=plan)


def collect(dirs: List[str]) -> List[Dict[str, Any]]:
    results = []
    for d in dirs:
        for p in sorted(Path(d).glob("*/result.json")):
            r = json.loads(p.read_text())
            if r.get("status") in ("running", "refused", "wrong_gpu", "module_origin") or "plan_key" not in r:
                continue
            if r.get("status") != "done":
                r.setdefault("error", dict(type="status", message=r.get("status")))
            r["_dir"] = str(p.parent)
            results.append(r)
    return results


def _f(v, fmt="%.1f") -> str:
    return "-" if v is None else (fmt % v)


def run_report(a) -> int:
    if not a.results:
        raise EB.Refused("--results is required")
    out = Path(a.out)
    if out.exists():
        print("ctxsweep: --out %s exists" % out, file=sys.stderr)
        return 2
    results = collect(a.results)
    if not results:
        print("ctxsweep: no results under %s" % a.results, file=sys.stderr)
        return 1
    rep = CS.sweep_report(results)
    lines = ["# context sweep (paper 4.2): %d results from %s" % (len(results), ", ".join(a.results)), ""]
    for S, blk in rep["contexts"].items():
        lines.append("## S = %s" % S)
        if "error" in blk:
            lines += ["ERROR: %s" % blk["error"], ""]
            continue
        lines += ["", "| cell@B | family | runs | batch | e2e tok/s | decode tok/s | used peak GB | over budget | errors |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, c in sorted(blk["cells"].items()):
            lines.append("| %s | %s | %d | %s | %s | %s | %s | %d | %d |" % (
                name, c["family"], c["runs"], c.get("batch"), EB._m(c.get("e2e_tok_s")), EB._m(c.get("decode_tok_s")),
                _f(c.get("used_peak_est_gb"), "%.2f"), c["over_budget"], c["errors"]))
        lines += ["", "| B | cell | e2e tok/s | decode tok/s (GPU units) [CI] | TTFT s | token p50 / p95 ms | vs | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for B, mb in blk["matched"].items():
            for name, c in sorted(mb["cells"].items()):
                cand = mb["candidates"].get(name)
                lines.append("| %s | %s | %s | %s [%s, %s] | %s | %s / %s | %s | %s |" % (
                    B, name, _f(c["e2e_tok_s"]), _f(c["decode_tok_s_gpu_units"]), _f(c["decode_ci"][0]),
                    _f(c["decode_ci"][1]), _f(c["ttft_s"], "%.2f"), _f(c["token_p50_ms"]), _f(c["token_p95_ms"]),
                    mb["baseline"] if cand else "(baseline)" if name == mb["baseline"] else "",
                    cand["verdict"] if cand else ""))
        eq = blk["equal_memory"]
        lines += ["", "equal memory: best per family %s; best baseline %s; candidates %s" % (
            {f: (b or {}).get("cell") for f, b in eq["best"].items()}, eq["best_baseline"], eq["speedup_e2e"]), ""]
    lines += ["LABEL: the spec family is LOCKSTEP (c = 1 + min_b n_acc per round): a LOWER BOUND of stage 3 until R2-M2"]
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.mode == "plan":
            return run_plan(a)
        if a.mode == "report":
            return run_report(a)
        return run_arm(a)
    except EB.Refused as e:
        print("ctxsweep: refused: %s" % e, file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
