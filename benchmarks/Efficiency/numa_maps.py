"""NUMA placement of host buffers from /proc/self/numa_maps, shared by worker_sweep.py and hisparse_repro.py. Pure
python, CPU-tested (retroinfer-eval tests/test_hisparse_copy_plan.py).

Review NB1 / NB11 (2026-09-23):
  * parse() reads /proc/self/numa_maps ONCE (each read walks every mapping of the process, ~267 GB of pinned host
    cache at B = 336) and returns every mapping (VMA);
  * range_pages() attributes a buffer [ptr, ptr + nbytes) to EVERY mapping it touches -- the mapping that contains
    ptr and every mapping that starts inside the range -- not only the mapping of the base pointer, so a buffer
    split across VMAs is checked whole. numa_maps gives each VMA's start but not its end, so a VMA that begins
    inside the buffer and runs past its end is counted whole: `covered_bytes` (pages x page size) is reported next
    to nbytes so an over-count is visible. A buffer whose pages are all on one node cannot be reported split.
"""
from typing import Dict, List, Optional

NUMA_MAPS = "/proc/self/numa_maps"


def parse_line(line: str) -> Optional[Dict]:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        start = int(parts[0], 16)
    except ValueError:
        return None
    nodes, page_kb = {}, 4
    for p in parts[2:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k.startswith("N") and k[1:].isdigit():
            nodes[k] = int(v)
        elif k == "kernelpagesize_kB":
            page_kb = int(v)
    return dict(start=start, policy=parts[1], pages_per_node=nodes, page_kb=page_kb,
                flags=[p for p in parts[2:] if not (p.startswith("N") and "=" in p)][:6])


def parse(text: Optional[str] = None) -> List[Dict]:
    """Every mapping, sorted by start address. `text` is for tests; None reads NUMA_MAPS once."""
    if text is None:
        with open(NUMA_MAPS) as f:
            text = f.read()
    vmas = [v for v in (parse_line(l) for l in text.splitlines()) if v is not None]
    return sorted(vmas, key=lambda v: v["start"])


def range_pages(vmas: List[Dict], ptr: int, nbytes: int) -> Optional[Dict]:
    """Pages per node of every mapping the buffer [ptr, ptr + nbytes) touches (see the module docstring)."""
    end = ptr + max(int(nbytes), 1)
    base = None
    hit = []
    for v in vmas:
        if v["start"] <= ptr:
            base = v
        elif v["start"] < end:
            hit.append(v)
    touched = ([base] if base is not None else []) + hit
    if not touched:
        return None
    ppn: Dict[str, int] = {}
    covered = 0
    for v in touched:
        for k, n in v["pages_per_node"].items():
            ppn[k] = ppn.get(k, 0) + n
            covered += n * v["page_kb"] * 1024
    return dict(pages_per_node=ppn, n_vmas=len(touched), mapping_start=hex(touched[0]["start"]), policy=touched[0]["policy"],
                covered_bytes=covered, nbytes=int(nbytes), flags=touched[0]["flags"])


def pages_at(ptr: int, vmas: Optional[List[Dict]] = None) -> Optional[Dict]:
    """The mapping that contains ptr alone (the old record; kept for comparison)."""
    try:
        vmas = parse() if vmas is None else vmas
    except OSError as e:
        return dict(error=str(e))
    best = None
    for v in vmas:
        if v["start"] <= ptr:
            best = v
    if best is None:
        return None
    return dict(mapping_start=hex(best["start"]), policy=best["policy"], pages_per_node=dict(best["pages_per_node"]), flags=best["flags"])


def node_of(rec: Optional[Dict]) -> str:
    """'0' / '1' / ... when every page is on one node, 's' when split, '?' when unknown."""
    if not rec or not rec.get("pages_per_node"):
        return "?"
    nodes = [k for k, n in rec["pages_per_node"].items() if n > 0]
    if len(nodes) == 1:
        return nodes[0][1:]
    return "s" if nodes else "?"


def all_on(rec: Optional[Dict], node: int) -> bool:
    """Every counted page on `node` (and at least one page counted)."""
    if not rec or "pages_per_node" not in rec:
        return False
    ppn = {k: n for k, n in rec["pages_per_node"].items() if n > 0}
    return bool(ppn) and set(ppn) == {"N%d" % node}
