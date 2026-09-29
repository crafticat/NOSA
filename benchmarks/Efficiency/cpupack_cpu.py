"""CPU placement for the CPU-packing transport: physical cores, the launch / transport-team / others split, affinity, the
thread census. Pure python (no torch import: the driver applies the early placement BEFORE importing torch, so every
thread torch / CUDA / Triton create at initialisation inherits the 'others' mask). CPU-tested (retroinfer-eval
tests/test_cpupack_transport.py).

LAYOUT (on the GPU's NUMA node; physical cores = one logical CPU per (package, core)):
  launch  1 core: the decode's Python launch thread ONLY (its mask is narrowed to it after warm-up)
  team    up to TEAM_MAX cores: the transport coordinator (= the OpenMP master of its index_select calls) and its OpenMP
          helpers. An arm with N cores uses team[:N]; N counts ALL transport CPU work.
  others  the rest: CUDA driver threads, Triton, nsys, and the main thread's own OpenMP team (untimed checks).
NEVER set OMP_PROC_BIND / OMP_PLACES: GOMP would bind the initial (launch) thread (calibration job 2179369: the main
thread's mask collapsed to one CPU). A GOMP helper inherits the mask of the thread that creates it, so the coordinator
narrows its own mask before its first parallel region, and team sizes are first used in increasing order (1, 2, 4, 8):
every helper is then created inside the smallest prefix of the team that uses it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

TEAM_MAX_DEF = 8


def read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def parse_cpulist(s: Optional[str]) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def topology(cpus: Sequence[int], sysroot: str = "/sys/devices/system") -> Dict[int, Dict]:
    """Per CPU: package, core id, SMT siblings, NUMA node (from sysfs)."""
    node_of: Dict[int, int] = {}
    nroot = os.path.join(sysroot, "node")
    if os.path.isdir(nroot):
        for d in os.listdir(nroot):
            if d.startswith("node") and d[4:].isdigit():
                for c in parse_cpulist(read(os.path.join(nroot, d, "cpulist"))):
                    node_of[c] = int(d[4:])
    out = {}
    for c in cpus:
        base = os.path.join(sysroot, "cpu", "cpu%d" % c, "topology")
        out[c] = dict(pkg=read(os.path.join(base, "physical_package_id")), core=read(os.path.join(base, "core_id")),
                      siblings=parse_cpulist(read(os.path.join(base, "thread_siblings_list"))) or [c], node=node_of.get(c))
    return out


def physical_cpus(cpus: Sequence[int], topo: Dict[int, Dict]) -> List[int]:
    """One logical CPU per physical core (the lowest allowed sibling), sorted."""
    seen, out = set(), []
    for c in sorted(cpus):
        t = topo.get(c, {})
        key = (t.get("pkg"), t.get("core")) if t.get("core") is not None else ("cpu", c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def choose(allowed: Sequence[int], topo: Dict[int, Dict], gpu_node: Optional[int], team_max: int = TEAM_MAX_DEF) -> Dict:
    """The launch / team / others split. Physical cores on the GPU's node first; refuses (ok=False) when fewer than
    1 + team_max + 1 physical cores exist there (a launch core, the full team, at least one core for everything else)."""
    phys = physical_cpus(allowed, topo)
    local = [c for c in phys if gpu_node is None or topo.get(c, {}).get("node") == gpu_node]
    notes = []
    if gpu_node is not None and len(local) < len(phys):
        notes.append("%d physical cores off the GPU node %s are not used for launch/team" % (len(phys) - len(local), gpu_node))
    ok = len(local) >= team_max + 2
    if not ok:
        notes.append("only %d physical cores on node %s; need %d" % (len(local), gpu_node, team_max + 2))
    launch = local[0] if local else (phys[0] if phys else None)
    team = local[1:1 + team_max]
    used = set([launch] + team)
    others = [c for c in sorted(allowed) if c not in used]
    smt_in_team = [c for c in team if len(topo.get(c, {}).get("siblings") or [c]) > 1]
    return dict(ok=ok, launch=launch, team=team, others=others, physical=phys, gpu_node=gpu_node, smt_siblings_in_team=smt_in_team,
                notes=notes)


def check_omp_env(env=None) -> List[str]:
    env = os.environ if env is None else env
    return ["%s=%r is set: GOMP would bind the launch thread; unset it" % (k, env[k]) for k in ("OMP_PROC_BIND", "OMP_PLACES")
            if env.get(k)]


def set_mask(cpus: Sequence[int]) -> List[int]:
    """The CALLING thread's mask (Linux: pid 0 = the calling thread)."""
    os.sched_setaffinity(0, set(int(c) for c in cpus))
    return sorted(os.sched_getaffinity(0))


def early_placement(team_max: int = TEAM_MAX_DEF, gpu_node: Optional[int] = None) -> Dict:
    """Run BEFORE `import torch`: choose the split, set the main thread's mask to the others pool and OMP_NUM_THREADS to its
    size (the main thread's own OpenMP team, used only for untimed checks). Returns the plan (recorded in the JSON)."""
    allowed = sorted(os.sched_getaffinity(0))
    topo = topology(allowed)
    plan = choose(allowed, topo, gpu_node, team_max)
    plan["allowed"] = allowed
    plan["omp_env_problems"] = check_omp_env()
    if plan["others"]:
        set_mask(plan["others"])
        os.environ["OMP_NUM_THREADS"] = str(max(1, len(plan["others"])))
    plan["main_mask_at_init"] = sorted(os.sched_getaffinity(0))
    return plan


def census() -> List[Dict]:
    """Every thread of this process: tid, comm, allowed CPUs, user/system ticks, the CPU it last ran on."""
    out = []
    base = "/proc/self/task"
    for tid in sorted(os.listdir(base), key=int) if os.path.isdir(base) else []:
        st = read(os.path.join(base, tid, "stat")) or ""
        status = read(os.path.join(base, tid, "status")) or ""
        allowed = next((l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list")), None)
        try:
            rest = st[st.rindex(")") + 2:].split()
            utime, stime, proc = int(rest[11]), int(rest[12]), int(rest[36])
            comm = st[st.index("(") + 1:st.rindex(")")]
        except (ValueError, IndexError):
            utime = stime = proc = -1
            comm = "?"
        out.append(dict(tid=int(tid), comm=comm, cpus_allowed=allowed, utime=utime, stime=stime, last_cpu=proc))
    return out


def census_delta(before: Sequence[Dict], after: Sequence[Dict]) -> List[Dict]:
    """Threads that consumed CPU between two censuses (ticks), with their allowed CPUs and last CPU."""
    b = {r["tid"]: r for r in before}
    out = []
    for r in after:
        p = b.get(r["tid"], dict(utime=0, stime=0))
        d = (r["utime"] - p["utime"]) + (r["stime"] - p["stime"])
        if d > 0:
            out.append(dict(tid=r["tid"], comm=r["comm"], ticks=d, cpus_allowed=r["cpus_allowed"], last_cpu=r["last_cpu"],
                            new=r["tid"] not in b))
    return out


def cpu_model() -> Optional[str]:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def smt_active() -> Optional[str]:
    return read("/sys/devices/system/cpu/smt/active")


def meminfo(path: str = "/proc/meminfo", keys=("MemTotal", "MemFree", "MemAvailable", "Mlocked", "Unevictable")) -> Dict:
    out = {}
    try:
        for line in open(path):
            k = line.split(":", 1)[0].strip()
            if k in keys:
                out[k] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return out


def node_meminfo(sysroot: str = "/sys/devices/system/node") -> Dict:
    out = {}
    if os.path.isdir(sysroot):
        for d in sorted(os.listdir(sysroot)):
            if d.startswith("node") and d[4:].isdigit():
                txt = read(os.path.join(sysroot, d, "meminfo")) or ""
                for line in txt.splitlines():
                    if "MemFree" in line or "MemTotal" in line:
                        parts = line.split()
                        out.setdefault(d, {})[parts[2].rstrip(":")] = " ".join(parts[3:])
    return out
