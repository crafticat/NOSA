"""Acceptance (spec A1, A2) in EXACT arithmetic (fractions.Fraction; no float anywhere).

A1 (sampled, primary). d_s ~ q_s, recorded at draft time: q_s is the distribution the draft ACTUALLY used, under the
availability it actually had. For i in I_m in order: accept with probability min(1, p_i(d_i) / q_i(d_i)), i.e. iff
U_accept * q_i(d_i) < p_i(d_i); at the first rejection the correction is sampled from norm(max(0, p_i - q_i)) with
U_residual. A residual of zero mass means the token cannot be rejected: it is accepted (zero-mass rule; in exact
arithmetic a rejection implies p_i(d_i) < q_i(d_i), so the residual mass is positive and the rule is a guard). The
verifier never recomputes q.
A2 (GREEDY, reported separately). q = one-hot argmax of the draft distribution (lowest token id on a tie); accept iff
argmax p_i == d_i; correction = argmax p_{i*}.

Sampling from a weight vector w (all w >= 0, total T > 0) with a uniform u: the first index v with
w_0 + ... + w_v > u * T (inverse CDF, fixed order). Reference semantics (cited, not copied): vLLM @99b44c7
vllm/v1/sample/rejection_sampler.py:829-900 (accept rule), :1004-1083 (residual); SGLang @87db743
python/sglang/kernels/ops/speculative/reject_sampling.py:5-160 (accept iff coin * q < p, residual over relu(p - q)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence

SAMPLED, GREEDY = "sampled", "greedy"
MODES = (SAMPLED, GREEDY)


def argmax(dist: Sequence[Fraction]) -> int:
    best, bi = None, -1
    for i, x in enumerate(dist):
        if best is None or x > best:
            best, bi = x, i
    return bi


def sample_index(weights: Sequence[Fraction], u: Fraction) -> int:
    """Inverse CDF over non-negative weights (exact). Raises on a zero total."""
    total = sum(weights, Fraction(0))
    if total <= 0:
        raise ValueError("sample_index: zero or negative total mass")
    target = u * total
    acc = Fraction(0)
    last = -1
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError("sample_index: negative weight")
        acc += w
        if w > 0:
            last = i
        if acc > target:
            return i
    return last                                    # unreachable for u < 1


def one_hot(V: int, i: int) -> List[Fraction]:
    return [Fraction(1) if j == i else Fraction(0) for j in range(V)]


def draft_token(mode: str, dist: Sequence[Fraction], u: Fraction):
    """(d, q): sampled -> d ~ dist with u, q = dist; greedy -> d = argmax, q = one-hot(d)."""
    if mode == SAMPLED:
        return sample_index(dist, u), list(dist)
    if mode == GREEDY:
        d = argmax(dist)
        return d, one_hot(len(dist), d)
    raise ValueError("mode %r" % (mode,))


@dataclass
class Outcome:
    a: int
    b: int
    n_acc: int                                     # accepted drafts (d_a .. d_{a + n_acc - 1})
    i_star: Optional[int]                          # first rejected index, None on full acceptance
    correction: Optional[int]                      # the correction token at i_star
    zero_mass_accepts: int = 0
    accepted: List[int] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return self.i_star is None

    @property
    def committed(self) -> int:
        return self.n_acc + (0 if self.i_star is None else 1)


def accept_cohort(mode: str, a: int, b: int, d: Dict[int, int], q: Dict[int, Sequence[Fraction]],
                  p: Dict[int, Sequence[Fraction]], uniform_of: Callable[[int, str], Fraction], mut=None) -> Outcome:
    """Decide cohort [a, b]. d[i], q[i] recorded at draft time; p[i] from the exact verifier; uniform_of(i, purpose)
    returns the keyed uniform of index i (the caller binds seed, uid, absolute position, epoch)."""
    if mode not in MODES:
        raise ValueError("mode %r" % (mode,))
    out = Outcome(a=a, b=b, n_acc=0, i_star=None, correction=None)
    greedy_rule = (mode == GREEDY) or bool(mut and mut.greedy_in_sampled)
    for i in range(a, b + 1):
        pi, di = p[i], d[i]
        if greedy_rule:
            g = argmax(pi)
            if g == di:
                out.n_acc += 1
                out.accepted.append(i)
                continue
            out.i_star, out.correction = i, g
            return out
        qi = pi if (mut and mut.verifier_recomputes_q) else q[i]
        if qi[di] <= 0:
            raise ValueError("A1: recorded q_%d(d_%d) = 0: d was not drawn from q" % (i, i))
        u = uniform_of(i, "accept")
        if u * qi[di] < pi[di]:                    # u < min(1, p/q)
            out.n_acc += 1
            out.accepted.append(i)
            continue
        if mut and mut.residual_no_clamp:
            w = [pv - qv for pv, qv in zip(pi, qi)]
            target = uniform_of(i, "residual") * sum(w, Fraction(0))
            acc, c = Fraction(0), len(w) - 1
            for v, x in enumerate(w):
                acc += x
                if acc > target:
                    c = v
                    break
            out.i_star, out.correction = i, c
            return out
        w = [max(Fraction(0), pv - qv) for pv, qv in zip(pi, qi)]
        if sum(w, Fraction(0)) == 0:
            out.zero_mass_accepts += 1             # cannot be rejected
            out.n_acc += 1
            out.accepted.append(i)
            continue
        out.i_star, out.correction = i, sample_index(w, uniform_of(i, "residual"))
        return out
    return out
