"""A1: the counter-based random generator. Every draw is a pure function of the key
(seed, request uid, absolute position, epoch, purpose in {draft, accept, residual}), so a serial twin reproduces each
draw without knowing how many draws came before (vLLM @99b44c7 rejection_sampler.py:659-673 draws in sequence from a
per-request generator; SGLang @87db743 speculative_hook.py:948-953 refuses rejection sampling under deterministic
inference because coins come from the global RNG; PEARL @fe7ddef src/engine.py:206, :231 keys by seed + position only).

Format (the twin re-implements it from this text): BLAKE2b with digest_size=8 and person=b"nosi-kstep-rng01" over
struct.pack("<qqqqB", seed, uid, pos, epoch, code), code = draft 1, accept 2, residual 3. The digest read as a
little-endian unsigned 64-bit integer, shifted right by 11, is k in [0, 2^53). The uniform is the exact rational k / 2^53.
"""
from __future__ import annotations

import hashlib
import struct
from fractions import Fraction

PURPOSES = {"draft": 1, "accept": 2, "residual": 3}
DENOM = 1 << 53
PERSON = b"nosi-kstep-rng01"


def u53(seed: int, uid: int, pos: int, epoch: int, purpose: str) -> int:
    code = PURPOSES[purpose]
    h = hashlib.blake2b(struct.pack("<qqqqB", int(seed), int(uid), int(pos), int(epoch), code), digest_size=8, person=PERSON)
    return int.from_bytes(h.digest(), "little") >> 11


def uniform(seed: int, uid: int, pos: int, epoch: int, purpose: str) -> Fraction:
    """Exact uniform in [0, 1)."""
    return Fraction(u53(seed, uid, pos, epoch, purpose), DENOM)
