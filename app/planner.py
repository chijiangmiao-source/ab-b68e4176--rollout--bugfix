"""Forwarding-table safety analysis and update-order planner.

A topology is a set of switches, each with an ``old_next`` and a ``new_next``
hop.  A hop target is either another declared switch id, the special exit
``DELIVER`` or the blackhole ``DROP``.  Updating a device atomically flips it
from ``old_next`` to ``new_next``.

A mixed forwarding state (identified by the set of already updated switches)
is *safe* iff walking from every ingress along the current next hops reaches
``DELIVER`` within at most ``N`` hops (``N`` = total number of switches).
Hitting ``DROP``, an unknown node, a repeated node or exceeding the hop bound
is unsafe.

The planner searches -- with full backtracking -- for a permutation of the
changed switches such that every prefix of the permutation yields a safe
mixed state, and returns the lexicographically smallest such permutation
(comparing switch ids by their raw UTF-8 bytes).  If no such permutation
exists it proves so by exhaustive search and returns ``None``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

DELIVER = "DELIVER"
DROP = "DROP"
RESERVED_TARGETS = (DELIVER, DROP)

MAX_SWITCHES = 60
MAX_PLANNED_STEPS = 22

# Encodings for resolved next-hop targets.
_MISSING = -3
_DROP = -2
_DELIVER = -1


def utf8_key(value: str) -> bytes:
    """Sort key implementing raw UTF-8 byte order comparison."""
    return value.encode("utf-8")


def canonical_topology(switches: Sequence[dict], ingresses: Sequence[str]) -> dict:
    """Canonical form of a topology used for stable hashing.

    Switches are sorted by id (UTF-8 byte order), ingresses sorted the same
    way, and only the semantically relevant fields are kept.
    """
    sw = sorted(
        (
            {"id": s["id"], "old_next": s["old_next"], "new_next": s["new_next"]}
            for s in switches
        ),
        key=lambda s: utf8_key(s["id"]),
    )
    ing = sorted(ingresses, key=utf8_key)
    return {"ingresses": list(ing), "switches": sw}


def stable_digest(obj) -> str:
    """Deterministic SHA-256 digest of a JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class Topology:
    """Resolved, index-based view of a topology for fast safety checks."""

    def __init__(self, switches: Sequence[dict], ingresses: Sequence[str]):
        self.switch_ids = [s["id"] for s in switches]
        self.index = {sid: i for i, sid in enumerate(self.switch_ids)}
        self.n = len(self.switch_ids)
        self.old_idx = [self._resolve(s["old_next"]) for s in switches]
        self.new_idx = [self._resolve(s["new_next"]) for s in switches]
        self.ingress_idx = [self.index[i] for i in ingresses]

    def _resolve(self, target: str) -> int:
        if target == DELIVER:
            return _DELIVER
        if target == DROP:
            return _DROP
        idx = self.index.get(target)
        return idx if idx is not None else _MISSING

    def walk_reaches_deliver(self, start: int, mask: int) -> bool:
        """Walk from ``start`` following the mixed table ``mask``.

        ``mask`` bit ``i`` set means switch ``i`` uses ``new_next``.
        Returns True iff DELIVER is reached within ``n`` hops without
        DROP / missing nodes / repeated nodes.
        """
        seen = 0
        node = start
        steps = 0
        n = self.n
        old_idx = self.old_idx
        new_idx = self.new_idx
        while True:
            if node == _DELIVER:
                return True
            if node < 0:  # DROP or missing node
                return False
            if (seen >> node) & 1:
                return False  # forwarding loop
            if steps >= n:
                return False  # exceeded the hop bound
            seen |= 1 << node
            steps += 1
            node = new_idx[node] if (mask >> node) & 1 else old_idx[node]

    def is_safe(self, mask: int) -> bool:
        for start in self.ingress_idx:
            if not self.walk_reaches_deliver(start, mask):
                return False
        return True


def diff_switches(switches: Sequence[dict]) -> list[str]:
    """Ids of switches whose next hop actually changes."""
    return [s["id"] for s in switches if s["old_next"] != s["new_next"]]


def full_mask(topo: Topology, diff_ids: Sequence[str]) -> int:
    mask = 0
    for sid in diff_ids:
        mask |= 1 << topo.index[sid]
    return mask


def find_safe_permutation(
    switches: Sequence[dict],
    ingresses: Sequence[str],
    diff_ids: Sequence[str],
) -> Optional[list[str]]:
    """Find the lexicographically smallest safe update permutation.

    Depth-first search with backtracking; candidates are tried in UTF-8
    byte order at every level, so the first complete permutation found is
    the lexicographically smallest one.  Sets that cannot be completed are
    memoised as dead ends.  Returns ``None`` iff no safe permutation exists
    (proven impossible by exhaustive search).
    """
    topo = Topology(switches, ingresses)
    order = sorted(diff_ids, key=utf8_key)
    bits = [1 << topo.index[d] for d in order]
    target = 0
    for b in bits:
        target |= b

    dead: set[int] = set()
    path: list[str] = []

    def dfs(mask: int) -> bool:
        if mask == target:
            return True
        for i, bit in enumerate(bits):
            if mask & bit:
                continue
            nxt = mask | bit
            if nxt in dead:
                continue
            if not topo.is_safe(nxt):
                dead.add(nxt)
                continue
            path.append(order[i])
            if dfs(nxt):
                return True
            path.pop()
            dead.add(nxt)
        return False

    return list(path) if dfs(0) else None
