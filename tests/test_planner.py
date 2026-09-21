"""Unit tests for the safety checker and the backtracking planner.

These tests are pure Python and do not need a database or the API.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.planner import (
    Topology,
    canonical_topology,
    diff_switches,
    find_safe_permutation,
    full_mask,
    stable_digest,
    utf8_key,
)


def make(switches, ingresses):
    """switches: {id: (old_next, new_next)}"""
    sw = [{"id": k, "old_next": v[0], "new_next": v[1]} for k, v in switches.items()]
    return sw, list(ingresses)


def brute_force_lex_min(switches, ingresses, diff_ids):
    """Exhaustively try every permutation; return the lex-min safe one."""
    topo = Topology(switches, ingresses)
    order = sorted(diff_ids, key=utf8_key)
    best = None
    for perm in itertools.permutations(order):
        mask = 0
        ok = True
        for sid in perm:
            mask |= 1 << topo.index[sid]
            if not topo.is_safe(mask):
                ok = False
                break
        if ok and (best is None or [utf8_key(x) for x in perm] < [utf8_key(x) for x in best]):
            best = list(perm)
    return best


# ---------------------------------------------------------------- safety

def test_safe_chain():
    sw, ing = make({"a": ("b", "b"), "b": ("DELIVER", "DELIVER")}, ["a"])
    topo = Topology(sw, ing)
    assert topo.is_safe(0)


def test_drop_is_unsafe():
    sw, ing = make({"a": ("DROP", "DROP")}, ["a"])
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


def test_missing_node_is_unsafe():
    sw, ing = make({"a": ("ghost", "ghost")}, ["a"])
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


def test_self_loop_is_unsafe_when_walked():
    sw, ing = make({"a": ("a", "a")}, ["a"])
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


def test_self_loop_on_unreached_node_is_safe():
    sw, ing = make({"a": ("DELIVER", "DELIVER"), "b": ("b", "b")}, ["a"])
    topo = Topology(sw, ing)
    assert topo.is_safe(0)


def test_two_node_loop_is_unsafe():
    sw, ing = make({"a": ("b", "b"), "b": ("a", "a")}, ["a"])
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


def test_hop_bound_exact():
    # chain of 5 switches then DELIVER: exactly 5 hops -> safe
    switches = {f"s{i}": (f"s{i+1}", f"s{i+1}") for i in range(1, 5)}
    switches["s5"] = ("DELIVER", "DELIVER")
    sw, ing = make(switches, ["s1"])
    topo = Topology(sw, ing)
    assert topo.is_safe(0)
    # chain of 6 nodes where s6 loops to s1: 6 nodes, never DELIVER
    switches["s6"] = ("s1", "s1")
    switches["s5"] = ("s6", "s6")
    sw, ing = make(switches, ["s1"])
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


def test_all_ingresses_must_be_safe():
    sw, ing = make(
        {"a": ("DELIVER", "DELIVER"), "b": ("DROP", "DROP")},
        ["a", "b"],
    )
    topo = Topology(sw, ing)
    assert not topo.is_safe(0)


# ---------------------------------------------------------------- planner

def test_planner_forces_order():
    # updating s2 before s1 creates a loop s1->s2->s1
    sw, ing = make(
        {
            "s1": ("s2", "s3"),
            "s2": ("DELIVER", "s1"),
            "s3": ("DELIVER", "DELIVER"),
        },
        ["s1"],
    )
    diffs = diff_switches(sw)
    assert diffs == ["s1", "s2"]
    assert find_safe_permutation(sw, ing, diffs) == ["s1", "s2"]


def test_planner_returns_none_when_no_safe_order():
    # initial state safe, but any update breaks the only walk.
    sw, ing = make({"a": ("DELIVER", "a")}, ["a"])
    diffs = diff_switches(sw)
    assert find_safe_permutation(sw, ing, diffs) is None


def test_planner_empty_diff():
    sw, ing = make({"a": ("DELIVER", "DELIVER")}, ["a"])
    assert find_safe_permutation(sw, ing, []) == []


def test_planner_lex_min_byte_order():
    # ids chosen so that UTF-8 byte order differs from naive expectations:
    # "Z" (0x5A) < "a" (0x61); both orders valid -> must pick "Z" first.
    sw, ing = make(
        {
            "a": ("DELIVER", "DELIVER"),
            "Z": ("DELIVER", "DELIVER"),
            "in": ("a", "Z"),
        },
        ["in"],
    )
    # only "in" is in the diff; craft a 2-diff case instead
    sw, ing = make(
        {
            "a": ("x", "DELIVER"),
            "Z": ("x", "DELIVER"),
            "x": ("DELIVER", "DELIVER"),
            "in": ("a", "a"),
        },
        ["in"],
    )
    diffs = diff_switches(sw)
    assert diffs == ["a", "Z"]
    assert find_safe_permutation(sw, ing, diffs) == ["Z", "a"]


def test_planner_matches_brute_force_random():
    rng = random.Random(20260919)
    for case in range(400):
        n = rng.randint(2, 6)
        ids = [f"s{i}" for i in range(n)]
        targets = ids + ["DELIVER", "DROP"]
        switches = {
            sid: (rng.choice(targets), rng.choice(targets)) for sid in ids
        }
        k = rng.randint(1, n)
        ingresses = rng.sample(ids, k)
        sw, ing = make(switches, ingresses)
        diffs = diff_switches(sw)
        if len(diffs) > 6:
            continue
        topo = Topology(sw, ing)
        if not topo.is_safe(0) or not topo.is_safe(full_mask(topo, diffs)):
            continue  # planner is only invoked on safe endpoints
        got = find_safe_permutation(sw, ing, diffs)
        want = brute_force_lex_min(sw, ing, diffs)
        assert got == want, f"case {case}: got {got}, want {want}, topo={switches}, ing={ingresses}"


def test_planner_scales_to_22_steps():
    # chain of 22 changed switches; every prefix must stay safe.
    n = 22
    switches = {}
    for i in range(n - 1):
        switches[f"s{i}"] = (f"s{i+1}", "DELIVER")
    # tail switch also changes, keeping both endpoints safe
    switches[f"s{n-1}"] = ("aux", "DELIVER")
    switches["aux"] = ("DELIVER", "DELIVER")
    sw, ing = make(switches, ["s0"])
    diffs = diff_switches(sw)
    assert len(diffs) == 22
    perm = find_safe_permutation(sw, ing, diffs)
    assert perm is not None
    assert sorted(perm) == sorted(diffs)
    # verify every prefix is actually safe
    topo = Topology(sw, ing)
    mask = 0
    for sid in perm:
        mask |= 1 << topo.index[sid]
        assert topo.is_safe(mask)


# ---------------------------------------------------------------- digests

def test_canonical_topology_order_independent():
    sw1 = [
        {"id": "b", "old_next": "DELIVER", "new_next": "DELIVER"},
        {"id": "a", "old_next": "b", "new_next": "DELIVER"},
    ]
    sw2 = list(reversed(sw1))
    c1 = canonical_topology(sw1, ["a"])
    c2 = canonical_topology(sw2, ["a"])
    assert c1 == c2
    assert stable_digest({"topology": c1}) == stable_digest({"topology": c2})


def test_stable_digest_format():
    assert stable_digest({"a": 1}).startswith("sha256:")
    assert len(stable_digest({"a": 1})) == len("sha256:") + 64
