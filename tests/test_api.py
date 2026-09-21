"""End-to-end acceptance tests for the migration service API.

Run inside the compose `verify` service against two API replicas sharing one
PostgreSQL database.  Every test may talk to either replica; cross-replica
tests are marked as such.
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.planner import canonical_topology, stable_digest

from conftest import (
    ack,
    advance,
    create_plan,
    create_rollout,
    drive_to_completion,
    fresh_api,
    key,
    lease,
    make_plan_and_rollout,
    simple_case,
    simple_topology,
    topo_payload,
)


def err(resp):
    return resp.json()["error"]["code"]


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_both_replicas(self, client, api1, api2):
        for base in (api1, api2):
            r = client.get(f"{base}/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
            assert r.json()["db"] == "up"


# ---------------------------------------------------------------------------
# plan validation
# ---------------------------------------------------------------------------

class TestPlanValidation:
    def test_empty_switch_list_rejected(self, client, api1):
        r = client.post(
            f"{api1}/plans",
            json={"idempotency_key": key(), "topology": {"switches": [], "ingresses": ["a"]}},
        )
        assert r.status_code == 422

    def test_too_many_switches_rejected(self, client, api1):
        switches = [{"id": f"s{i}", "old_next": "DELIVER", "new_next": "DELIVER"} for i in range(61)]
        r = client.post(
            f"{api1}/plans",
            json={"idempotency_key": key(), "topology": {"switches": switches, "ingresses": ["s0"]}},
        )
        assert r.status_code == 422

    def test_duplicate_switch_id(self, client, api1):
        topo = {
            "switches": [
                {"id": "a", "old_next": "DELIVER", "new_next": "DELIVER"},
                {"id": "a", "old_next": "DELIVER", "new_next": "DELIVER"},
            ],
            "ingresses": ["a"],
        }
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "DUPLICATE_SWITCH_ID"

    def test_reserved_switch_id(self, client, api1):
        topo = {
            "switches": [{"id": "DELIVER", "old_next": "DELIVER", "new_next": "DELIVER"}],
            "ingresses": ["DELIVER"],
        }
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "RESERVED_SWITCH_ID"

    def test_unknown_next_hop(self, client, api1):
        topo = {
            "switches": [{"id": "a", "old_next": "ghost", "new_next": "DELIVER"}],
            "ingresses": ["a"],
        }
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "UNKNOWN_NEXT_HOP"

    def test_unknown_ingress(self, client, api1):
        topo = {
            "switches": [{"id": "a", "old_next": "DELIVER", "new_next": "DELIVER"}],
            "ingresses": ["ghost"],
        }
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "UNKNOWN_INGRESS"

    def test_unsafe_initial_state(self, client, api1):
        topo = topo_payload({"a": ("a", "DELIVER")}, ["a"])  # self loop
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "UNSAFE_INITIAL_STATE"

    def test_unsafe_final_state(self, client, api1):
        topo = topo_payload({"a": ("DELIVER", "DROP")}, ["a"])
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "UNSAFE_FINAL_STATE"

    def test_diff_too_large(self, client, api1):
        switches = {f"s{i}": (f"s{i+1}", "DELIVER") for i in range(23)}
        switches["s23"] = ("DELIVER", "DELIVER")
        topo = topo_payload(switches, ["s0"])
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "DIFF_TOO_LARGE"

    def test_malformed_body(self, client, api1):
        r = client.post(f"{api1}/plans", json={"idempotency_key": key()})
        assert r.status_code == 422
        assert err(r) == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# plan creation / idempotency
# ---------------------------------------------------------------------------

class TestPlans:
    def test_create_and_get_plan(self, client, api1):
        topology, expected_order = simple_case()
        idem = key()
        r = client.post(f"{api1}/plans", json={"idempotency_key": idem, "topology": topology})
        assert r.status_code == 201
        plan = r.json()
        assert plan["status"] == "completed"
        # the only safe order for the simple topology is s1 before s2
        assert plan["permutation"] == expected_order
        # digest is stable and recomputed from canonical inputs
        canonical = canonical_topology(
            topology["switches"], topology["ingresses"]
        )
        assert plan["plan_digest"] == stable_digest(
            {"permutation": expected_order, "topology": canonical}
        )

        r = client.get(f"{api1}/plans/{plan['id']}")
        assert r.status_code == 200
        assert r.json() == plan

    def test_plan_idempotent_replay_same_params(self, client, api1):
        topology = simple_topology()
        idem = key()
        r1 = client.post(f"{api1}/plans", json={"idempotency_key": idem, "topology": topology})
        assert r1.status_code == 201
        # same key, semantically identical topology in different order
        shuffled = {
            "switches": list(reversed(topology["switches"])),
            "ingresses": list(topology["ingresses"]),
        }
        r2 = client.post(f"{api1}/plans", json={"idempotency_key": idem, "topology": shuffled})
        assert r2.status_code == 200
        assert r2.json()["id"] == r1.json()["id"]
        assert r2.json()["plan_digest"] == r1.json()["plan_digest"]

    def test_plan_idempotency_conflict(self, client, api1):
        idem = key()
        r1 = client.post(
            f"{api1}/plans", json={"idempotency_key": idem, "topology": simple_topology()}
        )
        assert r1.status_code == 201
        other = topo_payload({"a": ("DELIVER", "DROP"), "b": ("DELIVER", "DELIVER")}, ["b"])
        r2 = client.post(f"{api1}/plans", json={"idempotency_key": idem, "topology": other})
        assert r2.status_code == 409
        assert err(r2) == "IDEMPOTENCY_CONFLICT"

    def test_plan_concurrent_same_key_across_replicas(self, client, api1, api2):
        topology = simple_topology()
        idem = key()

        def post(base):
            with httpx.Client(timeout=10) as c:
                return c.post(f"{base}/plans", json={"idempotency_key": idem, "topology": topology})

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(post, [api1, api2] * 4))
        ids = {r.json()["id"] for r in results}
        assert len(ids) == 1
        assert all(r.status_code in (200, 201) for r in results)
        digests = {r.json()["plan_digest"] for r in results}
        assert len(digests) == 1

    def test_plan_not_found(self, client, api1):
        r = client.get(f"{api1}/plans/{uuid.uuid4()}")
        assert r.status_code == 404
        assert err(r) == "PLAN_NOT_FOUND"

    def test_plan_inputs_immutable_no_update_endpoint(self, client, api1):
        plan = create_plan(client, api1).json()
        assert client.put(f"{api1}/plans/{plan['id']}", json={}).status_code == 405
        assert client.patch(f"{api1}/plans/{plan['id']}", json={}).status_code == 405
        assert client.delete(f"{api1}/plans/{plan['id']}").status_code == 405


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

class TestRollouts:
    def test_create_rollout_and_status(self, client, api1):
        plan, rollout = make_plan_and_rollout(client, api1)
        assert rollout["status"] == "in_progress"
        assert rollout["total_steps"] == 2
        assert rollout["issued_count"] == 0
        assert rollout["plan_digest"] == plan["plan_digest"]
        assert rollout["lease"] is None
        assert rollout["commands"] == []

    def test_rollout_idempotent_replay(self, client, api1):
        plan = create_plan(client, api1).json()
        idem = key()
        r1 = create_rollout(client, api1, plan["id"], idem=idem)
        assert r1.status_code == 201
        r2 = create_rollout(client, api1, plan["id"], idem=idem)
        assert r2.status_code == 200
        assert r2.json()["id"] == r1.json()["id"]

    def test_rollout_idempotency_conflict(self, client, api1):
        p1 = create_plan(client, api1).json()
        p2 = create_plan(client, api1).json()
        idem = key()
        assert create_rollout(client, api1, p1["id"], idem=idem).status_code == 201
        r = create_rollout(client, api1, p2["id"], idem=idem)
        assert r.status_code == 409
        assert err(r) == "IDEMPOTENCY_CONFLICT"

    def test_rollout_for_missing_plan(self, client, api1):
        r = create_rollout(client, api1, str(uuid.uuid4()))
        assert r.status_code == 404
        assert err(r) == "PLAN_NOT_FOUND"

    def test_rollout_not_found(self, client, api1):
        r = client.get(f"{api1}/rollouts/{uuid.uuid4()}")
        assert r.status_code == 404
        assert err(r) == "ROLLOUT_NOT_FOUND"


# ---------------------------------------------------------------------------
# lease / epochs
# ---------------------------------------------------------------------------

class TestLease:
    def test_acquire_renew_takeover(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]

        r = lease(client, api1, rid, "c1", ttl=30)
        assert r.status_code == 200
        body = r.json()
        assert body["epoch"] == 1
        assert body["op_type"] == "acquire"
        assert body["holder"] == "c1"

        # idempotent replay of the same op returns the recorded result
        op = key()
        r1 = lease(client, api1, rid, "c1", ttl=30, op=op)
        r2 = lease(client, api1, rid, "c1", ttl=30, op=op)
        assert r1.json() == r2.json()

        # renew keeps the epoch
        r = lease(client, api1, rid, "c1", ttl=30)
        assert r.json()["epoch"] == 1
        assert r.json()["op_type"] == "renew"

        # another coordinator cannot take a live lease
        r = lease(client, api1, rid, "c2", ttl=30)
        assert r.status_code == 409
        assert err(r) == "LEASE_HELD"

    def test_lease_op_replay(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        op = key()
        r1 = lease(client, api1, rid, "c1", ttl=30, op=op)
        r2 = lease(client, api1, rid, "c1", ttl=30, op=op)
        assert r1.json() == r2.json()
        # same op_id with different parameters conflicts
        r3 = lease(client, api1, rid, "c1", ttl=31, op=op)
        assert r3.status_code == 409
        assert err(r3) == "IDEMPOTENCY_CONFLICT"

    def test_ttl_bounds(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        assert lease(client, api1, rid, "c1", ttl=4).status_code == 422
        assert lease(client, api1, rid, "c1", ttl=61).status_code == 422
        assert lease(client, api1, rid, "c1", ttl=5).status_code == 200
        assert lease(client, api1, rid, "c1", ttl=60).status_code == 200

    def test_takeover_assigns_monotonic_epochs(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        epochs = []
        holder = "c1"
        for i in range(3):
            r = lease(client, api1, rid, holder, ttl=5)
            assert r.status_code == 200
            epochs.append(r.json()["epoch"])
            holder = f"c{i+2}"
            time.sleep(6)  # let the lease expire (database clock)
        assert epochs == [1, 2, 3]

    def test_stale_epoch_cannot_advance(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        r = lease(client, api1, rid, "c1", ttl=5)
        old_epoch = r.json()["epoch"]
        time.sleep(6)
        r = lease(client, api1, rid, "c2", ttl=30)
        new_epoch = r.json()["epoch"]
        assert new_epoch > old_epoch

        # old holder with old epoch
        r = advance(client, api1, rid, "c1", old_epoch)
        assert r.status_code == 409
        assert err(r) in ("STALE_EPOCH", "NOT_LEASE_HOLDER")
        # old holder even with the new epoch number
        r = advance(client, api1, rid, "c1", new_epoch)
        assert r.status_code == 409
        assert err(r) == "NOT_LEASE_HOLDER"
        # current holder with the old epoch number
        r = advance(client, api1, rid, "c2", old_epoch)
        assert r.status_code == 409
        assert err(r) == "STALE_EPOCH"
        # current holder, current epoch works
        r = advance(client, api1, rid, "c2", new_epoch)
        assert r.status_code == 200

    def test_expired_lease_cannot_advance(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        r = lease(client, api1, rid, "c1", ttl=5)
        epoch = r.json()["epoch"]
        time.sleep(6)
        r = advance(client, api1, rid, "c1", epoch)
        assert r.status_code == 409
        assert err(r) == "LEASE_EXPIRED"

    def test_advance_without_lease(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        r = advance(client, api1, rollout["id"], "c1", 1)
        assert r.status_code == 409
        assert err(r) == "NO_ACTIVE_LEASE"


# ---------------------------------------------------------------------------
# advance / commands
# ---------------------------------------------------------------------------

class TestAdvance:
    def test_full_rollout_flow(self, client, api1):
        topology, order = simple_case()
        s1, s2 = order
        plan = create_plan(client, api1, topology).json()
        rollout = create_rollout(client, api1, plan["id"]).json()
        rid = rollout["id"]
        r = lease(client, api1, rid, "c1", ttl=30)
        epoch = r.json()["epoch"]

        # step 0
        r = advance(client, api1, rid, "c1", epoch)
        assert r.status_code == 200
        cmd0 = r.json()["command"]
        assert cmd0["step"] == 0
        assert cmd0["switch_id"] == s1
        assert cmd0["device_generation"] == 1
        assert cmd0["plan_digest"] == plan["plan_digest"]
        assert cmd0["status"] == "PENDING"

        # advancing again while pending returns the very same command
        r = advance(client, api1, rid, "c1", epoch)
        assert r.status_code == 200
        assert r.json()["command"] == cmd0

        # switch polls its pending command
        r = client.get(f"{api1}/switches/{s1}/pending-commands")
        assert r.status_code == 200
        assert [c["command_id"] for c in r.json()["commands"]] == [cmd0["command_id"]]

        # ack step 0
        r = ack(client, api1, rid, cmd0)
        assert r.status_code == 200
        assert r.json()["status"] == "APPLIED"
        assert r.json()["rollout_status"] == "in_progress"

        # step 1
        r = advance(client, api1, rid, "c1", epoch)
        cmd1 = r.json()["command"]
        assert cmd1["step"] == 1
        assert cmd1["switch_id"] == s2
        assert cmd1["device_generation"] == 1  # per-device generations

        r = ack(client, api1, rid, cmd1)
        assert r.json()["rollout_status"] == "completed"

        # rollout is completed now
        r = client.get(f"{api1}/rollouts/{rid}")
        assert r.json()["status"] == "completed"
        assert r.json()["completed_at"] is not None
        assert [c["status"] for c in r.json()["commands"]] == ["APPLIED", "APPLIED"]

        # no further progress possible
        r = advance(client, api1, rid, "c1", epoch)
        assert r.status_code == 409
        assert err(r) == "ROLLOUT_COMPLETED"
        r = lease(client, api1, rid, "c1", ttl=30)
        assert r.status_code == 409
        assert err(r) == "ROLLOUT_COMPLETED"

    def test_advance_op_replay(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
        op = key()
        r1 = advance(client, api1, rid, "c1", epoch, op=op)
        r2 = advance(client, api1, rid, "c1", epoch, op=op)
        assert r1.json() == r2.json()
        # same op id, different parameters -> conflict
        r3 = client.post(
            f"{api1}/rollouts/{rid}/advance",
            json={"coordinator_id": "c2", "epoch": epoch, "op_id": op},
        )
        assert r3.status_code == 409
        assert err(r3) == "IDEMPOTENCY_CONFLICT"

    def test_command_survives_takeover_unchanged(self, client, api1):
        """A lost response must never cause a new command id or generation."""
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        e1 = lease(client, api1, rid, "c1", ttl=5).json()["epoch"]
        r = advance(client, api1, rid, "c1", e1)
        cmd = r.json()["command"]

        time.sleep(6)  # c1's lease expires; c1 is considered crashed
        e2 = lease(client, api1, rid, "c2", ttl=30).json()["epoch"]
        assert e2 > e1
        r = advance(client, api1, rid, "c2", e2)
        assert r.status_code == 200
        again = r.json()["command"]
        assert again["command_id"] == cmd["command_id"]
        assert again["device_generation"] == cmd["device_generation"]
        assert again["step"] == cmd["step"]

    def test_concurrent_advance_single_command(self, client, api1, api2):
        """Hammering advance from both replicas creates exactly one command."""
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]


        def call(i):
            base = api1 if i % 2 == 0 else api2
            with httpx.Client(timeout=10) as c:
                return c.post(
                    f"{base}/rollouts/{rid}/advance",
                    json={"coordinator_id": "c1", "epoch": epoch, "op_id": key()},
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(call, range(10)))
        assert all(r.status_code == 200 for r in results)
        ids = {r.json()["command"]["command_id"] for r in results}
        assert len(ids) == 1
        status = client.get(f"{api1}/rollouts/{rid}").json()
        assert len(status["commands"]) == 1
        assert status["issued_count"] == 1

    def test_generations_increase_per_device_across_rollouts(self, client, api1):
        p = uuid.uuid4().hex[:10]
        sw, aux = f"{p}-sw", f"{p}-aux"
        topo = topo_payload({sw: ("DELIVER", "DROP"), aux: ("DELIVER", "DELIVER")}, [aux])
        gens = []
        cmds = []
        for _ in range(2):
            plan = create_plan(client, api1, topo).json()
            rollout = create_rollout(client, api1, plan["id"]).json()
            rid = rollout["id"]
            epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
            cmd = advance(client, api1, rid, "c1", epoch).json()["command"]
            gens.append(cmd["device_generation"])
            cmds.append((rid, cmd))
        assert gens == [1, 2]

        # ack the newer generation first: accepted
        rid2, cmd2 = cmds[1]
        r = ack(client, api1, rid2, cmd2)
        assert r.status_code == 200
        # the older, late acknowledgement is now stale and must be rejected
        rid1, cmd1 = cmds[0]
        r = ack(client, api1, rid1, cmd1)
        assert r.status_code == 409
        assert err(r) == "STALE_GENERATION"
        # and it did not change state
        status = client.get(f"{api1}/rollouts/{rid1}").json()
        assert status["commands"][0]["status"] == "PENDING"


# ---------------------------------------------------------------------------
# device generation survives rollout completion and service restarts
# ---------------------------------------------------------------------------

class TestDeviceGenerationAcrossRollouts:
    """A single switch may execute the same plan in several rollouts.

    Its ``device_generation`` must keep increasing across rollouts, idle
    periods and API restarts -- the switch uses it to tell new commands from
    old ones.
    """

    @staticmethod
    def _migration_topology(switch_id: str) -> dict:
        # The ingress flips old-path -> new-path; both exit switches lead
        # straight to DELIVER, so the only planned step is the ingress flip.
        return topo_payload(
            {
                switch_id: ("old-path", "new-path"),
                "old-path": ("DELIVER", "DELIVER"),
                "new-path": ("DELIVER", "DELIVER"),
            },
            [switch_id],
        )

    def _create_plan(self, client, base, switch_id):
        r = client.post(
            f"{base}/plans",
            json={
                "idempotency_key": key(),
                "topology": self._migration_topology(switch_id),
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["permutation"] == [switch_id]
        assert body["status"] == "completed"
        return body

    def _run_first_rollout(self, client, base, plan, coordinator="coord-A"):
        """Create rollout A, lease+advance, ack, complete.  Returns its command."""
        rollout = create_rollout(client, base, plan["id"])
        assert rollout.status_code == 201, rollout.text
        rid = rollout.json()["id"]
        epoch = lease(client, base, rid, coordinator, ttl=30).json()["epoch"]
        op = key()
        r = advance(client, base, rid, coordinator, epoch, op=op)
        assert r.status_code == 200, r.text
        cmd = r.json()["command"]
        assert cmd["step"] == 0
        assert cmd["switch_id"] == plan["permutation"][0]
        assert cmd["device_generation"] == 1
        # retrying the same advance op replays the *same* command id + generation
        r2 = advance(client, base, rid, coordinator, epoch, op=op)
        assert r2.status_code == 200
        assert r2.json()["command"]["command_id"] == cmd["command_id"]
        assert r2.json()["command"]["device_generation"] == 1
        r = ack(client, base, rid, cmd)
        assert r.status_code == 200, r.text
        assert r.json()["rollout_status"] == "completed"
        status = client.get(f"{base}/rollouts/{rid}").json()
        assert status["status"] == "completed"
        return rid, cmd

    def test_sequential_rollouts_same_plan_across_replicas(self, client, api1, api2):
        """A completes on api1; B (same plan) is driven via api2 -> gen 2."""
        switch = f"edge-repeat-{uuid.uuid4().hex[:8]}"
        plan = self._create_plan(client, api1, switch)

        # rollout A via api1
        rid_a, cmd_a = self._run_first_rollout(client, api1, plan, "coord-A")
        assert cmd_a["device_generation"] == 1

        # rollout B, same plan, leased/advanced through the *other* replica
        rb = create_rollout(client, api2, plan["id"])
        assert rb.status_code == 201, rb.text
        rid_b = rb.json()["id"]
        assert rid_b != rid_a
        epoch_b = lease(client, api2, rid_b, "coord-B", ttl=30).json()["epoch"]
        op_b = key()
        r = advance(client, api2, rid_b, "coord-B", epoch_b, op=op_b)
        assert r.status_code == 200, r.text
        cmd_b = r.json()["command"]
        assert cmd_b["device_generation"] == 2
        assert cmd_b["command_id"] != cmd_a["command_id"]
        assert cmd_b["switch_id"] == switch

        # same advance op retried on either replica returns the original
        # command id and generation (no extra issuance)
        for base in (api2, api1):
            r = advance(client, base, rid_b, "coord-B", epoch_b, op=op_b)
            assert r.status_code == 200
            assert r.json()["command"]["command_id"] == cmd_b["command_id"]
            assert r.json()["command"]["device_generation"] == 2

        # the switch can poll and acknowledge the second command
        pending = client.get(f"{api1}/switches/{switch}/pending-commands").json()
        assert [c["command_id"] for c in pending["commands"]] == [cmd_b["command_id"]]
        r = ack(client, api1, rid_b, cmd_b)
        assert r.status_code == 200
        assert r.json()["rollout_status"] == "completed"

        # the two consecutive commands carry generations [1, 2]
        assert [cmd_a["device_generation"], cmd_b["device_generation"]] == [1, 2]

        # each rollout's audit trail contains only its own create + command
        for rid, cid in ((rid_a, cmd_a["command_id"]), (rid_b, cmd_b["command_id"])):
            events = client.get(f"{api1}/rollouts/{rid}/audit").json()["events"]
            created = [e for e in events if e["event_type"] == "COMMAND_CREATED"]
            applied = [e for e in events if e["event_type"] == "COMMAND_APPLIED"]
            assert len(created) == 1
            assert len(applied) == 1
            assert created[0]["payload"]["command_id"] == cid
            assert applied[0]["payload"]["command_id"] == cid
            assert client.get(f"{api1}/rollouts/{rid}").json()["status"] == "completed"

        # a third rollout keeps climbing: generation 3
        rc = create_rollout(client, api1, plan["id"]).json()
        epoch_c = lease(client, api1, rc["id"], "coord-C", ttl=30).json()["epoch"]
        cmd_c = advance(client, api1, rc["id"], "coord-C", epoch_c).json()["command"]
        assert cmd_c["device_generation"] == 3
        assert ack(client, api2, rc["id"], cmd_c).json()["rollout_status"] == "completed"

    def test_generation_survives_api_restart(self, client, api1, api2):
        """After A completes, a cold-started API instance must issue gen 2."""
        switch = f"edge-repeat-{uuid.uuid4().hex[:8]}"
        plan = self._create_plan(client, api1, switch)
        rid_a, cmd_a = self._run_first_rollout(client, api1, plan, "coord-A")
        assert cmd_a["device_generation"] == 1

        # cold-start a completely fresh API process on the shared database
        with fresh_api() as api3:
            assert client.get(f"{api3}/health").status_code == 200
            rb = create_rollout(client, api3, plan["id"])
            assert rb.status_code == 201, rb.text
            rid_b = rb.json()["id"]
            epoch_b = lease(client, api3, rid_b, "coord-B", ttl=30).json()["epoch"]
            op_b = key()
            r = advance(client, api3, rid_b, "coord-B", epoch_b, op=op_b)
            assert r.status_code == 200, r.text
            cmd_b = r.json()["command"]
            assert cmd_b["device_generation"] == 2
            assert cmd_b["command_id"] != cmd_a["command_id"]

            # retried advance (including from a different replica) stays put
            for base in (api3, api2):
                r = advance(client, base, rid_b, "coord-B", epoch_b, op=op_b)
                assert r.status_code == 200
                assert r.json()["command"]["command_id"] == cmd_b["command_id"]
                assert r.json()["command"]["device_generation"] == 2

            # second command is queryable and ackable through the other replica
            pending = client.get(f"{api2}/switches/{switch}/pending-commands").json()
            assert [c["command_id"] for c in pending["commands"]] == [cmd_b["command_id"]]
            r = ack(client, api2, rid_b, cmd_b)
            assert r.status_code == 200
            assert r.json()["rollout_status"] == "completed"

        assert [cmd_a["device_generation"], cmd_b["device_generation"]] == [1, 2]
        # clean, per-rollout audit trails
        for rid, cid in ((rid_a, cmd_a["command_id"]), (rid_b, cmd_b["command_id"])):
            events = client.get(f"{api1}/rollouts/{rid}/audit").json()["events"]
            assert [e["event_type"] for e in events].count("COMMAND_CREATED") == 1
            assert [e["event_type"] for e in events].count("COMMAND_APPLIED") == 1
            assert all(
                e["payload"].get("command_id") == cid
                for e in events
                if e["event_type"] in ("COMMAND_CREATED", "COMMAND_APPLIED")
            )

        # restart again, then another rollout: generation keeps climbing
        with fresh_api() as api3:
            rd = create_rollout(client, api3, plan["id"]).json()
            epoch_d = lease(client, api3, rd["id"], "coord-D", ttl=30).json()["epoch"]
            cmd_d = advance(client, api3, rd["id"], "coord-D", epoch_d).json()["command"]
            assert cmd_d["device_generation"] == 3
            assert ack(client, api3, rd["id"], cmd_d).json()["rollout_status"] == "completed"


# ---------------------------------------------------------------------------
# acknowledgements
# ---------------------------------------------------------------------------

class TestAcks:
    def _one_step_pending(self, client, api1):
        plan, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
        cmd = advance(client, api1, rid, "c1", epoch).json()["command"]
        return plan, rid, cmd

    def test_ack_idempotent_replay(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        r1 = ack(client, api1, rid, cmd)
        assert r1.status_code == 200
        r2 = ack(client, api1, rid, cmd)
        assert r2.status_code == 200
        assert r2.json()["duplicate"] is True
        assert r2.json()["acked_at"] == r1.json()["acked_at"]

    def test_ack_unknown_command(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        r = client.post(
            f"{api1}/rollouts/{rollout['id']}/acks",
            json={
                "command_id": "nope",
                "switch_id": "s1",
                "step": 0,
                "plan_digest": "sha256:x",
                "device_generation": 1,
            },
        )
        assert r.status_code == 404
        assert err(r) == "COMMAND_NOT_FOUND"

    def test_ack_switch_mismatch(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        bad = dict(cmd, switch_id="s2")
        r = ack(client, api1, rid, bad)
        assert r.status_code == 409
        assert err(r) == "SWITCH_MISMATCH"

    def test_ack_step_mismatch(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        bad = dict(cmd, step=cmd["step"] + 1)
        r = ack(client, api1, rid, bad)
        assert r.status_code == 409
        assert err(r) == "STEP_MISMATCH"

    def test_ack_digest_mismatch(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        bad = dict(cmd, plan_digest="sha256:" + "0" * 64)
        r = ack(client, api1, rid, bad)
        assert r.status_code == 409
        assert err(r) == "PLAN_DIGEST_MISMATCH"

    def test_ack_generation_mismatch(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        bad = dict(cmd, device_generation=cmd["device_generation"] + 1)
        r = ack(client, api1, rid, bad)
        assert r.status_code == 409
        assert err(r) == "GENERATION_MISMATCH"

    def test_ack_wrong_rollout(self, client, api1):
        _, rid, cmd = self._one_step_pending(client, api1)
        _, rollout2 = make_plan_and_rollout(client, api1)
        r = ack(client, api1, rollout2["id"], cmd)
        assert r.status_code == 409
        assert err(r) == "ROLLOUT_MISMATCH"

    def test_late_ack_converges_after_takeover(self, client, api1):
        """Acks do not depend on the coordinator lease being alive."""
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        e1 = lease(client, api1, rid, "c1", ttl=5).json()["epoch"]
        cmd = advance(client, api1, rid, "c1", e1).json()["command"]
        time.sleep(6)  # lease expires; coordinator is considered gone
        # the switch's late acknowledgement is still accepted
        r = ack(client, api1, rid, cmd)
        assert r.status_code == 200
        # a new coordinator takes over and continues the plan
        e2 = lease(client, api1, rid, "c2", ttl=30).json()["epoch"]
        r = advance(client, api1, rid, "c2", e2)
        assert r.status_code == 200
        assert r.json()["command"]["step"] == 1

    def test_ack_does_not_create_next_command(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
        cmd = advance(client, api1, rid, "c1", epoch).json()["command"]
        ack(client, api1, rid, cmd)
        status = client.get(f"{api1}/rollouts/{rid}").json()
        assert len(status["commands"]) == 1  # ack alone never issues the next step


# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------

class TestAudit:
    def test_audit_trail(self, client, api1):
        topology, order = simple_case()
        plan = create_plan(client, api1, topology).json()
        rollout = create_rollout(client, api1, plan["id"]).json()
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
        drive_to_completion(client, api1, rid)

        r = client.get(f"{api1}/rollouts/{rid}/audit")
        assert r.status_code == 200
        events = r.json()["events"]
        types = [e["event_type"] for e in events]
        assert types[0] == "ROLLOUT_CREATED"
        assert "LEASE_ACQUIRED" in types
        assert types.count("COMMAND_CREATED") == 2
        assert types.count("COMMAND_APPLIED") == 2
        assert types[-1] == "ROLLOUT_COMPLETED"
        # ids are strictly increasing (ordered history)
        ids = [e["id"] for e in events]
        assert ids == sorted(ids)
        created = [e for e in events if e["event_type"] == "COMMAND_CREATED"]
        assert created[0]["payload"]["switch_id"] == order[0]
        assert created[0]["payload"]["epoch"] == epoch

    def test_audit_records_takeovers(self, client, api1):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]
        lease(client, api1, rid, "c1", ttl=5)
        time.sleep(6)
        lease(client, api1, rid, "c2", ttl=30)
        events = client.get(f"{api1}/rollouts/{rid}/audit").json()["events"]
        types = [e["event_type"] for e in events]
        assert "LEASE_ACQUIRED" in types
        assert "LEASE_TAKEN_OVER" in types
        takeover = [e for e in events if e["event_type"] == "LEASE_TAKEN_OVER"][0]
        assert takeover["payload"]["holder"] == "c2"
        assert takeover["payload"]["epoch"] == 2


# ---------------------------------------------------------------------------
# cross-replica behaviour
# ---------------------------------------------------------------------------

class TestCrossReplica:
    def test_full_flow_split_across_replicas(self, client, api1, api2):
        # plan on api1, read it on api2
        plan = create_plan(client, api1).json()
        assert client.get(f"{api2}/plans/{plan['id']}").json()["id"] == plan["id"]
        # rollout on api2, lease on api1
        rollout = create_rollout(client, api2, plan["id"]).json()
        rid = rollout["id"]
        epoch = lease(client, api1, rid, "c1", ttl=30).json()["epoch"]
        # advance on api2, poll on api1, ack on api2
        cmd = advance(client, api2, rid, "c1", epoch).json()["command"]
        pending = client.get(f"{api1}/switches/{cmd['switch_id']}/pending-commands").json()
        assert pending["commands"][0]["command_id"] == cmd["command_id"]
        r = ack(client, api2, rid, cmd)
        assert r.status_code == 200
        # status visible on api1
        status = client.get(f"{api1}/rollouts/{rid}").json()
        assert status["issued_count"] == 1

    def test_lease_contention_across_replicas(self, client, api1, api2):
        _, rollout = make_plan_and_rollout(client, api1)
        rid = rollout["id"]


        def acquire(base, coordinator):
            with httpx.Client(timeout=10) as c:
                return c.post(
                    f"{base}/rollouts/{rid}/lease",
                    json={"coordinator_id": coordinator, "ttl_seconds": 30, "op_id": key()},
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = []
            for i in range(8):
                base = api1 if i % 2 == 0 else api2
                futs.append(pool.submit(acquire, base, f"coord-{i}"))
            results = [f.result() for f in futs]
        winners = [r for r in results if r.status_code == 200]
        losers = [r for r in results if r.status_code == 409]
        assert len(winners) == 1
        assert all(err(r) == "LEASE_HELD" for r in losers)
        status = client.get(f"{api1}/rollouts/{rid}").json()
        assert status["lease"]["epoch"] == 1

    def test_concurrent_plan_creation_both_replicas(self, client, api1, api2):
        topology = simple_topology()
        idem = key()


        def post(base):
            with httpx.Client(timeout=10) as c:
                return c.post(f"{base}/plans", json={"idempotency_key": idem, "topology": topology})

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(post, [api1, api2, api1, api2, api1, api2]))
        assert len({r.json()["id"] for r in results}) == 1


# ---------------------------------------------------------------------------
# multi-ingress topologies
# ---------------------------------------------------------------------------

class TestMultiIngress:
    def test_all_ingresses_checked(self, client, api1):
        # safe for ingress i1, unsafe for ingress i2 initially
        topo = topo_payload(
            {
                "i1": ("a", "a"),
                "i2": ("b", "b"),
                "a": ("DELIVER", "DELIVER"),
                "b": ("i2", "i2"),
            },
            ["i1", "i2"],
        )
        r = client.post(f"{api1}/plans", json={"idempotency_key": key(), "topology": topo})
        assert r.status_code == 422
        assert err(r) == "UNSAFE_INITIAL_STATE"

    def test_multi_ingress_plan_and_run(self, client, api1):
        topo = topo_payload(
            {
                "i1": ("x", "x"),
                "i2": ("y", "y"),
                "x": ("y", "DELIVER"),
                "y": ("DELIVER", "x"),
            },
            ["i1", "i2"],
        )
        plan = create_plan(client, api1, topo).json()
        assert plan["status"] == "completed"
        assert sorted(plan["permutation"]) == ["x", "y"]
        rollout = create_rollout(client, api1, plan["id"]).json()
        drive_to_completion(client, api1, rollout["id"])
        status = client.get(f"{api1}/rollouts/{rollout['id']}").json()
        assert status["status"] == "completed"
