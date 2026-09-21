"""Business logic: plans, rollouts, leases, commands and acknowledgements.

Every state transition happens inside a single PostgreSQL transaction,
serialised by row-level locks (``SELECT ... FOR UPDATE``) and guarded by
unique constraints, so the invariants hold for any number of API replicas
sharing one database.  Lease expiry is evaluated with the database clock
(``now()``); application clocks are never trusted.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .errors import ApiError, conflict, not_found, unprocessable
from .planner import (
    MAX_PLANNED_STEPS,
    RESERVED_TARGETS,
    Topology,
    canonical_topology,
    diff_switches,
    find_safe_permutation,
    full_mask,
    stable_digest,
)

# Fixed namespace for deterministic command ids.
_COMMAND_NS = uuid.UUID("3f8a2c4e-7b1d-4e5f-9a6b-2c8d0e1f3a4b")


@contextmanager
def _tx():
    """A database transaction with dict rows."""
    with db.pool().connection() as conn:
        conn.row_factory = dict_row
        with conn.transaction():
            yield conn


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def _plan_view(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "idempotency_key": row["idempotency_key"],
        "status": row["status"],
        "permutation": row["permutation"],
        "plan_digest": row["plan_digest"],
        "topology": row["topology"],
        "diff_switches": row["diff_switches"],
        "created_at": row["created_at"],
    }


def _command_view(row: dict) -> dict:
    return {
        "command_id": row["command_id"],
        "rollout_id": str(row["rollout_id"]),
        "step": row["step"],
        "switch_id": row["switch_id"],
        "plan_digest": row["plan_digest"],
        "device_generation": row["device_generation"],
        "status": row["status"],
        "created_at": row["created_at"],
        "acked_at": row["acked_at"],
    }


def _lease_view(rollout_id: str, holder: str, epoch: int, expires_at, op_type: str, server_time) -> dict:
    return {
        "rollout_id": rollout_id,
        "holder": holder,
        "epoch": epoch,
        "expires_at": expires_at,
        "op_type": op_type,
        "server_time": server_time,
    }


def _parse_uuid(value: str, code: str, message: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise not_found(code, message)


# ---------------------------------------------------------------------------
# topology validation
# ---------------------------------------------------------------------------

def _validate_topology(switches: list[dict], ingresses: list[str]) -> list[str]:
    ids = [s["id"] for s in switches]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise unprocessable("DUPLICATE_SWITCH_ID", "switch ids must be unique", {"duplicates": dupes})
    reserved = [i for i in ids if i in RESERVED_TARGETS]
    if reserved:
        raise unprocessable(
            "RESERVED_SWITCH_ID",
            "switch ids must not collide with reserved hop targets",
            {"reserved": reserved, "reserved_targets": list(RESERVED_TARGETS)},
        )
    id_set = set(ids)
    allowed = id_set | set(RESERVED_TARGETS)
    for s in switches:
        for field in ("old_next", "new_next"):
            if s[field] not in allowed:
                raise unprocessable(
                    "UNKNOWN_NEXT_HOP",
                    f"next hop {s[field]!r} of switch {s['id']!r} is not a declared "
                    f"switch, DELIVER or DROP",
                    {"switch_id": s["id"], "field": field, "target": s[field]},
                )
    if len(set(ingresses)) != len(ingresses):
        raise unprocessable("DUPLICATE_INGRESS", "ingresses must be unique")
    for ing in ingresses:
        if ing not in id_set:
            raise unprocessable(
                "UNKNOWN_INGRESS",
                f"ingress {ing!r} is not a declared switch",
                {"ingress": ing},
            )
    diffs = diff_switches(switches)
    if len(diffs) > MAX_PLANNED_STEPS:
        raise unprocessable(
            "DIFF_TOO_LARGE",
            f"{len(diffs)} switches change their next hop; at most "
            f"{MAX_PLANNED_STEPS} can be planned",
            {"limit": MAX_PLANNED_STEPS, "actual": len(diffs)},
        )
    return diffs


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

def create_plan(payload) -> tuple[dict, int]:
    switches = [s.model_dump() for s in payload.topology.switches]
    ingresses = list(payload.topology.ingresses)
    diffs = _validate_topology(switches, ingresses)

    topo = Topology(switches, ingresses)
    if not topo.is_safe(0):
        raise unprocessable(
            "UNSAFE_INITIAL_STATE",
            "the initial forwarding state does not reach DELIVER from every ingress",
        )
    if not topo.is_safe(full_mask(topo, diffs)):
        raise unprocessable(
            "UNSAFE_FINAL_STATE",
            "the final forwarding state does not reach DELIVER from every ingress",
        )

    canonical = canonical_topology(switches, ingresses)
    request_hash = stable_digest({"topology": canonical})
    permutation = find_safe_permutation(switches, ingresses, diffs)
    status = "completed" if permutation is not None else "proven_impossible"
    plan_digest = stable_digest({"permutation": permutation, "topology": canonical})

    with _tx() as conn:
        row = conn.execute(
            """
            INSERT INTO plans (idempotency_key, request_hash, topology, diff_switches,
                               status, permutation, plan_digest)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING *
            """,
            (
                payload.idempotency_key,
                request_hash,
                Jsonb(canonical),
                Jsonb(diffs),
                status,
                Jsonb(permutation) if permutation is not None else None,
                plan_digest,
            ),
        ).fetchone()
        if row is not None:
            return _plan_view(row), 201

        existing = conn.execute(
            "SELECT * FROM plans WHERE idempotency_key = %s", (payload.idempotency_key,)
        ).fetchone()
        if existing["request_hash"] != request_hash:
            raise conflict(
                "IDEMPOTENCY_CONFLICT",
                "idempotency_key was already used with different plan parameters",
                {"idempotency_key": payload.idempotency_key},
            )
        return _plan_view(existing), 200


def get_plan(plan_id: str) -> dict:
    plan_id = _parse_uuid(plan_id, "PLAN_NOT_FOUND", "plan not found")
    with _tx() as conn:
        row = conn.execute("SELECT * FROM plans WHERE id = %s", (plan_id,)).fetchone()
    if row is None:
        raise not_found("PLAN_NOT_FOUND", "plan not found", {"plan_id": plan_id})
    return _plan_view(row)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def create_rollout(payload) -> tuple[dict, int]:
    plan_id = _parse_uuid(payload.plan_id, "PLAN_NOT_FOUND", "plan not found")
    with _tx() as conn:
        plan = conn.execute("SELECT * FROM plans WHERE id = %s", (plan_id,)).fetchone()
        if plan is None:
            raise not_found("PLAN_NOT_FOUND", "plan not found", {"plan_id": plan_id})
        if plan["status"] != "completed":
            raise conflict(
                "PLAN_NOT_EXECUTABLE",
                "cannot create a rollout for a plan without a safe permutation",
                {"plan_id": plan_id, "plan_status": plan["status"]},
            )
        total = len(plan["permutation"])
        empty = total == 0
        row = conn.execute(
            """
            INSERT INTO rollouts (plan_id, idempotency_key, plan_digest, status,
                                  total_steps, completed_at)
            VALUES (%s, %s, %s, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING *
            """,
            (
                plan_id,
                payload.idempotency_key,
                plan["plan_digest"],
                "in_progress" if not empty else "completed",
                total,
                empty,
            ),
        ).fetchone()
        if row is None:
            existing = conn.execute(
                "SELECT * FROM rollouts WHERE idempotency_key = %s", (payload.idempotency_key,)
            ).fetchone()
            if str(existing["plan_id"]) != plan_id:
                raise conflict(
                    "IDEMPOTENCY_CONFLICT",
                    "idempotency_key was already used with a different plan_id",
                    {"idempotency_key": payload.idempotency_key},
                )
            return get_rollout(str(existing["id"])), 200

        _record_event(
            conn,
            row["id"],
            "ROLLOUT_CREATED",
            {"plan_id": plan_id, "plan_digest": plan["plan_digest"], "total_steps": total},
        )
        rollout_id = str(row["id"])
    return get_rollout(rollout_id), 201


def get_rollout(rollout_id: str) -> dict:
    rollout_id = _parse_uuid(rollout_id, "ROLLOUT_NOT_FOUND", "rollout not found")
    with _tx() as conn:
        rollout = conn.execute("SELECT * FROM rollouts WHERE id = %s", (rollout_id,)).fetchone()
        if rollout is None:
            raise not_found("ROLLOUT_NOT_FOUND", "rollout not found", {"rollout_id": rollout_id})
        lease = conn.execute(
            "SELECT holder, epoch, expires_at, (expires_at <= now()) AS expired "
            "FROM leases WHERE rollout_id = %s",
            (rollout_id,),
        ).fetchone()
        commands = conn.execute(
            "SELECT * FROM commands WHERE rollout_id = %s ORDER BY step", (rollout_id,)
        ).fetchall()
    pending = next((c for c in commands if c["status"] == "PENDING"), None)
    return {
        "id": str(rollout["id"]),
        "plan_id": str(rollout["plan_id"]),
        "plan_digest": rollout["plan_digest"],
        "status": rollout["status"],
        "total_steps": rollout["total_steps"],
        "issued_count": rollout["issued_count"],
        "current_step": pending["step"] if pending else None,
        "lease": (
            {
                "holder": lease["holder"],
                "epoch": lease["epoch"],
                "expires_at": lease["expires_at"],
                "expired": lease["expired"],
            }
            if lease
            else None
        ),
        "commands": [_command_view(c) for c in commands],
        "created_at": rollout["created_at"],
        "completed_at": rollout["completed_at"],
    }


def get_audit(rollout_id: str) -> dict:
    rollout_id = _parse_uuid(rollout_id, "ROLLOUT_NOT_FOUND", "rollout not found")
    with _tx() as conn:
        rollout = conn.execute("SELECT id FROM rollouts WHERE id = %s", (rollout_id,)).fetchone()
        if rollout is None:
            raise not_found("ROLLOUT_NOT_FOUND", "rollout not found", {"rollout_id": rollout_id})
        events = conn.execute(
            "SELECT id, event_type, payload, created_at FROM events "
            "WHERE rollout_id = %s ORDER BY id",
            (rollout_id,),
        ).fetchall()
    return {"rollout_id": rollout_id, "events": events}


# ---------------------------------------------------------------------------
# lease (acquire / renew / takeover) -- all times come from the database clock
# ---------------------------------------------------------------------------

def acquire_lease(rollout_id: str, payload) -> dict:
    rollout_id = _parse_uuid(rollout_id, "ROLLOUT_NOT_FOUND", "rollout not found")
    request_hash = stable_digest(
        {"coordinator_id": payload.coordinator_id, "ttl_seconds": payload.ttl_seconds}
    )
    with _tx() as conn:
        rollout = conn.execute(
            "SELECT * FROM rollouts WHERE id = %s FOR UPDATE", (rollout_id,)
        ).fetchone()
        if rollout is None:
            raise not_found("ROLLOUT_NOT_FOUND", "rollout not found", {"rollout_id": rollout_id})

        op = conn.execute(
            "SELECT * FROM lease_ops WHERE rollout_id = %s AND op_id = %s",
            (rollout_id, payload.op_id),
        ).fetchone()
        if op is not None:
            if op["request_hash"] != request_hash:
                raise conflict(
                    "IDEMPOTENCY_CONFLICT",
                    "op_id was already used with different lease parameters",
                    {"op_id": payload.op_id},
                )
            return _lease_view(
                rollout_id, op["holder"], op["epoch"], op["expires_at"], op["op_type"], op["created_at"]
            )

        if rollout["status"] == "completed":
            raise conflict(
                "ROLLOUT_COMPLETED",
                "the rollout is already completed; coordination is no longer available",
                {"rollout_id": rollout_id},
            )

        lease = conn.execute(
            "SELECT * FROM leases WHERE rollout_id = %s", (rollout_id,)
        ).fetchone()
        clock = conn.execute(
            "SELECT now() AS now, now() + make_interval(secs => %s) AS expires",
            (payload.ttl_seconds,),
        ).fetchone()
        now, expires = clock["now"], clock["expires"]

        if lease is None:
            epoch, op_type = 1, "acquire"
        elif lease["expires_at"] <= now:
            epoch, op_type = lease["epoch"] + 1, "takeover"
        elif lease["holder"] == payload.coordinator_id:
            epoch, op_type = lease["epoch"], "renew"
        else:
            raise conflict(
                "LEASE_HELD",
                "the coordination lease is held by another coordinator",
                {"holder": lease["holder"], "epoch": lease["epoch"], "expires_at": lease["expires_at"]},
            )

        conn.execute(
            """
            INSERT INTO leases (rollout_id, epoch, holder, expires_at, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (rollout_id) DO UPDATE
            SET epoch = EXCLUDED.epoch, holder = EXCLUDED.holder,
                expires_at = EXCLUDED.expires_at, updated_at = now()
            """,
            (rollout_id, epoch, payload.coordinator_id, expires),
        )
        conn.execute(
            """
            INSERT INTO lease_ops (rollout_id, op_id, request_hash, holder, epoch, expires_at, op_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (rollout_id, payload.op_id, request_hash, payload.coordinator_id, epoch, expires, op_type),
        )
        _record_event(
            conn,
            rollout_id,
            {"acquire": "LEASE_ACQUIRED", "renew": "LEASE_RENEWED", "takeover": "LEASE_TAKEN_OVER"}[op_type],
            {
                "holder": payload.coordinator_id,
                "epoch": epoch,
                "ttl_seconds": payload.ttl_seconds,
                "expires_at": expires.isoformat(),
                "op_id": payload.op_id,
            },
        )
        return _lease_view(rollout_id, payload.coordinator_id, epoch, expires, op_type, now)


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------

def advance(rollout_id: str, payload) -> dict:
    rollout_id = _parse_uuid(rollout_id, "ROLLOUT_NOT_FOUND", "rollout not found")
    request_hash = stable_digest(
        {"coordinator_id": payload.coordinator_id, "epoch": payload.epoch}
    )
    with _tx() as conn:
        rollout = conn.execute(
            "SELECT * FROM rollouts WHERE id = %s FOR UPDATE", (rollout_id,)
        ).fetchone()
        if rollout is None:
            raise not_found("ROLLOUT_NOT_FOUND", "rollout not found", {"rollout_id": rollout_id})

        # Idempotent replay of a previously accepted advance operation.
        op = conn.execute(
            "SELECT * FROM advance_ops WHERE rollout_id = %s AND op_id = %s",
            (rollout_id, payload.op_id),
        ).fetchone()
        if op is not None:
            if op["request_hash"] != request_hash:
                raise conflict(
                    "IDEMPOTENCY_CONFLICT",
                    "op_id was already used with different advance parameters",
                    {"op_id": payload.op_id},
                )
            cmd = conn.execute(
                "SELECT * FROM commands WHERE command_id = %s", (op["command_id"],)
            ).fetchone()
            return _advance_view(cmd, rollout)

        if rollout["status"] == "completed":
            raise conflict(
                "ROLLOUT_COMPLETED",
                "the rollout is already completed",
                {"rollout_id": rollout_id},
            )

        lease = conn.execute(
            "SELECT * FROM leases WHERE rollout_id = %s", (rollout_id,)
        ).fetchone()
        if lease is None:
            raise conflict(
                "NO_ACTIVE_LEASE",
                "no coordination lease exists for this rollout",
                {"rollout_id": rollout_id},
            )
        if lease["holder"] != payload.coordinator_id:
            raise conflict(
                "NOT_LEASE_HOLDER",
                "the coordination lease is held by another coordinator",
                {"holder": lease["holder"], "epoch": lease["epoch"]},
            )
        if lease["epoch"] != payload.epoch:
            raise conflict(
                "STALE_EPOCH",
                "the supplied epoch is not the current lease epoch",
                {"current_epoch": lease["epoch"], "supplied_epoch": payload.epoch},
            )
        alive = conn.execute(
            "SELECT (expires_at > now()) AS alive FROM leases WHERE rollout_id = %s",
            (rollout_id,),
        ).fetchone()["alive"]
        if not alive:
            raise conflict(
                "LEASE_EXPIRED",
                "the coordination lease has expired",
                {"epoch": lease["epoch"], "expires_at": lease["expires_at"]},
            )

        # At most one unacknowledged command per rollout.  If the next step
        # already has a command (response lost, restart, takeover), return
        # the original command unchanged.
        cmd = conn.execute(
            "SELECT * FROM commands WHERE rollout_id = %s AND status = 'PENDING'",
            (rollout_id,),
        ).fetchone()
        if cmd is None:
            step = rollout["issued_count"]
            if step >= rollout["total_steps"]:
                raise conflict(
                    "NO_STEPS_REMAINING",
                    "all planned steps have already been issued",
                    {"rollout_id": rollout_id},
                )
            permutation = conn.execute(
                "SELECT permutation FROM plans WHERE id = %s", (rollout["plan_id"],)
            ).fetchone()["permutation"]
            switch_id = permutation[step]
            # Device-global generation: strictly increasing across *all*
            # rollouts for the switch.  The counter row is created once and
            # never deleted -- not after acks, not on rollout completion, not
            # while idle and not on restart -- so re-running a migration plan
            # for the same device always observes a higher generation.  The
            # UPSERT takes a row lock on the device row, so concurrent first
            # issuances from different rollouts/replicas serialise here and
            # receive consecutive generations.
            generation = conn.execute(
                """
                INSERT INTO device_state (switch_id, last_issued_generation, last_accepted_generation)
                VALUES (%s, 1, 0)
                ON CONFLICT (switch_id) DO UPDATE
                SET last_issued_generation = device_state.last_issued_generation + 1
                RETURNING last_issued_generation
                """,
                (switch_id,),
            ).fetchone()["last_issued_generation"]
            command_id = str(uuid.uuid5(_COMMAND_NS, f"{rollout_id}:{step}:{rollout['plan_digest']}"))
            cmd = conn.execute(
                """
                INSERT INTO commands (command_id, rollout_id, step, switch_id, plan_digest,
                                      device_generation)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (command_id, rollout_id, step, switch_id, rollout["plan_digest"], generation),
            ).fetchone()
            conn.execute(
                "UPDATE rollouts SET issued_count = issued_count + 1 WHERE id = %s",
                (rollout_id,),
            )
            _record_event(
                conn,
                rollout_id,
                "COMMAND_CREATED",
                {
                    "command_id": command_id,
                    "step": step,
                    "switch_id": switch_id,
                    "device_generation": generation,
                    "epoch": payload.epoch,
                    "coordinator_id": payload.coordinator_id,
                    "op_id": payload.op_id,
                },
            )
        conn.execute(
            """
            INSERT INTO advance_ops (rollout_id, op_id, request_hash, command_id)
            VALUES (%s, %s, %s, %s)
            """,
            (rollout_id, payload.op_id, request_hash, cmd["command_id"]),
        )
        return _advance_view(cmd, rollout)


def _advance_view(cmd: dict, rollout: dict) -> dict:
    return {
        "command": _command_view(cmd),
        "rollout_id": str(rollout["id"]),
        "rollout_status": rollout["status"],
        "total_steps": rollout["total_steps"],
    }


# ---------------------------------------------------------------------------
# acknowledgements (independent of the coordination lease on purpose)
# ---------------------------------------------------------------------------

def submit_ack(rollout_id: str, payload) -> dict:
    rollout_id = _parse_uuid(rollout_id, "ROLLOUT_NOT_FOUND", "rollout not found")
    with _tx() as conn:
        rollout = conn.execute(
            "SELECT * FROM rollouts WHERE id = %s FOR UPDATE", (rollout_id,)
        ).fetchone()
        if rollout is None:
            raise not_found("ROLLOUT_NOT_FOUND", "rollout not found", {"rollout_id": rollout_id})

        cmd = conn.execute(
            "SELECT * FROM commands WHERE command_id = %s", (payload.command_id,)
        ).fetchone()
        if cmd is None:
            raise not_found(
                "COMMAND_NOT_FOUND", "command not found", {"command_id": payload.command_id}
            )
        if str(cmd["rollout_id"]) != rollout_id:
            raise conflict(
                "ROLLOUT_MISMATCH",
                "the command belongs to a different rollout",
                {"command_rollout_id": str(cmd["rollout_id"])},
            )
        if cmd["switch_id"] != payload.switch_id:
            raise conflict(
                "SWITCH_MISMATCH",
                "the acknowledging switch does not match the command",
                {"expected": cmd["switch_id"], "got": payload.switch_id},
            )
        if cmd["step"] != payload.step:
            raise conflict(
                "STEP_MISMATCH",
                "the acknowledged step does not match the command",
                {"expected": cmd["step"], "got": payload.step},
            )
        if cmd["plan_digest"] != payload.plan_digest:
            raise conflict(
                "PLAN_DIGEST_MISMATCH",
                "the acknowledged plan digest does not match the command",
                {"expected": cmd["plan_digest"], "got": payload.plan_digest},
            )
        if cmd["device_generation"] != payload.device_generation:
            raise conflict(
                "GENERATION_MISMATCH",
                "the acknowledged device generation does not match the command",
                {"expected": cmd["device_generation"], "got": payload.device_generation},
            )

        if cmd["status"] == "APPLIED":
            # Identical acknowledgement re-submitted: return the first result.
            ack = conn.execute(
                "SELECT * FROM acks WHERE command_id = %s", (payload.command_id,)
            ).fetchone()
            return _ack_view(cmd, ack["accepted_at"], rollout["status"], duplicate=True)

        device = conn.execute(
            "SELECT * FROM device_state WHERE switch_id = %s FOR UPDATE", (cmd["switch_id"],)
        ).fetchone()
        accepted = device["last_accepted_generation"] if device else 0
        if payload.device_generation < accepted:
            raise conflict(
                "STALE_GENERATION",
                "the device has already accepted a newer generation",
                {
                    "switch_id": cmd["switch_id"],
                    "accepted_generation": accepted,
                    "supplied_generation": payload.device_generation,
                },
            )

        cmd = conn.execute(
            "UPDATE commands SET status = 'APPLIED', acked_at = now() "
            "WHERE command_id = %s RETURNING *",
            (payload.command_id,),
        ).fetchone()
        conn.execute(
            "UPDATE device_state SET last_accepted_generation = "
            "GREATEST(last_accepted_generation, %s) WHERE switch_id = %s",
            (payload.device_generation, cmd["switch_id"]),
        )
        ack = conn.execute(
            """
            INSERT INTO acks (command_id, rollout_id, switch_id, step, device_generation)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (payload.command_id, rollout_id, payload.switch_id, payload.step, payload.device_generation),
        ).fetchone()
        _record_event(
            conn,
            rollout_id,
            "COMMAND_APPLIED",
            {
                "command_id": payload.command_id,
                "step": payload.step,
                "switch_id": payload.switch_id,
                "device_generation": payload.device_generation,
            },
        )
        rollout_status = rollout["status"]
        if payload.step == rollout["total_steps"] - 1:
            conn.execute(
                "UPDATE rollouts SET status = 'completed', completed_at = now() WHERE id = %s",
                (rollout_id,),
            )
            _record_event(conn, rollout_id, "ROLLOUT_COMPLETED", {"total_steps": rollout["total_steps"]})
            rollout_status = "completed"
        return _ack_view(cmd, ack["accepted_at"], rollout_status, duplicate=False)


def _ack_view(cmd: dict, acked_at, rollout_status: str, duplicate: bool) -> dict:
    return {
        "command_id": cmd["command_id"],
        "rollout_id": str(cmd["rollout_id"]),
        "step": cmd["step"],
        "switch_id": cmd["switch_id"],
        "device_generation": cmd["device_generation"],
        "status": "APPLIED",
        "duplicate": duplicate,
        "acked_at": acked_at,
        "rollout_status": rollout_status,
    }


# ---------------------------------------------------------------------------
# switch-facing queries
# ---------------------------------------------------------------------------

def pending_commands(switch_id: str) -> dict:
    with _tx() as conn:
        rows = conn.execute(
            "SELECT * FROM commands WHERE switch_id = %s AND status = 'PENDING' "
            "ORDER BY created_at, command_id",
            (switch_id,),
        ).fetchall()
    return {"switch_id": switch_id, "commands": [_command_view(r) for r in rows]}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _record_event(conn, rollout_id, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO events (rollout_id, event_type, payload) VALUES (%s, %s, %s)",
        (str(rollout_id), event_type, Jsonb(payload)),
    )
