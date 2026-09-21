# Switch Migration Service

A pure-backend service that plans and safely executes the migration of
backbone switches from an old forwarding table to a new one.

* **Planning** — given a topology (`old_next` / `new_next` per switch), the
  planner searches — with full backtracking — for an update permutation
  such that **every prefix** of the permutation yields a safe mixed
  forwarding state, and returns the lexicographically smallest one
  (switch ids compared by raw UTF-8 bytes).  If no safe permutation exists
  it returns a definitive `proven_impossible`.
* **Execution** — a rollout strictly follows the persisted plan, one device
  command at a time, coordinated through database-clock leases with
  strictly increasing fencing epochs.  Commands, generations and
  acknowledgements are fully idempotent and survive coordinator takeovers,
  lost responses and service restarts.
* **Persistence** — everything (topology, immutable plans, leases, epochs,
  device generations, commands, acks, audit events) lives in PostgreSQL.
  Any number of stateless API replicas can share one database; all
  concurrency invariants are enforced by the database itself (row locks,
  unique constraints, `now()`), never by in-process state.

## Repository layout

```
app/
  main.py            FastAPI routes and error handlers
  service.py         business logic (plans, rollouts, lease, advance, acks)
  planner.py         safety checker + backtracking permutation search
  db.py              connection pool + migration runner
  errors.py          structured API errors
  schemas.py         request schemas
  migrations/001_init.sql
tests/
  test_planner.py    planner/safety unit tests (no DB needed)
  test_api.py        end-to-end acceptance tests (two replicas, one DB)
  conftest.py
Dockerfile
docker-compose.yml   db + api1 + api2 + one-shot verify
requirements.txt
```

## Running

Only Docker is required.

```bash
docker compose up --build -d db api1 api2   # start database + two API replicas
docker compose up --build verify            # run the acceptance suite (exits 0 on success)
docker compose down -v                      # tear down and wipe the volume
```

`docker compose up --build` also works; the `verify` service runs the tests
once both APIs are healthy and then exits.

Ports:

| service | host port                | container port |
|---------|--------------------------|----------------|
| api1    | `${API_PORT:-8000}`      | 8000           |
| api2    | `${API_PORT_2:-8001}`    | 8000           |

```bash
API_PORT=9000 API_PORT_2=9001 docker compose up --build -d api1 api2
```

Health check: `GET /health` → `{"status":"ok","db":"up"}` (503 when the
database is unreachable).

## Model

A topology has 1–60 switches with unique ids, one or more ingresses, and
per switch an `old_next` and `new_next` hop.  A hop target is another
declared switch id, the exit `DELIVER`, or the blackhole `DROP` (self loops
are ordinary hops).  A device update atomically flips one switch from
`old_next` to `new_next`.

A mixed state (any set of updated switches) is **safe** iff walking from
*every* ingress along the current hops reaches `DELIVER` within at most *N*
hops (*N* = total number of switches).  Hitting `DROP`, an unknown node, a
repeated node or exceeding the hop bound is unsafe.

Plan creation validates the initial and final states
(`422 UNSAFE_INITIAL_STATE` / `422 UNSAFE_FINAL_STATE`) and plans only
switches where `old_next != new_next` (at most 22, else
`422 DIFF_TOO_LARGE`).

### Planner

`find_safe_permutation` performs a depth-first search with backtracking over
the changed switches; candidates are tried in UTF-8 byte order at every
level, so the first complete permutation found is the lexicographically
smallest safe one.  Dead ends are memoised per updated-set.  If the search
exhausts, the plan is persisted with status `proven_impossible` (no
permutation) and cannot be executed.  Greedy "pick the smallest safe node"
without backtracking is deliberately not used.

Each plan carries a stable digest:

```
plan_digest = sha256( canonical_json({ "permutation": [...], "topology": canonical }) )
canonical   = switches sorted by id (UTF-8 bytes), ingresses sorted, minimal JSON
```

Plans are immutable once created.  `POST /plans` is idempotent on
`idempotency_key`: same key + same canonical parameters (even concurrently
across replicas) yields exactly one plan and the same response; same key +
different parameters yields `409 IDEMPOTENCY_CONFLICT`.

## Execution protocol

1. `POST /rollouts` binds a rollout to a completed plan.
2. A coordinator takes the lease: `POST /rollouts/{id}/lease`
   (`coordinator_id`, `ttl_seconds` 5–60, `op_id`).  Expiry is computed with
   the **database clock**.  Every takeover assigns a strictly increasing,
   never-reused `epoch`.  Re-acquiring with the same `op_id` replays the
   recorded result.
3. `POST /rollouts/{id}/advance` (`coordinator_id`, `epoch`, `op_id`)
   creates **at most one** command for the plan's next step — only while the
   caller holds the current, unexpired epoch.  The command fixes
   `rollout_id`, `step`, `switch_id`, `plan_digest`, a strictly increasing
   per-device `device_generation` and a deterministic `command_id`
   (uuid5 of `rollout_id:step:plan_digest`).  The per-device generation
   counter is permanent database state: it never resets, so acknowledged
   commands, completed rollouts, idle periods and service restarts cannot
   make a device see generation `1` (or any reused value) twice.  If the
   step already has a command (lost response, restart, takeover), the
   original command is returned unchanged — never a new id or generation.
4. Switches poll `GET /switches/{id}/pending-commands` and acknowledge with
   `POST /rollouts/{id}/acks`.  Acks are validated against the persisted
   command (switch, step, digest, generation) and against the device's
   accepted generation (`409 STALE_GENERATION` for late, superseded acks).
   Identical acks are replay-safe and return the first result.  Acks do
   **not** require a live lease, so a legal late ack still converges during
   a takeover — but an ack only closes its own step; it never issues the
   next command.
5. At most one unacknowledged command exists per rollout.  After the last
   step is acknowledged the rollout becomes `COMPLETED`; stale epochs can
   neither create commands nor complete a migration.

All state transitions run in single PostgreSQL transactions serialised by
`SELECT ... FOR UPDATE` row locks, so the invariants hold with multiple API
replicas and after any restart.

## API summary

| method & path                            | purpose                                   |
|------------------------------------------|-------------------------------------------|
| `POST /plans`                            | create plan (idempotent)                  |
| `GET  /plans/{id}`                       | fetch plan                                |
| `POST /rollouts`                         | create rollout (optional idempotency key) |
| `GET  /rollouts/{id}`                    | full execution status                     |
| `GET  /rollouts/{id}/audit`              | ordered audit trail                       |
| `POST /rollouts/{id}/lease`              | acquire / renew / take over coordination  |
| `POST /rollouts/{id}/advance`            | issue the next step's command             |
| `POST /rollouts/{id}/acks`               | submit an APPLIED acknowledgement         |
| `GET  /switches/{id}/pending-commands`   | commands a switch should apply            |
| `GET  /health`                           | health check                              |

### Example

```bash
# create a plan
curl -s -X POST localhost:8000/plans -H 'content-type: application/json' -d '{
  "idempotency_key": "plan-1",
  "topology": {
    "switches": [
      {"id": "s1", "old_next": "s2", "new_next": "s3"},
      {"id": "s2", "old_next": "DELIVER", "new_next": "s1"},
      {"id": "s3", "old_next": "DELIVER", "new_next": "DELIVER"}
    ],
    "ingresses": ["s1"]
  }}'
# -> {"id": "...", "status": "completed", "permutation": ["s1", "s2"], "plan_digest": "sha256:...", ...}

curl -s -X POST localhost:8000/rollouts -d '{"plan_id": "<plan-id>"}' -H 'content-type: application/json'
curl -s -X POST localhost:8000/rollouts/<rid>/lease -H 'content-type: application/json' \
     -d '{"coordinator_id": "c1", "ttl_seconds": 30, "op_id": "op-1"}'
curl -s -X POST localhost:8000/rollouts/<rid>/advance -H 'content-type: application/json' \
     -d '{"coordinator_id": "c1", "epoch": 1, "op_id": "op-2"}'
curl -s localhost:8000/switches/s1/pending-commands
curl -s -X POST localhost:8000/rollouts/<rid>/acks -H 'content-type: application/json' \
     -d '{"command_id": "...", "switch_id": "s1", "step": 0,
          "plan_digest": "sha256:...", "device_generation": 1}'
curl -s localhost:8000/rollouts/<rid>          # status
curl -s localhost:8000/rollouts/<rid>/audit    # audit trail
```

## Error responses

Errors are stable and machine-readable:
`{"error": {"code": "...", "message": "...", "details": {...}}}`

| status | code |
|--------|------|
| 404 | `PLAN_NOT_FOUND`, `ROLLOUT_NOT_FOUND`, `COMMAND_NOT_FOUND`, `NOT_FOUND` |
| 409 | `IDEMPOTENCY_CONFLICT`, `PLAN_NOT_EXECUTABLE`, `LEASE_HELD`, `NO_ACTIVE_LEASE`, `NOT_LEASE_HOLDER`, `STALE_EPOCH`, `LEASE_EXPIRED`, `ROLLOUT_COMPLETED`, `NO_STEPS_REMAINING`, `ROLLOUT_MISMATCH`, `SWITCH_MISMATCH`, `STEP_MISMATCH`, `PLAN_DIGEST_MISMATCH`, `GENERATION_MISMATCH`, `STALE_GENERATION` |
| 422 | `VALIDATION_ERROR`, `DUPLICATE_SWITCH_ID`, `RESERVED_SWITCH_ID`, `UNKNOWN_NEXT_HOP`, `UNKNOWN_INGRESS`, `DUPLICATE_INGRESS`, `DIFF_TOO_LARGE`, `UNSAFE_INITIAL_STATE`, `UNSAFE_FINAL_STATE` |

## Tests

`docker compose up --build verify` runs everything: planner unit tests
(including a brute-force lex-min cross-check) and the end-to-end acceptance
suite against both API replicas sharing one database — plan idempotency and
conflicts, lease epochs and fencing, takeover convergence, command
immutability, per-device generations, ack validation and replay, audit
trail, and cross-replica concurrency (parallel advances create exactly one
command; parallel lease acquisition elects exactly one holder).
