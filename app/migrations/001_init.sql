-- 001_init.sql -- schema for the switch migration service.
--
-- Everything correctness-relevant lives in the database: immutable plans,
-- rollouts, coordination leases with monotonically increasing epochs,
-- per-device generations, commands and acknowledgements.

CREATE TABLE IF NOT EXISTS plans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key TEXT NOT NULL UNIQUE,
    request_hash    TEXT NOT NULL,
    topology        JSONB NOT NULL,            -- canonical topology (sorted)
    diff_switches   JSONB NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('completed', 'proven_impossible')),
    permutation     JSONB,                     -- NULL iff proven_impossible
    plan_digest     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rollouts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id         UUID NOT NULL REFERENCES plans(id),
    idempotency_key TEXT UNIQUE,
    plan_digest     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'in_progress'
                    CHECK (status IN ('in_progress', 'completed')),
    issued_count    INT NOT NULL DEFAULT 0,    -- commands issued so far
    total_steps     INT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- One lease row per rollout.  epoch is strictly increasing and never
-- reused: every takeover sets epoch = previous epoch + 1.
CREATE TABLE IF NOT EXISTS leases (
    rollout_id  UUID PRIMARY KEY REFERENCES rollouts(id),
    epoch       BIGINT NOT NULL,
    holder      TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent operation log for lease acquire/renew/takeover.
CREATE TABLE IF NOT EXISTS lease_ops (
    rollout_id   UUID NOT NULL REFERENCES rollouts(id),
    op_id        TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    holder       TEXT NOT NULL,
    epoch        BIGINT NOT NULL,
    expires_at   TIMESTAMPTZ NOT NULL,
    op_type      TEXT NOT NULL CHECK (op_type IN ('acquire', 'renew', 'takeover')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (rollout_id, op_id)
);

-- Strictly increasing per-device generations, global across rollouts.
-- Rows are permanent and must never be deleted: the counters have to
-- survive acknowledged commands, completed rollouts, idle periods and
-- service restarts so a device can always tell newer commands from older
-- ones by their generation.
CREATE TABLE IF NOT EXISTS device_state (
    switch_id                 TEXT PRIMARY KEY,
    last_issued_generation    BIGINT NOT NULL DEFAULT 0,
    last_accepted_generation  BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS commands (
    command_id        TEXT PRIMARY KEY,        -- deterministic uuid5
    rollout_id        UUID NOT NULL REFERENCES rollouts(id),
    step              INT NOT NULL,
    switch_id         TEXT NOT NULL,
    plan_digest       TEXT NOT NULL,
    device_generation BIGINT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'PENDING'
                      CHECK (status IN ('PENDING', 'APPLIED')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    acked_at          TIMESTAMPTZ,
    UNIQUE (rollout_id, step)                  -- at most one command per step
);

-- Idempotent operation log for advance calls.
CREATE TABLE IF NOT EXISTS advance_ops (
    rollout_id   UUID NOT NULL REFERENCES rollouts(id),
    op_id        TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    command_id   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (rollout_id, op_id)
);

-- Accepted acknowledgements (one per command).
CREATE TABLE IF NOT EXISTS acks (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    command_id        TEXT NOT NULL UNIQUE REFERENCES commands(command_id),
    rollout_id        UUID NOT NULL,
    switch_id         TEXT NOT NULL,
    step              INT NOT NULL,
    device_generation BIGINT NOT NULL,
    accepted_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audit trail.
CREATE TABLE IF NOT EXISTS events (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    rollout_id  UUID NOT NULL,
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_commands_rollout ON commands(rollout_id, step);
CREATE INDEX IF NOT EXISTS idx_commands_switch_pending ON commands(switch_id) WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_events_rollout ON events(rollout_id, id);
