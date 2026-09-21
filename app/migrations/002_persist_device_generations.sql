-- 002_persist_device_generations.sql
--
-- Per-device generations must remain strictly increasing across *all*
-- rollouts for a switch: the same backbone switch can execute the same
-- migration plan repeatedly, and every command must outrank every earlier
-- one.  Earlier builds deleted idle device_state rows after rollout
-- completion and wiped them again on every API startup, which let the
-- counter restart at 1 for the next rollout (even after a restart).
--
-- Counters are now kept for the lifetime of the device.  Reconstruct any
-- rows that older builds may have deleted from the durable command history,
-- so a counter can never restart below a generation already handed out or
-- accepted.
INSERT INTO device_state (switch_id, last_issued_generation, last_accepted_generation)
SELECT switch_id,
       COALESCE(MAX(device_generation), 0),
       COALESCE(MAX(device_generation) FILTER (WHERE status = 'APPLIED'), 0)
FROM commands
GROUP BY switch_id
ON CONFLICT (switch_id) DO UPDATE
SET last_issued_generation =
        GREATEST(device_state.last_issued_generation, EXCLUDED.last_issued_generation),
    last_accepted_generation =
        GREATEST(device_state.last_accepted_generation, EXCLUDED.last_accepted_generation);
