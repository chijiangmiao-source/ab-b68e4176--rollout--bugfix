"""Database access: connection pool and schema migrations.

All correctness-relevant coordination (leases, epochs, idempotency, command
uniqueness) is enforced by PostgreSQL itself -- row locks, unique
constraints and the database clock -- so any number of API replicas can
share one database safely.  No in-process state is kept anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from psycopg_pool import ConnectionPool

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# Advisory lock serialising concurrent migrations from several API replicas.
_MIGRATION_LOCK_ID = 0x4D494752  # 'MIGR'

_pool: ConnectionPool | None = None


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/migration",
    )


def init_pool() -> None:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=database_url(), min_size=1, max_size=10, open=True)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("connection pool is not initialised")
    return _pool


def run_migrations() -> None:
    """Apply pending SQL migrations exactly once across all replicas."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with pool().connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        try:
            for path in files:
                sql = path.read_text(encoding="utf-8")
                # Migration files contain plain DDL statements only (no
                # functions or dollar-quoted bodies), so splitting on ';'
                # is safe here.
                for statement in sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
            conn.commit()
            _discard_idle_device_state(conn)
            conn.commit()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
            conn.commit()


def _discard_idle_device_state(conn) -> None:
    active = conn.execute(
        "SELECT DISTINCT switch_id FROM commands WHERE status = 'PENDING'"
    ).fetchall()
    active_ids = [row[0] for row in active]
    if active_ids:
        conn.execute(
            "DELETE FROM device_state WHERE NOT (switch_id = ANY(%s))",
            (active_ids,),
        )
    else:
        conn.execute("DELETE FROM device_state")
