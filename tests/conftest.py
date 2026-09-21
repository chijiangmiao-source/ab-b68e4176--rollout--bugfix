"""Shared fixtures for the API acceptance tests.

The tests talk to two independent API replicas (api1/api2) that share one
PostgreSQL database, proving that every concurrency invariant holds across
processes, not just inside one.
"""
from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

API1 = os.environ.get("API1_URL", "http://localhost:8000")
API2 = os.environ.get("API2_URL", "http://localhost:8001")

# A freshly started API process connects to the same database as the two
# long-running replicas.  Used by restart-recovery tests.
TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@db:5432/migration"
)

WAIT_TIMEOUT = float(os.environ.get("API_WAIT_TIMEOUT", "90"))


def _wait_for(url: str) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT
    while True:
        try:
            r = httpx.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"API at {url} did not become healthy in time")
        time.sleep(1)


@pytest.fixture(scope="session", autouse=True)
def wait_for_apis():
    _wait_for(API1)
    _wait_for(API2)


@pytest.fixture(scope="session")
def api1() -> str:
    return API1


@pytest.fixture(scope="session")
def api2() -> str:
    return API2


@pytest.fixture()
def client():
    with httpx.Client(timeout=10) as c:
        yield c


def key() -> str:
    return f"k-{uuid.uuid4()}"


def topo_payload(switches: dict, ingresses: list[str]) -> dict:
    return {
        "switches": [
            {"id": sid, "old_next": old, "new_next": new}
            for sid, (old, new) in switches.items()
        ],
        "ingresses": list(ingresses),
    }


def simple_case() -> tuple[dict, list[str]]:
    """A topology whose only safe update order is [s1, s2].

    Switch ids are unique per call so that device generations and pending
    commands never leak between tests sharing one database.
    """
    p = uuid.uuid4().hex[:10]
    s1, s2, s3 = f"{p}-s1", f"{p}-s2", f"{p}-s3"
    topo = topo_payload(
        {
            s1: (s2, s3),
            s2: ("DELIVER", s1),
            s3: ("DELIVER", "DELIVER"),
        },
        [s1],
    )
    return topo, [s1, s2]


def simple_topology() -> dict:
    """Two changed switches; the only safe order is [s1, s2] (unique ids)."""
    return simple_case()[0]


def create_plan(client: httpx.Client, base: str, topology: dict | None = None, idem: str | None = None):
    return client.post(
        f"{base}/plans",
        json={"idempotency_key": idem or key(), "topology": topology or simple_topology()},
    )


def create_rollout(client: httpx.Client, base: str, plan_id: str, idem: str | None = None):
    body = {"plan_id": plan_id}
    if idem:
        body["idempotency_key"] = idem
    return client.post(f"{base}/rollouts", json=body)


def make_plan_and_rollout(client: httpx.Client, base: str, topology: dict | None = None):
    plan = create_plan(client, base, topology).json()
    r = create_rollout(client, base, plan["id"])
    assert r.status_code == 201, r.text
    return plan, r.json()


def lease(client: httpx.Client, base: str, rid: str, coordinator: str, ttl: int = 30, op: str | None = None):
    return client.post(
        f"{base}/rollouts/{rid}/lease",
        json={"coordinator_id": coordinator, "ttl_seconds": ttl, "op_id": op or key()},
    )


def advance(client: httpx.Client, base: str, rid: str, coordinator: str, epoch: int, op: str | None = None):
    return client.post(
        f"{base}/rollouts/{rid}/advance",
        json={"coordinator_id": coordinator, "epoch": epoch, "op_id": op or key()},
    )


def ack(client: httpx.Client, base: str, rid: str, command: dict):
    return client.post(
        f"{base}/rollouts/{rid}/acks",
        json={
            "command_id": command["command_id"],
            "switch_id": command["switch_id"],
            "step": command["step"],
            "plan_digest": command["plan_digest"],
            "device_generation": command["device_generation"],
        },
    )


def drive_to_completion(client: httpx.Client, base: str, rid: str, coordinator: str = "c1"):
    """Acquire a lease and advance/ack until the rollout completes."""
    r = lease(client, base, rid, coordinator)
    assert r.status_code == 200, r.text
    epoch = r.json()["epoch"]
    for _ in range(50):
        r = advance(client, base, rid, coordinator, epoch)
        if r.status_code == 409 and r.json()["error"]["code"] == "ROLLOUT_COMPLETED":
            return
        assert r.status_code == 200, r.text
        cmd = r.json()["command"]
        r = ack(client, base, rid, cmd)
        assert r.status_code == 200, r.text
        if r.json()["rollout_status"] == "completed":
            return
    raise AssertionError("rollout did not complete")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def fresh_api(database_url: str | None = None):
    """Run a brand-new API process against the shared database.

    Simulates a service (re)start / a cold third replica: a clean process
    with no in-process state of any kind.  Yields the base URL of the
    short-lived instance and tears it down on exit.
    """
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["DATABASE_URL"] = database_url or TEST_DATABASE_URL
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log = tempfile.NamedTemporaryFile(prefix="fresh-api-", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log.close()
