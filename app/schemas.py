"""Request/response schemas (pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .planner import MAX_SWITCHES


class SwitchIn(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    old_next: str = Field(min_length=1, max_length=128)
    new_next: str = Field(min_length=1, max_length=128)


class TopologyIn(BaseModel):
    switches: list[SwitchIn] = Field(min_length=1, max_length=MAX_SWITCHES)
    ingresses: list[str] = Field(min_length=1, max_length=MAX_SWITCHES)


class PlanCreateIn(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=256)
    topology: TopologyIn


class RolloutCreateIn(BaseModel):
    plan_id: str = Field(min_length=1, max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)


class LeaseIn(BaseModel):
    coordinator_id: str = Field(min_length=1, max_length=256)
    ttl_seconds: int = Field(ge=5, le=60)
    op_id: str = Field(min_length=1, max_length=256)


class AdvanceIn(BaseModel):
    coordinator_id: str = Field(min_length=1, max_length=256)
    epoch: int = Field(ge=1)
    op_id: str = Field(min_length=1, max_length=256)


class AckIn(BaseModel):
    command_id: str = Field(min_length=1, max_length=128)
    switch_id: str = Field(min_length=1, max_length=128)
    step: int = Field(ge=0)
    plan_digest: str = Field(min_length=1, max_length=128)
    device_generation: int = Field(ge=1)
