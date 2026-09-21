"""HTTP API for the switch migration service."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import db, service
from .errors import ApiError
from .schemas import (
    AckIn,
    AdvanceIn,
    LeaseIn,
    PlanCreateIn,
    RolloutCreateIn,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_pool()
    db.run_migrations()
    yield
    db.close_pool()


app = FastAPI(title="Switch Migration Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# error handling: stable machine-readable error bodies
# ---------------------------------------------------------------------------

@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(exc.body()))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "request failed schema validation",
                    "details": {"errors": exc.errors()},
                }
            }
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(exc.detail), "details": {}}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "internal error", "details": {}}},
    )


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        with db.pool().connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "db": "up"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "error", "db": "down"})


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

@app.post("/plans")
def create_plan(payload: PlanCreateIn, response: Response):
    body, status = service.create_plan(payload)
    response.status_code = status
    return body


@app.get("/plans/{plan_id}")
def get_plan(plan_id: str):
    return service.get_plan(plan_id)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

@app.post("/rollouts")
def create_rollout(payload: RolloutCreateIn, response: Response):
    body, status = service.create_rollout(payload)
    response.status_code = status
    return body


@app.get("/rollouts/{rollout_id}")
def get_rollout(rollout_id: str):
    return service.get_rollout(rollout_id)


@app.get("/rollouts/{rollout_id}/audit")
def get_audit(rollout_id: str):
    return service.get_audit(rollout_id)


@app.post("/rollouts/{rollout_id}/lease")
def acquire_lease(rollout_id: str, payload: LeaseIn):
    return service.acquire_lease(rollout_id, payload)


@app.post("/rollouts/{rollout_id}/advance")
def advance(rollout_id: str, payload: AdvanceIn):
    return service.advance(rollout_id, payload)


@app.post("/rollouts/{rollout_id}/acks")
def submit_ack(rollout_id: str, payload: AckIn):
    return service.submit_ack(rollout_id, payload)


# ---------------------------------------------------------------------------
# switch-facing
# ---------------------------------------------------------------------------

@app.get("/switches/{switch_id}/pending-commands")
def pending_commands(switch_id: str):
    return service.pending_commands(switch_id)
