"""Structured API errors."""
from __future__ import annotations


class ApiError(Exception):
    """An error that maps to a stable HTTP status and machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def body(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# -- 404 ---------------------------------------------------------------------

def not_found(code: str, message: str, details: dict | None = None) -> ApiError:
    return ApiError(404, code, message, details)


# -- 409 ---------------------------------------------------------------------

def conflict(code: str, message: str, details: dict | None = None) -> ApiError:
    return ApiError(409, code, message, details)


# -- 422 ---------------------------------------------------------------------

def unprocessable(code: str, message: str, details: dict | None = None) -> ApiError:
    return ApiError(422, code, message, details)
