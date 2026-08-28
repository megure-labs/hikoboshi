from __future__ import annotations

from typing import NoReturn


class HikoboshiError(RuntimeError):
    """Base class for structured Hikoboshi Python errors."""

    code = "internal_error"

    def __init__(self, detail: str = "", *, code: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code
        self.status_code = self.code


class InvalidArgumentError(HikoboshiError):
    code = "invalid_argument"


class FailedPreconditionError(HikoboshiError):
    code = "failed_precondition"


class UnavailableError(HikoboshiError):
    code = "unavailable"


class UnimplementedError(HikoboshiError):
    code = "unimplemented"


class InternalError(HikoboshiError):
    code = "internal_error"


_BY_CODE: dict[str, type[HikoboshiError]] = {
    "invalid_argument": InvalidArgumentError,
    "failed_precondition": FailedPreconditionError,
    "unavailable": UnavailableError,
    "unimplemented": UnimplementedError,
    "internal_error": InternalError,
}


def error_from_core(exc: BaseException) -> HikoboshiError:
    if isinstance(exc, HikoboshiError):
        return exc
    code = str(getattr(exc, "code", getattr(exc, "status_code", "internal_error")))
    detail = str(getattr(exc, "detail", str(exc)))
    cls = _BY_CODE.get(code, InternalError)
    return cls(detail, code=code)


def raise_from_core(exc: BaseException) -> NoReturn:
    raise error_from_core(exc) from exc


__all__ = [
    "HikoboshiError",
    "InvalidArgumentError",
    "FailedPreconditionError",
    "UnavailableError",
    "UnimplementedError",
    "InternalError",
    "error_from_core",
    "raise_from_core",
]
