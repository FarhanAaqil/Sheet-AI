"""Typed domain errors shared by API and service boundaries."""


class SheetSenseError(Exception):
    code = "SHEETSENSE_ERROR"
    status_code = 500


class NotFoundError(SheetSenseError):
    code = "NOT_FOUND"
    status_code = 404


class AuthorizationError(SheetSenseError):
    code = "FORBIDDEN"
    status_code = 403


class ConflictError(SheetSenseError):
    code = "CONFLICT"
    status_code = 409


class DependencyUnavailableError(SheetSenseError):
    code = "DEPENDENCY_UNAVAILABLE"
    status_code = 503
