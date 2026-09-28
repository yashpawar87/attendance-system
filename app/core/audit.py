import logging

logger = logging.getLogger("attendance.audit")


def record(action: str, resource_type: str, resource_id: int | str, actor: str = "api-token") -> None:
    """Emit a non-biometric audit event; never pass images, vectors, or request bodies here."""
    logger.info("audit action=%s resource_type=%s resource_id=%s actor=%s", action, resource_type, resource_id, actor)
