"""Background tasks."""

from .utils import send_notification as _send_notification


def send_notification(user, message: str) -> None:
    """Queue notification task."""
    # In real Django, this would use Celery
    _send_notification(user, message)