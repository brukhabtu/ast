"""Utility functions."""

from typing import Dict, Any, Type
from .base import Model


def render_template(template_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Render a template with context."""
    return {
        "template": template_name,
        "context": context
    }


def get_object_or_404(model_class: Type[Model], **kwargs) -> Model:
    """Get object or raise 404."""
    try:
        return model_class.objects().get(**kwargs)
    except Exception:
        raise Http404(f"{model_class.__name__} not found")


class Http404(Exception):
    """404 error."""
    pass


def send_notification(user: "User", message: str) -> None:
    """Send notification to user."""
    print(f"Sending notification to {user.username}: {message}")