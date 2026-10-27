"""HTTP/WebSocket API around the `fatigue` package. No fatigue logic lives here: this layer only
translates between JSON and the same classes the CLI uses (LiveSession, PretripSession, route, ...)."""
from .app import create_app

__all__ = ["create_app"]
