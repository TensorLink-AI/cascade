"""Keep cascade's own loggers visible alongside bittensor.

bittensor's logging machine, on import, silences every *other* logger by setting
it to ``CRITICAL`` (it keeps only bittensor's own output). That swallows all
``cascade.*`` ``INFO``/``WARNING``/``ERROR`` messages from the trainer and
validator services, which makes them look hung when they are working fine.

:func:`restore_cascade_logging` imports bittensor (forcing that one-time silence
to happen now) and then puts the ``cascade.*`` loggers back at the requested
level. Call it once in a service ``main`` before entering the run loop.
"""

from __future__ import annotations

import contextlib
import logging


class _DropEmptyBittensor(logging.Filter):
    """Drop bittensor's empty-message ERROR records.

    bittensor's substrate/websocket layer logs a steady stream of ERROR records
    with an empty message during normal reconnect/keepalive — pure clutter that
    buries real logs. Records with a non-empty message (or from any other logger)
    pass through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.name.split(".", 1)[0] != "bittensor":
            return True
        try:
            return bool(record.getMessage().strip())
        except Exception:  # noqa: BLE001
            return True


def restore_cascade_logging(level_name: str = "INFO") -> None:
    """Restore ``cascade.*`` logger levels after bittensor silences them, and mute
    bittensor's empty-message reconnect noise."""
    # Force bittensor's logging machine to initialise now (it silences other
    # loggers on first import); doing it here means our restore below is the
    # last word. Best-effort: if bittensor isn't installed there's nothing to undo.
    with contextlib.suppress(Exception):
        import bittensor  # noqa: F401

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.getLogger("cascade").setLevel(level)
    for name in list(logging.root.manager.loggerDict):
        if name == "cascade" or name.startswith("cascade."):
            logging.getLogger(name).setLevel(level)

    # Mute the empty-message bittensor reconnect spam (idempotent).
    bt_log = logging.getLogger("bittensor")
    if not any(isinstance(f, _DropEmptyBittensor) for f in bt_log.filters):
        bt_log.addFilter(_DropEmptyBittensor())
