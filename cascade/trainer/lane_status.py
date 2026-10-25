"""Best-effort, rate-limited B/C status publication with independent lane clocks."""

import json
import time
from datetime import UTC, datetime


def publish_lane_status(sched, block, *, lane, status_key, last_published,
                         min_interval_s, clock, logger):
    driver = getattr(sched, f"lane_{lane}", None)
    if driver is None:
        return False
    now = (clock or time.monotonic)()
    key = id(sched)
    last = last_published.get(key)
    if last is not None and now - last[0] < min_interval_s:
        return False
    try:
        doc = {"lane": lane, "block": int(block), "as_of": datetime.now(UTC).isoformat(),
               **driver.status()}
        body = json.dumps(doc, sort_keys=True, default=str)
        if last is not None and last[1] == body:
            last_published[key] = (now, body)
            return False
        from ..shared.hippius import publish_public_json

        publish_public_json(sched.r.manifest_store(), status_key, json.loads(body))
        last_published[key] = (now, body)
        return True
    except Exception as e:  # noqa: BLE001 — discovery must never hold training/settlement
        logger.warning("lane %s status publish failed (ignored): %s", lane.upper(), e)
        last_published[key] = (now, last[1] if last else "")
        return False
