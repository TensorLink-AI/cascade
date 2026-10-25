"""Small concurrency primitives shared by long-running services."""

from concurrent.futures import ThreadPoolExecutor


def call_with_deadline(fn, seconds: float):
    """Bound the caller's wait for a blocking operation; propagate its result/error.

    A timeout does not terminate the underlying call. Its thread can continue
    until the operation returns. Do not use this as cancellation or as evidence
    that a remote operation failed. In particular, shutting down the executor
    must not wait for the timed-out call before returning to the service loop.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(fn).result(timeout=seconds)
    finally:
        executor.shutdown(wait=False)
