# retry.py
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)

def with_retry(
    fn,
    *,
    what: str,
    max_retries: int = 3,
    retry_delay: float = 0.2,
) -> bool:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            ok = bool(fn())
            if ok:
                return True
            raise RuntimeError("returned falsy")
        except Exception as e:
            last_err = e
            log.error(f"[RETRY] {what} failed attempt={attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)

    log.warning(f"[RETRY] {what} failed after {max_retries} attempts: {last_err}")
    return False
