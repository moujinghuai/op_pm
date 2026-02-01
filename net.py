# net.py
import time
import requests
from typing import Any, Dict, Optional, Callable

class Http:
    def __init__(self, retries: int = 2, backoff: float = 0.25):
        self.sess = requests.Session()
        self.retries = retries
        self.backoff = backoff

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        last = None
        for i in range(self.retries + 1):
            try:
                return fn()
            except Exception as e:
                last = e
                if i < self.retries:
                    time.sleep(self.backoff * (2 ** i))
        raise last

    def get(self, url: str, headers: Optional[Dict[str, str]] = None,
            params: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Any:
        def _do():
            r = self.sess.get(url, headers=headers, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        return self._with_retry(_do)

    def post(self, url: str, headers: Optional[Dict[str, str]] = None,
             json_body: Optional[Any] = None, timeout: float = 5.0) -> Any:
        def _do():
            r = self.sess.post(url, headers=headers, json=json_body, timeout=timeout)
            r.raise_for_status()
            return r.json()
        return self._with_retry(_do)

HTTP = Http()
