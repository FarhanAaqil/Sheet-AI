# =============================================================================
# SheetSense AI — Rate Limiting Middleware (Architecture §7, PRD FR-9)
# =============================================================================
# Implements sliding-window rate limiting per API key:
# - POST /chat: 60 requests/minute
# - POST /actions/{action_id}/confirm: 20 requests/minute
# Backed by Redis with in-memory fallback. Returns 429 Too Many Requests with
# a Retry-After header when limits are exceeded.
# =============================================================================

import os
import time
import uuid
import logging
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """
    Sliding-window rate limiter.
    Uses Redis if available, with in-memory deque fallback.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self._memory_windows: Dict[str, deque] = defaultdict(deque)

        if os.getenv("ENV", "").lower() != "local_no_redis":
            try:
                import redis
                client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                client.ping()
                self.redis_client = client
                logger.info(f"RateLimiter connected to Redis at '{self.redis_url}'.")
            except Exception as e:
                logger.warning(
                    f"RateLimiter could not connect to Redis ({e}). Using in-memory fallback."
                )


    def check_rate_limit(
        self,
        identifier: str,
        endpoint_group: str,
        max_requests: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int]:
        """
        Checks if identifier has exceeded max_requests in the rolling window.
        Returns: (is_allowed: bool, retry_after_seconds: int)
        """
        key = f"ratelimit:{identifier}:{endpoint_group}"
        now = time.time()

        if self.redis_client is not None:
            try:
                pipe = self.redis_client.pipeline()
                # 1. Purge requests older than window
                pipe.zremrangebyscore(key, 0, now - window_seconds)
                # 2. Add current request
                req_id = str(uuid.uuid4())
                pipe.zadd(key, {req_id: now})
                # 3. Count requests in current window
                pipe.zcard(key)
                # 4. Set TTL on key
                pipe.expire(key, window_seconds + 5)
                # 5. Get oldest request in window to calculate retry_after
                pipe.zrange(key, 0, 0, withscores=True)
                _, _, count, _, oldest = pipe.execute()

                if count > max_requests:
                    oldest_ts = oldest[0][1] if oldest else now
                    retry_after = max(1, int(window_seconds - (now - oldest_ts)))
                    return False, retry_after
                return True, 0
            except Exception as e:
                logger.warning(f"Redis rate limit check error ({e}); falling back to memory.")

        # In-memory sliding window fallback
        timestamps = self._memory_windows[key]
        # Remove expired timestamps
        cutoff = now - window_seconds
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            oldest_ts = timestamps[0]
            retry_after = max(1, int(window_seconds - (now - oldest_ts)))
            return False, retry_after

        timestamps.append(now)
        return True, 0


# Global singleton instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


def rate_limit(max_requests: int, window_seconds: int = 60, endpoint_group: str = "default"):
    """
    FastAPI dependency factory enforcing sliding-window rate limits.
    """
    async def dependency(
        request: Request,
        api_key: Optional[str] = Security(api_key_header),
    ):
        identifier = api_key or (request.client.host if request.client else "anonymous")
        limiter = get_rate_limiter()
        allowed, retry_after = limiter.check_rate_limit(
            identifier=identifier,
            endpoint_group=endpoint_group,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit of {max_requests} req/{window_seconds}s exceeded for '{endpoint_group}'. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        return True

    return dependency
