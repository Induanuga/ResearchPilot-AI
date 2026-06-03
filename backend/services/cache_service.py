"""
Disk-based caching service using diskcache.
Provides TTL-aware caching for expensive operations
(ArXiv queries, LLM calls, embeddings).
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable, Optional

import diskcache
from loguru import logger

from backend.config import get_settings


class CacheService:
    """
    Singleton disk cache backed by diskcache.Cache.
    Supports get/set with TTL and a decorator for auto-caching functions.
    """

    _instance: "CacheService | None" = None
    _cache: Optional[diskcache.Cache] = None

    def __new__(cls) -> "CacheService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._cache is None:
            settings = get_settings()
            os.makedirs(settings.cache_path, exist_ok=True)
            self._cache = diskcache.Cache(settings.cache_path)
            self._ttl = settings.cache_ttl
            logger.info(f"Cache initialised at {settings.cache_path} (TTL={self._ttl}s)")

    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, or None if missing/expired."""
        try:
            return self._cache.get(key)
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL override."""
        try:
            self._cache.set(key, value, expire=ttl or self._ttl)
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {e}")

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        try:
            self._cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            self._cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def make_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a deterministic cache key from arguments."""
        raw = json.dumps({"args": list(args), "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def cached(self, ttl: Optional[int] = None) -> Callable:
        """
        Decorator that caches function return values.

        Usage:
            @cache.cached(ttl=3600)
            def expensive_function(arg1, arg2):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = f"{func.__module__}.{func.__qualname__}:{self.make_key(*args, **kwargs)}"
                cached_result = self.get(key)
                if cached_result is not None:
                    logger.debug(f"Cache HIT: {func.__qualname__}")
                    return cached_result
                result = func(*args, **kwargs)
                self.set(key, result, ttl=ttl)
                logger.debug(f"Cache SET: {func.__qualname__}")
                return result
            return wrapper
        return decorator


# Module-level singleton
cache = CacheService()
