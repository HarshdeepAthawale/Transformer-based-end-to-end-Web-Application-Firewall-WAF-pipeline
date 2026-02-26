"""
Edge HTTP Cache — Two-tier CDN caching engine.

Tier 1: In-process LRU dict for <1ms hot cache.
Tier 2: Redis for shared cache across gateway instances.

Supports:
- Cache-Control parsing (max-age, s-maxage, no-cache, no-store, private, public)
- Vary header handling (different cache keys per Vary header values)
- ETag / If-None-Match conditional requests
- Last-Modified / If-Modified-Since conditional requests
- stale-while-revalidate (serve stale, refresh in background)
- Request coalescing (deduplicate concurrent cache-miss fetches)
- Cache tags for targeted purging
- Stats tracking (hits, misses, bytes saved)
"""

import asyncio
import hashlib
import json
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import redis.asyncio as aioredis

from gateway.config import gateway_config
from gateway.cache_tags import extract_cache_tags

logger = logging.getLogger("gateway.edge_cache")

# ─── Data structures ────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A cached HTTP response."""
    body: bytes
    status_code: int
    headers: dict[str, str]
    created_at: float
    max_age: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    stale_while_revalidate: int = 0
    cache_tags: list[str] = field(default_factory=list)
    vary_fields: list[str] = field(default_factory=list)

    @property
    def age(self) -> int:
        return int(time.time() - self.created_at)

    @property
    def is_fresh(self) -> bool:
        return self.age < self.max_age

    @property
    def is_stale_usable(self) -> bool:
        return self.age < (self.max_age + self.stale_while_revalidate)

    def to_redis(self) -> bytes:
        return json.dumps({
            "body_hex": self.body.hex(),
            "status_code": self.status_code,
            "headers": self.headers,
            "created_at": self.created_at,
            "max_age": self.max_age,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "stale_while_revalidate": self.stale_while_revalidate,
            "cache_tags": self.cache_tags,
            "vary_fields": self.vary_fields,
        }).encode()

    @classmethod
    def from_redis(cls, data: bytes) -> "CacheEntry":
        d = json.loads(data)
        return cls(
            body=bytes.fromhex(d["body_hex"]),
            status_code=d["status_code"],
            headers=d["headers"],
            created_at=d["created_at"],
            max_age=d["max_age"],
            etag=d.get("etag"),
            last_modified=d.get("last_modified"),
            stale_while_revalidate=d.get("stale_while_revalidate", 0),
            cache_tags=d.get("cache_tags", []),
            vary_fields=d.get("vary_fields", []),
        )


# ─── Cache key generation ───────────────────────────────────────────

def generate_cache_key(method: str, url: str, vary_headers: dict[str, str]) -> str:
    """Deterministic cache key from method + URL + Vary header values."""
    parts = [method.upper(), url]
    for k in sorted(vary_headers.keys()):
        parts.append(f"{k.lower()}={vary_headers[k]}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_vary_headers(request_headers: dict, vary_fields: list[str]) -> dict[str, str]:
    """Extract Vary-listed header values from request."""
    result = {}
    for field_name in vary_fields:
        key = field_name.strip().lower()
        if key == "*":
            return {"*": "*"}  # uncacheable
        for h_name, h_val in request_headers.items():
            if h_name.lower() == key:
                result[key] = h_val
                break
        else:
            result[key] = ""
    return result


# ─── Cache-Control parsing ──────────────────────────────────────────

def parse_cache_control(header: str) -> dict:
    """Parse Cache-Control header into dict of directives."""
    directives: dict = {}
    if not header:
        return directives
    for part in header.split(","):
        part = part.strip().lower()
        if "=" in part:
            key, val = part.split("=", 1)
            try:
                directives[key.strip()] = int(val.strip())
            except ValueError:
                directives[key.strip()] = val.strip()
        else:
            directives[part] = True
    return directives


# ─── Two-tier cache store ───────────────────────────────────────────

class EdgeCacheStore:
    """Two-tier cache: process-local LRU + Redis for cross-instance sharing."""

    def __init__(self, redis_url: str, local_max_entries: int, local_max_bytes: int, redis_prefix: str):
        self._redis_url = redis_url
        self._redis_prefix = redis_prefix
        self._redis: Optional[aioredis.Redis] = None

        # Local LRU cache
        self._local: OrderedDict[str, CacheEntry] = OrderedDict()
        self._local_max_entries = local_max_entries
        self._local_max_bytes = local_max_bytes
        self._local_bytes = 0

        # Stats
        self._hits = 0
        self._misses = 0
        self._bytes_saved = 0
        self._bytes_total = 0

    async def connect(self):
        try:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=False, socket_timeout=2)
            await self._redis.ping()
            logger.info("Edge cache Redis connected")
        except Exception as e:
            logger.warning(f"Edge cache Redis unavailable, local-only mode: {e}")
            self._redis = None

    async def close(self):
        if self._redis:
            await self._redis.close()

    def _redis_key(self, key: str) -> str:
        return f"{self._redis_prefix}{key}"

    def _evict_local(self):
        """Evict oldest entries if over limits."""
        while self._local_bytes > self._local_max_bytes or len(self._local) > self._local_max_entries:
            if not self._local:
                break
            _, entry = self._local.popitem(last=False)
            self._local_bytes -= len(entry.body)

    async def get(self, key: str) -> Optional[CacheEntry]:
        # Try local first
        if key in self._local:
            entry = self._local[key]
            self._local.move_to_end(key)
            if entry.is_fresh or entry.is_stale_usable:
                self._hits += 1
                self._bytes_saved += len(entry.body)
                self._bytes_total += len(entry.body)
                await self._incr_redis_stats("hits", len(entry.body))
                return entry
            else:
                del self._local[key]
                self._local_bytes -= len(entry.body)

        # Try Redis
        if self._redis:
            try:
                data = await self._redis.get(self._redis_key(key))
                if data:
                    entry = CacheEntry.from_redis(data)
                    if entry.is_fresh or entry.is_stale_usable:
                        # Promote to local cache
                        self._local[key] = entry
                        self._local_bytes += len(entry.body)
                        self._evict_local()
                        self._hits += 1
                        self._bytes_saved += len(entry.body)
                        self._bytes_total += len(entry.body)
                        await self._incr_redis_stats("hits", len(entry.body))
                        return entry
            except Exception as e:
                logger.debug(f"Redis cache get error: {e}")

        self._misses += 1
        self._bytes_total += 0
        await self._incr_redis_stats("misses", 0)
        return None

    async def set(self, key: str, entry: CacheEntry, url_index: Optional[str] = None) -> None:
        # Store locally
        if key in self._local:
            self._local_bytes -= len(self._local[key].body)
        self._local[key] = entry
        self._local_bytes += len(entry.body)
        self._local.move_to_end(key)
        self._evict_local()

        # Store in Redis with TTL
        if self._redis:
            try:
                ttl = entry.max_age + entry.stale_while_revalidate
                await self._redis.setex(self._redis_key(key), ttl, entry.to_redis())
                # Store tag mappings
                for tag in entry.cache_tags:
                    tag_key = f"{self._redis_prefix}tag:{tag}"
                    await self._redis.sadd(tag_key, key)
                    await self._redis.expire(tag_key, ttl)
                # URL index for purge by URL (all variants for this method+url)
                if url_index:
                    url_index_key = f"{self._redis_prefix}urlindex:{hashlib.md5(url_index.encode()).hexdigest()}"
                    await self._redis.sadd(url_index_key, key)
                    await self._redis.expire(url_index_key, ttl)
            except Exception as e:
                logger.debug(f"Redis cache set error: {e}")

    async def delete(self, key: str) -> bool:
        deleted = False
        if key in self._local:
            self._local_bytes -= len(self._local[key].body)
            del self._local[key]
            deleted = True
        if self._redis:
            try:
                result = await self._redis.delete(self._redis_key(key))
                deleted = deleted or result > 0
            except Exception:
                pass
        return deleted

    async def delete_by_prefix(self, prefix: str) -> int:
        count = 0
        # Local
        keys_to_delete = [k for k in self._local if k.startswith(prefix)]
        for k in keys_to_delete:
            self._local_bytes -= len(self._local[k].body)
            del self._local[k]
            count += 1
        # Redis (scan for prefix)
        if self._redis:
            try:
                cursor = 0
                pattern = f"{self._redis_prefix}{prefix}*"
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug(f"Redis prefix delete error: {e}")
        return count

    async def delete_by_url(self, method: str, url: str) -> int:
        """Purge all cache entries for this method+url (all Vary variants). Returns count purged."""
        url_index_key = f"{self._redis_prefix}urlindex:{hashlib.md5((method.upper() + '|' + url).encode()).hexdigest()}"
        count = 0
        if self._redis:
            try:
                members = await self._redis.smembers(url_index_key)
                for key in members:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    if key_str in self._local:
                        self._local_bytes -= len(self._local[key_str].body)
                        del self._local[key_str]
                    await self._redis.delete(self._redis_key(key_str))
                    count += 1
                await self._redis.delete(url_index_key)
            except Exception as e:
                logger.debug(f"Redis delete_by_url error: {e}")
        return count

    async def delete_by_tag(self, tag: str) -> int:
        count = 0
        tag_key = f"{self._redis_prefix}tag:{tag}"
        if self._redis:
            try:
                members = await self._redis.smembers(tag_key)
                for key in members:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    if key_str in self._local:
                        self._local_bytes -= len(self._local[key_str].body)
                        del self._local[key_str]
                    await self._redis.delete(self._redis_key(key_str))
                    count += 1
                await self._redis.delete(tag_key)
            except Exception as e:
                logger.debug(f"Redis tag delete error: {e}")
        else:
            # Local only: scan all entries for tag
            keys_to_delete = [
                k for k, v in self._local.items() if tag in v.cache_tags
            ]
            for k in keys_to_delete:
                self._local_bytes -= len(self._local[k].body)
                del self._local[k]
                count += 1
        return count

    async def delete_all(self) -> int:
        count = len(self._local)
        self._local.clear()
        self._local_bytes = 0
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=f"{self._redis_prefix}*", count=200)
                    if keys:
                        await self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug(f"Redis delete_all error: {e}")
        return count

    async def _incr_redis_stats(self, stat_type: str, bytes_amount: int):
        """Increment daily stats counters in Redis."""
        if not self._redis:
            return
        try:
            from datetime import datetime, timezone
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            pipe = self._redis.pipeline()
            stat_key = f"{self._redis_prefix}stats:{day}:{stat_type}"
            pipe.incr(stat_key)
            pipe.expire(stat_key, 86400 * 7)  # keep 7 days
            if bytes_amount > 0 and stat_type == "hits":
                bytes_key = f"{self._redis_prefix}stats:{day}:bytes_saved"
                pipe.incrby(bytes_key, bytes_amount)
                pipe.expire(bytes_key, 86400 * 7)
            await pipe.execute()
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            "local_entries": len(self._local),
            "local_bytes": self._local_bytes,
            "total_hits": self._hits,
            "total_misses": self._misses,
            "bytes_saved": self._bytes_saved,
            "bytes_total": self._bytes_total,
            "hit_ratio": round(self._hits / max(self._hits + self._misses, 1), 4),
        }


# ─── Request coalescer ──────────────────────────────────────────────

class RequestCoalescer:
    """Deduplicate concurrent cache-miss requests to the same URL.

    When multiple requests arrive for the same uncached resource,
    only the first one fetches from origin. Others wait and share
    the result.
    """

    def __init__(self, timeout: float = 30.0):
        self._pending: dict[str, asyncio.Event] = {}
        self._results: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._timeout = timeout

    async def get_or_fetch(self, key: str, fetch_fn: Callable) -> tuple[CacheEntry, str]:
        """Return (entry, status) where status is 'MISS' for fetcher or 'COALESCED' for waiters."""
        async with self._lock:
            if key in self._pending:
                event = self._pending[key]
                is_waiter = True
            else:
                event = asyncio.Event()
                self._pending[key] = event
                is_waiter = False

        if is_waiter:
            try:
                await asyncio.wait_for(event.wait(), timeout=self._timeout)
                entry = self._results.get(key)
                if entry:
                    return entry, "COALESCED"
            except asyncio.TimeoutError:
                pass
            # Fallback: fetch ourselves
            return await self._do_fetch(key, fetch_fn), "MISS"
        else:
            entry = await self._do_fetch(key, fetch_fn)
            return entry, "MISS"

    async def _do_fetch(self, key: str, fetch_fn: Callable) -> CacheEntry:
        try:
            entry = await fetch_fn()
            self._results[key] = entry
            return entry
        finally:
            async with self._lock:
                event = self._pending.pop(key, None)
                if event:
                    event.set()
            # Clean up result after a short delay
            await asyncio.sleep(0.1)
            self._results.pop(key, None)


# ─── Main edge cache orchestrator ───────────────────────────────────

class EdgeCache:
    """High-level cache orchestrator integrating store, coalescing, and HTTP semantics."""

    def __init__(self, store: EdgeCacheStore, coalescer: Optional[RequestCoalescer] = None):
        self.store = store
        self.coalescer = coalescer

    def is_cacheable_request(self, method: str, headers: dict) -> bool:
        """Only cache GET/HEAD without Authorization or Cache-Control: no-store."""
        if method.upper() not in ("GET", "HEAD"):
            return False
        # Check Authorization header
        for h_name in headers:
            if h_name.lower() == "authorization":
                return False
        # Check Cache-Control: no-store
        cc = ""
        for h_name, h_val in headers.items():
            if h_name.lower() == "cache-control":
                cc = h_val
                break
        if cc:
            directives = parse_cache_control(cc)
            if "no-store" in directives or "no-cache" in directives:
                return False
        # Check bypass cookie
        if gateway_config.EDGE_CACHE_BYPASS_COOKIE:
            cookie = ""
            for h_name, h_val in headers.items():
                if h_name.lower() == "cookie":
                    cookie = h_val
                    break
            if gateway_config.EDGE_CACHE_BYPASS_COOKIE in cookie:
                return False
        return True

    def is_cacheable_response(self, status_code: int, headers: dict) -> bool:
        """Check if response is cacheable based on status and headers."""
        if status_code not in (200, 203, 204, 206, 300, 301, 308, 404, 410):
            return False
        cc = ""
        for h_name, h_val in headers.items():
            if h_name.lower() == "cache-control":
                cc = h_val
                break
        directives = parse_cache_control(cc)
        if "no-store" in directives or "private" in directives:
            return False
        # Need explicit cacheability signal
        has_max_age = "max-age" in directives or "s-maxage" in directives
        has_expires = any(h.lower() == "expires" for h in headers)
        if not has_max_age and not has_expires and not cc:
            # No cache headers at all — apply default TTL if configured
            if gateway_config.EDGE_CACHE_DEFAULT_TTL > 0:
                return True
            return False
        return True

    def _compute_ttl(self, headers: dict) -> tuple[int, int]:
        """Compute (max_age, stale_while_revalidate) from response headers."""
        cc = ""
        for h_name, h_val in headers.items():
            if h_name.lower() == "cache-control":
                cc = h_val
                break
        directives = parse_cache_control(cc)

        # s-maxage takes priority over max-age for shared caches
        max_age = directives.get("s-maxage", directives.get("max-age", gateway_config.EDGE_CACHE_DEFAULT_TTL))
        if not isinstance(max_age, int):
            max_age = gateway_config.EDGE_CACHE_DEFAULT_TTL
        swr = directives.get("stale-while-revalidate", 0)
        if not isinstance(swr, int):
            swr = 0
        return max_age, swr

    def _get_vary_fields(self, headers: dict) -> list[str]:
        """Extract Vary field names from response."""
        for h_name, h_val in headers.items():
            if h_name.lower() == "vary":
                fields = [f.strip().lower() for f in h_val.split(",")]
                if "*" in fields:
                    return ["*"]
                return fields
        return []

    async def lookup(self, method: str, url: str, request_headers: dict) -> tuple[Optional[CacheEntry], str, str]:
        """Look up cache. Returns (entry_or_None, cache_key, hit_status)."""
        # We need vary fields from a previous response to compute the key correctly.
        # On first request we don't know vary, so use empty.
        # Check if we have a vary-hint stored in Redis.
        vary_fields = await self._get_stored_vary(url)
        vary_hdrs = get_vary_headers(request_headers, vary_fields)
        if "*" in vary_hdrs:
            return None, "", "BYPASS"

        cache_key = generate_cache_key(method, url, vary_hdrs)
        entry = await self.store.get(cache_key)

        if entry is None:
            return None, cache_key, "MISS"

        # Check conditional request headers from client
        if_none_match = None
        for h_name, h_val in request_headers.items():
            h_lower = h_name.lower()
            if h_lower == "if-none-match":
                if_none_match = h_val

        if if_none_match and entry.etag and if_none_match.strip('"') == entry.etag.strip('"'):
            return entry, cache_key, "REVALIDATED"

        if entry.is_fresh:
            return entry, cache_key, "HIT"
        elif entry.is_stale_usable:
            return entry, cache_key, "STALE"
        else:
            return None, cache_key, "MISS"

    async def store_response(self, cache_key: str, status_code: int, body: bytes,
                             response_headers: dict, path: str, method: str = "GET", request_url: Optional[str] = None) -> None:
        """Store a response in the cache. request_url is the full URL (path?query) used for the cache key / URL index."""
        vary_fields = self._get_vary_fields(response_headers)
        if "*" in vary_fields:
            return  # Vary: * is uncacheable

        max_age, swr = self._compute_ttl(response_headers)

        etag = None
        last_modified = None
        for h_name, h_val in response_headers.items():
            h_lower = h_name.lower()
            if h_lower == "etag":
                etag = h_val
            elif h_lower == "last-modified":
                last_modified = h_val

        tags = extract_cache_tags(response_headers, path)

        entry = CacheEntry(
            body=body,
            status_code=status_code,
            headers=dict(response_headers),
            created_at=time.time(),
            max_age=max_age,
            etag=etag,
            last_modified=last_modified,
            stale_while_revalidate=swr,
            cache_tags=tags,
            vary_fields=vary_fields,
        )

        url_index = (method.upper() + "|" + (request_url or path)) if request_url or path else None
        await self.store.set(cache_key, entry, url_index=url_index)

        # Store vary hint for this URL
        if vary_fields:
            await self._store_vary(request_url or path, vary_fields)

    async def _get_stored_vary(self, url: str) -> list[str]:
        """Get stored Vary fields for a URL from Redis."""
        if not self.store._redis:
            return []
        try:
            key = f"{self.store._redis_prefix}vary:{hashlib.md5(url.encode()).hexdigest()}"
            data = await self.store._redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return []

    async def _store_vary(self, url: str, vary_fields: list[str]) -> None:
        """Store Vary fields for a URL in Redis."""
        if not self.store._redis:
            return
        try:
            key = f"{self.store._redis_prefix}vary:{hashlib.md5(url.encode()).hexdigest()}"
            await self.store._redis.setex(key, 86400, json.dumps(vary_fields))
        except Exception:
            pass

    async def close(self):
        await self.store.close()


# ─── Purge subscription (Redis pub/sub) ─────────────────────────────

class CachePurgeSubscriber:
    """Subscribe to Redis pub/sub for cache purge commands from the backend."""

    def __init__(self, cache: EdgeCache, redis_url: str):
        self._cache = cache
        self._redis_url = redis_url
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._listen())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _listen(self):
        while True:
            try:
                r = aioredis.from_url(self._redis_url, decode_responses=True, socket_timeout=5)
                pubsub = r.pubsub()
                await pubsub.subscribe("waf:cache:purge")
                logger.info("Cache purge subscriber connected")
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        await self._handle_purge(message["data"])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cache purge subscriber error, reconnecting: {e}")
                await asyncio.sleep(2)

    async def _handle_purge(self, data: str):
        try:
            cmd = json.loads(data)
            purge_type = cmd.get("type", "all")
            values = cmd.get("values", [])

            if purge_type == "all":
                count = await self._cache.store.delete_all()
            elif purge_type == "url":
                count = 0
                for item in values:
                    if isinstance(item, dict):
                        method = item.get("method", "GET").upper()
                        url = item.get("url", "")
                    else:
                        method = "GET"
                        url = str(item)
                    if url:
                        count += await self._cache.store.delete_by_url(method, url)
            elif purge_type == "tag":
                count = 0
                for tag in values:
                    count += await self._cache.store.delete_by_tag(tag)
            elif purge_type == "prefix":
                count = 0
                for prefix in values:
                    count += await self._cache.store.delete_by_prefix(prefix)
            else:
                count = 0

            logger.info(f"Cache purge: type={purge_type}, keys_purged={count}")
        except Exception as e:
            logger.error(f"Cache purge handler error: {e}")


# ─── Factory ────────────────────────────────────────────────────────

def create_edge_cache() -> Optional[EdgeCache]:
    """Create EdgeCache if enabled, else return None."""
    if not gateway_config.EDGE_CACHE_ENABLED:
        return None

    store = EdgeCacheStore(
        redis_url=gateway_config.REDIS_URL,
        local_max_entries=gateway_config.EDGE_CACHE_LOCAL_MAX_ENTRIES,
        local_max_bytes=gateway_config.EDGE_CACHE_LOCAL_MAX_BYTES,
        redis_prefix=gateway_config.EDGE_CACHE_REDIS_PREFIX,
    )

    coalescer = None
    if gateway_config.EDGE_CACHE_COALESCE_ENABLED:
        coalescer = RequestCoalescer(timeout=gateway_config.EDGE_CACHE_COALESCE_TIMEOUT)

    return EdgeCache(store=store, coalescer=coalescer)


def create_purge_subscriber(cache: EdgeCache) -> CachePurgeSubscriber:
    """Create pub/sub subscriber for cache purge commands."""
    return CachePurgeSubscriber(cache=cache, redis_url=gateway_config.REDIS_URL)
