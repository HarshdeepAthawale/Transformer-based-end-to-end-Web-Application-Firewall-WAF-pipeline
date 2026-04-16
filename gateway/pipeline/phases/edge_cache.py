"""Phase wrapper for edge cache lookup (GET/HEAD only)."""

from typing import Optional

from gateway.pipeline.base import Phase, PhaseVerdict
from gateway.pipeline.context import PhaseContext, PhaseResult


class EdgeCachePhase(Phase):
    name = "edge_cache"
    order = 20
    requires_body = False

    def __init__(self, edge_cache=None):
        self._cache = edge_cache

    def should_run(self, ctx: PhaseContext) -> tuple[bool, Optional[str]]:
        if self._cache is None:
            return False, "no_edge_cache"
        if ctx.method not in ("GET", "HEAD"):
            return False, "non_cacheable_method"
        return True, None

    async def execute(self, ctx: PhaseContext) -> PhaseResult:
        full_url = ctx.path + ("?" + ctx.query_string if ctx.query_string else "")
        if not self._cache.is_cacheable_request(ctx.method, ctx.headers):
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.CONTINUE,
                data={"cache_status": "not_cacheable"},
            )

        entry, cache_key, hit_status = await self._cache.lookup(
            ctx.method, full_url, ctx.headers,
        )

        if hit_status in ("HIT", "STALE") and entry:
            # Store cache data for the caller to build a Response
            ctx.cache_ctx = (cache_key, full_url)
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="cache_hit",
                status_code=entry.status_code,
                response_headers={"X-Cache": hit_status},
                data={
                    "cache_status": hit_status,
                    "body": entry.body,
                    "status_code": entry.status_code,
                    "headers": dict(entry.headers),
                },
            )

        if hit_status == "REVALIDATED" and entry:
            return PhaseResult(
                phase_name=self.name,
                verdict=PhaseVerdict.SHORT_CIRCUIT,
                action="cache_hit",
                status_code=304,
                response_headers={"X-Cache": "REVALIDATED"},
                data={"cache_status": "REVALIDATED"},
            )

        # Cache miss -- store context for post-forward caching
        ctx.cache_ctx = (cache_key, full_url)
        return PhaseResult(
            phase_name=self.name,
            verdict=PhaseVerdict.CONTINUE,
            data={"cache_status": "MISS", "cache_key": cache_key},
        )
