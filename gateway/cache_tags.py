"""
Cache tag extraction from response headers and URL paths.
Tags enable targeted cache purging (e.g. purge all pages for a product).
"""


def extract_cache_tags(response_headers: dict, path: str) -> list[str]:
    """Extract cache tags from response headers and auto-generate path-based tags.

    Sources:
    - Cache-Tag header (comma-separated)
    - Surrogate-Key header (space-separated, Fastly convention)
    - Auto-generated path prefix tags

    Example: path=/api/users/123 -> tags:
      ['path:/api/users/123', 'prefix:/api/users', 'prefix:/api']
    """
    tags: list[str] = []

    # Cache-Tag header (comma-separated)
    cache_tag = response_headers.get("cache-tag") or response_headers.get("Cache-Tag")
    if cache_tag:
        tags.extend(t.strip() for t in cache_tag.split(",") if t.strip())

    # Surrogate-Key header (space-separated, Fastly/Varnish convention)
    surrogate_key = response_headers.get("surrogate-key") or response_headers.get("Surrogate-Key")
    if surrogate_key:
        tags.extend(k.strip() for k in surrogate_key.split() if k.strip())

    # Auto-generate path-based tags
    if path:
        clean_path = path.split("?")[0].rstrip("/")
        if clean_path:
            tags.append(f"path:{clean_path}")
            parts = clean_path.split("/")
            # Generate prefix tags for each path segment
            for i in range(2, len(parts)):
                prefix = "/".join(parts[:i])
                if prefix:
                    tags.append(f"prefix:{prefix}")

    return tags
