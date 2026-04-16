"""
Conditional filter helpers for pipeline phases.

Every filter is O(1) -- set lookups, string prefix checks, or boolean flags.
These are the "don't run expensive work" escape hatches from FL2.
"""

# Static asset extensions that skip ML inference
STATIC_EXTENSIONS = frozenset({
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".svg", ".woff", ".woff2", ".ttf", ".eot", ".map",
    ".webp", ".avif", ".mp4", ".webm", ".mp3", ".ogg",
    ".pdf", ".zip", ".gz", ".br", ".wasm",
})

# Path segments that indicate suspicious intent even on GET
_SUSPICIOUS_PATTERNS = ("../", "..\\", "%2e", "etc/passwd", "etc/shadow", "cmd", "exec")


def is_static_asset(path: str) -> bool:
    """O(1) check: does the path end with a known static asset extension?"""
    dot_idx = path.rfind(".")
    if dot_idx == -1:
        return False
    ext = path[dot_idx:].lower().split("?")[0]
    return ext in STATIC_EXTENSIONS


def is_benign_get(method: str, path: str, query_string: str) -> bool:
    """
    GET with no query string and no suspicious path segments.
    Used to skip ML inference on trivially safe requests.
    """
    if method != "GET":
        return False
    if query_string:
        return False
    path_lower = path.lower()
    return not any(s in path_lower for s in _SUSPICIOUS_PATTERNS)


def is_multipart(content_type: str) -> bool:
    """Check if the request has multipart form data (file upload)."""
    return "multipart/form-data" in (content_type or "").lower()


def is_login_path(path: str, method: str) -> bool:
    """Check if the request targets a login/auth endpoint."""
    if method != "POST":
        return False
    path_lower = path.lower()
    return any(seg in path_lower for seg in ("/login", "/auth", "/signin", "/sign-in"))
