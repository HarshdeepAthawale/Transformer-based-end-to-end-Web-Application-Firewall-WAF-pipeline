#!/usr/bin/env python3
"""
SQL Injection Attack Test Suite
Tests various SQL injection techniques against the WAF
"""

import requests
import json
import time
from typing import List, Dict

WAF_API = "http://localhost:3001/api/waf/check"

# SQL Injection Payloads - Mixed with normal-looking requests
SQL_INJECTION_ATTACKS = [
    # Classic SQL Injection
    {"name": "Classic OR bypass", "path": "/api/users", "query": {"id": "1 OR 1=1"}},
    {"name": "OR with comment", "path": "/api/login", "query": {"user": "admin'--"}},
    {"name": "OR with quotes", "path": "/api/search", "query": {"q": "' OR '1'='1"}},

    # UNION-based Injection
    {"name": "UNION SELECT", "path": "/api/products", "query": {"id": "1 UNION SELECT * FROM users--"}},
    {"name": "UNION with NULL", "path": "/api/items", "query": {"cat": "1 UNION SELECT NULL,NULL,NULL--"}},
    {"name": "UNION column enum", "path": "/api/data", "query": {"id": "1 UNION SELECT username,password FROM users--"}},

    # Error-based Injection
    {"name": "Error-based extractvalue", "path": "/api/user", "query": {"id": "1 AND EXTRACTVALUE(1,CONCAT(0x7e,(SELECT @@version)))"}},
    {"name": "Error-based updatexml", "path": "/api/profile", "query": {"uid": "1 AND UPDATEXML(1,CONCAT(0x7e,(SELECT user())),1)"}},

    # Time-based Blind Injection
    {"name": "Time-based SLEEP", "path": "/api/check", "query": {"id": "1; WAITFOR DELAY '0:0:5'--"}},
    {"name": "Time-based IF", "path": "/api/verify", "query": {"token": "1' AND IF(1=1,SLEEP(5),0)--"}},
    {"name": "MySQL benchmark", "path": "/api/validate", "query": {"code": "1' AND BENCHMARK(10000000,SHA1('test'))--"}},

    # Boolean-based Blind
    {"name": "Boolean AND true", "path": "/api/exists", "query": {"email": "test@test.com' AND 1=1--"}},
    {"name": "Boolean AND false", "path": "/api/exists", "query": {"email": "test@test.com' AND 1=2--"}},
    {"name": "Boolean substring", "path": "/api/find", "query": {"name": "admin' AND SUBSTRING(password,1,1)='a'--"}},

    # Stacked Queries
    {"name": "Stacked DROP", "path": "/api/update", "query": {"id": "1; DROP TABLE users;--"}},
    {"name": "Stacked INSERT", "path": "/api/save", "query": {"data": "1; INSERT INTO admins VALUES('hacker','pass');--"}},
    {"name": "Stacked UPDATE", "path": "/api/modify", "query": {"item": "1; UPDATE users SET role='admin';--"}},

    # Second Order Injection
    {"name": "Stored XSS+SQL", "path": "/api/register", "body": {"username": "admin'--", "email": "test@test.com"}},
    {"name": "Profile poison", "path": "/api/profile/update", "body": {"bio": "Hello'; DROP TABLE posts;--"}},

    # Encoding Bypasses
    {"name": "URL encoded", "path": "/api/search", "query": {"q": "%27%20OR%20%271%27%3D%271"}},
    {"name": "Double URL encode", "path": "/api/find", "query": {"term": "%2527%2520OR%25201%253D1"}},
    {"name": "Unicode bypass", "path": "/api/query", "query": {"input": "admin%u0027--"}},
    {"name": "Hex encoding", "path": "/api/exec", "query": {"cmd": "0x27204f5220313d31"}},

    # NoSQL Injection (MongoDB style)
    {"name": "MongoDB $ne", "path": "/api/auth", "body": {"username": {"$ne": ""}, "password": {"$ne": ""}}},
    {"name": "MongoDB $gt", "path": "/api/users", "query": {"age": {"$gt": ""}}},
    {"name": "MongoDB $regex", "path": "/api/search", "body": {"name": {"$regex": ".*"}}},
    {"name": "MongoDB $where", "path": "/api/find", "body": {"$where": "this.password.length > 0"}},

    # PostgreSQL specific
    {"name": "PG version", "path": "/api/info", "query": {"id": "1; SELECT version();--"}},
    {"name": "PG copy", "path": "/api/export", "query": {"table": "users; COPY users TO '/tmp/dump.txt';--"}},

    # MySQL specific
    {"name": "MySQL into outfile", "path": "/api/backup", "query": {"data": "1' INTO OUTFILE '/var/www/shell.php'--"}},
    {"name": "MySQL load_file", "path": "/api/read", "query": {"file": "1' UNION SELECT LOAD_FILE('/etc/passwd')--"}},

    # MSSQL specific
    {"name": "MSSQL xp_cmdshell", "path": "/api/admin", "query": {"cmd": "'; EXEC xp_cmdshell('whoami');--"}},
    {"name": "MSSQL openrowset", "path": "/api/import", "query": {"source": "'; SELECT * FROM OPENROWSET('SQLOLEDB','server';'sa';'pass','SELECT 1');--"}},

    # WAF Bypass Techniques
    {"name": "Comment bypass", "path": "/api/test", "query": {"id": "1'/**/OR/**/1=1--"}},
    {"name": "Case variation", "path": "/api/test", "query": {"id": "1' oR 1=1--"}},
    {"name": "Whitespace bypass", "path": "/api/test", "query": {"id": "1'\t\nOR\t\n1=1--"}},
    {"name": "Null byte", "path": "/api/test", "query": {"id": "1'%00OR 1=1--"}},

    # Mixed with legitimate-looking data
    {"name": "Hidden in JSON", "path": "/api/order", "body": {"product": "Laptop", "quantity": "1; DROP TABLE orders;--"}},
    {"name": "Hidden in search", "path": "/api/products/search", "query": {"q": "iPhone' UNION SELECT credit_card FROM payments--"}},
]

def test_payload(attack: Dict) -> Dict:
    """Test a single attack payload against WAF"""
    payload = {
        "method": "POST" if "body" in attack else "GET",
        "path": attack["path"],
        "query_params": attack.get("query", {}),
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    }

    if "body" in attack:
        payload["body"] = json.dumps(attack["body"]) if isinstance(attack["body"], dict) else attack["body"]

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
    print("SQL INJECTION ATTACK TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(SQL_INJECTION_ATTACKS)} SQL injection payloads...\n")

    detected = 0
    missed = 0
    errors = 0

    for i, attack in enumerate(SQL_INJECTION_ATTACKS, 1):
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
        print(f"[{i:02d}/{len(SQL_INJECTION_ATTACKS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}")
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests:  {len(SQL_INJECTION_ATTACKS)}")
    print(f"\033[92mBlocked:      {detected}\033[0m")
    print(f"\033[91mMissed:       {missed}\033[0m")
    print(f"\033[93mErrors:       {errors}\033[0m")
    print(f"Detection Rate: {detected/(len(SQL_INJECTION_ATTACKS)-errors)*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
