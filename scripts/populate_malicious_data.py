#!/usr/bin/env python3
"""
Populate data/malicious/ with categorized payload files.

Extracts payloads from:
  - scripts/attack_tests/07_header_injection.py
  - scripts/attack_tests/08_ldap_xpath_injection.py
  - scripts/attack_tests/09_dos_patterns.py
  - PayloadsAllTheThings/CRLF Injection/
  - PayloadsAllTheThings/Server Side Template Injection/
  - PayloadsAllTheThings/LDAP Injection/ (if exists)
  - tests/payloads/malicious_payloads.py

Output: JSON files in data/malicious/ with the training-data dict format:
  {"method": "GET"|"POST", "url": "/path", "params": {...}, "headers": {...}, "json": {...}}
"""
import json
import re
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_MALICIOUS = PROJECT_ROOT / "data" / "malicious"
PAT_DIR = PROJECT_ROOT / "PayloadsAllTheThings"

sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attack_to_training(attack: Dict) -> Dict:
    """Convert attack test dict to training format."""
    entry = {"method": "POST" if "body" in attack else "GET", "url": attack.get("path", "/")}
    if attack.get("query"):
        entry["params"] = attack["query"]
    if attack.get("body"):
        body = attack["body"]
        entry["json"] = body if isinstance(body, (dict, list)) else {"data": body}
    if attack.get("headers"):
        entry["headers"] = attack["headers"]
    return entry


def extract_header_injection() -> List[Dict]:
    """Extract header injection payloads from test 07."""
    results = []
    path = SCRIPT_DIR / "attack_tests" / "07_header_injection.py"
    if path.exists():
        mod = _load_module("header_test", path)
        for attack in getattr(mod, "HEADER_INJECTION_ATTACKS", []):
            results.append(_attack_to_training(attack))
    return results


def extract_ldap_xpath_ssti() -> tuple:
    """Extract LDAP/XPATH/SSTI payloads from test 08, split by category."""
    ldap, xpath, ssti = [], [], []
    path = SCRIPT_DIR / "attack_tests" / "08_ldap_xpath_injection.py"
    if path.exists():
        mod = _load_module("ldap_test", path)
        for attack in getattr(mod, "INJECTION_ATTACKS", []):
            name = attack.get("name", "").lower()
            entry = _attack_to_training(attack)
            if "ldap" in name or "jndi" in name or "log4shell" in name:
                ldap.append(entry)
            elif "xpath" in name or "xquery" in name:
                xpath.append(entry)
            elif any(k in name for k in ("ssti", "jinja", "twig", "freemarker", "velocity",
                                          "pebble", "smarty", "thymeleaf", "erb", "mako",
                                          "handlebars", "razor", "pug", "nunjucks", "template",
                                          "el ", "ognl", "spel", "mvel")):
                ssti.append(entry)
            else:
                # Default to SSTI bucket for injection attacks
                ssti.append(entry)
    return ldap, xpath, ssti


def extract_dos_patterns() -> List[Dict]:
    """Extract DoS patterns from test 09."""
    results = []
    path = SCRIPT_DIR / "attack_tests" / "09_dos_patterns.py"
    if path.exists():
        mod = _load_module("dos_test", path)
        for attack in getattr(mod, "DOS_ATTACKS", []):
            results.append(_attack_to_training(attack))
    return results


def extract_crlf_from_pat() -> List[Dict]:
    """Extract CRLF payloads from PayloadsAllTheThings."""
    results = []
    crlf_file = PAT_DIR / "CRLF Injection" / "Files" / "crlfinjection.txt"
    if crlf_file.exists():
        for line in crlf_file.read_text().strip().splitlines():
            line = line.strip()
            if line:
                results.append({"method": "GET", "url": line, "params": {}})
                results.append({"method": "GET", "url": "/api/redirect", "params": {"url": f"http://example.com{line}"}})
                results.append({"method": "GET", "url": "/api/page", "params": {"next": line}})
    return results


def extract_ssti_from_pat() -> List[Dict]:
    """Extract SSTI payloads from PayloadsAllTheThings markdown files."""
    results = []
    ssti_dir = PAT_DIR / "Server Side Template Injection"
    if not ssti_dir.exists():
        return results

    code_block_re = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)
    inline_code_re = re.compile(r"`([^`]{5,120})`")
    template_markers = {"{{", "${", "#{", "<%", "{%", "{#", "[[", "$F{"}

    payloads = set()
    for md_file in ssti_dir.glob("*.md"):
        if md_file.name == "README.md":
            continue
        content = md_file.read_text(errors="ignore")
        for match in code_block_re.finditer(content):
            block = match.group(1).strip()
            for line in block.splitlines():
                line = line.strip()
                if line and 5 < len(line) < 300 and not line.startswith("#") and not line.startswith("//"):
                    payloads.add(line)
        for match in inline_code_re.finditer(content):
            payload = match.group(1).strip()
            if any(m in payload for m in template_markers):
                payloads.add(payload)

    paths = ["/api/template", "/api/render", "/api/search", "/api/format", "/api/page"]
    params = ["input", "template", "name", "query", "content"]
    for i, p in enumerate(payloads):
        path = paths[i % len(paths)]
        param = params[i % len(params)]
        results.append({"method": "GET", "url": path, "params": {param: p}})
        results.append({"method": "POST", "url": path, "json": {param: p}})

    return results


def _write_json(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  {path.name}: {len(data)} entries")


def main():
    print("=" * 60)
    print("POPULATE data/malicious/ — Categorized Payload Files")
    print("=" * 60)

    # Header injection (CRLF + smuggling + header attacks)
    header = extract_header_injection()
    crlf_pat = extract_crlf_from_pat()
    all_header = header + crlf_pat
    _write_json(DATA_MALICIOUS / "header_injection.json", all_header)

    # LDAP / XPATH / SSTI split
    ldap, xpath, ssti_from_test = extract_ldap_xpath_ssti()
    ssti_from_pat = extract_ssti_from_pat()
    all_ssti = ssti_from_test + ssti_from_pat

    _write_json(DATA_MALICIOUS / "ldap_injection.json", ldap)
    _write_json(DATA_MALICIOUS / "xpath_injection.json", xpath)
    _write_json(DATA_MALICIOUS / "ssti.json", all_ssti)

    # DoS patterns
    dos = extract_dos_patterns()
    _write_json(DATA_MALICIOUS / "dos_patterns.json", dos)

    # Summary
    total = len(all_header) + len(ldap) + len(xpath) + len(all_ssti) + len(dos)
    print(f"\nTotal: {total} categorized malicious training samples")
    print(f"Written to: {DATA_MALICIOUS}/")
    print("\nCategories:")
    print(f"  header_injection.json: {len(all_header)}")
    print(f"  ldap_injection.json:   {len(ldap)}")
    print(f"  xpath_injection.json:  {len(xpath)}")
    print(f"  ssti.json:             {len(all_ssti)}")
    print(f"  dos_patterns.json:     {len(dos)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
