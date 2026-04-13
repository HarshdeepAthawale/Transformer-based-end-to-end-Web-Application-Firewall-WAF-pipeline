#!/usr/bin/env python3
"""
Build a multi-class labeled dataset for WAF classifier training.

Combines data from:
  1. HuggingFace notesbymuneeb/ai-waf-dataset (re-labeled from binary to 8-class)
  2. Local benign requests (data/training/benign_requests.json)
  3. Attack test payloads (scripts/attack_tests/*.py) - already categorized by file
  4. Categorized malicious data (data/malicious/*.json)
  5. Malicious pool + missed samples (scripts/data/)

Output: data/training/multiclass_dataset.json

Label map (8 classes):
  0: benign
  1: sqli
  2: xss
  3: rce  (command injection)
  4: path_traversal
  5: xxe
  6: ssrf
  7: other_attack  (header injection, LDAP, SSTI, DoS, XPath, misc)

Usage:
    python scripts/build_multiclass_dataset.py
    python scripts/build_multiclass_dataset.py --no-hf  # skip HuggingFace download
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_PATH = TRAINING_DIR / "multiclass_dataset.json"

LABEL_MAP = {
    "benign": 0,
    "sqli": 1,
    "xss": 2,
    "rce": 3,
    "path_traversal": 4,
    "xxe": 5,
    "ssrf": 6,
    "other_attack": 7,
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# Pattern-based auto-labeling for malicious samples
# ---------------------------------------------------------------------------

# Patterns are checked in priority order. First match wins.
# More specific patterns come first to avoid mis-classification.

XXE_PATTERNS = re.compile(
    r"<!ENTITY|<!DOCTYPE\s.*SYSTEM|<!DOCTYPE\s.*PUBLIC|"
    r"SYSTEM\s+[\"'](?:file|http|ftp|php|expect)://|"
    r"<!ELEMENT|<!ATTLIST|"
    r"<!DOCTYPE\s+\w+\s*\[|"
    r"xmlns:xi=|"
    r"<\?xml\b.*<!|"
    r"&xxe;|&ext;|&file;",
    re.IGNORECASE,
)

SSRF_PATTERNS = re.compile(
    r"169\.254\.169\.254|"
    r"metadata\.google\.internal|"
    r"100\.100\.100\.200|"  # Alibaba metadata
    r"http://127\.|http://localhost|http://0\.0\.0\.0|"
    r"http://\[::1\]|http://0x7f|http://2130706433|"
    r"http://10\.\d|http://172\.(1[6-9]|2\d|3[01])\.|http://192\.168\.|"
    r"gopher://|dict://|file:///|"
    r"url\s*=\s*https?://(?:127|10\.|172\.|192\.168|localhost)|"
    r"@127\.0\.0\.1|@localhost",
    re.IGNORECASE,
)

SQLI_PATTERNS = re.compile(
    r"(?:UNION\s+(?:ALL\s+)?SELECT)|"
    r"(?:OR\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+)|"
    r"(?:AND\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+)|"
    r"(?:SELECT\s+.*\s+FROM\s+\w+)|"
    r"(?:INSERT\s+INTO)|(?:UPDATE\s+\w+\s+SET)|(?:DELETE\s+FROM)|"
    r"(?:DROP\s+TABLE)|(?:ALTER\s+TABLE)|"
    r"(?:SLEEP\s*\()|(?:WAITFOR\s+DELAY)|(?:BENCHMARK\s*\()|"
    r"(?:EXTRACTVALUE\s*\()|(?:UPDATEXML\s*\()|"
    r"(?:LOAD_FILE\s*\()|(?:INTO\s+(?:OUT|DUMP)FILE)|"
    r"(?:xp_cmdshell)|(?:OPENROWSET)|"
    r"(?:information_schema)|"
    r"(?:'\s*(?:OR|AND)\s+['\d])|"
    r"(?:--\s*$)|"
    r"(?:'\s*;\s*(?:DROP|INSERT|UPDATE|DELETE|SELECT))|"
    r"(?:\$(?:ne|gt|lt|gte|lte|regex|where|exists)\b)|"
    r"(?:'\s*OR\s+')",
    re.IGNORECASE,
)

XSS_PATTERNS = re.compile(
    r"<script[\s>]|</script>|"
    r"javascript\s*:|"
    r"on(?:error|load|click|mouseover|focus|blur|submit|change|input|"
    r"mouseout|mouseenter|mouseleave|keydown|keyup|keypress|"
    r"abort|animationend|beforeunload|contextmenu|dblclick|drag|"
    r"dragend|dragenter|dragleave|dragover|dragstart|drop|"
    r"ended|hashchange|input|invalid|loadeddata|message|"
    r"pageshow|pagehide|pointerdown|pointerup|popstate|"
    r"resize|scroll|toggle|touchstart|touchend|transitionend|"
    r"unload|wheel)\s*=|"
    r"<img\s[^>]*on\w+\s*=|"
    r"<svg[\s/]|<iframe[\s>]|<embed[\s>]|<object[\s>]|"
    r"<body\s[^>]*on\w+|"
    r"alert\s*\(|confirm\s*\(|prompt\s*\(|"
    r"document\.cookie|document\.location|document\.write|"
    r"eval\s*\(|"
    r"String\.fromCharCode|"
    r"<marquee\b|<details\b[^>]*ontoggle|"
    r"expression\s*\(",
    re.IGNORECASE,
)

RCE_PATTERNS = re.compile(
    r";\s*(?:ls|cat|id|whoami|uname|pwd|dir|net\s|ping\s|nslookup|curl\s|wget\s)|"
    r"\|\s*(?:cat|id|whoami|ls|bash|sh|cmd|powershell)|"
    r"&&\s*(?:cat|id|whoami|ls|bash|sh|curl|wget)|"
    r"\$\((?:cat|id|whoami|ls|curl|wget|bash|sh)|"
    r"`(?:cat|id|whoami|ls|curl|wget|bash|sh)`|"
    r"(?:bash|sh|cmd|powershell)\s+-[a-z]*\s|"
    r"/bin/(?:bash|sh|dash|zsh)|"
    r"(?:nc|ncat|netcat)\s+-[a-z]*\s|"
    r"(?:python|perl|ruby|php|node)\s+-[a-z]*\s|"
    r"exec\s*\(|system\s*\(|passthru\s*\(|popen\s*\(|"
    r"os\.(?:system|popen|exec)|"
    r"subprocess\.|"
    r"(?:rm\s+-rf|chmod\s+[0-9]+|chown\s)|"
    r"reverse\s*shell|"
    r"\bnew\s+ProcessBuilder|Runtime\.getRuntime\(\)\.exec|"
    r"import\s+os\b|__import__\s*\(",
    re.IGNORECASE,
)

PATH_TRAVERSAL_PATTERNS = re.compile(
    r"\.\./|\.\.\\|"
    r"%2e%2e[/%]|%2e%2e%2f|%252e%252e|"
    r"/etc/(?:passwd|shadow|hosts|group|sudoers)|"
    r"/proc/self/|"
    r"/var/log/|"
    r"(?:c:|C:)[\\/](?:windows|boot\.ini|win\.ini)|"
    r"\.ssh/|id_rsa|authorized_keys|"
    r"web\.config|\.htaccess|\.htpasswd|"
    r"WEB-INF/|META-INF/|"
    r"(?:file|php)://(?:filter|input|expect)|"
    r"php://filter/|"
    r"http[s]?://[^/]+/.*\.(php|asp|jsp|txt|log|ini|conf|bak)\b.*(?:=|%3d).*\.\.|"
    r"/\.env\b|\.git/config",
    re.IGNORECASE,
)


def classify_text(text: str) -> str:
    """Classify a malicious payload into one of 7 attack categories.

    Returns the attack label (not 'benign').
    Patterns are checked in specificity order: XXE > SSRF > SQLi > XSS > RCE > Path Traversal.
    """
    # XXE first (very distinctive patterns)
    if XXE_PATTERNS.search(text):
        return "xxe"
    # SSRF (check before SQLi because some SSRF payloads contain SQL-like strings)
    if SSRF_PATTERNS.search(text):
        return "ssrf"
    # SQLi
    if SQLI_PATTERNS.search(text):
        return "sqli"
    # Path traversal (check before XSS/RCE since ../ is very distinctive)
    if PATH_TRAVERSAL_PATTERNS.search(text):
        return "path_traversal"
    # XSS
    if XSS_PATTERNS.search(text):
        return "xss"
    # RCE / Command injection
    if RCE_PATTERNS.search(text):
        return "rce"
    # Fallback
    return "other_attack"


# ---------------------------------------------------------------------------
# Request dict -> HTTP text conversion (matches training/inference format)
# ---------------------------------------------------------------------------

def req_to_text(req: Dict) -> str:
    """Convert a request dict to raw HTTP text for training."""
    method = req.get("method", "GET")
    path = req.get("url", req.get("path", "/"))
    params = req.get("params", req.get("query", req.get("query_params")))
    headers = req.get("headers")
    body = req.get("json", req.get("body"))

    full_path = path
    if params and isinstance(params, dict):
        qs = "&".join(f"{k}={v}" for k, v in params.items() if isinstance(v, str))
        if qs:
            full_path = f"{path}?{qs}"

    lines = [f"{method} {full_path} HTTP/1.1"]

    if headers and isinstance(headers, dict):
        skip = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
        for key, value in headers.items():
            if key.lower() not in skip:
                lines.append(f"{key}: {value}")

    if body is not None:
        lines.append("")
        if isinstance(body, dict):
            lines.append(json.dumps(body))
        elif isinstance(body, str):
            lines.append(body)
        else:
            lines.append(str(body))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_hf_dataset() -> List[Dict]:
    """Load HuggingFace dataset and re-label malicious samples into 8 classes."""
    try:
        from datasets import load_dataset
        print("Loading HuggingFace dataset: notesbymuneeb/ai-waf-dataset ...")
        ds = load_dataset("notesbymuneeb/ai-waf-dataset", split="train")
        print(f"  Loaded {len(ds)} samples")
    except Exception as e:
        print(f"  WARNING: Could not load HuggingFace dataset: {e}")
        print("  Use --no-hf flag to skip. Continuing without HF data.")
        return []

    rows = []
    relabel_counts = Counter()
    for row in ds:
        text = row["text"]
        label = row["label"]
        if label == "benign":
            rows.append({"text": text, "label": "benign"})
            relabel_counts["benign"] += 1
        else:
            # Re-label malicious into specific category
            attack_label = classify_text(text)
            rows.append({"text": text, "label": attack_label})
            relabel_counts[attack_label] += 1

    print("  HF dataset re-labeling results:")
    for lbl in LABEL_MAP:
        if relabel_counts[lbl]:
            print(f"    {lbl}: {relabel_counts[lbl]}")
    return rows


def load_benign_requests(max_samples: int = 5000) -> List[Dict]:
    """Load benign requests from local data/training/benign_requests.json."""
    path = TRAINING_DIR / "benign_requests.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping benign requests")
        return []

    print(f"Loading benign requests from {path} ...")
    with open(path) as f:
        entries = json.load(f)

    # These already have 'text' field in HTTP format
    rows = []
    for entry in entries:
        text = entry.get("text", "")
        if text.strip():
            rows.append({"text": text, "label": "benign"})

    # Subsample if too many (avoid massive class imbalance)
    if len(rows) > max_samples:
        random.seed(42)
        rows = random.sample(rows, max_samples)

    print(f"  Loaded {len(rows)} benign samples (capped at {max_samples})")
    return rows


def load_attack_test_payloads() -> List[Dict]:
    """Extract payloads from scripts/attack_tests/*.py files.

    These are already categorized by filename.
    """
    attack_dir = SCRIPT_DIR / "attack_tests"
    if not attack_dir.exists():
        print(f"  WARNING: {attack_dir} not found")
        return []

    # Map filename prefix to label
    file_label_map = {
        "01_sql_injection": "sqli",
        "02_xss_attacks": "xss",
        "03_command_injection": "rce",
        "04_path_traversal": "path_traversal",
        "05_xxe_attacks": "xxe",
        "06_ssrf_attacks": "ssrf",
        "07_header_injection": "other_attack",
        "08_ldap_xpath_injection": "other_attack",
        "09_dos_patterns": "other_attack",
        "10_mixed_blended": None,  # auto-classify mixed
    }

    print("Loading attack test payloads ...")
    rows = []

    for py_file in sorted(attack_dir.glob("*.py")):
        if py_file.name in ("run_all_tests.py", "__init__.py"):
            continue
        if py_file.name == "11_fp_regression.py":
            # These are benign false-positive regression samples
            continue

        stem = py_file.stem
        label = file_label_map.get(stem)

        # Parse the Python file to extract payload dicts
        payloads = _extract_payloads_from_py(py_file)
        if not payloads:
            continue

        count = 0
        for payload in payloads:
            text = req_to_text(payload)
            if label is not None:
                rows.append({"text": text, "label": label})
            else:
                # Auto-classify mixed payloads
                auto_label = classify_text(text)
                rows.append({"text": text, "label": auto_label})
            count += 1

        assigned_label = label or "auto-classified"
        print(f"  {py_file.name}: {count} payloads -> {assigned_label}")

    print(f"  Total attack test payloads: {len(rows)}")
    return rows


def _extract_payloads_from_py(py_path: Path) -> List[Dict]:
    """Extract attack payload dicts from a test file.

    Executes the file in a restricted namespace to capture the payload list,
    since many files use expressions that ast.literal_eval cannot handle.
    """
    content = py_path.read_text()
    payloads = []

    # Build a minimal namespace with only builtins (no requests, no I/O)
    namespace: Dict = {"__builtins__": {"len": len, "range": range, "str": str,
                                         "int": int, "float": float, "list": list,
                                         "dict": dict, "print": lambda *a, **k: None,
                                         "True": True, "False": False, "None": None,
                                         "isinstance": isinstance, "enumerate": enumerate,
                                         "format": format, "type": type, "set": set,
                                         "tuple": tuple, "sorted": sorted, "min": min,
                                         "max": max, "abs": abs, "round": round,
                                         "zip": zip, "map": map, "filter": filter,
                                         "reversed": reversed, "hasattr": hasattr,
                                         "getattr": getattr, "setattr": setattr,
                                         "KeyError": KeyError, "ValueError": ValueError,
                                         "TypeError": TypeError, "Exception": Exception,
                                         "RuntimeError": RuntimeError}}

    # Only exec the lines up to the first function definition to grab constants
    # This avoids importing requests, json, time etc.
    lines = content.split("\n")
    safe_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
        # Skip import lines
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        safe_lines.append(line)

    safe_code = "\n".join(safe_lines)

    try:
        exec(safe_code, namespace)  # noqa: S102
    except Exception:
        return []

    # Find the payload list (uppercase variable that is a list of dicts)
    for var_name, val in namespace.items():
        if var_name.startswith("_") or not var_name[0].isupper():
            continue
        if isinstance(val, list) and len(val) > 3:
            for item in val:
                if isinstance(item, dict) and ("path" in item or "url" in item):
                    payloads.append(item)

    return payloads


def load_malicious_json_files() -> List[Dict]:
    """Load categorized malicious payloads from data/malicious/*.json."""
    mal_dir = DATA_DIR / "malicious"
    if not mal_dir.exists():
        print(f"  WARNING: {mal_dir} not found")
        return []

    # Map filename to label
    file_label_map = {
        "dos_patterns": "other_attack",
        "header_injection": "other_attack",
        "ldap_injection": "other_attack",
        "ssti": "other_attack",
        "xpath_injection": "other_attack",
    }

    print("Loading data/malicious/*.json ...")
    rows = []

    for json_file in sorted(mal_dir.glob("*.json")):
        stem = json_file.stem
        label = file_label_map.get(stem, "other_attack")

        try:
            with open(json_file) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  WARNING: Could not parse {json_file}")
            continue

        if not isinstance(entries, list):
            continue

        count = 0
        for entry in entries:
            if isinstance(entry, dict):
                text = req_to_text(entry)
                rows.append({"text": text, "label": label})
                count += 1

        print(f"  {json_file.name}: {count} samples -> {label}")

    print(f"  Total malicious JSON samples: {len(rows)}")
    return rows


def load_malicious_pool() -> List[Dict]:
    """Load scripts/data/malicious_pool.json and auto-classify."""
    pool_path = SCRIPT_DIR / "data" / "malicious_pool.json"
    if not pool_path.exists():
        print(f"  WARNING: {pool_path} not found")
        return []

    print(f"Loading malicious pool from {pool_path} ...")
    with open(pool_path) as f:
        entries = json.load(f)

    rows = []
    label_counts = Counter()
    for entry in entries:
        if isinstance(entry, dict):
            text = req_to_text(entry)
            label = classify_text(text)
            rows.append({"text": text, "label": label})
            label_counts[label] += 1

    print(f"  Malicious pool: {len(rows)} samples auto-classified:")
    for lbl, cnt in label_counts.most_common():
        print(f"    {lbl}: {cnt}")
    return rows


def load_missed_malicious() -> List[Dict]:
    """Load scripts/data/missed_malicious_samples.json and auto-classify."""
    path = SCRIPT_DIR / "data" / "missed_malicious_samples.json"
    if not path.exists():
        return []

    print(f"Loading missed malicious samples from {path} ...")
    with open(path) as f:
        entries = json.load(f)

    rows = []
    for entry in entries:
        if isinstance(entry, dict):
            text = req_to_text(entry)
            label = classify_text(text)
            rows.append({"text": text, "label": label})

    print(f"  Missed malicious: {len(rows)} samples")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build multi-class WAF training dataset")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace dataset download")
    parser.add_argument("--max-benign", type=int, default=5000, help="Max benign samples from local corpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Building Multi-Class WAF Dataset")
    print("=" * 60)

    all_rows: List[Dict] = []

    # 1. HuggingFace dataset (re-labeled)
    if not args.no_hf:
        all_rows.extend(load_hf_dataset())
    else:
        print("Skipping HuggingFace dataset (--no-hf)")

    print()

    # 2. Local benign requests
    all_rows.extend(load_benign_requests(max_samples=args.max_benign))
    print()

    # 3. Attack test payloads (already categorized)
    all_rows.extend(load_attack_test_payloads())
    print()

    # 4. data/malicious/*.json
    all_rows.extend(load_malicious_json_files())
    print()

    # 5. Malicious pool (auto-classified)
    all_rows.extend(load_malicious_pool())
    print()

    # 6. Missed malicious samples
    all_rows.extend(load_missed_malicious())
    print()

    # Deduplicate by text
    seen = set()
    unique_rows = []
    for row in all_rows:
        text_hash = hash(row["text"][:500])  # Use first 500 chars for dedup
        if text_hash not in seen:
            seen.add(text_hash)
            unique_rows.append(row)

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples (before dedup): {len(all_rows)}")
    print(f"Total samples (after dedup):  {len(unique_rows)}")
    print()

    label_counts = Counter(row["label"] for row in unique_rows)
    print("Class distribution:")
    for label_name in LABEL_MAP:
        count = label_counts.get(label_name, 0)
        label_id = LABEL_MAP[label_name]
        print(f"  {label_id}: {label_name:<20s} {count:>6d}")
    print(f"  {'TOTAL':<23s} {len(unique_rows):>6d}")

    # Shuffle
    random.shuffle(unique_rows)

    # Save as JSON with integer labels
    output_data = []
    for row in unique_rows:
        output_data.append({
            "text": row["text"],
            "label": row["label"],
            "label_id": LABEL_MAP[row["label"]],
        })

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=None)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {file_size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
