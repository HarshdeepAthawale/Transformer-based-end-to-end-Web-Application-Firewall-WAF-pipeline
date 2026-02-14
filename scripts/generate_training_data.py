#!/usr/bin/env python3
"""
Generate Benign Training Data for WAF Anomaly Model

Crawls Juice Shop, WebGoat, and DVWA web applications to generate
benign HTTP request data. Uses the ingestion + parsing pipeline to
normalize requests into the model input format.

Two modes:
  1. Live crawl: Sends HTTP requests to running apps and captures them
  2. Log parse: Reads existing access logs from apps

Output: data/training/benign_requests.json

Usage:
    # Crawl all apps (must be running on their ports)
    python scripts/generate_training_data.py --mode crawl --apps juice-shop,webgoat,dvwa

    # Parse existing log files
    python scripts/generate_training_data.py --mode logs --log-dir data/raw

    # Generate synthetic benign requests (no apps needed)
    python scripts/generate_training_data.py --mode synthetic --min-per-app 10000

    # Combined: synthetic + crawl (recommended)
    python scripts/generate_training_data.py --mode combined --min-per-app 10000
"""

import argparse
import itertools
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.parsing.pipeline import ParsingPipeline
from backend.parsing.log_parser import HTTPRequest
from backend.ingestion.batch_reader import read_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App configurations
# ---------------------------------------------------------------------------

APP_CONFIGS = {
    "juice-shop": {
        "base_url": "http://localhost:8080",
        "endpoints": [
            # Public pages
            ("GET", "/", {}),
            ("GET", "/api/Products", {}),
            ("GET", "/rest/products/search", {"q": ""}),
            ("GET", "/rest/products/search", {"q": "apple"}),
            ("GET", "/rest/products/search", {"q": "juice"}),
            ("GET", "/rest/products/search", {"q": "banana"}),
            ("GET", "/api/Products/1", {}),
            ("GET", "/api/Products/2", {}),
            ("GET", "/api/Products/3", {}),
            ("GET", "/api/Feedbacks", {}),
            ("GET", "/api/Challenges", {}),
            ("GET", "/api/Quantitys", {}),
            ("GET", "/api/Deliverys", {}),
            ("GET", "/api/Recyclables", {}),
            ("GET", "/rest/admin/application-version", {}),
            ("GET", "/rest/languages", {}),
            ("GET", "/assets/public/images/products/apple_juice.jpg", {}),
            ("GET", "/assets/public/images/products/apple_pressings.jpg", {}),
            ("GET", "/assets/public/favicon_js.ico", {}),
            ("GET", "/ftp", {}),
            ("GET", "/robots.txt", {}),
            ("GET", "/sitemap.xml", {}),
            ("GET", "/.well-known/security.txt", {}),
            # Auth endpoints
            ("POST", "/rest/user/login", {"email": "user@example.com", "password": "password123"}),
            ("POST", "/api/Users", {"email": "newuser@test.com", "password": "test1234", "passwordRepeat": "test1234"}),
            ("GET", "/rest/user/whoami", {}),
            ("GET", "/api/SecurityQuestions", {}),
            # Shopping
            ("GET", "/api/BasketItems", {}),
            ("GET", "/rest/basket/1", {}),
            ("POST", "/api/BasketItems", {"ProductId": 1, "BasketId": "1", "quantity": 1}),
        ],
    },
    "webgoat": {
        "base_url": "http://localhost:8081",
        "endpoints": [
            ("GET", "/WebGoat/login", {}),
            ("GET", "/WebGoat/registration", {}),
            ("POST", "/WebGoat/login", {"username": "guest", "password": "guest"}),
            ("GET", "/WebGoat/start.mvc", {}),
            ("GET", "/WebGoat/service/lessonoverview.mvc", {}),
            ("GET", "/WebGoat/lesson/SqlInjection.lesson", {}),
            ("GET", "/WebGoat/lesson/CrossSiteScripting.lesson", {}),
            ("GET", "/WebGoat/lesson/PathTraversal.lesson", {}),
            ("GET", "/WebGoat/lesson/IDOR.lesson", {}),
            ("GET", "/WebGoat/lesson/VulnerableComponents.lesson", {}),
            ("GET", "/WebGoat/lesson/AuthBypass.lesson", {}),
            ("GET", "/WebGoat/lesson/JWT.lesson", {}),
            ("GET", "/WebGoat/lesson/PasswordReset.lesson", {}),
            ("GET", "/WebGoat/lesson/Challenges.lesson", {}),
            ("GET", "/WebGoat/service/hint.mvc", {}),
            ("GET", "/WebGoat/images/webgoat2.png", {}),
            ("GET", "/WebGoat/css/main.css", {}),
            ("GET", "/WebGoat/js/main.js", {}),
            ("GET", "/WebGoat/favicon.ico", {}),
            ("POST", "/WebGoat/register.mvc", {"username": "testuser", "password": "testpass", "agree": "agree"}),
        ],
    },
    "dvwa": {
        "base_url": "http://localhost:8082",
        "endpoints": [
            ("GET", "/", {}),
            ("GET", "/login.php", {}),
            ("POST", "/login.php", {"username": "admin", "password": "password", "Login": "Login"}),
            ("GET", "/index.php", {}),
            ("GET", "/about.php", {}),
            ("GET", "/instructions.php", {}),
            ("GET", "/setup.php", {}),
            ("GET", "/security.php", {}),
            ("GET", "/phpinfo.php", {}),
            ("GET", "/vulnerabilities/sqli/", {}),
            ("GET", "/vulnerabilities/sqli/", {"id": "1", "Submit": "Submit"}),
            ("GET", "/vulnerabilities/sqli/", {"id": "2", "Submit": "Submit"}),
            ("GET", "/vulnerabilities/xss_r/", {}),
            ("GET", "/vulnerabilities/xss_r/", {"name": "John"}),
            ("GET", "/vulnerabilities/xss_s/", {}),
            ("GET", "/vulnerabilities/exec/", {}),
            ("GET", "/vulnerabilities/fi/", {"page": "include.php"}),
            ("GET", "/vulnerabilities/csrf/", {}),
            ("GET", "/vulnerabilities/upload/", {}),
            ("GET", "/vulnerabilities/captcha/", {}),
            ("GET", "/vulnerabilities/brute/", {}),
            ("GET", "/dvwa/css/main.css", {}),
            ("GET", "/dvwa/js/dvwaPage.js", {}),
            ("GET", "/favicon.ico", {}),
            ("GET", "/dvwa/images/logo.png", {}),
        ],
    },
}

# Common benign User-Agent strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
]

# Common benign referers
REFERERS = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "",  # direct access
    "-",
]

# Benign search terms for parameter variation
SEARCH_TERMS = [
    "apple", "juice", "banana", "orange", "water", "milk", "bread",
    "cheese", "pizza", "salad", "coffee", "tea", "sugar", "salt",
    "rice", "pasta", "chicken", "fish", "beef", "pork", "eggs",
    "butter", "oil", "vinegar", "sauce", "spice", "herb", "fruit",
    "vegetable", "nut", "seed", "grain", "flour", "honey", "jam",
    "product", "item", "order", "cart", "checkout", "payment",
    "delivery", "shipping", "return", "refund", "review", "rating",
    "user", "profile", "settings", "account", "dashboard", "report",
]

USERNAMES = [
    "john", "jane", "admin", "user", "test", "demo", "guest",
    "alice", "bob", "charlie", "dave", "eve", "frank", "grace",
]

# Common benign API paths for synthetic generation
GENERIC_API_PATHS = [
    "/api/products", "/api/products/{id}", "/api/users", "/api/users/{id}",
    "/api/orders", "/api/orders/{id}", "/api/cart", "/api/cart/items",
    "/api/search", "/api/categories", "/api/reviews", "/api/ratings",
    "/api/auth/login", "/api/auth/register", "/api/auth/logout",
    "/api/profile", "/api/settings", "/api/notifications",
    "/api/dashboard", "/api/reports", "/api/health", "/api/status",
    "/static/js/app.js", "/static/css/style.css", "/static/images/logo.png",
    "/favicon.ico", "/robots.txt", "/sitemap.xml",
    "/", "/index.html", "/about", "/contact", "/help", "/faq",
    "/login", "/register", "/forgot-password", "/reset-password",
]


# ---------------------------------------------------------------------------
# Crawl mode: send requests to running apps
# ---------------------------------------------------------------------------

def crawl_app(
    app_name: str,
    pipeline: ParsingPipeline,
    min_requests: int = 1000,
    timeout: float = 5.0,
) -> list[dict]:
    """Crawl a running app and capture normalized requests."""
    config = APP_CONFIGS.get(app_name)
    if not config:
        logger.warning("Unknown app: %s", app_name)
        return []

    base_url = config["base_url"]
    endpoints = config["endpoints"]
    results = []
    session = requests.Session()

    logger.info("Crawling %s at %s (target: %d requests)", app_name, base_url, min_requests)

    # Check if app is reachable
    try:
        resp = session.get(base_url, timeout=timeout)
        logger.info("%s is reachable (status=%d)", app_name, resp.status_code)
    except requests.RequestException as e:
        logger.error("%s is not reachable: %s", app_name, e)
        return []

    # Crawl endpoints in rounds until we hit min_requests
    round_num = 0
    while len(results) < min_requests:
        round_num += 1
        random.shuffle(endpoints)

        for method, path, params in endpoints:
            if len(results) >= min_requests:
                break

            # Vary parameters
            varied_params = _vary_params(params)
            ua = random.choice(USER_AGENTS)
            referer = random.choice(REFERERS)

            headers = {"User-Agent": ua}
            if referer and referer != "-":
                headers["Referer"] = referer

            try:
                if method == "GET":
                    url = f"{base_url}{path}"
                    resp = session.get(url, params=varied_params, headers=headers, timeout=timeout)
                else:
                    url = f"{base_url}{path}"
                    if any(isinstance(v, (dict, list)) for v in varied_params.values()):
                        resp = session.post(url, json=varied_params, headers=headers, timeout=timeout)
                        headers["Content-Type"] = "application/json"
                    else:
                        resp = session.post(url, data=varied_params, headers=headers, timeout=timeout)
                        headers["Content-Type"] = "application/x-www-form-urlencoded"

                # Build HTTPRequest from what we sent
                request = HTTPRequest(
                    method=method,
                    path=path,
                    query_params=varied_params if method == "GET" else {},
                    headers=headers,
                    body=json.dumps(varied_params) if method == "POST" else None,
                )
                serialized = pipeline.process_request(request)
                results.append({
                    "text": serialized,
                    "app": app_name,
                    "source": "crawl",
                })

            except requests.RequestException:
                continue

            # Small delay to avoid hammering
            time.sleep(random.uniform(0.01, 0.05))

        if round_num > 100:
            logger.warning("Hit max rounds for %s, got %d/%d", app_name, len(results), min_requests)
            break

    logger.info("Crawled %s: %d requests captured", app_name, len(results))
    return results


def _vary_params(params: dict) -> dict:
    """Add slight variation to request parameters for diversity."""
    varied = {}
    for key, value in params.items():
        if key in ("q", "query", "search", "name"):
            varied[key] = random.choice(SEARCH_TERMS)
        elif key in ("id", "ProductId", "BasketId"):
            varied[key] = str(random.randint(1, 100))
        elif key in ("page",):
            varied[key] = str(random.randint(1, 20))
        elif key in ("quantity",):
            varied[key] = random.randint(1, 5)
        elif key in ("username", "email"):
            user = random.choice(USERNAMES)
            varied[key] = f"{user}@example.com" if key == "email" else user
        elif key in ("password", "passwordRepeat"):
            varied[key] = f"pass{random.randint(1000, 9999)}"
        else:
            varied[key] = value
    return varied


# ---------------------------------------------------------------------------
# Log parse mode: read existing log files
# ---------------------------------------------------------------------------

def parse_log_files(
    log_dir: str,
    pipeline: ParsingPipeline,
    app_name: str = "unknown",
) -> list[dict]:
    """Parse existing log files into training data."""
    log_path = Path(log_dir)
    results = []

    if not log_path.exists():
        logger.warning("Log directory not found: %s", log_dir)
        return []

    log_files = list(log_path.glob("*.log")) + list(log_path.glob("*.log.gz"))
    logger.info("Found %d log files in %s", len(log_files), log_dir)

    for log_file in log_files:
        for chunk in read_chunks(str(log_file), chunk_size=1000):
            for line in chunk:
                serialized = pipeline.process(line)
                if serialized:
                    results.append({
                        "text": serialized,
                        "app": app_name,
                        "source": "log",
                    })

    logger.info("Parsed %d requests from logs in %s", len(results), log_dir)
    return results


# ---------------------------------------------------------------------------
# Synthetic mode: generate diverse benign requests without running apps
# ---------------------------------------------------------------------------

def generate_synthetic(
    app_name: str,
    pipeline: ParsingPipeline,
    count: int = 10000,
) -> list[dict]:
    """Generate synthetic benign requests for an app."""
    config = APP_CONFIGS.get(app_name)
    if not config:
        logger.warning("Unknown app: %s, using generic endpoints", app_name)
        endpoints = [(random.choice(["GET", "POST"]), p, {}) for p in GENERIC_API_PATHS]
    else:
        endpoints = config["endpoints"]

    results = []
    logger.info("Generating %d synthetic benign requests for %s", count, app_name)

    # Extra query param keys for diversity
    extra_param_keys = [
        "page", "limit", "offset", "sort", "order", "filter", "lang",
        "format", "callback", "v", "t", "ref", "source", "utm_source",
        "utm_medium", "tab", "section", "view", "mode",
    ]
    extra_param_values = [
        "1", "2", "5", "10", "20", "50", "100",
        "asc", "desc", "name", "date", "price", "rating",
        "en", "fr", "de", "es", "json", "xml", "html",
        "list", "grid", "compact", "detailed", "dark", "light",
    ]
    accept_headers = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "application/json",
        "text/html",
        "*/*",
        "application/json, text/plain, */*",
    ]

    for i in range(count):
        method, path, base_params = random.choice(endpoints)

        # Replace {id} placeholders with random IDs
        path = path.replace("{id}", str(random.randint(1, 1000)))

        # Vary params
        params = _vary_params(base_params)

        # Add random extra params for GET to increase diversity
        if method == "GET" and random.random() < 0.6:
            n_extra = random.randint(1, 3)
            for _ in range(n_extra):
                key = random.choice(extra_param_keys)
                val = random.choice(extra_param_values)
                params[key] = val

        # Random headers
        ua = random.choice(USER_AGENTS)
        referer = random.choice(REFERERS)
        headers = {"User-Agent": ua}
        if referer and referer not in ("", "-"):
            headers["Referer"] = referer

        # Sometimes add Accept header
        if random.random() < 0.4:
            headers["Accept"] = random.choice(accept_headers)

        # Sometimes add Accept-Language
        if random.random() < 0.2:
            headers["Accept-Language"] = random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.8", "fr-FR,fr;q=0.9", "de-DE,de;q=0.9"])

        # Occasionally add content-type for POST
        body = None
        if method == "POST" and params:
            headers["Content-Type"] = random.choice([
                "application/json",
                "application/x-www-form-urlencoded",
            ])
            if headers["Content-Type"] == "application/json":
                body = json.dumps(params)
            else:
                body = urlencode(params)

        request = HTTPRequest(
            method=method,
            path=path,
            query_params=params if method == "GET" else {},
            headers=headers,
            body=body,
        )
        serialized = pipeline.process_request(request)
        results.append({
            "text": serialized,
            "app": app_name,
            "source": "synthetic",
        })

    logger.info("Generated %d synthetic requests for %s", len(results), app_name)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benign training data for WAF anomaly model"
    )
    parser.add_argument(
        "--mode", choices=["crawl", "logs", "synthetic", "combined"],
        default="synthetic",
        help="Generation mode (default: synthetic)",
    )
    parser.add_argument(
        "--apps", default="juice-shop,webgoat,dvwa",
        help="Comma-separated app names to generate data for",
    )
    parser.add_argument(
        "--min-per-app", type=int, default=10000,
        help="Minimum requests per app (default: 10000)",
    )
    parser.add_argument(
        "--log-dir", default="data/raw",
        help="Directory containing log files (for --mode logs)",
    )
    parser.add_argument(
        "--output", default="data/training/benign_requests.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--crawl-timeout", type=float, default=5.0,
        help="HTTP request timeout for crawling (seconds)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apps = [a.strip() for a in args.apps.split(",")]
    output_path = PROJECT_ROOT / args.output

    pipeline = ParsingPipeline()
    all_data: list[dict] = []

    for app in apps:
        app_data: list[dict] = []

        if args.mode in ("crawl", "combined"):
            logger.info("=== Crawling %s ===", app)
            crawl_data = crawl_app(app, pipeline, args.min_per_app, args.crawl_timeout)
            app_data.extend(crawl_data)

        if args.mode == "logs":
            logger.info("=== Parsing logs for %s ===", app)
            log_data = parse_log_files(args.log_dir, pipeline, app)
            app_data.extend(log_data)

        if args.mode in ("synthetic", "combined"):
            remaining = max(0, args.min_per_app - len(app_data))
            if remaining > 0 or args.mode == "synthetic":
                target = remaining if args.mode == "combined" else args.min_per_app
                logger.info("=== Generating %d synthetic for %s ===", target, app)
                synthetic_data = generate_synthetic(app, pipeline, target)
                app_data.extend(synthetic_data)

        logger.info("Total for %s: %d requests", app, len(app_data))
        all_data.extend(app_data)

    # Deduplicate by text
    seen = set()
    unique_data = []
    for item in all_data:
        if item["text"] not in seen:
            seen.add(item["text"])
            unique_data.append(item)

    logger.info(
        "Total: %d requests (%d unique) across %d apps",
        len(all_data), len(unique_data), len(apps),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unique_data, f, indent=2)

    logger.info("Saved to %s", output_path)

    # Print summary
    print("\n=== Training Data Summary ===")
    for app in apps:
        count = sum(1 for d in unique_data if d["app"] == app)
        print(f"  {app}: {count} requests")
    print(f"  Total unique: {len(unique_data)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
