#!/usr/bin/env python3
"""
Generate Training Data from Web Applications

Generate benign traffic from DVWA, Juice Shop, and WebGoat applications.
"""
import argparse
import json
import time
import requests
from pathlib import Path
from loguru import logger
from typing import List
import random

from backend.ml.ingestion.ingestion import LogIngestionSystem
from backend.ml.parsing.pipeline import ParsingPipeline


def generate_benign_requests_dvwa(base_url: str, num_requests: int = 1000) -> List[str]:
    """Generate benign requests for DVWA"""
    logger.info(f"Generating {num_requests} benign requests for DVWA...")
    
    endpoints = [
        "/",
        "/login.php",
        "/index.php",
        "/instructions.php",
        "/setup.php",
        "/security.php"
    ]
    
    requests_list = []
    session = requests.Session()
    
    # Login first
    try:
        login_data = {
            "username": "admin",
            "password": "password",
            "Login": "Login"
        }
        session.post(f"{base_url}/login.php", data=login_data, timeout=5)
    except:
        pass
    
    for i in range(num_requests):
        endpoint = random.choice(endpoints)
        try:
            if endpoint == "/login.php":
                response = session.get(f"{base_url}{endpoint}", timeout=5)
            else:
                response = session.get(f"{base_url}{endpoint}", timeout=5)
            
            # Simulate log line format
            log_line = f'127.0.0.1 - - [{time.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "GET {endpoint} HTTP/1.1" {response.status_code} {len(response.content)}'
            requests_list.append(log_line)
        except:
            pass
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_requests} requests")
    
    return requests_list


def generate_benign_requests_juice_shop(base_url: str, num_requests: int = 1000) -> List[str]:
    """Generate benign requests for Juice Shop"""
    logger.info(f"Generating {num_requests} benign requests for Juice Shop...")
    
    endpoints = [
        "/",
        "/#/",
        "/#/search",
        "/#/contact",
        "/#/about",
        "/api/Products",
        "/api/Categories",
        "/rest/products",
        "/rest/products/search"
    ]
    
    requests_list = []
    
    for i in range(num_requests):
        endpoint = random.choice(endpoints)
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            log_line = f'127.0.0.1 - - [{time.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "GET {endpoint} HTTP/1.1" {response.status_code} {len(response.content)}'
            requests_list.append(log_line)
        except:
            pass
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_requests} requests")
    
    return requests_list


def generate_benign_requests_webgoat(base_url: str, num_requests: int = 1000) -> List[str]:
    """Generate benign requests for WebGoat"""
    logger.info(f"Generating {num_requests} benign requests for WebGoat...")
    
    endpoints = [
        "/WebGoat/",
        "/WebGoat/start.mvc",
        "/WebGoat/login.mvc",
        "/WebGoat/welcome.mvc"
    ]
    
    requests_list = []
    
    for i in range(num_requests):
        endpoint = random.choice(endpoints)
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            log_line = f'127.0.0.1 - - [{time.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "GET {endpoint} HTTP/1.1" {response.status_code} {len(response.content)}'
            requests_list.append(log_line)
        except:
            pass
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_requests} requests")
    
    return requests_list


def process_logs_to_normalized(log_lines: List[str]) -> List[str]:
    """Process log lines to normalized request strings"""
    parser = ParsingPipeline()
    normalized = []
    
    for log_line in log_lines:
        try:
            norm = parser.process_log_line(log_line)
            if norm:
                normalized.append(norm)
        except:
            continue
    
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Generate training data from web applications")
    parser.add_argument("--dvwa-url", type=str, help="DVWA base URL")
    parser.add_argument("--juice-shop-url", type=str, help="Juice Shop base URL")
    parser.add_argument("--webgoat-url", type=str, help="WebGoat base URL")
    parser.add_argument("--log-file", type=str, help="Path to existing log file to process")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--samples-per-app", type=int, default=10000, help="Samples per application")
    parser.add_argument("--max-samples", type=int, help="Maximum total samples")
    
    args = parser.parse_args()
    
    all_normalized = []
    
    # Option 1: Process existing log file
    if args.log_file:
        logger.info(f"Processing log file: {args.log_file}")
        ingestion = LogIngestionSystem()
        log_lines = []
        
        for line in ingestion.ingest_batch(args.log_file):
            log_lines.append(line)
        
        normalized = process_logs_to_normalized(log_lines)
        all_normalized.extend(normalized)
        logger.info(f"Processed {len(normalized)} requests from log file")
    
    # Option 2: Generate from web applications
    if args.dvwa_url:
        log_lines = generate_benign_requests_dvwa(args.dvwa_url, args.samples_per_app)
        normalized = process_logs_to_normalized(log_lines)
        all_normalized.extend(normalized)
        logger.info(f"Generated {len(normalized)} normalized requests from DVWA")
    
    if args.juice_shop_url:
        log_lines = generate_benign_requests_juice_shop(args.juice_shop_url, args.samples_per_app)
        normalized = process_logs_to_normalized(log_lines)
        all_normalized.extend(normalized)
        logger.info(f"Generated {len(normalized)} normalized requests from Juice Shop")
    
    if args.webgoat_url:
        log_lines = generate_benign_requests_webgoat(args.webgoat_url, args.samples_per_app)
        normalized = process_logs_to_normalized(log_lines)
        all_normalized.extend(normalized)
        logger.info(f"Generated {len(normalized)} normalized requests from WebGoat")
    
    # Limit total samples if specified
    if args.max_samples and len(all_normalized) > args.max_samples:
        all_normalized = all_normalized[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Remove duplicates
    all_normalized = list(set(all_normalized))
    logger.info(f"Total unique normalized requests: {len(all_normalized)}")
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_normalized, f, indent=2)
    
    logger.info(f"Training data saved to {args.output}")


if __name__ == "__main__":
    main()
