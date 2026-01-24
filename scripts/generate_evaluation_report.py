#!/usr/bin/env python3
"""
Evaluation Report Generator

Generate comprehensive evaluation report from test results
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


def generate_evaluation_report(
    results: Dict,
    output_path: str,
    test_type: str = "comprehensive"
):
    """
    Generate evaluation report
    
    Args:
        results: Dictionary with test results
        output_path: Path to save report
        test_type: Type of test ("accuracy", "performance", "comprehensive")
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_type': test_type,
        'summary': {
            'total_tests': results.get('total_tests', 0),
            'passed': results.get('passed', 0),
            'failed': results.get('failed', 0),
            'accuracy': results.get('accuracy', 0.0)
        },
        'accuracy_metrics': {
            'true_positive_rate': results.get('tpr', 0.0),
            'false_positive_rate': results.get('fpr', 0.0),
            'precision': results.get('precision', 0.0),
            'recall': results.get('recall', 0.0),
            'f1_score': results.get('f1_score', 0.0),
            'true_positives': results.get('tp', 0),
            'false_positives': results.get('fp', 0),
            'true_negatives': results.get('tn', 0),
            'false_negatives': results.get('fn', 0)
        },
        'performance_metrics': {
            'avg_latency_ms': results.get('avg_latency', 0.0),
            'median_latency_ms': results.get('median_latency', 0.0),
            'p95_latency_ms': results.get('p95_latency', 0.0),
            'p99_latency_ms': results.get('p99_latency', 0.0),
            'min_latency_ms': results.get('min_latency', 0.0),
            'max_latency_ms': results.get('max_latency', 0.0),
            'throughput_req_per_sec': results.get('throughput', 0.0)
        },
        'detection_by_category': results.get('detection_by_category', {}),
        'details': results.get('details', [])
    }
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary
    print_report(report)
    
    return report


def print_report(report: Dict):
    """Print evaluation report summary"""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Test Type: {report['test_type']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"\nSummary:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Accuracy: {report['summary']['accuracy']:.2%}")
    
    print(f"\nAccuracy Metrics:")
    print(f"  True Positive Rate (TPR/Recall): {report['accuracy_metrics']['true_positive_rate']:.2%}")
    print(f"  False Positive Rate (FPR): {report['accuracy_metrics']['false_positive_rate']:.2%}")
    print(f"  Precision: {report['accuracy_metrics']['precision']:.2%}")
    print(f"  Recall: {report['accuracy_metrics']['recall']:.2%}")
    print(f"  F1 Score: {report['accuracy_metrics']['f1_score']:.2%}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {report['accuracy_metrics']['true_positives']}, "
          f"FP: {report['accuracy_metrics']['false_positives']}")
    print(f"    TN: {report['accuracy_metrics']['true_negatives']}, "
          f"FN: {report['accuracy_metrics']['false_negatives']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Latency: {report['performance_metrics']['avg_latency_ms']:.2f}ms")
    print(f"  Median Latency: {report['performance_metrics']['median_latency_ms']:.2f}ms")
    print(f"  P95 Latency: {report['performance_metrics']['p95_latency_ms']:.2f}ms")
    print(f"  P99 Latency: {report['performance_metrics']['p99_latency_ms']:.2f}ms")
    print(f"  Min Latency: {report['performance_metrics']['min_latency_ms']:.2f}ms")
    print(f"  Max Latency: {report['performance_metrics']['max_latency_ms']:.2f}ms")
    print(f"  Throughput: {report['performance_metrics']['throughput_req_per_sec']:.2f} req/s")
    
    if report.get('detection_by_category'):
        print(f"\nDetection by Category:")
        for category, metrics in report['detection_by_category'].items():
            rate = metrics.get('rate', 0.0)
            detected = metrics.get('detected', 0)
            total = metrics.get('total', 0)
            print(f"  {category}: {rate:.2%} ({detected}/{total})")
    
    print("=" * 60)


def merge_reports(report_paths: List[str], output_path: str) -> Dict:
    """Merge multiple test reports into one"""
    all_results = {
        'accuracy_metrics': {
            'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': []
        },
        'performance_metrics': {
            'avg_latency': [], 'p95_latency': [], 'throughput': []
        },
        'detection_by_category': {}
    }
    
    for report_path in report_paths:
        with open(report_path, 'r') as f:
            report = json.load(f)
            
            # Aggregate accuracy metrics
            acc = report.get('accuracy_metrics', {})
            if acc.get('true_positive_rate'):
                all_results['accuracy_metrics']['tpr'].append(acc['true_positive_rate'])
            if acc.get('false_positive_rate'):
                all_results['accuracy_metrics']['fpr'].append(acc['false_positive_rate'])
            if acc.get('precision'):
                all_results['accuracy_metrics']['precision'].append(acc['precision'])
            if acc.get('recall'):
                all_results['accuracy_metrics']['recall'].append(acc['recall'])
            if acc.get('f1_score'):
                all_results['accuracy_metrics']['f1_score'].append(acc['f1_score'])
            
            # Aggregate performance metrics
            perf = report.get('performance_metrics', {})
            if perf.get('avg_latency_ms'):
                all_results['performance_metrics']['avg_latency'].append(perf['avg_latency_ms'])
            if perf.get('p95_latency_ms'):
                all_results['performance_metrics']['p95_latency'].append(perf['p95_latency_ms'])
            if perf.get('throughput_req_per_sec'):
                all_results['performance_metrics']['throughput'].append(perf['throughput_req_per_sec'])
    
    # Calculate averages
    import statistics
    
    merged_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'merged',
        'accuracy_metrics': {
            'true_positive_rate': statistics.mean(all_results['accuracy_metrics']['tpr']) if all_results['accuracy_metrics']['tpr'] else 0.0,
            'false_positive_rate': statistics.mean(all_results['accuracy_metrics']['fpr']) if all_results['accuracy_metrics']['fpr'] else 0.0,
            'precision': statistics.mean(all_results['accuracy_metrics']['precision']) if all_results['accuracy_metrics']['precision'] else 0.0,
            'recall': statistics.mean(all_results['accuracy_metrics']['recall']) if all_results['accuracy_metrics']['recall'] else 0.0,
            'f1_score': statistics.mean(all_results['accuracy_metrics']['f1_score']) if all_results['accuracy_metrics']['f1_score'] else 0.0
        },
        'performance_metrics': {
            'avg_latency_ms': statistics.mean(all_results['performance_metrics']['avg_latency']) if all_results['performance_metrics']['avg_latency'] else 0.0,
            'p95_latency_ms': statistics.mean(all_results['performance_metrics']['p95_latency']) if all_results['performance_metrics']['p95_latency'] else 0.0,
            'throughput_req_per_sec': statistics.mean(all_results['performance_metrics']['throughput']) if all_results['performance_metrics']['throughput'] else 0.0
        }
    }
    
    # Generate merged report
    return generate_evaluation_report(merged_results, output_path, test_type='merged')


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results", help="Path to test results JSON file")
    parser.add_argument("--output", default="reports/evaluation_report.json", help="Output report path")
    parser.add_argument("--merge", nargs='+', help="Merge multiple reports (provide paths)")
    parser.add_argument("--type", default="comprehensive", choices=["accuracy", "performance", "comprehensive"],
                        help="Test type")
    
    args = parser.parse_args()
    
    if args.merge:
        # Merge reports
        logger.info(f"Merging {len(args.merge)} reports...")
        merge_reports(args.merge, args.output)
    elif args.results:
        # Generate from single results file
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        generate_evaluation_report(results, args.output, args.type)
    else:
        logger.error("Either --results or --merge must be provided")
        sys.exit(1)


if __name__ == "__main__":
    main()
