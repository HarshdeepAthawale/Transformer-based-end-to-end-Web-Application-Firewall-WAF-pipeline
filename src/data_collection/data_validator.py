"""
Data Validator

Validate quality and consistency of collected traffic data
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import re
from loguru import logger


class DataValidator:
    """Validate collected traffic data quality"""

    def __init__(self):
        self.validation_rules = {
            'http_methods': {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'},
            'required_fields': {'method', 'path', 'query_params', 'headers', 'label'},
            'max_path_length': 2048,
            'max_body_length': 1048576,  # 1MB
            'valid_content_types': {
                'application/json', 'application/x-www-form-urlencoded',
                'multipart/form-data', 'text/plain', 'application/xml'
            }
        }

    def validate_dataset(self, data_file: str) -> Dict[str, any]:
        """
        Comprehensive validation of a dataset file

        Args:
            data_file: Path to JSON dataset file

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating dataset: {data_file}")

        if not Path(data_file).exists():
            return {'valid': False, 'error': f'File not found: {data_file}'}

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {'valid': False, 'error': f'Invalid JSON: {e}'}

        if not isinstance(data, list):
            return {'valid': False, 'error': 'Dataset must be a JSON array'}

        results = {
            'valid': True,
            'total_samples': len(data),
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        # Basic structure validation
        structure_issues = self._validate_structure(data)
        results['issues'].extend(structure_issues)

        # Content validation
        content_issues = self._validate_content(data)
        results['issues'].extend(content_issues)

        # Statistical analysis
        results['statistics'] = self._analyze_statistics(data)

        # Quality checks
        quality_warnings = self._check_quality(data)
        results['warnings'].extend(quality_warnings)

        # Overall validity
        results['valid'] = len(results['issues']) == 0

        logger.info(f"Validation complete: {len(results['issues'])} issues, {len(results['warnings'])} warnings")

        return results

    def _validate_structure(self, data: List[Dict]) -> List[str]:
        """Validate basic data structure"""
        issues = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                issues.append(f"Sample {i}: Not a dictionary object")
                continue

            # Check required fields
            missing_fields = self.validation_rules['required_fields'] - set(item.keys())
            if missing_fields:
                issues.append(f"Sample {i}: Missing required fields: {missing_fields}")

            # Validate method
            if 'method' in item and item['method'] not in self.validation_rules['http_methods']:
                issues.append(f"Sample {i}: Invalid HTTP method: {item['method']}")

            # Validate path
            if 'path' in item:
                if not isinstance(item['path'], str):
                    issues.append(f"Sample {i}: Path must be string")
                elif len(item['path']) > self.validation_rules['max_path_length']:
                    issues.append(f"Sample {i}: Path too long ({len(item['path'])} chars)")
                elif not item['path'].startswith('/'):
                    issues.append(f"Sample {i}: Path must start with '/'")

            # Validate query_params
            if 'query_params' in item and not isinstance(item['query_params'], dict):
                issues.append(f"Sample {i}: query_params must be dictionary")

            # Validate headers
            if 'headers' in item and not isinstance(item['headers'], dict):
                issues.append(f"Sample {i}: headers must be dictionary")

            # Validate body
            if 'body' in item and item['body'] is not None:
                if not isinstance(item['body'], str):
                    issues.append(f"Sample {i}: body must be string or null")
                elif len(item['body']) > self.validation_rules['max_body_length']:
                    issues.append(f"Sample {i}: body too large ({len(item['body'])} bytes)")

            # Validate label
            if 'label' in item and item['label'] not in [0, 1]:
                issues.append(f"Sample {i}: label must be 0 or 1")

        return issues

    def _validate_content(self, data: List[Dict]) -> List[str]:
        """Validate content quality and consistency"""
        issues = []

        for i, item in enumerate(data):
            # Check for suspicious content in benign samples
            if item.get('label') == 0:  # Benign
                suspicious_patterns = [
                    r'<script[^>]*>.*?</script>',
                    r'union.*select',
                    r'1=1',
                    r'\.\./',
                    r'\.\.\\',
                    r'drop\s+table',
                    r'exec\s*\('
                ]

                text_content = self._extract_text_content(item)
                for pattern in suspicious_patterns:
                    if re.search(pattern, text_content, re.IGNORECASE):
                        issues.append(f"Sample {i}: Benign sample contains suspicious pattern: {pattern}")

            # Check for malformed URLs in paths
            if 'path' in item:
                path = item['path']
                if 'javascript:' in path or 'vbscript:' in path:
                    issues.append(f"Sample {i}: Dangerous protocol in path")

            # Validate content-type headers
            if 'headers' in item and 'Content-Type' in item['headers']:
                content_type = item['headers']['Content-Type'].lower()
                if not any(valid in content_type for valid in self.validation_rules['valid_content_types']):
                    issues.append(f"Sample {i}: Unusual content-type: {content_type}")

        return issues

    def _analyze_statistics(self, data: List[Dict]) -> Dict[str, any]:
        """Analyze dataset statistics"""
        stats = {
            'label_distribution': Counter(item.get('label', -1) for item in data),
            'method_distribution': Counter(item.get('method', 'UNKNOWN') for item in data),
            'attack_types': Counter(),
            'user_types': Counter(),
            'path_patterns': Counter(),
            'content_length_stats': {'min': float('inf'), 'max': 0, 'avg': 0, 'total': 0}
        }

        total_content_length = 0
        valid_samples = 0

        for item in data:
            # Attack types (malicious only)
            if item.get('label') == 1:
                attack_type = item.get('attack_type', 'unknown')
                stats['attack_types'][attack_type] += 1

            # User types (benign only)
            if item.get('label') == 0:
                user_type = item.get('user_type', 'unknown')
                stats['user_types'][user_type] += 1

            # Path patterns
            if 'path' in item:
                path = item['path']
                # Simplify path pattern
                pattern = re.sub(r'\d+', '<ID>', path)
                pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<UUID>', pattern)
                stats['path_patterns'][pattern] += 1

            # Content length stats
            content_length = self._calculate_content_length(item)
            if content_length > 0:
                stats['content_length_stats']['min'] = min(stats['content_length_stats']['min'], content_length)
                stats['content_length_stats']['max'] = max(stats['content_length_stats']['max'], content_length)
                total_content_length += content_length
                valid_samples += 1

        # Calculate average content length
        if valid_samples > 0:
            stats['content_length_stats']['avg'] = total_content_length / valid_samples

        # Convert counters to dictionaries for JSON serialization
        stats['attack_types'] = dict(stats['attack_types'])
        stats['user_types'] = dict(stats['user_types'])
        stats['path_patterns'] = dict(stats['path_patterns'].most_common(20))  # Top 20 patterns

        return stats

    def _check_quality(self, data: List[Dict]) -> List[str]:
        """Check dataset quality issues"""
        warnings = []

        # Check class balance
        labels = [item.get('label', -1) for item in data]
        label_counts = Counter(labels)

        if 0 in label_counts and 1 in label_counts:
            benign_count = label_counts[0]
            malicious_count = label_counts[1]
            total = benign_count + malicious_count
            malicious_ratio = malicious_count / total

            if malicious_ratio < 0.15 or malicious_ratio > 0.25:
                warnings.append(f"Unbalanced classes: {malicious_ratio:.1%} malicious (target: 20%)")

        # Check diversity
        methods = set(item.get('method', 'UNKNOWN') for item in data)
        if len(methods) < 3:
            warnings.append(f"Limited HTTP method diversity: {len(methods)} methods")

        # Check for duplicate samples
        sample_signatures = []
        for item in data:
            # Create a signature based on key fields
            signature = (
                item.get('method', ''),
                item.get('path', ''),
                str(item.get('query_params', {})),
                str(item.get('body', ''))
            )
            sample_signatures.append(signature)

        unique_signatures = len(set(sample_signatures))
        if unique_signatures < len(data) * 0.95:  # Less than 95% unique
            duplicate_ratio = 1 - (unique_signatures / len(data))
            warnings.append(f"High duplicate ratio: {duplicate_ratio:.1%}")

        # Check for missing metadata
        metadata_missing = sum(1 for item in data if 'metadata' not in item)
        if metadata_missing > 0:
            warnings.append(f"Missing metadata in {metadata_missing} samples")

        return warnings

    def _extract_text_content(self, item: Dict) -> str:
        """Extract text content from request for analysis"""
        content_parts = []

        # Path
        if 'path' in item:
            content_parts.append(item['path'])

        # Query parameters
        if 'query_params' in item:
            for key, value in item['query_params'].items():
                content_parts.extend([key, str(value)])

        # Body
        if 'body' in item and item['body']:
            content_parts.append(item['body'])

        return ' '.join(content_parts)

    def _calculate_content_length(self, item: Dict) -> int:
        """Calculate total content length of a request"""
        length = 0

        # Path
        if 'path' in item:
            length += len(item['path'])

        # Query string
        if 'query_params' in item:
            query_str = '&'.join(f"{k}={v}" for k, v in item['query_params'].items())
            length += len(query_str)

        # Headers
        if 'headers' in item:
            for name, value in item['headers'].items():
                length += len(name) + len(str(value)) + 2  # +2 for ': '

        # Body
        if 'body' in item and item['body']:
            length += len(item['body'])

        return length

    def validate_all_datasets(self, data_dir: str = "data") -> Dict[str, Dict]:
        """Validate all dataset splits"""
        data_path = Path(data_dir)
        results = {}

        # Validate each split
        for split in ['training', 'validation', 'test']:
            data_file = data_path / split / f"{split.replace('ing', '')}_data.json"
            if data_file.exists():
                results[split] = self.validate_dataset(str(data_file))

        # Cross-validation checks
        cross_issues = self._validate_cross_dataset_consistency(results)
        if cross_issues:
            results['cross_validation_issues'] = cross_issues

        return results

    def _validate_cross_dataset_consistency(self, results: Dict[str, Dict]) -> List[str]:
        """Check consistency across dataset splits"""
        issues = []

        if not all(split in results for split in ['training', 'validation', 'test']):
            return issues

        # Check feature consistency
        train_stats = results['training']['statistics']
        val_stats = results['validation']['statistics']
        test_stats = results['test']['statistics']

        # Check method distribution consistency
        train_methods = set(train_stats['method_distribution'].keys())
        val_methods = set(val_stats['method_distribution'].keys())
        test_methods = set(test_stats['method_distribution'].keys())

        if not (val_methods.issubset(train_methods) and test_methods.issubset(train_methods)):
            issues.append("Validation/test sets contain HTTP methods not seen in training")

        # Check attack type consistency (malicious samples)
        train_attacks = set(train_stats.get('attack_types', {}).keys())
        val_attacks = set(val_stats.get('attack_types', {}).keys())
        test_attacks = set(test_stats.get('attack_types', {}).keys())

        unseen_val_attacks = val_attacks - train_attacks
        unseen_test_attacks = test_attacks - train_attacks

        if unseen_val_attacks:
            issues.append(f"Validation set has unseen attack types: {unseen_val_attacks}")
        if unseen_test_attacks:
            issues.append(f"Test set has unseen attack types: {unseen_test_attacks}")

        return issues

    def generate_validation_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive validation report"""
        report_lines = [
            "# Dataset Validation Report",
            "",
            f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary"
        ]

        total_issues = sum(len(r.get('issues', [])) for r in results.values() if isinstance(r, dict))
        total_warnings = sum(len(r.get('warnings', [])) for r in results.values() if isinstance(r, dict))

        report_lines.extend([
            f"- Total Issues: {total_issues}",
            f"- Total Warnings: {total_warnings}",
            f"- Datasets Validated: {len([r for r in results.values() if isinstance(r, dict) and r.get('valid', False)])}",
            "",
            "## Detailed Results"
        ])

        for dataset_name, result in results.items():
            if not isinstance(result, dict):
                continue

            report_lines.extend([
                f"",
                f"### {dataset_name.title()} Dataset",
                f"- Valid: {'✅' if result.get('valid', False) else '❌'}",
                f"- Samples: {result.get('total_samples', 0)}",
                f"- Issues: {len(result.get('issues', []))}",
                f"- Warnings: {len(result.get('warnings', []))}"
            ])

            if result.get('issues'):
                report_lines.append("- Issues:")
                for issue in result['issues'][:10]:  # Show first 10 issues
                    report_lines.append(f"  - {issue}")
                if len(result['issues']) > 10:
                    report_lines.append(f"  - ... and {len(result['issues']) - 10} more")

            if result.get('warnings'):
                report_lines.append("- Warnings:")
                for warning in result['warnings'][:5]:  # Show first 5 warnings
                    report_lines.append(f"  - {warning}")

        return '\n'.join(report_lines)