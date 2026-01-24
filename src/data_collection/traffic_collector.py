"""
Traffic Collector

Orchestrate collection of malicious and benign traffic for balanced dataset
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

from .malicious_generator import MaliciousTrafficGenerator, MaliciousRequest
from .benign_generator import BenignTrafficGenerator, BenignRequest


class TrafficCollector:
    """Collect and balance malicious/benign traffic for training"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.malicious_dir = self.output_dir / "malicious"
        self.benign_dir = self.output_dir / "benign"
        self.training_dir = self.output_dir / "training"
        self.validation_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"

        # Create directories
        for dir_path in [self.malicious_dir, self.benign_dir,
                        self.training_dir, self.validation_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize generators
        self.malicious_gen = MaliciousTrafficGenerator(seed=42)
        self.benign_gen = BenignTrafficGenerator(seed=42)

        logger.info(f"TrafficCollector initialized with output dir: {output_dir}")

    def collect_balanced_dataset(
        self,
        total_samples: int = 100000,
        malicious_ratio: float = 0.2,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, int]:
        """
        Collect a balanced dataset with specified ratios

        Args:
            total_samples: Total number of samples to generate
            malicious_ratio: Ratio of malicious samples (0.0-1.0)
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set

        Returns:
            Dictionary with sample counts per split
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Split ratios must sum to 1.0"

        # Calculate sample counts
        malicious_count = int(total_samples * malicious_ratio)
        benign_count = total_samples - malicious_count

        train_malicious = int(malicious_count * train_ratio)
        train_benign = int(benign_count * train_ratio)

        val_malicious = int(malicious_count * val_ratio)
        val_benign = int(benign_count * val_ratio)

        test_malicious = malicious_count - train_malicious - val_malicious
        test_benign = benign_count - train_benign - val_benign

        logger.info(f"Generating dataset: {total_samples} total samples")
        logger.info(f"  Malicious: {malicious_count} ({malicious_ratio:.1%})")
        logger.info(f"  Benign: {benign_count} ({1-malicious_ratio:.1%})")
        logger.info(f"Splits - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")

        # Generate malicious samples
        logger.info("Generating malicious traffic...")
        malicious_samples = self.malicious_gen.generate_batch(malicious_count)
        self.malicious_gen.save_requests(malicious_samples, self.malicious_dir / "malicious_samples.json")

        # Generate benign samples
        logger.info("Generating benign traffic...")
        benign_samples = self.benign_gen.generate_batch(benign_count)
        self.benign_gen.save_requests(benign_samples, self.benign_dir / "benign_samples.json")

        # Split datasets
        logger.info("Splitting datasets...")

        # Training set
        train_malicious_samples = malicious_samples[:train_malicious]
        train_benign_samples = benign_samples[:train_benign]
        self._save_split("train", train_malicious_samples, train_benign_samples)

        # Validation set
        val_start = train_malicious
        val_malicious_samples = malicious_samples[val_start:val_start + val_malicious]
        val_benign_samples = benign_samples[train_benign:train_benign + val_benign]
        self._save_split("val", val_malicious_samples, val_benign_samples)

        # Test set
        test_start_mal = val_start + val_malicious
        test_start_ben = train_benign + val_benign
        test_malicious_samples = malicious_samples[test_start_mal:]
        test_benign_samples = benign_samples[test_start_ben:]
        self._save_split("test", test_malicious_samples, test_benign_samples)

        # Return statistics
        stats = {
            'total_samples': total_samples,
            'malicious_samples': malicious_count,
            'benign_samples': benign_count,
            'train_malicious': len(train_malicious_samples),
            'train_benign': len(train_benign_samples),
            'val_malicious': len(val_malicious_samples),
            'val_benign': len(val_benign_samples),
            'test_malicious': len(test_malicious_samples),
            'test_benign': len(test_benign_samples)
        }

        logger.info("Dataset collection complete!")
        logger.info(f"  Training: {len(train_malicious_samples) + len(train_benign_samples)} samples")
        logger.info(f"  Validation: {len(val_malicious_samples) + len(val_benign_samples)} samples")
        logger.info(f"  Test: {len(test_malicious_samples) + len(test_benign_samples)} samples")

        return stats

    def collect_temporal_sequences(
        self,
        num_sequences: int = 1000,
        avg_sequence_length: int = 20
    ) -> Dict[str, int]:
        """
        Collect temporal attack sequences for advanced training

        Args:
            num_sequences: Number of attack sequences to generate
            avg_sequence_length: Average length of each sequence

        Returns:
            Statistics about generated sequences
        """
        logger.info(f"Generating {num_sequences} temporal attack sequences...")

        all_sequences = []

        for i in range(num_sequences):
            # Vary sequence length around average
            seq_length = max(5, int(random.gauss(avg_sequence_length, avg_sequence_length * 0.2)))
            sequence = self.malicious_gen.generate_temporal_sequence(seq_length)
            all_sequences.extend(sequence)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_sequences} sequences")

        # Save sequences
        output_file = self.malicious_dir / "temporal_sequences.json"
        self.malicious_gen.save_requests(all_sequences, output_file)

        stats = {
            'num_sequences': num_sequences,
            'avg_sequence_length': avg_sequence_length,
            'total_samples': len(all_sequences),
            'sequences_file': str(output_file)
        }

        logger.info(f"Generated {len(all_sequences)} samples in {num_sequences} temporal sequences")
        return stats

    def collect_user_sessions(
        self,
        num_sessions: int = 5000,
        avg_session_length: int = 15
    ) -> Dict[str, int]:
        """
        Collect realistic user session data

        Args:
            num_sessions: Number of user sessions to generate
            avg_session_length: Average requests per session

        Returns:
            Statistics about generated sessions
        """
        logger.info(f"Generating {num_sessions} user sessions...")

        all_sessions = []

        for i in range(num_sessions):
            # Vary session length around average
            session_length = max(3, int(random.gauss(avg_session_length, avg_session_length * 0.3)))
            session = self.benign_gen.generate_user_session(session_length)
            all_sessions.extend(session)

            if (i + 1) % 500 == 0:
                logger.info(f"Generated {i + 1}/{num_sessions} sessions")

        # Save sessions
        output_file = self.benign_dir / "user_sessions.json"
        self.benign_gen.save_requests(all_sessions, output_file)

        stats = {
            'num_sessions': num_sessions,
            'avg_session_length': avg_session_length,
            'total_samples': len(all_sessions),
            'sessions_file': str(output_file)
        }

        logger.info(f"Generated {len(all_sessions)} samples in {num_sessions} user sessions")
        return stats

    def _save_split(self, split_name: str, malicious_samples: List[MaliciousRequest],
                   benign_samples: List[BenignRequest]):
        """Save a dataset split to disk"""
        if split_name == "train":
            output_dir = self.training_dir
        elif split_name == "val":
            output_dir = self.validation_dir
        else:
            output_dir = self.test_dir

        # Combine and shuffle samples
        combined_samples = []
        combined_samples.extend([{
            'method': req.method,
            'path': req.path,
            'query_params': req.query_params,
            'headers': req.headers,
            'body': req.body,
            'attack_type': req.attack_type,
            'attack_family': req.attack_family,
            'severity': req.severity,
            'metadata': req.metadata,
            'label': 1
        } for req in malicious_samples])

        combined_samples.extend([{
            'method': req.method,
            'path': req.path,
            'query_params': req.query_params,
            'headers': req.headers,
            'body': req.body,
            'user_type': req.user_type,
            'metadata': req.metadata,
            'label': 0
        } for req in benign_samples])

        # Shuffle the combined dataset
        random.shuffle(combined_samples)

        # Save to JSON
        output_file = output_dir / f"{split_name}_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(combined_samples)} samples to {output_file}")
        logger.info(f"  Malicious: {len(malicious_samples)}")
        logger.info(f"  Benign: {len(benign_samples)}")

    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about collected datasets"""
        stats = {}

        for split in ['training', 'validation', 'test']:
            split_dir = getattr(self, f"{split}_dir")
            data_file = split_dir / f"{split.replace('ing', '')}_data.json"

            if data_file.exists():
                with open(data_file, 'r') as f:
                    data = json.load(f)

                malicious = sum(1 for item in data if item.get('label') == 1)
                benign = sum(1 for item in data if item.get('label') == 0)

                stats[split] = {
                    'total': len(data),
                    'malicious': malicious,
                    'benign': benign,
                    'malicious_ratio': malicious / len(data) if data else 0
                }

        return stats

    def validate_dataset_quality(self) -> Dict[str, any]:
        """Validate the quality of collected datasets"""
        issues = []
        recommendations = []

        stats = self.get_dataset_stats()

        # Check class balance
        for split, split_stats in stats.items():
            ratio = split_stats['malicious_ratio']
            if abs(ratio - 0.2) > 0.05:  # More than 5% deviation from 20%
                issues.append(f"{split} set has unbalanced classes (malicious: {ratio:.1%})")
                recommendations.append(f"Regenerate {split} data with proper 80/20 benign/malicious ratio")

        # Check minimum sample sizes
        min_samples = 10000
        for split, split_stats in stats.items():
            if split_stats['total'] < min_samples:
                issues.append(f"{split} set has insufficient samples ({split_stats['total']})")
                recommendations.append(f"Generate at least {min_samples} samples for {split} set")

        # Check data diversity
        training_file = self.training_dir / "train_data.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                data = json.load(f)

            # Check method diversity
            methods = set(item['method'] for item in data)
            if len(methods) < 3:
                issues.append("Limited HTTP method diversity in training data")
                recommendations.append("Include more HTTP methods (GET, POST, PUT, DELETE, etc.)")

            # Check attack type diversity
            attack_types = set(item.get('attack_type', 'benign') for item in data if item.get('label') == 1)
            if len(attack_types) < 5:
                issues.append("Limited attack type diversity")
                recommendations.append("Include more OWASP Top 10 attack categories")

        return {
            'issues': issues,
            'recommendations': recommendations,
            'quality_score': max(0, 100 - len(issues) * 10)  # Simple quality score
        }