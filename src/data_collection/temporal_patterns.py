"""
Temporal Patterns Generator

Generate time-based attack sequences and user behavior patterns
"""
import random
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from .malicious_generator import MaliciousTrafficGenerator, MaliciousRequest
from .benign_generator import BenignTrafficGenerator, BenignRequest


@dataclass
class TemporalSequence:
    """Represents a temporal sequence of requests"""
    sequence_id: str
    requests: List[MaliciousRequest | BenignRequest]
    sequence_type: str
    start_time: datetime
    duration_seconds: int
    metadata: Dict[str, any]


class TemporalPatternGenerator:
    """Generate temporal patterns for advanced training"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.sequence_id_counter = 0

        # Attack sequence patterns
        self.attack_sequences = {
            'reconnaissance': {
                'description': 'Initial reconnaissance phase',
                'duration_range': (30, 300),  # 30 seconds to 5 minutes
                'request_patterns': [
                    ('path_traversal', 0.4),
                    ('sql_injection', 0.3),
                    ('xss', 0.3)
                ],
                'intensity_curve': 'low_start'  # Slow start, low intensity
            },
            'brute_force': {
                'description': 'Brute force authentication attempts',
                'duration_range': (60, 600),  # 1 to 10 minutes
                'request_patterns': [
                    ('command_injection', 1.0)
                ],
                'intensity_curve': 'constant'  # Steady stream
            },
            'data_exfiltration': {
                'description': 'Data theft after initial compromise',
                'duration_range': (120, 1800),  # 2 to 30 minutes
                'request_patterns': [
                    ('path_traversal', 0.6),
                    ('sql_injection', 0.4)
                ],
                'intensity_curve': 'bursty'  # Bursts of activity
            },
            'zero_day_exploit': {
                'description': 'Advanced persistent threat',
                'duration_range': (300, 3600),  # 5 minutes to 1 hour
                'request_patterns': [
                    ('xxe', 0.5),
                    ('ldap_injection', 0.3),
                    ('command_injection', 0.2)
                ],
                'intensity_curve': 'stealthy'  # Very low and slow
            }
        }

        # User behavior patterns
        self.user_patterns = {
            'casual_browsing': {
                'description': 'Normal web browsing behavior',
                'session_length_range': (180, 1800),  # 3 to 30 minutes
                'request_frequency': (2, 10),  # 2-10 requests per minute
                'behavior_phases': ['discovery', 'engagement', 'completion']
            },
            'goal_oriented': {
                'description': 'Task-focused user behavior',
                'session_length_range': (120, 900),  # 2 to 15 minutes
                'request_frequency': (5, 15),  # 5-15 requests per minute
                'behavior_phases': ['search', 'action', 'completion']
            },
            'api_client': {
                'description': 'API client behavior',
                'session_length_range': (60, 600),  # 1 to 10 minutes
                'request_frequency': (10, 30),  # 10-30 requests per minute
                'behavior_phases': ['auth', 'usage', 'cleanup']
            },
            'mobile_user': {
                'description': 'Mobile app user behavior',
                'session_length_range': (90, 1200),  # 1.5 to 20 minutes
                'request_frequency': (3, 8),  # 3-8 requests per minute
                'behavior_phases': ['app_start', 'usage', 'background']
            }
        }

        # Initialize generators
        self.malicious_gen = MaliciousTrafficGenerator(seed=seed)
        self.benign_gen = BenignTrafficGenerator(seed=seed)

        logger.info("TemporalPatternGenerator initialized")

    def generate_attack_sequence(self, sequence_type: str = None) -> TemporalSequence:
        """Generate a temporal attack sequence"""
        self.sequence_id_counter += 1

        # Select sequence type
        if sequence_type is None:
            sequence_type = random.choice(list(self.attack_sequences.keys()))

        sequence_config = self.attack_sequences[sequence_type]

        # Generate sequence duration
        duration = random.randint(*sequence_config['duration_range'])

        # Generate start time (within last 24 hours)
        start_time = datetime.now() - timedelta(hours=random.randint(0, 24))

        # Generate requests based on intensity curve
        requests = self._generate_sequence_requests(
            sequence_config,
            duration,
            start_time
        )

        metadata = {
            'sequence_type': sequence_type,
            'description': sequence_config['description'],
            'duration_seconds': duration,
            'intensity_curve': sequence_config['intensity_curve'],
            'total_requests': len(requests),
            'avg_requests_per_minute': len(requests) / (duration / 60),
            'attack_patterns': sequence_config['request_patterns']
        }

        return TemporalSequence(
            sequence_id=f"attack_seq_{self.sequence_id_counter}",
            requests=requests,
            sequence_type=sequence_type,
            start_time=start_time,
            duration_seconds=duration,
            metadata=metadata
        )

    def generate_user_session(self, user_type: str = None) -> TemporalSequence:
        """Generate a temporal user session"""
        self.sequence_id_counter += 1

        # Select user type
        if user_type is None:
            user_type = random.choice(list(self.user_patterns.keys()))

        session_config = self.user_patterns[user_type]

        # Generate session duration
        duration = random.randint(*session_config['session_length_range'])

        # Generate start time
        start_time = datetime.now() - timedelta(hours=random.randint(0, 24))

        # Generate requests based on user behavior
        requests = self._generate_session_requests(
            session_config,
            duration,
            start_time,
            user_type
        )

        metadata = {
            'sequence_type': 'user_session',
            'user_type': user_type,
            'description': session_config['description'],
            'duration_seconds': duration,
            'behavior_phases': session_config['behavior_phases'],
            'total_requests': len(requests),
            'avg_requests_per_minute': len(requests) / (duration / 60)
        }

        return TemporalSequence(
            sequence_id=f"session_{self.sequence_id_counter}",
            requests=requests,
            sequence_type='user_session',
            start_time=start_time,
            duration_seconds=duration,
            metadata=metadata
        )

    def generate_mixed_traffic(self, duration_hours: int = 1) -> List[TemporalSequence]:
        """Generate mixed traffic with both attacks and normal users"""
        sequences = []
        total_duration_seconds = duration_hours * 3600

        # Generate attack sequences (1-3 per hour)
        num_attacks = random.randint(1, 3) * duration_hours
        for _ in range(num_attacks):
            attack_seq = self.generate_attack_sequence()
            sequences.append(attack_seq)

        # Generate user sessions (10-20 per hour)
        num_sessions = random.randint(10, 20) * duration_hours
        for _ in range(num_sessions):
            user_session = self.generate_user_session()
            sequences.append(user_session)

        # Sort by start time
        sequences.sort(key=lambda x: x.start_time)

        logger.info(f"Generated mixed traffic: {num_attacks} attacks, {num_sessions} user sessions")
        return sequences

    def _generate_sequence_requests(
        self,
        sequence_config: Dict,
        duration: int,
        start_time: datetime
    ) -> List[MaliciousRequest]:
        """Generate requests for an attack sequence"""
        requests = []

        # Calculate number of requests based on duration and intensity
        base_requests = duration // 10  # 1 request every 10 seconds base
        intensity_multiplier = self._get_intensity_multiplier(sequence_config['intensity_curve'])
        num_requests = int(base_requests * intensity_multiplier)

        # Select attack types for this sequence
        attack_types = []
        for attack_type, weight in sequence_config['request_patterns']:
            attack_types.extend([attack_type] * int(weight * 100))
        attack_types.extend(['path_traversal'] * 20)  # Default fallback

        # Generate requests with temporal spacing
        current_time = start_time
        for i in range(num_requests):
            # Select attack type
            attack_type = random.choice(attack_types)

            # Generate request
            request = self.malicious_gen.generate_request(attack_type)

            # Add temporal metadata
            request.metadata['sequence_position'] = i + 1
            request.metadata['total_in_sequence'] = num_requests
            request.metadata['timestamp'] = current_time.isoformat()
            request.metadata['phase'] = self._get_attack_phase(i, num_requests, sequence_config['intensity_curve'])

            requests.append(request)

            # Add temporal spacing (variable based on intensity)
            spacing_seconds = self._get_request_spacing(sequence_config['intensity_curve'], duration)
            current_time += timedelta(seconds=spacing_seconds)

        return requests

    def _generate_session_requests(
        self,
        session_config: Dict,
        duration: int,
        start_time: datetime,
        user_type: str
    ) -> List[BenignRequest]:
        """Generate requests for a user session"""
        requests = []

        # Calculate number of requests
        min_freq, max_freq = session_config['request_frequency']
        avg_requests_per_minute = random.uniform(min_freq, max_freq)
        num_requests = int((duration / 60) * avg_requests_per_minute)

        # Generate requests with temporal spacing
        current_time = start_time
        phases = session_config['behavior_phases']

        for i in range(num_requests):
            # Generate request
            request = self.benign_gen.generate_request(user_type)

            # Add temporal metadata
            request.metadata['sequence_position'] = i + 1
            request.metadata['total_in_sequence'] = num_requests
            request.metadata['timestamp'] = current_time.isoformat()
            request.metadata['phase'] = self._get_session_phase(i, num_requests, phases)

            requests.append(request)

            # Add realistic spacing (exponential distribution)
            spacing_seconds = random.expovariate(1.0 / 6.0)  # Average 6 seconds between requests
            current_time += timedelta(seconds=spacing_seconds)

        return requests

    def _get_intensity_multiplier(self, curve_type: str) -> float:
        """Get intensity multiplier based on curve type"""
        multipliers = {
            'low_start': 0.7,
            'constant': 1.0,
            'bursty': 1.5,
            'stealthy': 0.3
        }
        return multipliers.get(curve_type, 1.0)

    def _get_attack_phase(self, position: int, total: int, curve_type: str) -> str:
        """Determine attack phase based on position and curve type"""
        progress = position / total

        if curve_type == 'low_start':
            if progress < 0.3:
                return 'recon'
            elif progress < 0.7:
                return 'exploit'
            else:
                return 'exfiltrate'
        elif curve_type == 'bursty':
            if progress % 0.3 < 0.1:  # Bursty pattern
                return 'burst'
            else:
                return 'quiet'
        elif curve_type == 'stealthy':
            return 'stealth'
        else:  # constant
            return 'active'

    def _get_session_phase(self, position: int, total: int, phases: List[str]) -> str:
        """Determine user session phase"""
        progress = position / total
        phase_index = int(progress * len(phases))
        return phases[min(phase_index, len(phases) - 1)]

    def _get_request_spacing(self, curve_type: str, total_duration: int) -> int:
        """Get spacing between requests based on curve type"""
        base_spacing = total_duration // 100  # Default spacing

        if curve_type == 'bursty':
            # Bursty: mix short and long spacing
            if random.random() < 0.3:
                return random.randint(1, 5)  # Burst
            else:
                return random.randint(30, 120)  # Quiet periods
        elif curve_type == 'stealthy':
            return random.randint(60, 300)  # Very slow
        elif curve_type == 'low_start':
            return random.randint(10, 60)  # Moderate
        else:  # constant
            return random.randint(5, 20)  # Steady

    def save_sequences(self, sequences: List[TemporalSequence], filepath: str):
        """Save temporal sequences to JSON file"""
        data = []
        for seq in sequences:
            sequence_data = {
                'sequence_id': seq.sequence_id,
                'sequence_type': seq.sequence_type,
                'start_time': seq.start_time.isoformat(),
                'duration_seconds': seq.duration_seconds,
                'metadata': seq.metadata,
                'requests': []
            }

            for req in seq.requests:
                if hasattr(req, 'attack_type'):  # MaliciousRequest
                    request_data = {
                        'type': 'malicious',
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
                    }
                else:  # BenignRequest
                    request_data = {
                        'type': 'benign',
                        'method': req.method,
                        'path': req.path,
                        'query_params': req.query_params,
                        'headers': req.headers,
                        'body': req.body,
                        'user_type': req.user_type,
                        'metadata': req.metadata,
                        'label': 0
                    }

                sequence_data['requests'].append(request_data)

            data.append(sequence_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved {len(sequences)} temporal sequences to {filepath}")

    def load_sequences(self, filepath: str) -> List[TemporalSequence]:
        """Load temporal sequences from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sequences = []
        for seq_data in data:
            requests = []

            for req_data in seq_data['requests']:
                metadata = req_data['metadata']
                # Convert timestamp back to datetime if needed
                if 'timestamp' in metadata and isinstance(metadata['timestamp'], str):
                    try:
                        metadata['timestamp'] = datetime.fromisoformat(metadata['timestamp'])
                    except:
                        pass  # Keep as string if parsing fails

                if req_data['type'] == 'malicious':
                    from .malicious_generator import MaliciousRequest
                    request = MaliciousRequest(
                        method=req_data['method'],
                        path=req_data['path'],
                        query_params=req_data['query_params'],
                        headers=req_data['headers'],
                        body=req_data['body'],
                        attack_type=req_data['attack_type'],
                        attack_family=req_data['attack_family'],
                        severity=req_data['severity'],
                        metadata=metadata
                    )
                else:
                    from .benign_generator import BenignRequest
                    request = BenignRequest(
                        method=req_data['method'],
                        path=req_data['path'],
                        query_params=req_data['query_params'],
                        headers=req_data['headers'],
                        body=req_data['body'],
                        user_type=req_data['user_type'],
                        metadata=metadata
                    )

                requests.append(request)

            sequence = TemporalSequence(
                sequence_id=seq_data['sequence_id'],
                requests=requests,
                sequence_type=seq_data['sequence_type'],
                start_time=datetime.fromisoformat(seq_data['start_time']),
                duration_seconds=seq_data['duration_seconds'],
                metadata=seq_data['metadata']
            )

            sequences.append(sequence)

        logger.info(f"Loaded {len(sequences)} temporal sequences from {filepath}")
        return sequences