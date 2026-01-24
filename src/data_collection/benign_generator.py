"""
Benign Traffic Generator

Generate realistic benign HTTP requests for WAF training
"""
import random
import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from urllib.parse import urlencode
import uuid
from loguru import logger


@dataclass
class BenignRequest:
    """Represents a benign HTTP request"""
    method: str
    path: str
    query_params: Dict[str, str]
    headers: Dict[str, str]
    body: Optional[str]
    user_type: str
    metadata: Dict[str, any]


class BenignTrafficGenerator:
    """Generate comprehensive benign HTTP traffic"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.request_id = 0

        # User types and their behavior patterns
        self.user_types = {
            'authenticated_user': {
                'weight': 0.6,
                'session_auth': True,
                'api_usage': 0.8,
                'admin_actions': 0.1
            },
            'guest_user': {
                'weight': 0.25,
                'session_auth': False,
                'api_usage': 0.3,
                'admin_actions': 0.0
            },
            'admin_user': {
                'weight': 0.1,
                'session_auth': True,
                'api_usage': 0.9,
                'admin_actions': 0.8
            },
            'api_client': {
                'weight': 0.05,
                'session_auth': True,  # API keys
                'api_usage': 1.0,
                'admin_actions': 0.2
            },
            'casual_browsing': {
                'weight': 0.4,
                'session_auth': False,
                'api_usage': 0.2,
                'admin_actions': 0.0
            },
            'goal_oriented': {
                'weight': 0.3,
                'session_auth': True,
                'api_usage': 0.5,
                'admin_actions': 0.1
            },
            'mobile_user': {
                'weight': 0.3,
                'session_auth': True,
                'api_usage': 0.4,
                'admin_actions': 0.0
            }
        }

        # Realistic user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",
            "PostmanRuntime/7.36.3",
            "python-requests/2.31.0",
            "curl/8.4.0"
        ]

        # Application endpoints by category
        self.endpoints = {
            'ecommerce': {
                'public': ['/', '/products', '/categories', '/search', '/about', '/contact'],
                'authenticated': ['/cart', '/checkout', '/orders', '/profile', '/wishlist'],
                'admin': ['/admin/products', '/admin/orders', '/admin/users', '/admin/dashboard']
            },
            'social': {
                'public': ['/', '/explore', '/trending'],
                'authenticated': ['/feed', '/profile', '/messages', '/notifications', '/settings'],
                'admin': ['/admin/users', '/admin/content', '/admin/analytics']
            },
            'blog': {
                'public': ['/', '/posts', '/categories', '/tags', '/archive'],
                'authenticated': ['/dashboard', '/write', '/drafts', '/profile'],
                'admin': ['/admin/posts', '/admin/users', '/admin/comments']
            },
            'api': {
                'public': ['/api/health', '/api/docs'],
                'authenticated': ['/api/users/me', '/api/posts', '/api/comments', '/api/likes'],
                'admin': ['/api/admin/users', '/api/admin/stats', '/api/admin/logs']
            }
        }

        # Content types for different request types
        self.content_types = {
            'json': 'application/json',
            'form': 'application/x-www-form-urlencoded',
            'multipart': 'multipart/form-data',
            'xml': 'application/xml',
            'text': 'text/plain'
        }

        logger.info("BenignTrafficGenerator initialized with realistic user patterns")

    def generate_request(self, user_type: str = None) -> BenignRequest:
        """Generate a single benign request"""
        self.request_id += 1

        # Select user type if not specified
        if user_type is None:
            user_types = list(self.user_types.keys())
            weights = [self.user_types[ut]['weight'] for ut in user_types]
            user_type = random.choices(user_types, weights=weights)[0]

        user_config = self.user_types[user_type]

        # Determine request characteristics
        is_api_request = random.random() < user_config['api_usage']
        is_admin_action = random.random() < user_config['admin_actions']

        # Generate request components
        method = self._select_method(user_type, is_api_request)
        app_category = self._select_app_category()
        path = self._generate_path(app_category, user_type, is_admin_action)
        query_params = self._generate_query_params(path, method)
        headers = self._generate_headers(user_type, method, is_api_request)
        body = self._generate_body(method, is_api_request)

        # Generate metadata
        metadata = {
            'request_id': self.request_id,
            'timestamp': '2024-01-24T10:00:00Z',
            'session_id': str(uuid.uuid4()) if user_config['session_auth'] else None,
            'user_id': f"user_{random.randint(1, 10000)}" if user_config['session_auth'] else None,
            'ip_address': f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
            'response_time_ms': random.randint(50, 500),
            'status_code': random.choices([200, 201, 204, 301, 302, 400, 404, 500],
                                        weights=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])[0]
        }

        return BenignRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body,
            user_type=user_type,
            metadata=metadata
        )

    def generate_batch(self, count: int, user_distribution: Dict[str, float] = None) -> List[BenignRequest]:
        """Generate a batch of benign requests"""
        requests = []

        # Default distribution if not specified
        if user_distribution is None:
            user_distribution = {ut: config['weight'] for ut, config in self.user_types.items()}

        user_types = list(user_distribution.keys())
        weights = list(user_distribution.values())

        for _ in range(count):
            user_type = random.choices(user_types, weights=weights)[0]
            request = self.generate_request(user_type)
            requests.append(request)

        logger.info(f"Generated {count} benign requests")
        return requests

    def generate_user_session(self, session_length: int = 20) -> List[BenignRequest]:
        """Generate a realistic user session with temporal patterns"""
        # Start with authentication if needed
        session_requests = []

        user_type = random.choice(['authenticated_user', 'admin_user'])
        user_config = self.user_types[user_type]

        # Authentication request (if applicable)
        if user_config['session_auth'] and random.random() < 0.3:
            auth_request = self._create_auth_request()
            session_requests.append(auth_request)

        # Main session activity
        for i in range(session_length):
            request = self.generate_request(user_type)

            # Add session progression metadata
            request.metadata['session_step'] = i + 1
            request.metadata['total_steps'] = session_length

            # Simulate user behavior patterns
            if i < session_length * 0.3:
                # Early session: browsing/exploration
                request.metadata['behavior'] = 'exploration'
            elif i < session_length * 0.8:
                # Middle session: engagement
                request.metadata['behavior'] = 'engagement'
            else:
                # Late session: completion/exit
                request.metadata['behavior'] = 'completion'

            session_requests.append(request)

        return session_requests

    def _select_method(self, user_type: str, is_api: bool) -> str:
        """Select appropriate HTTP method"""
        if is_api:
            # API requests use more varied methods
            methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
            weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        else:
            # Web requests are mostly GET
            methods = ['GET', 'POST', 'PUT', 'DELETE']
            weights = [0.8, 0.15, 0.03, 0.02]

        return random.choices(methods, weights=weights)[0]

    def _select_app_category(self) -> str:
        """Select application category"""
        categories = ['ecommerce', 'social', 'blog', 'api']
        weights = [0.4, 0.3, 0.2, 0.1]  # Ecommerce most common
        return random.choices(categories, weights=weights)[0]

    def _generate_path(self, app_category: str, user_type: str, is_admin: bool) -> str:
        """Generate realistic path based on user type and application"""
        endpoints = self.endpoints[app_category]

        if is_admin and user_type == 'admin_user':
            path_options = endpoints['admin']
        elif user_type in ['authenticated_user', 'admin_user']:
            path_options = endpoints['authenticated'] + endpoints['public']
        else:
            path_options = endpoints['public']

        base_path = random.choice(path_options)

        # Add dynamic segments for realism
        if 'users' in base_path or 'profile' in base_path:
            base_path = base_path.replace('users', f'users/{random.randint(1, 1000)}')
        elif 'products' in base_path:
            base_path = base_path.replace('products', f'products/{random.randint(1, 500)}')
        elif 'posts' in base_path:
            base_path = base_path.replace('posts', f'posts/{random.randint(1, 1000)}')
        elif 'orders' in base_path:
            base_path = base_path.replace('orders', f'orders/{random.randint(1000, 9999)}')

        # Add query parameter indicators sometimes
        if random.random() < 0.3:
            base_path += f"?id={random.randint(1, 1000)}"

        return base_path

    def _generate_query_params(self, path: str, method: str) -> Dict[str, str]:
        """Generate realistic query parameters"""
        params = {}

        # Path-specific parameters
        if 'search' in path or 'query' in path:
            params['q'] = random.choice(['laptop', 'shoes', 'book', 'phone', 'watch'])
            params['limit'] = str(random.randint(10, 100))
            params['offset'] = str(random.randint(0, 1000))

        elif 'products' in path or 'categories' in path:
            params['page'] = str(random.randint(1, 20))
            params['per_page'] = str(random.choice([12, 24, 48, 96]))
            params['sort'] = random.choice(['price_asc', 'price_desc', 'name', 'rating'])
            if random.random() < 0.5:
                params['category'] = random.choice(['electronics', 'clothing', 'books', 'home'])

        elif 'feed' in path or 'posts' in path:
            params['limit'] = str(random.randint(10, 50))
            params['before'] = f"post_{random.randint(1000, 9999)}"
            if random.random() < 0.3:
                params['filter'] = random.choice(['trending', 'following', 'latest'])

        # Common pagination parameters
        if 'page' not in params and random.random() < 0.4:
            params['page'] = str(random.randint(1, 10))

        # Add some noise parameters occasionally
        if random.random() < 0.2:
            noise_params = {
                'utm_source': 'google',
                'utm_medium': 'organic',
                'utm_campaign': 'brand',
                'ref': 'homepage',
                'lang': 'en'
            }
            param_name = random.choice(list(noise_params.keys()))
            params[param_name] = noise_params[param_name]

        return params

    def _generate_headers(self, user_type: str, method: str, is_api: bool) -> Dict[str, str]:
        """Generate realistic HTTP headers"""
        headers = {}

        # User agent
        headers['User-Agent'] = random.choice(self.user_agents)

        # Accept headers
        if is_api:
            headers['Accept'] = 'application/json'
        else:
            headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'

        headers['Accept-Language'] = random.choice([
            'en-US,en;q=0.9',
            'en-GB,en;q=0.8',
            'en-US,en;q=0.9,es;q=0.8',
            'fr-FR,fr;q=0.9,en;q=0.8'
        ])

        # Content-Type for requests with bodies
        if method in ['POST', 'PUT', 'PATCH']:
            if is_api:
                headers['Content-Type'] = 'application/json'
            else:
                content_types = ['application/x-www-form-urlencoded', 'multipart/form-data']
                headers['Content-Type'] = random.choice(content_types)

        # Authentication headers (for authenticated users)
        if user_type in ['authenticated_user', 'admin_user']:
            if random.random() < 0.8:  # 80% have session cookies
                headers['Cookie'] = f"session_id={uuid.uuid4()}; user_id={random.randint(1, 10000)}"

        elif user_type == 'api_client':
            headers['Authorization'] = f"Bearer {uuid.uuid4()}"
            headers['X-API-Key'] = ''.join(random.choices('abcdef0123456789', k=32))

        # Referrer
        if random.random() < 0.6:
            referrers = [
                'https://www.google.com/',
                'https://www.facebook.com/',
                'https://twitter.com/',
                'https://www.linkedin.com/',
                'https://github.com/',
                'https://stackoverflow.com/',
                'https://news.ycombinator.com/',
                'https://reddit.com/'
            ]
            headers['Referer'] = random.choice(referrers)

        # Additional headers
        headers['Accept-Encoding'] = 'gzip, deflate, br'
        headers['Connection'] = 'keep-alive'
        headers['Upgrade-Insecure-Requests'] = '1'

        return headers

    def _generate_body(self, method: str, is_api: bool) -> Optional[str]:
        """Generate request body for POST/PUT requests"""
        if method not in ['POST', 'PUT', 'PATCH']:
            return None

        if random.random() < 0.3:  # 30% of requests don't have bodies
            return None

        if is_api:
            # JSON body for API requests
            body_templates = [
                {"data": {"name": "John Doe", "email": "john@example.com"}},
                {"query": "SELECT * FROM users", "params": {"limit": 10}},
                {"comment": "This is a test comment", "rating": 5},
                {"product_id": random.randint(1, 1000), "quantity": random.randint(1, 5)},
                {"username": "testuser", "password": "testpass123"}
            ]
            return json.dumps(random.choice(body_templates))

        else:
            # Form data for web requests
            form_templates = [
                "name=John+Doe&email=john%40example.com&message=Hello+World",
                f"username=user{random.randint(1,1000)}&password=pass{random.randint(1000,9999)}",
                f"product_id={random.randint(1,500)}&quantity={random.randint(1,10)}&cart_id={uuid.uuid4()}",
                f"title=Test+Post&content=This+is+a+test+post&tags=test%2Cexample",
                f"search_query=laptop&category=electronics&min_price=500&max_price=2000"
            ]
            return random.choice(form_templates)

    def _create_auth_request(self) -> BenignRequest:
        """Create an authentication request"""
        auth_paths = ['/login', '/auth/login', '/api/auth/login', '/signin']
        path = random.choice(auth_paths)

        body = json.dumps({
            "username": f"user_{random.randint(1, 1000)}",
            "password": "password123",
            "remember_me": random.choice([True, False])
        })

        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        return BenignRequest(
            method='POST',
            path=path,
            query_params={},
            headers=headers,
            body=body,
            user_type='guest_user',  # Before authentication
            metadata={
                'request_id': self.request_id,
                'auth_attempt': True,
                'timestamp': '2024-01-24T10:00:00Z'
            }
        )

    def save_requests(self, requests: List[BenignRequest], filepath: str):
        """Save benign requests to JSON file"""
        data = []
        for req in requests:
            request_dict = {
                'method': req.method,
                'path': req.path,
                'query_params': req.query_params,
                'headers': req.headers,
                'body': req.body,
                'user_type': req.user_type,
                'metadata': req.metadata,
                'label': 0  # 0 = benign
            }
            data.append(request_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(requests)} benign requests to {filepath}")

    def load_requests(self, filepath: str) -> List[BenignRequest]:
        """Load benign requests from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        requests = []
        for item in data:
            request = BenignRequest(
                method=item['method'],
                path=item['path'],
                query_params=item['query_params'],
                headers=item['headers'],
                body=item['body'],
                user_type=item['user_type'],
                metadata=item['metadata']
            )
            requests.append(request)

        logger.info(f"Loaded {len(requests)} benign requests from {filepath}")
        return requests