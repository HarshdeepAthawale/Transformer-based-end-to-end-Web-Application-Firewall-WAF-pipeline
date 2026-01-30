
---
license: mit
tags:
- http-requests
- security
- ai-waf
- synthetic-data
---

# Synthetic HTTP Requests Dataset for AI WAF Training

This dataset is synthetically generated and contains a diverse set of HTTP requests, labeled as either 'benign' or 'malicious'. It is designed for training and evaluating Web Application Firewalls (WAFs), particularly those based on AI/ML models.

The dataset aims to provide a comprehensive collection of both common and sophisticated attack vectors, alongside a wide array of legitimate traffic patterns.

## Dataset Composition:

**Total Benign Requests:** 8658
**Total Malicious Requests:** 3291

### Malicious Request Categories:

The malicious portion of the dataset includes, but is not limited to, the following attack types, often with multiple variations and obfuscation techniques:

*   **SQL Injection (SQLi):** Variations include Union-based, Error-based, Time-based blind, Boolean-based blind, Stacked Queries, Out-of-Band, and advanced obfuscation. Payloads are injected via URL parameters, paths, headers (Cookies), and request bodies (JSON, form-urlencoded).
*   **Cross-Site Scripting (XSS):** Includes Reflected XSS (e.g., `<script>`, `<img> onerror`), Stored/Reflected XSS. Payloads are found in URL parameters, paths, User-Agent headers, JSON bodies (as values or keys), form data, and custom HTTP headers, utilizing techniques like JavaScript URIs, `String.fromCharCode`, and Base64 encoding.
*   **Command Injection:** Exploits via URL parameters, paths, HTTP headers, JSON bodies, and form data, using semicolons, pipes, subshell execution (`$(...)`, ` `` `), logical operators (`&&`, `||`), and newline obfuscation.
*   **Path Traversal / Directory Traversal:** Attempts to access restricted files/directories using `../` sequences (plain and encoded), null bytes, and absolute paths, injected into URL parameters, URL paths, cookie values, and JSON bodies.
*   **Server-Side Template Injection (SSTI):** Payloads targeting various template engines, including basic evaluation, object navigation for RCE, placed in URL parameters, paths, headers, JSON bodies, and form data.
*   **Server-Side Request Forgery (SSRF):** Exploits using `http://`, `https://`, `file:///`, `dict://`, `gopher://` schemes. Techniques include IP address obfuscation (decimal, octal, hex) and blind SSRF. Payloads are delivered via URL parameters, paths, JSON bodies, and custom headers.
*   **CRLF Injection / HTTP Response Splitting:** Injection of `\r\n` characters to split headers or inject content, via URL parameters, paths, HTTP headers, and JSON bodies.
*   **XML External Entity (XXE) Injection:** Includes file disclosure, SSRF through XXE, and out-of-band data exfiltration using parameter entities. Payloads are delivered in direct XML request bodies and as part of XML file uploads (`multipart/form-data`).
*   **Log Injection:** Forging log entries or injecting HTML/scripts into log data intended for web-based viewers. Payloads are inserted via URL parameters, User-Agent, JSON bodies, and form data using CRLF sequences or null bytes.
*   **NoSQL Injection:** Targeting databases like MongoDB using operators (`$ne`, `$gt`), JavaScript evaluation (`$where`), time-based blind techniques, and syntax breaking. Payloads are found in URL parameters, JSON bodies, HTTP headers, and form data.
*   **LDAP Injection:** Exploiting LDAP filters through direct injection, blind techniques, attribute retrieval, and null byte usage. Payloads are placed in URL parameters, JSON bodies, form data, and cookie values.
*   **XPath Injection:** String-based manipulation, blind techniques, accessing all XML nodes, and data exfiltration using XPath functions. Payloads are injected into URL parameters, JSON bodies, XML request bodies, and form data.
*   **Open Redirect:** Redirecting users to malicious sites using direct URLs, obfuscated URLs (e.g., `//evil.com`, `legit.com@evil.com`), and data URIs, via URL parameters and JSON bodies.
*   **Header Injection:** Includes Host header injection (for cache poisoning/routing), `X-Forwarded-Host` manipulation, injection of arbitrary custom headers to influence application logic (e.g., `X-User-Role: admin`), and Referer spoofing.
*   **Server-Side Includes (SSI) Injection:** Basic SSI directives (`<!--#echo var="..." -->`, `<!--#exec cmd="..." -->`), file inclusion, and command execution, delivered via URL parameters, form data, HTTP headers, and JSON bodies.
*   **HTTP Parameter Pollution (HPP):** Supplying multiple instances of the same parameter to confuse parsing or bypass security filters, potentially leading to vulnerabilities like SQLi or SSRF. Applied to URL query strings and POST form data, sometimes with URL-encoded parameter names.
*   **Mass Assignment:** Sending unexpected parameters (e.g., `isAdmin: true`, `account_balance: 99999`) in JSON or form-data bodies to modify sensitive object properties without authorization. Covers nested objects and array syntax for parameter binding.
*   **Regex Denial of Service (ReDoS):** Input strings crafted to exploit inefficient regular expressions, leading to excessive CPU consumption (polynomial or exponential backtracking). Payloads delivered via URL query parameters, JSON bodies, and form data.
*   **Text-based Insecure Deserialization (Mimicry):** Payloads containing text snippets characteristic of known deserialization gadget chains for Java (e.g., CommonsCollections), .NET (e.g., TypeConfuseDelegate), PHP (e.g., Phar, Monolog gadgets), and Python (e.g., Pickle opcodes). These are not full serialized objects but strings that WAFs might flag, transported in JSON values, XML CDATA, Base64 encoded form fields, or custom headers.

### Benign Request Categories:

The benign dataset mirrors a wide range of legitimate user and system activities, including:

*   **Standard Web Browsing (GET):** Accessing HTML pages (various site types like blogs, e-commerce, forums), static assets (CSS, JS, images like JPG/PNG/SVG/GIF, fonts), special files (`robots.txt`, `sitemap.xml`, `favicon.ico`), documents (PDF), and feeds (RSS/Atom). These requests feature diverse, realistic query parameters for search, filtering, sorting, tracking, pagination, and include URL-encoded values and international characters.
*   **API Interactions (GET, POST, PUT, PATCH, DELETE):**
    *   **Data Retrieval (GET):** Fetching collections or specific resources (JSON & XML), API schemas (OpenAPI/Swagger), health check endpoints. Includes various authentication methods (Bearer tokens, API keys in headers).
    *   **Data Modification (POST, PUT, PATCH):** Creating resources, full updates, partial updates using JSON, XML, and JSON Patch formats. Payloads range from simple to complex nested structures and large bodies. Benign Base64 encoded data within JSON values is also included.
    *   **Resource Deletion (DELETE):** Standard deletions by ID, and conditional deletions using ETags (`If-Match`).
*   **User Authentication Flows (POST):** Logins, registrations, and password reset requests using both `application/x-www-form-urlencoded` (form-based) and `application/json` (API-based). OAuth token requests are also represented.
*   **Form Submissions (POST):** Standard HTML form submissions for contact forms, comments, profile updates, newsletter signups, polls, etc., using `application/x-www-form-urlencoded` and `multipart/form-data` (for text fields). These include benign hidden fields, callback URLs, and varied content lengths.
*   **File Uploads (POST):** `multipart/form-data` requests for uploading images (JPG, PNG) and documents (PDF, DOCX), including single and multiple file uploads.
*   **Background AJAX/Fetch Operations:** Requests typical of Single Page Applications (SPAs), including GETs for data and POST/PUTs for actions, using JSON or plain text bodies, with headers like `X-Requested-With`.
*   **Legitimate Bot Traffic:** GET requests from common search engine crawlers (Googlebot, Bingbot), social media bots, and generic crawlers, fetching HTML and `robots.txt`.
*   **CORS Preflight Requests (OPTIONS):** For various HTTP methods (GET, POST, PUT, DELETE) and custom headers, indicating cross-origin resource sharing checks.
*   **Resource Metadata Checks (HEAD):** Requests to fetch headers for HTML pages, large files, and API endpoints without retrieving the body.
*   **Challenging Benign Scenarios:** A special category of benign requests designed to superficially resemble malicious patterns but are legitimate. This includes:
    *   Usage of parameter names often targeted in attacks (e.g., `id`, `file`, `cmd`, `url`) but with safe, contextually appropriate values.
    *   JSON keys that might appear in mass assignment attacks (e.g., `isAdmin`) but with benign, non-privileged values (e.g., `false`).
    *   Text content (in comments, messages) that naturally includes SQL keywords or HTML/JS syntax as part of a discussion (not for execution).
    *   Benign uses of HTTP Parameter Pollution (e.g., multiple filter selections).
    *   Benign Base64 encoded data in cookies or JSON values.
    *   Long but harmless parameter values.
    *   And many other variations designed to test the precision of a WAF.

This dataset is intended for research and development purposes to advance the capabilities of security solutions.
