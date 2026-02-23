#!/usr/bin/env python3
"""
False Positive Regression Test Suite
Tests that legitimate benign traffic is NOT blocked by the WAF.
Target: <1% false positive rate on realistic benign requests.

This suite contains 200+ realistic benign requests that should NEVER be blocked.
Categories:
  - Standard API CRUD operations
  - Search and pagination
  - Authentication flows
  - File and media requests
  - JSON API payloads with complex data
  - GraphQL queries (legitimate)
  - Webhook and callback patterns
  - Requests with legitimate special characters
  - Requests with headers that look suspicious but are legitimate
  - Long URLs and complex query strings
  - Multipart/form-data patterns
  - Internationalized content
"""

import requests
import json
import time
import os
from typing import Dict

WAF_API = os.getenv("API_SERVER_URL", "http://localhost:3001") + "/api/waf/check"

# Benign requests that should NOT be blocked
BENIGN_REQUESTS = [
    # --- Standard API CRUD ---
    {"name": "GET homepage", "path": "/", "method": "GET"},
    {"name": "GET products list", "path": "/api/products", "method": "GET", "query": {"page": "1", "limit": "20"}},
    {"name": "GET product by ID", "path": "/api/products/42", "method": "GET"},
    {"name": "GET user profile", "path": "/api/users/123/profile", "method": "GET"},
    {"name": "POST create product", "path": "/api/products", "method": "POST", "body": {"name": "Wireless Mouse", "price": 29.99, "category": "electronics"}},
    {"name": "PUT update product", "path": "/api/products/42", "method": "PUT", "body": {"name": "Wireless Mouse Pro", "price": 39.99}},
    {"name": "DELETE remove item", "path": "/api/cart/items/7", "method": "DELETE"},
    {"name": "PATCH update status", "path": "/api/orders/100", "method": "PATCH", "body": {"status": "shipped"}},
    {"name": "GET categories", "path": "/api/categories", "method": "GET"},
    {"name": "GET inventory", "path": "/api/inventory", "method": "GET", "query": {"warehouse": "main"}},

    # --- Search and Pagination ---
    {"name": "Search products", "path": "/api/search", "method": "GET", "query": {"q": "wireless headphones", "sort": "price", "order": "asc"}},
    {"name": "Search with filters", "path": "/api/search", "method": "GET", "query": {"q": "laptop", "brand": "Dell", "minPrice": "500", "maxPrice": "2000"}},
    {"name": "Paginated results", "path": "/api/products", "method": "GET", "query": {"page": "5", "limit": "50", "offset": "200"}},
    {"name": "Sort by date", "path": "/api/posts", "method": "GET", "query": {"sort": "created_at", "order": "desc", "limit": "25"}},
    {"name": "Full text search", "path": "/api/search", "method": "POST", "body": {"query": "best wireless headphones 2024", "filters": {"category": "audio", "inStock": True}}},
    {"name": "Autocomplete query", "path": "/api/autocomplete", "method": "GET", "query": {"q": "jav", "limit": "10"}},
    {"name": "Faceted search", "path": "/api/search", "method": "GET", "query": {"q": "shoes", "size": "10", "color": "black", "brand": "Nike"}},
    {"name": "Search empty query", "path": "/api/search", "method": "GET", "query": {"q": ""}},
    {"name": "Search special chars", "path": "/api/search", "method": "GET", "query": {"q": "C++ programming tutorial"}},
    {"name": "Search with ampersand", "path": "/api/search", "method": "GET", "query": {"q": "Tom & Jerry episodes"}},

    # --- Authentication Flows ---
    {"name": "Login request", "path": "/api/auth/login", "method": "POST", "body": {"email": "user@example.com", "password": "MySecurePass123!"}},
    {"name": "Register user", "path": "/api/auth/register", "method": "POST", "body": {"email": "newuser@example.com", "name": "John Doe", "password": "StrongP@ss1"}},
    {"name": "Forgot password", "path": "/api/auth/forgot-password", "method": "POST", "body": {"email": "user@example.com"}},
    {"name": "Reset password", "path": "/api/auth/reset-password", "method": "POST", "body": {"token": "abc123def456", "password": "NewSecureP@ss1"}},
    {"name": "Refresh token", "path": "/api/auth/refresh", "method": "POST", "body": {"refresh_token": "eyJhbGciOiJIUzI1NiJ9.eyJ0b2tlbiI6InJlZnJlc2gifQ.sig"}},
    {"name": "Logout request", "path": "/api/auth/logout", "method": "POST"},
    {"name": "OAuth callback", "path": "/api/auth/callback", "method": "GET", "query": {"code": "abc123", "state": "xyz789"}},
    {"name": "Verify email", "path": "/api/auth/verify-email", "method": "GET", "query": {"token": "verification_token_123"}},
    {"name": "MFA verify", "path": "/api/auth/mfa/verify", "method": "POST", "body": {"code": "123456"}},
    {"name": "SSO redirect", "path": "/api/auth/sso", "method": "GET", "query": {"provider": "google", "redirect": "/dashboard"}},

    # --- File and Media Requests ---
    {"name": "GET static JS", "path": "/static/js/app.bundle.js", "method": "GET"},
    {"name": "GET static CSS", "path": "/static/css/main.min.css", "method": "GET"},
    {"name": "GET image", "path": "/static/images/logo.png", "method": "GET"},
    {"name": "GET favicon", "path": "/favicon.ico", "method": "GET"},
    {"name": "GET manifest", "path": "/manifest.json", "method": "GET"},
    {"name": "GET sitemap", "path": "/sitemap.xml", "method": "GET"},
    {"name": "GET robots.txt", "path": "/robots.txt", "method": "GET"},
    {"name": "Download report", "path": "/api/reports/download", "method": "GET", "query": {"id": "report-2024-Q1", "format": "pdf"}},
    {"name": "Upload avatar", "path": "/api/users/123/avatar", "method": "POST", "headers": {"Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"}},
    {"name": "GET thumbnail", "path": "/api/images/thumb/product-42.webp", "method": "GET", "query": {"w": "200", "h": "200"}},

    # --- Complex JSON API Payloads ---
    {"name": "Create order", "path": "/api/orders", "method": "POST", "body": {"items": [{"id": 1, "qty": 2, "price": 29.99}, {"id": 5, "qty": 1, "price": 49.99}], "shipping": {"address": "123 Main St", "city": "Springfield", "state": "IL", "zip": "62701"}, "payment": {"method": "credit_card", "last4": "4242"}}},
    {"name": "Update preferences", "path": "/api/users/123/preferences", "method": "PUT", "body": {"theme": "dark", "language": "en", "notifications": {"email": True, "push": False, "sms": True}, "timezone": "America/Chicago"}},
    {"name": "Batch update", "path": "/api/products/batch", "method": "POST", "body": {"operations": [{"op": "update", "id": 1, "data": {"price": 19.99}}, {"op": "update", "id": 2, "data": {"price": 24.99}}, {"op": "delete", "id": 3}]}},
    {"name": "Analytics event", "path": "/api/analytics/events", "method": "POST", "body": {"event": "page_view", "properties": {"page": "/products", "referrer": "https://google.com/search?q=best+deals", "timestamp": "2024-01-15T10:30:00Z"}}},
    {"name": "Complex filter", "path": "/api/reports/generate", "method": "POST", "body": {"dateRange": {"start": "2024-01-01", "end": "2024-12-31"}, "metrics": ["revenue", "orders", "customers"], "groupBy": "month", "filters": {"region": ["US", "EU", "APAC"]}}},
    {"name": "Nested JSON body", "path": "/api/config/update", "method": "POST", "body": {"database": {"host": "db.example.com", "port": 5432, "name": "myapp"}, "cache": {"ttl": 3600, "maxSize": "256MB"}, "logging": {"level": "info", "format": "json"}}},
    {"name": "Array of objects", "path": "/api/contacts/import", "method": "POST", "body": [{"name": "Alice", "email": "alice@example.com"}, {"name": "Bob", "email": "bob@example.com"}, {"name": "Charlie", "email": "charlie@example.com"}]},
    {"name": "Form submission", "path": "/api/contact", "method": "POST", "body": {"name": "Jane Smith", "email": "jane@example.com", "subject": "Product inquiry", "message": "Hi, I'd like to know more about your enterprise plan. Can you send me pricing details?"}},
    {"name": "Survey response", "path": "/api/surveys/response", "method": "POST", "body": {"surveyId": "s123", "answers": [{"q": "How satisfied are you?", "a": "Very satisfied"}, {"q": "Would you recommend us?", "a": "Yes, definitely"}]}},
    {"name": "Comment with code", "path": "/api/comments", "method": "POST", "body": {"text": "You can use `console.log('hello')` to debug. For loops, try `for (let i = 0; i < 10; i++) { ... }`", "postId": 42}},

    # --- GraphQL (Legitimate) ---
    {"name": "GraphQL user query", "path": "/api/graphql", "method": "POST", "body": {"query": "{ user(id: 1) { name email } }"}},
    {"name": "GraphQL products", "path": "/api/graphql", "method": "POST", "body": {"query": "{ products(first: 10) { edges { node { id name price } } pageInfo { hasNextPage } } }"}},
    {"name": "GraphQL mutation", "path": "/api/graphql", "method": "POST", "body": {"query": "mutation { updateProfile(name: \"John\") { id name } }", "variables": {}}},
    {"name": "GraphQL with vars", "path": "/api/graphql", "method": "POST", "body": {"query": "query GetUser($id: ID!) { user(id: $id) { name email } }", "variables": {"id": "123"}}},
    {"name": "GraphQL fragments", "path": "/api/graphql", "method": "POST", "body": {"query": "fragment UserFields on User { name email } query { user(id: 1) { ...UserFields } }"}},

    # --- Webhook and Callback Patterns ---
    {"name": "Stripe webhook", "path": "/api/webhooks/stripe", "method": "POST", "headers": {"Stripe-Signature": "t=1234,v1=abc123"}, "body": {"type": "payment_intent.succeeded", "data": {"object": {"id": "pi_123", "amount": 2000}}}},
    {"name": "GitHub webhook", "path": "/api/webhooks/github", "method": "POST", "headers": {"X-GitHub-Event": "push", "X-Hub-Signature-256": "sha256=abc123"}, "body": {"ref": "refs/heads/main", "commits": [{"message": "Fix typo in README"}]}},
    {"name": "Slack event", "path": "/api/webhooks/slack", "method": "POST", "body": {"type": "event_callback", "event": {"type": "message", "text": "Hello team!"}}},
    {"name": "Health check", "path": "/health", "method": "GET"},
    {"name": "Readiness probe", "path": "/ready", "method": "GET"},
    {"name": "Liveness probe", "path": "/healthz", "method": "GET"},

    # --- Requests with Legitimate Special Characters ---
    {"name": "Search math expr", "path": "/api/search", "method": "GET", "query": {"q": "what is 1=1 in mathematics"}},
    {"name": "Search OR keyword", "path": "/api/search", "method": "GET", "query": {"q": "how to use OR operator in Python"}},
    {"name": "Search SQL topic", "path": "/api/search", "method": "GET", "query": {"q": "SQL UNION operator tutorial"}},
    {"name": "Search template topic", "path": "/api/search", "method": "GET", "query": {"q": "Jinja2 template syntax examples"}},
    {"name": "Search LDAP docs", "path": "/api/search", "method": "GET", "query": {"q": "LDAP connection timeout troubleshooting"}},
    {"name": "Search XPath tutorial", "path": "/api/docs", "method": "GET", "query": {"topic": "XPath selectors tutorial"}},
    {"name": "Forum post SQL", "path": "/api/forum", "method": "POST", "body": {"title": "How to use SELECT * FROM table", "body": "I need to query all rows from a table in PostgreSQL"}},
    {"name": "Code review body", "path": "/api/code-review", "method": "POST", "body": {"code": "if (user.role === 'admin') { return true; }", "language": "javascript"}},
    {"name": "Regex in search", "path": "/api/search", "method": "GET", "query": {"q": "regex pattern ^[a-zA-Z0-9]+$"}},
    {"name": "HTML in docs", "path": "/api/docs", "method": "POST", "body": {"content": "Use <strong>bold</strong> and <em>italic</em> for emphasis in HTML"}},
    {"name": "Curly braces search", "path": "/api/search", "method": "GET", "query": {"q": "how to escape curly braces {{ }} in Jinja2"}},
    {"name": "Dollar sign search", "path": "/api/search", "method": "GET", "query": {"q": "JavaScript template literals ${variable}"}},
    {"name": "Angle brackets search", "path": "/api/search", "method": "GET", "query": {"q": "C++ vector<int> template usage"}},
    {"name": "Percent encoding search", "path": "/api/search", "method": "GET", "query": {"q": "URL encoding %20 spaces explained"}},

    # --- Requests with Legitimate Headers ---
    {"name": "Standard browser req", "path": "/api/data", "method": "GET", "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9,fr;q=0.8", "Accept-Encoding": "gzip, deflate, br"}},
    {"name": "API with Bearer token", "path": "/api/profile", "method": "GET", "headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "Accept": "application/json"}},
    {"name": "Request with cookies", "path": "/api/dashboard", "method": "GET", "headers": {"Cookie": "session=abc123def456; preferences=dark_mode; _ga=GA1.2.123456.789"}},
    {"name": "CORS preflight", "path": "/api/data", "method": "OPTIONS", "headers": {"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Content-Type, Authorization"}},
    {"name": "Content negotiation", "path": "/api/data", "method": "GET", "headers": {"Accept": "application/json, text/plain, */*", "Accept-Encoding": "gzip, deflate, br", "Accept-Language": "en-US,en;q=0.9"}},
    {"name": "Custom request ID", "path": "/api/orders", "method": "GET", "headers": {"X-Request-ID": "req-550e8400-e29b-41d4-a716-446655440000", "X-Correlation-ID": "corr-123-456"}},
    {"name": "API key header", "path": "/api/external/data", "method": "GET", "headers": {"X-API-Key": "pk_live_12345abcdef67890", "Accept": "application/json"}},
    {"name": "Forwarded header", "path": "/api/data", "method": "GET", "headers": {"X-Forwarded-For": "203.0.113.50", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "app.example.com"}},
    {"name": "If-None-Match cache", "path": "/api/products/42", "method": "GET", "headers": {"If-None-Match": "\"33a64df551425fcc55e4d42a148795d9f25f89d4\"", "If-Modified-Since": "Wed, 21 Oct 2024 07:28:00 GMT"}},
    {"name": "Range request", "path": "/api/files/large-report.pdf", "method": "GET", "headers": {"Range": "bytes=0-1023"}},

    # --- Long URLs and Complex Query Strings ---
    {"name": "Long filter query", "path": "/api/products", "method": "GET", "query": {"category": "electronics", "subcategory": "smartphones", "brand": "Samsung", "minPrice": "200", "maxPrice": "1500", "color": "black", "storage": "256GB", "rating": "4", "inStock": "true", "sort": "price", "order": "asc", "page": "1", "limit": "20"}},
    {"name": "UTM params", "path": "/landing", "method": "GET", "query": {"utm_source": "google", "utm_medium": "cpc", "utm_campaign": "spring_sale_2024", "utm_content": "banner_v2", "utm_term": "wireless+headphones+deal", "ref": "homepage_hero"}},
    {"name": "Long path", "path": "/api/v2/organizations/org-123/teams/team-456/members/user-789/permissions", "method": "GET"},
    {"name": "Encoded query", "path": "/api/search", "method": "GET", "query": {"q": "O'Brien's restaurant & bar", "location": "San Francisco, CA"}},
    {"name": "Date range query", "path": "/api/analytics", "method": "GET", "query": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z", "granularity": "day", "metrics": "pageviews,sessions,bounce_rate"}},
    {"name": "Complex encoded path", "path": "/api/files/Documents%20and%20Settings/user/My%20Documents/report.pdf", "method": "GET"},

    # --- Internationalized Content ---
    {"name": "Chinese content", "path": "/api/posts", "method": "POST", "body": {"title": "你好世界", "content": "这是一个测试帖子"}},
    {"name": "Japanese search", "path": "/api/search", "method": "GET", "query": {"q": "東京タワー観光ガイド"}},
    {"name": "Arabic content", "path": "/api/messages", "method": "POST", "body": {"text": "مرحبا بالعالم"}},
    {"name": "Korean search", "path": "/api/search", "method": "GET", "query": {"q": "서울 맛집 추천"}},
    {"name": "Emoji content", "path": "/api/comments", "method": "POST", "body": {"text": "Great product! Love it! 👍🎉✨", "rating": 5}},
    {"name": "German umlaut", "path": "/api/users", "method": "POST", "body": {"name": "Müller", "city": "München"}},
    {"name": "French accents", "path": "/api/profile", "method": "PUT", "body": {"name": "François", "bio": "Développeur à Paris"}},
    {"name": "Mixed script", "path": "/api/search", "method": "GET", "query": {"q": "café résumé naïve"}},

    # --- E-commerce Flows ---
    {"name": "Add to cart", "path": "/api/cart/add", "method": "POST", "body": {"productId": 42, "quantity": 2, "variant": {"size": "M", "color": "blue"}}},
    {"name": "Apply coupon", "path": "/api/cart/coupon", "method": "POST", "body": {"code": "SAVE20"}},
    {"name": "Checkout", "path": "/api/checkout", "method": "POST", "body": {"cartId": "cart-123", "shipping": {"method": "express", "address": {"line1": "123 Main St", "city": "Chicago", "state": "IL", "zip": "60601"}}}},
    {"name": "Track order", "path": "/api/orders/tracking", "method": "GET", "query": {"orderId": "ORD-2024-12345"}},
    {"name": "Product review", "path": "/api/products/42/reviews", "method": "POST", "body": {"rating": 4, "title": "Good quality", "text": "The product works well. Battery life could be better, but overall I'm satisfied with the purchase."}},
    {"name": "Wishlist add", "path": "/api/wishlist", "method": "POST", "body": {"productId": 42}},
    {"name": "Price compare", "path": "/api/products/compare", "method": "GET", "query": {"ids": "42,43,44,45"}},

    # --- Admin/Dashboard APIs ---
    {"name": "Dashboard stats", "path": "/api/admin/dashboard", "method": "GET", "query": {"period": "30d"}},
    {"name": "User list paginated", "path": "/api/admin/users", "method": "GET", "query": {"page": "1", "limit": "50", "role": "customer", "status": "active"}},
    {"name": "Export data", "path": "/api/admin/export", "method": "POST", "body": {"type": "users", "format": "csv", "dateRange": {"from": "2024-01-01", "to": "2024-12-31"}}},
    {"name": "Audit log", "path": "/api/admin/audit-log", "method": "GET", "query": {"action": "login", "user": "admin@example.com", "limit": "100"}},
    {"name": "Settings update", "path": "/api/admin/settings", "method": "PUT", "body": {"siteName": "My Store", "maintenanceMode": False, "allowRegistration": True}},

    # --- Notification and Messaging ---
    {"name": "Send notification", "path": "/api/notifications/send", "method": "POST", "body": {"userId": 123, "type": "email", "template": "order_confirmation", "data": {"orderId": "ORD-123"}}},
    {"name": "Get notifications", "path": "/api/notifications", "method": "GET", "query": {"unread": "true", "limit": "20"}},
    {"name": "Mark read", "path": "/api/notifications/mark-read", "method": "POST", "body": {"ids": [1, 2, 3, 4, 5]}},
    {"name": "Chat message", "path": "/api/chat/messages", "method": "POST", "body": {"conversationId": "conv-123", "text": "Hi there! Can you help me with my order?"}},
    {"name": "Email subscribe", "path": "/api/newsletter/subscribe", "method": "POST", "body": {"email": "user@example.com", "preferences": ["deals", "new_products"]}},

    # --- Social Features ---
    {"name": "Create post", "path": "/api/posts", "method": "POST", "body": {"text": "Just finished reading an amazing book! Highly recommend 'Clean Code' by Robert C. Martin.", "tags": ["books", "programming"]}},
    {"name": "Like post", "path": "/api/posts/42/like", "method": "POST"},
    {"name": "Follow user", "path": "/api/users/456/follow", "method": "POST"},
    {"name": "Share post", "path": "/api/posts/42/share", "method": "POST", "body": {"platform": "twitter", "message": "Check this out!"}},
    {"name": "Report content", "path": "/api/posts/42/report", "method": "POST", "body": {"reason": "spam", "details": "This post appears to be automated spam"}},

    # --- Dev/Debug (Legitimate) ---
    {"name": "API version", "path": "/api/version", "method": "GET"},
    {"name": "API docs", "path": "/api/docs", "method": "GET"},
    {"name": "OpenAPI spec", "path": "/api/openapi.json", "method": "GET"},
    {"name": "Metrics endpoint", "path": "/metrics", "method": "GET"},
    {"name": "Status page", "path": "/api/status", "method": "GET"},

    # --- Edge Cases That Look Suspicious But Are Benign ---
    {"name": "Password with specials", "path": "/api/auth/login", "method": "POST", "body": {"email": "user@example.com", "password": "P@ssw0rd!#$%^&*()_+-=[]{}|;':\",./<>?"}},
    {"name": "Code snippet body", "path": "/api/snippets", "method": "POST", "body": {"language": "python", "code": "import os\npath = os.path.join('/home', 'user')\nprint(f'Path: {path}')"}},
    {"name": "SQL docs post", "path": "/api/forum", "method": "POST", "body": {"title": "Understanding SQL JOINs", "body": "SELECT users.name, orders.total FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.total > 100"}},
    {"name": "Shell tutorial", "path": "/api/tutorials", "method": "POST", "body": {"title": "Bash scripting basics", "content": "Use `ls -la` to list files. Pipe with `|` and redirect with `>`. Example: cat file.txt | grep 'error' > errors.log"}},
    {"name": "XML config sample", "path": "/api/docs/examples", "method": "POST", "body": {"title": "Spring Config", "content": "<?xml version=\"1.0\"?><beans><bean id=\"myBean\" class=\"com.example.MyClass\"/></beans>"}},
    {"name": "Regex tutorial", "path": "/api/tutorials", "method": "POST", "body": {"title": "Regex patterns", "content": "Email: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$ URL: https?://[\\w.-]+(?:\\.[\\w.-]+)+"}},
    {"name": "Template docs post", "path": "/api/docs", "method": "POST", "body": {"title": "Jinja2 Guide", "content": "Use {{ variable }} for output, {% for item in items %} for loops, and {# comment #} for comments."}},
    {"name": "Math expression", "path": "/api/calculator", "method": "POST", "body": {"expression": "2 * (3 + 4) / 7 - 1"}},
    {"name": "URL in body", "path": "/api/bookmarks", "method": "POST", "body": {"url": "https://example.com/page?id=123&ref=main", "title": "Useful resource"}},
    {"name": "Path in config", "path": "/api/settings", "method": "PUT", "body": {"logPath": "/var/log/app/output.log", "dataDir": "/opt/data/processed"}},

    # --- Additional Standard Operations ---
    {"name": "Bulk delete", "path": "/api/items/bulk-delete", "method": "POST", "body": {"ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}},
    {"name": "File metadata", "path": "/api/files/metadata", "method": "GET", "query": {"path": "/documents/report.pdf"}},
    {"name": "Calendar event", "path": "/api/calendar/events", "method": "POST", "body": {"title": "Team Meeting", "start": "2024-03-15T14:00:00Z", "end": "2024-03-15T15:00:00Z", "attendees": ["alice@example.com", "bob@example.com"]}},
    {"name": "Task create", "path": "/api/tasks", "method": "POST", "body": {"title": "Review PR #42", "description": "Check the new authentication module for security issues", "assignee": "dev@example.com", "priority": "high"}},
    {"name": "Tag management", "path": "/api/tags", "method": "POST", "body": {"name": "JavaScript", "color": "#f7df1e"}},
    {"name": "Geolocation lookup", "path": "/api/geo/lookup", "method": "GET", "query": {"lat": "41.8781", "lng": "-87.6298"}},
    {"name": "Currency convert", "path": "/api/currency/convert", "method": "GET", "query": {"from": "USD", "to": "EUR", "amount": "100"}},
    {"name": "Timezone list", "path": "/api/timezones", "method": "GET", "query": {"region": "America"}},

    # --- Mobile API Patterns ---
    {"name": "Device register", "path": "/api/devices/register", "method": "POST", "body": {"deviceId": "abc123", "platform": "ios", "pushToken": "apns_token_xyz", "appVersion": "2.1.0"}},
    {"name": "Sync data", "path": "/api/sync", "method": "POST", "body": {"lastSyncTimestamp": "2024-01-15T10:30:00Z", "deviceId": "abc123"}},
    {"name": "App config", "path": "/api/config/mobile", "method": "GET", "headers": {"X-App-Version": "2.1.0", "X-Platform": "ios"}},
    {"name": "Push notification", "path": "/api/push/send", "method": "POST", "body": {"token": "fcm_token_abc", "title": "New message", "body": "You have a new message from Alice"}},

    # --- Legitimate redirect URLs ---
    {"name": "OAuth redirect legit", "path": "/api/auth/callback", "method": "GET", "query": {"redirect_uri": "https://myapp.com/callback", "code": "auth_code_123"}},
    {"name": "Post-login redirect", "path": "/api/auth/login", "method": "POST", "body": {"email": "user@example.com", "password": "pass123", "redirect": "/dashboard"}},
    {"name": "SSO return URL", "path": "/api/auth/sso/return", "method": "GET", "query": {"token": "sso_token_123", "returnTo": "https://app.example.com/home"}},

    # --- Batch/Bulk Operations ---
    {"name": "Batch fetch users", "path": "/api/users/batch", "method": "POST", "body": {"ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}},
    {"name": "Bulk status update", "path": "/api/orders/bulk-status", "method": "PUT", "body": {"orderIds": ["ORD-001", "ORD-002", "ORD-003"], "status": "processing"}},
    {"name": "Import CSV data", "path": "/api/import", "method": "POST", "body": {"format": "csv", "mapping": {"col1": "name", "col2": "email", "col3": "phone"}, "skipHeader": True}},

    # --- Content Management ---
    {"name": "Create blog post", "path": "/api/cms/posts", "method": "POST", "body": {"title": "10 Tips for Better Code Reviews", "content": "Code reviews are essential for maintaining code quality. Here are some tips: 1. Be constructive 2. Focus on the code, not the person 3. Keep reviews small", "status": "draft", "tags": ["engineering", "best-practices"]}},
    {"name": "Update page", "path": "/api/cms/pages/about", "method": "PUT", "body": {"content": "We are a team of passionate developers building tools for the modern web.", "metadata": {"lastUpdated": "2024-01-15", "author": "admin"}}},
    {"name": "Media library", "path": "/api/cms/media", "method": "GET", "query": {"type": "image", "page": "1", "limit": "24"}},

    # --- Realistic headers with potentially-suspicious-looking values ---
    {"name": "Legit long cookie", "path": "/api/data", "method": "GET", "headers": {"Cookie": "session=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMiLCJuYW1lIjoiSm9obiIsImlhdCI6MTUxNjIzOTAyMn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c; _ga=GA1.2.12345.67890; _gid=GA1.2.11111.22222"}},
    {"name": "Legit cache headers", "path": "/api/products/42", "method": "GET", "headers": {"Cache-Control": "no-cache", "Pragma": "no-cache", "If-None-Match": "W/\"abc123\""}},
    {"name": "Legit multipart CT", "path": "/api/upload", "method": "POST", "headers": {"Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryABCDEF1234567890"}},
    {"name": "Legit XML content", "path": "/api/data", "method": "POST", "headers": {"Content-Type": "application/xml", "Accept": "application/xml"}},
    {"name": "Legit gzip encoding", "path": "/api/data", "method": "POST", "headers": {"Content-Encoding": "gzip", "Content-Type": "application/json"}},
    {"name": "Legit websocket upg", "path": "/ws/chat", "method": "GET", "headers": {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}},

    # --- Search queries that look like attacks but aren't ---
    {"name": "Search SELECT keyword", "path": "/api/search", "method": "GET", "query": {"q": "how to SELECT data from PostgreSQL table"}},
    {"name": "Search DROP TABLE docs", "path": "/api/search", "method": "GET", "query": {"q": "PostgreSQL DROP TABLE IF EXISTS syntax"}},
    {"name": "Search script tag", "path": "/api/search", "method": "GET", "query": {"q": "how to use script tag in HTML5"}},
    {"name": "Search eval function", "path": "/api/search", "method": "GET", "query": {"q": "Python eval() vs exec() differences"}},
    {"name": "Search passwd file", "path": "/api/search", "method": "GET", "query": {"q": "Linux /etc/passwd file format explained"}},
    {"name": "Search LDAP protocol", "path": "/api/search", "method": "GET", "query": {"q": "how does LDAP authentication work"}},
    {"name": "Search XPath basics", "path": "/api/search", "method": "GET", "query": {"q": "XPath selector examples for web scraping"}},
    {"name": "Forum shell scripting", "path": "/api/forum", "method": "POST", "body": {"title": "Help with shell script", "body": "I need to run `cat /var/log/syslog | grep error` but it's not working as expected"}},
    {"name": "Discuss SQL injection", "path": "/api/forum", "method": "POST", "body": {"title": "Preventing SQL injection in Node.js", "body": "Always use parameterized queries like `db.query('SELECT * FROM users WHERE id = $1', [userId])` instead of string concatenation"}},
    {"name": "Discuss OWASP", "path": "/api/forum", "method": "POST", "body": {"title": "OWASP Top 10 summary", "body": "The OWASP Top 10 includes: Injection, Broken Authentication, XSS, Insecure Direct Object References..."}},

    # --- Additional edge cases ---
    {"name": "Empty body POST", "path": "/api/ping", "method": "POST", "body": {}},
    {"name": "Null body POST", "path": "/api/heartbeat", "method": "POST"},
    {"name": "Very long benign val", "path": "/api/notes", "method": "POST", "body": {"content": "This is a very long note about the quarterly sales report. " * 20}},
    {"name": "Boolean params", "path": "/api/products", "method": "GET", "query": {"inStock": "true", "featured": "false", "onSale": "true"}},
    {"name": "Negative number param", "path": "/api/offset", "method": "GET", "query": {"timezone": "-5", "offset": "-100"}},
    {"name": "UUID param", "path": "/api/sessions/550e8400-e29b-41d4-a716-446655440000", "method": "GET"},
    {"name": "Base64 in body", "path": "/api/images/upload", "method": "POST", "body": {"filename": "photo.jpg", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
    {"name": "Comma separated IDs", "path": "/api/products", "method": "GET", "query": {"ids": "1,2,3,4,5,6,7,8,9,10"}},

    # --- Additional benign requests to reach 200+ ---
    {"name": "Weather API call", "path": "/api/weather", "method": "GET", "query": {"city": "London", "units": "metric"}},
    {"name": "Translation request", "path": "/api/translate", "method": "POST", "body": {"text": "Hello world", "source": "en", "target": "fr"}},
    {"name": "PDF generation", "path": "/api/pdf/generate", "method": "POST", "body": {"template": "invoice", "data": {"invoiceId": "INV-001", "total": 150.00}}},
    {"name": "QR code generate", "path": "/api/qrcode", "method": "POST", "body": {"data": "https://example.com/product/42", "size": 256}},
    {"name": "Email send legit", "path": "/api/email/send", "method": "POST", "body": {"to": "user@example.com", "subject": "Order Confirmation", "body": "Thank you for your order #12345."}},
    {"name": "SMS send legit", "path": "/api/sms/send", "method": "POST", "body": {"phone": "+1234567890", "message": "Your verification code is 123456"}},
    {"name": "Cron job trigger", "path": "/api/cron/daily-report", "method": "POST", "headers": {"X-Cron-Key": "internal_secret_123"}},
    {"name": "Feature flags", "path": "/api/features", "method": "GET", "query": {"userId": "user-123", "platform": "web"}},
    {"name": "AB test assign", "path": "/api/ab-test/assign", "method": "POST", "body": {"experiment": "checkout_flow_v2", "userId": "user-123"}},
    {"name": "Rate limit check", "path": "/api/rate-limit/status", "method": "GET", "headers": {"X-API-Key": "pk_test_12345"}},
    {"name": "Sitemap index", "path": "/sitemap-index.xml", "method": "GET"},
    {"name": "OpenSearch desc", "path": "/opensearch.xml", "method": "GET"},
    {"name": "Service worker", "path": "/sw.js", "method": "GET"},
    {"name": "Web manifest", "path": "/.well-known/assetlinks.json", "method": "GET"},
    {"name": "Apple assoc", "path": "/.well-known/apple-app-site-association", "method": "GET"},
    {"name": "Security.txt", "path": "/.well-known/security.txt", "method": "GET"},
    {"name": "ACME challenge", "path": "/.well-known/acme-challenge/token123", "method": "GET"},
    {"name": "Logging event", "path": "/api/logs", "method": "POST", "body": {"level": "info", "message": "User logged in successfully", "timestamp": "2024-01-15T10:30:00Z", "userId": "user-123"}},
    {"name": "Deep link handler", "path": "/api/deeplink", "method": "GET", "query": {"path": "/product/42", "source": "push_notification"}},
    {"name": "Versioned API call", "path": "/api/v3/users/123", "method": "GET", "headers": {"Accept": "application/vnd.myapp.v3+json"}},
]


def test_payload(request: Dict) -> Dict:
    """Test a single benign request against WAF — expecting it to NOT be blocked."""
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    headers = {**base_headers, **request.get("headers", {})}

    payload = {
        "method": request.get("method", "GET"),
        "path": request["path"],
        "query_params": request.get("query", {}),
        "headers": headers,
    }

    body = request.get("body")
    if body is not None:
        payload["body"] = json.dumps(body) if isinstance(body, (dict, list)) else str(body)

    try:
        response = requests.post(WAF_API, json=payload, timeout=10)
        result = response.json()
        return {
            "name": request["name"],
            "blocked": result.get("is_anomaly", False),
            "score": result.get("anomaly_score", 0),
            "time_ms": result.get("processing_time_ms", 0),
        }
    except Exception as e:
        return {"name": request["name"], "error": str(e)}


def main():
    print("=" * 70)
    print("FALSE POSITIVE REGRESSION TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(BENIGN_REQUESTS)} benign requests...")
    print("Target: <1% false positive rate (benign requests should NOT be blocked)\n")

    false_positives = 0
    correct = 0
    errors = 0

    fp_details = []

    for i, req in enumerate(BENIGN_REQUESTS, 1):
        result = test_payload(req)

        if "error" in result:
            status = "\033[93m[ERROR]\033[0m"
            errors += 1
        elif result["blocked"]:
            status = "\033[91m[FALSE POSITIVE]\033[0m"
            false_positives += 1
            fp_details.append(result)
        else:
            status = "\033[92m[OK]\033[0m"
            correct += 1

        score = result.get("score", 0)
        print(
            f"[{i:03d}/{len(BENIGN_REQUESTS)}] {status} {result['name'][:40]:<40} Score: {score:.4f}"
        )
        time.sleep(0.02)

    total_tested = len(BENIGN_REQUESTS) - errors
    fp_rate = false_positives / total_tested * 100 if total_tested > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Benign Tests:  {len(BENIGN_REQUESTS)}")
    print(f"\033[92mCorrectly Allowed:   {correct}\033[0m")
    print(f"\033[91mFalse Positives:     {false_positives}\033[0m")
    print(f"\033[93mErrors:              {errors}\033[0m")
    print(f"False Positive Rate: {fp_rate:.2f}%")

    if fp_rate < 1.0:
        print(f"\033[92mSTATUS: PASS (FP rate {fp_rate:.2f}% < 1.0% target)\033[0m")
    else:
        print(f"\033[91mSTATUS: FAIL (FP rate {fp_rate:.2f}% >= 1.0% target)\033[0m")

    if fp_details:
        print(f"\n\033[91mFalse Positive Details:\033[0m")
        for fp in fp_details:
            print(f"  - {fp['name']}: score={fp.get('score', 0):.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
