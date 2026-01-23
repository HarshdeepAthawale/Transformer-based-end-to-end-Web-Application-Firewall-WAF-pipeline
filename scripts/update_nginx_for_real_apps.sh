#!/bin/bash
# Update Nginx configuration to proxy to real applications

set -e

echo "zenbook" | sudo -S cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup.real-apps 2>/dev/null || true

# Update Nginx upstream configuration
echo "zenbook" | sudo -S tee /etc/nginx/conf.d/waf-real-apps.conf > /dev/null << 'EOF'
# Upstream servers for real applications
upstream juice_shop {
    server localhost:8080;
}

upstream webgoat {
    server localhost:8081;
}

upstream dvwa {
    server localhost:8082;
}

# Server block for Juice Shop
server {
    listen 80;
    server_name juice-shop.local;

    location / {
        proxy_pass http://juice_shop;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Server block for WebGoat
server {
    listen 80;
    server_name webgoat.local;

    location / {
        proxy_pass http://webgoat;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Server block for DVWA
server {
    listen 80;
    server_name dvwa.local;

    location / {
        proxy_pass http://dvwa;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Default server - route to Juice Shop
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass http://juice_shop;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

echo "✓ Nginx configuration updated for real applications"
echo "zenbook" | sudo -S nginx -t && echo "✓ Nginx configuration test passed" || echo "✗ Nginx configuration test failed"
echo "zenbook" | sudo -S systemctl reload nginx && echo "✓ Nginx reloaded" || echo "✗ Nginx reload failed"
