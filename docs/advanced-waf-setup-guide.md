# Advanced WAF Setup Guide - Complete AI Integration

## Overview
This guide provides step-by-step instructions for setting up the **Advanced WAF with full AI integration** using OpenResty and Lua scripting for real-time anomaly detection.

## Prerequisites
- Arch Linux system
- Root/sudo access
- Internet connection for package downloads
- Python virtual environment with dependencies

## 🚀 Complete Setup Process

### Step 1: Install OpenResty with Lua Support

```bash
# Run the OpenResty installation script
sudo ./scripts/setup_openresty_arch.sh
```

This script will:
- ✅ Install OpenResty from AUR (or build from source)
- ✅ Install required Lua modules (lua-resty-http, lua-cjson)
- ✅ Create systemd service for OpenResty
- ✅ Set up configuration directories

### Step 2: Configure Advanced WAF Integration

```bash
# Run the advanced WAF configuration
sudo ./scripts/setup_complete_advanced_waf.sh
```

This script will:
- ✅ Configure OpenResty for WAF integration
- ✅ Set up Lua scripting for request interception
- ✅ Configure real-time AI anomaly detection
- ✅ Set up rate limiting and security headers
- ✅ Enable comprehensive logging

### Step 3: Verify the Setup

```bash
# Run verification tests
sudo ./scripts/verify_advanced_waf.sh
```

Expected output should show 80%+ success rate.

## 🔧 Manual Component Setup

If you prefer to set up components individually:

### 3.1 Start WAF Service

```bash
# Navigate to project directory
cd /path/to/waf-project

# Activate Python virtual environment
source venv/bin/activate

# Start WAF service with real model
python scripts/start_waf_service.py \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 4 \
    --log_level info
```

### 3.2 Start Backend Application

```bash
# Example: Juice Shop (OWASP vulnerable app)
cd applications/app1-juice-shop
npm start

# Or any other app on port 8080
```

## 🧪 Testing the Advanced WAF

### Basic Functionality Test

```bash
# Test health endpoints
curl http://127.0.0.1:8000/health
curl http://localhost/waf-metrics

# Test basic proxying
curl -v http://localhost/

# Test API endpoints
curl -v http://localhost/api/test
```

### AI Anomaly Detection Tests

```bash
# Test normal request (should pass)
curl -v "http://localhost/api/products?page=1&limit=10"

# Test suspicious requests (should be blocked/analyzed)
curl -v "http://localhost/api/users?id=1' OR '1'='1"
curl -v "http://localhost/api/search?<script>alert('xss')</script>"

# Test direct WAF service
curl -X POST http://127.0.0.1:8000/check \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","path":"/api/test","query_params":{"id":"1 UNION SELECT * FROM users"}}'
```

### Security Feature Tests

```bash
# Check security headers
curl -I http://localhost/

# Test rate limiting (rapid requests)
for i in {1..20}; do curl -s http://localhost/ > /dev/null & done

# Test automated tool blocking
curl -H "User-Agent: sqlmap/1.6.5" http://localhost/
```

## 📊 Monitoring and Logs

### Real-time Monitoring

```bash
# WAF service metrics
curl http://localhost/waf-metrics

# OpenResty access logs
tail -f /var/log/openresty/access.log

# OpenResty error logs
tail -f /var/log/openresty/error.log

# Systemd service logs
journalctl -u openresty -f
```

### Log Analysis

```bash
# Search for WAF blocks
grep "WAF BLOCK" /var/log/openresty/access.log

# Search for anomalies
grep "anomaly" /var/log/openresty/access.log

# Monitor request processing times
grep "processing_time" /var/log/openresty/access.log
```

## 🔧 Configuration Tuning

### Adjust Anomaly Detection Threshold

```bash
# Via API (runtime adjustment)
curl -X POST http://127.0.0.1:8000/update-threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.3}'

# Or restart service with new threshold
python scripts/start_waf_service.py --threshold 0.3
```

### Modify Rate Limiting

Edit `/usr/local/openresty/nginx/conf/sites-available/waf-integration`:

```nginx
# Adjust these values
limit_req_zone $binary_remote_addr zone=waf_check:10m rate=200r/s;  # Increase from 100
limit_req_zone $binary_remote_addr zone=backend:10m rate=2000r/s; # Increase from 1000
```

Then reload:
```bash
sudo systemctl reload openresty
```

## 🚨 Troubleshooting

### Common Issues

**1. OpenResty won't start**
```bash
# Check configuration
sudo openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf

# Check logs
journalctl -u openresty --no-pager -n 50

# Manual start for debugging
sudo openresty -c /usr/local/openresty/nginx/conf/nginx.conf
```

**2. Lua module not found**
```bash
# Check Lua paths
lua -e "print(package.path)"
lua -e "print(package.cpath)"

# Verify module installation
ls -la /usr/local/openresty/lualib/resty/
```

**3. WAF service connection refused**
```bash
# Check if WAF service is running
ps aux | grep waf_service

# Check service health
curl http://127.0.0.1:8000/health

# Restart service
pkill -f waf_service
python scripts/start_waf_service.py
```

**4. Backend application not responding**
```bash
# Check backend on port 8080
curl http://127.0.0.1:8080/

# Check if backend is running
netstat -tlnp | grep 8080
```

### Performance Optimization

**1. Increase worker processes**
Edit OpenResty config and adjust:
```nginx
worker_processes auto;  # or specific number
worker_connections 102400;  # increase from 65536
```

**2. Optimize Lua caching**
Add to nginx.conf:
```nginx
lua_code_cache on;
lua_shared_dict waf_cache 10m;
```

**3. Database connection pooling**
Ensure backend apps use connection pooling to handle increased load.

## 📈 Performance Benchmarks

### Expected Performance (on typical hardware)

- **Throughput**: 500+ requests/second with AI analysis
- **Latency**: <50ms average, <200ms P95
- **Memory**: ~200MB base + model size
- **CPU**: Minimal additional load beyond normal proxying

### Scaling Recommendations

- **Single Instance**: Up to 1000 req/sec
- **Load Balancer**: Multiple OpenResty instances
- **WAF Service**: Horizontal scaling with multiple workers
- **Caching**: Redis for session/state management

## 🔒 Security Considerations

### Production Deployment

1. **SSL/TLS Configuration**
   ```nginx
   listen 443 ssl http2;
   ssl_certificate /path/to/cert.pem;
   ssl_certificate_key /path/to/key.pem;
   ```

2. **Access Control**
   ```nginx
   allow 192.168.1.0/24;
   deny all;
   ```

3. **Log Security**
   - Rotate logs regularly
   - Secure log storage
   - Monitor for sensitive data leakage

4. **Rate Limiting Tuning**
   - Adjust based on legitimate traffic patterns
   - Implement progressive delays
   - Use geo-blocking for known bad actors

## 🎯 Next Steps

### Phase 7: Real-Time Non-Blocking Detection
- Implement async anomaly detection
- Add machine learning model updates
- Create dashboard for real-time monitoring
- Implement alert system

### Advanced Features
- Integration with SIEM systems
- Custom rule engine
- Threat intelligence feeds
- Automated response actions

## 📞 Support

### Logs to Check
- `/var/log/openresty/error.log` - OpenResty errors
- `/var/log/openresty/access.log` - Access logs with WAF data
- WAF service terminal output - Model inference logs
- `journalctl -u openresty` - Systemd service logs

### Configuration Files
- `/usr/local/openresty/nginx/conf/nginx.conf` - Main config
- `/usr/local/openresty/nginx/conf/sites-available/waf-integration` - WAF config
- `config/config.yaml` - WAF service settings

The advanced WAF setup provides enterprise-grade AI-powered web application protection with real-time anomaly detection, comprehensive logging, and production-ready reliability! 🚀