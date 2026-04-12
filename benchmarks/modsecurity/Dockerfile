FROM owasp/modsecurity-crs:nginx

# The OWASP CRS image comes with:
#   - ModSecurity v3 compiled into nginx
#   - OWASP Core Rule Set v4 pre-installed
#   - Detection mode by default (ANOMALY_INBOUND score threshold = 5)
#
# We switch to blocking mode and return 403 on detected attacks.

ENV MODSEC_RULE_ENGINE=On
ENV ANOMALY_INBOUND=5
ENV ANOMALY_OUTBOUND=4
ENV BLOCKING_PARANOIA=1

# Backend is a simple echo server (nginx returns 200 for everything not blocked)
ENV BACKEND=http://localhost:8888

# Add a minimal upstream echo server config
COPY echo-backend.conf /etc/nginx/conf.d/echo-backend.conf

EXPOSE 80
