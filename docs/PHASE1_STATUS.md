# Phase 1 — Edge Network & Performance: What’s Built vs Remaining

This document is an honest status of **Phase 1** from the CLOUDFLARE_KILLER_PLAN: what is implemented end-to-end and what is still missing.

---

## 1.1 Global Anycast Edge Network (CDN Layer)

| Item | Status | Notes |
|------|--------|--------|
| **Edge PoP architecture** |  Not built | Deploying to Fly.io / Workers / bare-metal is **infra/DevOps**, not app code. No automation added. |
| **Anycast DNS routing** |  Not built | Would need PowerDNS + GeoIP2 and BGP/GeoDNS; out of scope for this pass. |
| **Edge caching layer** |  Built | Two-tier cache (in-process LRU + Redis), Cache-Control/ETag/Vary, stale-while-revalidate. **Wired into gateway**: GET/HEAD go through cache; X-Cache: HIT/MISS/STALE. Enable with `EDGE_CACHE_ENABLED=true`. |
| **Cache purge API** |  Built | `POST /api/v1/cache/purge` — body: `urls`, `tags`, `prefixes`, or `purge_everything`. Publishes to Redis; gateways subscribe and purge. Purge-by-URL uses a URL index (all variants for that URL). |
| **Cache analytics** |  Built | `GET /api/v1/cache/analytics` — hit ratio, bandwidth saved, per-day stats from Redis (gateways write stats). |
| **Request coalescing** |  Built | In `gateway/edge_cache.py`: `RequestCoalescer` deduplicates concurrent cache-miss fetches to the same key. Coalescing is **not** yet used in the proxy path (only lookup → miss → forward → store). |

---

## 1.2 DNS-as-a-Service

| Item | Status | Notes |
|------|--------|--------|
| **Authoritative DNS server** |  Not built | No PowerDNS or other DNS server; no DNSSEC signing. |
| **DNS proxy mode** |  Data model only | `DNSRecord.proxied` exists (“orange cloud” toggle). No gateway or DNS server that actually proxies based on it. |
| **DNS analytics** |  Not built | No query volume, latency, or NXDOMAIN tracking. |
| **Automatic DNS record import** |  Not built | No zone scan/import. |
| **DNS zones & records CRUD** |  Built | `GET/POST/PATCH/DELETE /api/v1/dns/zones`, `GET/POST/PATCH/DELETE /api/v1/dns/zones/{zone_id}/records`. |

---

## 1.3 SSL/TLS Management

| Item | Status | Notes |
|------|--------|--------|
| **Automatic HTTPS (ACME)** |  Not built | Config has `ACME_*` and `CERT_*` vars; no ACME client, no HTTP-01/DNS-01, no cert issuance. |
| **Full / Strict SSL modes** |  Data model only | `SSLSettings.ssl_mode` (off, flexible, full, full_strict). No TLS termination in this app; modes are not enforced. |
| **Automatic HTTPS rewrites** |  Stub | `gateway/https_rewrite.py`: `should_redirect_to_https`, `build_hsts_header`. Not wired into gateway response (no redirect or HSTS injection yet). |
| **Custom certificates** |  Built | `POST/GET/PATCH/DELETE /api/v1/ssl/certificates` — store cert PEM and (encrypted) key. No loading into a TLS terminator. |
| **TLS 1.3 + 0-RTT** |  Data model only | `SSLSettings.min_tls_version`, `tls_0rtt_enabled`. Not enforced (no terminator). |
| **HSTS preload** |  Data model only | `SSLSettings.hsts_*`. Not sent in responses. |
| **Minimum TLS version** |  Data model only | Stored in `SSLSettings`; not enforced. |
| **Client certificate mTLS** |  Data model only | `SSLSettings.mtls_enabled`, `mtls_ca_cert_pem`. Not enforced. |
| **Certificate transparency monitoring** |  Not built | No CT log monitoring or alerts. |
| **SSL settings CRUD** |  Built | `GET/POST/PATCH/DELETE /api/v1/ssl/settings`, `GET/PATCH/DELETE /api/v1/ssl/settings/{domain}`. |

---

## Summary

- **Implemented end-to-end (usable):**
  - Edge HTTP cache in gateway (with purge + analytics APIs and Redis pub/sub).
  - Cache purge API and cache analytics API.
  - DNS zones and records CRUD APIs.
  - SSL certificates and SSL settings CRUD APIs.
  - `https_rewrite` helper module (stub; not yet wired for redirect/HSTS).

- **Implemented partially (data/API only, no runtime behavior):**
  - DNS “proxy” flag and SSL modes/settings (stored only; no DNS server or TLS terminator).

- **Not implemented:**
  - Edge PoP deployment, Anycast/GeoDNS, authoritative DNS, DNSSEC, ACME/auto HTTPS, HTTPS redirect/HSTS in responses, CT monitoring, and request coalescing in the proxy path.

To **run Phase 1 cache**: set `EDGE_CACHE_ENABLED=true` and `REDIS_URL`; gateways will cache GET/HEAD and subscribe to purge. Use `POST /api/v1/cache/purge` and `GET /api/v1/cache/analytics` from the backend.
