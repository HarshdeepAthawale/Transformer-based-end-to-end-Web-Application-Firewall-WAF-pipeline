# AI-Powered Web Application Firewall

Protect your web apps with Transformer-based anomaly detection. No signatures, no rules. Just learned behavior.

---

## The Problem

Traditional WAFs rely on attack signatures and rule sets. They miss zero-days, novel exploits, and evasive techniques. Keeping rules up to date is a constant battle.

## The Solution

This WAF uses unsupervised anomaly detection. It learns what normal traffic looks like from your benign requests, then flags deviations in real time. Attacks that look nothing like your baseline get blocked—even if they have never been seen before.

---

## Key Features

**Zero-Day Ready** — Learns from benign traffic; no attack signatures required.

**Real-Time Protection** — Low-latency inference; blocks threats before they hit your app.

**Live Dashboard** — Next.js dashboard with metrics, charts, alerts, and activity feed.

**Production-Ready** — Docker Compose, Nginx, PostgreSQL, Redis; one command to run.

**Rate Limiting** — Redis-backed per-IP throttling; configurable requests per minute.

**DDoS Protection** — Burst detection, request size limits, and temporary IP blocking.

**Continuous Learning** — Fine-tune on new traffic; adapt to evolving patterns.

---

## Quick Start

```bash
git clone https://github.com/HarshdeepAthawale/Transformer-based-end-to-end-Web-Application-Firewall-WAF-pipeline.git
cd Transformer-based-end-to-end-Web-Application-Firewall-WAF-pipeline
cp .env.example .env
docker-compose up -d
```

- Open **http://localhost:3000** for the dashboard

To run only the WAF gateway with Redis (rate limiting and DDoS): `docker compose -f docker-compose.gateway.yml up -d`.

See [docs/README.md](docs/README.md) for full setup, environment variables, and deployment.

---

## Architecture

B2B SaaS model: each customer connects their apps; traffic flows through the WAF before reaching their origins.

```
                    Customer Traffic (HTTPS)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│            WAF Gateway (Reverse Proxy · Edge)               │
│     Rate Limit → DDoS Check → Inspect → Score → Allow/Block  │
└──────────────────────┬──────────────────────────────────────┘
                       │ Events, metrics (per tenant)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                     Customer Dashboard                       │
│                   (Next.js · Multi-tenant)                   │
│         Per-org metrics, alerts, config, API keys             │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API + WebSocket
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                     Backend API Server                       │
│                  (FastAPI · Multi-tenant)                    │
│         WAF config + ML Model + Tenant isolation              │
└──┬─────────┬────────┬────────────────────────────────┬──────┘
   │         │        │                                │
   ↓         ↓        ↓                                ↓
┌─────┐  ┌──────┐ ┌──────────────────────────┐   ┌─────────┐
│Redis│  │Postgres│ WAF ML Model              │   │WebSocket│
└─────┘  └──────┘ │(DistilBERT Fine-tuned)   │   └─────────┘
                  └──────────────────────────┘

                 ↓ Protects (per customer) ↓

┌──────────────────────────────────────────────────────────────┐
│                   Customer Origins (B2B)                     │
│     app.customer-a.com · api.customer-b.com · ...            │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

PyTorch · DistilBERT · FastAPI · Next.js · Docker · PostgreSQL · Redis

---

## Project Structure

| Directory       | Purpose                                      |
| --------------- | -------------------------------------------- |
| `backend/`      | FastAPI API, WAF middleware, ML inference    |
| `frontend/`     | Next.js dashboard                            |
| `gateway/`      | Reverse proxy + WAF inspection               |
| `applications/` | Demo apps (Juice Shop, WebGoat, DVWA) for testing |
| `models/`       | Trained DistilBERT model                     |
| `scripts/`      | Fine-tuning, stress tests, threshold sweeps  |
| `docs/`         | Phase guides and detailed documentation      |

---

## Documentation

Full documentation lives in [docs/](docs/), including phase-by-phase implementation guides, architecture notes, and deployment procedures.
