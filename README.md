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
- Open **http://localhost:8080** for the protected app (when using the gateway)

See [docs/README.md](docs/README.md) for full setup, environment variables, and deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard                       │
│                   (Next.js on port 3000)                    │
│         Real-time metrics, traffic logs, charts             │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API + WebSocket
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                     Backend API Server                       │
│                  (FastAPI on port 3001)                     │
│         WAF Middleware + ML Model + Database                │
└──┬─────────┬────────┬────────────────────────────────┬──────┘
   │         │        │                                │
   ↓         ↓        ↓                                ↓
┌─────┐  ┌──────┐ ┌──────────────────────────┐   ┌─────────┐
│Redis│  │Postgres│ WAF ML Model              │   │WebSocket│
└─────┘  └──────┘ │(DistilBERT Fine-tuned)   │   └─────────┘
                  └──────────────────────────┘

                 ↓ Protects ↓

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Juice Shop  │  │   WebGoat    │  │     DVWA     │
│  Port 8080   │  │  Port 8081   │  │  Port 8082   │
└──────────────┘  └──────────────┘  └──────────────┘
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
| `applications/` | Juice Shop, WebGoat, DVWA (protected apps)   |
| `models/`       | Trained DistilBERT model                     |
| `scripts/`      | Fine-tuning, stress tests, threshold sweeps  |
| `docs/`         | Phase guides and detailed documentation      |

---

## Documentation

Full documentation lives in [docs/](docs/), including phase-by-phase implementation guides, architecture notes, and deployment procedures.
