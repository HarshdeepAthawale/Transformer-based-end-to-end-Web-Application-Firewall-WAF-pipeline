# Transformer-Based End-to-End Web Application Firewall (WAF) Pipeline

---

## Slide 1: Title
- **Title:** AI-Powered Web Application Firewall: Transformer-Based Anomaly Detection
- **Subtitle:** End-to-end WAF pipeline with zero-day readiness and real-time protection
- **Your name / Course / Date** (add as needed)

---

## Slide 2: The Problem
- **Traditional WAFs rely on attack signatures and rule sets**
  - Miss zero-day attacks, novel exploits, and evasive techniques
  - Keeping rules up to date is a constant, manual battle
  - Signature-based systems cannot generalize to unseen attacks
- **Result:** Web applications remain exposed to emerging and unknown threats

---

## Slide 3: Our Solution
- **Unsupervised anomaly detection using a Transformer (DistilBERT)**
  - Learns what **normal** traffic looks like from benign requests only
  - Flags **deviations** in real time—no attack signatures required
  - Attacks that look nothing like the baseline get blocked, **even if never seen before**
- **Tagline:** *No signatures, no rules. Just learned behavior.*

---

## Slide 4: Why Transformers?
- **Pre-trained language understanding** (DistilBERT) captures structure and semantics of HTTP requests
- **Fine-tuned on HTTP request sequences** for binary classification (benign vs malicious)
- **Efficient for real-time inference** — lighter than BERT, suitable for edge/production
- **Anomaly detection approach:** train on benign traffic only; deviations = anomalies

---

## Slide 5: High-Level Architecture
- **B2B SaaS style:** customer traffic flows through the WAF before reaching their applications
- **WAF Gateway (Edge):** Reverse proxy → Rate limit → DDoS check → Inspect (ML) → Score → Allow/Block
- **Backend API (FastAPI):** WAF config, ML model serving, tenant isolation, PostgreSQL, Redis
- **Dashboard (Next.js):** Multi-tenant; per-org metrics, alerts, config, API keys
- **Protected origins:** Customer apps (e.g. app.customer-a.com, api.customer-b.com)

---

## Slide 6: Request Flow (Pipeline)
1. **Traffic** enters the WAF Gateway (HTTPS).
2. **Rate limiting** (Redis-backed, per-IP).
3. **DDoS check** (burst detection, request size limits, temporary IP blocking).
4. **WAF inspection:** Request (method, path, query, headers, body) sent to ML service.
5. **Model inference:** Fine-tuned DistilBERT returns **attack score (0–100)** and label (benign/malicious).
6. **Decision:** Block, challenge, or allow based on configurable thresholds.
7. **Events and metrics** sent to backend; dashboard shows alerts and activity.

---

## Slide 7: ML Pipeline — Training
- **Training data:** 26,589+ unique benign samples from three vulnerable apps (Juice Shop, WebGoat, DVWA)
- **Modes:** Crawl (live apps), log parsing (Nginx/Apache), synthetic generation, or combined
- **Normalization:** Replace dynamic values (UUIDs, timestamps, IDs); preserve request structure for generalization
- **Model:** DistilBERT fine-tuned as binary classifier (benign vs malicious); optional synthetic attack payloads for semi-supervised training
- **Output:** Saved model (e.g. `models/waf-distilbert`) for deployment

---

## Slide 8: ML Pipeline — Inference
- **Input:** Raw HTTP request (method, path, query, headers, body) — optionally truncated for latency
- **Processing:** Parsing pipeline normalizes request → tokenizer (max length 512) → model forward pass
- **Output:** Label (benign/malicious), confidence, **attack score 0–100**, latency metrics
- **Configurable:** Block/challenge thresholds; fail-open or fail-closed when model unavailable

---

## Slide 9: Key Features
- **Zero-day ready** — Learns from benign traffic; no attack signatures
- **Real-time protection** — Low-latency inference; blocks threats before they hit the app
- **Live dashboard** — Next.js: metrics, charts, alerts, activity feed
- **Production-ready** — Docker Compose, Nginx, PostgreSQL, Redis; one command to run
- **Rate limiting** — Redis-backed per-IP throttling; configurable requests per minute
- **DDoS protection** — Burst detection, request size limits, temporary IP blocking
- **Continuous learning** — Fine-tune on new traffic; adapt to evolving patterns

---

## Slide 10: Tech Stack
- **ML:** PyTorch, Hugging Face Transformers, DistilBERT (fine-tuned)
- **Backend:** FastAPI, multi-tenant API, WAF config, ML serving
- **Frontend:** Next.js dashboard
- **Infrastructure:** Docker, Nginx (reverse proxy), PostgreSQL, Redis
- **Demo/Testing:** OWASP Juice Shop, OWASP WebGoat, DVWA (intentionally vulnerable apps)

---

## Slide 11: Project Structure (Codebase)
- **backend/** — FastAPI API, WAF middleware, ML inference (WAFClassifier)
- **frontend/** — Next.js dashboard
- **gateway/** — Reverse proxy + WAF inspection (rate limit, DDoS, ML score)
- **applications/** — Demo apps: Juice Shop, WebGoat, DVWA for testing
- **models/** — Trained DistilBERT model
- **scripts/** — Fine-tuning, training data generation, stress tests, threshold sweeps
- **docs/** — Phase-by-phase implementation guides (10 phases, Day 1–10)

---

## Slide 12: Implementation Phases (10-Day Pipeline)
1. **Phase 1:** Environment setup & web app deployment  
2. **Phase 2:** Log ingestion (batch + streaming)  
3. **Phase 3:** Request parsing & normalization  
4. **Phase 4:** Tokenization & sequence preparation  
5. **Phase 5:** Transformer (DistilBERT) architecture & training  
6. **Phase 6:** WAF integration with web server (Nginx/Apache)  
7. **Phase 7:** Real-time non-blocking detection (async, batching, timeout)  
8. **Phase 8:** Continuous learning & incremental model updates  
9. **Phase 9:** Testing, validation & performance tuning  
10. **Phase 10:** Deployment, monitoring & demo preparation  

---

## Slide 13: Demo Applications
- **OWASP Juice Shop** — Modern vulnerable web app (REST API, auth, shopping); 30 endpoints, ~9k benign samples
- **OWASP WebGoat** — Security lessons (SQLi, XSS, auth, JWT, etc.); 20 endpoints, ~8.6k samples
- **DVWA** — Damn Vulnerable Web Application (SQLi, XSS, file upload, etc.); 25 endpoints, ~8.9k samples  
- Used for **benign traffic generation** and **attack testing** to validate the WAF.

---

## Slide 14: Quick Start (For Judges/Demo)
- Clone repo → `cp .env.example .env` → `docker-compose up -d`
- Dashboard: **http://localhost:3000**
- Gateway-only (rate limit + DDoS): `docker compose -f docker-compose.gateway.yml up -d`
- Full stack: dashboard + backend + gateway + Redis + PostgreSQL

---

## Slide 15: Why This Matters (Impact)
- **Security:** Protects web apps from known and **unknown** attacks without maintaining huge rule sets
- **Scalability:** Same pipeline can serve multiple tenants (B2B); model can be fine-tuned per customer
- **Research/Education:** Demonstrates applying NLP/Transformers to security (anomaly detection on HTTP traffic)
- **Practical:** Docker-based, config-driven, with dashboard and APIs — close to production use

---

## Slide 16: Conclusion & Thank You
- **Summary:** We built an end-to-end WAF pipeline that uses a fine-tuned Transformer to detect malicious HTTP requests by learning normal behavior — no signatures, zero-day ready.
- **Deliverables:** Gateway (proxy + rate limit + DDoS + ML), Backend API, Dashboard, trained model, scripts, and documentation.
- **Thank you** — Questions?

---

## Notes for Iris.ai
- Use this content as **slide-by-slide source** for generating the PPT.
- Each "Slide N" block can map to one or two slides; adjust based on preferred density.
- Add institution logo, your name, and date on title and thank-you slides.
- Suggested tone: **professional, academic, clear** — suitable for a professor and 150 students.
