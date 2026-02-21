# Iris AI — Full PPT Prompt: AI-Powered WAF Presentation

Copy everything below the line into Iris AI for PPT generation.
---

## Design & Visual Style (STRICT)

- **Background:** Use a **full black** background on every slide. Hex: **#000000**. No gradients to gray—pure black only.
- **Primary accent colors:**
  - **Cyan:** Use for titles, key headings, and primary callouts. Hex: **#00D9FF** or **#00E5FF**. Use for slide titles and section headers.
  - **Parrot green:** Use for bullet highlights, checkmarks, “good/safe” concepts, and secondary emphasis. Hex: **#39FF14** or **#00FF7F** (spring green). Use for solution bullets, “allow,” “benign,” “zero-day ready.”
- **Additional cool colors (use sparingly for variety):**
  - **Coral / neon pink:** **#FF6B6B** or **#EC4899** — for “malicious,” “block,” “attack” when you need contrast.
  - **Teal / mint:** **#2DD4BF** or **#5EEAD4** — for tech highlights (ML, Transformer, inference) if you need a third accent.
- **Body text:** Light gray or off-white for readability on black: **#E5E7EB** or **#D1D5DB**. Avoid pure white (#FFFFFF) for long paragraphs.
- **Style:** Modern, tech/security vibe. Clean sans-serif fonts. Subtle borders or glows in cyan/green on key boxes. No clip art; use simple shapes, icons, or minimal diagrams. Keep slides uncluttered with plenty of negative space (black).

---

## Slide 1: Title
- **Title:** AI-Powered Web Application Firewall: Transformer-Based Anomaly Detection  
- **Subtitle:** End-to-end WAF pipeline with zero-day readiness and real-time protection  
- **Footer:** Your name / Course / Date (placeholder)

---

## Slide 2: Introduction
- **Title:** Introduction  
- **Body:** We built an end-to-end Web Application Firewall (WAF) pipeline that uses a fine-tuned Transformer to protect web applications from malicious and unknown attacks. Instead of relying on attack signatures and rule sets, our system learns what normal traffic looks like and blocks deviations in real time—so apps stay protected even against zero-day and evasive threats. From a production-ready gateway (rate limiting, DDoS protection, and ML inspection) to a multi-tenant backend and dashboard, we deliver a complete solution that is zero-day ready and suitable for B2B use. Our goal is to show how modern NLP and anomaly detection can make WAFs more effective with less operational burden. Let this presentation guide you through the problem, the approach, and the architecture.

---

## Slide 3: Problem Statement
- **Title:** Problem Statement  
- **Body (3 points):**  
  1. **Zero-day and evasive attacks slip through** — Traditional WAFs rely on attack signatures and rule sets (e.g. OWASP ModSecurity CRS) that match known patterns like SQL injection and XSS. They miss zero-day exploits, novel attacks, and evasive techniques (encoding, fragmentation, obfuscation).  
  2. **Rules are a constant, manual battle** — New CVEs and bypasses appear continuously. Keeping rules up to date and tuning them is time-consuming and app-specific.  
  3. **No generalization to unseen attacks** — Signature-based systems cannot generalize; they only block what is explicitly described in rules. Web applications remain exposed to emerging and unknown threats.

---

## Slide 4: Our Innovative Solutions
- **Title:** Our Innovative Solutions  
- **Body (6 solution cards, use 2 rows of 3 or list):**  
  1. **Learn Normal Behavior (No Signatures)** — WAF learns from benign traffic only; flags deviations in real time; zero-day and evasive attacks blocked.  
  2. **Real-Time Anomaly Detection at the Edge** — Every request scored by DistilBERT; low-latency inference; 0–100 attack score; configurable block/challenge thresholds.  
  3. **End-to-End Pipeline** — Gateway + backend + dashboard; Docker, PostgreSQL, Redis; one command to run.  
  4. **Multi-Tenant B2B Ready** — Tenant isolation by API key; per-org metrics, alerts, config; optional per-tenant models.  
  5. **Tune Without Retraining** — Adjust block/challenge thresholds; fail-open or fail-closed; dashboard shows score distribution.  
  6. **Rate Limiting and DDoS Protection** — Redis-backed rate limit; burst detection; temporary IP blocking.

---

## Slide 5: Why Transformers?
- **Title:** Why Transformers?  
- **Body:**  
  - Pre-trained language understanding (DistilBERT): captures structure and semantics of token sequences; HTTP requests treated as sequences; attention works for abnormal patterns.  
  - Fine-tuned on HTTP request sequences for binary classification (benign vs malicious); optional semi-supervised training with synthetic attack payloads.  
  - Efficiency: DistilBERT ~40% smaller and faster than BERT — suitable for real-time inference at the edge.  
  - Anomaly detection: train on benign traffic only; deviations = anomalies; no large labeled attack corpus needed.  
  - Why not classic ML? Transformers learn from raw request text; less feature engineering; better generalization.

---

## Slide 6: High-Level Architecture
- **Title:** High-Level Architecture  
- **Body:**  
  - B2B SaaS: customer traffic → WAF gateway → applications; multi-tenant isolation by API key / tenant ID.  
  - **WAF Gateway (Edge):** Reverse proxy → Rate limit → DDoS check → WAF inspect (ML) → Score → Allow/Block → forward to origin.  
  - **Backend API (FastAPI):** WAF config, ML model serving, PostgreSQL, Redis; receives events and metrics.  
  - **Dashboard (Next.js):** Per-org metrics, alerts, activity feed, config, API keys; REST + WebSocket.  
  - **Protected origins:** Customer apps (e.g. app.customer-a.com); gateway forwards only after checks pass.  
- **Visual:** Simple flowchart: Client → Gateway (Proxy → Rate Limit → DDoS → ML Inspect → Decision) → Backend/Dashboard; Decision → Allow → Origins.

---

## Slide 7: Request Flow (Pipeline)
- **Title:** Request Flow (Pipeline) — Step by Step  
- **Body (numbered):**  
  1. Traffic enters WAF Gateway (HTTPS); TLS terminated.  
  2. Rate limiting: Redis-backed; per-IP/per-tenant; on exceed → 429 + Retry-After.  
  3. DDoS check: burst detection, max request size, temporary IP blocking; on exceed → 413 or 429.  
  4. WAF inspection: request serialized and sent to ML service.  
  5. Model inference: parse → normalize → tokenize (max 512) → DistilBERT → attack score (0–100), label.  
  6. Decision: block (403), challenge, or allow; fail-open/fail-closed when ML unavailable.  
  7. Events and metrics to backend; dashboard shows alerts and charts.

---

## Slide 8: ML Pipeline — Training
- **Title:** ML Pipeline — Training (Detail)  
- **Body:**  
  - Training data: 26,589+ unique benign samples from Juice Shop, WebGoat, DVWA; crawl, log parsing, synthetic.  
  - Normalization: replace dynamic values (UUIDs, timestamps, session IDs) with placeholders; preserve structure.  
  - Tokenization: HTTP-aware tokenizer; vocab from training data; max length 512.  
  - Model: DistilBERT base; binary classifier (benign vs malicious); optional semi-supervised with synthetic attacks.  
  - Output: saved model and vocabulary; versioned for rollback.

---

## Slide 9: ML Pipeline — Inference
- **Title:** ML Pipeline — Inference (Detail)  
- **Body:**  
  - Input: raw HTTP request; optional truncation for latency (< 100 ms p99).  
  - Processing: parse → same normalization as training → tokenize → pad/truncate 512 → model forward pass.  
  - Output: label, confidence, attack score 0–100, latency metrics.  
  - Configurable: block threshold (e.g. ≥ 80), challenge threshold (e.g. 50–79); fail-open/fail-closed.  
  - Async or batched inference; timeout to avoid blocking gateway.

---

## Slide 10: Attack Score and Thresholds
- **Title:** Attack Score and Thresholds (Detail)  
- **Body:**  
  - Attack score 0–100: single scalar; derived from model confidence; higher = more likely malicious.  
  - Block threshold (e.g. 80): score ≥ 80 → 403 Forbidden.  
  - Challenge threshold (e.g. 50–79): optional CAPTCHA or allow with flag.  
  - Allow (score < 50): request forwarded; event sent for analytics.  
  - Tuning: adjust thresholds without retraining; dashboard shows score distribution.

---

## Slide 11: Normalization — Why It Matters
- **Title:** Normalization — Why It Matters  
- **Body:**  
  - Goal: learn patterns (e.g. “path has ID segment”), not specific values (e.g. user ID 12345).  
  - Replace: UUIDs, timestamps, session/CSRF tokens, numeric IDs with placeholders ({UUID}, {TIMESTAMP}, {ID}).  
  - Preserve: method, path structure, query names, header names, body structure.  
  - Result: same normalization in training and inference; better generalization.

---

## Slide 12: Key Features
- **Title:** Key Features (Detail)  
- **Body:** Zero-day ready • Real-time protection • Live dashboard (Next.js) • Production-ready (Docker Compose) • Rate limiting (Redis) • DDoS protection • Continuous learning (fine-tune, versioning, hot-swap).

---

## Slide 13: Tech Stack
- **Title:** Tech Stack (Detail)  
- **Body:**  
  - ML: PyTorch, Hugging Face Transformers, DistilBERT; HTTP-aware tokenizer.  
  - Backend: FastAPI; PostgreSQL; Redis.  
  - Frontend: Next.js; REST + WebSocket.  
  - Infrastructure: Docker, Nginx, PostgreSQL, Redis.  
  - Demo: Juice Shop, WebGoat, DVWA.

---

## Slide 14: Project Structure
- **Title:** Project Structure (Codebase)  
- **Body:** backend/ (FastAPI, WAF, ML) • frontend/ (Next.js dashboard) • gateway/ (proxy, rate limit, DDoS, WAF inspect) • applications/ (Juice Shop, WebGoat, DVWA) • models/ (DistilBERT, vocab) • scripts/ (training, stress tests) • docs/ (phases, architecture).

---

## Slide 15: Implementation Phases (10-Day)
- **Title:** Implementation Phases (10-Day Pipeline)  
- **Body:** Phase 1: Environment & app deployment • Phase 2: Log ingestion • Phase 3: Parsing & normalization • Phase 4: Tokenization • Phase 5: DistilBERT training • Phase 6: WAF integration • Phase 7: Real-time detection • Phase 8: Continuous learning • Phase 9: Testing & validation • Phase 10: Deployment & demo.

---

## Slide 16: Demo Applications
- **Title:** Demo Applications (Detail)  
- **Body:** OWASP Juice Shop (~9k samples) • OWASP WebGoat (~8.6k) • DVWA (~8.9k). Crawl/replay for benign data; run attacks to validate WAF blocks.

---

## Slide 17: Quick Start (For Judges/Demo)
- **Title:** Quick Start (For Judges/Demo)  
- **Body:** Clone → cp .env.example .env • docker-compose up -d • Dashboard: http://localhost:3000 • Gateway-only option • Verify with benign and malicious requests.

---

## Slide 18: Why This Matters (Impact)
- **Title:** Why This Matters (Impact)  
- **Body:** Security: protects from known and unknown attacks • Scalability: multi-tenant B2B • Research/Education: NLP/Transformers for security • Practical: Docker, config-driven, production-ready.

---

## Slide 19: Conclusion & Thank You
- **Title:** Conclusion & Thank You  
- **Body:** Summary: End-to-end WAF with DistilBERT; learns normal behavior; zero-day ready; gateway + backend + dashboard. Deliverables: Gateway, Backend API, Dashboard, model, scripts, docs. Thank you — Questions?  
- **Footer:** Institution logo, your name, date.

---

## Reminder for Iris AI
- Use **only** the colors specified: black #000000 background; cyan and parrot green as primary accents; coral/neon pink and teal/mint as optional secondary accents; light gray/off-white for body text. No purple or gold.
- Keep every slide background **pure black (#000000)**.
- Tone: professional, academic, clear. Suitable for professor and large audience (~150 students).
- One slide per section above (19 slides total). Add a simple flowchart/diagram for slides 6 and 7 if the tool supports it; otherwise use bullet flow.
