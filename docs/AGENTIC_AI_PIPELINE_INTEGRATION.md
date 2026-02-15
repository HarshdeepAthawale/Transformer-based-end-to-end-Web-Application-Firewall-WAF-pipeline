# Agentic AI Pipeline — Integration & Agent Design

**Scope:** Multi-agent, tool-using pipeline (agentic AI), not the existing ML anomaly model. The Transformer WAF stays as the detection layer; agents sit on top for reasoning, investigation, triage, and remediation.

---

## 1. Agentic vs Regular ML (Clarification)

| Aspect | Your existing pipeline (ML) | Agentic AI pipeline (this doc) |
|--------|---------------------------|--------------------------------|
| **What it is** | DistilBERT anomaly model: one forward pass per request → score → allow/block | LLM-based agents that use **tools** and **reasoning loops** over your data |
| **Role** | Real-time request classification (sensor) | Investigation, triage, summarization, suggested actions (copilot / SOC assistant) |
| **Input** | Single HTTP request (method, path, headers, body) | User intent (natural language) + context from tools |
| **Output** | Anomaly score, allow/block | Answers, summaries, suggested rules, triage labels |
| **Integration** | WAF service + middleware | New backend module + Copilot UI in dashboard |

They **coexist**: ML handles traffic in the hot path; agents handle analyst workflows and product UX.

---

## 2. How Many Agents and Their Purposes

Use **4 specialist agents + 1 orchestrator** = **5 agents** total. Each specialist has a focused role and its own tool set; the orchestrator routes the user query and can chain specialists.

### 2.1 Agent roster

| # | Agent | Purpose | When it runs |
|---|--------|---------|----------------|
| 1 | **Orchestrator (Router)** | Understands user intent, picks which specialist(s) to call, aggregates answers, returns final response | Every user turn; decides workflow |
| 2 | **Investigator** | Deep-dive on alerts, traffic, and threats: “what happened?”, “why was this blocked?”, “show me requests from IP X” | Queries about events, causality, logs |
| 3 | **Triage** | Assess alerts/events: true positive vs false positive, severity, similarity to past incidents | “Is this a real attack?”, “Prioritize these alerts” |
| 4 | **Remediation** | Suggests and drafts actions: block IP, add rule, adjust geo/bot config; no direct writes without approval | “What should I do?”, “Suggest a rule for this” |
| 5 | **Reporter** | Summaries, trends, top-N lists, time-bound reports (e.g. “last 24h attacks”, “weekly summary”) | Dashboards-in-chat, executive summary |

### 2.2 Why 5 (and not 1 or 10)

- **One agent:** Would need many tools and a very long system prompt; harder to keep behavior focused and to tune per task.
- **Many tiny agents:** Overhead and latency; harder to maintain. Your product doesn’t need 10+ micro-agents for v1.
- **Five:** Matches the SOC flow (investigate → triage → remediate) plus routing and reporting. Clear boundaries, one agent per “job” in the pipeline.

---

## 3. Agentic Pipeline Architecture

### 3.1 Flow (high level)

```
User message (e.g. "Why did we block 192.168.1.5? Suggest a rule.")
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (Router)                                           │
│  - Classifies intent (investigate / triage / remediate / report) │
│  - Calls one or more specialists with sub-questions or context   │
│  - Merges results and formats final reply                        │
└─────────────────────────────────────────────────────────────────┘
        │
        ├──► Investigator (tools: alerts, traffic, threats, WAF check)
        ├──► Triage         (tools: alerts, traffic, threat stats, history)
        ├──► Remediation   (tools: rules, IP/geo/bot config, draft actions)
        └──► Reporter      (tools: metrics, charts, analytics, trends)
        │
        ▼
Response to user (+ optional suggested actions for UI)
```

### 3.2 Coordination model

- **Orchestrator in the driver’s seat:** Single entry point. It has a small, fixed set of “meta-tools”: *invoke_investigator*, *invoke_triage*, *invoke_remediation*, *invoke_reporter*. It can call one or several in sequence (e.g. first Investigator, then Remediation with that context).
- **Specialists are tool-using agents:** Each has its own LLM call + real tools (your backend APIs). No direct agent-to-agent calls; all go through the orchestrator.
- **Implementation options:** LangGraph (graph with orchestrator node + 4 specialist nodes) or a custom loop in FastAPI: orchestrator runs → gets tool calls → runs specialist(s) → passes results back to orchestrator → final answer.

### 3.3 Tool-to-API mapping per agent

Your existing routes stay as-is; each agent gets a **tool layer** that calls them (internal HTTP or in-process controller calls).

| Agent | Tools (conceptual) | Backend APIs / source |
|-------|--------------------|------------------------|
| **Investigator** | get_active_alerts, get_alert_history, get_traffic, get_traffic_by_ip, get_threats, get_threat_type, waf_check_request | `/api/alerts/*`, `/api/traffic/*`, `/api/threats/*`, `/api/waf/check` |
| **Triage** | get_active_alerts, get_alert_history, get_traffic_for_alert, get_threat_stats, get_similar_incidents | `/api/alerts/*`, `/api/traffic/*`, `/api/threats/stats`, (similarity: same data, agent-side or simple heuristic) |
| **Remediation** | get_security_rules, get_blacklist_whitelist, get_geo_rules, get_bot_signatures, draft_block_ip, draft_rule, draft_geo_rule | `/api/rules/*`, `/api/ip/*`, `/api/geo/*`, `/api/bots/*` — draft only; no POST until user confirms |
| **Reporter** | get_metrics_realtime, get_metrics_historical, get_charts_requests, get_charts_threats, get_analytics_overview, get_analytics_trends, get_analytics_summary | `/api/metrics/*`, `/api/charts/*`, `/api/analytics/*` |
| **Orchestrator** | invoke_investigator, invoke_triage, invoke_remediation, invoke_reporter (with params) | Internal: calls specialist agents |

---

## 4. How This Integrates Into Your Product

### 4.1 Backend

- **New module:** e.g. `backend/agents/` (or `backend/copilot/`).
  - `orchestrator.py` — router agent + loop that invokes specialists.
  - `investigator.py`, `triage.py`, `remediation.py`, `reporter.py` — one module per specialist; each defines its tools and one “run(message, context)” entry.
  - `tools/` — shared tool implementations that call your existing controllers or HTTP clients to `/api/alerts`, `/api/traffic`, etc.
  - `schemas.py` — request/response models for chat and suggested actions.
- **New routes:** e.g. under `backend/routes/agents.py` or `copilot.py`:
  - `POST /api/agents/chat` — main entry: body `{ "message": "...", "conversation_id?": "..." }`, returns `{ "reply": "...", "suggested_actions": [...] }`.
  - `POST /api/agents/chat/stream` — optional SSE/streaming for the reply.
- **Registration in `main.py`:** Include the new router with prefix `/api/agents` (or `/api/copilot`). Reuse existing auth and rate limiting.

### 4.2 Frontend (dashboard)

- **Entry point:** Add a **Copilot** entry in the sidebar (e.g. “Copilot” or “AI Assistant”) that links to `/copilot`, or a floating chat button that opens a slide-over panel on any page.
- **Page:** e.g. `frontend/app/copilot/page.tsx` — chat UI: message list, input, and display of suggested actions (e.g. “Add to blocklist”, “View rule”) that trigger existing dashboard flows (e.g. open IP management with IP pre-filled).
- **API usage:** Call `POST /api/agents/chat` (and optionally stream endpoint) from the chat component; send auth (session/token) as with the rest of the app.

### 4.3 Data flow (single turn)

1. User types in Copilot UI → `POST /api/agents/chat` with `message` and optional `conversation_id`.
2. Backend runs **Orchestrator** with the message; orchestrator may call **Investigator**, **Triage**, **Remediation**, **Reporter** via its meta-tools.
3. Each specialist, when invoked, uses its tools (your APIs) and returns a structured summary to the orchestrator.
4. Orchestrator produces final **reply** and optional **suggested_actions** (e.g. `{ "type": "block_ip", "payload": { "ip": "..." } }`).
5. Frontend shows the reply and renders buttons/links for suggested actions (human-in-the-loop: “Block IP” opens IP management or a confirmation modal, no direct POST from the agent).

### 4.4 Where the “pipeline” lives

- **Agentic pipeline** = Orchestrator + 4 specialists + their tools. It runs only when the user interacts with the Copilot (on-demand), not on every HTTP request.
- **ML pipeline** = ingestion → parsing → normalization → tokenization → DistilBERT → score → allow/block. It runs in the WAF service/middleware on every request.
- Integration point: the **Investigator** (and optionally Remediation) can call **WAF check** as a tool (e.g. “is this sample request malicious?”) so the agentic pipeline can *use* the ML pipeline as a tool, but the two pipelines stay separate.

---

## 5. Suggested Implementation Order

1. **Scaffold** — Create `backend/agents/`, tool stubs that call your existing APIs, and one route `POST /api/agents/chat` that runs a **single** agent (e.g. Investigator only) with tools, no orchestrator yet.
2. **Orchestrator + Investigator** — Add the orchestrator that can route to Investigator; implement Investigator’s tools against `/api/alerts`, `/api/traffic`, `/api/threats`, `/api/waf/check`.
3. **Reporter** — Add Reporter agent and orchestrator routing for summary/trend queries; connect to `/api/metrics`, `/api/charts`, `/api/analytics`.
4. **Triage** — Add Triage agent and routing for “is this real?”, “prioritize” type questions.
5. **Remediation** — Add Remediation with draft-only tools and suggested_actions in the response; wire UI to suggested actions (confirm then call existing APIs).
6. **Frontend** — Copilot page + (optional) floating button + suggested-action handling.

This keeps the agentic pipeline clearly separated from your regular ML and gives you a concrete agent count (5), roles, and a path to integrate into the product.
