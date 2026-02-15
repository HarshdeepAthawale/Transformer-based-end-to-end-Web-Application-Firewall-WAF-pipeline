# AI Agents for the WAF Platform — Research & Approach

**Purpose:** Ground research and recommended approach for adding AI agents to the Transformer-based WAF pipeline (dashboard, backend, WAF service).

---

## 1. What “AI Agents” Mean Here

In this context, **AI agents** are systems that:

- Use an **LLM** (or similar) for reasoning and language understanding.
- Call **tools** (APIs, DB queries, scripts) to read data and take actions.
- Operate in a **loop**: plan → use tools → interpret results → decide next step or respond.
- Can be **single-agent** (one LLM with many tools) or **multi-agent** (specialist agents that coordinate).

For your platform, the two main directions are:

1. **Agents that enhance the WAF product** — e.g. a “WAF Copilot” for investigation, triage, and automation inside your dashboard/backend.
2. **Protecting AI agents and LLM apps** — e.g. extending the WAF to secure agent tool-calls and LLM inputs (prompt injection, tool abuse), sometimes called “WAF for AI” or GAF (Generative Application Firewall).

This doc focuses on **(1)** as the primary “adding AI agents to our platform” use case, with **(2)** noted as a future product angle.

---

## 2. Industry Direction (2024–2025)

| Source | What they do |
|--------|----------------|
| **Microsoft Security Copilot + Azure WAF** | Natural language investigation of WAF logs/events, top rules triggered, top offending IPs; investigation → triage → remediation in SOC workflows. |
| **Akamai / Cloudflare** | AI inside WAF for detection and policy; Cloudflare “Firewall for AI” for protecting LLM/agent surfaces. |
| **F5** | WAF extended to protect MCP (Model Context Protocol) servers and agent tool endpoints. |
| **GAF (Generative Application Firewall)** | New layer for LLM/agent apps: prompt filters, guardrails, tool-call checks in one enforcement point. |

Takeaways:

- **WAF + AI Copilot** is the dominant pattern: NL queries over WAF data, guided investigation/triage/remediation.
- **Agent tool security** (MCP, tool-calling APIs) is an emerging need; your WAF can later protect those endpoints.

---

## 3. Recommended Approach: WAF Copilot (Agents for Your Platform)

### 3.1 High-level idea

Add an **AI assistant (Copilot)** that:

- Talks to your **existing backend** (alerts, traffic, threats, metrics, security rules) via **tools**.
- Lets users ask in **natural language** and get answers or suggested actions (e.g. “Why did we block this IP?”, “Summarize last 24h attacks”, “Suggest a rule for this pattern”).
- Optionally **automates** repetitive tasks (e.g. triage, adding IP to blocklist, drafting rule changes) with human-in-the-loop.

### 3.2 Where it fits in your stack

You already have:

- **Backend:** FastAPI, DB, APIs for alerts, traffic, threats, metrics, security rules, IP management, etc.
- **Frontend:** Next.js dashboard (overview, traffic, alerts, analytics, threats).
- **WAF service:** FastAPI microservice with classifier (anomaly score, allow/block).

The agent layer sits **between the user and your APIs**:

```
User (Dashboard / Chat UI)
        ↓
   WAF Copilot (Agent)
   - LLM (reasoning)
   - Tool layer (calls your backend APIs + DB)
        ↓
   Your Backend (FastAPI) + DB + WAF Service
```

So: **one primary agent** (the Copilot) with many **tools** that map to your existing routes and services.

### 3.3 Core “tools” the agent should use

These map directly to what you already have:

| Tool | Backend / source | Purpose |
|------|-------------------|--------|
| Get active alerts | `GET /alerts/active` | List current alerts for context. |
| Get alert history | `GET /alerts/history` | Trends, recurring issues. |
| Get traffic / requests | Traffic API | Inspect requests, top IPs, methods, paths. |
| Get threats | Threats API | Threat summary and categories. |
| Get metrics / charts | Metrics / charts APIs | High-level stats and time series. |
| Get security rules | Security rules API | Current rules for “why was this blocked?”. |
| Get IP / geo / blocklist | IP management, geo, threat intel | Explain blocks, suggest blocklist. |
| (Optional) WAF check | WAF service `check_request` | “Is this request malicious?” on demand. |

The agent **orchestrator** (see below) decides when to call which tool and how to combine results into an answer or recommendation.

### 3.4 Architecture options

**Option A — Single-agent with tools (recommended first step)**

- One LLM (e.g. OpenAI / Azure / open model via API).
- A **tool layer**: each tool = one or more backend calls (REST or internal).
- Orchestration: **LangChain**, **LangGraph**, or **custom FastAPI** “agent” endpoint that:
  - Receives user message.
  - Calls LLM with tool definitions; LLM returns “call tool X with params Y”.
  - Backend runs tool (calls your APIs), returns result to LLM.
  - Repeats until LLM responds to the user (or max steps).
- **No MCP required** for v1; REST from your backend to your own APIs is enough.

**Option B — MCP (Model Context Protocol) for tools**

- Expose your backend capabilities as **MCP servers** (resources + tools).
- The same Copilot LLM connects to MCP to get “tools” and “resources”.
- Pros: standard protocol, reusable by other MCP clients. Cons: extra moving parts and security surface (see MCP security notes below).
- **Recommendation:** Consider MCP once the Copilot is stable and you want to open the same tools to other agents or products.

**Option C — Multi-agent (later)**

- E.g. “Investigator” agent (queries alerts/traffic), “Triage” agent (suggests true/false positive), “Remediation” agent (suggests rules or blocklist changes).
- Coordination via a “supervisor” or LangGraph-style graph.
- **Recommendation:** Start with a single agent; split into specialists only if you see clear workflow boundaries (e.g. triage vs remediation).

### 3.5 How you’ll use it (concrete)

1. **Dashboard integration**
   - Add a **Copilot panel or chat** in the dashboard (e.g. next to Alerts / Activity or in the header).
   - User asks: “What were the top attack types in the last 24 hours?” or “Why is IP X blocked?”
   - Frontend sends the message to a new backend endpoint, e.g. `POST /api/copilot/chat`.
   - Backend runs the agent (LLM + tools), streams or returns the reply; optional “suggested actions” (e.g. “Add to blocklist”, “View rule”).

2. **Backend**
   - New module, e.g. `backend/copilot/` or `backend/agents/`:
     - **Agent orchestration:** prompt + tool definitions, loop (LLM → tool calls → LLM).
     - **Tool implementations:** thin wrappers over your existing routes or DB (alerts, traffic, threats, metrics, rules, IP/geo).
   - One public endpoint: `POST /copilot/chat` (and optionally `GET /copilot/stream` for streaming).
   - Auth: reuse your existing auth; only authenticated users can hit the Copilot.

3. **Safety and control**
   - **Read-heavy first:** most tools only read data; few or no “write” tools initially (e.g. only “suggest rule” that copies to clipboard or opens a form, no direct “create rule”).
   - **Human-in-the-loop:** any destructive or sensitive action (block IP, change rule) as a suggested step that the user confirms in the UI.
   - **Audit:** log all Copilot requests and tool calls (who, when, which tools, which params) for compliance and debugging.

4. **Cost and latency**
   - Use a single, cost-effective model for v1; cache repeated queries (e.g. “top attacks last 24h”) if needed.
   - Set max steps per turn (e.g. 5–10) to avoid long loops; timeout per tool call.

---

## 4. Securing AI Agents (WAF for AI) — Later Product Angle

Your platform can later **protect** AI agents and LLM apps:

- **Prompt injection / jailbreaks:** detect or block malicious prompts.
- **Tool abuse:** validate and rate-limit tool-call payloads (e.g. MCP or REST tools used by the agent).
- **Data exfiltration / PII:** mask or block sensitive data in prompts or tool responses.

This aligns with “Firewall for AI” and GAF: your existing WAF pipeline (parsing, normalization, Transformer model) could be extended with:

- A **semantic/session-aware** layer for LLM traffic (prompts, tool calls, responses).
- Rules or a small classifier for “suspicious prompt” or “suspicious tool call”.

Implementation can be a separate service or a mode in your existing WAF service that handles “LLM request” vs “HTTP request” differently. No need to do this in the first phase of “adding AI agents to our platform.”

---

## 5. MCP and Security (If You Adopt MCP Later)

If you expose tools via **Model Context Protocol**:

- **Authentication and authorization:** per-user or per-service tokens; scope tools (e.g. read-only vs write).
- **Input validation:** validate and sanitize all inputs to MCP tools (you’re already used to this for WAF).
- **Audit and logging:** log every tool call and result (for security and compliance).
- **Sandboxing:** run agent or MCP server in a constrained environment; limit which internal APIs it can call.
- **Human-in-the-loop:** sensitive actions (block IP, delete rule) should not be fully automated; require approval in the UI.

---

## 6. Summary: Approach and How You’ll Use It

| Aspect | Recommendation |
|--------|----------------|
| **What to build first** | A **WAF Copilot**: one AI agent with tools over your existing backend (alerts, traffic, threats, metrics, rules, IP/geo). |
| **Where it lives** | New backend module (`/copilot` or `/agents`) + chat/panel in the existing Next.js dashboard. |
| **Architecture** | Single-agent, LLM + tool-calling; orchestration via LangChain/LangGraph or custom loop in FastAPI. |
| **Tools** | Read-only at first (alerts, traffic, threats, metrics, rules); write actions as “suggestions” with human confirmation. |
| **How users use it** | Natural language in the dashboard: investigation (“Why was this blocked?”), summaries (“Top attacks last 24h”), and suggested next steps (view rule, add to blocklist). |
| **Later** | Multi-agent if needed; MCP if you want a standard tool protocol; “WAF for AI” to protect LLM/agent endpoints. |

This gives you a clear, industry-aligned path: **add a Copilot-style agent that uses your current APIs as tools**, then iterate toward multi-agent or “WAF for AI” as product requirements evolve.

---

## 7. Agentic AI Pipeline (Multi-Agent Integration)

For an **agentic AI pipeline** (multi-agent, tool-using, not regular ML), see:

**[AGENTIC_AI_PIPELINE_INTEGRATION.md](./AGENTIC_AI_PIPELINE_INTEGRATION.md)**

That doc defines:
- **5 agents:** 1 Orchestrator (router) + 4 specialists (Investigator, Triage, Remediation, Reporter)
- **Per-agent purposes and tool-to-API mapping**
- **How the pipeline integrates** into the product (backend module, routes, frontend Copilot)
- **Orchestrator-led coordination** and implementation order
