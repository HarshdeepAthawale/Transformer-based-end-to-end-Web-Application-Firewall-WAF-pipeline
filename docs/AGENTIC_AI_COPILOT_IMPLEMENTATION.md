# Agentic AI Copilot — Implementation Documentation

> 5-Agent AI pipeline for the WAF Dashboard. Users chat with a security copilot that can investigate threats, analyze traffic, explain concepts, perform forensics, and suggest/execute remediation actions.

---

## Architecture Overview

```
User Message
     │
     ▼
┌──────────┐   keyword regex    ┌──────────────┐
│  Router   │ ─────────────────▶│  Specialist   │
│ (0ms)     │   (no LLM call)   │  Agent        │
└──────────┘                    └──────┬───────┘
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌──────────┐
                    │  Tool 1  │ │  Tool 2  │ │  Tool N  │
                    │ (wraps   │ │ (wraps   │ │ (wraps   │
                    │ ctrl)    │ │ ctrl)    │ │ ctrl)    │
                    └──────────┘ └──────────┘ └──────────┘
                          │            │            │
                          ▼            ▼            ▼
                    ┌─────────────────────────────────────┐
                    │  Existing Controllers & Services     │
                    │  (alerts, threats, traffic, etc.)    │
                    └─────────────────────────────────────┘
```

**Key design decisions:**
- Intent routing via keyword regex — zero-latency classification, no nested LLM call
- Tools call existing controllers in-process (no HTTP overhead)
- SSE streaming with tool-use progress events
- Experience store with keyword-based retrieval for learning from feedback
- 30-second timeout on agent execution (configurable)
- Sliding-window rate limiter (10 RPM per IP, configurable)

---

## File Inventory

### New Files Created (26)

#### Backend — Agent Pipeline (`backend/agents/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `context.py` | `AgentContext` dataclass — holds `db`, `waf_service`, `user_id`, `session_id` |
| `llm_client.py` | `AsyncOpenAI` client factory from env vars (`OPENAI_API_KEY`, `AGENT_BASE_URL`, `AGENT_MODEL`) |
| `base_agent.py` | `BaseAgent` class — OpenAI tool-calling loop (max 6 steps), streaming + non-streaming modes |
| `router.py` | `classify_intent()` — regex keyword classifier returning `AgentIntent` enum |
| `orchestrator.py` | `Orchestrator` class — main entry point: classify → retrieve experience → run specialist → save |
| `experience_store.py` | `ExperienceStore` — save/retrieve/feedback for conversation turns, keyword LIKE search |

#### Backend — Tools (`backend/agents/tools/`)

| File | Tools (28 total) | Wraps |
|------|-------------------|-------|
| `registry.py` | `ToolRegistry` + `ToolDef` dataclass | — |
| `investigation_tools.py` | `get_active_alerts`, `get_recent_threats`, `get_threat_stats`, `get_recent_traffic`, `get_realtime_metrics`, `get_threats_by_type`, `get_traffic_by_endpoint` | `alerts`, `threats`, `traffic`, `metrics` controllers |
| `remediation_tools.py` | `block_ip`, `unblock_ip`, `whitelist_ip`, `dismiss_alert`, `acknowledge_alert`, `create_security_rule`, `create_geo_rule` | `ip_management`, `alerts`, `security_rules`, `geo_rules` controllers |
| `analytics_tools.py` | `get_analytics_overview`, `get_analytics_trends`, `get_analytics_summary`, `get_request_chart`, `get_threat_chart` | `analytics`, `charts` controllers |
| `forensics_tools.py` | `get_audit_logs`, `get_ip_reputation`, `get_traffic_by_ip`, `get_threat_timeline` | `audit`, `ip_management`, `traffic`, `threats` controllers |
| `explainer_tools.py` | `get_bot_signatures`, `get_security_rules`, `get_owasp_rules`, `get_ip_reputation_explain`, `explain_threat_type` | `bot_detection`, `security_rules`, `ip_management` controllers + static knowledge dict |

#### Backend — Specialist Agents (`backend/agents/specialists/`)

| File | Intent | Tools Used |
|------|--------|------------|
| `investigator.py` | INVESTIGATE | 7 investigation tools |
| `analyst.py` | ANALYZE | 5 analytics + 1 metrics tool |
| `explainer.py` | EXPLAIN | 5 explainer tools |
| `forensics.py` | FORENSICS | 4 forensics tools |
| `remediation.py` | REMEDIATE | 7 remediation + 4 investigation tools, `extract_suggested_actions()` parser |

#### Backend — Model & Routes

| File | Purpose |
|------|---------|
| `backend/models/agent_experience.py` | `AgentExperience` SQLAlchemy model — stores conversation turns, feedback, keywords |
| `backend/routes/agent.py` | 5 API endpoints under `/api/agent/` |
| `backend/middleware/agent_rate_limit.py` | Sliding-window rate limiter for `/api/agent/*` |

#### Frontend

| File | Purpose |
|------|---------|
| `frontend/lib/agent-api.ts` | `agentApi` client — `chatStream()` (SSE parser), `chat()`, `submitFeedback()`, `executeAction()` |
| `frontend/app/copilot/page.tsx` | Copilot page — starter prompts, streaming chat, session management |
| `frontend/components/copilot/chat-message.tsx` | Message bubble — `react-markdown` rendering, feedback buttons, suggested actions |
| `frontend/components/copilot/intent-badge.tsx` | Colored badge showing agent intent (Investigating, Remediating, etc.) |
| `frontend/components/copilot/typing-indicator.tsx` | Animated spinner with tool name label |
| `frontend/components/copilot/suggested-action-button.tsx` | Action button with loading/done/error states |

### Existing Files Modified (4)

| File | Change |
|------|--------|
| `requirements.txt` | Added `openai>=1.0.0` |
| `backend/database.py` | Imported `agent_experience` model in `init_db()` |
| `backend/main.py` | Added `("agent", "/api/agent", "ai-agent")` to `advanced_routes`; registered `AgentRateLimitMiddleware` |
| `frontend/components/sidebar.tsx` | Added `{ icon: Bot, label: 'AI Copilot', href: '/copilot' }` after Overview |

### Dependency Added

| Package | Where | Purpose |
|---------|-------|---------|
| `openai>=1.0.0` | `requirements.txt` (Python) | AsyncOpenAI client for LLM calls |
| `react-markdown` | `package.json` (npm) | Markdown rendering in chat messages |

---

## API Reference

### `POST /api/agent/chat`

Non-streaming chat endpoint.

**Request:**
```json
{
  "message": "show me active alerts",
  "session_id": "optional-session-uuid"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "content": "Here are the currently active alerts...",
    "intent": "investigate",
    "experience_id": 42,
    "session_id": "uuid",
    "suggested_actions": [],
    "tools_used": ["get_active_alerts"]
  },
  "timestamp": "2026-02-15T12:00:00"
}
```

### `POST /api/agent/chat/stream`

SSE streaming chat. Same request body as `/chat`.

**SSE Events:**
```
data: {"type": "intent", "intent": "investigate"}
data: {"type": "tool_use", "tool": "get_active_alerts"}
data: {"type": "token", "content": "Here "}
data: {"type": "token", "content": "are the "}
...
data: {"type": "done", "experience_id": 42, "session_id": "uuid", "suggested_actions": [], "intent": "investigate"}
```

Error event:
```
data: {"type": "error", "message": "Request timed out"}
```

### `POST /api/agent/feedback/{experience_id}`

**Request:**
```json
{
  "score": 1,
  "text": "optional feedback text"
}
```

Score must be `1` (thumbs up) or `-1` (thumbs down).

### `POST /api/agent/action/execute`

Execute a suggested remediation action.

**Request:**
```json
{
  "action": "block_ip",
  "params": {"ip": "1.2.3.4", "reason": "Malicious activity"}
}
```

Whitelisted actions: `block_ip`, `unblock_ip`, `whitelist_ip`, `dismiss_alert`, `acknowledge_alert`, `create_security_rule`, `create_geo_rule`.

### `GET /api/agent/history?session_id={uuid}`

Returns conversation history for a session.

---

## Intent Router

The router uses regex keyword patterns to classify messages. No LLM call — deterministic and fast.

| Intent | Priority | Trigger Keywords (subset) |
|--------|----------|--------------------------|
| REMEDIATE | 1 (highest) | block, unblock, whitelist, ban, dismiss, create rule, mitigate, fix, stop attack |
| FORENSICS | 2 | forensic, audit, timeline, trace, who changed, ip reputation, track |
| INVESTIGATE | 3 | alert, threat, attack, suspicious, breach, what's happening, under attack |
| ANALYZE | 4 | analyze, trend, traffic pattern, metric, chart, summary, how many |
| EXPLAIN | 5 (lowest) | explain, what is, how does, teach, owasp, best practice, definition |

Default (no match): INVESTIGATE

---

## Experience Store

### How It Works

1. **Save**: After each agent response, the orchestrator saves the turn with extracted keywords
2. **Retrieve**: Before running a specialist, the orchestrator queries for similar past experiences using keyword LIKE matching + intent filter
3. **Enrich**: Relevant past context is appended to the user message as context
4. **Feedback**: Users can rate responses (thumbs up/down), which influences retrieval ranking

### Keyword Extraction

- Tokenizes message into words
- Removes stop words (a, the, is, show, get, help, please, etc.)
- Removes words shorter than 3 characters
- Deduplicates preserving order
- Stored as space-separated string for `LIKE '%keyword%'` queries

### Retrieval Ranking

1. Positive feedback first (`feedback_score DESC`)
2. Most recent first (`timestamp DESC`)
3. Maximum 3 results per query

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for OpenAI-compatible provider |
| `AGENT_BASE_URL` | `None` | Base URL override (Groq, Together, local vLLM, etc.) |
| `AGENT_MODEL` | `gpt-4o-mini` | Model name |
| `AGENT_TIMEOUT_SECONDS` | `30` | Max seconds per agent execution |
| `AGENT_RATE_LIMIT_RPM` | `10` | Max requests per minute per IP for agent endpoints |

### Compatible Providers

The `llm_client.py` factory uses `AsyncOpenAI` which works with any OpenAI-compatible API:

- **OpenAI**: Set `OPENAI_API_KEY` (no base URL needed)
- **Groq**: Set `AGENT_BASE_URL=https://api.groq.com/openai/v1` + Groq API key
- **Together AI**: Set `AGENT_BASE_URL=https://api.together.xyz/v1` + Together key
- **Local vLLM**: Set `AGENT_BASE_URL=http://localhost:8000/v1` + any dummy key

---

## Frontend Copilot UI

### Page: `/copilot`

- **Empty state**: 6 starter prompt cards covering all 5 intents
- **Chat area**: Scrollable message list with auto-scroll on new content
- **Input area**: Text input with Enter-to-send, disabled during streaming
- **Session persistence**: Session ID maintained across messages in a conversation

### Component Hierarchy

```
CopilotPage
├── Sidebar (with AI Copilot nav item)
├── Header
└── main
    ├── Empty state (starter prompt cards)
    │   └── StarterPromptCard × 6
    ├── Message list
    │   └── ChatMessage × N
    │       ├── IntentBadge
    │       ├── ReactMarkdown (assistant content)
    │       ├── SuggestedActionButton × N
    │       └── Feedback buttons (thumbs up/down)
    ├── TypingIndicator (during tool use)
    └── Input form
```

### Design Language

Follows the Positivus design system used throughout the dashboard:
- `rounded-none` (sharp corners)
- CSS variables: `--positivus-black`, `--positivus-white`, `--positivus-green`, `--positivus-gray`, `--positivus-gray-dark`
- Font: `var(--font-space-grotesk)` for UI text
- 2px solid borders

### Streaming Flow

1. User sends message → user bubble appears immediately
2. Empty assistant bubble appears
3. `onIntent` → intent badge renders
4. `onToolUse` → typing indicator shows tool name
5. `onToken` → content streams into assistant bubble (markdown rendered live)
6. `onDone` → experience ID set, suggested actions appear, feedback buttons enabled

---

## Verification

### Backend Smoke Tests

```bash
# Non-streaming chat
curl -X POST http://localhost:3001/api/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "show me active alerts"}'

# Streaming chat
curl -N -X POST http://localhost:3001/api/agent/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message": "analyze traffic trends"}'

# Feedback
curl -X POST http://localhost:3001/api/agent/feedback/1 \
  -H 'Content-Type: application/json' \
  -d '{"score": 1}'

# Execute action
curl -X POST http://localhost:3001/api/agent/action/execute \
  -H 'Content-Type: application/json' \
  -d '{"action": "block_ip", "params": {"ip": "1.2.3.4", "reason": "test"}}'

# History
curl http://localhost:3001/api/agent/history?session_id=YOUR_SESSION_ID
```

### Frontend

1. Navigate to `/copilot`
2. Verify sidebar shows "AI Copilot" item after Overview
3. Click a starter prompt card
4. Verify: intent badge → tool indicator → streaming content → feedback buttons
5. Test thumbs up/down
6. For remediation queries, test suggested action buttons

### Experience Retrieval

1. Send: "show me active alerts"
2. Rate the response (thumbs up)
3. Send: "what alerts are currently active?"
4. The second response should include context from the first (visible in enriched message if you check server logs)

---

## Database Schema

### `agent_experiences` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | DATETIME | UTC creation time |
| `session_id` | VARCHAR(64) | Conversation session |
| `user_id` | INTEGER | Nullable user reference |
| `user_message` | TEXT | Original user input |
| `agent_intent` | VARCHAR(32) | Classified intent |
| `agent_response` | TEXT | Final agent response |
| `tools_used` | TEXT (JSON) | List of tool names called |
| `tool_call_count` | INTEGER | Number of tool calls |
| `suggested_actions` | TEXT (JSON) | Extracted action suggestions |
| `feedback_score` | INTEGER | 1 (up), -1 (down), NULL |
| `feedback_text` | TEXT | Optional feedback text |
| `feedback_at` | DATETIME | When feedback was given |
| `keywords` | TEXT | Space-separated keywords for retrieval |
| `steps_taken` | INTEGER | Tool-calling loop steps |
| `response_length` | INTEGER | Character count of response |

**Indexes:**
- `ix_agent_exp_session_ts` — (session_id, timestamp)
- `ix_agent_exp_intent_score` — (agent_intent, feedback_score)
