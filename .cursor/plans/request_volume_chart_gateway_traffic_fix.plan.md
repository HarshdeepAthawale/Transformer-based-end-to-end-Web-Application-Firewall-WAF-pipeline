---
name: ""
overview: ""
todos: []
isProject: false
---

# Plan: Fix Request Volume & Threats Chart (No Data from Gateway Traffic)

## Problem

The **Request Volume & Threats** chart shows "No data available / Waiting for traffic data..." because:

1. **Chart data source**: For ranges 1h, 6h, 24h the frontend uses the "realtime" path and calls `GET /api/traffic/recent`, which reads from the `**traffic_logs**` table.
2. **Who writes to `traffic_logs**`: Only the **backend WAF middleware** writes to `traffic_logs`, when requests hit the FastAPI backend and go through that middleware.
3. **Gateway traffic**: All traffic through the **gateway** is reported only to `POST /api/events/ingest`, which writes to `**security_events**` only. No `TrafficLog` rows are created from gateway events.

So gateway traffic never populates the table the chart uses → chart stays empty.

---

## Goal

Ensure **Request Volume & Threats** (and related traffic views) show data when traffic goes through the **gateway**, by making ingested gateway events also feed the traffic pipeline that the chart uses.

---

## Approach: Write TrafficLog from Events Ingest (Recommended)

**Idea**: When the backend receives events at `POST /api/events/ingest`, create a **TrafficLog** row for each event in addition to the existing **SecurityEvent** row. The chart and `GET /api/traffic/recent` already read from `traffic_logs`, so no frontend or chart API changes are required.

### 1. Map Ingest Event → TrafficLog

- **event_type**: `"allow"` → `was_blocked = 0`; any other type (e.g. `rate_limit`, `blacklist`, `ddos_*`, `bot_*`, `waf_*`) → `was_blocked = 1`.
- **path** → `endpoint` (or `"/"` if missing).
- **method** → `method` (or `"GET"` if missing).
- **ip** → `ip`.
- **attack_score** → can be stored as `anomaly_score` (float) for blocked requests; optional.
- **bot_score** / **bot_action** → can drive `threat_type` (e.g. `"bot_block"`) for blocked requests; optional.
- Fields not provided by gateway: use defaults — `status_code=0`, `response_size=0`, `processing_time_ms=0`, `user_agent=None`, `query_string=None`, `request_body=None`, `country_code=None`. These are acceptable for chart aggregation (request count + blocked vs allowed).

### 2. Backend Changes


| Location                                  | Change                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**backend/routes/events.py**`            | In `ingest_events`: after creating each `SecurityEvent`, call a small helper (or `TrafficService.create_traffic_log`) to create a `TrafficLog` from the same event. Use the mapping above. Reuse the same `db` session; commit once per batch (after the loop) to keep ingest performant.                                 |
| `**backend/services/traffic_service.py**` | Optional: add a method e.g. `create_traffic_log_from_ingest_event(event_type, ip, method, path, ...)` that accepts ingest-shaped args and uses defaults for missing fields, so the route stays thin.                                                                                                                      |
| **WebSocket broadcast**                   | After creating a `TrafficLog` from ingest, call the same broadcast used by the WAF middleware (e.g. `broadcast_update_sync("traffic", traffic_log.to_dict())`) so the realtime chart updates when gateway events are ingested. Ensure the broadcast helper is importable from the events route (or from a shared module). |


### 3. Optional: Feature Flag

- Add a config flag (e.g. `EVENTS_INGEST_WRITE_TRAFFIC_LOG=true`) so you can disable writing to `traffic_logs` from ingest if you ever need to (e.g. to avoid double-counting in a hybrid setup). Default: `true`.

### 4. Threat Type Mapping (Optional)

- For blocked events, set `threat_type` from `event_type` when possible, e.g. `rate_limit` → `"rate_limit"`, `waf_block` → `"waf"`, `ddos_burst` → `"ddos"`, `bot_block` → `"bot"`, etc. This improves Top Threat Types and analytics without changing APIs.

---

## Alternative: Chart Reads from SecurityEvent

If you prefer **not** to duplicate data into `traffic_logs`:

- **Option B**: Extend the charts/traffic pipeline to support a **secondary source**: when `traffic_logs` has no (or insufficient) data in the requested range, aggregate from `**security_events**` by time bucket, counting events as requests and deriving blocked vs allowed from `event_type`. This would require:
  - Backend: charts service (and optionally `GET /api/traffic/recent`) to query `SecurityEvent` when TrafficLog is empty or as a fallback.
  - Consistent semantics: one event = one request; `event_type == 'allow'` = allowed, else blocked.

This is more invasive (two data sources, fallback logic, possible differences in schema) and still leaves "recent traffic" list potentially empty unless you also expose security_events as "traffic". The **recommended approach** remains writing TrafficLog from ingest.

---

## Implementation Checklist

- **1. TrafficLog from ingest**  
In `backend/routes/events.py` (`ingest_events`), for each event in `body.events`:
  - Keep existing SecurityEvent creation.
  - Map event to TrafficLog fields (event_type → was_blocked; path → endpoint; method; ip; defaults for status_code, response_size, processing_time_ms, etc.).
  - Call `TrafficService(db).create_traffic_log(...)` (or a dedicated helper that uses defaults). Prefer a single commit after the loop for the whole batch.
- **2. Reuse session / batch commit**  
Create all SecurityEvent and TrafficLog rows in the same request; commit once (or per batch) to avoid many small transactions.
- **3. WebSocket broadcast**  
After each new TrafficLog from ingest, call the same traffic broadcast used by the WAF middleware so the realtime chart updates.
- **4. (Optional) Config flag**  
Add `EVENTS_INGEST_WRITE_TRAFFIC_LOG` (default true) and skip creating TrafficLog when false.
- **5. (Optional) threat_type**  
Set `threat_type` on TrafficLog from `event_type` for blocked events.
- **6. Verify**  
  - Start gateway + backend with `BACKEND_EVENTS_URL` set to backend ingest.
  - Send traffic through the gateway (browse upstream app or run a small script).
  - Open dashboard → Request Volume & Threats (e.g. Last 24 hours); confirm data appears and updates.

---

## Files to Touch


| File                                  | Purpose                                                       |
| ------------------------------------- | ------------------------------------------------------------- |
| `backend/routes/events.py`            | Create TrafficLog per ingested event; optional broadcast.     |
| `backend/services/traffic_service.py` | Optional: `create_traffic_log_from_ingest_event(...)` helper. |
| `backend/config.py`                   | Optional: `EVENTS_INGEST_WRITE_TRAFFIC_LOG` flag.             |


No frontend or chart API changes required; the chart already reads from `traffic_logs` and will see the new rows.

---

## Summary

- **Root cause**: Chart reads from `traffic_logs`; only backend middleware wrote there; gateway only wrote to `security_events`.
- **Fix**: On `POST /api/events/ingest`, create one **TrafficLog** per event (with event_type → was_blocked, path → endpoint, defaults for missing fields) and broadcast traffic updates so the Request Volume & Threats chart and realtime views show gateway traffic.

