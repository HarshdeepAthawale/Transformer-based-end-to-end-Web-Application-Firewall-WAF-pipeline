# Frontend–Backend Connection Guide

The frontend dashboard requires the backend API server to be running for full functionality (metrics, charts, alerts, traffic, threats, etc.).

## Quick Start

### 1. Start the Backend

```bash
# From project root
cd backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 3001 --reload
```

Or from project root:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 3001 --reload
```

### 2. Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend runs on **http://localhost:3000**. All API requests (including the AI Copilot) go through the Next.js server via **proxy**: the browser calls `http://localhost:3000/api/*`, and Next.js rewrites those to the backend (default **http://localhost:3001**). The backend only needs to be reachable from the machine running the Next.js server, not directly from the browser.

### 3. Verify Connection

- Open http://localhost:3000/dashboard
- Metrics, charts, and data should load when the backend is reachable
- Open the Copilot page; queries should work as long as the backend is running
- If you see "No data available", empty charts, or "Failed to connect to the AI agent", ensure the backend is running on port 3001

## Configuration

### Environment Variables

| Variable | Where | Default | Description |
|----------|--------|---------|-------------|
| `BACKEND_URL` | Server (next.config / Node) | `http://localhost:3001` | URL the Next.js server uses to proxy `/api/*` to the backend. In Docker, set to `http://backend:3001`. |
| `NEXT_PUBLIC_WS_URL` | Frontend (browser) | `ws://localhost:3001/ws/` | WebSocket URL for real-time updates (browser connects directly to backend). |

API requests from the frontend **always** use the proxy (relative `/api/*`). The proxy target is controlled by `BACKEND_URL` when running the Next.js server.

## Docker Deployment

When using Docker Compose (see [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)), set `BACKEND_URL=http://backend:3001` so the frontend container can proxy API requests to the backend container. The browser only talks to the frontend; the frontend server proxies to the backend.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty charts, no metrics | Backend not running | Start the backend on port 3001 |
| Network errors in console | Backend unreachable | Check backend is running; verify port 3001 is not blocked |
| **Failed to connect to the AI agent** | Backend not running or not reachable from Next server | Start the backend on port 3001. Ensure `BACKEND_URL` (or default `http://localhost:3001`) is reachable from the process running `npm run dev` (or the frontend container). |
| **User API not found** / **Users module not loaded** | Backend missing PyJWT or users route not registered | Run `pip install 'PyJWT>=2.8.0'` in the backend environment, then restart the backend. With Docker, rebuild: `docker compose build backend && docker compose up -d backend`. |
| User Management shows 404 | Frontend can’t reach backend `/api/users` | Ensure backend is running. Proxy is always used; check that `BACKEND_URL` points at the running backend. |
| Buttons not navigating | Usually fixed | View All Alerts, View Activity Log, Details now link to their pages |
