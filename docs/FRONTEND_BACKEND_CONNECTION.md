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

The frontend runs on **http://localhost:3000**. When `NEXT_PUBLIC_API_URL` is not set, the frontend proxies API requests to **http://localhost:3001** via Next.js rewrites.

### 3. Verify Connection

- Open http://localhost:3000/dashboard
- Metrics, charts, and data should load when the backend is reachable
- If you see "No data available" or empty charts, ensure the backend is running on port 3001

## Configuration

### Environment Variables (frontend/.env.local)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | (empty) | Leave empty for proxy mode. Set to `http://localhost:3001` for direct API calls. |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:3001/ws/` | WebSocket URL for real-time updates. |

### Proxy Mode (default)

When `NEXT_PUBLIC_API_URL` is **not set**, the frontend uses Next.js rewrites to proxy `/api/*` to `http://localhost:3001/api/*`. This avoids CORS issues in local development.

### Direct API Mode

Set `NEXT_PUBLIC_API_URL=http://localhost:3001` when you want the frontend to call the backend directly. The backend must allow CORS from `http://localhost:3000` (already configured).

## Docker Deployment

When using Docker Compose (see [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)), both services run in containers. Set `NEXT_PUBLIC_API_URL` to the backend service URL that the browser can reach (e.g. `http://localhost:3001` if the backend port is exposed).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Empty charts, no metrics | Backend not running | Start the backend on port 3001 |
| Network errors in console | Backend unreachable | Check backend is running; verify port 3001 is not blocked |
| **User API not found** / **Users module not loaded** | Backend missing PyJWT or users route not registered | Run `pip install 'PyJWT>=2.8.0'` in the backend environment, then restart the backend. With Docker, rebuild: `docker compose build backend && docker compose up -d backend`. |
| User Management shows 404 | Frontend can’t reach backend `/api/users` | Ensure backend is running. If using Docker, set `NEXT_PUBLIC_API_URL=http://localhost:3001` (or the URL the browser uses to reach the API). |
| Buttons not navigating | Usually fixed | View All Alerts, View Activity Log, Details now link to their pages |
