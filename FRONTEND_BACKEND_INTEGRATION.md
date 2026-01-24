# Frontend-Backend Integration Guide

## Overview

The frontend has been fully updated to integrate with the backend API server. All components now use real data from the backend with no mock or hardcoded values.

## Configuration

### Environment Variables

Create a `.env.local` file in the `frontend/` directory:

```bash
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_WS_URL=ws://localhost:3001/ws/
```

### API Base URLs

- **REST API**: `http://localhost:3001` (default)
- **WebSocket**: `ws://localhost:3001/ws/` (default)

## Component Updates

### ✅ Updated Components

1. **Metrics Overview** (`components/metrics-overview.tsx`)
   - Removed hardcoded initial values
   - Uses `useRealTimeData` hook for live metrics
   - Receives real-time updates via WebSocket

2. **Charts Section** (`components/charts-section.tsx`)
   - Fetches data from `/api/charts/*` endpoints
   - Supports time range queries
   - Real-time chart updates via WebSocket

3. **Alerts Section** (`components/alerts-section.tsx`)
   - Fetches active alerts from `/api/alerts/active`
   - Real-time alert updates via WebSocket
   - Supports alert dismissal and acknowledgment

4. **Activity Feed** (`components/activity-feed.tsx`)
   - Fetches recent activities from `/api/activities/recent`
   - Real-time activity updates via WebSocket

5. **Threats Page** (`app/threats/page.tsx`)
   - Fetches threats from `/api/threats/recent`
   - Real-time threat updates via WebSocket
   - Threat statistics and filtering

6. **Traffic Page** (`app/traffic/page.tsx`)
   - Fetches traffic logs from `/api/traffic/recent`
   - Real-time traffic updates via WebSocket
   - Traffic filtering and search

7. **Analytics Page** (`app/analytics/page.tsx`)
   - Fetches analytics from `/api/analytics/overview`
   - Time range support
   - Chart visualizations

8. **Performance Page** (`app/performance/page.tsx`)
   - Fetches performance data from `/api/charts/performance`
   - CPU, memory, and latency metrics
   - Performance charts

9. **Security Page** (`app/security/page.tsx`)
   - Fetches security checks from `/api/security/checks`
   - Compliance score from `/api/security/compliance-score`
   - Security health monitoring

## API Integration

### API Client (`lib/api.ts`)

All API endpoints are defined in `lib/api.ts`:

- `metricsApi` - Real-time and historical metrics
- `alertsApi` - Alert management
- `activitiesApi` - Activity feed
- `chartsApi` - Chart data
- `trafficApi` - Traffic logs
- `threatsApi` - Threat detection
- `securityApi` - Security checks
- `analyticsApi` - Analytics and trends

### WebSocket Manager

The `WebSocketManager` class handles:
- Connection management
- Automatic reconnection
- Message subscription/unsubscription
- Real-time updates for:
  - Metrics
  - Alerts
  - Activities
  - Threats
  - Traffic

### Real-time Data Hook (`hooks/use-real-time-data.ts`)

The `useRealTimeData` hook provides:
- Real-time metrics
- Connection status
- Automatic WebSocket connection
- Fallback polling if WebSocket fails
- Data freshness indicators

## Data Flow

1. **Initial Load**: Components fetch data from REST API on mount
2. **Real-time Updates**: WebSocket provides live updates
3. **Fallback**: If WebSocket fails, components poll REST API every 30 seconds
4. **Error Handling**: All components handle errors gracefully with loading and error states

## Response Format

All API responses follow this format:

```typescript
{
  success: boolean
  data: T
  message?: string
  timestamp: string
}
```

## WebSocket Messages

WebSocket messages follow this format:

```typescript
{
  type: 'metrics' | 'alert' | 'activity' | 'threat' | 'traffic'
  data: any
  timestamp: string
}
```

## Error Handling

All components include:
- Loading states while fetching data
- Error states for failed requests
- Empty states when no data is available
- Retry mechanisms via WebSocket reconnection

## Testing the Integration

1. **Start the backend**:
```bash
python scripts/start_api_server.py
```

2. **Start the frontend**:
```bash
cd frontend
npm run dev
```

3. **Verify**:
   - Dashboard shows real-time metrics
   - Charts display data from backend
   - Alerts appear in real-time
   - Activity feed updates live
   - All pages load data from API

## Troubleshooting

### WebSocket Connection Issues

- Check that backend is running on port 3001
- Verify WebSocket URL is `ws://localhost:3001/ws/`
- Check browser console for connection errors
- Ensure CORS is configured correctly in backend

### API Request Failures

- Verify backend is running
- Check API base URL in `.env.local`
- Check browser network tab for failed requests
- Verify backend logs for errors

### No Data Displayed

- Check if database has data
- Verify API endpoints return data
- Check browser console for errors
- Ensure background workers are running

## Next Steps

- Add authentication/authorization
- Implement API key management
- Add request/response interceptors
- Implement offline mode with caching
- Add data export functionality
