---
name: remove-mock-data-realtime-backend
overview: Replace all mock data with real-time backend API calls since the team is in phase 5 of backend development
todos:
  - id: create-api-service
    content: Create lib/api.ts with centralized API functions and TypeScript interfaces for all endpoints
    status: completed
  - id: replace-realtime-hook
    content: Replace useRealTimeData hook with actual API calls and WebSocket support
    status: completed
  - id: update-dashboard-components
    content: Update alerts-section.tsx, activity-feed.tsx, charts-section.tsx, and metrics-overview.tsx to use API calls
    status: completed
  - id: update-traffic-page
    content: Replace hardcoded trafficData in app/traffic/page.tsx with API calls
    status: completed
  - id: update-performance-page
    content: Replace hardcoded performanceData in app/performance/page.tsx with API calls
    status: completed
  - id: update-analytics-page
    content: Replace hardcoded analyticsData in app/analytics/page.tsx with API calls
    status: completed
  - id: update-threats-page
    content: Replace hardcoded threats in app/threats/page.tsx with API calls
    status: completed
  - id: update-security-page
    content: Replace hardcoded securityChecks in app/security/page.tsx with API calls
    status: completed
  - id: implement-websockets
    content: Implement WebSocket connection for real-time data updates across components
    status: completed
  - id: add-loading-states
    content: Add proper loading states, error boundaries, and empty state handling throughout the app
    status: completed
isProject: false
---

## Overview

The dashboard currently uses extensive mock/simulated data across all components. Since the team is in phase 5 of backend development, we need to replace all mock data with real-time API calls to the backend.

## Current Mock Data Locations

### Core Components

1. **`use-real-time-data.ts`** - Simulated metrics with setInterval updates
2. **`alerts-section.tsx`** - Hardcoded alerts array
3. **`activity-feed.tsx`** - Hardcoded activities array  
4. **`charts-section.tsx`** - Multiple hardcoded data arrays (requestData, threatData, performanceData)
5. **`metrics-overview.tsx`** - Uses simulated data from useRealTimeData hook

### Page Components

6. **`app/traffic/page.tsx`** - Hardcoded trafficData array
7. **`app/performance/page.tsx`** - Hardcoded performanceData array
8. **`app/analytics/page.tsx`** - Hardcoded analyticsData array
9. **`app/threats/page.tsx`** - Hardcoded threats array
10. **`app/security/page.tsx`** - Hardcoded securityChecks array

## Implementation Plan

### 1. Create API Service Layer

- Create `lib/api.ts` with centralized API functions
- Implement proper error handling and loading states
- Add WebSocket/real-time connection support
- Define TypeScript interfaces for all API responses

### 2. Replace useRealTimeData Hook

- Replace simulated data with actual API calls
- Implement WebSocket connection for real-time updates
- Add reconnection logic and error states
- Update metrics fetching to use backend endpoints

### 3. Update Dashboard Components

- Modify all components to fetch data from APIs instead of using hardcoded arrays
- Implement proper loading states and error boundaries
- Add real-time data subscriptions where appropriate
- Update component interfaces to handle API response types

### 4. Update Page Components

- Replace all hardcoded data in page components with API calls
- Implement proper data fetching patterns
- Add loading skeletons and error states
- Ensure all components handle empty states gracefully

### 5. API Endpoints Structure

Based on phase 5 backend, assume these endpoints exist:

```
/api/metrics/realtime - Real-time metrics
/api/alerts - Active alerts
/api/activities - Recent activities
/api/charts/requests - Request volume data
/api/charts/threats - Threat distribution data
/api/charts/performance - Performance metrics
/api/traffic/recent - Recent traffic logs
/api/threats/recent - Recent threats
/api/security/checks - Security compliance checks
/api/analytics/overview - Analytics data
```

### 6. Real-time Updates

- Implement WebSocket connection for live data updates
- Add fallback to polling for components that don't require instant updates
- Handle connection states (connecting, connected, disconnected, reconnecting)

## Expected Outcomes

- All dashboard components will display real data from the backend
- Real-time updates will work through WebSocket connections
- Proper loading states and error handling throughout the app
- No more mock or simulated data in the codebase
- Dashboard becomes fully functional with live backend data