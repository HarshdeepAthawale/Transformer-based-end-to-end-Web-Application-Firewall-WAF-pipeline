// API service layer for WAF Dashboard backend integration

// Base API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws/'

// TypeScript interfaces for API responses
export interface AlertData {
  id: number
  type: 'critical' | 'warning' | 'info'
  severity: 'high' | 'medium' | 'low'
  title: string
  description: string
  time: string
  icon: string
  source: string
  actions: string[]
  timestamp: string
}

export interface ActivityData {
  id: number
  type: 'blocked' | 'allowed'
  title: string
  details: string
  time: string
  ip?: string
  endpoint?: string
  timestamp: string
}

export interface RealTimeMetrics {
  requests: number
  blocked: number
  attackRate: number
  responseTime: number
  threatsPerMinute: number
  uptime: number
  activeConnections: number
  timestamp: string
}

export interface ChartDataPoint {
  time: string
  requests?: number
  blocked?: number
  allowed?: number
  sql?: number
  xss?: number
  ddos?: number
  other?: number
  latency?: number
  cpu?: number
  memory?: number
  date?: string
}

export interface TrafficData {
  ip: string
  method: string
  endpoint: string
  status: number
  size: string
  time: string
  timestamp: string
  userAgent?: string
}

export interface ThreatData {
  id: number
  type: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  source: string
  endpoint: string
  blocked: boolean
  time: string
  timestamp: string
  details?: string
}

export interface SecurityCheck {
  id: number
  name: string
  status: 'pass' | 'fail' | 'warning'
  message: string
  lastChecked: string
  details?: string
}

export interface ApiResponse<T> {
  success: boolean
  data: T
  message?: string
  timestamp: string
}

export interface WebSocketMessage {
  type: 'metrics' | 'alert' | 'activity' | 'threat' | 'traffic'
  data: any
  timestamp: string
}

// API Error class
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

// Generic fetch wrapper with error handling
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new ApiError(response.status, `API request failed: ${response.statusText}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// Metrics API
export const metricsApi = {
  getRealtime: (): Promise<ApiResponse<RealTimeMetrics>> =>
    apiRequest('/api/metrics/realtime'),

  getHistorical: (timeRange: string): Promise<ApiResponse<RealTimeMetrics[]>> =>
    apiRequest(`/api/metrics/historical?range=${timeRange}`),
}

// Alerts API
export const alertsApi = {
  getActive: (): Promise<ApiResponse<AlertData[]>> =>
    apiRequest('/api/alerts/active'),

  getHistorical: (timeRange: string): Promise<ApiResponse<AlertData[]>> =>
    apiRequest(`/api/alerts/history?range=${timeRange}`),

  dismiss: (alertId: number): Promise<ApiResponse<void>> =>
    apiRequest(`/api/alerts/${alertId}/dismiss`, { method: 'POST' }),

  acknowledge: (alertId: number): Promise<ApiResponse<void>> =>
    apiRequest(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' }),
}

// Activities API
export const activitiesApi = {
  getRecent: (limit: number = 10): Promise<ApiResponse<ActivityData[]>> =>
    apiRequest(`/api/activities/recent?limit=${limit}`),

  getByTimeRange: (timeRange: string): Promise<ApiResponse<ActivityData[]>> =>
    apiRequest(`/api/activities?range=${timeRange}`),
}

// Charts API
export const chartsApi = {
  getRequests: (timeRange: string): Promise<ApiResponse<ChartDataPoint[]>> =>
    apiRequest(`/api/charts/requests?range=${timeRange}`),

  getThreats: (timeRange: string): Promise<ApiResponse<ChartDataPoint[]>> =>
    apiRequest(`/api/charts/threats?range=${timeRange}`),

  getPerformance: (timeRange: string): Promise<ApiResponse<ChartDataPoint[]>> =>
    apiRequest(`/api/charts/performance?range=${timeRange}`),
}

// Traffic API
export const trafficApi = {
  getRecent: (limit: number = 50): Promise<ApiResponse<TrafficData[]>> =>
    apiRequest(`/api/traffic/recent?limit=${limit}`),

  getByTimeRange: (timeRange: string): Promise<ApiResponse<TrafficData[]>> =>
    apiRequest(`/api/traffic?range=${timeRange}`),

  getByEndpoint: (endpoint: string, timeRange: string): Promise<ApiResponse<TrafficData[]>> =>
    apiRequest(`/api/traffic/endpoint/${encodeURIComponent(endpoint)}?range=${timeRange}`),
}

// Threats API
export const threatsApi = {
  getRecent: (limit: number = 20): Promise<ApiResponse<ThreatData[]>> =>
    apiRequest(`/api/threats/recent?limit=${limit}`),

  getByTimeRange: (timeRange: string): Promise<ApiResponse<ThreatData[]>> =>
    apiRequest(`/api/threats?range=${timeRange}`),

  getByType: (type: string, timeRange: string): Promise<ApiResponse<ThreatData[]>> =>
    apiRequest(`/api/threats/type/${type}?range=${timeRange}`),

  getStats: (timeRange: string): Promise<ApiResponse<Record<string, number>>> =>
    apiRequest(`/api/threats/stats?range=${timeRange}`),
}

// Security API
export const securityApi = {
  getChecks: (): Promise<ApiResponse<SecurityCheck[]>> =>
    apiRequest('/api/security/checks'),

  runCheck: (checkId: number): Promise<ApiResponse<SecurityCheck>> =>
    apiRequest(`/api/security/checks/${checkId}/run`, { method: 'POST' }),

  getComplianceScore: (): Promise<ApiResponse<{ score: number; total: number }>> =>
    apiRequest('/api/security/compliance-score'),
}

// Analytics API
export const analyticsApi = {
  getOverview: (timeRange: string): Promise<ApiResponse<ChartDataPoint[]>> =>
    apiRequest(`/api/analytics/overview?range=${timeRange}`),

  getTrends: (metric: string, timeRange: string): Promise<ApiResponse<ChartDataPoint[]>> =>
    apiRequest(`/api/analytics/trends/${metric}?range=${timeRange}`),

  getSummary: (timeRange: string): Promise<ApiResponse<Record<string, any>>> =>
    apiRequest(`/api/analytics/summary?range=${timeRange}`),
}

// WebSocket connection management
export class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private subscribers: Map<string, (data: any) => void> = new Map()

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(WS_BASE_URL)

        this.ws.onopen = () => {
          console.log('[WebSocket] Connected to backend')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.notifySubscribers(message.type, message.data)
          } catch (error) {
            console.error('[WebSocket] Failed to parse message:', error)
          }
        }

        this.ws.onclose = () => {
          console.log('[WebSocket] Connection closed')
          this.attemptReconnect()
        }

        this.ws.onerror = (error) => {
          console.error('[WebSocket] Connection error:', error)
          reject(error)
        }

      } catch (error) {
        reject(error)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    console.log(`[WebSocket] Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`)

    setTimeout(() => {
      this.connect().catch(() => {
        // Reconnection failed, will retry
      })
    }, delay)
  }

  subscribe(type: string, callback: (data: any) => void) {
    this.subscribers.set(type, callback)
  }

  unsubscribe(type: string) {
    this.subscribers.delete(type)
  }

  private notifySubscribers(type: string, data: any) {
    const callback = this.subscribers.get(type)
    if (callback) {
      callback(data)
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Export singleton WebSocket manager
export const wsManager = new WebSocketManager()