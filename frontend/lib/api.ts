// API service layer for WAF Dashboard backend integration

// Base API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'
// WebSocket URL - note: the backend route is /ws/ so the full path is ws://localhost:3001/ws/
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || (typeof window !== 'undefined' 
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:3001/ws/`
  : 'ws://localhost:3001/ws/')

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
  was_blocked?: boolean
  wasBlocked?: boolean
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
  type: 'metrics' | 'alert' | 'activity' | 'threat' | 'traffic' | 'connection' | 'subscribed' | 'unsubscribed'
  data: any
  timestamp?: string
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
    
    // Handle network errors gracefully (backend not running)
    // These are expected when backend is offline, so mark them specially
    const isNetworkError = error instanceof TypeError && 
      (error.message.includes('fetch') || 
       error.message.includes('Failed to fetch') ||
       error.message.includes('NetworkError'))
    
    if (isNetworkError) {
      // Network error - backend likely not running
      // Create a special error that can be handled gracefully
      const networkError = new ApiError(0, 'Backend server not available')
      // Add a flag to identify network errors
      ;(networkError as any).isNetworkError = true
      throw networkError
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

// WAF API
export interface WAFCheckRequest {
  method: string
  path: string
  query_params?: Record<string, any>
  headers?: Record<string, any>
  body?: string
}

export interface WAFCheckResponse {
  anomaly_score: number
  is_anomaly: boolean
  threshold: number
  processing_time_ms: number
  model_version?: string
}

export interface WAFStats {
  service_available: boolean
  total_requests: number
  anomalies_detected: number
  average_processing_time_ms: number
  threshold: number
  model_loaded?: boolean
}

export interface WAFConfig {
  threshold: number
  model_available: boolean
  service_enabled: boolean
}

export interface WAFModelInfo {
  model_loaded: boolean
  model_path: string | null
  vocab_path: string | null
  model_exists: boolean
  vocab_exists: boolean
  threshold: number
  version: string
}

// IP Management API
export interface IPEntry {
  id: number
  ip: string
  list_type: 'blacklist' | 'whitelist'
  reason?: string
  source: string
  created_at: string
  expires_at?: string
  is_active: boolean
}

export interface IPReputation {
  ip: string
  reputation_score: number
  threat_level: 'low' | 'medium' | 'high' | 'critical'
  is_blacklisted: boolean
  is_whitelisted: boolean
  last_seen: string
  total_requests: number
  blocked_requests: number
}

export const ipApi = {
  getBlacklist: (limit: number = 100): Promise<ApiResponse<IPEntry[]>> =>
    apiRequest(`/api/ip/blacklist?limit=${limit}`),

  getWhitelist: (limit: number = 100): Promise<ApiResponse<IPEntry[]>> =>
    apiRequest(`/api/ip/whitelist?limit=${limit}`),

  addToBlacklist: (ip: string, reason?: string, durationHours?: number, source: string = 'manual'): Promise<ApiResponse<IPEntry>> =>
    apiRequest('/api/ip/blacklist', {
      method: 'POST',
      body: JSON.stringify({ ip, reason, duration_hours: durationHours, source }),
    }),

  addToWhitelist: (ip: string, reason?: string): Promise<ApiResponse<IPEntry>> =>
    apiRequest('/api/ip/whitelist', {
      method: 'POST',
      body: JSON.stringify({ ip, reason }),
    }),

  removeFromList: (ip: string, listType: 'blacklist' | 'whitelist'): Promise<ApiResponse<void>> =>
    apiRequest(`/api/ip/${ip}?list_type=${listType}`, { method: 'DELETE' }),

  getReputation: (ip: string): Promise<ApiResponse<IPReputation>> =>
    apiRequest(`/api/ip/${ip}/reputation`),
}

// Geo Rules API
export interface GeoRule {
  id: number
  rule_type: 'allow' | 'deny'
  country_code: string
  country_name: string
  priority: number
  exception_ips?: string[]
  reason?: string
  is_active: boolean
  created_at: string
}

export interface GeoStats {
  country_code: string
  country_name: string
  total_requests: number
  blocked_requests: number
  threat_count: number
}

export const geoApi = {
  getRules: (activeOnly: boolean = true): Promise<ApiResponse<GeoRule[]>> =>
    apiRequest(`/api/geo/rules?active_only=${activeOnly}`),

  createRule: (rule: {
    rule_type: 'allow' | 'deny'
    country_code: string
    country_name: string
    priority?: number
    exception_ips?: string[]
    reason?: string
  }): Promise<ApiResponse<GeoRule>> =>
    apiRequest('/api/geo/rules', {
      method: 'POST',
      body: JSON.stringify(rule),
    }),

  getStats: (range: string = '24h'): Promise<ApiResponse<GeoStats[]>> =>
    apiRequest(`/api/geo/stats?range=${range}`),
}

// Bot Detection API
export interface BotSignature {
  id: number
  user_agent_pattern: string
  name: string
  category: 'search_engine' | 'scraper' | 'malicious' | 'monitoring' | 'unknown'
  action: 'block' | 'allow' | 'challenge' | 'monitor'
  is_whitelisted: boolean
  is_active: boolean
  created_at: string
}

export const botApi = {
  getSignatures: (activeOnly: boolean = true): Promise<ApiResponse<BotSignature[]>> =>
    apiRequest(`/api/bots/signatures?active_only=${activeOnly}`),

  addSignature: (signature: {
    user_agent_pattern: string
    name: string
    category: string
    action?: string
    is_whitelisted?: boolean
  }): Promise<ApiResponse<BotSignature>> =>
    apiRequest('/api/bots/signatures', {
      method: 'POST',
      body: JSON.stringify(signature),
    }),
}

// Threat Intelligence API
export interface ThreatIntel {
  id: number
  threat_type: 'ip' | 'domain' | 'signature'
  value: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  category: string
  source: string
  description?: string
  expires_at?: string
  is_active: boolean
  created_at: string
}

export interface ThreatCheckResult {
  is_threat: boolean
  threat_level: 'low' | 'medium' | 'high' | 'critical'
  matches: ThreatIntel[]
}

export const threatIntelApi = {
  getFeeds: (threatType?: string, activeOnly: boolean = true, limit: number = 100): Promise<ApiResponse<ThreatIntel[]>> =>
    apiRequest(`/api/threat-intel/feeds?${threatType ? `threat_type=${threatType}&` : ''}active_only=${activeOnly}&limit=${limit}`),

  addThreat: (threat: {
    threat_type: string
    value: string
    severity: string
    category: string
    source: string
    description?: string
    expires_at?: string
  }): Promise<ApiResponse<ThreatIntel>> =>
    apiRequest('/api/threat-intel/feeds', {
      method: 'POST',
      body: JSON.stringify(threat),
    }),

  checkIP: (ip: string): Promise<ApiResponse<ThreatCheckResult>> =>
    apiRequest(`/api/threat-intel/check/${ip}`),
}

// Security Rules API
export interface SecurityRule {
  id: number
  name: string
  rule_type: string
  pattern: string
  applies_to: string
  action: 'block' | 'log' | 'alert' | 'redirect' | 'challenge'
  priority: 'high' | 'medium' | 'low'
  description?: string
  owasp_category?: string
  is_active: boolean
  created_at: string
}

export const securityRulesApi = {
  getRules: (activeOnly: boolean = true): Promise<ApiResponse<SecurityRule[]>> =>
    apiRequest(`/api/rules?active_only=${activeOnly}`),

  createRule: (rule: {
    name: string
    rule_type: string
    pattern: string
    applies_to?: string
    action?: string
    priority?: string
    description?: string
    owasp_category?: string
  }): Promise<ApiResponse<SecurityRule>> =>
    apiRequest('/api/rules', {
      method: 'POST',
      body: JSON.stringify(rule),
    }),

  getOWASPRules: (): Promise<ApiResponse<SecurityRule[]>> =>
    apiRequest('/api/rules/owasp'),
}

// Users API
export interface User {
  id: number
  username: string
  email: string
  role: 'admin' | 'operator' | 'viewer'
  full_name?: string
  is_active: boolean
  last_login?: string
  created_at: string
  created_by?: string
}

export const usersApi = {
  getUsers: (): Promise<ApiResponse<User[]>> =>
    apiRequest('/api/users'),

  getCurrentUser: (): Promise<ApiResponse<User>> =>
    apiRequest('/api/users/me'),

  createUser: (user: {
    username: string
    email: string
    password: string
    role?: string
    full_name?: string
  }): Promise<ApiResponse<User>> =>
    apiRequest('/api/users', {
      method: 'POST',
      body: JSON.stringify(user),
    }),

  login: (username: string, password: string): Promise<ApiResponse<{ token: string; user: User }>> =>
    apiRequest('/api/users/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
}

// Audit Logs API
export interface AuditLog {
  id: number
  timestamp: string
  user_id?: number
  username?: string
  ip_address?: string
  action: 'create' | 'update' | 'delete' | 'view' | 'login' | 'logout' | 'block' | 'unblock' | 'config_change' | 'rule_change'
  resource_type: string
  resource_id?: number
  description: string
  details?: string
  success: boolean
  error_message?: string
}

export const auditApi = {
  getLogs: (params: {
    limit?: number
    action?: string
    resource_type?: string
    start_time?: string
  } = {}): Promise<ApiResponse<AuditLog[]>> => {
    const query = new URLSearchParams()
    if (params.limit) query.append('limit', params.limit.toString())
    if (params.action) query.append('action', params.action)
    if (params.resource_type) query.append('resource_type', params.resource_type)
    if (params.start_time) query.append('start_time', params.start_time)
    return apiRequest(`/api/audit/logs?${query.toString()}`)
  },

  getLog: (logId: number): Promise<ApiResponse<AuditLog>> =>
    apiRequest(`/api/audit/logs/${logId}`),
}

export const wafApi = {
  check: (request: WAFCheckRequest): Promise<ApiResponse<WAFCheckResponse>> =>
    apiRequest('/api/waf/check', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  checkBatch: (requests: WAFCheckRequest[]): Promise<ApiResponse<WAFCheckResponse[]>> =>
    apiRequest('/api/waf/check/batch', {
      method: 'POST',
      body: JSON.stringify(requests),
    }),

  getStats: (): Promise<ApiResponse<WAFStats>> =>
    apiRequest('/api/waf/stats'),

  getConfig: (): Promise<ApiResponse<WAFConfig>> =>
    apiRequest('/api/waf/config'),

  updateConfig: (threshold: number): Promise<ApiResponse<WAFConfig>> =>
    apiRequest('/api/waf/config', {
      method: 'PUT',
      body: JSON.stringify({ threshold }),
    }),

  getModelInfo: (): Promise<ApiResponse<WAFModelInfo>> =>
    apiRequest('/api/waf/model-info'),
}

// WebSocket connection management
export class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private reconnectDelay = 1000
  private subscribers: Map<string, Set<(data: any) => void>> = new Map()
  private messageQueue: WebSocketMessage[] = []
  private pingInterval: NodeJS.Timeout | null = null
  private subscribedTypes: Set<string> = new Set()
  private connectionState: 'connecting' | 'connected' | 'disconnected' = 'disconnected'

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Only connect in browser environment
      if (typeof window === 'undefined') {
        console.debug('[WebSocket] Skipping connection in server environment')
        resolve()
        return
      }

      try {
        if (this.ws?.readyState === WebSocket.OPEN) {
          resolve()
          return
        }

        // Clean up existing connection if any
        if (this.ws) {
          try {
            this.ws.close()
          } catch (e) {
            // Ignore cleanup errors
          }
          this.ws = null
        }

        this.connectionState = 'connecting'
        
        // Validate WebSocket URL
        if (!WS_BASE_URL || WS_BASE_URL === '') {
          console.error('[WebSocket] Invalid WebSocket URL')
          this.connectionState = 'disconnected'
          reject(new Error('WebSocket URL not configured'))
          return
        }

        console.log(`[WebSocket] Attempting to connect to ${WS_BASE_URL}`)
        this.ws = new WebSocket(WS_BASE_URL)
        
        // Set connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.warn('[WebSocket] Connection timeout')
            if (this.ws) {
              this.ws.close()
            }
            this.connectionState = 'disconnected'
            this.attemptReconnect()
          }
        }, 10000) // 10 second timeout

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout)
          console.log('[WebSocket] Connected to backend')
          this.connectionState = 'connected'
          this.reconnectAttempts = 0
          
          // Send queued messages
          this.flushMessageQueue()
          
          // Subscribe to previously subscribed types
          if (this.subscribedTypes.size > 0) {
            this.send({
              type: 'subscribe',
              message_types: Array.from(this.subscribedTypes)
            })
          }
          
          // Start ping interval
          this.startPingInterval()
          
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            
            // Handle connection status messages
            if (message.type === 'connection') {
              console.log('[WebSocket] Connection status:', message.data)
              return
            }
            
            if (message.type === 'subscribed' || message.type === 'unsubscribed') {
              console.log('[WebSocket] Subscription update:', message.data)
              return
            }
            
            // Notify all subscribers for this message type
            this.notifySubscribers(message.type, message.data)
          } catch (error) {
            console.error('[WebSocket] Failed to parse message:', error)
          }
        }

        this.ws.onclose = (event) => {
          console.log(`[WebSocket] Connection closed (code: ${event.code}, reason: ${event.reason || 'none'})`)
          this.connectionState = 'disconnected'
          this.stopPingInterval()
          
          // Only attempt reconnect if it wasn't a manual close
          if (event.code !== 1000) {
            this.attemptReconnect()
          }
        }

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout)
          // WebSocket errors are common when server is not running
          // Log as debug to avoid console spam
          console.debug('[WebSocket] Connection error (backend may not be running):', error)
          this.connectionState = 'disconnected'
          // Don't reject here - let onclose handle reconnection
          // This allows the connection to attempt reconnection automatically
        }

      } catch (error) {
        console.error('[WebSocket] Failed to create WebSocket:', error)
        this.connectionState = 'disconnected'
        // Don't reject immediately - allow reconnection attempts
        setTimeout(() => {
          this.attemptReconnect()
        }, 1000)
      }
    })
  }

  private send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      // Queue message for when connection is established
      this.messageQueue.push(data as WebSocketMessage)
    }
  }

  private flushMessageQueue() {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift()
      if (message) {
        this.ws.send(JSON.stringify(message))
      }
    }
  }

  private startPingInterval() {
    this.stopPingInterval()
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' })
      }
    }, 30000) // Ping every 30 seconds
  }

  private stopPingInterval() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval)
      this.pingInterval = null
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn(`[WebSocket] Max reconnection attempts (${this.maxReconnectAttempts}) reached. WebSocket will remain disconnected.`)
      console.warn('[WebSocket] Please check if the backend server is running on', WS_BASE_URL)
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    )

    console.log(`[WebSocket] Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`)

    setTimeout(() => {
      if (this.connectionState === 'disconnected') {
        this.connect().catch((err) => {
          console.debug('[WebSocket] Reconnection attempt failed:', err)
          // Will retry on next attempt
        })
      }
    }, delay)
  }

  subscribe(type: string, callback: (data: any) => void) {
    if (!this.subscribers.has(type)) {
      this.subscribers.set(type, new Set())
    }
    this.subscribers.get(type)!.add(callback)
    
    // Subscribe on server if connected
    if (!this.subscribedTypes.has(type)) {
      this.subscribedTypes.add(type)
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({
          type: 'subscribe',
          message_types: [type]
        })
      }
    }
  }

  unsubscribe(type: string) {
    const callbacks = this.subscribers.get(type)
    if (callbacks) {
      callbacks.clear()
    }
    
    // Unsubscribe on server if connected
    if (this.subscribedTypes.has(type)) {
      this.subscribedTypes.delete(type)
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({
          type: 'unsubscribe',
          message_types: [type]
        })
      }
    }
  }

  private notifySubscribers(type: string, data: any) {
    const callbacks = this.subscribers.get(type)
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`[WebSocket] Error in subscriber callback for ${type}:`, error)
        }
      })
    }
  }

  disconnect() {
    this.stopPingInterval()
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.connectionState = 'disconnected'
    this.subscribers.clear()
    this.messageQueue = []
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  get connectionStatus(): 'connecting' | 'connected' | 'disconnected' {
    return this.connectionState
  }
}

// Export singleton WebSocket manager
export const wsManager = new WebSocketManager()