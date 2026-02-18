// API service layer for WAF Dashboard backend integration

// Use relative path so all API requests go through Next.js proxy (/api/* -> backend). Backend URL is
// configured server-side via BACKEND_URL (next.config.mjs). This avoids connection failures when
// the browser cannot reach the backend directly (e.g. Docker, or backend not on same host).
const API_BASE_URL = ''
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
  threatsPerMinute: number
  uptime: number
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
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    }
    // Attach backend JWT when on client so /api/users and other auth routes work
    if (typeof window !== 'undefined') {
      try {
        const { getSession } = await import('next-auth/react')
        const session = await getSession()
        const backendToken = (session?.user as { backendToken?: string } | undefined)?.backendToken
        if (backendToken) {
          headers['Authorization'] = `Bearer ${backendToken}`
        }
      } catch {
        // next-auth not available or not in provider (e.g. tests)
      }
    }
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    })

    if (!response.ok) {
      let message = `API request failed: ${response.statusText}`
      try {
        const errBody = await response.json()
        if (typeof errBody?.detail === 'string') {
          message = errBody.detail
        }
      } catch {
        // Ignore JSON parse errors
      }
      throw new ApiError(response.status, message)
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

  getRateLimit: (timeRange: string): Promise<ApiResponse<{ time: string; count: number }[]>> =>
    apiRequest(`/api/charts/rate-limit?range=${timeRange}`),

  getDdos: (timeRange: string): Promise<ApiResponse<{ time: string; count: number }[]>> =>
    apiRequest(`/api/charts/ddos?range=${timeRange}`),
}

// Events API (rate limit, DDoS)
export interface SecurityEventData {
  id: number
  timestamp: string
  event_type: string
  ip: string
  method?: string
  path?: string
  details?: string
  attack_score?: number
  block_duration_seconds?: number
  bot_score?: number
}

export interface EventsStats {
  rate_limit_count: number
  ddos_count: number
  blacklist_count?: number
  waf_block_count?: number
  avg_attack_score?: number | null
}

export interface DosOverviewData {
  stats: EventsStats
  chart_rate_limit: { time: string; count: number }[]
  chart_ddos: { time: string; count: number }[]
  chart_blacklist?: { time: string; count: number }[]
  recent_rate_limit: SecurityEventData[]
  recent_ddos: SecurityEventData[]
  recent_blacklist?: SecurityEventData[]
}

export const eventsApi = {
  getStats: (range: string = '24h'): Promise<ApiResponse<EventsStats>> =>
    apiRequest(`/api/events/stats?range=${range}`),

  getRateLimitEvents: (range: string = '24h'): Promise<ApiResponse<SecurityEventData[]>> =>
    apiRequest(`/api/events/rate-limit?range=${range}`),

  getDdosEvents: (range: string = '24h'): Promise<ApiResponse<SecurityEventData[]>> =>
    apiRequest(`/api/events/ddos?range=${range}`),

  getDosOverview: (range: string = '24h', limit?: number): Promise<ApiResponse<DosOverviewData>> =>
    apiRequest(`/api/events/dos-overview?range=${range}${limit != null ? `&limit=${limit}` : ''}`),

  getWafEvents: (range: string = '24h', limit?: number): Promise<ApiResponse<SecurityEventData[]>> =>
    apiRequest(`/api/events/waf?range=${range}${limit != null ? `&limit=${limit}` : ''}`),
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

export interface VerifiedBot {
  id: number
  name: string
  user_agent_pattern: string
  source: 'manual' | 'remote'
  synced_at: string | null
  created_at: string
}

export interface BotScoreBand {
  id: number
  min_score: number
  max_score: number
  action: 'allow' | 'challenge' | 'block'
  priority: number
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

  // Bot management (score bands, verified bots)
  getScoreBands: (): Promise<ApiResponse<BotScoreBand[]>> =>
    apiRequest('/api/bot/score-bands'),

  updateScoreBands: (bands: { min_score: number; max_score: number; action: string }[]): Promise<ApiResponse<BotScoreBand[]>> =>
    apiRequest('/api/bot/score-bands', {
      method: 'PUT',
      body: JSON.stringify({ bands }),
    }),

  getVerifiedBots: (): Promise<ApiResponse<VerifiedBot[]>> =>
    apiRequest('/api/bot/verified'),

  addVerifiedBot: (bot: { name: string; user_agent_pattern: string }): Promise<ApiResponse<VerifiedBot>> =>
    apiRequest('/api/bot/verified', {
      method: 'POST',
      body: JSON.stringify(bot),
    }),

  deleteVerifiedBot: (id: number): Promise<ApiResponse<void>> =>
    apiRequest(`/api/bot/verified/${id}`, { method: 'DELETE' }),

  syncVerifiedBots: (): Promise<ApiResponse<{ synced: number }>> =>
    apiRequest('/api/bot/verified/sync', { method: 'POST' }),
}

// Bot events (for bot-detection page)
export const botEventsApi = {
  getBotEvents: (range: string = '24h', limit?: number): Promise<ApiResponse<SecurityEventData[]>> =>
    apiRequest(`/api/events/bot?range=${range}${limit != null ? `&limit=${limit}` : ''}`),
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

// Managed rule packs (OWASP CRS + feed-synced)
export interface ManagedRulePack {
  id: number
  name: string
  pack_id: string
  source_url: string | null
  version: string | null
  enabled: boolean
  last_synced_at: string | null
  created_at: string
  updated_at: string
  rule_count?: number
}

export interface ManagedRulesResponse {
  packs: Array<{
    pack_id: string
    name: string
    version: string
    enabled: boolean
    last_synced_at: string | null
    rules: Array<{ id: number; name: string; pattern: string; applies_to: string; action: string }>
  }>
}

export const managedRulesApi = {
  getPacks: (enabledOnly: boolean = false): Promise<ApiResponse<ManagedRulePack[]>> =>
    apiRequest(`/api/rules/managed/packs?enabled_only=${enabledOnly}`),

  getManagedRules: (enabledOnly: boolean = true): Promise<ManagedRulesResponse> =>
    apiRequest(`/api/rules/managed?enabled_only=${enabledOnly}`),

  togglePack: (packId: string, enabled: boolean): Promise<ApiResponse<ManagedRulePack>> =>
    apiRequest(`/api/rules/managed/packs/${encodeURIComponent(packId)}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    }),

  syncNow: (packId?: string): Promise<ApiResponse<{ rules_created: number; rules_updated: number; version: string }>> =>
    apiRequest('/api/rules/managed/sync', {
      method: 'POST',
      body: JSON.stringify(packId != null ? { pack_id: packId } : {}),
    }),
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

// Settings API (account preferences)
export interface AccountSettings {
  theme?: string
  default_time_range?: string
  notifications?: boolean
  email_alerts?: boolean
  auto_block_threats?: boolean
  alert_severity_critical?: boolean
  alert_severity_high?: boolean
  alert_severity_medium?: boolean
  webhook_url?: string
  alert_emails?: string
}

export interface RetentionSettings {
  metrics_days: number
  traffic_days: number
  alerts_days: number
  threats_days: number
}

export const settingsApi = {
  get: (): Promise<ApiResponse<AccountSettings>> =>
    apiRequest('/api/settings'),

  update: (payload: Partial<AccountSettings>): Promise<ApiResponse<AccountSettings>> =>
    apiRequest('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  getRetention: (): Promise<ApiResponse<RetentionSettings>> =>
    apiRequest('/api/settings/retention'),
}

// API Keys (current user)
export interface ApiKeyMeta {
  id: string
  name: string
  prefix: string
  created_at: string
}

export interface ApiKeyCreated {
  key: string
  id: string
  name: string
  created_at: string
}

export const apiKeysApi = {
  list: (): Promise<ApiResponse<ApiKeyMeta[]>> =>
    apiRequest('/api/users/me/keys'),

  create: (params: { name?: string }): Promise<ApiResponse<ApiKeyCreated>> =>
    apiRequest('/api/users/me/keys', {
      method: 'POST',
      body: JSON.stringify({ name: params.name || '' }),
    }),

  revoke: (keyId: string): Promise<ApiResponse<{ revoked: string }>> =>
    apiRequest(`/api/users/me/keys/${keyId}`, { method: 'DELETE' }),
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