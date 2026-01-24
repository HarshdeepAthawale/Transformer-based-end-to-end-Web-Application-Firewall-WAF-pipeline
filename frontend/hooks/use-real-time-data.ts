'use client';

import { useEffect, useState, useCallback } from 'react'
import { metricsApi, RealTimeMetrics, wsManager, ApiError } from '@/lib/api'

export function useRealTimeData() {
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [isOnline, setIsOnline] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('connecting')
  const [metrics, setMetrics] = useState<RealTimeMetrics>({
    requests: 0,
    blocked: 0,
    attackRate: 0,
    responseTime: 0,
    threatsPerMinute: 0,
    uptime: 0,
    activeConnections: 0,
    timestamp: new Date().toISOString(),
  })
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch initial metrics
  const fetchMetrics = useCallback(async () => {
    try {
      setError(null)
      const response = await metricsApi.getRealtime()
      if (response.success) {
        setMetrics(response.data)
        setLastUpdate(new Date(response.data.timestamp || Date.now()))
        setIsOnline(true)
        setConnectionStatus('connected')
      }
    } catch (err) {
      console.error('[useRealTimeData] Failed to fetch metrics:', err)
      setError(err instanceof ApiError ? err.message : 'Failed to fetch metrics')
      setIsOnline(false)
      setConnectionStatus('disconnected')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // WebSocket connection and real-time updates
  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout

    const connectWebSocket = async () => {
      try {
        setConnectionStatus('connecting')
        await wsManager.connect()
        setConnectionStatus('connected')
        setIsOnline(true)

        // Subscribe to real-time metrics updates
        wsManager.subscribe('metrics', (data: RealTimeMetrics) => {
          setMetrics(data)
          setLastUpdate(new Date(data.timestamp || Date.now()))
          setIsOnline(true)
          setError(null)
        })

        // Subscribe to connection status updates
        wsManager.subscribe('connection', (status: 'connected' | 'disconnected') => {
          setConnectionStatus(status)
          setIsOnline(status === 'connected')
        })

      } catch (err) {
        console.error('[useRealTimeData] WebSocket connection failed:', err)
        setConnectionStatus('disconnected')
        setIsOnline(false)

        // Attempt to reconnect after delay
        reconnectTimeout = setTimeout(connectWebSocket, 5000)
      }
    }

    connectWebSocket()

    // Fallback polling every 30 seconds if WebSocket fails
    const pollingInterval = setInterval(() => {
      if (!wsManager.isConnected) {
        fetchMetrics()
      }
    }, 30000)

    // Initial fetch
    fetchMetrics()

    return () => {
      clearTimeout(reconnectTimeout)
      clearInterval(pollingInterval)
      wsManager.disconnect()
    }
  }, [fetchMetrics])

  const getRecentTime = () => {
    const now = new Date()
    const diff = Math.floor((now.getTime() - lastUpdate.getTime()) / 1000)

    if (diff < 60) return `${diff}s ago`
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    return `${Math.floor(diff / 3600)}h ago`
  }

  const getStatusIndicator = () => {
    if (connectionStatus === 'disconnected') return 'offline'
    if (error) return 'warning'
    const timeSinceUpdate = Date.now() - lastUpdate.getTime()
    if (timeSinceUpdate > 15000) return 'warning'
    return 'online'
  }

  const getFreshnessLevel = () => {
    if (connectionStatus === 'disconnected') return 'stale'
    const timeSinceUpdate = Date.now() - lastUpdate.getTime()
    if (timeSinceUpdate < 3000) return 'fresh'
    if (timeSinceUpdate < 10000) return 'recent'
    return 'stale'
  }

  const refresh = () => {
    fetchMetrics()
  }

  return {
    lastUpdate,
    isOnline,
    connectionStatus,
    metrics,
    error,
    isLoading,
    getRecentTime,
    getStatusIndicator,
    getFreshnessLevel,
    refresh,
  }
}
