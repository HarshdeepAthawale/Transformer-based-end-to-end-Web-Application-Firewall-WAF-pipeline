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
    threatsPerMinute: 0,
    uptime: 0,
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
      // Check if it's a network error (backend not running)
      const isNetworkError = err instanceof ApiError && 
        (err.status === 0 || (err as any).isNetworkError)
      
      if (isNetworkError) {
        // Network errors are expected if backend isn't running
        // Don't log as error, just set offline status
        console.debug('[useRealTimeData] Backend not available, will retry')
        setError(null) // Don't show error for connection issues
      } else if (err instanceof ApiError && err.status !== 0) {
        // Actual API error (not network)
        console.error('[useRealTimeData] Failed to fetch metrics:', err)
        setError(err.message)
      } else {
        // Other errors
        console.debug('[useRealTimeData] Error fetching metrics:', err)
        setError(null)
      }
      setIsOnline(false)
      setConnectionStatus('disconnected')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // WebSocket connection and real-time updates
  useEffect(() => {
    // Update connection status based on WebSocket manager state
    const updateConnectionStatus = () => {
      const status = wsManager.connectionStatus
      setConnectionStatus(status)
      setIsOnline(status === 'connected')
    }

    // Initial connection attempt (non-blocking)
    const connectWebSocket = async () => {
      try {
        setConnectionStatus('connecting')
        // Don't await - let it connect in background
        wsManager.connect().then(() => {
          updateConnectionStatus()
        }).catch(() => {
          // Connection failed - WebSocket manager will handle reconnection
          // This is expected if backend isn't running, so don't log as error
          updateConnectionStatus()
        })
      } catch (err) {
        // Silently handle - WebSocket manager will retry
        updateConnectionStatus()
      }
    }

    // Subscribe to real-time metrics updates
    const handleMetricsUpdate = (data: RealTimeMetrics) => {
      setMetrics(data)
      setLastUpdate(new Date(data.timestamp || Date.now()))
      setIsOnline(true)
      setError(null)
      updateConnectionStatus()
    }

    // Subscribe to metrics updates
    wsManager.subscribe('metrics', handleMetricsUpdate)

    // Initial connection attempt
    connectWebSocket()

    // Monitor connection status periodically
    const statusCheckInterval = setInterval(() => {
      updateConnectionStatus()
    }, 2000) // Check every 2 seconds

    // Fallback polling every 30 seconds if WebSocket fails
    const pollingInterval = setInterval(() => {
      if (!wsManager.isConnected) {
        fetchMetrics()
      }
    }, 30000)

    // Initial fetch
    fetchMetrics()

    return () => {
      clearInterval(statusCheckInterval)
      clearInterval(pollingInterval)
      wsManager.unsubscribe('metrics')
      // Don't disconnect - let other components use the connection
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
