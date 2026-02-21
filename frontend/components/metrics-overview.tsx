'use client'

import { ArrowUpRight, ArrowDownRight, AlertTriangle, Shield, BarChart3, RefreshCw } from 'lucide-react'
import { useRealTimeData } from '@/hooks/use-real-time-data'
import { useEffect, useState } from 'react'
import { wsManager, RealTimeMetrics } from '@/lib/api'

export function MetricsOverview() {
  const { metrics: realTimeMetrics, refresh } = useRealTimeData()
  const [animatedValues, setAnimatedValues] = useState({
    requests: 0,
    blocked: 0,
    attackRate: 0,
    threatsPerMinute: 0,
  })
  const [previousMetrics, setPreviousMetrics] = useState<RealTimeMetrics | null>(null)

  // Subscribe to real-time metrics updates
  useEffect(() => {
    const handleMetricsUpdate = (data: RealTimeMetrics) => {
      setPreviousMetrics(realTimeMetrics)
    }

    wsManager.subscribe('metrics', handleMetricsUpdate)

    return () => {
      wsManager.unsubscribe('metrics')
    }
  }, [realTimeMetrics])

  // Calculate trends
  const getTrend = (current: number, previous: number | null) => {
    if (previous === null || previous === 0) return { change: '0%', trend: 'neutral' as const }
    const change = ((current - previous) / previous) * 100
    return {
      change: `${change > 0 ? '+' : ''}${change.toFixed(1)}%`,
      trend: change > 0 ? 'up' as const : change < 0 ? 'down' as const : 'neutral' as const
    }
  }

  // Animate value changes (throttled to reduce re-renders and lag)
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedValues(prev => {
        const next = {
          requests: prev.requests + (realTimeMetrics.requests - prev.requests) * 0.1,
          blocked: prev.blocked + (realTimeMetrics.blocked - prev.blocked) * 0.1,
          attackRate: prev.attackRate + (realTimeMetrics.attackRate - prev.attackRate) * 0.1,
          threatsPerMinute: prev.threatsPerMinute + (realTimeMetrics.threatsPerMinute - prev.threatsPerMinute) * 0.1,
        }
        // Skip update if effectively unchanged to avoid unnecessary re-renders
        const eps = 0.5
        if (
          Math.abs(next.requests - prev.requests) < eps &&
          Math.abs(next.blocked - prev.blocked) < eps &&
          Math.abs(next.attackRate - prev.attackRate) < eps &&
          Math.abs(next.threatsPerMinute - prev.threatsPerMinute) < eps
        ) {
          return prev
        }
        return next
      })
    }, 250)

    return () => clearInterval(interval)
  }, [realTimeMetrics])

  // Calculate trends from previous metrics
  const requestsTrend = getTrend(realTimeMetrics.requests, previousMetrics?.requests || null)
  const blockedTrend = getTrend(realTimeMetrics.blocked, previousMetrics?.blocked || null)
  const attackRateTrend = getTrend(realTimeMetrics.attackRate, previousMetrics?.attackRate || null)

  const metrics = [
    {
      label: 'Total Requests',
      value: animatedValues.requests >= 1000000 
        ? `${(animatedValues.requests / 1000000).toFixed(1)}M`
        : animatedValues.requests >= 1000
        ? `${(animatedValues.requests / 1000).toFixed(1)}K`
        : Math.round(animatedValues.requests).toLocaleString(),
      change: requestsTrend.change,
      trend: requestsTrend.trend,
      icon: BarChart3,
      priority: 'normal' as const,
      isLive: true,
    },
    {
      label: 'Threats Blocked',
      value: Math.round(animatedValues.blocked).toLocaleString(),
      change: blockedTrend.change,
      trend: blockedTrend.trend,
      icon: Shield,
      priority: 'normal' as const,
      isLive: true,
    },
    {
      label: 'Attack Rate',
      value: `${animatedValues.attackRate.toFixed(1)}%`,
      change: attackRateTrend.change,
      trend: attackRateTrend.trend,
      icon: AlertTriangle,
      priority: animatedValues.attackRate > 10 ? 'critical' : 'normal',
      isLive: true,
    },
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        const TrendIcon = metric.trend === 'up' ? ArrowUpRight : ArrowDownRight
        const trendColor = metric.trend === 'up'
          ? (metric.priority === 'critical' ? 'var(--destructive)' : 'var(--positivus-green)')
          : 'var(--positivus-green)'

        return (
          <div
            key={metric.label}
            className="p-6 border-2 relative group rounded-md flex flex-col min-h-[140px]"
            style={{
              backgroundColor: 'var(--positivus-white)',
              borderColor: 'var(--positivus-gray)',
              animationDelay: `${index * 0.1}s`,
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <div
                className="p-2 rounded-md shrink-0"
                style={{ backgroundColor: 'var(--positivus-green-bg)' }}
              >
                <Icon size={20} style={{ color: 'var(--positivus-green)' }} />
              </div>
              <div className="flex items-center gap-2 min-w-0">
                <div className="flex items-center gap-1 text-sm font-medium shrink-0" style={{ color: trendColor }}>
                  {metric.trend !== 'neutral' && <TrendIcon size={14} />}
                  {metric.change}
                </div>
                <button
                  onClick={refresh}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-accent shrink-0"
                  title="Refresh"
                  style={{ color: 'var(--positivus-gray-dark)' }}
                >
                  <RefreshCw size={12} />
                </button>
              </div>
            </div>
            <div className="space-y-1 flex-1">
              <p className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>{metric.label}</p>
              <p className="text-2xl font-semibold leading-tight" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                {metric.value}
              </p>
              {metric.isLive && (
                <div className="flex items-center gap-2 mt-2">
                  <div
                    className="w-2 h-2 rounded-full animate-pulse shrink-0"
                    style={{ backgroundColor: 'var(--positivus-green)' }}
                  />
                  <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>Real-time</p>
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
