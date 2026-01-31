'use client'

import { ArrowUpRight, ArrowDownRight, AlertTriangle, Shield, Zap, BarChart3, RefreshCw } from 'lucide-react'
import { useRealTimeData } from '@/hooks/use-real-time-data'
import { useEffect, useState } from 'react'
import { metricsApi, wsManager, RealTimeMetrics } from '@/lib/api'

export function MetricsOverview() {
  const { metrics: realTimeMetrics, getFreshnessLevel, refresh } = useRealTimeData()
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

  // Animate value changes
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedValues(prev => ({
        requests: prev.requests + (realTimeMetrics.requests - prev.requests) * 0.1,
        blocked: prev.blocked + (realTimeMetrics.blocked - prev.blocked) * 0.1,
        attackRate: prev.attackRate + (realTimeMetrics.attackRate - prev.attackRate) * 0.1,
        threatsPerMinute: prev.threatsPerMinute + (realTimeMetrics.threatsPerMinute - prev.threatsPerMinute) * 0.1,
      }))
    }, 100)

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
      color: 'text-muted-foreground',
      bgColor: 'bg-card',
      borderColor: 'border-l-border',
      glowColor: '',
      priority: 'normal',
      isLive: true,
    },
    {
      label: 'Threats Blocked',
      value: Math.round(animatedValues.blocked).toLocaleString(),
      change: blockedTrend.change,
      trend: blockedTrend.trend,
      icon: Shield,
      color: 'text-muted-foreground',
      bgColor: 'bg-card',
      borderColor: 'border-l-border',
      glowColor: '',
      priority: 'normal',
      isLive: true,
    },
    {
      label: 'Attack Rate',
      value: `${animatedValues.attackRate.toFixed(1)}%`,
      change: attackRateTrend.change,
      trend: attackRateTrend.trend,
      icon: AlertTriangle,
      color: 'text-muted-foreground',
      bgColor: 'bg-card',
      borderColor: 'border-l-border',
      glowColor: '',
      priority: animatedValues.attackRate > 10 ? 'critical' : 'normal',
      isLive: true,
    },
    
  ]

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        const TrendIcon = metric.trend === 'up' ? ArrowUpRight : ArrowDownRight
        const trendColor = metric.trend === 'up'
          ? (metric.priority === 'critical' ? 'text-security-critical' : 'text-security-low')
          : 'text-security-low'

        return (
          <div
            key={metric.label}
            className={`${metric.bgColor} rounded-lg ${metric.borderColor} p-6 border border-l-4 relative group`}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`p-2 rounded-md bg-muted ${metric.color}`}>
                <Icon size={20} />
              </div>
              <div className="flex items-center gap-2">
                <div className={`flex items-center gap-1 text-sm font-medium ${trendColor}`}>
                  {metric.trend !== 'neutral' && <TrendIcon size={14} />}
                  {metric.change}
                </div>
                <button
                  onClick={refresh}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded"
                  title="Refresh"
                >
                  <RefreshCw size={12} className="text-muted-foreground" />
                </button>
              </div>
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">{metric.label}</p>
              <p className={`text-2xl font-semibold text-foreground ${
                metric.priority === 'critical' ? 'text-foreground' : ''
              }`}>{metric.value}</p>
              {metric.isLive && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <p className="text-xs text-muted-foreground">Real-time</p>
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
