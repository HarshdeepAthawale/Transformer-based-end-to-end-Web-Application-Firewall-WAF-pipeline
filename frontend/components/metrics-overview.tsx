'use client'

import { ArrowUpRight, ArrowDownRight, AlertTriangle, Shield, Zap, BarChart3, RefreshCw } from 'lucide-react'
import { useRealTimeData } from '@/hooks/use-real-time-data'
import { useEffect, useState } from 'react'

export function MetricsOverview() {
  const { metrics: realTimeMetrics, getFreshnessLevel } = useRealTimeData()
  const [animatedValues, setAnimatedValues] = useState({
    requests: 2400000,
    blocked: 1247,
    attackRate: 2.4,
    responseTime: 45,
  })

  // Animate value changes
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedValues(prev => ({
        requests: prev.requests + (realTimeMetrics.requests - prev.requests) * 0.1,
        blocked: prev.blocked + (realTimeMetrics.blocked - prev.blocked) * 0.1,
        attackRate: prev.attackRate + (realTimeMetrics.attackRate - prev.attackRate) * 0.1,
        responseTime: prev.responseTime + (realTimeMetrics.responseTime - prev.responseTime) * 0.1,
      }))
    }, 100)

    return () => clearInterval(interval)
  }, [realTimeMetrics])

  const metrics = [
    {
      label: 'Total Requests',
      value: `${(animatedValues.requests / 1000000).toFixed(1)}M`,
      change: '+12.5%',
      trend: 'up',
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
      value: animatedValues.blocked.toLocaleString(),
      change: '+8.2%',
      trend: 'up',
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
      change: '-3.1%',
      trend: 'down',
      icon: AlertTriangle,
      color: 'text-muted-foreground',
      bgColor: 'bg-card',
      borderColor: 'border-l-border',
      glowColor: '',
      priority: 'critical',
      isLive: true,
    },
    {
      label: 'Response Time',
      value: `${Math.round(animatedValues.responseTime)}ms`,
      change: '-5.2%',
      trend: 'down',
      icon: Zap,
      color: 'text-muted-foreground',
      bgColor: 'bg-card',
      borderColor: 'border-l-border',
      glowColor: '',
      priority: 'normal',
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
            className={`${metric.bgColor} rounded-lg ${metric.borderColor} p-6 border border-l-4`}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`p-2 rounded-md bg-muted ${metric.color}`}>
                <Icon size={20} />
              </div>
              <div className={`flex items-center gap-1 text-sm font-medium ${trendColor}`}>
                <TrendIcon size={14} />
                {metric.change}
              </div>
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">{metric.label}</p>
              <p className={`text-2xl font-semibold text-foreground ${
                metric.priority === 'critical' ? 'text-foreground' : ''
              }`}>{metric.value}</p>
              {metric.isLive && (
                <p className="text-xs text-muted-foreground">Real-time</p>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
