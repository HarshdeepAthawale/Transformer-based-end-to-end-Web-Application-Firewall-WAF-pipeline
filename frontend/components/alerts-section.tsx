'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { AlertTriangle, CheckCircle, AlertCircle, Zap, Shield, Ban, Eye, X, Loader2, AlertCircle as AlertIcon } from 'lucide-react'
import { alertsApi, AlertData, wsManager } from '@/lib/api'

export function AlertsSection() {
  const router = useRouter()
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch alerts on mount
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        setError(null)
        const response = await alertsApi.getActive()
        if (response.success) {
          setAlerts(response.data)
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[AlertsSection] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[AlertsSection] Failed to fetch alerts:', err)
          setError('Failed to load alerts')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchAlerts()

    // Subscribe to real-time alert updates
    wsManager.subscribe('alert', (newAlert: AlertData) => {
      setAlerts(prev => [newAlert, ...prev.slice(0, 9)]) // Keep only latest 10 alerts
    })

    return () => {
      wsManager.unsubscribe('alert')
    }
  }, [])

  // Icon mapping for alerts
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical':
        return AlertTriangle
      case 'warning':
        return AlertCircle
      case 'info':
        return CheckCircle
      default:
        return AlertIcon
    }
  }

  const getAlertStyles = (type: string) => {
    switch (type) {
      case 'critical':
        return 'bg-card border text-muted-foreground'
      case 'warning':
        return 'bg-card border text-muted-foreground'
      case 'info':
        return 'bg-card border text-muted-foreground'
      default:
        return 'bg-card border text-muted-foreground'
    }
  }

  const getAlertIconColor = (type: string) => {
    switch (type) {
      case 'critical':
        return 'text-muted-foreground'
      case 'warning':
        return 'text-muted-foreground'
      case 'info':
        return 'text-muted-foreground'
      default:
        return 'text-muted-foreground'
    }
  }

  const getActionButton = (action: string) => {
    switch (action) {
      case 'block_ip':
        return { icon: Ban, label: 'Block IP', color: 'text-security-critical hover:bg-security-critical/10' }
      case 'investigate':
        return { icon: Eye, label: 'Investigate', color: 'text-foreground hover:bg-foreground/10' }
      case 'dismiss':
        return { icon: X, label: 'Dismiss', color: 'text-muted-foreground hover:bg-muted/10' }
      case 'rate_limit':
        return { icon: Shield, label: 'Rate Limit', color: 'text-security-high hover:bg-security-high/10' }
      case 'monitor':
        return { icon: Eye, label: 'Monitor', color: 'text-security-medium hover:bg-security-medium/10' }
      case 'optimize':
        return { icon: Zap, label: 'Optimize', color: 'text-security-low hover:bg-security-low/10' }
      case 'view_details':
        return { icon: Eye, label: 'View Details', color: 'text-foreground hover:bg-foreground/10' }
      default:
        return { icon: Eye, label: 'View', color: 'text-muted-foreground hover:bg-muted/10' }
    }
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin h-6 w-6" style={{ color: 'var(--positivus-gray-dark)' }} />
          <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>Loading alerts...</span>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <div className="flex items-center justify-center py-8">
          <AlertIcon className="h-6 w-6 text-destructive" />
          <span className="ml-2 text-destructive">{error}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
      <div className="flex items-center justify-between gap-4 mb-4 md:mb-6">
        <h2 className="text-lg font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Recent Alerts</h2>
        <span className="text-xs px-3 py-1 rounded-md font-medium shrink-0" style={{ backgroundColor: 'var(--positivus-gray)', color: 'var(--positivus-gray-dark)' }}>
          {alerts.length} Active
        </span>
      </div>

      <div className="space-y-3">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8">
            <CheckCircle className="h-12 w-12 mb-4" style={{ color: 'var(--positivus-green)' }} />
            <p className="text-center" style={{ color: 'var(--positivus-gray-dark)' }}>No active alerts</p>
          </div>
        ) : (
          alerts.map((alert) => {
            const Icon = getAlertIcon(alert.type)
            return (
              <div
                key={alert.id}
                className="p-3 rounded-md border-2 border-l-4"
                style={{
                  backgroundColor: 'var(--positivus-white)',
                  borderColor: 'var(--positivus-gray)',
                  borderLeftColor: alert.type === 'critical' ? 'var(--destructive)' : 'var(--positivus-green)',
                }}
              >
                <div className="flex gap-3">
                  <Icon className="flex-shrink-0 mt-0.5" size={16} style={{ color: alert.type === 'critical' ? 'var(--destructive)' : 'var(--positivus-green)' }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <p className="font-medium text-sm">{alert.title}</p>
                        <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                          alert.severity === 'high' ? 'bg-muted text-muted-foreground' :
                          alert.severity === 'medium' ? 'bg-muted text-muted-foreground' :
                          'bg-muted text-muted-foreground'
                        }`}>
                          {alert.severity}
                        </span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mb-2 font-mono">{alert.description}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">{alert.time}</span>
                      <span className="text-xs text-muted-foreground">{alert.source}</span>
                    </div>
                    <div className="flex gap-1">
                      {alert.actions.slice(0, 2).map((action) => {
                        const actionBtn = getActionButton(action)
                        const ActionIcon = actionBtn.icon
                        return (
                          <button
                            key={action}
                            className="p-1 rounded text-xs hover:bg-muted text-muted-foreground"
                            title={actionBtn.label}
                          >
                            <ActionIcon size={12} />
                          </button>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })
        )}
      </div>

      <button
        onClick={() => router.push('/audit-logs')}
        className="w-full mt-4 py-2 text-sm font-medium rounded-md border-2 transition-colors hover:bg-accent"
        style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
      >
        View All Alerts
      </button>
    </div>
  )
}
