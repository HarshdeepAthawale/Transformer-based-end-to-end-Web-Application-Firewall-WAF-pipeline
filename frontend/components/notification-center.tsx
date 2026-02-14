'use client'

import { useEffect, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Bell, AlertTriangle, AlertCircle, CheckCircle, X, Loader2 } from 'lucide-react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { alertsApi, type AlertData, wsManager } from '@/lib/api'

const MAX_ALERTS = 10

function getAlertIcon(type: string) {
  switch (type) {
    case 'critical':
      return AlertTriangle
    case 'warning':
      return AlertCircle
    case 'info':
      return CheckCircle
    default:
      return AlertCircle
  }
}

export function NotificationCenter() {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [alerts, setAlerts] = useState<AlertData[]>([])
  const [loading, setLoading] = useState(false)

  const fetchAlerts = useCallback(async () => {
    setLoading(true)
    try {
      const res = await alertsApi.getActive()
      if (res?.success && Array.isArray(res.data)) {
        setAlerts(res.data.slice(0, MAX_ALERTS))
      }
    } catch {
      setAlerts([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      fetchAlerts()
    }
  }, [open, fetchAlerts])

  useEffect(() => {
    const handler = (newAlert: AlertData) => {
      setAlerts((prev) => [newAlert, ...prev.slice(0, MAX_ALERTS - 1)])
    }
    wsManager.subscribe('alert', handler)
    return () => wsManager.unsubscribe('alert')
  }, [])

  const handleDismiss = async (e: React.MouseEvent, alertId: number) => {
    e.preventDefault()
    e.stopPropagation()
    try {
      await alertsApi.dismiss(alertId)
      setAlerts((prev) => prev.filter((a) => a.id !== alertId))
    } catch {
      // ignore
    }
  }

  const count = alerts.length

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="p-2 rounded-none transition-colors relative hover:bg-accent"
          style={{ color: 'var(--positivus-black)' }}
          aria-label="Notifications"
        >
          <Bell size={20} />
          {count > 0 && (
            <span
              className="absolute top-1 right-1 min-w-[18px] h-[18px] px-1 flex items-center justify-center text-[10px] font-semibold bg-destructive text-destructive-foreground rounded-full"
              aria-hidden
            >
              {count > 99 ? '99+' : count}
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[380px] p-0 max-h-[420px] flex flex-col">
        <div className="p-3 border-b flex items-center justify-between">
          <h3 className="font-semibold text-sm">Notifications</h3>
          {count > 0 && (
            <span className="text-xs text-muted-foreground">{count} active</span>
          )}
        </div>
        <div className="overflow-auto flex-1 min-h-0">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="animate-spin h-6 w-6 text-muted-foreground" />
            </div>
          ) : alerts.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No active alerts
            </div>
          ) : (
            <ul className="divide-y">
              {alerts.map((alert) => {
                const Icon = getAlertIcon(alert.type)
                return (
                  <li key={alert.id} className="p-3 hover:bg-muted/50 transition-colors">
                    <div className="flex gap-2">
                      <Icon
                        className="flex-shrink-0 mt-0.5"
                        size={16}
                        style={{
                          color:
                            alert.type === 'critical'
                              ? 'var(--destructive)'
                              : 'var(--positivus-green)',
                        }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{alert.title}</p>
                        <p className="text-xs text-muted-foreground truncate mt-0.5">
                          {alert.description}
                        </p>
                        <div className="flex items-center justify-between mt-2">
                          <span className="text-xs text-muted-foreground">
                            {alert.time ?? alert.timestamp?.slice(11, 19)} · {alert.severity}
                          </span>
                          <button
                            type="button"
                            onClick={(e) => handleDismiss(e, alert.id)}
                            className="p-1 rounded hover:bg-muted text-muted-foreground"
                            title="Dismiss"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
        <div className="p-2 border-t">
          <button
            type="button"
            onClick={() => {
              setOpen(false)
              router.push('/dashboard')
            }}
            className="w-full py-2 text-sm font-medium rounded-md hover:bg-muted transition-colors"
          >
            View all
          </button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
