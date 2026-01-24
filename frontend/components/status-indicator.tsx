'use client'

import { useEffect, useState } from 'react'

interface StatusIndicatorProps {
  status: 'online' | 'warning' | 'critical'
  label?: string
}

export function StatusIndicator({ status, label = 'System Status' }: StatusIndicatorProps) {
  const [isAnimating, setIsAnimating] = useState(true)

  useEffect(() => {
    if (status === 'online') {
      setIsAnimating(true)
    } else {
      setIsAnimating(false)
    }
  }, [status])

  const getStatusColor = () => {
    switch (status) {
      case 'online':
        return 'bg-green-500'
      case 'warning':
        return 'bg-amber-500'
      case 'critical':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'online':
        return ''
      case 'warning':
        return 'Warning - Issues Detected'
      case 'critical':
        return 'Critical - Action Required'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="relative w-3 h-3">
        <div className={`w-3 h-3 rounded-full ${getStatusColor()} ${isAnimating ? 'animate-pulse' : ''}`} />
        {isAnimating && (
          <div
            className={`absolute inset-0 rounded-full ${getStatusColor()} animate-ping`}
            style={{ animationDuration: '1.5s' }}
          />
        )}
      </div>
      <span className="text-sm font-medium text-muted-foreground">{getStatusText()}</span>
    </div>
  )
}
