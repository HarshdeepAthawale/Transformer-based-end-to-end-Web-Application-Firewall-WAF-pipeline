'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Shield, AlertTriangle, CheckCircle, Clock, Globe, Loader2, AlertCircle } from 'lucide-react'
import { activitiesApi, ActivityData, wsManager } from '@/lib/api'

export function ActivityFeed() {
  const router = useRouter()
  const [activities, setActivities] = useState<ActivityData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch activities on mount
  useEffect(() => {
    const fetchActivities = async () => {
      try {
        setError(null)
        const response = await activitiesApi.getRecent(10)
        if (response.success) {
          setActivities(response.data)
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[ActivityFeed] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[ActivityFeed] Failed to fetch activities:', err)
          setError('Failed to load activities')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchActivities()

    // Subscribe to real-time activity updates
    wsManager.subscribe('activity', (newActivity: ActivityData) => {
      setActivities(prev => [newActivity, ...prev.slice(0, 9)]) // Keep only latest 10 activities
    })

    return () => {
      wsManager.unsubscribe('activity')
    }
  }, [])

  // Icon mapping for activities
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'blocked':
        return AlertTriangle
      case 'allowed':
        return CheckCircle
      default:
        return Globe
    }
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <h2 className="text-lg font-semibold mb-4 md:mb-6" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Activity Feed</h2>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin h-6 w-6" style={{ color: 'var(--positivus-gray-dark)' }} />
          <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>Loading activities...</span>
        </div>
      </div>
    )
  }

  // Error state - show graceful empty state
  if (error) {
    return (
      <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <h2 className="text-lg font-semibold mb-4 md:mb-6" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Activity Feed</h2>
        <div className="flex flex-col items-center justify-center py-8">
          <CheckCircle className="h-12 w-12 mb-4" style={{ color: 'var(--positivus-green)' }} />
          <p className="text-center" style={{ color: 'var(--positivus-gray-dark)' }}>No recent activity</p>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-md p-4 md:p-6 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
      <h2 className="text-lg font-semibold mb-4 md:mb-6" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>Activity Feed</h2>

      <div className="space-y-3">
        {activities.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8">
            <CheckCircle className="h-12 w-12 mb-4" style={{ color: 'var(--positivus-green)' }} />
            <p className="text-center" style={{ color: 'var(--positivus-gray-dark)' }}>No recent activities</p>
          </div>
        ) : (
          activities.map((activity) => {
            const Icon = getActivityIcon(activity.type)
            const isBlocked = activity.type === 'blocked'

            return (
              <div key={activity.id} className="flex gap-3 pb-3 last:border-0 last:pb-0" style={{ borderBottom: '1px solid var(--positivus-gray)' }}>
                <div
                  className={`flex-shrink-0 p-2 rounded-md ${isBlocked ? 'animate-pulse' : ''}`}
                  style={{ backgroundColor: isBlocked ? 'var(--destructive-bg)' : 'var(--positivus-green-bg)' }}
                >
                  <Icon size={16} style={{ color: isBlocked ? 'var(--destructive)' : 'var(--positivus-green)' }} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{activity.title}</p>
                  <p className="text-xs text-muted-foreground truncate font-mono" title={activity.details}>
                    {activity.details}
                  </p>
                </div>
                <p className="text-xs text-muted-foreground flex-shrink-0 whitespace-nowrap hidden sm:block">{activity.time}</p>
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
        View Activity Log
      </button>
    </div>
  )
}
