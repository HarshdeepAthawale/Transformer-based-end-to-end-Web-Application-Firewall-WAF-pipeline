'use client'

import { useEffect, useState } from 'react'
import { Shield, AlertTriangle, CheckCircle, Clock, Globe, Loader2, AlertCircle } from 'lucide-react'
import { activitiesApi, ActivityData, wsManager } from '@/lib/api'

export function ActivityFeed() {
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
      <div className="bg-card rounded-lg border border-border p-4 md:p-6">
        <h2 className="text-lg font-semibold mb-4 md:mb-6 text-foreground security-text-metric">Activity Feed</h2>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin h-6 w-6 text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading activities...</span>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="bg-card rounded-lg border border-border p-4 md:p-6">
        <h2 className="text-lg font-semibold mb-4 md:mb-6 text-foreground security-text-metric">Activity Feed</h2>
        <div className="flex items-center justify-center py-8">
          <AlertCircle className="h-6 w-6 text-destructive" />
          <span className="ml-2 text-destructive">{error}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-card rounded-lg border border-border p-4 md:p-6">
      <h2 className="text-lg font-semibold mb-4 md:mb-6 text-foreground security-text-metric">Activity Feed</h2>

      <div className="space-y-3">
        {activities.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No recent activities</p>
          </div>
        ) : (
          activities.map((activity) => {
            const Icon = getActivityIcon(activity.type)
            const isBlocked = activity.type === 'blocked'

            return (
              <div key={activity.id} className="flex gap-3 pb-3 border-b border-border last:border-0 last:pb-0">
                <div className={`flex-shrink-0 p-2 rounded-lg ${
                  isBlocked
                    ? 'bg-muted animate-pulse'
                    : 'bg-muted'
                }`}>
                  <Icon className={
                    isBlocked
                      ? 'text-muted-foreground'
                      : 'text-muted-foreground'
                  } size={16} />
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

      <button className="w-full mt-4 py-2 text-sm font-medium text-accent hover:bg-accent/10 rounded-lg transition-colors">
        View Activity Log
      </button>
    </div>
  )
}
