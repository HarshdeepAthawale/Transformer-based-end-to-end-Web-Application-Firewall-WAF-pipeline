'use client'

import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { MetricsOverview } from '@/components/metrics-overview'
import { ChartsSection } from '@/components/charts-section'
import { AlertsSection } from '@/components/alerts-section'
import { ActivityFeed } from '@/components/activity-feed'
import { useRealTimeData } from '@/hooks/use-real-time-data'
import { useState } from 'react'

export default function Page() {
  const [timeRange, setTimeRange] = useState('24h')
  const { lastUpdate, connectionStatus, getRecentTime, getStatusIndicator } = useRealTimeData()

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <div className="p-6 space-y-6">
            {/* Real-time Status Indicator */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-sm ${
                  connectionStatus === 'connected' ? 'bg-green-500' :
                  connectionStatus === 'connecting' ? 'bg-amber-500' :
                  'bg-red-500'
                }`} />
                <span className="text-sm text-muted-foreground">
                  {connectionStatus === 'connected' ? 'Live' :
                   connectionStatus === 'connecting' ? 'Connecting...' :
                   'Offline'}
                </span>
                <span className="text-xs text-muted-foreground">
                  Updated {getRecentTime()}
                </span>
              </div>
            </div>

            <MetricsOverview />
            {/* Responsive Dashboard Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
              {/* Main Content Area - Charts */}
              <div className="xl:col-span-8">
                <ChartsSection timeRange={timeRange} />
              </div>

              {/* Sidebar Content - Alerts & Activity */}
              <div className="xl:col-span-4 space-y-6">
                <AlertsSection />
                <ActivityFeed />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
