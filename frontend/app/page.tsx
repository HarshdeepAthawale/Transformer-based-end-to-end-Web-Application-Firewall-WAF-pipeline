'use client'

import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { MetricsOverview } from '@/components/metrics-overview'
import { ChartsSection } from '@/components/charts-section'
import { AlertsSection } from '@/components/alerts-section'
import { ActivityFeed } from '@/components/activity-feed'
import { ErrorBoundary } from '@/components/error-boundary'
import { useState, useEffect } from 'react'
import { wsManager } from '@/lib/api'

export default function Page() {
  const [timeRange, setTimeRange] = useState('24h')

  // Ensure WebSocket is connected
  useEffect(() => {
    // Only attempt connection if not already connected or connecting
    if (!wsManager.isConnected && wsManager.connectionStatus !== 'connecting') {
      wsManager.connect().catch(err => {
        // Error is already logged by WebSocketManager
        // The manager will automatically attempt reconnection
        console.debug('[Overview] WebSocket connection attempt failed, will retry automatically')
      })
    }
  }, [])

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <ErrorBoundary>
                <MetricsOverview />
              </ErrorBoundary>

              {/* Responsive Dashboard Grid */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                {/* Main Content Area - Charts */}
                <div className="xl:col-span-8">
                  <ErrorBoundary>
                    <ChartsSection timeRange={timeRange} />
                  </ErrorBoundary>
                </div>

                {/* Sidebar Content - Alerts & Activity */}
                <div className="xl:col-span-4 space-y-6">
                  <ErrorBoundary>
                    <AlertsSection />
                  </ErrorBoundary>
                  <ErrorBoundary>
                    <ActivityFeed />
                  </ErrorBoundary>
                </div>
              </div>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
