'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertCircle } from 'lucide-react'
import { analyticsApi, ChartDataPoint } from '@/lib/api'

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [analyticsData, setAnalyticsData] = useState<ChartDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch analytics data
  useEffect(() => {
    const fetchAnalyticsData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await analyticsApi.getOverview(timeRange)
        if (response.success) {
          setAnalyticsData(response.data)
        }
      } catch (err) {
        console.error('[AnalyticsPage] Failed to fetch analytics data:', err)
        setError('Failed to load analytics data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchAnalyticsData()
  }, [timeRange])

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Analytics</h2>
                <p className="text-muted-foreground">Detailed analytics and statistics for your WAF</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <Loader2 className="animate-spin h-8 w-8 text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading analytics data...</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Analytics</h2>
                <p className="text-muted-foreground">Detailed analytics and statistics for your WAF</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <span className="ml-2 text-destructive">{error}</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Analytics</h2>
              <p className="text-muted-foreground">Detailed analytics and statistics for your WAF</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Request Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={analyticsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="date" stroke="var(--muted-foreground)" />
                    <YAxis stroke="var(--muted-foreground)" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                      labelStyle={{ color: 'var(--foreground)' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="requests" stroke="var(--chart-1)" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Blocked vs Allowed</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="date" stroke="var(--muted-foreground)" />
                    <YAxis stroke="var(--muted-foreground)" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                      labelStyle={{ color: 'var(--foreground)' }}
                    />
                    <Legend />
                    <Bar dataKey="blocked" fill="var(--chart-5)" />
                    <Bar dataKey="allowed" fill="var(--chart-1)" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
