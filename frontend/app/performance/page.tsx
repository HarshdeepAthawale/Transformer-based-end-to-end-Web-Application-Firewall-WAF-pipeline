'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertCircle } from 'lucide-react'
import { chartsApi, ChartDataPoint } from '@/lib/api'

export default function PerformancePage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [performanceData, setPerformanceData] = useState<ChartDataPoint[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch performance data
  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await chartsApi.getPerformance(timeRange)
        if (response.success) {
          setPerformanceData(response.data)
        }
      } catch (err) {
        console.error('[PerformancePage] Failed to fetch performance data:', err)
        setError('Failed to load performance data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchPerformanceData()
  }, [timeRange])

  // Calculate metrics from data
  const avgCpu = performanceData.length > 0
    ? Math.round(performanceData.reduce((sum, d) => sum + (d.cpu || 0), 0) / performanceData.length)
    : 0
  const avgMemory = performanceData.length > 0
    ? Math.round(performanceData.reduce((sum, d) => sum + (d.memory || 0), 0) / performanceData.length)
    : 0
  const avgLatency = performanceData.length > 0
    ? Math.round(performanceData.reduce((sum, d) => sum + (d.latency || 0), 0) / performanceData.length)
    : 0
  const peakCpu = performanceData.length > 0
    ? Math.max(...performanceData.map(d => d.cpu || 0))
    : 0
  const peakMemory = performanceData.length > 0
    ? Math.max(...performanceData.map(d => d.memory || 0))
    : 0
  const peakLatency = performanceData.length > 0
    ? Math.max(...performanceData.map(d => d.latency || 0))
    : 0

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
                <h2 className="text-2xl font-bold mb-2">System Performance</h2>
                <p className="text-muted-foreground">Monitor system resources and performance metrics</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <Loader2 className="animate-spin h-8 w-8 text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading performance data...</span>
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
                <h2 className="text-2xl font-bold mb-2">System Performance</h2>
                <p className="text-muted-foreground">Monitor system resources and performance metrics</p>
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
              <h2 className="text-2xl font-bold mb-2">System Performance</h2>
              <p className="text-muted-foreground">Monitor system resources and performance metrics</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-6">
                <p className="text-muted-foreground text-sm mb-2">Average CPU Usage</p>
                <p className="text-3xl font-bold">{avgCpu}%</p>
                <p className="text-xs text-muted-foreground mt-2">Peak: {peakCpu}%</p>
              </Card>

              <Card className="p-6">
                <p className="text-muted-foreground text-sm mb-2">Average Memory Usage</p>
                <p className="text-3xl font-bold">{avgMemory}%</p>
                <p className="text-xs text-muted-foreground mt-2">Peak: {peakMemory}%</p>
              </Card>

              <Card className="p-6">
                <p className="text-muted-foreground text-sm mb-2">Average Latency</p>
                <p className="text-3xl font-bold">{avgLatency}ms</p>
                <p className="text-xs text-muted-foreground mt-2">Peak: {peakLatency}ms</p>
              </Card>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">CPU & Memory Usage</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="time" stroke="var(--muted-foreground)" />
                  <YAxis stroke="var(--muted-foreground)" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                    labelStyle={{ color: 'var(--foreground)' }}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="cpu" fill="var(--chart-1)" stroke="var(--chart-1)" opacity={0.6} />
                  <Area type="monotone" dataKey="memory" fill="var(--chart-2)" stroke="var(--chart-2)" opacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Response Latency</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="time" stroke="var(--muted-foreground)" />
                  <YAxis stroke="var(--muted-foreground)" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                    labelStyle={{ color: 'var(--foreground)' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="latency" stroke="var(--chart-3)" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </main>
      </div>
    </div>
  )
}
