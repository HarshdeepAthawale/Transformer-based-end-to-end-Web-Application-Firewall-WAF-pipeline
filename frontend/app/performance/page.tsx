'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertCircle, Cpu, HardDrive, Activity } from 'lucide-react'
import { chartsApi, ChartDataPoint, wsManager } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'

export default function PerformancePage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [performanceData, setPerformanceData] = useState<ChartDataPoint[]>([])
  const [currentMetrics, setCurrentMetrics] = useState({ cpu: 0, memory: 0, latency: 0 })
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
          // Set current metrics from latest data point
          if (response.data.length > 0) {
            const latest = response.data[response.data.length - 1]
            setCurrentMetrics({
              cpu: latest.cpu || 0,
              memory: latest.memory || 0,
              latency: latest.latency || 0,
            })
          }
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[PerformancePage] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[PerformancePage] Failed to fetch performance data:', err)
          setError('Failed to load performance data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchPerformanceData()

    // Subscribe to real-time performance updates
    const handlePerformanceUpdate = (data: any) => {
      const now = new Date().toISOString()
      setPerformanceData(prev => {
        const newPoint: ChartDataPoint = {
          time: now,
          cpu: data.cpu || 0,
          memory: data.memory || 0,
          latency: data.latency || 0,
        }
        setCurrentMetrics({
          cpu: data.cpu || 0,
          memory: data.memory || 0,
          latency: data.latency || 0,
        })
        return [...prev.slice(-59), newPoint]
      })
    }

    wsManager.subscribe('performance', handlePerformanceUpdate)

    return () => {
      wsManager.unsubscribe('performance')
    }
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
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">System Performance</h2>
                <p className="text-muted-foreground">Monitor system resources and performance metrics</p>
              </div>

              {/* Current Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-500/10 rounded-lg">
                        <Cpu className="w-5 h-5 text-blue-600" />
                      </div>
                      <div>
                        <p className="text-muted-foreground text-sm">CPU Usage</p>
                        <p className="text-2xl font-bold">{Math.round(currentMetrics.cpu)}%</p>
                      </div>
                    </div>
                  </div>
                  <Progress value={currentMetrics.cpu} className="h-2" />
                  <p className="text-xs text-muted-foreground mt-2">
                    Peak: {peakCpu}% | Avg: {avgCpu}%
                  </p>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-500/10 rounded-lg">
                        <HardDrive className="w-5 h-5 text-green-600" />
                      </div>
                      <div>
                        <p className="text-muted-foreground text-sm">Memory Usage</p>
                        <p className="text-2xl font-bold">{Math.round(currentMetrics.memory)}%</p>
                      </div>
                    </div>
                  </div>
                  <Progress value={currentMetrics.memory} className="h-2" />
                  <p className="text-xs text-muted-foreground mt-2">
                    Peak: {peakMemory}% | Avg: {avgMemory}%
                  </p>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-orange-500/10 rounded-lg">
                        <Activity className="w-5 h-5 text-orange-600" />
                      </div>
                      <div>
                        <p className="text-muted-foreground text-sm">Response Latency</p>
                        <p className="text-2xl font-bold">{Math.round(currentMetrics.latency)}ms</p>
                      </div>
                    </div>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-orange-500 transition-all"
                      style={{ width: `${Math.min((currentMetrics.latency / 1000) * 100, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Peak: {peakLatency}ms | Avg: {avgLatency}ms
                  </p>
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
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
