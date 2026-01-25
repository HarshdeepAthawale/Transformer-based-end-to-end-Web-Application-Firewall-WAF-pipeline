'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Loader2, AlertCircle, Download } from 'lucide-react'
import { analyticsApi, ChartDataPoint, wsManager } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d']

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [analyticsData, setAnalyticsData] = useState<ChartDataPoint[]>([])
  const [summary, setSummary] = useState<Record<string, any>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch analytics data
  useEffect(() => {
    const fetchAnalyticsData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const [overviewResponse, summaryResponse] = await Promise.all([
          analyticsApi.getOverview(timeRange),
          analyticsApi.getSummary(timeRange)
        ])
        if (overviewResponse.success) {
          setAnalyticsData(overviewResponse.data)
        }
        if (summaryResponse.success) {
          setSummary(summaryResponse.data)
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[AnalyticsPage] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[AnalyticsPage] Failed to fetch analytics data:', err)
          setError('Failed to load analytics data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchAnalyticsData()

    // Subscribe to real-time updates
    const handleMetricsUpdate = (data: any) => {
      const now = new Date().toISOString()
      setAnalyticsData(prev => {
        const newPoint: ChartDataPoint = {
          time: now,
          requests: data.total_requests || 0,
          blocked: data.blocked_requests || 0,
          allowed: data.allowed_requests || 0,
        }
        return [...prev.slice(-59), newPoint]
      })
    }

    wsManager.subscribe('metrics', handleMetricsUpdate)

    return () => {
      wsManager.unsubscribe('metrics')
    }
  }, [timeRange])

  // Prepare pie chart data for attack types
  const attackTypeData = [
    { name: 'SQL Injection', value: summary.sql_injection || 0 },
    { name: 'XSS', value: summary.xss || 0 },
    { name: 'DDoS', value: summary.ddos || 0 },
    { name: 'Other', value: summary.other || 0 },
  ].filter(item => item.value > 0)

  const handleExport = () => {
    const csv = [
      ['Time', 'Requests', 'Blocked', 'Allowed'].join(','),
      ...analyticsData.map(d => [
        d.time || d.date || '',
        d.requests || 0,
        d.blocked || 0,
        d.allowed || 0
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `analytics-${new Date().toISOString()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

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
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold mb-2">Analytics</h2>
                  <p className="text-muted-foreground">Detailed analytics and statistics for your WAF</p>
                </div>
                <Button onClick={handleExport} variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  Export Data
                </Button>
              </div>

              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Total Requests</p>
                  <p className="text-2xl font-bold">{summary.total_requests?.toLocaleString() || 0}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Blocked</p>
                  <p className="text-2xl font-bold">{summary.blocked_requests?.toLocaleString() || 0}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Allowed</p>
                  <p className="text-2xl font-bold">{summary.allowed_requests?.toLocaleString() || 0}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Attack Rate</p>
                  <p className="text-2xl font-bold">
                    {summary.attack_rate ? `${summary.attack_rate.toFixed(1)}%` : '0%'}
                  </p>
                </Card>
              </div>

              {/* Charts Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Request Trends</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={analyticsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="time" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                        labelStyle={{ color: 'var(--foreground)' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="requests" stroke="#0088FE" strokeWidth={2} name="Total Requests" />
                      <Line type="monotone" dataKey="blocked" stroke="#FF8042" strokeWidth={2} name="Blocked" />
                      <Line type="monotone" dataKey="allowed" stroke="#00C49F" strokeWidth={2} name="Allowed" />
                    </LineChart>
                  </ResponsiveContainer>
                </Card>

                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Blocked vs Allowed</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={analyticsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="time" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                        labelStyle={{ color: 'var(--foreground)' }}
                      />
                      <Legend />
                      <Bar dataKey="blocked" fill="#FF8042" name="Blocked" />
                      <Bar dataKey="allowed" fill="#00C49F" name="Allowed" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                {attackTypeData.length > 0 && (
                  <Card className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Attack Type Distribution</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={attackTypeData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {attackTypeData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Card>
                )}

                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Request Volume Over Time</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={analyticsData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="time" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: 'var(--card)', border: `1px solid var(--border)` }}
                        labelStyle={{ color: 'var(--foreground)' }}
                      />
                      <Legend />
                      <Area type="monotone" dataKey="requests" stroke="#0088FE" fill="#0088FE" fillOpacity={0.6} name="Requests" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
