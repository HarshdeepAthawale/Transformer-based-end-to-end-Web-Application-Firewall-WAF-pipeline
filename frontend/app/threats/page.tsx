'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { AlertTriangle, Shield, Target, Loader2, AlertCircle } from 'lucide-react'
import { threatsApi, ThreatData, wsManager } from '@/lib/api'

export default function ThreatsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [threats, setThreats] = useState<ThreatData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch threats data
  useEffect(() => {
    const fetchThreats = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await threatsApi.getRecent(20)
        if (response.success) {
          setThreats(response.data)
        }
      } catch (err) {
        console.error('[ThreatsPage] Failed to fetch threats:', err)
        setError('Failed to load threats data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchThreats()

    // Subscribe to real-time threat updates
    wsManager.subscribe('threat', (newThreat: ThreatData) => {
      setThreats(prev => [newThreat, ...prev.slice(0, 19)]) // Keep latest 20 threats
    })

    return () => {
      wsManager.unsubscribe('threat')
    }
  }, [])

  // Calculate threat statistics
  const criticalCount = threats.filter(t => t.severity === 'critical').length
  const totalThreats = threats.length

  const severityColors = {
    critical: 'bg-destructive/10 text-destructive border-destructive/20',
    high: 'bg-orange-500/10 text-orange-600 border-orange-500/20',
    medium: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20',
    low: 'bg-muted text-muted-foreground border-muted-foreground/20',
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
                <h2 className="text-2xl font-bold mb-2">Threat Detection</h2>
                <p className="text-muted-foreground">Monitor and manage detected threats and security incidents</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <Loader2 className="animate-spin h-8 w-8 text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading threats data...</span>
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
                <h2 className="text-2xl font-bold mb-2">Threat Detection</h2>
                <p className="text-muted-foreground">Monitor and manage detected threats and security incidents</p>
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
              <h2 className="text-2xl font-bold mb-2">Threat Detection</h2>
              <p className="text-muted-foreground">Monitor and manage detected threats and security incidents</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <AlertTriangle className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Critical Threats</p>
                    <p className="text-2xl font-bold">{criticalCount}</p>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <Target className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Active Attacks</p>
                    <p className="text-2xl font-bold">{totalThreats}</p>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <Shield className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Blocked Today</p>
                    <p className="text-2xl font-bold">342</p>
                  </div>
                </div>
              </Card>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Threats</h3>
              <div className="space-y-3">
                {threats.map((threat) => (
                  <div
                    key={threat.id}
                    className="flex items-center justify-between p-4 bg-card/50 rounded-lg border border-border hover:border-accent/50 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <h4 className="font-semibold">{threat.type}</h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${severityColors[threat.severity as keyof typeof severityColors]}`}>
                          {threat.severity}
                        </span>
                      </div>
                      <div className="flex gap-4 text-sm text-muted-foreground">
                        <span>Source: <span className="text-foreground font-mono">{threat.source}</span></span>
                        <span>Endpoint: <span className="text-foreground font-mono">{threat.endpoint}</span></span>
                        <span>{threat.time}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {threat.blocked && (
                        <span className="px-3 py-1 bg-muted text-muted-foreground rounded text-xs font-medium">
                          Blocked
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </main>
      </div>
    </div>
  )
}
