'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Loader2, AlertCircle } from 'lucide-react'
import { trafficApi, TrafficData, wsManager } from '@/lib/api'

export default function TrafficPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [trafficData, setTrafficData] = useState<TrafficData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch traffic data
  useEffect(() => {
    const fetchTraffic = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await trafficApi.getRecent(50)
        if (response.success) {
          setTrafficData(response.data)
        }
      } catch (err) {
        console.error('[TrafficPage] Failed to fetch traffic data:', err)
        setError('Failed to load traffic data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchTraffic()

    // Subscribe to real-time traffic updates
    wsManager.subscribe('traffic', (newTraffic: TrafficData) => {
      setTrafficData(prev => [newTraffic, ...prev.slice(0, 49)]) // Keep latest 50 entries
    })

    return () => {
      wsManager.unsubscribe('traffic')
    }
  }, [])

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
                <h2 className="text-2xl font-bold mb-2">Traffic Monitor</h2>
                <p className="text-muted-foreground">Real-time traffic and request monitoring</p>
              </div>
              <Card className="p-6">
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="animate-spin h-6 w-6 text-muted-foreground" />
                  <span className="ml-2 text-muted-foreground">Loading traffic data...</span>
                </div>
              </Card>
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
                <h2 className="text-2xl font-bold mb-2">Traffic Monitor</h2>
                <p className="text-muted-foreground">Real-time traffic and request monitoring</p>
              </div>
              <Card className="p-6">
                <div className="flex items-center justify-center py-16">
                  <AlertCircle className="h-6 w-6 text-destructive" />
                  <span className="ml-2 text-destructive">{error}</span>
                </div>
              </Card>
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
              <h2 className="text-2xl font-bold mb-2">Traffic Monitor</h2>
              <p className="text-muted-foreground">Real-time traffic and request monitoring</p>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Recent Requests</h3>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border hover:bg-transparent">
                      <TableHead className="text-muted-foreground">IP Address</TableHead>
                      <TableHead className="text-muted-foreground">Method</TableHead>
                      <TableHead className="text-muted-foreground">Endpoint</TableHead>
                      <TableHead className="text-muted-foreground">Status</TableHead>
                      <TableHead className="text-muted-foreground">Size</TableHead>
                      <TableHead className="text-muted-foreground">Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trafficData.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">
                          No traffic data available
                        </TableCell>
                      </TableRow>
                    ) : (
                      trafficData.map((request) => (
                        <TableRow key={request.timestamp} className="border-border hover:bg-card/50">
                          <TableCell className="font-mono text-sm">{request.ip}</TableCell>
                          <TableCell>
                            <span className="px-2 py-1 bg-black/20 text-black rounded text-xs font-medium">
                              {request.method}
                            </span>
                          </TableCell>
                          <TableCell className="font-mono text-sm">{request.endpoint}</TableCell>
                          <TableCell>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              request.status >= 200 && request.status < 300 ? 'bg-muted text-muted-foreground' :
                              request.status >= 400 ? 'bg-muted text-muted-foreground' : 'bg-muted text-muted-foreground'
                            }`}>
                              {request.status}
                          </span>
                        </TableCell>
                        <TableCell className="text-sm">{request.size}</TableCell>
                        <TableCell className="text-sm">{request.time}</TableCell>
                      </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </Card>
          </div>
        </main>
      </div>
    </div>
  )
}
