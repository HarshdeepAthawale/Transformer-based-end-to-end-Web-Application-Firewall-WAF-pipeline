'use client'

import { useEffect, useState, useMemo } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { AlertTriangle, Shield, Target, Loader2, AlertCircle, Search, Eye, Ban, Download } from 'lucide-react'
import { threatsApi, ThreatData, wsManager } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'

export default function ThreatsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [threats, setThreats] = useState<ThreatData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [threatStats, setThreatStats] = useState<Record<string, number>>({})
  
  // Filtering state
  const [searchQuery, setSearchQuery] = useState('')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedThreat, setSelectedThreat] = useState<ThreatData | null>(null)
  const [showDetails, setShowDetails] = useState(false)

  // Fetch threats data and stats
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const [threatsResponse, statsResponse] = await Promise.all([
          threatsApi.getByTimeRange(timeRange),
          threatsApi.getStats(timeRange)
        ])
        if (threatsResponse.success) {
          setThreats(threatsResponse.data)
        }
        if (statsResponse.success) {
          setThreatStats(statsResponse.data)
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[ThreatsPage] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[ThreatsPage] Failed to fetch threats:', err)
          setError('Failed to load threats data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()

    // Subscribe to real-time threat updates
    const handleThreatUpdate = (newThreat: ThreatData) => {
      setThreats(prev => [newThreat, ...prev.slice(0, 499)]) // Keep latest 500 threats
    }

    wsManager.subscribe('threat', handleThreatUpdate)

    return () => {
      wsManager.unsubscribe('threat')
    }
  }, [timeRange])

  // Filter threats
  const filteredThreats = useMemo(() => {
    let filtered = [...threats]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(threat =>
        threat.type.toLowerCase().includes(query) ||
        threat.source.toLowerCase().includes(query) ||
        threat.endpoint.toLowerCase().includes(query) ||
        threat.details?.toLowerCase().includes(query)
      )
    }

    // Apply severity filter
    if (severityFilter !== 'all') {
      filtered = filtered.filter(threat => threat.severity === severityFilter)
    }

    // Apply type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter(threat => threat.type.toLowerCase() === typeFilter.toLowerCase())
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(threat => 
        statusFilter === 'blocked' ? threat.blocked : !threat.blocked
      )
    }

    return filtered
  }, [threats, searchQuery, severityFilter, typeFilter, statusFilter])

  // Calculate threat statistics
  const criticalCount = filteredThreats.filter(t => t.severity === 'critical').length
  const totalThreats = filteredThreats.length
  const blockedCount = filteredThreats.filter(t => t.blocked).length

  const severityColors = {
    critical: 'bg-red-500/10 text-red-600 border-red-500/20',
    high: 'bg-orange-500/10 text-orange-600 border-orange-500/20',
    medium: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20',
    low: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
  }

  const handleExport = () => {
    const csv = [
      ['Type', 'Severity', 'Source', 'Endpoint', 'Blocked', 'Time'].join(','),
      ...filteredThreats.map(t => [
        t.type,
        t.severity,
        t.source,
        t.endpoint,
        t.blocked ? 'Yes' : 'No',
        t.time
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `threats-${new Date().toISOString()}.csv`
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
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Threat Detection</h2>
                <p className="text-muted-foreground">Monitor and manage detected threats and security incidents</p>
              </div>

              {/* Statistics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-red-500/10 rounded-lg">
                      <AlertTriangle className="w-6 h-6 text-red-600" />
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">Critical Threats</p>
                      <p className="text-2xl font-bold">{criticalCount}</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-orange-500/10 rounded-lg">
                      <Target className="w-6 h-6 text-orange-600" />
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">Total Threats</p>
                      <p className="text-2xl font-bold">{totalThreats}</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-green-500/10 rounded-lg">
                      <Shield className="w-6 h-6 text-green-600" />
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">Blocked</p>
                      <p className="text-2xl font-bold">{blockedCount}</p>
                    </div>
                  </div>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-blue-500/10 rounded-lg">
                      <Shield className="w-6 h-6 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-muted-foreground text-sm">Block Rate</p>
                      <p className="text-2xl font-bold">
                        {totalThreats > 0 ? ((blockedCount / totalThreats) * 100).toFixed(1) : 0}%
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Filters */}
              <Card className="p-4">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search threats..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <Select value={severityFilter} onValueChange={setSeverityFilter}>
                    <SelectTrigger className="w-full md:w-[180px]">
                      <SelectValue placeholder="Severity" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Severities</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="low">Low</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={typeFilter} onValueChange={setTypeFilter}>
                    <SelectTrigger className="w-full md:w-[180px]">
                      <SelectValue placeholder="Type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      {Object.keys(threatStats).map(type => (
                        <SelectItem key={type} value={type}>{type}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-full md:w-[180px]">
                      <SelectValue placeholder="Status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="blocked">Blocked</SelectItem>
                      <SelectItem value="allowed">Allowed</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button onClick={handleExport} variant="outline">
                    <Download className="mr-2 h-4 w-4" />
                    Export
                  </Button>
                </div>
              </Card>

              {/* Threats List */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Threats ({filteredThreats.length})</h3>
                <div className="space-y-3">
                  {filteredThreats.length === 0 ? (
                    <div className="text-center py-8">
                      <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">No threats found</p>
                    </div>
                  ) : (
                    filteredThreats.slice(0, 100).map((threat) => (
                      <div
                        key={threat.id}
                        className="flex items-center justify-between p-4 bg-card/50 rounded-lg border border-border hover:border-accent/50 transition-colors cursor-pointer"
                        onClick={() => {
                          setSelectedThreat(threat)
                          setShowDetails(true)
                        }}
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-1">
                            <h4 className="font-semibold">{threat.type}</h4>
                            <span className={`px-2 py-1 rounded text-xs font-medium border ${severityColors[threat.severity as keyof typeof severityColors]}`}>
                              {threat.severity}
                            </span>
                            {threat.blocked && (
                              <span className="px-2 py-1 bg-green-500/10 text-green-600 rounded text-xs font-medium">
                                Blocked
                              </span>
                            )}
                          </div>
                          <div className="flex gap-4 text-sm text-muted-foreground">
                            <span>Source: <span className="text-foreground font-mono">{threat.source}</span></span>
                            <span>Endpoint: <span className="text-foreground font-mono">{threat.endpoint}</span></span>
                            <span>{threat.time}</span>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedThreat(threat)
                            setShowDetails(true)
                          }}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              </Card>

              {/* Threat Details Dialog */}
              <Dialog open={showDetails} onOpenChange={setShowDetails}>
                <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>Threat Details</DialogTitle>
                    <DialogDescription>Complete information about the detected threat</DialogDescription>
                  </DialogHeader>
                  {selectedThreat && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold mb-2">Threat Type</h4>
                          <p>{selectedThreat.type}</p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">Severity</h4>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${severityColors[selectedThreat.severity as keyof typeof severityColors]}`}>
                            {selectedThreat.severity}
                          </span>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">Source IP</h4>
                          <p className="font-mono">{selectedThreat.source}</p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">Endpoint</h4>
                          <p className="font-mono break-all">{selectedThreat.endpoint}</p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">Status</h4>
                          <p>{selectedThreat.blocked ? 'Blocked' : 'Allowed'}</p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">Time</h4>
                          <p>{selectedThreat.time}</p>
                        </div>
                      </div>
                      {selectedThreat.details && (
                        <div>
                          <h4 className="font-semibold mb-2">Details</h4>
                          <p className="text-sm text-muted-foreground">{selectedThreat.details}</p>
                        </div>
                      )}
                      <div className="flex gap-2">
                        <Button variant="destructive">
                          <Ban className="mr-2 h-4 w-4" />
                          Block IP
                        </Button>
                        <Button variant="outline" onClick={handleExport}>
                          <Download className="mr-2 h-4 w-4" />
                          Export Report
                        </Button>
                      </div>
                    </div>
                  )}
                </DialogContent>
              </Dialog>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
