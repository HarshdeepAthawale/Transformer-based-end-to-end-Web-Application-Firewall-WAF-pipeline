'use client'

import { useEffect, useState, useMemo, useRef, useCallback } from 'react'
import { usePathname } from 'next/navigation'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Loader2, AlertCircle, Search, Filter, Download, ArrowUpDown, Eye, RefreshCw, Radio, ArrowUp } from 'lucide-react'
import { trafficApi, TrafficData, wsManager } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'

// Utility function to format time in IST with 12-hour format and AM/PM
function formatTimeIST(timestamp: string | Date): string {
  try {
    let date: Date
    
    if (typeof timestamp === 'string') {
      // Backend sends UTC timestamps - ensure they're treated as UTC
      // If timestamp doesn't have timezone info (no Z or +/-), assume it's UTC
      let timestampStr = timestamp.trim()
      
      // Check if it already has timezone indicator
      const hasTimezone = timestampStr.endsWith('Z') || 
                         timestampStr.includes('+') || 
                         (timestampStr.includes('-') && timestampStr.length > 19 && timestampStr[19] === '-' || timestampStr[19] === '+')
      
      // If no timezone info, treat as UTC by appending 'Z'
      if (!hasTimezone && timestampStr.length > 0) {
        // Remove any trailing space or invalid chars, then add Z
        timestampStr = timestampStr.replace(/[^\d\-:T\s]/g, '') + 'Z'
      }
      
      date = new Date(timestampStr)
      
      // Validate the date
      if (isNaN(date.getTime())) {
        // If parsing failed, try treating the original string as UTC manually
        // Parse format: "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
        const cleaned = timestamp.replace(/[^\d\-:T\s]/g, ' ').trim()
        const parts = cleaned.split(/[\sT]/)
        if (parts.length >= 2) {
          const [datePart, timePart] = parts
          const [year, month, day] = datePart.split('-').map(Number)
          const [hour, minute, second] = timePart.split(':').map(Number)
          date = new Date(Date.UTC(year, month - 1, day, hour || 0, minute || 0, second || 0))
        } else {
          date = new Date(timestamp) // Last resort
        }
      }
    } else {
      date = timestamp
    }
    
    // Validate date
    if (isNaN(date.getTime())) {
      console.warn('Invalid timestamp:', timestamp)
      return 'Invalid Time'
    }
    
    // Use Intl.DateTimeFormat to convert to IST and format in 12-hour format
    // This automatically handles UTC to IST conversion
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Kolkata',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    })
    
    return formatter.format(date)
  } catch (error) {
    console.error('Error formatting time:', error, timestamp)
    // Fallback: manual UTC to IST conversion
    try {
      let date: Date
      if (typeof timestamp === 'string') {
        // Try to parse as UTC
        const utcStr = timestamp.endsWith('Z') ? timestamp : timestamp + 'Z'
        date = new Date(utcStr)
        if (isNaN(date.getTime())) {
          date = new Date(timestamp)
        }
      } else {
        date = timestamp
      }
      
      if (isNaN(date.getTime())) {
        return 'Invalid Time'
      }
      
      // Get UTC time and add IST offset (UTC+5:30 = 5.5 hours)
      const utcTime = date.getTime()
      const istOffset = 5.5 * 60 * 60 * 1000 // IST is UTC+5:30
      const istTime = new Date(utcTime + istOffset)
      
      const hours = istTime.getUTCHours()
      const minutes = istTime.getUTCMinutes()
      const seconds = istTime.getUTCSeconds()
      const ampm = hours >= 12 ? 'PM' : 'AM'
      const displayHours = hours % 12 || 12
      
      return `${displayHours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} ${ampm}`
    } catch (fallbackError) {
      return 'Invalid Time'
    }
  }
}

export default function TrafficPage() {
  const pathname = usePathname()
  const [timeRange, setTimeRange] = useState('24h')
  const [trafficData, setTrafficData] = useState<TrafficData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isRealTime, setIsRealTime] = useState(true)
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date())
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [showScrollToTop, setShowScrollToTop] = useState(false)
  const tableContainerRef = useRef<HTMLDivElement>(null)
  const scrollPositionRef = useRef<number>(0)
  const shouldAutoScrollRef = useRef<boolean>(true)
  
  // Filtering and sorting state
  const [searchQuery, setSearchQuery] = useState('')
  const [methodFilter, setMethodFilter] = useState<string>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [sortField, setSortField] = useState<keyof TrafficData | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

  const [selectedTraffic, setSelectedTraffic] = useState<TrafficData | null>(null)
  const [showDetails, setShowDetails] = useState(false)

  // Fetch traffic data
  const fetchTraffic = useCallback(async (showLoading = true) => {
    try {
      if (showLoading) setIsLoading(true)
      setIsRefreshing(true)
      setError(null)
      const response = await trafficApi.getRecent(200) // Get more recent logs
      if (response.success) {
        // Merge with existing data, avoiding duplicates
        setTrafficData(prev => {
          const existingIds = new Set(prev.map(t => t.timestamp))
          const newData = response.data.filter(t => !existingIds.has(t.timestamp))
          const merged = [...newData, ...prev]
          // Sort by timestamp descending and keep latest 500
          return merged
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
            .slice(0, 500)
        })
        setLastUpdateTime(new Date())
      }
    } catch (err: any) {
      // Only show error if it's not a network error (backend not running)
      if (err?.isNetworkError) {
        console.debug('[TrafficPage] Backend not available')
        setError(null) // Don't show error for network issues
      } else {
        console.error('[TrafficPage] Failed to fetch traffic data:', err)
        setError('Failed to load traffic data')
      }
    } finally {
      if (showLoading) setIsLoading(false)
      setIsRefreshing(false)
    }
  }, [])

  // Handle scroll position to determine if we should auto-scroll
  useEffect(() => {
    const container = tableContainerRef.current
    if (!container) return

    const handleScroll = () => {
      scrollPositionRef.current = container.scrollTop
      // If user scrolled down more than 100px, don't auto-scroll
      shouldAutoScrollRef.current = container.scrollTop < 100
      // Show scroll to top button if scrolled down
      setShowScrollToTop(container.scrollTop > 200)
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  // Auto-scroll to top when new logs arrive (if user is near top)
  useEffect(() => {
    if (shouldAutoScrollRef.current && tableContainerRef.current) {
      tableContainerRef.current.scrollTop = 0
    }
  }, [trafficData.length])

  // Fetch traffic data and set up real-time updates
  useEffect(() => {
    // Reset state when navigating to this page
    setIsLoading(true)
    setError(null)

    // Fetch fresh data
    fetchTraffic()

    // Subscribe to real-time traffic updates via WebSocket
    const handleTrafficUpdate = (newTraffic: TrafficData) => {
      setTrafficData(prev => {
        // Check if this log already exists (avoid duplicates)
        const exists = prev.some(t => t.timestamp === newTraffic.timestamp)
        if (exists) return prev

        // Add new log at the top, keep latest 500 entries
        const updated = [newTraffic, ...prev.slice(0, 499)]
        setLastUpdateTime(new Date())
        return updated
      })

      // Auto-scroll if user is near top
      if (shouldAutoScrollRef.current && tableContainerRef.current) {
        tableContainerRef.current.scrollTop = 0
      }
    }

    wsManager.subscribe('traffic', handleTrafficUpdate)

    // Polling fallback for real-time updates (every 2 seconds if WebSocket not connected)
    const pollingInterval = setInterval(() => {
      if (!wsManager.isConnected && isRealTime) {
        fetchTraffic(false) // Don't show loading spinner for polling
      }
    }, 2000)

    // Also poll every 5 seconds as backup even if WebSocket is connected
    const backupPollingInterval = setInterval(() => {
      if (isRealTime) {
        fetchTraffic(false)
      }
    }, 5000)

    return () => {
      wsManager.unsubscribe('traffic')
      clearInterval(pollingInterval)
      clearInterval(backupPollingInterval)
    }
  }, [pathname, fetchTraffic, isRealTime])

  // Filter and sort traffic data
  const filteredAndSortedData = useMemo(() => {
    let filtered = [...trafficData]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(traffic =>
        traffic.ip.toLowerCase().includes(query) ||
        traffic.endpoint.toLowerCase().includes(query) ||
        traffic.method.toLowerCase().includes(query) ||
        traffic.userAgent?.toLowerCase().includes(query)
      )
    }

    // Apply method filter
    if (methodFilter !== 'all') {
      filtered = filtered.filter(traffic => traffic.method === methodFilter)
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      if (statusFilter === '2xx') {
        filtered = filtered.filter(traffic => traffic.status >= 200 && traffic.status < 300)
      } else if (statusFilter === '4xx') {
        filtered = filtered.filter(traffic => traffic.status >= 400 && traffic.status < 500)
      } else if (statusFilter === '5xx') {
        filtered = filtered.filter(traffic => traffic.status >= 500)
      }
    }

    // Apply sorting
    if (sortField) {
      filtered.sort((a, b) => {
        const aVal = a[sortField]
        const bVal = b[sortField]
        
        if (aVal === undefined || aVal === null) return 1
        if (bVal === undefined || bVal === null) return -1
        
        if (typeof aVal === 'string' && typeof bVal === 'string') {
          return sortDirection === 'asc'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal)
        }
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
        }
        
        return 0
      })
    } else {
      // Default sort by timestamp (newest first) - always default to desc
      filtered.sort((a, b) => {
        const aTime = new Date(a.timestamp).getTime()
        const bTime = new Date(b.timestamp).getTime()
        // If no sort field is set, always use desc (newest first)
        const direction = sortField === null ? 'desc' : sortDirection
        return direction === 'asc' ? aTime - bTime : bTime - aTime
      })
    }

    return filtered
  }, [trafficData, searchQuery, methodFilter, statusFilter, sortField, sortDirection])

  // Calculate statistics
  const statistics = useMemo(() => {
    const total = filteredAndSortedData.length
    const uniqueIPs = new Set(filteredAndSortedData.map(t => t.ip)).size
    const methods = filteredAndSortedData.reduce((acc, t) => {
      acc[t.method] = (acc[t.method] || 0) + 1
      return acc
    }, {} as Record<string, number>)
    const topEndpoint = Object.entries(
      filteredAndSortedData.reduce((acc, t) => {
        acc[t.endpoint] = (acc[t.endpoint] || 0) + 1
        return acc
      }, {} as Record<string, number>)
    ).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A'

    return { total, uniqueIPs, methods, topEndpoint }
  }, [filteredAndSortedData])

  const handleSort = (field: keyof TrafficData) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const handleExport = () => {
    const csv = [
      ['IP', 'Method', 'Endpoint', 'Status', 'Size', 'Time'].join(','),
      ...filteredAndSortedData.map(t => [
        t.ip,
        t.method,
        t.endpoint,
        t.status,
        t.size,
        formatTimeIST(t.timestamp)
      ].join(','))
    ].join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `traffic-${new Date().toISOString()}.csv`
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
                <h2 className="text-2xl font-bold mb-2">Traffic Monitor</h2>
                <p className="text-muted-foreground">Live real-time traffic and request monitoring</p>
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
                <p className="text-muted-foreground">Live real-time traffic and request monitoring</p>
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
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Traffic Monitor</h2>
                <p className="text-muted-foreground">Live real-time traffic and request monitoring</p>
              </div>

              {/* Statistics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Total Requests</p>
                  <p className="text-2xl font-bold">{statistics.total.toLocaleString()}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Unique IPs</p>
                  <p className="text-2xl font-bold">{statistics.uniqueIPs.toLocaleString()}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Top Method</p>
                  <p className="text-2xl font-bold">{Object.keys(statistics.methods).sort((a, b) => statistics.methods[b] - statistics.methods[a])[0] || 'N/A'}</p>
                </Card>
                <Card className="p-4">
                  <p className="text-sm text-muted-foreground mb-1">Top Endpoint</p>
                  <p className="text-lg font-bold font-mono truncate" title={statistics.topEndpoint}>{statistics.topEndpoint}</p>
                </Card>
              </div>

              {/* Filters and Search */}
              <Card className="p-4">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search by IP, endpoint, method..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <Select value={methodFilter} onValueChange={setMethodFilter}>
                    <SelectTrigger className="w-full md:w-[180px]">
                      <SelectValue placeholder="Method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Methods</SelectItem>
                      <SelectItem value="GET">GET</SelectItem>
                      <SelectItem value="POST">POST</SelectItem>
                      <SelectItem value="PUT">PUT</SelectItem>
                      <SelectItem value="DELETE">DELETE</SelectItem>
                      <SelectItem value="PATCH">PATCH</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-full md:w-[180px]">
                      <SelectValue placeholder="Status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="2xx">2xx Success</SelectItem>
                      <SelectItem value="4xx">4xx Client Error</SelectItem>
                      <SelectItem value="5xx">5xx Server Error</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button onClick={handleExport} variant="outline">
                    <Download className="mr-2 h-4 w-4" />
                    Export
                  </Button>
                </div>
              </Card>

              {/* Traffic Table */}
              <Card className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold">Traffic Logs ({filteredAndSortedData.length})</h3>
                  </div>
                  <Button
                    onClick={() => {
                      fetchTraffic(false)
                      if (tableContainerRef.current) {
                        tableContainerRef.current.scrollTop = 0
                        shouldAutoScrollRef.current = true
                      }
                    }}
                    variant="outline"
                    size="sm"
                    disabled={isRefreshing}
                    className="gap-2"
                  >
                    <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
                    Latest Logs
                  </Button>
                </div>
                <div className="relative">
                  <div 
                    className="overflow-x-auto overflow-y-auto max-h-[600px]" 
                    ref={tableContainerRef}
                    style={{ scrollBehavior: 'smooth' }}
                  >
                    <Table>
                    <TableHeader>
                      <TableRow className="border-border hover:bg-transparent">
                        <TableHead className="text-muted-foreground cursor-pointer" onClick={() => handleSort('ip')}>
                          <div className="flex items-center gap-2">
                            IP Address
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </TableHead>
                        <TableHead className="text-muted-foreground cursor-pointer" onClick={() => handleSort('method')}>
                          <div className="flex items-center gap-2">
                            Method
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </TableHead>
                        <TableHead className="text-muted-foreground cursor-pointer" onClick={() => handleSort('endpoint')}>
                          <div className="flex items-center gap-2">
                            Endpoint
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </TableHead>
                        <TableHead className="text-muted-foreground cursor-pointer" onClick={() => handleSort('status')}>
                          <div className="flex items-center gap-2">
                            Status
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </TableHead>
                        <TableHead className="text-muted-foreground">Size</TableHead>
                        <TableHead className="text-muted-foreground cursor-pointer" onClick={() => handleSort('time')}>
                          <div className="flex items-center gap-2">
                            Time
                            <ArrowUpDown className="h-3 w-3" />
                          </div>
                        </TableHead>
                        <TableHead className="text-muted-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredAndSortedData.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                            {isLoading ? 'Loading traffic data...' : 'No traffic data available'}
                          </TableCell>
                        </TableRow>
                      ) : (
                        filteredAndSortedData.slice(0, 100).map((request) => (
                          <TableRow 
                            key={request.timestamp} 
                            className="border-border hover:bg-card/50 cursor-pointer"
                            onClick={() => {
                              setSelectedTraffic(request)
                              setShowDetails(true)
                            }}
                          >
                            <TableCell className="font-mono text-sm">{request.ip}</TableCell>
                            <TableCell>
                              <span className="px-2 py-1 bg-muted text-muted-foreground rounded text-xs font-medium">
                                {request.method}
                              </span>
                            </TableCell>
                            <TableCell className="font-mono text-sm max-w-xs truncate" title={request.endpoint}>
                              {request.endpoint}
                            </TableCell>
                            <TableCell>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                request.status >= 200 && request.status < 300 ? 'bg-green-500/10 text-green-600' :
                                request.status >= 400 && request.status < 500 ? 'bg-yellow-500/10 text-yellow-600' :
                                request.status >= 500 ? 'bg-red-500/10 text-red-600' :
                                'bg-muted text-muted-foreground'
                              }`}>
                                {request.status}
                              </span>
                            </TableCell>
                            <TableCell className="text-sm">{request.size}</TableCell>
                            <TableCell className="text-sm">{formatTimeIST(request.timestamp)}</TableCell>
                            <TableCell>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setSelectedTraffic(request)
                                  setShowDetails(true)
                                }}
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                  </div>
                  {/* Scroll to top button */}
                  {showScrollToTop && (
                    <Button
                      onClick={() => {
                        if (tableContainerRef.current) {
                          tableContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' })
                          shouldAutoScrollRef.current = true
                        }
                      }}
                      className="absolute bottom-4 right-4 rounded-full shadow-lg"
                      size="sm"
                      variant="default"
                    >
                      <ArrowUp className="h-4 w-4 mr-1" />
                      Top
                    </Button>
                  )}
                </div>
              </Card>

              {/* Details Dialog */}
              <Dialog open={showDetails} onOpenChange={setShowDetails}>
                <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>Request Details</DialogTitle>
                    <DialogDescription>Full details of the selected request</DialogDescription>
                  </DialogHeader>
                  {selectedTraffic && (
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-semibold mb-2">Basic Information</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">IP Address:</span>
                            <p className="font-mono">{selectedTraffic.ip}</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Method:</span>
                            <p>{selectedTraffic.method}</p>
                          </div>
                          <div className="col-span-2">
                            <span className="text-muted-foreground">Endpoint:</span>
                            <p className="font-mono break-all">{selectedTraffic.endpoint}</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Status:</span>
                            <p>{selectedTraffic.status}</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Size:</span>
                            <p>{selectedTraffic.size}</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Time:</span>
                            <p>{formatTimeIST(selectedTraffic.timestamp)}</p>
                          </div>
                          {selectedTraffic.userAgent && (
                            <div className="col-span-2">
                              <span className="text-muted-foreground">User Agent:</span>
                              <p className="font-mono text-xs break-all">{selectedTraffic.userAgent}</p>
                            </div>
                          )}
                        </div>
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
