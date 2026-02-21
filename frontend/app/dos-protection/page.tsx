'use client'

import { useEffect, useState, useMemo } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/components/ui/chart'
import { AreaChart, Area, XAxis, CartesianGrid } from 'recharts'
import { Gauge, Ban, ShieldCheck, AlertCircle, Loader2, ShieldX, ShieldAlert, Activity } from 'lucide-react'
import { eventsApi, ipApi, ddosApi, type SecurityEventData, type AdaptiveDdosStats } from '@/lib/api'
import { useTimezone } from '@/contexts/timezone-context'
import { formatTimeLocal } from '@/lib/chart-utils'

const MAX_CHART_POINTS = 60

/** Build a full timeline of buckets for the range so the area chart renders continuous "up down" shape (like Request Volume & Threats). */
function buildFullTimelineBuckets(timeRange: string): string[] {
  const now = new Date()
  const buckets: string[] = []
  const pad = (n: number) => String(n).padStart(2, '0')
  if (timeRange === '1h') {
    for (let i = 1; i >= 0; i--) {
      const d = new Date(now)
      d.setHours(d.getHours() - i)
      d.setMinutes(0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:00:00`)
    }
  } else if (timeRange === '6h') {
    for (let i = 5; i >= 0; i--) {
      const d = new Date(now)
      d.setHours(d.getHours() - i)
      d.setMinutes(0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:00:00`)
    }
  } else if (timeRange === '24h') {
    for (let i = 23; i >= 0; i--) {
      const d = new Date(now)
      d.setHours(d.getHours() - i)
      d.setMinutes(0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:00:00`)
    }
  } else if (timeRange === '7d') {
    for (let i = 6; i >= 0; i--) {
      const d = new Date(now)
      d.setDate(d.getDate() - i)
      d.setHours(0, 0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} 00:00:00`)
    }
  } else if (timeRange === '30d') {
    for (let i = 29; i >= 0; i--) {
      const d = new Date(now)
      d.setDate(d.getDate() - i)
      d.setHours(0, 0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} 00:00:00`)
    }
  } else if (timeRange === '90d') {
    for (let i = 89; i >= 0; i--) {
      const d = new Date(now)
      d.setDate(d.getDate() - i)
      d.setHours(0, 0, 0, 0)
      buckets.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} 00:00:00`)
    }
  }
  return buckets
}

const securityEventsChartConfig = {
  rateLimit: { label: 'Rate Limit Hits', color: 'var(--chart-1)' },
  ddos: { label: 'DDoS Blocks', color: 'var(--chart-2)' },
  blacklist: { label: 'Blacklist Blocks', color: 'var(--chart-3)' },
} satisfies ChartConfig

const CHART_TIME_RANGES = [
  { value: '1h', label: 'Last 1 hour' },
  { value: '6h', label: 'Last 6 hours' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: '90d', label: 'Last 3 months' },
] as const

export default function DosProtectionPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [overviewData, setOverviewData] = useState<{
    stats: { rate_limit_count: number; ddos_count: number; blacklist_count?: number }
    chart_rate_limit: { time: string; count: number }[]
    chart_ddos: { time: string; count: number }[]
    chart_blacklist?: { time: string; count: number }[]
    recent_rate_limit: SecurityEventData[]
    recent_ddos: SecurityEventData[]
    recent_blacklist?: SecurityEventData[]
  } | null>(null)
  const [wafEvents, setWafEvents] = useState<SecurityEventData[]>([])
  const [wafBlockCount, setWafBlockCount] = useState(0)
  const [activeTab, setActiveTab] = useState<'rate_limit' | 'ddos' | 'blacklist' | 'waf'>('rate_limit')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [blacklistLoading, setBlacklistLoading] = useState<string | null>(null)
  const [adaptiveDdos, setAdaptiveDdos] = useState<AdaptiveDdosStats | null>(null)
  const { timezone } = useTimezone()

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const [res, wafRes, statsRes, adaptiveRes] = await Promise.all([
          eventsApi.getDosOverview(timeRange, 100),
          eventsApi.getWafEvents(timeRange, 100),
          eventsApi.getStats(timeRange),
          ddosApi.getAdaptiveDdosStats(),
        ])
        if (res.success && res.data) {
          setOverviewData(res.data)
        }
        if (wafRes.success && wafRes.data) {
          setWafEvents(wafRes.data)
        }
        if (statsRes.success && statsRes.data) {
          setWafBlockCount(statsRes.data.waf_block_count ?? 0)
        }
        if (adaptiveRes.success && adaptiveRes.data) {
          setAdaptiveDdos(adaptiveRes.data)
        }
      } catch (err: unknown) {
        const e = err as { isNetworkError?: boolean; message?: string }
        if (!e?.isNetworkError) {
          setError(e?.message || 'Failed to load DoS/DDoS data')
        }
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [timeRange])

  const securityEventsData = useMemo(() => {
    if (!overviewData) return []
    const isDaily = ['7d', '30d', '90d'].includes(timeRange)
    const toKey = (t: string) => {
      if (!t) return ''
      if (isDaily) return t.slice(0, 10) + ' 00:00:00' // YYYY-MM-DD 00:00:00
      return t
    }
    const rateLimitMap = new Map<string, number>()
    overviewData.chart_rate_limit.forEach((p) => {
      const k = toKey(p.time || '')
      if (k) rateLimitMap.set(k, (rateLimitMap.get(k) ?? 0) + (p.count ?? 0))
    })
    const ddosMap = new Map<string, number>()
    overviewData.chart_ddos.forEach((p) => {
      const k = toKey(p.time || '')
      if (k) ddosMap.set(k, (ddosMap.get(k) ?? 0) + (p.count ?? 0))
    })
    const blacklistMap = new Map<string, number>()
    ;(overviewData.chart_blacklist ?? []).forEach((p) => {
      const k = toKey(p.time || '')
      if (k) blacklistMap.set(k, (blacklistMap.get(k) ?? 0) + (p.count ?? 0))
    })
    const buckets = buildFullTimelineBuckets(timeRange)
    if (buckets.length === 0) {
      const allTimes = new Set([...rateLimitMap.keys(), ...ddosMap.keys(), ...blacklistMap.keys()])
      return Array.from(allTimes)
        .filter(Boolean)
        .sort()
        .map((time) => ({
          time,
          timeFormatted: formatTimeLocal(time, timezone),
          rateLimit: rateLimitMap.get(time) ?? 0,
          ddos: ddosMap.get(time) ?? 0,
          blacklist: blacklistMap.get(time) ?? 0,
        }))
        .slice(-MAX_CHART_POINTS)
    }
    return buckets.map((time) => ({
      time,
      timeFormatted: formatTimeLocal(time, timezone),
      rateLimit: rateLimitMap.get(time) ?? 0,
      ddos: ddosMap.get(time) ?? 0,
      blacklist: blacklistMap.get(time) ?? 0,
    }))
  }, [overviewData, timezone, timeRange])

  const handleAddToBlacklist = async (ip: string, reason: string) => {
    setBlacklistLoading(ip)
    setError(null)
    try {
      await ipApi.addToBlacklist(ip, reason, undefined, 'dos_protection')
      // Success: IP is now blacklisted and enforced at gateway (Redis-synced)
      // Optional: refetch to refresh view
      const res = await eventsApi.getDosOverview(timeRange, 100)
      if (res.success && res.data) setOverviewData(res.data)
    } catch (err: unknown) {
      const e = err as { isNetworkError?: boolean; message?: string }
      if (!e?.isNetworkError) {
        setError(e?.message || 'Failed to block IP')
      }
    } finally {
      setBlacklistLoading(null)
    }
  }

  const rateLimitCount = overviewData?.stats?.rate_limit_count ?? 0
  const ddosCount = overviewData?.stats?.ddos_count ?? 0
  const blacklistCount = overviewData?.stats?.blacklist_count ?? 0
  const rateLimitEvents = overviewData?.recent_rate_limit ?? []
  const ddosEvents = overviewData?.recent_ddos ?? []
  const blacklistEvents = overviewData?.recent_blacklist ?? []
  const currentEvents =
    activeTab === 'rate_limit'
      ? rateLimitEvents
      : activeTab === 'ddos'
        ? ddosEvents
        : activeTab === 'waf'
          ? wafEvents
          : blacklistEvents

  return (
    <div className="flex h-screen" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto p-6">
          <ErrorBoundary>
            <div className="max-w-7xl mx-auto space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                  DoS/DDoS Protection
                </h2>
                <p className="text-muted-foreground">
                  Monitor rate limit hits and DDoS blocks, analyze events, and apply mitigation with best practices
                </p>
              </div>

              {error && (
                <Card className="p-4 border-destructive" style={{ backgroundColor: 'rgba(var(--destructive), 0.1)' }}>
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                </Card>
              )}

              {/* Metric cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <Gauge size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      Rate Limit Hits
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : rateLimitCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <Ban size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      DDoS Blocks
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : ddosCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <ShieldX size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      Blacklist Blocks
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : blacklistCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <ShieldAlert size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      WAF Blocks
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : wafBlockCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
              </div>

              {/* Adaptive DDoS */}
              <Card
                className="border-2"
                style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
              >
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Activity size={20} style={{ color: 'var(--positivus-green)' }} />
                    <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Adaptive DDoS
                    </CardTitle>
                  </div>
                  <CardDescription>
                    Auto-tuned burst threshold from traffic baseline (P{adaptiveDdos?.config?.percentile ?? 95}); gateway reads from Redis.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {!adaptiveDdos ? (
                    <p className="text-muted-foreground text-sm">Loading…</p>
                  ) : !adaptiveDdos.enabled ? (
                    <p className="text-muted-foreground text-sm">Adaptive DDoS is disabled. Enable ADAPTIVE_DDOS_ENABLED in backend and gateway to auto-tune the burst threshold.</p>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Current threshold</p>
                        <p className="text-lg font-semibold" style={{ color: 'var(--positivus-black)' }}>
                          {adaptiveDdos.current_threshold != null ? adaptiveDdos.current_threshold : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Baseline (P{adaptiveDdos.config?.percentile ?? 95})</p>
                        <p className="text-lg font-semibold" style={{ color: 'var(--positivus-black)' }}>
                          {adaptiveDdos.baseline_percentile_value != null ? adaptiveDdos.baseline_percentile_value : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Last updated</p>
                        <p className="font-mono text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                          {adaptiveDdos.last_updated ? formatTimeLocal(adaptiveDdos.last_updated, timezone) : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Config</p>
                        <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                          ×{adaptiveDdos.config?.multiplier ?? '—'} min {adaptiveDdos.config?.threshold_min ?? '—'} max {adaptiveDdos.config?.threshold_max ?? '—'} · {adaptiveDdos.learning_window_minutes}m window
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Chart */}
              <Card
                className="pt-0 border-2"
                style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
              >
                <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row" style={{ borderColor: 'var(--positivus-gray)' }}>
                  <div className="grid flex-1 gap-1">
                    <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Rate Limit Hits, DDoS Blocks & Blacklist Blocks
                    </CardTitle>
                    <CardDescription>
                      Compare rate limit hits, DDoS blocks, and blacklist blocks over the selected period
                    </CardDescription>
                  </div>
                  <Select value={timeRange} onValueChange={setTimeRange}>
                    <SelectTrigger
                      className="w-[160px] rounded-lg sm:ml-auto border-2"
                      style={{ borderColor: 'var(--positivus-gray)' }}
                      aria-label="Select time range"
                    >
                      <SelectValue placeholder="Time range" />
                    </SelectTrigger>
                    <SelectContent className="rounded-xl">
                      {CHART_TIME_RANGES.map((r) => (
                        <SelectItem key={r.value} value={r.value} className="rounded-lg">
                          {r.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardHeader>
                <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
                  {loading || securityEventsData.length === 0 ? (
                    <div
                      className="flex flex-col items-center justify-center h-[250px] border-2 border-dashed rounded-md"
                      style={{ borderColor: 'var(--positivus-gray)' }}
                    >
                      <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                        {loading ? 'Loading...' : 'No rate limit, DDoS or blacklist events yet'}
                      </p>
                    </div>
                  ) : (
                    <ChartContainer config={securityEventsChartConfig} className="aspect-auto h-[250px] w-full">
                      <AreaChart data={securityEventsData}>
                        <defs>
                          <linearGradient id="fillRateLimitDos" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--color-rateLimit)" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="var(--color-rateLimit)" stopOpacity={0.1} />
                          </linearGradient>
                          <linearGradient id="fillDdosDos" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--color-ddos)" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="var(--color-ddos)" stopOpacity={0.1} />
                          </linearGradient>
                          <linearGradient id="fillBlacklistDos" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--color-blacklist)" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="var(--color-blacklist)" stopOpacity={0.1} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid vertical={false} />
                        <XAxis
                          dataKey="time"
                          tickLine={false}
                          axisLine={false}
                          tickMargin={8}
                          minTickGap={24}
                          tickFormatter={(value) => {
                            try {
                              const date = new Date(value)
                              if (timeRange === '1h' || timeRange === '6h' || timeRange === '24h') {
                                return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
                              }
                              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
                            } catch {
                              return value
                            }
                          }}
                        />
                        <ChartTooltip
                          cursor={false}
                          content={
                            <ChartTooltipContent
                              labelFormatter={(value) => {
                                try {
                                  const d = new Date(value)
                                  if (timeRange === '1h' || timeRange === '6h' || timeRange === '24h') {
                                    return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
                                  }
                                  return d.toLocaleDateString('en-US', {
                                    month: 'short',
                                    day: 'numeric',
                                    hour: 'numeric',
                                    minute: '2-digit',
                                  })
                                } catch {
                                  return value
                                }
                              }}
                              indicator="dot"
                            />
                          }
                        />
                        <Area
                          dataKey="rateLimit"
                          type="natural"
                          fill="url(#fillRateLimitDos)"
                          stroke="var(--color-rateLimit)"
                          stackId="a"
                          radius={4}
                          dot={false}
                          isAnimationActive={true}
                        />
                        <Area
                          dataKey="ddos"
                          type="natural"
                          fill="url(#fillDdosDos)"
                          stroke="var(--color-ddos)"
                          stackId="a"
                          radius={0}
                          dot={false}
                          isAnimationActive={true}
                        />
                        <Area
                          dataKey="blacklist"
                          type="natural"
                          fill="url(#fillBlacklistDos)"
                          stroke="var(--color-blacklist)"
                          stackId="a"
                          radius={4}
                          dot={false}
                          isAnimationActive={true}
                        />
                        <ChartLegend content={<ChartLegendContent />} />
                      </AreaChart>
                    </ChartContainer>
                  )}
                </CardContent>
              </Card>

              {/* Event tables */}
              <Card className="border-2 overflow-hidden" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                <div className="flex gap-2 p-4 border-b" style={{ borderColor: 'var(--positivus-gray)' }}>
                  <Button
                    variant={activeTab === 'rate_limit' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveTab('rate_limit')}
                    className="gap-2"
                  >
                    <Gauge size={16} />
                    Rate Limit Events ({rateLimitEvents.length})
                  </Button>
                  <Button
                    variant={activeTab === 'ddos' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveTab('ddos')}
                    className="gap-2"
                  >
                    <Ban size={16} />
                    DDoS Events ({ddosEvents.length})
                  </Button>
                  <Button
                    variant={activeTab === 'blacklist' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveTab('blacklist')}
                    className="gap-2"
                  >
                    <ShieldX size={16} />
                    Blacklist Events ({blacklistEvents.length})
                  </Button>
                  <Button
                    variant={activeTab === 'waf' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setActiveTab('waf')}
                    className="gap-2"
                  >
                    <ShieldAlert size={16} />
                    WAF Events ({wafEvents.length})
                  </Button>
                </div>
                <div className="overflow-x-auto">
                  {loading ? (
                    <div className="flex items-center justify-center py-16">
                      <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                    </div>
                  ) : currentEvents.length === 0 ? (
                    <div className="py-12 text-center text-muted-foreground">
                      No {activeTab === 'rate_limit' ? 'rate limit' : activeTab === 'ddos' ? 'DDoS' : activeTab === 'waf' ? 'WAF' : 'blacklist'} events in the selected period
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Time</TableHead>
                          <TableHead>IP</TableHead>
                          <TableHead>Method</TableHead>
                          <TableHead>Path</TableHead>
                          {activeTab === 'waf' && <TableHead>Attack Score</TableHead>}
                          <TableHead>Details</TableHead>
                          <TableHead className="w-[120px]">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {currentEvents.map((ev) => (
                          <TableRow key={ev.id}>
                            <TableCell className="text-sm text-muted-foreground">
                              {new Date(ev.timestamp).toLocaleString()}
                            </TableCell>
                            <TableCell className="font-mono text-sm">{ev.ip}</TableCell>
                            <TableCell className="text-sm">{ev.method || '-'}</TableCell>
                            <TableCell className="text-sm max-w-[200px] truncate">{ev.path || '-'}</TableCell>
                            {activeTab === 'waf' && (
                              <TableCell>
                                {ev.attack_score != null ? (
                                  <span
                                    className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium"
                                    style={{
                                      backgroundColor:
                                        ev.attack_score >= 70
                                          ? 'hsl(0 84% 60% / 0.15)'
                                          : ev.attack_score >= 30
                                            ? 'hsl(45 93% 47% / 0.15)'
                                            : 'hsl(142 76% 36% / 0.15)',
                                      color:
                                        ev.attack_score >= 70
                                          ? 'hsl(0 84% 40%)'
                                          : ev.attack_score >= 30
                                            ? 'hsl(45 93% 30%)'
                                            : 'hsl(142 76% 26%)',
                                    }}
                                  >
                                    {ev.attack_score}
                                    <span className="ml-1">
                                      {ev.attack_score >= 70 ? 'High' : ev.attack_score >= 30 ? 'Medium' : 'Low'}
                                    </span>
                                  </span>
                                ) : (
                                  <span className="text-muted-foreground">&mdash;</span>
                                )}
                              </TableCell>
                            )}
                            <TableCell className="text-sm max-w-[200px] truncate text-muted-foreground">
                              {ev.details || '-'}
                            </TableCell>
                            <TableCell>
                              {activeTab === 'blacklist' ? (
                                <span className="text-xs text-muted-foreground">Already blocked</span>
                              ) : (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="gap-1"
                                  disabled={blacklistLoading === ev.ip}
                                  onClick={() => handleAddToBlacklist(ev.ip, `${activeTab === 'waf' ? 'WAF' : 'DoS'} event: ${ev.event_type}`)}
                                >
                                  {blacklistLoading === ev.ip ? (
                                    <Loader2 className="h-3 w-3 animate-spin" />
                                  ) : (
                                    <Ban className="h-3 w-3" />
                                  )}
                                  Block IP
                                </Button>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </div>
              </Card>

              {/* Best practices */}
              <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                <div className="p-4 border-b" style={{ borderColor: 'var(--positivus-gray)' }}>
                  <h3 className="text-lg font-semibold flex items-center gap-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    <ShieldCheck size={20} style={{ color: 'var(--positivus-green)' }} />
                    Best Practices for DoS/DDoS Mitigation
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Industry-recommended approaches to analyze and mitigate denial-of-service attacks
                  </p>
                </div>
                <Accordion type="multiple" className="w-full">
                  <AccordionItem value="rate-limiting" className="border-b px-4" style={{ borderColor: 'var(--positivus-gray)' }}>
                    <AccordionTrigger>Rate Limiting Best Practices</AccordionTrigger>
                    <AccordionContent>
                      <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                        <li>Use sliding-window or token-bucket algorithms for accurate throttling</li>
                        <li>Configure per-IP limits (e.g. 120 req/min) and burst allowance for legitimate spikes</li>
                        <li>Return 429 Too Many Requests with Retry-After header when limits are exceeded</li>
                        <li>Decide fail-open vs fail-closed when Redis or rate limit backend is unavailable</li>
                        <li>Consider per-endpoint limits for sensitive routes (login, API keys, etc.)</li>
                      </ul>
                    </AccordionContent>
                  </AccordionItem>
                  <AccordionItem value="ddos" className="border-b px-4" style={{ borderColor: 'var(--positivus-gray)' }}>
                    <AccordionTrigger>DDoS Mitigation Best Practices</AccordionTrigger>
                    <AccordionContent>
                      <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                        <li>Enforce max request body size before reading the body to protect against size-based attacks</li>
                        <li>Use burst detection (e.g. 50 requests in 5 seconds) to catch rapid-fire traffic</li>
                        <li>Set appropriate block duration for abusive IPs (e.g. 60 seconds)</li>
                        <li>Use layered defense: edge/CDN, WAF, then application</li>
                        <li>Monitor and tune thresholds based on baseline traffic patterns</li>
                      </ul>
                    </AccordionContent>
                  </AccordionItem>
                  <AccordionItem value="config" className="px-4">
                    <AccordionTrigger>Configuration Reference</AccordionTrigger>
                    <AccordionContent>
                      <p className="text-sm text-muted-foreground mb-3">
                        Rate limit and DDoS protection are configured via environment variables on the gateway. Key settings:
                      </p>
                      <div className="text-sm space-y-2 font-mono bg-muted/30 p-4 rounded-md">
                        <p><strong>Rate limiting:</strong> RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS_PER_MINUTE (120), RATE_LIMIT_WINDOW_SECONDS (60), RATE_LIMIT_BURST (20)</p>
                        <p><strong>DDoS:</strong> DDOS_ENABLED, DDOS_MAX_BODY_BYTES (10MB), DDOS_BURST_THRESHOLD (50), DDOS_BURST_WINDOW_SECONDS (5), DDOS_BLOCK_DURATION_SECONDS (60)</p>
                      </div>
                      <p className="text-sm text-muted-foreground mt-3">
                        See docs/rate-limiting.md and docs/ddos-protection.md for full documentation.
                      </p>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </Card>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
