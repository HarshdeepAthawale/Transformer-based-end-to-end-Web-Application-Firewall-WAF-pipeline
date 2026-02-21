'use client'

import { useEffect, useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/components/ui/chart'
import { AreaChart, Area, XAxis, CartesianGrid } from 'recharts'
import { Loader2 } from 'lucide-react'
import { dashboardApi, type DashboardUnifiedData, type SecurityEventData } from '@/lib/api'
import { useTimezone } from '@/contexts/timezone-context'
import { formatChartAxisLabel, formatTimeLocal, CHART_TIME_RANGES } from '@/lib/chart-utils'

const DASHBOARD_RANGES = [
  { value: '1h', label: '1h' },
  { value: '6h', label: '6h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: '30d', label: '30d' },
] as const

const EVENT_TYPE_OPTIONS = [
  { value: '', label: 'All types' },
  { value: 'waf_block', label: 'WAF block' },
  { value: 'rate_limit', label: 'Rate limit' },
  { value: 'ddos_burst', label: 'DDoS' },
  { value: 'bot_block', label: 'Bot block' },
  { value: 'upload_scan_infected', label: 'Upload infected' },
  { value: 'credential_leak_block', label: 'Credential leak' },
  { value: 'firewall_ai_prompt_block', label: 'Firewall AI' },
] as const

/** Merge chart series into one timeline: each bucket has keys seriesName: count. */
function mergeSeriesIntoTimeline(series: { name: string; data: { time: string; count: number }[] }[]) {
  const map = new Map<string, Record<string, number>>()
  for (const s of series) {
    for (const point of s.data) {
      const t = point.time
      if (!map.has(t)) map.set(t, { time: t, timeFormatted: t })
      const row = map.get(t)!
      row[s.name] = point.count
    }
  }
  return Array.from(map.values()).sort((a, b) => (a.time < b.time ? -1 : 1))
}

function buildChartConfig(series: { name: string }[]): ChartConfig {
  const config: ChartConfig = {}
  series.forEach((s, i) => {
    ;(config as Record<string, { label: string; color: string }>)[s.name] = {
      label: s.name.replace(/_/g, ' '),
      color: `var(--chart-${(i % 5) + 1})`,
    }
  })
  return config
}

interface UnifiedSecuritySectionProps {
  timeRange: string
  onTimeRangeChange?: (range: string) => void
}

export function UnifiedSecuritySection({ timeRange, onTimeRangeChange }: UnifiedSecuritySectionProps) {
  const [data, setData] = useState<DashboardUnifiedData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [eventTypeFilter, setEventTypeFilter] = useState('')
  const { timezone } = useTimezone()

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    dashboardApi
      .getDashboardUnified(timeRange, 50, eventTypeFilter || undefined)
      .then((res) => {
        if (!cancelled && res.success && res.data) setData(res.data)
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message || 'Failed to load dashboard')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [timeRange, eventTypeFilter])

  const overview = data?.overview
  const series = data?.charts?.series ?? []
  const recentEvents = data?.recent_events ?? []
  const chartData = mergeSeriesIntoTimeline(series).map((row) => ({
    ...row,
    timeFormatted: formatChartAxisLabel(row.time, timeRange, timezone),
  }))
  const unifiedChartConfig = buildChartConfig(series)

  return (
    <div className="space-y-6">
      <div>
        <h2
          className="text-xl font-bold"
          style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
        >
          Unified security
        </h2>
        <p className="text-sm text-muted-foreground">
          Aggregated WAF, rate limit, DDoS, bot, upload scan, credential leak, and Firewall-for-AI from security events.
        </p>
      </div>

      {error && (
        <Card className="p-4 border-destructive" style={{ backgroundColor: 'rgba(var(--destructive), 0.1)' }}>
          <p className="text-sm text-destructive">{error}</p>
        </Card>
      )}

      {/* Overview cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {loading && !overview ? (
          <div className="col-span-full flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--positivus-green)' }} />
          </div>
        ) : overview ? (
          <>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>WAF blocks</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.waf_block_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Rate limit</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.rate_limit_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>DDoS</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.ddos_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Bot blocks</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.bot_block_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Upload infected</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.upload_scan_infected_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Credential leak</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.credential_leak_block_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Firewall AI</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.firewall_ai_block_count}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Avg attack score</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.avg_attack_score ?? '—'}</p>
            </Card>
            <Card className="p-4 border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Avg bot score</p>
              <p className="text-xl font-semibold" style={{ color: 'var(--positivus-black)' }}>{overview.avg_bot_score ?? '—'}</p>
            </Card>
          </>
        ) : null}
      </div>

      {/* Time-series chart */}
      <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
              Events over time
            </CardTitle>
            <CardDescription>By category (hourly)</CardDescription>
          </div>
          {onTimeRangeChange && (
            <Select value={timeRange} onValueChange={onTimeRangeChange}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {DASHBOARD_RANGES.map((r) => (
                  <SelectItem key={r.value} value={r.value}>{r.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </CardHeader>
        <CardContent>
          {chartData.length === 0 ? (
            <div className="h-[220px] flex items-center justify-center border-2 border-dashed rounded-md" style={{ borderColor: 'var(--positivus-gray)' }}>
              <p className="text-sm text-muted-foreground">No event data in this range</p>
            </div>
          ) : (
            <ChartContainer config={unifiedChartConfig} className="h-[220px] w-full">
              <AreaChart data={chartData}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="timeFormatted" tickLine={false} axisLine={false} tickMargin={8} minTickGap={40} />
                <ChartTooltip content={<ChartTooltipContent />} />
                {series.map((s, i) => (
                  <Area
                    key={s.name}
                    type="monotone"
                    dataKey={s.name}
                    stackId="1"
                    stroke={`var(--chart-${(i % 5) + 1})`}
                    fill={`var(--chart-${(i % 5) + 1})`}
                    fillOpacity={0.6}
                  />
                ))}
                <ChartLegend content={<ChartLegendContent />} />
              </AreaChart>
            </ChartContainer>
          )}
        </CardContent>
      </Card>

      {/* Recent events table */}
      <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
              Recent events
            </CardTitle>
            <CardDescription>Security events in selected range</CardDescription>
          </div>
          <Select value={eventTypeFilter} onValueChange={setEventTypeFilter}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Event type" />
            </SelectTrigger>
            <SelectContent>
              {EVENT_TYPE_OPTIONS.map((o) => (
                <SelectItem key={o.value || 'all'} value={o.value}>{o.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent>
          {loading && recentEvents.length === 0 ? (
            <div className="h-[200px] flex items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--positivus-green)' }} />
            </div>
          ) : recentEvents.length === 0 ? (
            <div className="h-[200px] flex items-center justify-center border-2 border-dashed rounded-md" style={{ borderColor: 'var(--positivus-gray)' }}>
              <p className="text-sm text-muted-foreground">No events</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow style={{ borderColor: 'var(--positivus-gray)' }}>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Time</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Type</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>IP</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Path</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Attack</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Bot</TableHead>
                    <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Details</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {recentEvents.map((ev: SecurityEventData) => (
                    <TableRow key={ev.id} style={{ borderColor: 'var(--positivus-gray)' }}>
                      <TableCell className="text-xs" style={{ color: 'var(--positivus-black)' }}>
                        {formatTimeLocal(ev.timestamp ?? '', timezone)}
                      </TableCell>
                      <TableCell className="text-xs font-mono" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.event_type}</TableCell>
                      <TableCell className="text-xs font-mono" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.ip}</TableCell>
                      <TableCell className="text-xs font-mono max-w-[180px] truncate" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.path ?? '—'}</TableCell>
                      <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{ev.attack_score ?? '—'}</TableCell>
                      <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{ev.bot_score ?? '—'}</TableCell>
                      <TableCell className="text-xs max-w-[200px] truncate" style={{ color: 'var(--positivus-gray-dark)' }} title={ev.details ?? ''}>{ev.details ?? '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
