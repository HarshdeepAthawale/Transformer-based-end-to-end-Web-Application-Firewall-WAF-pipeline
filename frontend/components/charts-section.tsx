'use client'

import { useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from '@/components/ui/chart'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useChartData } from '@/hooks/use-chart-data'
import { formatTimeIST, CHART_TIME_RANGES } from '@/lib/chart-utils'

const requestVolumeChartConfig = {
  requests: { label: 'Total Requests', color: 'var(--chart-1)' },
  blocked: { label: 'Blocked Threats', color: 'var(--chart-2)' },
} satisfies ChartConfig

const topThreatChartConfig = {
  count: { label: 'Blocked', color: 'var(--chart-1)' },
  name: { label: 'Threat Type' },
} satisfies ChartConfig

interface ChartsSectionProps {
  timeRange: string
  onTimeRangeChange?: (range: string) => void
}

export function ChartsSection({ timeRange, onTimeRangeChange }: ChartsSectionProps) {
  const router = useRouter()
  const { requestData, topThreatTypes, isLoading, error } = useChartData(timeRange)

  const formattedRequestData = useMemo(
    () =>
      requestData.map((point) => ({
        ...point,
        timeFormatted: formatTimeIST(point.time || ''),
      })),
    [requestData]
  )

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div
          className="rounded-md p-4 md:p-6 border-2"
          style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
        >
          <div className="flex items-center justify-center py-16">
            <div
              className="animate-spin rounded-full h-8 w-8 border-b-2"
              style={{ borderColor: 'var(--positivus-green)' }}
            />
            <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>
              Loading chart data...
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div
          className="rounded-md p-4 md:p-6 border-2"
          style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
        >
          <div className="flex items-center justify-center py-16">
            <div className="text-destructive">Failed to load chart data: {error}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Request Volume & Threats */}
        <div
          className="rounded-md p-4 md:p-6 xl:col-span-2 border-2"
          style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
        >
          <div className="mb-4 md:mb-6 flex items-center gap-2 flex-wrap sm:flex-row">
            <div className="grid flex-1 gap-1 min-w-0">
              <h2
                className="text-lg font-semibold"
                style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
              >
                Request Volume & Threats
              </h2>
              <p className="text-sm text-muted-foreground hidden sm:block">
                Request volume and blocked threats over time
              </p>
            </div>
            {onTimeRangeChange && (
              <Select value={timeRange} onValueChange={onTimeRangeChange}>
                <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto shrink-0" aria-label="Select time range">
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
            )}
          </div>
          {formattedRequestData.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md"
              style={{ borderColor: 'var(--positivus-gray)' }}
            >
              <p className="text-sm mb-2" style={{ color: 'var(--positivus-gray-dark)' }}>
                No data available
              </p>
              <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                Waiting for traffic data...
              </p>
            </div>
          ) : (
            <ChartContainer config={requestVolumeChartConfig} className="aspect-auto h-[250px] w-full">
              <AreaChart data={formattedRequestData}>
                <defs>
                  <linearGradient id="fillRequests" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-requests)" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="var(--color-requests)" stopOpacity={0.1} />
                  </linearGradient>
                  <linearGradient id="fillBlocked" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-blocked)" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="var(--color-blocked)" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="timeFormatted"
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  minTickGap={32}
                  interval="preserveStartEnd"
                />
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent labelFormatter={(value) => value} indicator="dot" />}
                />
                <Area
                  dataKey="blocked"
                  type="natural"
                  fill="url(#fillBlocked)"
                  stroke="var(--color-blocked)"
                />
                <Area
                  dataKey="requests"
                  type="natural"
                  fill="url(#fillRequests)"
                  stroke="var(--color-requests)"
                />
                <ChartLegend content={<ChartLegendContent />} />
              </AreaChart>
            </ChartContainer>
          )}
        </div>

        {/* Top 10 Threat Types */}
        <div
          className="rounded-md p-4 md:p-6 xl:col-span-2 border-2"
          style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
        >
          <div className="flex items-center justify-between gap-4 mb-4 md:mb-6">
            <div className="grid flex-1 gap-1 min-w-0">
              <h2
                className="text-lg font-semibold"
                style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
              >
                Top 10 Threat Types
              </h2>
              <p className="text-sm text-muted-foreground hidden sm:block">
                Most frequently blocked attack types
              </p>
            </div>
            <button
              onClick={() => router.push('/threats')}
              className="px-3 py-1 text-xs rounded-lg border-2 shrink-0 hover:bg-accent"
              style={{
                backgroundColor: 'var(--positivus-gray)',
                borderColor: 'var(--positivus-gray)',
                color: 'var(--positivus-gray-dark)',
              }}
            >
              Details
            </button>
          </div>
          {topThreatTypes.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md text-center"
              style={{ borderColor: 'var(--positivus-gray)' }}
            >
              <p className="text-sm font-medium mb-1" style={{ color: 'var(--positivus-gray-dark)' }}>
                No threats detected
              </p>
              <p className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                Top attack types will appear when threats are blocked
              </p>
            </div>
          ) : (
            <ChartContainer
              config={topThreatChartConfig}
              className="aspect-auto w-full"
              style={{ height: Math.max(250, topThreatTypes.length * 36) }}
            >
              <BarChart
                accessibilityLayer
                data={topThreatTypes}
                layout="vertical"
                margin={{ left: 8, right: 12, top: 8, bottom: 8 }}
              >
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} tickMargin={8} allowDecimals={false} />
                <YAxis type="category" dataKey="name" tickLine={false} axisLine={false} tickMargin={8} width={140} />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      className="w-[180px]"
                      nameKey="count"
                      labelFormatter={(value) => value}
                      formatter={(value) => `${Number(value).toLocaleString()} blocked`}
                    />
                  }
                />
                <Bar dataKey="count" fill="var(--color-count)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ChartContainer>
          )}
        </div>
      </div>
    </div>
  )
}
