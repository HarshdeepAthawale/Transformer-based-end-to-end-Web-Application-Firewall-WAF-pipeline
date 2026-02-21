'use client'

import { useState, useEffect, useMemo } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { GeoAttackMap } from '@/components/geo-attack-map'
import { geoApi, GeoRule, GeoStats } from '@/lib/api'
import { Globe, Plus, Search, MapPin, AlertCircle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from '@/components/ui/chart'

const geoStatsChartConfig = {
  total_requests: { label: 'Total Requests', color: 'var(--chart-1)' },
  blocked_requests: { label: 'Blocked', color: 'var(--chart-2)' },
} satisfies ChartConfig

const topThreatCountriesChartConfig = {
  count: { label: 'Threats', color: 'var(--chart-1)' },
  name: { label: 'Country' },
} satisfies ChartConfig

const STATS_RANGE_OPTIONS = [
  { value: '1h', label: 'Last 1 hour' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
] as const
type StatsRange = (typeof STATS_RANGE_OPTIONS)[number]['value']

export default function GeoRulesPage() {
  const [rules, setRules] = useState<GeoRule[]>([])
  const [stats, setStats] = useState<GeoStats[]>([])
  const [statsRange, setStatsRange] = useState<StatsRange>('24h')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newRuleType, setNewRuleType] = useState<'allow' | 'deny'>('deny')
  const [newCountryCode, setNewCountryCode] = useState('')
  const [newCountryName, setNewCountryName] = useState('')
  const [newPriority, setNewPriority] = useState(0)
  const [newReason, setNewReason] = useState('')

  useEffect(() => {
    fetchRules()
  }, [])

  useEffect(() => {
    fetchStats()
  }, [statsRange])

  const fetchRules = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await geoApi.getRules(true)
      if (response.success) setRules(response.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch geo rules')
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await geoApi.getStats(statsRange)
      if (response.success && Array.isArray(response.data)) {
        setStats(response.data)
      } else {
        setStats([])
      }
    } catch (err: any) {
      // Silently handle network errors
      setStats([])
    }
  }

  const handleAdd = async () => {
    if (!newCountryCode.trim() || !newCountryName.trim()) return

    setLoading(true)
    setError(null)
    try {
      await geoApi.createRule({
        rule_type: newRuleType,
        country_code: newCountryCode,
        country_name: newCountryName,
        priority: newPriority,
        reason: newReason || undefined,
      })
      setNewCountryCode('')
      setNewCountryName('')
      setNewPriority(0)
      setNewReason('')
      setShowAddDialog(false)
      fetchRules()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to create rule')
      }
    } finally {
      setLoading(false)
    }
  }

  const filteredRules = rules.filter(rule =>
    rule.country_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rule.country_code.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const geoStatsChartData = useMemo(
    () => (Array.isArray(stats) ? stats.slice(0, 10) : []),
    [stats]
  )

  const topThreatCountriesData = useMemo(() => {
    if (!Array.isArray(stats) || stats.length === 0) return []
    return [...stats]
      .sort((a, b) => b.threat_count - a.threat_count)
      .slice(0, 10)
      .map((s) => ({ name: s.country_name, count: s.threat_count }))
  }, [stats])

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <ErrorBoundary>
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold">Geo Rules</h1>
                  <p className="text-muted-foreground mt-1">Manage geographic access rules</p>
                </div>
                <Button onClick={() => setShowAddDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Rule
                </Button>
              </div>

              {error && (
                <Card className="p-4 bg-destructive/10 border-destructive">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                </Card>
              )}

              <Card>
                <div className="p-6">
                  <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                    <div className="flex items-center gap-2">
                      <Globe className="h-5 w-5 text-muted-foreground" />
                      <h3 className="text-lg font-semibold">Attack origins</h3>
                    </div>
                    <Select
                      value={statsRange}
                      onValueChange={(v) => setStatsRange(v as StatsRange)}
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {STATS_RANGE_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    Where blocked requests are coming from. Marker size reflects blocked request count.
                    <span className="block mt-1 text-xs opacity-80">Drag to pan • Scroll to zoom • Hover markers for details</span>
                  </p>
                  {stats.length > 0 ? (
                    <div className="relative w-full min-h-[32vh] aspect-[2/1] rounded-lg overflow-hidden bg-muted/30">
                      <GeoAttackMap stats={stats} className="absolute inset-0" />
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center min-h-[55vh] rounded-lg bg-muted/30 text-muted-foreground text-sm text-center px-4">
                      <MapPin className="h-12 w-12 mb-2 opacity-50" />
                      <p>No geographic data for the selected period.</p>
                      <p className="mt-1">Traffic or threats with country attribution will appear here.</p>
                      <p className="mt-2 text-xs opacity-80">
                        Ensure GeoIP database is configured (see docs/GEOIP_SETUP.md) or run{' '}
                        <code className="bg-muted px-1 rounded">scripts/seed_geo_traffic.py</code> for demo data.
                      </p>
                    </div>
                  )}
                </div>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div
                  className="rounded-md p-4 md:p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <h3
                    className="text-lg font-semibold mb-4"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    Geographic Statistics ({statsRange === '1h' ? '1h' : statsRange === '24h' ? '24h' : statsRange === '7d' ? '7d' : '30d'})
                  </h3>
                  {geoStatsChartData.length === 0 ? (
                    <div
                      className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md"
                      style={{ borderColor: 'var(--positivus-gray)' }}
                    >
                      <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                        No statistics available
                      </p>
                    </div>
                  ) : (
                    <ChartContainer config={geoStatsChartConfig} className="aspect-auto h-[300px] w-full">
                      <BarChart data={geoStatsChartData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                        <CartesianGrid vertical={false} />
                        <XAxis
                          dataKey="country_code"
                          tickLine={false}
                          axisLine={false}
                          tickMargin={8}
                          minTickGap={32}
                        />
                        <YAxis tickLine={false} axisLine={false} tickMargin={8} allowDecimals={false} />
                        <ChartTooltip
                          content={
                            <ChartTooltipContent
                              labelFormatter={(value) => value}
                              formatter={(value) => Number(value).toLocaleString()}
                            />
                          }
                        />
                        <ChartLegend content={<ChartLegendContent />} />
                        <Bar dataKey="total_requests" fill="var(--color-total_requests)" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="blocked_requests" fill="var(--color-blocked_requests)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ChartContainer>
                  )}
                </div>

                <div
                  className="rounded-md p-4 md:p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <h3
                    className="text-lg font-semibold mb-4"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    Top Threat Countries
                  </h3>
                  <p className="text-sm text-muted-foreground hidden sm:block mb-4">
                    Countries with the highest threat counts
                  </p>
                  {topThreatCountriesData.length === 0 ? (
                    <div
                      className="flex flex-col items-center justify-center h-[300px] border-2 border-dashed rounded-md"
                      style={{ borderColor: 'var(--positivus-gray)' }}
                    >
                      <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                        No statistics available
                      </p>
                    </div>
                  ) : (
                    <ChartContainer
                      config={topThreatCountriesChartConfig}
                      className="aspect-auto w-full"
                      style={{ height: Math.max(250, topThreatCountriesData.length * 36) }}
                    >
                      <BarChart
                        accessibilityLayer
                        data={topThreatCountriesData}
                        layout="vertical"
                        margin={{ left: 8, right: 12, top: 8, bottom: 8 }}
                      >
                        <CartesianGrid horizontal={false} />
                        <XAxis
                          type="number"
                          tickLine={false}
                          axisLine={false}
                          tickMargin={8}
                          allowDecimals={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          tickLine={false}
                          axisLine={false}
                          tickMargin={8}
                          width={140}
                        />
                        <ChartTooltip
                          content={
                            <ChartTooltipContent
                              className="w-[180px]"
                              nameKey="count"
                              labelFormatter={(value) => value}
                              formatter={(value) => `${Number(value).toLocaleString()} threats`}
                            />
                          }
                        />
                        <Bar dataKey="count" fill="var(--color-count)" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ChartContainer>
                  )}
                </div>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search countries..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10"
                      />
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium">Country</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Code</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Rule Type</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Priority</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Reason</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRules.map((rule) => (
                        <tr key={rule.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 font-medium">{rule.country_name}</td>
                          <td className="px-4 py-3">
                            <Badge variant="outline">{rule.country_code}</Badge>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={rule.rule_type === 'allow' ? 'default' : 'destructive'}>
                              {rule.rule_type}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">{rule.priority}</td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {rule.reason || '-'}
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={rule.is_active ? 'default' : 'secondary'}>
                              {rule.is_active ? 'Active' : 'Inactive'}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredRules.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground">No geo rules found</div>
                  )}
                </div>
              </Card>
            </div>

            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create Geo Rule</DialogTitle>
                  <DialogDescription>
                    Create a rule to allow or deny traffic from a specific country
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Rule Type</label>
                    <Select value={newRuleType} onValueChange={(v: 'allow' | 'deny') => setNewRuleType(v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="allow">Allow</SelectItem>
                        <SelectItem value="deny">Deny</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Country Code (ISO 3166-1 alpha-2)</label>
                    <Input
                      value={newCountryCode}
                      onChange={(e) => setNewCountryCode(e.target.value.toUpperCase())}
                      placeholder="US"
                      maxLength={2}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Country Name</label>
                    <Input
                      value={newCountryName}
                      onChange={(e) => setNewCountryName(e.target.value)}
                      placeholder="United States"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Priority</label>
                    <Input
                      type="number"
                      value={newPriority}
                      onChange={(e) => setNewPriority(parseInt(e.target.value) || 0)}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Reason (optional)</label>
                    <Input
                      value={newReason}
                      onChange={(e) => setNewReason(e.target.value)}
                      placeholder="Reason for this rule"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newCountryCode.trim() || !newCountryName.trim()}>
                      Create Rule
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
