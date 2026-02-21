'use client'

import { useEffect, useState } from 'react'
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { ShieldAlert, Lock, Loader2 } from 'lucide-react'
import { credentialLeakApi, type CredentialLeakEventData, type CredentialLeakStats } from '@/lib/api'
import { useTimezone } from '@/contexts/timezone-context'
import { formatTimeLocal } from '@/lib/chart-utils'

const TIME_RANGES = [
  { value: '1h', label: 'Last 1 hour' },
  { value: '6h', label: 'Last 6 hours' },
  { value: '24h', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
] as const

function parseDetails(details: string | undefined): Record<string, unknown> {
  if (!details) return {}
  try {
    return JSON.parse(details) as Record<string, unknown>
  } catch {
    return {}
  }
}

export default function CredentialProtectionPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [stats, setStats] = useState<CredentialLeakStats | null>(null)
  const [events, setEvents] = useState<CredentialLeakEventData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { timezone } = useTimezone()

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const [statsRes, eventsRes] = await Promise.all([
          credentialLeakApi.getCredentialLeakStats(timeRange),
          credentialLeakApi.getCredentialLeakEvents(timeRange, 100),
        ])
        if (statsRes.success && statsRes.data) setStats(statsRes.data)
        if (eventsRes.success && eventsRes.data) setEvents(eventsRes.data)
      } catch (err: unknown) {
        const e = err as { isNetworkError?: boolean; message?: string }
        if (!e?.isNetworkError) {
          setError(e?.message || 'Failed to load credential protection data')
        }
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [timeRange])

  const blockedCount = stats?.blocked_count ?? 0
  const flaggedCount = stats?.flagged_count ?? 0

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
                  Credential protection
                </h2>
                <p className="text-muted-foreground">
                  Have I Been Pwned (HIBP) integration: block or flag login attempts using breached passwords. No passwords stored.
                </p>
              </div>

              {error && (
                <Card className="p-4 border-destructive" style={{ backgroundColor: 'rgba(var(--destructive), 0.1)' }}>
                  <div className="flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                </Card>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <ShieldAlert size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      Blocked
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : blockedCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
                <Card
                  className="p-6 border-2"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-md" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
                      <Lock size={20} style={{ color: 'var(--positivus-green)' }} />
                    </div>
                    <span className="text-sm font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>
                      Flagged
                    </span>
                  </div>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    {loading ? '...' : flaggedCount.toLocaleString()}
                  </p>
                  <p className="text-xs mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>Last {timeRange}</p>
                </Card>
              </div>

              <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2" style={{ borderColor: 'var(--positivus-gray)' }}>
                  <div>
                    <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Recent events
                    </CardTitle>
                    <CardDescription>
                      Blocked and flagged login attempts (no password or hash in UI)
                    </CardDescription>
                  </div>
                  <Select value={timeRange} onValueChange={setTimeRange}>
                    <SelectTrigger className="w-[160px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {TIME_RANGES.map((r) => (
                        <SelectItem key={r.value} value={r.value}>{r.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                    </div>
                  ) : events.length === 0 ? (
                    <p className="text-muted-foreground py-8 text-center">No credential leak events in this period.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow style={{ borderColor: 'var(--positivus-gray)' }}>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Time</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>IP</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Path</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Username</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Action</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {events.map((ev) => {
                          const d = parseDetails(ev.details)
                          const username = d.credential_leak_username as string | undefined
                          return (
                            <TableRow key={ev.id} style={{ borderColor: 'var(--positivus-gray)' }}>
                              <TableCell style={{ color: 'var(--positivus-black)' }}>
                                {formatTimeLocal(ev.timestamp ?? '', timezone)}
                              </TableCell>
                              <TableCell className="font-mono text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.ip}</TableCell>
                              <TableCell className="font-mono text-sm max-w-[200px] truncate" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.path ?? '—'}</TableCell>
                              <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{username ?? '—'}</TableCell>
                              <TableCell>
                                {ev.event_type === 'credential_leak_block' ? (
                                  <span className="text-destructive font-medium">Blocked</span>
                                ) : (
                                  <span className="text-muted-foreground">Flagged</span>
                                )}
                              </TableCell>
                            </TableRow>
                          )
                        })}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
