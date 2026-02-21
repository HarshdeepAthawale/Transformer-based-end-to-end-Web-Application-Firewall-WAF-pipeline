'use client'

import { useState, useEffect } from 'react'
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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
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
import { Switch } from '@/components/ui/switch'
import { firewallAiApi, type LLMEndpointData, type FirewallAIEventData } from '@/lib/api'
import { Cpu, Plus, Pencil, Trash2, ShieldAlert, Loader2 } from 'lucide-react'
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

function reasonLabel(eventType: string): string {
  if (eventType.includes('prompt')) return 'Prompt injection'
  if (eventType.includes('pii')) return 'PII'
  if (eventType.includes('abuse')) return 'Abuse rate'
  return eventType
}

export default function FirewallAIPage() {
  const [endpoints, setEndpoints] = useState<LLMEndpointData[]>([])
  const [events, setEvents] = useState<FirewallAIEventData[]>([])
  const [timeRange, setTimeRange] = useState('24h')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [formPath, setFormPath] = useState('')
  const [formMethods, setFormMethods] = useState('POST')
  const [formLabel, setFormLabel] = useState('llm')
  const [formActive, setFormActive] = useState(true)
  const { timezone } = useTimezone()

  const fetchEndpoints = async () => {
    try {
      const res = await firewallAiApi.getLlmEndpoints(false)
      if (res.success && res.data) setEndpoints(res.data)
    } catch (e) {
      if (!(e as { isNetworkError?: boolean })?.isNetworkError) {
        setError((e as Error).message || 'Failed to load endpoints')
      }
    }
  }

  const fetchEvents = async () => {
    try {
      const res = await firewallAiApi.getFirewallAiEvents(timeRange, 100)
      if (res.success && res.data) setEvents(res.data)
    } catch (e) {
      if (!(e as { isNetworkError?: boolean })?.isNetworkError) {
        setError((e as Error).message || 'Failed to load events')
      }
    }
  }

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([fetchEndpoints(), fetchEvents()]).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetchEvents()
  }, [timeRange])

  const openAdd = () => {
    setEditingId(null)
    setFormPath('')
    setFormMethods('POST')
    setFormLabel('llm')
    setFormActive(true)
    setShowForm(true)
  }

  const openEdit = (ep: LLMEndpointData) => {
    setEditingId(ep.id)
    setFormPath(ep.path_pattern)
    setFormMethods(ep.methods || 'POST')
    setFormLabel(ep.label || 'llm')
    setFormActive(ep.is_active)
    setShowForm(true)
  }

  const handleSave = async () => {
    if (!formPath.trim()) return
    setError(null)
    try {
      if (editingId != null) {
        await firewallAiApi.updateLlmEndpoint(editingId, {
          path_pattern: formPath.trim(),
          methods: formMethods,
          label: formLabel.trim(),
          is_active: formActive,
        })
      } else {
        await firewallAiApi.createLlmEndpoint({
          path_pattern: formPath.trim(),
          methods: formMethods,
          label: formLabel.trim(),
          is_active: formActive,
        })
      }
      setShowForm(false)
      fetchEndpoints()
    } catch (e) {
      setError((e as Error).message || 'Failed to save endpoint')
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this LLM endpoint?')) return
    setError(null)
    try {
      await firewallAiApi.deleteLlmEndpoint(id)
      fetchEndpoints()
    } catch (e) {
      setError((e as Error).message || 'Failed to delete endpoint')
    }
  }

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
                  Firewall for AI
                </h2>
                <p className="text-muted-foreground">
                  Protect LLM endpoints with prompt-injection detection, PII checks, and abuse rate limiting.
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

              {/* LLM Endpoints */}
              <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0">
                  <div>
                    <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      LLM Endpoints
                    </CardTitle>
                    <CardDescription>
                      Path patterns and methods that receive Firewall-for-AI checks
                    </CardDescription>
                  </div>
                  <Button onClick={openAdd} size="sm" style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}>
                    <Plus className="h-4 w-4 mr-2" /> Add
                  </Button>
                </CardHeader>
                <CardContent>
                  {loading && !endpoints.length ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                    </div>
                  ) : endpoints.length === 0 ? (
                    <p className="text-muted-foreground py-6 text-center">No LLM endpoints. Add one to protect a path.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow style={{ borderColor: 'var(--positivus-gray)' }}>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Path pattern</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Methods</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Label</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Active</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }} className="w-[100px]">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {endpoints.map((ep) => (
                          <TableRow key={ep.id} style={{ borderColor: 'var(--positivus-gray)' }}>
                            <TableCell className="font-mono text-sm" style={{ color: 'var(--positivus-black)' }}>{ep.path_pattern}</TableCell>
                            <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{ep.methods}</TableCell>
                            <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{ep.label}</TableCell>
                            <TableCell>{ep.is_active ? 'Yes' : 'No'}</TableCell>
                            <TableCell>
                              <div className="flex gap-2">
                                <Button variant="ghost" size="icon" onClick={() => openEdit(ep)}><Pencil className="h-4 w-4" /></Button>
                                <Button variant="ghost" size="icon" onClick={() => handleDelete(ep.id)}><Trash2 className="h-4 w-4 text-destructive" /></Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>

              {/* Events */}
              <Card className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2" style={{ borderColor: 'var(--positivus-gray)' }}>
                  <div>
                    <CardTitle className="text-lg" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Firewall for AI events
                    </CardTitle>
                    <CardDescription>
                      Blocked prompts, PII, and abuse rate events
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
                  {events.length === 0 ? (
                    <p className="text-muted-foreground py-8 text-center">No events in this period.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow style={{ borderColor: 'var(--positivus-gray)' }}>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Time</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Path</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Reason</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>Action</TableHead>
                          <TableHead style={{ color: 'var(--positivus-gray-dark)' }}>IP</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {events.map((ev) => {
                          const d = parseDetails(ev.details)
                          const reason = (d.firewall_ai_reason as string) || reasonLabel(ev.event_type)
                          const action = (d.firewall_ai_action as string) || 'block'
                          return (
                            <TableRow key={ev.id} style={{ borderColor: 'var(--positivus-gray)' }}>
                              <TableCell style={{ color: 'var(--positivus-black)' }}>
                                {formatTimeLocal(ev.timestamp ?? '', timezone)}
                              </TableCell>
                              <TableCell className="font-mono text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>{ev.path ?? '—'}</TableCell>
                              <TableCell>{reason}</TableCell>
                              <TableCell>{action}</TableCell>
                              <TableCell style={{ color: 'var(--positivus-gray-dark)' }}>{ev.ip}</TableCell>
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

      <Dialog open={showForm} onOpenChange={setShowForm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingId != null ? 'Edit LLM endpoint' : 'Add LLM endpoint'}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label>Path pattern</Label>
              <Input
                placeholder="/api/chat or /v1/completions"
                value={formPath}
                onChange={(e) => setFormPath(e.target.value)}
              />
            </div>
            <div className="grid gap-2">
              <Label>Methods</Label>
              <Select value={formMethods} onValueChange={setFormMethods}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="POST">POST</SelectItem>
                  <SelectItem value="GET">GET</SelectItem>
                  <SelectItem value="POST,GET">POST,GET</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label>Label</Label>
              <Input
                placeholder="chat"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={formActive} onCheckedChange={setFormActive} />
              <Label>Active</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowForm(false)}>Cancel</Button>
            <Button onClick={handleSave} style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
