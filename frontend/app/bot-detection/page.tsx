'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  botApi,
  botEventsApi,
  BotSignature,
  VerifiedBot,
  BotScoreBand,
} from '@/lib/api'
import { Bot, Plus, Search, AlertCircle, RefreshCw, Trash2, Shield } from 'lucide-react'
import { useTimezone } from '@/contexts/timezone-context'
import { formatTimeLocal } from '@/lib/chart-utils'

export default function BotDetectionPage() {
  const [signatures, setSignatures] = useState<BotSignature[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newPattern, setNewPattern] = useState('')
  const [newName, setNewName] = useState('')
  const [newCategory, setNewCategory] = useState('unknown')
  const [newAction, setNewAction] = useState('block')
  const [newWhitelisted, setNewWhitelisted] = useState(false)

  // Score bands
  const [scoreBands, setScoreBands] = useState<BotScoreBand[]>([])
  const [showBandsEdit, setShowBandsEdit] = useState(false)
  const [editingBands, setEditingBands] = useState<{ min_score: number; max_score: number; action: string }[]>([])

  // Verified bots
  const [verifiedBots, setVerifiedBots] = useState<VerifiedBot[]>([])
  const [showVerifiedAdd, setShowVerifiedAdd] = useState(false)
  const [verifiedName, setVerifiedName] = useState('')
  const [verifiedPattern, setVerifiedPattern] = useState('')
  const [syncLoading, setSyncLoading] = useState(false)

  // Bot events
  const [botEvents, setBotEvents] = useState<Array<{ id: number; timestamp: string; event_type: string; ip: string; path?: string; bot_score?: number; details?: string }>>([])
  const [timeRange, setTimeRange] = useState('24h')
  const { timezone } = useTimezone()

  useEffect(() => {
    fetchSignatures()
  }, [])

  useEffect(() => {
    fetchScoreBands()
    fetchVerifiedBots()
  }, [])

  useEffect(() => {
    fetchBotEvents()
  }, [timeRange])

  const fetchSignatures = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await botApi.getSignatures(true)
      if (response.success) setSignatures(response.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch bot signatures')
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchScoreBands = async () => {
    try {
      const res = await botApi.getScoreBands()
      if (res.success && res.data) setScoreBands(res.data)
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to fetch score bands')
    }
  }

  const fetchVerifiedBots = async () => {
    try {
      const res = await botApi.getVerifiedBots()
      if (res.success && res.data) setVerifiedBots(res.data)
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to fetch verified bots')
    }
  }

  const fetchBotEvents = async () => {
    try {
      const res = await botEventsApi.getBotEvents(timeRange, 50)
      if (res.success && res.data) setBotEvents(res.data)
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to fetch bot events')
    }
  }

  const handleSaveBands = async () => {
    setLoading(true)
    setError(null)
    try {
      await botApi.updateScoreBands(editingBands)
      setShowBandsEdit(false)
      fetchScoreBands()
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to update score bands')
    } finally {
      setLoading(false)
    }
  }

  const handleAddVerifiedBot = async () => {
    if (!verifiedName.trim() || !verifiedPattern.trim()) return
    setLoading(true)
    setError(null)
    try {
      await botApi.addVerifiedBot({ name: verifiedName, user_agent_pattern: verifiedPattern })
      setVerifiedName('')
      setVerifiedPattern('')
      setShowVerifiedAdd(false)
      fetchVerifiedBots()
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to add verified bot')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteVerifiedBot = async (id: number) => {
    setLoading(true)
    setError(null)
    try {
      await botApi.deleteVerifiedBot(id)
      fetchVerifiedBots()
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Failed to delete verified bot')
    } finally {
      setLoading(false)
    }
  }

  const handleSyncVerifiedBots = async () => {
    setSyncLoading(true)
    setError(null)
    try {
      await botApi.syncVerifiedBots()
      fetchVerifiedBots()
    } catch (err: any) {
      if (!err?.isNetworkError) setError(err?.message || 'Sync failed (BOT_VERIFIED_SYNC_URL may not be configured)')
    } finally {
      setSyncLoading(false)
    }
  }

  const openBandsEdit = () => {
    setEditingBands(scoreBands.map((b) => ({ min_score: b.min_score, max_score: b.max_score, action: b.action })))
    setShowBandsEdit(true)
  }

  const handleAdd = async () => {
    if (!newPattern.trim() || !newName.trim()) return

    setLoading(true)
    setError(null)
    try {
      await botApi.addSignature({
        user_agent_pattern: newPattern,
        name: newName,
        category: newCategory,
        action: newAction,
        is_whitelisted: newWhitelisted,
      })
      setNewPattern('')
      setNewName('')
      setNewCategory('unknown')
      setNewAction('block')
      setNewWhitelisted(false)
      setShowAddDialog(false)
      fetchSignatures()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to add signature')
      }
    } finally {
      setLoading(false)
    }
  }

  const filteredSignatures = signatures.filter(sig =>
    sig.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    sig.user_agent_pattern.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const categoryColors: Record<string, string> = {
    search_engine: 'default',
    scraper: 'secondary',
    malicious: 'destructive',
    monitoring: 'outline',
    unknown: 'outline',
  }

  const actionColors: Record<string, string> = {
    block: 'destructive',
    allow: 'default',
    challenge: 'secondary',
    monitor: 'outline',
  }

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
                  <h1 className="text-3xl font-bold">Bot Detection</h1>
                  <p className="text-muted-foreground mt-1">Manage bot detection signatures</p>
                </div>
                <Button onClick={() => setShowAddDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Signature
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

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Total Signatures</div>
                  <div className="text-2xl font-bold mt-1">{signatures.length}</div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Active</div>
                  <div className="text-2xl font-bold mt-1">
                    {signatures.filter(s => s.is_active).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Whitelisted</div>
                  <div className="text-2xl font-bold mt-1">
                    {signatures.filter(s => s.is_whitelisted).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Malicious</div>
                  <div className="text-2xl font-bold mt-1">
                    {signatures.filter(s => s.category === 'malicious').length}
                  </div>
                </Card>
              </div>

              <Tabs defaultValue="signatures" className="space-y-4">
                <TabsList>
                  <TabsTrigger value="signatures">Signatures</TabsTrigger>
                  <TabsTrigger value="score-bands">Score Bands</TabsTrigger>
                  <TabsTrigger value="verified-bots">Verified Bots</TabsTrigger>
                  <TabsTrigger value="events">Bot Events</TabsTrigger>
                </TabsList>

                <TabsContent value="signatures">
                  <Card>
                    <div className="p-4 border-b">
                      <div className="flex items-center gap-4">
                        <div className="flex-1 relative">
                          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                          <Input
                            placeholder="Search signatures..."
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
                            <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Pattern</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Category</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Action</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredSignatures.map((sig) => (
                            <tr key={sig.id} className="border-b hover:bg-muted/50">
                              <td className="px-4 py-3 font-medium">{sig.name}</td>
                              <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                                {sig.user_agent_pattern}
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant={categoryColors[sig.category] as any}>{sig.category}</Badge>
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant={actionColors[sig.action] as any}>{sig.action}</Badge>
                              </td>
                              <td className="px-4 py-3">
                                <div className="flex items-center gap-2">
                                  <Badge variant={sig.is_active ? 'default' : 'secondary'}>
                                    {sig.is_active ? 'Active' : 'Inactive'}
                                  </Badge>
                                  {sig.is_whitelisted && <Badge variant="outline">Whitelisted</Badge>}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {filteredSignatures.length === 0 && (
                        <div className="p-8 text-center text-muted-foreground">No bot signatures found</div>
                      )}
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="score-bands">
                  <Card>
                    <div className="p-4 border-b flex items-center justify-between">
                      <h3 className="font-medium">Score Bands</h3>
                      <Button variant="outline" size="sm" onClick={openBandsEdit}>
                        Edit Bands
                      </Button>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-muted">
                          <tr>
                            <th className="px-4 py-3 text-left text-sm font-medium">Min Score</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Max Score</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {scoreBands.map((b) => (
                            <tr key={b.id} className="border-b hover:bg-muted/50">
                              <td className="px-4 py-3">{b.min_score}</td>
                              <td className="px-4 py-3">{b.max_score}</td>
                              <td className="px-4 py-3">
                                <Badge variant={b.action === 'block' ? 'destructive' : b.action === 'challenge' ? 'secondary' : 'default'}>
                                  {b.action}
                                </Badge>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {scoreBands.length === 0 && (
                        <div className="p-8 text-center text-muted-foreground">No score bands configured</div>
                      )}
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="verified-bots">
                  <Card>
                    <div className="p-4 border-b flex items-center justify-between">
                      <h3 className="font-medium">Verified Bots (Allowlist)</h3>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={handleSyncVerifiedBots} disabled={syncLoading}>
                          {syncLoading ? (
                            <RefreshCw className="h-4 w-4 animate-spin mr-1" />
                          ) : (
                            <RefreshCw className="h-4 w-4 mr-1" />
                          )}
                          Sync from URL
                        </Button>
                        <Button size="sm" onClick={() => setShowVerifiedAdd(true)}>
                          <Plus className="h-4 w-4 mr-1" />
                          Add
                        </Button>
                      </div>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-muted">
                          <tr>
                            <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Pattern</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Source</th>
                            <th className="px-4 py-3 text-left text-sm font-medium"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {verifiedBots.map((vb) => (
                            <tr key={vb.id} className="border-b hover:bg-muted/50">
                              <td className="px-4 py-3 font-medium">{vb.name}</td>
                              <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                                {vb.user_agent_pattern}
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant="outline">{vb.source}</Badge>
                              </td>
                              <td className="px-4 py-3">
                                <Button variant="ghost" size="sm" onClick={() => handleDeleteVerifiedBot(vb.id)}>
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {verifiedBots.length === 0 && (
                        <div className="p-8 text-center text-muted-foreground">No verified bots</div>
                      )}
                    </div>
                  </Card>
                </TabsContent>

                <TabsContent value="events">
                  <Card>
                    <div className="p-4 border-b flex items-center justify-between">
                      <h3 className="font-medium">Recent Bot Block/Challenge Events</h3>
                      <Select value={timeRange} onValueChange={setTimeRange}>
                        <SelectTrigger className="w-[120px]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1h">Last 1h</SelectItem>
                          <SelectItem value="6h">Last 6h</SelectItem>
                          <SelectItem value="24h">Last 24h</SelectItem>
                          <SelectItem value="7d">Last 7d</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-muted">
                          <tr>
                            <th className="px-4 py-3 text-left text-sm font-medium">Time</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Type</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">IP</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Path</th>
                            <th className="px-4 py-3 text-left text-sm font-medium">Bot Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {botEvents.map((ev) => (
                            <tr key={ev.id} className="border-b hover:bg-muted/50">
                              <td className="px-4 py-3 text-sm text-muted-foreground">
                                {ev.timestamp ? formatTimeLocal(ev.timestamp, timezone) : '-'}
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant={ev.event_type === 'bot_block' ? 'destructive' : 'secondary'}>
                                  {ev.event_type}
                                </Badge>
                              </td>
                              <td className="px-4 py-3 font-mono text-sm">{ev.ip}</td>
                              <td className="px-4 py-3 text-sm truncate max-w-[200px]">{ev.path || '-'}</td>
                              <td className="px-4 py-3">
                                {ev.bot_score != null ? (
                                  <Badge variant="outline">{ev.bot_score}</Badge>
                                ) : (
                                  '-'
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {botEvents.length === 0 && (
                        <div className="p-8 text-center text-muted-foreground">No bot events in this range</div>
                      )}
                    </div>
                  </Card>
                </TabsContent>
              </Tabs>

              <Dialog open={showBandsEdit} onOpenChange={setShowBandsEdit}>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Edit Score Bands</DialogTitle>
                    <DialogDescription>
                      Score 1–99: low = automated, high = human. Bands are evaluated by priority order.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 max-h-[60vh] overflow-y-auto">
                    {editingBands.map((b, i) => (
                      <div key={i} className="flex gap-4 items-center">
                        <Input
                          type="number"
                          min={1}
                          max={99}
                          value={b.min_score}
                          onChange={(e) => {
                            const next = [...editingBands]
                            next[i] = { ...next[i], min_score: parseInt(e.target.value) || 1 }
                            setEditingBands(next)
                          }}
                          className="w-20"
                        />
                        <span className="text-muted-foreground">–</span>
                        <Input
                          type="number"
                          min={1}
                          max={99}
                          value={b.max_score}
                          onChange={(e) => {
                            const next = [...editingBands]
                            next[i] = { ...next[i], max_score: parseInt(e.target.value) || 99 }
                            setEditingBands(next)
                          }}
                          className="w-20"
                        />
                        <Select
                          value={b.action}
                          onValueChange={(v) => {
                            const next = [...editingBands]
                            next[i] = { ...next[i], action: v }
                            setEditingBands(next)
                          }}
                        >
                          <SelectTrigger className="w-28">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="allow">Allow</SelectItem>
                            <SelectItem value="challenge">Challenge</SelectItem>
                            <SelectItem value="block">Block</SelectItem>
                          </SelectContent>
                        </Select>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setEditingBands(editingBands.filter((_, j) => j !== i))}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setEditingBands([...editingBands, { min_score: 70, max_score: 99, action: 'allow' }])
                      }
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      Add Band
                    </Button>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowBandsEdit(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleSaveBands} disabled={loading || editingBands.length === 0}>
                      Save
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>

              <Dialog open={showVerifiedAdd} onOpenChange={setShowVerifiedAdd}>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add Verified Bot</DialogTitle>
                    <DialogDescription>
                      Allowlisted bots receive a high bot score and pass through based on score bands.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">Name</label>
                      <Input
                        value={verifiedName}
                        onChange={(e) => setVerifiedName(e.target.value)}
                        placeholder="Googlebot"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">User Agent Pattern (regex)</label>
                      <Input
                        value={verifiedPattern}
                        onChange={(e) => setVerifiedPattern(e.target.value)}
                        placeholder=".*Googlebot.*"
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setShowVerifiedAdd(false)}>
                        Cancel
                      </Button>
                      <Button
                        onClick={handleAddVerifiedBot}
                        disabled={loading || !verifiedName.trim() || !verifiedPattern.trim()}
                      >
                        Add
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Bot Signature</DialogTitle>
                  <DialogDescription>
                    Create a signature to detect and handle specific bots
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Name</label>
                    <Input
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="Googlebot"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">User Agent Pattern</label>
                    <Input
                      value={newPattern}
                      onChange={(e) => setNewPattern(e.target.value)}
                      placeholder=".*Googlebot.*"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Category</label>
                    <Select value={newCategory} onValueChange={setNewCategory}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="search_engine">Search Engine</SelectItem>
                        <SelectItem value="scraper">Scraper</SelectItem>
                        <SelectItem value="malicious">Malicious</SelectItem>
                        <SelectItem value="monitoring">Monitoring</SelectItem>
                        <SelectItem value="unknown">Unknown</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Action</label>
                    <Select value={newAction} onValueChange={setNewAction}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="block">Block</SelectItem>
                        <SelectItem value="allow">Allow</SelectItem>
                        <SelectItem value="challenge">Challenge</SelectItem>
                        <SelectItem value="monitor">Monitor</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="whitelisted"
                      checked={newWhitelisted}
                      onChange={(e) => setNewWhitelisted(e.target.checked)}
                      className="rounded"
                    />
                    <label htmlFor="whitelisted" className="text-sm font-medium">
                      Whitelist this signature
                    </label>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newPattern.trim() || !newName.trim()}>
                      Add Signature
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
