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
import { threatIntelApi, ThreatIntel, ThreatCheckResult } from '@/lib/api'
import { Shield, Plus, Search, AlertTriangle, Eye, AlertCircle } from 'lucide-react'

export default function ThreatIntelligencePage() {
  const [threats, setThreats] = useState<ThreatIntel[]>([])
  const [threatTypeFilter, setThreatTypeFilter] = useState<string>('all')
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [showCheckDialog, setShowCheckDialog] = useState(false)
  const [checkIP, setCheckIP] = useState('')
  const [checkResult, setCheckResult] = useState<ThreatCheckResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newThreatType, setNewThreatType] = useState('ip')
  const [newValue, setNewValue] = useState('')
  const [newSeverity, setNewSeverity] = useState('medium')
  const [newCategory, setNewCategory] = useState('')
  const [newSource, setNewSource] = useState('')
  const [newDescription, setNewDescription] = useState('')

  useEffect(() => {
    fetchThreats()
  }, [threatTypeFilter])

  const fetchThreats = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await threatIntelApi.getFeeds(
        threatTypeFilter === 'all' ? undefined : threatTypeFilter,
        true,
        1000
      )
      if (response.success) {
        let filtered = response.data
        if (severityFilter !== 'all') {
          filtered = filtered.filter(t => t.severity === severityFilter)
        }
        setThreats(filtered)
      }
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch threat intelligence')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = async () => {
    if (!newValue.trim() || !newCategory.trim() || !newSource.trim()) return

    setLoading(true)
    setError(null)
    try {
      await threatIntelApi.addThreat({
        threat_type: newThreatType,
        value: newValue,
        severity: newSeverity,
        category: newCategory,
        source: newSource,
        description: newDescription || undefined,
      })
      setNewValue('')
      setNewCategory('')
      setNewSource('')
      setNewDescription('')
      setShowAddDialog(false)
      fetchThreats()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to add threat')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleCheckIP = async () => {
    if (!checkIP.trim()) return

    setLoading(true)
    setError(null)
    try {
      const response = await threatIntelApi.checkIP(checkIP)
      if (response.success) {
        setCheckResult(response.data)
      }
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to check IP')
      }
    } finally {
      setLoading(false)
    }
  }

  const filteredThreats = threats.filter(threat =>
    threat.value.toLowerCase().includes(searchQuery.toLowerCase()) ||
    threat.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
    threat.source.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const severityColors: Record<string, string> = {
    critical: 'destructive',
    high: 'destructive',
    medium: 'default',
    low: 'secondary',
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <ErrorBoundary>
            <div className="max-w-7xl mx-auto space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold">Threat Intelligence</h1>
                  <p className="text-muted-foreground mt-1">Manage threat intelligence feeds</p>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => setShowCheckDialog(true)}>
                    <Eye className="mr-2 h-4 w-4" />
                    Check IP
                  </Button>
                  <Button onClick={() => setShowAddDialog(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    Add Threat
                  </Button>
                </div>
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
                  <div className="text-sm text-muted-foreground">Total Threats</div>
                  <div className="text-2xl font-bold mt-1">{threats.length}</div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Critical</div>
                  <div className="text-2xl font-bold mt-1 text-destructive">
                    {threats.filter(t => t.severity === 'critical').length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">High</div>
                  <div className="text-2xl font-bold mt-1 text-orange-500">
                    {threats.filter(t => t.severity === 'high').length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Active</div>
                  <div className="text-2xl font-bold mt-1">
                    {threats.filter(t => t.is_active).length}
                  </div>
                </Card>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search threats..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10"
                      />
                    </div>
                    <Select value={threatTypeFilter} onValueChange={setThreatTypeFilter}>
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Types</SelectItem>
                        <SelectItem value="ip">IP</SelectItem>
                        <SelectItem value="domain">Domain</SelectItem>
                        <SelectItem value="signature">Signature</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={severityFilter} onValueChange={setSeverityFilter}>
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Severities</SelectItem>
                        <SelectItem value="critical">Critical</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium">Type</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Value</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Severity</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Category</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Source</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Added</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredThreats.map((threat) => (
                        <tr key={threat.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3">
                            <Badge variant="outline">{threat.threat_type}</Badge>
                          </td>
                          <td className="px-4 py-3 font-mono text-sm">{threat.value}</td>
                          <td className="px-4 py-3">
                            <Badge variant={severityColors[threat.severity] as any}>
                              {threat.severity}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">{threat.category}</td>
                          <td className="px-4 py-3">
                            <Badge variant="secondary">{threat.source}</Badge>
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {new Date(threat.created_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={threat.is_active ? 'default' : 'secondary'}>
                              {threat.is_active ? 'Active' : 'Inactive'}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredThreats.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground">No threats found</div>
                  )}
                </div>
              </Card>
            </div>

            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Threat Intelligence</DialogTitle>
                  <DialogDescription>
                    Add a new threat to the intelligence feed
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Threat Type</label>
                    <Select value={newThreatType} onValueChange={setNewThreatType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ip">IP Address</SelectItem>
                        <SelectItem value="domain">Domain</SelectItem>
                        <SelectItem value="signature">Signature</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Value</label>
                    <Input
                      value={newValue}
                      onChange={(e) => setNewValue(e.target.value)}
                      placeholder={newThreatType === 'ip' ? '192.168.1.1' : 'example.com'}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Severity</label>
                    <Select value={newSeverity} onValueChange={setNewSeverity}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="critical">Critical</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Category</label>
                    <Input
                      value={newCategory}
                      onChange={(e) => setNewCategory(e.target.value)}
                      placeholder="malware, phishing, etc."
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Source</label>
                    <Input
                      value={newSource}
                      onChange={(e) => setNewSource(e.target.value)}
                      placeholder="Source of this intelligence"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Description (optional)</label>
                    <Input
                      value={newDescription}
                      onChange={(e) => setNewDescription(e.target.value)}
                      placeholder="Additional details"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newValue.trim() || !newCategory.trim() || !newSource.trim()}>
                      Add Threat
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog open={showCheckDialog} onOpenChange={setShowCheckDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Check IP Threat</DialogTitle>
                  <DialogDescription>
                    Check if an IP address is in the threat intelligence database
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">IP Address</label>
                    <Input
                      value={checkIP}
                      onChange={(e) => setCheckIP(e.target.value)}
                      placeholder="192.168.1.1"
                    />
                  </div>
                  <Button onClick={handleCheckIP} disabled={loading || !checkIP.trim()} className="w-full">
                    Check IP
                  </Button>
                  {checkResult && (
                    <div className="mt-4 p-4 bg-muted rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className={`h-5 w-5 ${checkResult.is_threat ? 'text-destructive' : 'text-green-500'}`} />
                        <span className="font-semibold">
                          {checkResult.is_threat ? 'Threat Detected' : 'No Threat Found'}
                        </span>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">Threat Level: </span>
                          <Badge variant={checkResult.threat_level === 'critical' || checkResult.threat_level === 'high' ? 'destructive' : 'default'}>
                            {checkResult.threat_level}
                          </Badge>
                        </div>
                        {checkResult.matches.length > 0 && (
                          <div>
                            <span className="text-muted-foreground">Matches: </span>
                            <span className="font-medium">{checkResult.matches.length}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
