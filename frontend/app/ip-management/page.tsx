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
import { ipApi, IPEntry, IPReputation } from '@/lib/api'
import { Ban, Shield, Plus, Search, Trash2, Eye, AlertCircle } from 'lucide-react'

export default function IPManagementPage() {
  const [blacklist, setBlacklist] = useState<IPEntry[]>([])
  const [whitelist, setWhitelist] = useState<IPEntry[]>([])
  const [activeTab, setActiveTab] = useState<'blacklist' | 'whitelist'>('blacklist')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [selectedIP, setSelectedIP] = useState<string | null>(null)
  const [reputation, setReputation] = useState<IPReputation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newIP, setNewIP] = useState('')
  const [newReason, setNewReason] = useState('')
  const [durationHours, setDurationHours] = useState<number | undefined>()

  useEffect(() => {
    fetchLists()
  }, [])

  const fetchLists = async () => {
    setLoading(true)
    setError(null)
    try {
      const [blacklistRes, whitelistRes] = await Promise.all([
        ipApi.getBlacklist(1000),
        ipApi.getWhitelist(1000),
      ])
      if (blacklistRes.success) setBlacklist(blacklistRes.data)
      if (whitelistRes.success) setWhitelist(whitelistRes.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch IP lists')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = async () => {
    if (!newIP.trim()) return

    setLoading(true)
    setError(null)
    try {
      if (activeTab === 'blacklist') {
        await ipApi.addToBlacklist(newIP, newReason || undefined, durationHours, 'manual')
      } else {
        await ipApi.addToWhitelist(newIP, newReason || undefined)
      }
      setNewIP('')
      setNewReason('')
      setDurationHours(undefined)
      setShowAddDialog(false)
      fetchLists()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to add IP')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleRemove = async (ip: string) => {
    if (!confirm(`Remove ${ip} from ${activeTab}?`)) return

    setLoading(true)
    try {
      await ipApi.removeFromList(ip, activeTab)
      fetchLists()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to remove IP')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleViewReputation = async (ip: string) => {
    setSelectedIP(ip)
    setLoading(true)
    try {
      const response = await ipApi.getReputation(ip)
      if (response.success) {
        setReputation(response.data)
      }
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch reputation')
      }
    } finally {
      setLoading(false)
    }
  }

  const currentList = activeTab === 'blacklist' ? blacklist : whitelist
  const filteredList = currentList.filter(entry =>
    entry.ip.toLowerCase().includes(searchQuery.toLowerCase()) ||
    entry.reason?.toLowerCase().includes(searchQuery.toLowerCase())
  )

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
                  <h1 className="text-3xl font-bold">IP Management</h1>
                  <p className="text-muted-foreground mt-1">Manage IP blacklist and whitelist</p>
                </div>
                <Button onClick={() => setShowAddDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add IP
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

              <div className="flex gap-4">
                <Button
                  variant={activeTab === 'blacklist' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('blacklist')}
                  className="flex items-center gap-2"
                >
                  <Ban className="h-4 w-4" />
                  Blacklist ({blacklist.length})
                </Button>
                <Button
                  variant={activeTab === 'whitelist' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('whitelist')}
                  className="flex items-center gap-2"
                >
                  <Shield className="h-4 w-4" />
                  Whitelist ({whitelist.length})
                </Button>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search IPs..."
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
                        <th className="px-4 py-3 text-left text-sm font-medium">IP Address</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Reason</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Source</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Added</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Expires</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredList.map((entry) => (
                        <tr key={entry.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 font-mono text-sm">{entry.ip}</td>
                          <td className="px-4 py-3 text-sm">{entry.reason || '-'}</td>
                          <td className="px-4 py-3">
                            <Badge variant="outline">{entry.source}</Badge>
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {new Date(entry.created_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {entry.expires_at ? new Date(entry.expires_at).toLocaleString() : 'Never'}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleViewReputation(entry.ip)}
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleRemove(entry.ip)}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredList.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground">
                      No IPs found in {activeTab}
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Add IP Dialog */}
            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add IP to {activeTab === 'blacklist' ? 'Blacklist' : 'Whitelist'}</DialogTitle>
                  <DialogDescription>
                    {activeTab === 'blacklist'
                      ? 'Block this IP address from accessing your services'
                      : 'Allow this IP address to bypass security checks'}
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">IP Address</label>
                    <Input
                      value={newIP}
                      onChange={(e) => setNewIP(e.target.value)}
                      placeholder="192.168.1.1"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Reason</label>
                    <Input
                      value={newReason}
                      onChange={(e) => setNewReason(e.target.value)}
                      placeholder="Reason for adding this IP"
                    />
                  </div>
                  {activeTab === 'blacklist' && (
                    <div>
                      <label className="text-sm font-medium">Duration (hours, optional)</label>
                      <Input
                        type="number"
                        value={durationHours || ''}
                        onChange={(e) => setDurationHours(e.target.value ? parseInt(e.target.value) : undefined)}
                        placeholder="Leave empty for permanent"
                      />
                    </div>
                  )}
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newIP.trim()}>
                      Add IP
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            {/* Reputation Dialog */}
            {selectedIP && (
              <Dialog open={!!selectedIP} onOpenChange={() => setSelectedIP(null)}>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>IP Reputation: {selectedIP}</DialogTitle>
                  </DialogHeader>
                  {reputation ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Reputation Score</p>
                          <p className="text-2xl font-bold">{reputation.reputation_score.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Threat Level</p>
                          <Badge variant={reputation.threat_level === 'critical' ? 'destructive' : 'default'}>
                            {reputation.threat_level}
                          </Badge>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Total Requests</p>
                          <p className="text-lg">{reputation.total_requests}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Blocked Requests</p>
                          <p className="text-lg text-destructive">{reputation.blocked_requests}</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted-foreground">Loading reputation data...</p>
                  )}
                </DialogContent>
              </Dialog>
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
