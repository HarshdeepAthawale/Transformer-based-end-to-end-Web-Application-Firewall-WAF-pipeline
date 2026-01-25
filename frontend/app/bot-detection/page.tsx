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
import { botApi, BotSignature } from '@/lib/api'
import { Bot, Plus, Search, AlertCircle } from 'lucide-react'

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

  useEffect(() => {
    fetchSignatures()
  }, [])

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
      <div className="flex-1 flex flex-col overflow-hidden">
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
                            <Badge variant={categoryColors[sig.category] as any}>
                              {sig.category}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={actionColors[sig.action] as any}>
                              {sig.action}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <Badge variant={sig.is_active ? 'default' : 'secondary'}>
                                {sig.is_active ? 'Active' : 'Inactive'}
                              </Badge>
                              {sig.is_whitelisted && (
                                <Badge variant="outline">Whitelisted</Badge>
                              )}
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
