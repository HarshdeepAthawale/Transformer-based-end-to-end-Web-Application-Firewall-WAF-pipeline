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
import { securityRulesApi, SecurityRule } from '@/lib/api'
import { Shield, Plus, Search, FileText, AlertCircle } from 'lucide-react'

export default function SecurityRulesPage() {
  const [rules, setRules] = useState<SecurityRule[]>([])
  const [owaspRules, setOwaspRules] = useState<SecurityRule[]>([])
  const [activeTab, setActiveTab] = useState<'custom' | 'owasp'>('custom')
  const [searchQuery, setSearchQuery] = useState('')
  const [showAddDialog, setShowAddDialog] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [newName, setNewName] = useState('')
  const [newRuleType, setNewRuleType] = useState('regex')
  const [newPattern, setNewPattern] = useState('')
  const [newAppliesTo, setNewAppliesTo] = useState('all')
  const [newAction, setNewAction] = useState('block')
  const [newPriority, setNewPriority] = useState('medium')
  const [newDescription, setNewDescription] = useState('')
  const [newOwaspCategory, setNewOwaspCategory] = useState('')

  useEffect(() => {
    fetchRules()
    fetchOwaspRules()
  }, [])

  const fetchRules = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await securityRulesApi.getRules(true)
      if (response.success) setRules(response.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch security rules')
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchOwaspRules = async () => {
    try {
      const response = await securityRulesApi.getOWASPRules()
      if (response.success) setOwaspRules(response.data)
    } catch (err: any) {
      // Silently handle network errors
    }
  }

  const handleAdd = async () => {
    if (!newName.trim() || !newPattern.trim()) return

    setLoading(true)
    setError(null)
    try {
      await securityRulesApi.createRule({
        name: newName,
        rule_type: newRuleType,
        pattern: newPattern,
        applies_to: newAppliesTo,
        action: newAction,
        priority: newPriority,
        description: newDescription || undefined,
        owasp_category: newOwaspCategory || undefined,
      })
      setNewName('')
      setNewPattern('')
      setNewAppliesTo('all')
      setNewAction('block')
      setNewPriority('medium')
      setNewDescription('')
      setNewOwaspCategory('')
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

  const currentRules = activeTab === 'custom' ? rules : owaspRules
  const filteredRules = currentRules.filter(rule =>
    rule.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rule.pattern.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rule.description?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const actionColors: Record<string, string> = {
    block: 'destructive',
    log: 'secondary',
    alert: 'default',
    redirect: 'outline',
    challenge: 'secondary',
  }

  const priorityColors: Record<string, string> = {
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
                  <h1 className="text-3xl font-bold">Security Rules</h1>
                  <p className="text-muted-foreground mt-1">Manage custom and OWASP security rules</p>
                </div>
                {activeTab === 'custom' && (
                  <Button onClick={() => setShowAddDialog(true)}>
                    <Plus className="mr-2 h-4 w-4" />
                    Add Rule
                  </Button>
                )}
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
                  variant={activeTab === 'custom' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('custom')}
                >
                  Custom Rules ({rules.length})
                </Button>
                <Button
                  variant={activeTab === 'owasp' ? 'default' : 'outline'}
                  onClick={() => setActiveTab('owasp')}
                >
                  OWASP Rules ({owaspRules.length})
                </Button>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search rules..."
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
                        <th className="px-4 py-3 text-left text-sm font-medium">Type</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Pattern</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Applies To</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Action</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Priority</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredRules.map((rule) => (
                        <tr key={rule.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3">
                            <div>
                              <div className="font-medium">{rule.name}</div>
                              {rule.description && (
                                <div className="text-xs text-muted-foreground mt-1">{rule.description}</div>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant="outline">{rule.rule_type}</Badge>
                          </td>
                          <td className="px-4 py-3 font-mono text-xs text-muted-foreground max-w-xs truncate">
                            {rule.pattern}
                          </td>
                          <td className="px-4 py-3 text-sm">{rule.applies_to}</td>
                          <td className="px-4 py-3">
                            <Badge variant={actionColors[rule.action] as any}>
                              {rule.action}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={priorityColors[rule.priority] as any}>
                              {rule.priority}
                            </Badge>
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
                    <div className="p-8 text-center text-muted-foreground">No rules found</div>
                  )}
                </div>
              </Card>
            </div>

            <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Create Security Rule</DialogTitle>
                  <DialogDescription>
                    Create a custom security rule to detect and handle specific patterns
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Rule Name</label>
                    <Input
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="SQL Injection Detection"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Rule Type</label>
                      <Select value={newRuleType} onValueChange={setNewRuleType}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="regex">Regex</SelectItem>
                          <SelectItem value="string">String Match</SelectItem>
                          <SelectItem value="header">Header</SelectItem>
                          <SelectItem value="body">Body</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Priority</label>
                      <Select value={newPriority} onValueChange={setNewPriority}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="high">High</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="low">Low</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Pattern</label>
                    <Input
                      value={newPattern}
                      onChange={(e) => setNewPattern(e.target.value)}
                      placeholder=".*(union|select|insert).*"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Applies To</label>
                      <Input
                        value={newAppliesTo}
                        onChange={(e) => setNewAppliesTo(e.target.value)}
                        placeholder="all, /api/*, etc."
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Action</label>
                      <Select value={newAction} onValueChange={setNewAction}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="block">Block</SelectItem>
                          <SelectItem value="log">Log</SelectItem>
                          <SelectItem value="alert">Alert</SelectItem>
                          <SelectItem value="redirect">Redirect</SelectItem>
                          <SelectItem value="challenge">Challenge</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Description (optional)</label>
                    <Input
                      value={newDescription}
                      onChange={(e) => setNewDescription(e.target.value)}
                      placeholder="Description of this rule"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">OWASP Category (optional)</label>
                    <Input
                      value={newOwaspCategory}
                      onChange={(e) => setNewOwaspCategory(e.target.value)}
                      placeholder="A01, A02, etc."
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setShowAddDialog(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleAdd} disabled={loading || !newName.trim() || !newPattern.trim()}>
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
