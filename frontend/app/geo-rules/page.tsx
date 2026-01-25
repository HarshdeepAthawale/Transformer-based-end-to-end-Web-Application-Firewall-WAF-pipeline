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
import { geoApi, GeoRule, GeoStats } from '@/lib/api'
import { Globe, Plus, Search, MapPin, AlertCircle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function GeoRulesPage() {
  const [rules, setRules] = useState<GeoRule[]>([])
  const [stats, setStats] = useState<GeoStats[]>([])
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
    fetchStats()
  }, [])

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
      const response = await geoApi.getStats('24h')
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

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <div className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Geographic Statistics (24h)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={Array.isArray(stats) ? stats.slice(0, 10) : []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="country_code" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="total_requests" fill="#8884d8" name="Total Requests" />
                        <Bar dataKey="blocked_requests" fill="#82ca9d" name="Blocked" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card>
                  <div className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Top Threat Countries</h3>
                    <div className="space-y-2">
                      {Array.isArray(stats) && stats.length > 0 ? (
                        stats
                          .sort((a, b) => b.threat_count - a.threat_count)
                          .slice(0, 5)
                          .map((stat) => (
                            <div key={stat.country_code} className="flex items-center justify-between p-2 bg-muted rounded">
                              <div className="flex items-center gap-2">
                                <MapPin className="h-4 w-4" />
                                <span className="font-medium">{stat.country_name}</span>
                              </div>
                              <Badge variant="destructive">{stat.threat_count} threats</Badge>
                            </div>
                          ))
                      ) : (
                        <div className="p-4 text-center text-muted-foreground text-sm">No statistics available</div>
                      )}
                    </div>
                  </div>
                </Card>
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
