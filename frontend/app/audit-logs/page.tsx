'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { auditApi, AuditLog } from '@/lib/api'
import { FileText, Search, Eye, AlertCircle } from 'lucide-react'

export default function AuditLogsPage() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [actionFilter, setActionFilter] = useState<string>('all')
  const [resourceTypeFilter, setResourceTypeFilter] = useState<string>('all')
  const [limit, setLimit] = useState(100)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchLogs()
  }, [actionFilter, resourceTypeFilter, limit])

  const fetchLogs = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await auditApi.getLogs({
        limit,
        action: actionFilter !== 'all' ? actionFilter : undefined,
        resource_type: resourceTypeFilter !== 'all' ? resourceTypeFilter : undefined,
      })
      if (response.success) setLogs(response.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch audit logs')
      }
    } finally {
      setLoading(false)
    }
  }

  const filteredLogs = logs.filter(log =>
    log.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    log.username?.toLowerCase().includes(searchQuery.toLowerCase()) ||
    log.ip_address?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const actionColors: Record<string, string> = {
    create: 'default',
    update: 'secondary',
    delete: 'destructive',
    view: 'outline',
    login: 'default',
    logout: 'secondary',
    block: 'destructive',
    unblock: 'default',
    config_change: 'secondary',
    rule_change: 'default',
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
                  <h1 className="text-3xl font-bold">Audit Logs</h1>
                  <p className="text-muted-foreground mt-1">View system audit and security logs</p>
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
                  <div className="text-sm text-muted-foreground">Total Logs</div>
                  <div className="text-2xl font-bold mt-1">{logs.length}</div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Successful</div>
                  <div className="text-2xl font-bold mt-1 text-green-500">
                    {logs.filter(l => l.success).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Failed</div>
                  <div className="text-2xl font-bold mt-1 text-destructive">
                    {logs.filter(l => !l.success).length}
                  </div>
                </Card>
                <Card className="p-4">
                  <div className="text-sm text-muted-foreground">Unique Users</div>
                  <div className="text-2xl font-bold mt-1">
                    {new Set(logs.map(l => l.username).filter(Boolean)).size}
                  </div>
                </Card>
              </div>

              <Card>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search logs..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-10"
                      />
                    </div>
                    <Select value={actionFilter} onValueChange={setActionFilter}>
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Actions</SelectItem>
                        <SelectItem value="create">Create</SelectItem>
                        <SelectItem value="update">Update</SelectItem>
                        <SelectItem value="delete">Delete</SelectItem>
                        <SelectItem value="login">Login</SelectItem>
                        <SelectItem value="logout">Logout</SelectItem>
                        <SelectItem value="block">Block</SelectItem>
                        <SelectItem value="config_change">Config Change</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={resourceTypeFilter} onValueChange={setResourceTypeFilter}>
                      <SelectTrigger className="w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Resources</SelectItem>
                        <SelectItem value="ip">IP</SelectItem>
                        <SelectItem value="rule">Rule</SelectItem>
                        <SelectItem value="user">User</SelectItem>
                        <SelectItem value="config">Config</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={limit.toString()} onValueChange={(v) => setLimit(parseInt(v))}>
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="50">50</SelectItem>
                        <SelectItem value="100">100</SelectItem>
                        <SelectItem value="200">200</SelectItem>
                        <SelectItem value="500">500</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-muted">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium">Timestamp</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">User</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Action</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Resource</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Description</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">IP Address</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
                        <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredLogs.map((log) => (
                        <tr key={log.id} className="border-b hover:bg-muted/50">
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {new Date(log.timestamp).toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-sm">{log.username || '-'}</td>
                          <td className="px-4 py-3">
                            <Badge variant={actionColors[log.action] as any}>
                              {log.action}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <div>
                              <div className="text-sm font-medium">{log.resource_type}</div>
                              {log.resource_id && (
                                <div className="text-xs text-muted-foreground">ID: {log.resource_id}</div>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-sm max-w-xs truncate">{log.description}</td>
                          <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                            {log.ip_address || '-'}
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={log.success ? 'default' : 'destructive'}>
                              {log.success ? 'Success' : 'Failed'}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setSelectedLog(log)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filteredLogs.length === 0 && (
                    <div className="p-8 text-center text-muted-foreground">No audit logs found</div>
                  )}
                </div>
              </Card>
            </div>

            {selectedLog && (
              <Dialog open={!!selectedLog} onOpenChange={() => setSelectedLog(null)}>
                <DialogContent className="max-w-2xl">
                  <DialogHeader>
                    <DialogTitle>Audit Log Details</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Timestamp</p>
                        <p className="font-medium">{new Date(selectedLog.timestamp).toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">User</p>
                        <p className="font-medium">{selectedLog.username || '-'}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Action</p>
                        <Badge variant={actionColors[selectedLog.action] as any}>
                          {selectedLog.action}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Resource Type</p>
                        <p className="font-medium">{selectedLog.resource_type}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">IP Address</p>
                        <p className="font-mono text-sm">{selectedLog.ip_address || '-'}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Status</p>
                        <Badge variant={selectedLog.success ? 'default' : 'destructive'}>
                          {selectedLog.success ? 'Success' : 'Failed'}
                        </Badge>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Description</p>
                      <p className="font-medium">{selectedLog.description}</p>
                    </div>
                    {selectedLog.details && (
                      <div>
                        <p className="text-sm text-muted-foreground">Details</p>
                        <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-auto">
                          {JSON.stringify(JSON.parse(selectedLog.details), null, 2)}
                        </pre>
                      </div>
                    )}
                    {selectedLog.error_message && (
                      <div>
                        <p className="text-sm text-muted-foreground">Error Message</p>
                        <p className="text-sm text-destructive">{selectedLog.error_message}</p>
                      </div>
                    )}
                  </div>
                </DialogContent>
              </Dialog>
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
