'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { managedRulesApi, ManagedRulePack } from '@/lib/api'
import { Shield, RefreshCw, AlertCircle, Package } from 'lucide-react'

export default function ManagedRulesPage() {
  const pathname = usePathname()
  const [packs, setPacks] = useState<ManagedRulePack[]>([])
  const [feedUrlConfigured, setFeedUrlConfigured] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(false)
  const [syncing, setSyncing] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setPacks([])
    setError(null)
    const load = async () => {
      try {
        const configRes = await managedRulesApi.getConfig()
        if (configRes.success && configRes.data) {
          setFeedUrlConfigured(configRes.data.feed_url_configured)
        }
      } catch {
        setFeedUrlConfigured(false)
      }
      fetchPacks()
    }
    load()
  }, [pathname])

  const fetchPacks = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await managedRulesApi.getPacks(false)
      if (res.success && res.data) setPacks(res.data)
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to fetch rule packs')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleToggle = async (packId: string, enabled: boolean) => {
    setError(null)
    try {
      await managedRulesApi.togglePack(packId, enabled)
      fetchPacks()
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Failed to update pack')
      }
    }
  }

  const handleSync = async (packId?: string) => {
    setError(null)
    setSyncing(packId ?? 'default')
    try {
      const res = await managedRulesApi.syncNow(packId)
      if (res.success) {
        fetchPacks()
      } else {
        setError((res as any)?.message || 'Sync failed')
      }
    } catch (err: any) {
      if (!err?.isNetworkError) {
        setError(err?.message || 'Sync failed')
      }
    } finally {
      setSyncing(null)
    }
  }

  const formatDate = (s: string | null) => {
    if (!s) return '—'
    try {
      const d = new Date(s)
      return d.toLocaleString()
    } catch {
      return s
    }
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
                  <h1 className="text-3xl font-bold">Managed Rules</h1>
                  <p className="text-muted-foreground mt-1">
                    OWASP CRS and feed-synced rule packs — enable/disable and sync from config
                  </p>
                </div>
                <Button
                  onClick={() => handleSync()}
                  disabled={!!syncing}
                  variant="outline"
                >
                  <RefreshCw className={`mr-2 h-4 w-4 ${syncing ? 'animate-spin' : ''}`} />
                  Sync now
                </Button>
              </div>

              {(error || feedUrlConfigured === false) && (
                <Card className="p-4 bg-destructive/10 border-destructive">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="h-4 w-4 text-destructive" />
                    <p className="text-sm text-destructive">
                      {error ?? 'MANAGED_RULES_FEED_URL is not set'}
                    </p>
                  </div>
                </Card>
              )}

              <Card>
                <div className="p-4 border-b">
                  <p className="text-sm text-muted-foreground">
                    Packs are synced from the configured feed URL. Enable a pack to apply its rules at the gateway.
                  </p>
                </div>
                <div className="overflow-x-auto">
                  {loading ? (
                    <div className="p-8 text-center text-muted-foreground">Loading packs…</div>
                  ) : packs.length === 0 ? (
                    <div className="p-8 text-center text-muted-foreground flex flex-col items-center gap-2">
                      <Package className="h-10 w-10 opacity-50" />
                      <p>No rule packs yet.</p>
                      <p className="text-sm">
                        {feedUrlConfigured === true
                          ? 'Click Sync now to create a pack from the configured feed.'
                          : 'Set MANAGED_RULES_FEED_URL and click "Sync now" to create a pack.'}
                      </p>
                      <Button onClick={() => handleSync()} disabled={!!syncing} variant="outline" className="mt-2">
                        <RefreshCw className={`mr-2 h-4 w-4 ${syncing ? 'animate-spin' : ''}`} />
                        Sync now
                      </Button>
                    </div>
                  ) : (
                    <table className="w-full">
                      <thead className="bg-muted">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Pack ID</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Version</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Last synced</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Rules</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Enabled</th>
                          <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {packs.map((pack) => (
                          <tr key={pack.pack_id} className="border-b hover:bg-muted/50">
                            <td className="px-4 py-3 font-medium">{pack.name}</td>
                            <td className="px-4 py-3">
                              <Badge variant="outline">{pack.pack_id}</Badge>
                            </td>
                            <td className="px-4 py-3 text-sm text-muted-foreground">{pack.version ?? '—'}</td>
                            <td className="px-4 py-3 text-sm text-muted-foreground">{formatDate(pack.last_synced_at)}</td>
                            <td className="px-4 py-3">{pack.rule_count ?? 0}</td>
                            <td className="px-4 py-3">
                              <Switch
                                checked={pack.enabled}
                                onCheckedChange={(checked) => handleToggle(pack.pack_id, checked)}
                              />
                            </td>
                            <td className="px-4 py-3">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => handleSync(pack.pack_id)}
                                disabled={syncing === pack.pack_id}
                              >
                                <RefreshCw className={`h-4 w-4 ${syncing === pack.pack_id ? 'animate-spin' : ''}`} />
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </Card>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
