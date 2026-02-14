'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import {
  Sun,
  Moon,
  Monitor,
  Bell,
  Shield,
  Zap,
  Lock,
  Users,
  Database,
  FileText,
  ChevronRight,
  Key,
  Copy,
  Trash2,
} from 'lucide-react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import {
  settingsApi,
  wafApi,
  apiKeysApi,
  type AccountSettings,
  type ApiKeyMeta,
  type ApiKeyCreated,
  type RetentionSettings,
} from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'

const SECTIONS = [
  { id: 'general', label: 'General', icon: Sun },
  { id: 'security', label: 'Security & WAF', icon: Shield },
  { id: 'notifications', label: 'Notifications & Alerts', icon: Bell },
  { id: 'api-keys', label: 'API & Keys', icon: Key },
  { id: 'team', label: 'Team & Access', icon: Users },
  { id: 'retention', label: 'Data & Retention', icon: Database },
  { id: 'audit', label: 'Audit', icon: FileText },
] as const

type SectionId = (typeof SECTIONS)[number]['id']

export default function SettingsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [section, setSection] = useState<SectionId>('general')
  const [settings, setSettings] = useState<AccountSettings | null>(null)
  const [retention, setRetention] = useState<RetentionSettings | null>(null)
  const [wafThreshold, setWafThreshold] = useState(0.5)
  const [apiKeys, setApiKeys] = useState<ApiKeyMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle')
  const [saveMessage, setSaveMessage] = useState('')
  const [apiKeysError, setApiKeysError] = useState<string | null>(null)
  const { theme, setTheme } = useTheme()

  // Create API key dialog
  const [createKeyOpen, setCreateKeyOpen] = useState(false)
  const [newKeyName, setNewKeyName] = useState('')
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null)
  const [createKeyLoading, setCreateKeyLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [settingsRes, retentionRes] = await Promise.all([
          settingsApi.get(),
          settingsApi.getRetention(),
        ])
        if (!cancelled && settingsRes?.success && settingsRes.data) {
          setSettings(settingsRes.data)
          setTheme(settingsRes.data.theme || 'system')
        }
        if (!cancelled && retentionRes?.success && retentionRes.data) {
          setRetention(retentionRes.data)
        }
      } catch {
        if (!cancelled) setSettings(null)
      }
    }
    load()
    return () => { cancelled = true }
  }, [setTheme])

  useEffect(() => {
    let cancelled = false
    wafApi.getConfig().then((res) => {
      if (!cancelled && res?.success && res.data?.threshold != null) {
        setWafThreshold(res.data.threshold)
      }
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (section !== 'api-keys') return
    setApiKeysError(null)
    apiKeysApi.list().then((res) => {
      if (res?.success && Array.isArray(res.data)) setApiKeys(res.data)
    }).catch((e: { status?: number }) => {
      setApiKeysError(e?.status === 401 ? 'Sign in with your account to manage API keys.' : 'Failed to load API keys.')
      setApiKeys([])
    })
  }, [section])

  const updateSetting = (key: keyof AccountSettings, value: unknown) => {
    setSettings((prev) => (prev ? { ...prev, [key]: value } : { [key]: value } as AccountSettings))
  }

  const saveSettings = async () => {
    if (!settings) return
    setSaveStatus('saving')
    setSaveMessage('')
    try {
      await settingsApi.update(settings)
      setSaveStatus('success')
      setSaveMessage('Settings saved.')
    } catch (e: unknown) {
      setSaveStatus('error')
      setSaveMessage(e instanceof Error ? e.message : 'Failed to save.')
    }
    setTimeout(() => setSaveStatus('idle'), 3000)
  }

  const saveWafConfig = async () => {
    setSaveStatus('saving')
    try {
      await wafApi.updateConfig(wafThreshold)
      setSaveStatus('success')
      setSaveMessage('WAF config saved.')
    } catch (e: unknown) {
      setSaveStatus('error')
      setSaveMessage(e instanceof Error ? e.message : 'Failed to save WAF config.')
    }
    setTimeout(() => setSaveStatus('idle'), 3000)
  }

  const handleCreateKey = async () => {
    setCreateKeyLoading(true)
    setCreatedKey(null)
    try {
      const res = await apiKeysApi.create({ name: newKeyName })
      if (res?.success && res.data) {
        setCreatedKey(res.data)
        setApiKeys((prev) => [...prev, { id: res.data!.id, name: res.data!.name, prefix: res.data!.key.slice(0, 12) + '…', created_at: res.data!.created_at }])
      }
    } catch {
      setSaveMessage('Failed to create API key.')
    }
    setCreateKeyLoading(false)
  }

  const handleRevokeKey = async (keyId: string) => {
    try {
      await apiKeysApi.revoke(keyId)
      setApiKeys((prev) => prev.filter((k) => k.id !== keyId))
    } catch {
      setSaveMessage('Failed to revoke key.')
    }
  }

  const copyKey = (key: string) => {
    navigator.clipboard.writeText(key)
    setSaveMessage('Copied to clipboard.')
    setTimeout(() => setSaveMessage(''), 2000)
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <div className="p-6 max-w-4xl mx-auto">
            <div className="mb-6">
              <h2 className="text-2xl font-bold mb-1">Settings</h2>
              <p className="text-muted-foreground text-sm">Manage dashboard settings and preferences</p>
            </div>

            <Tabs value={section} onValueChange={(v) => setSection(v as SectionId)} className="space-y-6">
              <TabsList className="flex flex-wrap h-auto gap-1 bg-muted p-2">
                {SECTIONS.map(({ id, label, icon: Icon }) => (
                  <TabsTrigger key={id} value={id} className="gap-2">
                    <Icon className="h-4 w-4" />
                    {label}
                  </TabsTrigger>
                ))}
              </TabsList>

              {saveStatus !== 'idle' && (
                <p className={saveStatus === 'error' ? 'text-destructive text-sm' : 'text-muted-foreground text-sm'}>
                  {saveStatus === 'saving' ? 'Saving…' : saveMessage}
                </p>
              )}

              <ErrorBoundary>
                <TabsContent value="general" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="font-semibold">Theme</h3>
                        <p className="text-sm text-muted-foreground">Choose light, dark, or system preference</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => { setTheme('light'); updateSetting('theme', 'light') }}
                          className={`p-2 rounded-md transition-colors ${theme === 'light' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}
                          title="Light"
                        >
                          <Sun size={18} />
                        </button>
                        <button
                          type="button"
                          onClick={() => { setTheme('dark'); updateSetting('theme', 'dark') }}
                          className={`p-2 rounded-md transition-colors ${theme === 'dark' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}
                          title="Dark"
                        >
                          <Moon size={18} />
                        </button>
                        <button
                          type="button"
                          onClick={() => { setTheme('system'); updateSetting('theme', 'system') }}
                          className={`p-2 rounded-md transition-colors ${theme === 'system' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}
                          title="System"
                        >
                          <Monitor size={18} />
                        </button>
                      </div>
                    </div>
                  </Card>
                  <Card className="p-6">
                    <div className="space-y-2">
                      <Label>Default time range</Label>
                      <select
                        className="w-full max-w-xs rounded-md border border-input bg-background px-3 py-2 text-sm"
                        value={settings?.default_time_range ?? '24h'}
                        onChange={(e) => updateSetting('default_time_range', e.target.value)}
                      >
                        <option value="1h">Last 1 hour</option>
                        <option value="24h">Last 24 hours</option>
                        <option value="7d">Last 7 days</option>
                        <option value="30d">Last 30 days</option>
                      </select>
                    </div>
                  </Card>
                  <Button onClick={saveSettings} disabled={saveStatus === 'saving'}>Save</Button>
                </TabsContent>

                <TabsContent value="security" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <h3 className="font-semibold mb-2">WAF detection threshold</h3>
                    <p className="text-sm text-muted-foreground mb-4">Higher values = fewer blocks; lower = stricter.</p>
                    <div className="flex items-center gap-4">
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={wafThreshold}
                        onChange={(e) => setWafThreshold(parseFloat(e.target.value))}
                        className="flex-1 max-w-xs"
                      />
                      <span className="text-sm font-mono">{wafThreshold.toFixed(2)}</span>
                    </div>
                    <Button className="mt-4" onClick={saveWafConfig} disabled={saveStatus === 'saving'}>Save WAF config</Button>
                  </Card>
                  <Card className="p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="font-semibold">Auto block threats</h3>
                        <p className="text-sm text-muted-foreground">Automatically block detected threats</p>
                      </div>
                      <Switch
                        checked={settings?.auto_block_threats ?? true}
                        onCheckedChange={(v) => updateSetting('auto_block_threats', v)}
                      />
                    </div>
                  </Card>
                  <div className="grid gap-4 sm:grid-cols-2">
                    {[
                      { title: 'Security Rules', href: '/security-rules', desc: 'Manage WAF rules' },
                      { title: 'Geo Rules', href: '/geo-rules', desc: 'Geographic access rules' },
                      { title: 'IP Management', href: '/ip-management', desc: 'Blacklist and whitelist' },
                      { title: 'Bot Detection', href: '/bot-detection', desc: 'Bot signatures' },
                      { title: 'Threat Intel', href: '/threat-intelligence', desc: 'Threat feeds' },
                    ].map(({ title, href, desc }) => (
                      <Link key={href} href={href}>
                        <Card className="p-4 flex items-center justify-between hover:bg-muted/50 transition-colors">
                          <div>
                            <p className="font-medium">{title}</p>
                            <p className="text-sm text-muted-foreground">{desc}</p>
                          </div>
                          <ChevronRight className="h-5 w-5 text-muted-foreground" />
                        </Card>
                      </Link>
                    ))}
                  </div>
                  <Button onClick={saveSettings} disabled={saveStatus === 'saving'}>Save preferences</Button>
                </TabsContent>

                <TabsContent value="notifications" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="font-semibold">Notifications</h3>
                        <p className="text-sm text-muted-foreground">Receive alerts and updates</p>
                      </div>
                      <Switch
                        checked={settings?.notifications ?? true}
                        onCheckedChange={(v) => updateSetting('notifications', v)}
                      />
                    </div>
                  </Card>
                  <Card className="p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="font-semibold">Email alerts</h3>
                        <p className="text-sm text-muted-foreground">Send critical alerts via email</p>
                      </div>
                      <Switch
                        checked={settings?.email_alerts ?? true}
                        onCheckedChange={(v) => updateSetting('email_alerts', v)}
                      />
                    </div>
                  </Card>
                  <Card className="p-6">
                    <h3 className="font-semibold mb-2">Alert severity</h3>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label>Critical</Label>
                        <Switch
                          checked={settings?.alert_severity_critical ?? true}
                          onCheckedChange={(v) => updateSetting('alert_severity_critical', v)}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label>High</Label>
                        <Switch
                          checked={settings?.alert_severity_high ?? true}
                          onCheckedChange={(v) => updateSetting('alert_severity_high', v)}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label>Medium</Label>
                        <Switch
                          checked={settings?.alert_severity_medium ?? false}
                          onCheckedChange={(v) => updateSetting('alert_severity_medium', v)}
                        />
                      </div>
                    </div>
                  </Card>
                  <Card className="p-6">
                    <Label className="mb-2 block">Webhook URL (optional)</Label>
                    <Input
                      placeholder="https://..."
                      value={settings?.webhook_url ?? ''}
                      onChange={(e) => updateSetting('webhook_url', e.target.value)}
                    />
                  </Card>
                  <Card className="p-6">
                    <Label className="mb-2 block">Alert recipient emails (optional)</Label>
                    <Input
                      placeholder="admin@example.com, ops@example.com"
                      value={settings?.alert_emails ?? ''}
                      onChange={(e) => updateSetting('alert_emails', e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground mt-2">Comma-separated; used for email alerts when Email alerts is on. Leave empty to use admin users.</p>
                  </Card>
                  <Button onClick={saveSettings} disabled={saveStatus === 'saving'}>Save</Button>
                </TabsContent>

                <TabsContent value="api-keys" className="space-y-6 mt-0">
                  {apiKeysError && (
                    <Card className="p-4 border-destructive/50 bg-destructive/10">
                      <p className="text-sm text-destructive">{apiKeysError}</p>
                    </Card>
                  )}
                  <Card className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold">API keys</h3>
                      <Button onClick={() => { setCreatedKey(null); setNewKeyName(''); setCreateKeyOpen(true) }}>Create key</Button>
                    </div>
                    <ul className="space-y-2">
                      {apiKeys.length === 0 && !apiKeysError && <li className="text-muted-foreground text-sm">No API keys yet.</li>}
                      {apiKeys.map((k) => (
                        <li key={k.id} className="flex items-center justify-between py-2 border-b last:border-0">
                          <div>
                            <p className="font-medium">{k.name}</p>
                            <p className="text-xs text-muted-foreground font-mono">{k.prefix} · {k.created_at?.slice(0, 10)}</p>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleRevokeKey(k.id)}><Trash2 className="h-4 w-4" /></Button>
                        </li>
                      ))}
                    </ul>
                  </Card>

                  <Dialog open={createKeyOpen} onOpenChange={setCreateKeyOpen}>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>{createdKey ? 'API key created' : 'Create API key'}</DialogTitle>
                        <DialogDescription>
                          {createdKey
                            ? 'Copy the key now; it will not be shown again.'
                            : 'Give this key a name to identify it later.'}
                        </DialogDescription>
                      </DialogHeader>
                      {createdKey ? (
                        <div className="space-y-2">
                          <div className="flex gap-2">
                            <Input readOnly value={createdKey.key} className="font-mono text-sm" />
                            <Button variant="outline" size="icon" onClick={() => copyKey(createdKey.key)}><Copy className="h-4 w-4" /></Button>
                          </div>
                          <Button className="w-full" onClick={() => { setCreateKeyOpen(false); setCreatedKey(null) }}>Done</Button>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div>
                            <Label>Name</Label>
                            <Input
                              placeholder="e.g. Production"
                              value={newKeyName}
                              onChange={(e) => setNewKeyName(e.target.value)}
                            />
                          </div>
                          <Button className="w-full" onClick={handleCreateKey} disabled={createKeyLoading}>
                            {createKeyLoading ? 'Creating…' : 'Create'}
                          </Button>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                </TabsContent>

                <TabsContent value="team" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <h3 className="font-semibold mb-2">Roles</h3>
                    <ul className="text-sm text-muted-foreground space-y-1 mb-4">
                      <li><strong className="text-foreground">Admin</strong> — Full access, user and settings management.</li>
                      <li><strong className="text-foreground">Operator</strong> — Manage rules, IPs, and security config.</li>
                      <li><strong className="text-foreground">Viewer</strong> — Read-only access to dashboard and logs.</li>
                    </ul>
                    <Link href="/users">
                      <Button>Manage users</Button>
                    </Link>
                  </Card>
                </TabsContent>

                <TabsContent value="retention" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <h3 className="font-semibold mb-4">Data retention (read-only)</h3>
                    <p className="text-sm text-muted-foreground mb-4">Configured via server environment. Contact your administrator to change.</p>
                    {retention && (
                      <ul className="space-y-2 text-sm">
                        <li>Metrics: {retention.metrics_days} days</li>
                        <li>Traffic: {retention.traffic_days} days</li>
                        <li>Alerts: {retention.alerts_days} days</li>
                        <li>Threats: {retention.threats_days} days</li>
                      </ul>
                    )}
                  </Card>
                </TabsContent>

                <TabsContent value="audit" className="space-y-6 mt-0">
                  <Card className="p-6">
                    <h3 className="font-semibold mb-2">Audit logs</h3>
                    <p className="text-sm text-muted-foreground mb-4">View all configuration changes and security events.</p>
                    <Link href="/audit-logs">
                      <Button>View audit logs</Button>
                    </Link>
                  </Card>
                </TabsContent>
              </ErrorBoundary>
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  )
}
