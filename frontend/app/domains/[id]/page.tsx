'use client'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { dnsApi, type DNSZoneData, type DNSRecordData } from '@/lib/api'
import {
  ArrowLeft,
  Plus,
  Trash2,
  Edit2,
  Loader2,
  Cloud,
  CloudOff,
  CheckCircle,
  Clock,
  AlertTriangle,
  Lock,
  Server,
  Save,
  X,
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'

const RECORD_TYPES = ['A', 'AAAA', 'CNAME', 'MX', 'TXT', 'SRV', 'CAA', 'NS'] as const

const RECORD_TYPE_PLACEHOLDERS: Record<string, string> = {
  A: '192.168.1.1',
  AAAA: '2001:db8::1',
  CNAME: 'target.example.com',
  MX: 'mail.example.com',
  TXT: 'v=spf1 include:_spf.google.com ~all',
  SRV: '0 5 5269 xmpp-server.example.com',
  CAA: '0 issue "letsencrypt.org"',
  NS: 'ns1.example.com',
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, { bg: string; color: string; icon: React.ComponentType<any> }> = {
    active: { bg: 'var(--positivus-green-bg)', color: 'var(--positivus-green)', icon: CheckCircle },
    pending: { bg: '#fef3c7', color: '#d97706', icon: Clock },
    error: { bg: '#fee2e2', color: '#dc2626', icon: AlertTriangle },
  }
  const s = styles[status] ?? styles.pending
  const Icon = s.icon
  return (
    <span
      className="inline-flex items-center gap-1 text-xs font-medium px-2 py-1 rounded"
      style={{ backgroundColor: s.bg, color: s.color }}
    >
      <Icon size={12} />
      {status}
    </span>
  )
}

function ProxyToggle({
  proxied,
  onToggle,
  disabled,
}: {
  proxied: boolean
  onToggle: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors disabled:opacity-50"
      style={{
        backgroundColor: proxied ? '#f481200d' : 'var(--positivus-gray)',
        color: proxied ? '#f48120' : 'var(--positivus-gray-dark)',
        border: `1px solid ${proxied ? '#f4812033' : 'var(--positivus-gray)'}`,
      }}
      title={proxied ? 'Proxied - traffic flows through WAF' : 'DNS only - traffic goes directly to origin'}
    >
      {proxied ? <Cloud size={14} style={{ color: '#f48120' }} /> : <CloudOff size={14} />}
      {proxied ? 'Proxied' : 'DNS only'}
    </button>
  )
}

function AddRecordDialog({
  zoneId,
  onCreated,
}: {
  zoneId: number
  onCreated: () => void
}) {
  const [open, setOpen] = useState(false)
  const [recordType, setRecordType] = useState('A')
  const [name, setName] = useState('')
  const [content, setContent] = useState('')
  const [ttl, setTtl] = useState(300)
  const [priority, setPriority] = useState<number | ''>('')
  const [proxied, setProxied] = useState(false)
  const [comment, setComment] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const showPriority = recordType === 'MX' || recordType === 'SRV'

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim() || !content.trim()) return
    setLoading(true)
    setError(null)
    try {
      await dnsApi.createRecord(zoneId, {
        name: name.trim(),
        record_type: recordType,
        content: content.trim(),
        ttl,
        priority: showPriority && priority !== '' ? Number(priority) : undefined,
        proxied,
        comment: comment.trim() || undefined,
      })
      setOpen(false)
      setName('')
      setContent('')
      setTtl(300)
      setPriority('')
      setProxied(false)
      setComment('')
      setRecordType('A')
      onCreated()
    } catch (err: any) {
      setError(err?.message || 'Failed to create record')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button
          className="flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors"
          style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}
        >
          <Plus size={16} />
          Add Record
        </button>
      </DialogTrigger>
      <DialogContent className="border-2 max-w-lg" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <DialogHeader>
          <DialogTitle style={{ fontFamily: 'var(--font-space-grotesk)', color: 'var(--positivus-black)' }}>
            Add DNS Record
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>Type</label>
              <select
                value={recordType}
                onChange={(e) => setRecordType(e.target.value)}
                className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
                style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
              >
                {RECORD_TYPES.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="@ or subdomain"
                className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
                style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
                required
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>Content</label>
            <input
              type="text"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder={RECORD_TYPE_PLACEHOLDERS[recordType] ?? ''}
              className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)] font-mono"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
              required
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>TTL (seconds)</label>
              <input
                type="number"
                value={ttl}
                onChange={(e) => setTtl(Number(e.target.value))}
                min={60}
                className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
                style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
              />
            </div>
            {showPriority && (
              <div>
                <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>Priority</label>
                <input
                  type="number"
                  value={priority}
                  onChange={(e) => setPriority(e.target.value === '' ? '' : Number(e.target.value))}
                  min={0}
                  placeholder="10"
                  className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
                  style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
                />
              </div>
            )}
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="proxied"
                checked={proxied}
                onChange={(e) => setProxied(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="proxied" className="text-sm flex items-center gap-1" style={{ color: 'var(--positivus-black)' }}>
                <Cloud size={14} style={{ color: proxied ? '#f48120' : 'var(--positivus-gray-dark)' }} />
                Proxy through WAF
              </label>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>Comment (optional)</label>
            <input
              type="text"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="e.g. Main web server"
              className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="px-4 py-2 text-sm rounded-md border-2 transition-colors hover:bg-accent"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !name.trim() || !content.trim()}
              className="px-4 py-2 text-sm rounded-md font-medium transition-colors disabled:opacity-50"
              style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : 'Add Record'}
            </button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function RecordRow({
  record,
  zoneId,
  onUpdated,
  onDeleted,
}: {
  record: DNSRecordData
  zoneId: number
  onUpdated: () => void
  onDeleted: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [content, setContent] = useState(record.content)
  const [ttl, setTtl] = useState(record.ttl)
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    try {
      await dnsApi.updateRecord(zoneId, record.id, { content, ttl })
      setEditing(false)
      onUpdated()
    } catch {
      // Failed silently
    } finally {
      setSaving(false)
    }
  }

  const handleProxyToggle = async () => {
    try {
      await dnsApi.updateRecord(zoneId, record.id, { proxied: !record.proxied })
      onUpdated()
    } catch {
      // Failed silently
    }
  }

  const handleDelete = async () => {
    try {
      await dnsApi.deleteRecord(zoneId, record.id)
      onDeleted()
    } catch {
      // Failed silently
    }
  }

  return (
    <tr style={{ borderBottom: '1px solid var(--positivus-gray)' }}>
      <td className="px-4 py-3">
        <span
          className="inline-block px-2 py-0.5 text-xs font-mono font-medium rounded"
          style={{ backgroundColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
        >
          {record.record_type}
        </span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm font-mono" style={{ color: 'var(--positivus-black)' }}>
          {record.name}
        </span>
      </td>
      <td className="px-4 py-3">
        {editing ? (
          <input
            type="text"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="w-full px-2 py-1 border-2 rounded text-sm font-mono outline-none focus:border-[var(--positivus-green)]"
            style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
          />
        ) : (
          <span className="text-sm font-mono" style={{ color: 'var(--positivus-gray-dark)' }}>
            {record.content}
          </span>
        )}
      </td>
      <td className="px-4 py-3">
        {editing ? (
          <input
            type="number"
            value={ttl}
            onChange={(e) => setTtl(Number(e.target.value))}
            min={60}
            className="w-20 px-2 py-1 border-2 rounded text-sm outline-none focus:border-[var(--positivus-green)]"
            style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
          />
        ) : (
          <span className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
            {record.ttl === 1 ? 'Auto' : `${record.ttl}s`}
          </span>
        )}
      </td>
      <td className="px-4 py-3">
        <ProxyToggle proxied={record.proxied} onToggle={handleProxyToggle} />
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-1">
          {editing ? (
            <>
              <button
                onClick={handleSave}
                disabled={saving}
                className="p-1.5 rounded transition-colors hover:bg-accent"
                style={{ color: 'var(--positivus-green)' }}
                title="Save"
              >
                {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              </button>
              <button
                onClick={() => { setEditing(false); setContent(record.content); setTtl(record.ttl) }}
                className="p-1.5 rounded transition-colors hover:bg-accent"
                style={{ color: 'var(--positivus-gray-dark)' }}
                title="Cancel"
              >
                <X size={14} />
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setEditing(true)}
                className="p-1.5 rounded transition-colors hover:bg-accent"
                style={{ color: 'var(--positivus-gray-dark)' }}
                title="Edit"
              >
                <Edit2 size={14} />
              </button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <button
                    className="p-1.5 rounded transition-colors hover:bg-accent"
                    style={{ color: 'var(--positivus-gray-dark)' }}
                    title="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                </AlertDialogTrigger>
                <AlertDialogContent className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
                  <AlertDialogHeader>
                    <AlertDialogTitle style={{ fontFamily: 'var(--font-space-grotesk)', color: 'var(--positivus-black)' }}>
                      Delete record?
                    </AlertDialogTitle>
                    <AlertDialogDescription>
                      Delete {record.record_type} record for {record.name} pointing to {record.content}?
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel className="border-2" style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}>
                      Cancel
                    </AlertDialogCancel>
                    <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground">
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          )}
        </div>
      </td>
    </tr>
  )
}

export default function DomainDetailPage() {
  const params = useParams()
  const router = useRouter()
  const zoneId = Number(params.id)

  const [zone, setZone] = useState<DNSZoneData | null>(null)
  const [records, setRecords] = useState<DNSRecordData[]>([])
  const [loading, setLoading] = useState(true)
  const [filterType, setFilterType] = useState<string>('')

  const fetchData = useCallback(async () => {
    try {
      const [z, r] = await Promise.all([
        dnsApi.getZone(zoneId),
        dnsApi.getRecords(zoneId),
      ])
      setZone(z)
      setRecords(Array.isArray(r) ? r : [])
    } catch {
      // Zone not found
    } finally {
      setLoading(false)
    }
  }, [zoneId])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const filteredRecords = filterType
    ? records.filter((r) => r.record_type === filterType)
    : records

  if (loading) {
    return (
      <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          <Header />
          <main className="flex-1 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
          </main>
        </div>
      </div>
    )
  }

  if (!zone) {
    return (
      <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          <Header />
          <main className="flex-1 flex flex-col items-center justify-center">
            <p className="text-lg" style={{ color: 'var(--positivus-gray-dark)' }}>Domain not found</p>
            <button
              onClick={() => router.push('/domains')}
              className="mt-4 px-4 py-2 text-sm rounded-md border-2 transition-colors hover:bg-accent"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
            >
              Back to Domains
            </button>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              {/* Back + Domain Header */}
              <div>
                <button
                  onClick={() => router.push('/domains')}
                  className="flex items-center gap-1 text-sm mb-4 transition-colors hover:opacity-70"
                  style={{ color: 'var(--positivus-gray-dark)' }}
                >
                  <ArrowLeft size={16} />
                  Back to Domains
                </button>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-4">
                    <h2
                      className="text-2xl font-bold"
                      style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                    >
                      {zone.domain}
                    </h2>
                    <StatusBadge status={zone.status} />
                  </div>
                  <AddRecordDialog zoneId={zoneId} onCreated={fetchData} />
                </div>
              </div>

              {/* Zone Info Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div
                  className="p-4 border-2 rounded-md"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Server size={14} style={{ color: 'var(--positivus-gray-dark)' }} />
                    <span className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Provider</span>
                  </div>
                  <p className="text-sm font-semibold" style={{ color: 'var(--positivus-black)' }}>{zone.provider}</p>
                </div>
                <div
                  className="p-4 border-2 rounded-md"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Lock size={14} style={{ color: 'var(--positivus-gray-dark)' }} />
                    <span className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>DNSSEC</span>
                  </div>
                  <p className="text-sm font-semibold" style={{ color: 'var(--positivus-black)' }}>
                    {zone.dnssec_enabled ? 'Enabled' : 'Disabled'}
                  </p>
                </div>
                <div
                  className="p-4 border-2 rounded-md"
                  style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Cloud size={14} style={{ color: 'var(--positivus-gray-dark)' }} />
                    <span className="text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>DNS Records</span>
                  </div>
                  <p className="text-sm font-semibold" style={{ color: 'var(--positivus-black)' }}>
                    {records.length} record{records.length !== 1 ? 's' : ''}
                  </p>
                </div>
              </div>

              {/* DNS Records Table */}
              <div
                className="border-2 rounded-md overflow-hidden"
                style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
              >
                <div className="px-4 py-3 flex items-center justify-between flex-wrap gap-3" style={{ borderBottom: '2px solid var(--positivus-gray)' }}>
                  <h3
                    className="text-lg font-semibold"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    DNS Records
                  </h3>
                  <div className="flex items-center gap-2">
                    <select
                      value={filterType}
                      onChange={(e) => setFilterType(e.target.value)}
                      className="px-2 py-1 border-2 rounded text-sm outline-none"
                      style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
                    >
                      <option value="">All types</option>
                      {RECORD_TYPES.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {filteredRecords.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12">
                    <CloudOff className="h-12 w-12 mb-3" style={{ color: 'var(--positivus-gray-dark)', opacity: 0.4 }} />
                    <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                      {records.length === 0 ? 'No DNS records yet' : 'No records match this filter'}
                    </p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr style={{ borderBottom: '2px solid var(--positivus-gray)' }}>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Type</th>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Name</th>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Content</th>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>TTL</th>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Proxy</th>
                          <th className="px-4 py-3 text-left text-xs font-medium" style={{ color: 'var(--positivus-gray-dark)' }}>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRecords.map((record) => (
                          <RecordRow
                            key={record.id}
                            record={record}
                            zoneId={zoneId}
                            onUpdated={fetchData}
                            onDeleted={fetchData}
                          />
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
