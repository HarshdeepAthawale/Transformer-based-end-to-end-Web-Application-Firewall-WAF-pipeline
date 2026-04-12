'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { useDomain } from '@/contexts/domain-context'
import { dnsApi, type DNSZoneData } from '@/lib/api'
import {
  Globe,
  Plus,
  Trash2,
  ExternalLink,
  Shield,
  Loader2,
  Server,
  Lock,
  CheckCircle,
  Clock,
  AlertTriangle,
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

function AddDomainDialog({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  const [domain, setDomain] = useState('')
  const [provider, setProvider] = useState('manual')
  const [dnssec, setDnssec] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!domain.trim()) return
    setLoading(true)
    setError(null)
    try {
      await dnsApi.createZone({ domain: domain.trim(), provider, dnssec_enabled: dnssec })
      setOpen(false)
      setDomain('')
      setProvider('manual')
      setDnssec(false)
      onCreated()
    } catch (err: any) {
      setError(err?.message || 'Failed to add domain')
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
          Add Domain
        </button>
      </DialogTrigger>
      <DialogContent className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
        <DialogHeader>
          <DialogTitle style={{ fontFamily: 'var(--font-space-grotesk)', color: 'var(--positivus-black)' }}>
            Add a new domain
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-4">
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>
              Domain
            </label>
            <input
              type="text"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="example.com"
              className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: 'var(--positivus-black)' }}>
              DNS Provider
            </label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full px-3 py-2 border-2 rounded-md text-sm outline-none focus:border-[var(--positivus-green)]"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)', backgroundColor: 'var(--positivus-white)' }}
            >
              <option value="manual">Manual</option>
              <option value="cloudflare">Cloudflare</option>
              <option value="route53">AWS Route 53</option>
              <option value="powerdns">PowerDNS</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="dnssec"
              checked={dnssec}
              onChange={(e) => setDnssec(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="dnssec" className="text-sm" style={{ color: 'var(--positivus-black)' }}>
              Enable DNSSEC
            </label>
          </div>
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
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
              disabled={loading || !domain.trim()}
              className="px-4 py-2 text-sm rounded-md font-medium transition-colors disabled:opacity-50"
              style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : 'Add Domain'}
            </button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function DomainCard({
  zone,
  onDelete,
}: {
  zone: DNSZoneData
  onDelete: (id: number) => void
}) {
  const router = useRouter()

  return (
    <div
      className="p-6 border-2 rounded-md group transition-colors hover:border-[var(--positivus-green)]"
      style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="p-2 rounded-md shrink-0" style={{ backgroundColor: 'var(--positivus-green-bg)' }}>
            <Globe size={20} style={{ color: 'var(--positivus-green)' }} />
          </div>
          <div className="min-w-0">
            <h3
              className="font-semibold text-lg truncate"
              style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              {zone.domain}
            </h3>
            <StatusBadge status={zone.status} />
          </div>
        </div>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <button
              className="p-2 rounded-md opacity-0 group-hover:opacity-100 transition-opacity hover:bg-accent"
              style={{ color: 'var(--positivus-gray-dark)' }}
              title="Delete domain"
            >
              <Trash2 size={16} />
            </button>
          </AlertDialogTrigger>
          <AlertDialogContent className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
            <AlertDialogHeader>
              <AlertDialogTitle style={{ fontFamily: 'var(--font-space-grotesk)', color: 'var(--positivus-black)' }}>
                Delete {zone.domain}?
              </AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently delete the domain and all its DNS records. This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel
                className="border-2"
                style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
              >
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={() => onDelete(zone.id)}
                className="bg-destructive text-destructive-foreground"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
          <Server size={14} />
          <span>Provider: {zone.provider}</span>
        </div>
        <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
          <Lock size={14} />
          <span>DNSSEC: {zone.dnssec_enabled ? 'Enabled' : 'Disabled'}</span>
        </div>
        {zone.created_at && (
          <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
            <Clock size={14} />
            <span>Added {new Date(zone.created_at).toLocaleDateString()}</span>
          </div>
        )}
      </div>

      <button
        onClick={() => router.push(`/domains/${zone.id}`)}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium rounded-md border-2 transition-colors hover:bg-accent"
        style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
      >
        <ExternalLink size={14} />
        Manage DNS Records
      </button>
    </div>
  )
}

export default function DomainsPage() {
  const [zones, setZones] = useState<DNSZoneData[]>([])
  const [loading, setLoading] = useState(true)
  const { refreshDomains } = useDomain()

  const fetchZones = async () => {
    setLoading(true)
    try {
      const data = await dnsApi.getZones()
      setZones(Array.isArray(data) ? data : [])
    } catch {
      setZones([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchZones()
  }, [])

  const handleDelete = async (id: number) => {
    try {
      await dnsApi.deleteZone(id)
      await fetchZones()
      await refreshDomains()
    } catch {
      // Failed silently
    }
  }

  const handleCreated = async () => {
    await fetchZones()
    await refreshDomains()
  }

  return (
    <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2
                    className="text-2xl font-bold"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    Domains
                  </h2>
                  <p className="text-sm mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>
                    Manage your domains and DNS records
                  </p>
                </div>
                <AddDomainDialog onCreated={handleCreated} />
              </div>

              {/* Domain Grid */}
              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                  <span className="ml-2" style={{ color: 'var(--positivus-gray-dark)' }}>Loading domains...</span>
                </div>
              ) : zones.length === 0 ? (
                <div
                  className="flex flex-col items-center justify-center py-16 border-2 border-dashed rounded-md"
                  style={{ borderColor: 'var(--positivus-gray)', backgroundColor: 'var(--positivus-white)' }}
                >
                  <Shield className="h-16 w-16 mb-4" style={{ color: 'var(--positivus-gray-dark)', opacity: 0.4 }} />
                  <h3
                    className="text-lg font-semibold mb-2"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    No domains yet
                  </h3>
                  <p className="text-sm mb-4 text-center max-w-md" style={{ color: 'var(--positivus-gray-dark)' }}>
                    Add your first domain to start protecting it with the WAF. You can configure DNS records and enable proxy protection.
                  </p>
                  <AddDomainDialog onCreated={handleCreated} />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {zones.map((zone) => (
                    <DomainCard key={zone.id} zone={zone} onDelete={handleDelete} />
                  ))}
                </div>
              )}
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
