'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { toxicCombinationsApi, type ToxicCombinationData } from '@/lib/api'
import {
  AlertTriangle,
  Shield,
  Loader2,
  CheckCircle,
  Clock,
  Eye,
  Globe,
  RefreshCw,
  ChevronRight,
} from 'lucide-react'

const SEVERITY_STYLES: Record<string, { bg: string; color: string; border: string }> = {
  critical: { bg: '#fee2e2', color: '#dc2626', border: '#fca5a5' },
  high: { bg: '#fff7ed', color: '#ea580c', border: '#fdba74' },
  medium: { bg: '#fef3c7', color: '#d97706', border: '#fcd34d' },
  low: { bg: 'var(--positivus-green-bg)', color: 'var(--positivus-green)', border: 'var(--positivus-green)' },
}

const PATTERN_LABELS: Record<string, string> = {
  admin_probing: 'Admin Endpoint Probing',
  debug_probing: 'Debug Parameter Exposure',
  sqli_success: 'SQL Injection with Success',
  coordinated_evasion: 'Coordinated Rate Limit Evasion',
  idor_detection: 'Predictable ID Enumeration (IDOR)',
  payment_anomaly: 'Payment Flow Anomaly',
}

function SeverityBadge({ severity }: { severity: string }) {
  const s = SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.medium
  return (
    <span
      className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-1 rounded uppercase"
      style={{ backgroundColor: s.bg, color: s.color }}
    >
      {severity}
    </span>
  )
}

function ToxicCombinationCard({
  item,
  onStatusChange,
}: {
  item: ToxicCombinationData
  onStatusChange: (id: number, status: string) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className="border-2 rounded-md p-5 transition-colors"
      style={{
        backgroundColor: 'var(--positivus-white)',
        borderColor: 'var(--positivus-gray)',
        borderLeftColor: SEVERITY_STYLES[item.severity]?.color ?? 'var(--positivus-gray)',
        borderLeftWidth: 4,
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <SeverityBadge severity={item.severity} />
            <h3
              className="font-semibold text-base"
              style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              {PATTERN_LABELS[item.pattern_name] ?? item.pattern_name}
            </h3>
          </div>
          <p className="text-sm mb-3" style={{ color: 'var(--positivus-gray-dark)' }}>
            {item.description}
          </p>

          <div className="flex items-center gap-4 text-xs flex-wrap" style={{ color: 'var(--positivus-gray-dark)' }}>
            {item.affected_path && (
              <span className="flex items-center gap-1">
                <Globe size={12} />
                <code className="font-mono">{item.affected_path}</code>
              </span>
            )}
            <span className="flex items-center gap-1">
              <Eye size={12} />
              {item.event_count} events
            </span>
            {item.first_seen && (
              <span className="flex items-center gap-1">
                <Clock size={12} />
                {new Date(item.first_seen).toLocaleString()}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {item.status === 'active' && (
            <button
              onClick={() => onStatusChange(item.id, 'investigating')}
              className="px-3 py-1 text-xs rounded border-2 transition-colors hover:bg-accent"
              style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
            >
              Investigate
            </button>
          )}
          {item.status === 'investigating' && (
            <button
              onClick={() => onStatusChange(item.id, 'resolved')}
              className="px-3 py-1 text-xs rounded font-medium transition-colors"
              style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}
            >
              Resolve
            </button>
          )}
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-1.5 rounded transition-colors hover:bg-accent"
            style={{ color: 'var(--positivus-gray-dark)' }}
          >
            <ChevronRight size={16} className={`transition-transform ${expanded ? 'rotate-90' : ''}`} />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--positivus-gray)' }}>
          <h4 className="text-xs font-semibold mb-2 uppercase" style={{ color: 'var(--positivus-gray-dark)' }}>
            Contributing Signals
          </h4>
          <div className="space-y-2">
            {item.signals.map((signal, i) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                <span
                  className="inline-block px-1.5 py-0.5 text-xs rounded font-mono shrink-0"
                  style={{ backgroundColor: 'var(--positivus-gray)', color: 'var(--positivus-gray-dark)' }}
                >
                  {signal.type}
                </span>
                <span style={{ color: 'var(--positivus-black)' }}>{signal.detail}</span>
              </div>
            ))}
          </div>

          {item.source_ips.length > 0 && (
            <div className="mt-3">
              <h4 className="text-xs font-semibold mb-1 uppercase" style={{ color: 'var(--positivus-gray-dark)' }}>
                Source IPs
              </h4>
              <div className="flex flex-wrap gap-1">
                {item.source_ips.slice(0, 10).map((ip) => (
                  <span key={ip} className="text-xs font-mono px-2 py-0.5 rounded" style={{ backgroundColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}>
                    {ip}
                  </span>
                ))}
                {item.source_ips.length > 10 && (
                  <span className="text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                    +{item.source_ips.length - 10} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function SecurityInsightsPage() {
  const [combinations, setCombinations] = useState<ToxicCombinationData[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [combResult, statsResult] = await Promise.all([
        toxicCombinationsApi.getAll(),
        toxicCombinationsApi.getStats(),
      ])
      if (combResult.success) setCombinations(combResult.data)
      if (statsResult.success) setStats(statsResult.data)
    } catch {
      setCombinations([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const handleStatusChange = async (id: number, status: string) => {
    try {
      await toxicCombinationsApi.updateStatus(id, status)
      await fetchData()
    } catch {
      // Failed silently
    }
  }

  return (
    <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2
                    className="text-2xl font-bold"
                    style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                  >
                    Security Insights
                  </h2>
                  <p className="text-sm mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>
                    Toxic combinations: when small signals converge into security incidents
                  </p>
                </div>
                <button
                  onClick={fetchData}
                  className="flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium border-2 transition-colors hover:bg-accent"
                  style={{ borderColor: 'var(--positivus-gray)', color: 'var(--positivus-black)' }}
                >
                  <RefreshCw size={14} />
                  Refresh
                </button>
              </div>

              {/* Stats Cards */}
              {stats && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {['critical', 'high', 'medium', 'low'].map((sev) => (
                    <div
                      key={sev}
                      className="p-4 border-2 rounded-md"
                      style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
                    >
                      <p className="text-xs font-medium uppercase" style={{ color: SEVERITY_STYLES[sev]?.color ?? 'var(--positivus-gray-dark)' }}>
                        {sev}
                      </p>
                      <p className="text-2xl font-bold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                        {stats.by_severity?.[sev] ?? 0}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* Combinations List */}
              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                </div>
              ) : combinations.length === 0 ? (
                <div
                  className="flex flex-col items-center justify-center py-16 border-2 border-dashed rounded-md"
                  style={{ borderColor: 'var(--positivus-gray)', backgroundColor: 'var(--positivus-white)' }}
                >
                  <CheckCircle className="h-16 w-16 mb-4" style={{ color: 'var(--positivus-green)' }} />
                  <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    No toxic combinations detected
                  </h3>
                  <p className="text-sm text-center max-w-md" style={{ color: 'var(--positivus-gray-dark)' }}>
                    The system continuously monitors for multi-signal threat patterns. No converging threats found in the current window.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {combinations.map((item) => (
                    <ToxicCombinationCard key={item.id} item={item} onStatusChange={handleStatusChange} />
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
