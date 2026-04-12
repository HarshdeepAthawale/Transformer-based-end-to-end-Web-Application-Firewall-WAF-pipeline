'use client'

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { ErrorBoundary } from '@/components/error-boundary'
import { emergencyRulesApi, type EmergencyRuleData } from '@/lib/api'
import {
  Zap,
  Plus,
  Trash2,
  Loader2,
  Shield,
  ExternalLink,
  ToggleLeft,
  ToggleRight,
  Rocket,
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

function RuleCard({
  rule,
  onToggle,
  onDelete,
}: {
  rule: EmergencyRuleData
  onToggle: (id: number, enabled: boolean) => void
  onDelete: (id: number) => void
}) {
  return (
    <div
      className="p-5 border-2 rounded-md"
      style={{
        backgroundColor: 'var(--positivus-white)',
        borderColor: rule.enabled ? 'var(--positivus-green)' : 'var(--positivus-gray)',
        opacity: rule.enabled ? 1 : 0.7,
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <Zap size={16} style={{ color: rule.enabled ? 'var(--positivus-green)' : 'var(--positivus-gray-dark)' }} />
            <h3
              className="font-semibold"
              style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              {rule.name}
            </h3>
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded uppercase"
              style={{
                backgroundColor: rule.severity === 'critical' ? '#fee2e2' : '#fff7ed',
                color: rule.severity === 'critical' ? '#dc2626' : '#ea580c',
              }}
            >
              {rule.severity}
            </span>
          </div>

          {rule.description && (
            <p className="text-sm mb-2" style={{ color: 'var(--positivus-gray-dark)' }}>
              {rule.description}
            </p>
          )}

          <div className="flex items-center gap-4 text-xs flex-wrap" style={{ color: 'var(--positivus-gray-dark)' }}>
            <span>Action: <strong>{rule.action}</strong></span>
            <span>Hits: <strong>{rule.hit_count.toLocaleString()}</strong></span>
            {rule.cves.length > 0 && (
              <span className="flex items-center gap-1">
                {rule.cves.map((cve) => (
                  <code key={cve} className="font-mono px-1 py-0.5 rounded" style={{ backgroundColor: 'var(--positivus-gray)' }}>
                    {cve}
                  </code>
                ))}
              </span>
            )}
          </div>

          {rule.patterns.length > 0 && (
            <div className="mt-2 space-y-1">
              {rule.patterns.map((p, i) => (
                <div key={i} className="text-xs font-mono" style={{ color: 'var(--positivus-gray-dark)' }}>
                  {p.field} {p.op} <span style={{ color: 'var(--positivus-black)' }}>"{p.value}"</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onToggle(rule.id, !rule.enabled)}
            className="p-2 rounded transition-colors hover:bg-accent"
            title={rule.enabled ? 'Disable' : 'Enable'}
          >
            {rule.enabled ? (
              <ToggleRight size={20} style={{ color: 'var(--positivus-green)' }} />
            ) : (
              <ToggleLeft size={20} style={{ color: 'var(--positivus-gray-dark)' }} />
            )}
          </button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <button className="p-2 rounded transition-colors hover:bg-accent" style={{ color: 'var(--positivus-gray-dark)' }}>
                <Trash2 size={16} />
              </button>
            </AlertDialogTrigger>
            <AlertDialogContent className="border-2" style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}>
              <AlertDialogHeader>
                <AlertDialogTitle style={{ fontFamily: 'var(--font-space-grotesk)' }}>Delete emergency rule?</AlertDialogTitle>
                <AlertDialogDescription>This will permanently delete "{rule.name}". This action cannot be undone.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel className="border-2" style={{ borderColor: 'var(--positivus-gray)' }}>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => onDelete(rule.id)} className="bg-destructive text-destructive-foreground">Delete</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>
    </div>
  )
}

function TemplateCard({ template, onDeploy }: { template: any; onDeploy: (key: string) => void }) {
  return (
    <div
      className="p-4 border-2 rounded-md flex items-center justify-between gap-4"
      style={{ backgroundColor: 'var(--positivus-white)', borderColor: 'var(--positivus-gray)' }}
    >
      <div className="min-w-0">
        <h4 className="font-medium text-sm" style={{ color: 'var(--positivus-black)' }}>{template.name}</h4>
        <div className="flex gap-2 mt-1 flex-wrap">
          {template.cves?.map((cve: string) => (
            <code key={cve} className="text-xs font-mono px-1 py-0.5 rounded" style={{ backgroundColor: 'var(--positivus-gray)', color: 'var(--positivus-gray-dark)' }}>
              {cve}
            </code>
          ))}
        </div>
      </div>
      <button
        onClick={() => onDeploy(template.key)}
        className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded-md shrink-0 transition-colors"
        style={{ backgroundColor: 'var(--positivus-green)', color: 'var(--positivus-black)' }}
      >
        <Rocket size={12} />
        Deploy
      </button>
    </div>
  )
}

export default function EmergencyRulesPage() {
  const [rules, setRules] = useState<EmergencyRuleData[]>([])
  const [templates, setTemplates] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [rulesResult, templatesResult] = await Promise.all([
        emergencyRulesApi.getAll(),
        emergencyRulesApi.getTemplates(),
      ])
      if (rulesResult.success) setRules(rulesResult.data)
      if (templatesResult.success) setTemplates(templatesResult.data)
    } catch {
      setRules([])
      setTemplates([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const handleToggle = async (id: number, enabled: boolean) => {
    try {
      await emergencyRulesApi.toggle(id, enabled)
      await fetchData()
    } catch { /* silent */ }
  }

  const handleDelete = async (id: number) => {
    try {
      await emergencyRulesApi.delete(id)
      await fetchData()
    } catch { /* silent */ }
  }

  const handleDeploy = async (templateKey: string) => {
    try {
      await emergencyRulesApi.deployTemplate(templateKey)
      await fetchData()
    } catch { /* silent */ }
  }

  return (
    <div className="flex h-screen text-foreground" style={{ backgroundColor: 'var(--positivus-gray)' }}>
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Header />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                  Emergency Rules
                </h2>
                <p className="text-sm mt-1" style={{ color: 'var(--positivus-gray-dark)' }}>
                  Rapid-deploy rules for zero-day threat response. Checked before ML inference for instant blocking.
                </p>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="h-8 w-8 animate-spin" style={{ color: 'var(--positivus-green)' }} />
                </div>
              ) : (
                <>
                  {/* Active Rules */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Active Rules ({rules.length})
                    </h3>
                    {rules.length === 0 ? (
                      <div
                        className="flex flex-col items-center justify-center py-12 border-2 border-dashed rounded-md"
                        style={{ borderColor: 'var(--positivus-gray)', backgroundColor: 'var(--positivus-white)' }}
                      >
                        <Shield className="h-12 w-12 mb-3" style={{ color: 'var(--positivus-gray-dark)', opacity: 0.4 }} />
                        <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                          No emergency rules deployed. Use templates below for quick deployment.
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {rules.map((rule) => (
                          <RuleCard key={rule.id} rule={rule} onToggle={handleToggle} onDelete={handleDelete} />
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Zero-Day Templates */}
                  <div>
                    <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                      Zero-Day Pattern Templates
                    </h3>
                    <p className="text-sm mb-3" style={{ color: 'var(--positivus-gray-dark)' }}>
                      Pre-built detection patterns for known zero-day exploits. Deploy instantly with one click.
                    </p>
                    <div className="space-y-2">
                      {templates.map((template) => (
                        <TemplateCard key={template.key} template={template} onDeploy={handleDeploy} />
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
