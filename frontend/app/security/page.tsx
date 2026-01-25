'use client'

import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Card } from '@/components/ui/card'
import { CheckCircle, AlertCircle, Lock, Key, Shield, Loader2, XCircle } from 'lucide-react'
import { securityApi, SecurityCheck } from '@/lib/api'
import { ErrorBoundary } from '@/components/error-boundary'

export default function SecurityPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [securityChecks, setSecurityChecks] = useState<SecurityCheck[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [securityScore, setSecurityScore] = useState(0)

  // Fetch security data
  useEffect(() => {
    const fetchSecurityData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const [checksResponse, scoreResponse] = await Promise.all([
          securityApi.getChecks(),
          securityApi.getComplianceScore(),
        ])

        if (checksResponse.success) {
          setSecurityChecks(checksResponse.data)
        }

        if (scoreResponse.success) {
          setSecurityScore(scoreResponse.data.score)
        }
      } catch (err: any) {
        // Only show error if it's not a network error (backend not running)
        if (err?.isNetworkError) {
          console.debug('[SecurityPage] Backend not available')
          setError(null) // Don't show error for network issues
        } else {
          console.error('[SecurityPage] Failed to fetch security data:', err)
          setError('Failed to load security data')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchSecurityData()
  }, [])

  // Calculate security statistics
  const passedChecks = securityChecks.filter(check => check.status === 'pass').length
  const totalChecks = securityChecks.length

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Security Settings</h2>
                <p className="text-muted-foreground">Manage security policies and compliance settings</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <Loader2 className="animate-spin h-8 w-8 text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Loading security data...</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
          <main className="flex-1 overflow-auto">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Security Settings</h2>
                <p className="text-muted-foreground">Manage security policies and compliance settings</p>
              </div>
              <div className="flex items-center justify-center py-16">
                <AlertCircle className="h-8 w-8 text-destructive" />
                <span className="ml-2 text-destructive">{error}</span>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header timeRange={timeRange} onTimeRangeChange={setTimeRange} />
        <main className="flex-1 overflow-auto">
          <ErrorBoundary>
            <div className="p-6 space-y-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Security Settings</h2>
              <p className="text-muted-foreground">Manage security policies and compliance settings</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <Shield className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Security Score</p>
                    <p className="text-3xl font-bold">{securityScore}/100</p>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <Lock className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Policies Active</p>
                    <p className="text-3xl font-bold">{passedChecks}</p>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-muted rounded-lg">
                    <Key className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-muted-foreground text-sm">Checks Total</p>
                    <p className="text-3xl font-bold">{totalChecks}</p>
                  </div>
                </div>
              </Card>
            </div>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Security Health Checks</h3>
              <div className="space-y-3">
                {securityChecks.map((check) => (
                  <div
                    key={check.id}
                    className="flex items-center justify-between p-4 bg-card/50 rounded-lg border border-border hover:border-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      {check.status === 'pass' ? (
                        <CheckCircle className="w-5 h-5 text-muted-foreground" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-muted-foreground" />
                      )}
                      <div>
                        <h4 className="font-semibold">{check.name}</h4>
                        <p className="text-sm text-muted-foreground">{check.message}</p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded text-xs font-medium ${
                      check.status === 'pass'
                        ? 'bg-muted text-muted-foreground'
                        : 'bg-muted text-muted-foreground'
                    }`}>
                      {check.status === 'pass' ? 'Pass' : 'Warning'}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
            </div>
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
