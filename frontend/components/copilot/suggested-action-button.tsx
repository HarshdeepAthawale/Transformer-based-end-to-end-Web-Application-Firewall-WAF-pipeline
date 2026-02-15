'use client'

import { useState } from 'react'
import { Ban, Check, Loader2, Shield, AlertTriangle, Globe } from 'lucide-react'
import type { SuggestedAction } from '@/lib/agent-api'

const ACTION_ICONS: Record<string, typeof Ban> = {
  block_ip: Ban,
  unblock_ip: Shield,
  whitelist_ip: Shield,
  dismiss_alert: AlertTriangle,
  acknowledge_alert: AlertTriangle,
  create_security_rule: Shield,
  create_geo_rule: Globe,
}

interface Props {
  action: SuggestedAction
  onExecute: (action: SuggestedAction) => Promise<void>
}

export function SuggestedActionButton({ action, onExecute }: Props) {
  const [state, setState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const Icon = ACTION_ICONS[action.action] || Shield

  const handleClick = async () => {
    setState('loading')
    try {
      await onExecute(action)
      setState('done')
    } catch {
      setState('error')
      setTimeout(() => setState('idle'), 2000)
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={state === 'loading' || state === 'done'}
      className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-none border-2 transition-colors disabled:opacity-60"
      style={{
        borderColor: state === 'done' ? '#22c55e' : state === 'error' ? '#ef4444' : 'var(--positivus-gray)',
        backgroundColor: state === 'done' ? '#DCFCE7' : 'var(--positivus-white)',
        color: 'var(--positivus-black)',
        fontFamily: 'var(--font-space-grotesk)',
      }}
    >
      {state === 'loading' ? (
        <Loader2 size={14} className="animate-spin" />
      ) : state === 'done' ? (
        <Check size={14} style={{ color: '#22c55e' }} />
      ) : (
        <Icon size={14} />
      )}
      {state === 'done' ? 'Done' : state === 'error' ? 'Failed' : action.label}
    </button>
  )
}
