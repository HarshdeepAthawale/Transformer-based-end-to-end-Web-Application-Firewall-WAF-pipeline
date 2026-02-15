'use client'

const INTENT_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  investigate: { label: 'Investigating', color: 'var(--positivus-black)', bg: '#FDE68A' },
  remediate: { label: 'Remediating', color: 'var(--positivus-white)', bg: '#EF4444' },
  analyze: { label: 'Analyzing', color: 'var(--positivus-black)', bg: 'var(--positivus-green)' },
  explain: { label: 'Explaining', color: 'var(--positivus-black)', bg: '#A5B4FC' },
  forensics: { label: 'Forensics', color: 'var(--positivus-white)', bg: '#7C3AED' },
}

export function IntentBadge({ intent }: { intent: string }) {
  const config = INTENT_CONFIG[intent] || { label: intent, color: '#000', bg: '#e5e7eb' }
  return (
    <span
      className="inline-flex items-center px-2.5 py-0.5 text-xs font-semibold rounded-none"
      style={{
        color: config.color,
        backgroundColor: config.bg,
        fontFamily: 'var(--font-space-grotesk)',
      }}
    >
      {config.label}
    </span>
  )
}
