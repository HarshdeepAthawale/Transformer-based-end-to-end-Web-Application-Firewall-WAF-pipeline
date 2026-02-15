'use client'

import { Loader2 } from 'lucide-react'

export function TypingIndicator({ toolName }: { toolName?: string }) {
  return (
    <div className="flex items-center gap-2 py-2 px-3" style={{ color: 'var(--positivus-gray-dark)' }}>
      <Loader2 size={16} className="animate-spin" />
      <span className="text-sm" style={{ fontFamily: 'var(--font-space-grotesk)' }}>
        {toolName ? `Using ${toolName.replace(/_/g, ' ')}...` : 'Thinking...'}
      </span>
    </div>
  )
}
