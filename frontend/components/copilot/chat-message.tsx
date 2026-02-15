'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ThumbsUp, ThumbsDown, User, Bot, Wrench } from 'lucide-react'
import { IntentBadge } from './intent-badge'
import { SuggestedActionButton } from './suggested-action-button'
import { agentApi } from '@/lib/agent-api'
import type { SuggestedAction } from '@/lib/agent-api'

export interface ChatMessageData {
  role: 'user' | 'assistant'
  content: string
  intent?: string
  experienceId?: number
  toolsUsed?: string[]
  suggestedActions?: SuggestedAction[]
}

interface Props {
  message: ChatMessageData
}

export function ChatMessage({ message }: Props) {
  const [feedback, setFeedback] = useState<1 | -1 | null>(null)
  const isUser = message.role === 'user'

  const handleFeedback = async (score: 1 | -1) => {
    if (!message.experienceId || feedback !== null) return
    try {
      await agentApi.submitFeedback(message.experienceId, score)
      setFeedback(score)
    } catch {
      // Silently fail
    }
  }

  const handleAction = async (action: SuggestedAction) => {
    await agentApi.executeAction(action.action, action.params)
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-none"
        style={{
          backgroundColor: isUser ? 'var(--positivus-black)' : 'var(--positivus-green)',
        }}
      >
        {isUser ? (
          <User size={16} style={{ color: 'var(--positivus-white)' }} />
        ) : (
          <Bot size={16} style={{ color: 'var(--positivus-black)' }} />
        )}
      </div>

      {/* Message body */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        {/* Intent badge + tools */}
        {!isUser && message.intent && (
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <IntentBadge intent={message.intent} />
            {message.toolsUsed && message.toolsUsed.length > 0 && (
              <span className="inline-flex items-center gap-1 text-xs" style={{ color: 'var(--positivus-gray-dark)' }}>
                <Wrench size={12} />
                {message.toolsUsed.length} tool{message.toolsUsed.length > 1 ? 's' : ''} used
              </span>
            )}
          </div>
        )}

        {/* Content */}
        <div
          className="rounded-none px-4 py-3 border-2"
          style={{
            backgroundColor: isUser ? 'var(--positivus-black)' : 'var(--positivus-white)',
            color: isUser ? 'var(--positivus-white)' : 'var(--positivus-black)',
            borderColor: isUser ? 'var(--positivus-black)' : 'var(--positivus-gray)',
          }}
        >
          {isUser ? (
            <p className="text-sm" style={{ fontFamily: 'var(--font-space-grotesk)' }}>{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none text-sm [&_table]:text-xs [&_th]:px-2 [&_td]:px-2 [&_th]:py-1 [&_td]:py-1 [&_table]:border-collapse [&_th]:border [&_td]:border" style={{ fontFamily: 'var(--font-space-grotesk)' }}>
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Suggested actions */}
        {!isUser && message.suggestedActions && message.suggestedActions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {message.suggestedActions.map((action, i) => (
              <SuggestedActionButton key={i} action={action} onExecute={handleAction} />
            ))}
          </div>
        )}

        {/* Feedback buttons */}
        {!isUser && message.experienceId && (
          <div className="flex gap-1 mt-2">
            <button
              onClick={() => handleFeedback(1)}
              disabled={feedback !== null}
              className="p-1 rounded-none transition-colors hover:bg-green-50"
              style={{ color: feedback === 1 ? '#22c55e' : 'var(--positivus-gray-dark)' }}
              title="Helpful"
            >
              <ThumbsUp size={14} />
            </button>
            <button
              onClick={() => handleFeedback(-1)}
              disabled={feedback !== null}
              className="p-1 rounded-none transition-colors hover:bg-red-50"
              style={{ color: feedback === -1 ? '#ef4444' : 'var(--positivus-gray-dark)' }}
              title="Not helpful"
            >
              <ThumbsDown size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
