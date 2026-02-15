'use client'

import { useState, useRef, useEffect } from 'react'
import { Sidebar } from '@/components/sidebar'
import { Header } from '@/components/header'
import { Send, Bot, Search, BarChart3, Shield, FileSearch, HelpCircle, AlertTriangle, Square } from 'lucide-react'
import { ChatMessage, type ChatMessageData } from '@/components/copilot/chat-message'
import { TypingIndicator } from '@/components/copilot/typing-indicator'
import { agentApi, type SuggestedAction } from '@/lib/agent-api'

const STARTER_PROMPTS = [
  { icon: AlertTriangle, label: 'Active Alerts', prompt: 'Show me all active alerts and their severity', color: '#FDE68A' },
  { icon: Search, label: 'Investigate Threats', prompt: 'What are the most recent threats detected?', color: '#FCA5A5' },
  { icon: BarChart3, label: 'Traffic Analysis', prompt: 'Analyze traffic trends for the last 24 hours', color: 'var(--positivus-green)' },
  { icon: Shield, label: 'Block Attacker', prompt: 'Help me block a suspicious IP address', color: '#FDE68A' },
  { icon: FileSearch, label: 'Forensics', prompt: 'Show me the audit trail for recent configuration changes', color: '#C4B5FD' },
  { icon: HelpCircle, label: 'Explain SQL Injection', prompt: 'Explain what SQL injection is and how the WAF protects against it', color: '#A5B4FC' },
]

export default function CopilotPage() {
  const [messages, setMessages] = useState<ChatMessageData[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const [currentTool, setCurrentTool] = useState<string | undefined>(undefined)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isStreaming, currentTool])

  const stopStreaming = () => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
  }

  const sendMessage = async (text: string) => {
    if (!text.trim() || isStreaming) return

    stopStreaming()
    abortRef.current = new AbortController()

    const userMsg: ChatMessageData = { role: 'user', content: text.trim() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsStreaming(true)
    setCurrentTool(undefined)

    // Build streaming assistant message
    let content = ''
    let intent = ''
    let toolsUsed: string[] = []

    // Add placeholder assistant message
    const assistantIdx = messages.length + 1
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      await agentApi.chatStream(text.trim(), sessionId, {
        onIntent: (i) => {
          intent = i
          setMessages(prev => {
            const updated = [...prev]
            updated[assistantIdx] = { ...updated[assistantIdx], intent: i }
            return updated
          })
        },
        onToolUse: (tool) => {
          toolsUsed.push(tool)
          setCurrentTool(tool)
        },
        onToken: (token) => {
          setCurrentTool(undefined)
          content += token
          setMessages(prev => {
            const updated = [...prev]
            updated[assistantIdx] = {
              ...updated[assistantIdx],
              content,
              intent,
              toolsUsed: [...toolsUsed],
            }
            return updated
          })
        },
        onDone: (data) => {
          setSessionId(data.session_id)
          const finalContent = (data.content && data.content.length > 0) ? data.content : content
          setMessages(prev => {
            const updated = [...prev]
            updated[assistantIdx] = {
              ...updated[assistantIdx],
              content: finalContent,
              intent: data.intent || intent,
              experienceId: data.experience_id,
              toolsUsed: [...toolsUsed],
              suggestedActions: data.suggested_actions,
            }
            return updated
          })
        },
        onError: (msg) => {
          setMessages(prev => {
            const updated = [...prev]
            updated[assistantIdx] = {
              ...updated[assistantIdx],
              content: content || `Error: ${msg}`,
              intent,
              toolsUsed: [...toolsUsed],
            }
            return updated
          })
        },
        onAbort: () => {
          if (content) {
            setMessages(prev => {
              const updated = [...prev]
              updated[assistantIdx] = { ...updated[assistantIdx], content: content + '\n\n_Cancelled_' }
              return updated
            })
          }
        },
      }, abortRef.current.signal)
    } catch (err) {
      const isAborted = err instanceof Error && err.name === 'AbortError'
      if (!isAborted) {
        const errMsg = err instanceof Error ? err.message : 'Unknown error'
        const isNetwork =
          errMsg.toLowerCase().includes('fetch') ||
          errMsg.toLowerCase().includes('network') ||
          errMsg.toLowerCase().includes('failed to fetch')
        const display =
          content ||
          (isNetwork
            ? 'Failed to connect to the AI agent. Please check that the backend is running on port 3001.'
            : `Error: ${errMsg}`)
        setMessages(prev => {
          const updated = [...prev]
          updated[assistantIdx] = { ...updated[assistantIdx], content: display }
          return updated
        })
      } else if (content) {
        setMessages(prev => {
          const updated = [...prev]
          updated[assistantIdx] = { ...updated[assistantIdx], content: content + '\n\n_Cancelled_' }
          return updated
        })
      }
    } finally {
      setIsStreaming(false)
      setCurrentTool(undefined)
      abortRef.current = null
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    sendMessage(input)
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 flex flex-col overflow-hidden bg-background">
          {/* Chat area */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
            {isEmpty ? (
              /* Empty state with starter prompts */
              <div className="max-w-2xl mx-auto flex flex-col items-center justify-center h-full gap-8">
                <div className="text-center">
                  <div
                    className="inline-flex items-center justify-center w-16 h-16 rounded-none mb-4"
                    style={{ backgroundColor: 'var(--positivus-green)' }}
                  >
                    <Bot size={32} style={{ color: 'var(--positivus-black)' }} />
                  </div>
                  <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                    WAF Security Copilot
                  </h2>
                  <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                    Ask me about threats, traffic, alerts, or security concepts. I can also help you take action.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full">
                  {STARTER_PROMPTS.map((sp) => {
                    const Icon = sp.icon
                    return (
                      <button
                        key={sp.label}
                        onClick={() => sendMessage(sp.prompt)}
                        className="flex items-start gap-3 p-4 rounded-none border-2 text-left transition-all hover:shadow-md"
                        style={{
                          borderColor: 'var(--positivus-gray)',
                          backgroundColor: 'var(--positivus-white)',
                        }}
                      >
                        <div className="flex-shrink-0 p-1.5 rounded-none" style={{ backgroundColor: sp.color }}>
                          <Icon size={16} style={{ color: 'var(--positivus-black)' }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                            {sp.label}
                          </p>
                          <p className="text-xs mt-0.5 line-clamp-2" style={{ color: 'var(--positivus-gray-dark)' }}>
                            {sp.prompt}
                          </p>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            ) : (
              /* Message list */
              <div className="max-w-3xl mx-auto space-y-6">
                {messages.map((msg, i) => (
                  <ChatMessage key={i} message={msg} />
                ))}
                {isStreaming && currentTool && (
                  <TypingIndicator toolName={currentTool} />
                )}
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="border-t-2 px-4 py-4 md:px-8" style={{ borderColor: 'var(--positivus-gray)', backgroundColor: 'var(--positivus-white)' }}>
            <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about threats, traffic, or security..."
                disabled={isStreaming}
                className="flex-1 px-4 py-3 text-sm rounded-none border-2 focus:outline-none focus:border-[var(--positivus-black)] disabled:opacity-50"
                style={{
                  borderColor: 'var(--positivus-gray)',
                  fontFamily: 'var(--font-space-grotesk)',
                  color: 'var(--positivus-black)',
                  backgroundColor: 'var(--positivus-white)',
                }}
              />
              {isStreaming ? (
                <button
                  type="button"
                  onClick={stopStreaming}
                  className="px-4 py-3 rounded-none transition-opacity hover:opacity-80 flex items-center gap-2"
                  style={{
                    backgroundColor: '#dc2626',
                    color: 'white',
                  }}
                >
                  <Square size={14} fill="currentColor" />
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="px-4 py-3 rounded-none transition-opacity disabled:opacity-30"
                  style={{
                    backgroundColor: 'var(--positivus-black)',
                    color: 'var(--positivus-white)',
                  }}
                >
                  <Send size={18} />
                </button>
              )}
            </form>
          </div>
        </main>
      </div>
    </div>
  )
}
