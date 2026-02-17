// Agent API client with SSE streaming support
// Use relative path so requests go through Next.js proxy (/api/* -> backend). Avoids "Failed to connect" when
// NEXT_PUBLIC_API_URL points at backend and backend is not reachable from the browser (e.g. Docker).
const API_BASE_URL = ''

export interface AgentResponse {
  content: string
  intent: string
  experience_id: number
  session_id: string
  suggested_actions: SuggestedAction[]
  tools_used: string[]
}

export interface SuggestedAction {
  action: string
  params: Record<string, any>
  label: string
}

export interface StreamCallbacks {
  onIntent?: (intent: string) => void
  onToolUse?: (tool: string) => void
  onToken?: (token: string) => void
  onDone?: (data: { experience_id: number; session_id: string; suggested_actions: SuggestedAction[]; intent: string; content?: string }) => void
  onError?: (message: string) => void
  onAbort?: () => void
}

export const agentApi = {
  /** Non-streaming chat */
  async chat(message: string, sessionId?: string): Promise<AgentResponse> {
    const res = await fetch(`${API_BASE_URL}/api/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || err.message || 'Chat request failed')
    }
    const json = await res.json()
    return json.data
  },

  /** Streaming chat via SSE (supports AbortSignal for cancellation) */
  async chatStream(
    message: string,
    sessionId: string | undefined,
    callbacks: StreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/api/agent/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      callbacks.onError?.(err.detail || 'Stream request failed')
      return
    }

    const reader = res.body?.getReader()
    if (!reader) {
      callbacks.onError?.('No response body')
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (signal?.aborted) {
          reader.cancel()
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue

          try {
            const event = JSON.parse(raw)
            switch (event.type) {
              case 'intent':
                callbacks.onIntent?.(event.intent)
                break
              case 'tool_use':
                callbacks.onToolUse?.(event.tool)
                break
              case 'token':
                callbacks.onToken?.(event.content)
                break
              case 'done':
                callbacks.onDone?.(event)
                break
              case 'error':
                callbacks.onError?.(event.message)
                break
            }
          } catch {
            // Skip malformed events
          }
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        callbacks.onAbort?.()
        return
      }
      throw e
    }
  },

  /** Submit feedback (thumbs up/down) */
  async submitFeedback(experienceId: number, score: 1 | -1): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/api/agent/feedback/${experienceId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ score }),
    })
    if (!res.ok) throw new Error('Failed to submit feedback')
  },

  /** Execute a suggested action */
  async executeAction(action: string, params: Record<string, any>): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/api/agent/action/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, params }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'Action execution failed')
    }
    const json = await res.json()
    return json.data
  },
}
