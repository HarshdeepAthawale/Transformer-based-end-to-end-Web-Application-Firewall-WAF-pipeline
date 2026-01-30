'use client'

import { useEffect } from 'react'

export default function ErrorReporter() {
  useEffect(() => {
    // Error reporting setup
    const handleError = (event: ErrorEvent) => {
      console.error('[ErrorReporter] Uncaught error:', event.error)
    }

    const handleRejection = (event: PromiseRejectionEvent) => {
      console.error('[ErrorReporter] Unhandled rejection:', event.reason)
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleRejection)
    }
  }, [])

  return null
}
