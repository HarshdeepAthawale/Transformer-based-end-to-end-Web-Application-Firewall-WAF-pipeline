'use client'

import { useState, Suspense } from 'react'
import { signIn } from 'next-auth/react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { usersApi } from '@/lib/api'
import { Header } from '@/components/ui/header-1'

type AuthMode = 'signin' | 'signup'

const greenTextureDots = {
  backgroundImage: 'radial-gradient(rgba(197, 226, 70, 0.45) 1px, transparent 1px)',
  backgroundSize: '24px 24px',
} as const

const greenTextureGradient = {
  background: 'linear-gradient(135deg, transparent 60%, rgba(197, 226, 70, 0.04) 100%)',
} as const

function LoginForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const callbackUrl = searchParams.get('callbackUrl') ?? '/dashboard'
  const [mode, setMode] = useState<AuthMode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (mode === 'signin') {
        const result = await signIn('credentials', {
          email,
          password,
          redirect: false,
        })
        if (result?.error) {
          setError('Invalid email or password')
          return
        }
        router.push(callbackUrl)
        router.refresh()
      } else {
        if (!name.trim()) {
          setError('Name is required')
          return
        }
        if (password.length < 8) {
          setError('Password must be at least 8 characters')
          return
        }
        try {
          await usersApi.register({
            username: email,
            email,
            password,
            full_name: name.trim(),
          })
        } catch (err: unknown) {
          const msg =
            err instanceof Error ? err.message : 'Failed to create account. Please try again.'
          setError(msg)
          return
        }
        const result = await signIn('credentials', {
          email,
          password,
          redirect: false,
        })
        if (result?.error) {
          setError('Account created but sign in failed. Please try signing in.')
          return
        }
        router.push(callbackUrl)
        router.refresh()
      }
    } catch {
      setError('Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleSignIn = () => {
    setLoading(true)
    signIn('google', { callbackUrl })
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header variant="light" />
      <div
        className="relative flex-1 flex items-center justify-center p-4 bg-white"
        style={{ fontFamily: 'var(--font-space-grotesk)' }}
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={greenTextureDots}
          aria-hidden
        />
        <div
          className="absolute inset-0 pointer-events-none"
          style={greenTextureGradient}
          aria-hidden
        />
        <div className="relative w-full max-w-md">
          <Link
            href="/"
            className="flex items-center justify-center gap-2 mb-8"
            style={{ color: '#191A23' }}
          >
            <Shield className="w-8 h-8" style={{ color: '#C5E246' }} />
            <span className="text-xl font-bold">
              WAF
            </span>
          </Link>

          <div
            className="p-8 rounded-lg border-2"
            style={{
              backgroundColor: '#ffffff',
              borderColor: '#F2F2F2',
            }}
          >
            <div className="flex mb-6 border-b" style={{ borderColor: '#F2F2F2' }}>
              <button
                type="button"
                onClick={() => { setMode('signin'); setError('') }}
                className={`flex-1 py-3 text-sm font-medium transition-colors ${
                  mode === 'signin'
                    ? 'border-b-2'
                    : 'opacity-60 hover:opacity-100'
                }`}
                style={{
                  color: mode === 'signin' ? '#191A23' : '#707A89',
                  borderColor: mode === 'signin' ? '#C5E246' : 'transparent',
                }}
              >
                Sign in
              </button>
              <button
                type="button"
                onClick={() => { setMode('signup'); setError('') }}
                className={`flex-1 py-3 text-sm font-medium transition-colors ${
                  mode === 'signup'
                    ? 'border-b-2'
                    : 'opacity-60 hover:opacity-100'
                }`}
                style={{
                  color: mode === 'signup' ? '#191A23' : '#707A89',
                  borderColor: mode === 'signup' ? '#C5E246' : 'transparent',
                }}
              >
                Sign up
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {mode === 'signup' && (
                <div>
                  <Label htmlFor="name" className="text-sm font-medium" style={{ color: '#191A23' }}>
                    Name
                  </Label>
                  <Input
                    id="name"
                    type="text"
                    placeholder="Your name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    className="mt-1 rounded-none border-2 placeholder:text-[#707A89]"
                    style={{
                      borderColor: '#F2F2F2',
                      color: '#191A23',
                    }}
                  />
                </div>
              )}
              <div>
                <Label htmlFor="email" className="text-sm font-medium" style={{ color: '#191A23' }}>
                  Email
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="mt-1 rounded-none border-2 placeholder:text-[#707A89]"
                  style={{
                    borderColor: '#F2F2F2',
                    color: '#191A23',
                  }}
                />
              </div>
              <div>
                <Label htmlFor="password" className="text-sm font-medium" style={{ color: '#191A23' }}>
                  Password
                </Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="mt-1 rounded-none border-2 placeholder:text-[#707A89]"
                  style={{
                    borderColor: '#F2F2F2',
                    color: '#191A23',
                  }}
                />
              </div>
              {error && (
                <p className="text-sm" style={{ color: '#dc2626' }}>
                  {error}
                </p>
              )}
              <Button
                type="submit"
                disabled={loading}
                className="w-full rounded-none font-semibold py-3"
                style={{
                  backgroundColor: '#C5E246',
                  color: '#191A23',
                }}
              >
                {loading ? 'Please wait...' : mode === 'signin' ? 'Sign in' : 'Sign up'}
              </Button>
            </form>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div
                  className="w-full border-t"
                  style={{ borderColor: '#F2F2F2' }}
                />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span
                  className="px-2 bg-white"
                  style={{ color: '#707A89' }}
                >
                  Or continue with
                </span>
              </div>
            </div>

            <Button
              type="button"
              variant="outline"
              onClick={handleGoogleSignIn}
              disabled={loading}
              className="w-full rounded-none font-medium py-3 border-2 bg-white hover:bg-[#E8F5B8]"
              style={{
                borderColor: '#191A23',
                color: '#191A23',
                backgroundColor: '#ffffff',
              }}
            >
              <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="currentColor"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="currentColor"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="currentColor"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Google
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div
        className="min-h-screen flex items-center justify-center bg-white relative"
        style={{ fontFamily: 'var(--font-space-grotesk)' }}
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={greenTextureDots}
          aria-hidden
        />
        <div
          className="absolute inset-0 pointer-events-none"
          style={greenTextureGradient}
          aria-hidden
        />
        <div className="relative animate-pulse" style={{ color: '#191A23' }}>Loading...</div>
      </div>
    }>
      <LoginForm />
    </Suspense>
  )
}
