'use client'

import Link from 'next/link'
import { Shield, ArrowRight, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function HeroSection() {
  return (
    <section
      className="relative min-h-screen flex items-center overflow-hidden pt-14"
      style={{ backgroundColor: '#ffffff' }}
    >
      {/* Subtle pattern overlay - black dots only */}
      <div
        className="absolute inset-0 opacity-[0.12] pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(#000000 1px, transparent 1px)`,
          backgroundSize: '24px 24px',
        }}
        aria-hidden
      />
      {/* Very subtle green tint for depth */}
      <div
        className="absolute top-0 right-0 w-1/2 h-full pointer-events-none"
        style={{
          background: 'linear-gradient(135deg, transparent 60%, rgba(197, 226, 70, 0.04) 100%)',
        }}
        aria-hidden
      />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 lg:py-12 w-full">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left: Copy */}
          <div className="order-2 lg:order-1 text-center lg:text-left">
            {/* Badge / pill */}
            <div
              className="inline-flex items-center gap-2 rounded-full border px-4 py-1.5 mb-6"
              style={{
                borderColor: 'var(--positivus-green)',
                color: 'var(--positivus-green)',
                backgroundColor: 'var(--positivus-green-bg)',
              }}
            >
              <span className="h-2 w-2 rounded-full bg-[var(--positivus-green)] animate-pulse" />
              <span className="text-sm font-semibold">AI-Powered Security</span>
            </div>

            <h1
              className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold leading-[1.05] tracking-tight mb-6"
              style={{
                color: '#000000',
                fontFamily: 'var(--font-space-grotesk)',
              }}
            >
              Secure your web apps with{' '}
              <span style={{ color: 'var(--positivus-green)' }}>intelligent</span>
              <br className="hidden sm:block" />
              protection
            </h1>
            <p
              className="text-lg sm:text-xl max-w-xl mx-auto lg:mx-0 mb-8 leading-relaxed"
              style={{ color: '#000000' }}
            >
              Enterprise-grade Web Application Firewall that stops attacks before they reach your
              applications. Real-time threat detection, zero-config deployment.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Button
                asChild
                size="lg"
                className="rounded-full font-semibold px-8 py-6 text-base gap-2 group"
                style={{
                  backgroundColor: 'var(--positivus-green)',
                  color: 'var(--positivus-black)',
                }}
              >
                <Link href="/dashboard">
                  Get started free
                  <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="rounded-full font-semibold px-8 py-6 text-base gap-2 border-2"
                style={{
                  backgroundColor: '#ffffff',
                  borderColor: 'var(--positivus-green)',
                  color: 'var(--positivus-green)',
                }}
              >
                <Link href="#products" className="inline-flex items-center gap-2">
                  <Play className="size-4" />
                  See how it works
                </Link>
              </Button>
            </div>

            {/* Trust line */}
            <p className="mt-8 text-sm" style={{ color: '#000000' }}>
              No credit card required · Deploy in minutes
            </p>
          </div>

          {/* Right: Visual */}
          <div className="order-1 lg:order-2 flex justify-center lg:justify-end">
            <div className="relative w-full max-w-md aspect-square">
              {/* Glow behind shield */}
              <div
                className="absolute inset-0 rounded-3xl opacity-30 blur-3xl"
                style={{ background: 'var(--positivus-green)' }}
                aria-hidden
              />
              <div className="relative w-full h-full rounded-3xl flex items-center justify-center">
                <Shield
                  className="w-40 h-40 sm:w-52 sm:h-52 lg:w-64 lg:h-64 drop-shadow-2xl relative z-10"
                  style={{ color: 'var(--positivus-green)' }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
