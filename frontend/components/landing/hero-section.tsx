import Link from 'next/link'
import { Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function HeroSection() {
  return (
    <section className="relative overflow-hidden bg-[var(--positivus-white)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          <div className="order-2 lg:order-1">
            <h1
              className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight mb-6"
              style={{
                color: 'var(--positivus-black)',
                fontFamily: 'var(--font-space-grotesk)',
              }}
            >
              Secure your web apps with{' '}
              <span style={{ color: 'var(--positivus-green)' }}>AI-powered</span> protection
            </h1>
            <p
              className="text-lg sm:text-xl max-w-xl mb-8 leading-relaxed"
              style={{ color: 'var(--positivus-gray-dark)' }}
            >
              Enterprise-grade Web Application Firewall that stops attacks before they reach your
              applications. Real-time threat detection, zero-config deployment, and intelligent
              learning that adapts to your traffic.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                asChild
                size="lg"
                className="rounded-none font-semibold px-8 py-3 text-lg"
                style={{
                  backgroundColor: 'var(--positivus-black)',
                  color: 'var(--positivus-white)',
                }}
              >
                <Link href="/dashboard">Get started free</Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="rounded-none font-semibold px-8 py-3 text-lg border-2"
                style={{
                  borderColor: 'var(--positivus-black)',
                  color: 'var(--positivus-black)',
                  backgroundColor: 'transparent',
                }}
              >
                <Link href="#products">See how it works</Link>
              </Button>
            </div>
          </div>
          <div className="order-1 lg:order-2 flex justify-center lg:justify-end">
            <div
              className="w-full max-w-md aspect-square rounded-2xl flex items-center justify-center"
              style={{ backgroundColor: 'var(--positivus-green-bg)' }}
            >
              <Shield
                className="w-48 h-48 lg:w-64 lg:h-64"
                style={{ color: 'var(--positivus-green)' }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
