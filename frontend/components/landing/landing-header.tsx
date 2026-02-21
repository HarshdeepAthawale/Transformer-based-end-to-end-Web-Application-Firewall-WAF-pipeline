'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

const navLinks = [
  { href: '#about', label: 'About' },
  { href: '#products', label: 'Products' },
  { href: '#solutions', label: 'Solutions' },
  { href: '#customers', label: 'Customers' },
  { href: '#pricing', label: 'Pricing' },
]

export function LandingHeader() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 border-b bg-[#000000]"
      style={{
        borderColor: 'rgba(255,255,255,0.1)',
      }}
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 lg:h-20 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span
              className="text-xl lg:text-2xl font-bold font-[family-name:var(--font-space-grotesk)]"
              style={{ color: '#ffffff' }}
            >
              WAF
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-[15px] font-medium transition-colors hover:text-[var(--positivus-green)]"
                style={{ color: '#ffffff' }}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <Button
              asChild
              className="hidden md:inline-flex rounded-full font-semibold px-6 py-2.5"
              style={{
                backgroundColor: '#0d0f12',
                color: '#ffffff',
              }}
            >
              <Link href="/login">Sign in</Link>
            </Button>
            <button
              type="button"
              className="md:hidden p-2"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
              style={{ color: '#ffffff' }}
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {mobileMenuOpen && (
          <div
            className="md:hidden py-4 border-t"
            style={{ borderColor: 'rgba(255,255,255,0.2)' }}
          >
            <nav className="flex flex-col gap-4">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-[15px] font-medium"
                  style={{ color: '#ffffff' }}
                >
                  {link.label}
                </Link>
              ))}
              <Button
                asChild
                className="w-full rounded-full font-semibold mt-2"
                style={{
                  backgroundColor: 'var(--positivus-green)',
                  color: '#000000',
                }}
              >
                <Link href="/login">Sign in</Link>
              </Button>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}
