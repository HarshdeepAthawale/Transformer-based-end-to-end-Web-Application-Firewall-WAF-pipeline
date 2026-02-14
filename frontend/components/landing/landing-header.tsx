'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/theme-toggle'

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
    <header className="sticky top-0 z-50 bg-[var(--positivus-white)] border-b border-[var(--positivus-gray)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-20 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span
              className="text-2xl font-bold font-[family-name:var(--font-space-grotesk)]"
              style={{ color: 'var(--positivus-black)' }}
            >
              WAF
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-[15px] font-medium transition-colors hover:opacity-70"
                style={{ color: 'var(--positivus-black)' }}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <ThemeToggle />
            <Button
              asChild
              className="hidden md:inline-flex rounded-none font-semibold px-6 py-2.5"
              style={{
                backgroundColor: 'var(--positivus-black)',
                color: 'var(--positivus-white)',
              }}
            >
              <Link href="/login">Sign in</Link>
            </Button>
            <button
              type="button"
              className="md:hidden p-2"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" style={{ color: 'var(--positivus-gray-dark)' }} />
              ) : (
                <Menu className="h-6 w-6" style={{ color: 'var(--positivus-gray-dark)' }} />
              )}
            </button>
          </div>
        </div>

        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-[var(--positivus-gray)]">
            <nav className="flex flex-col gap-4">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-[15px] font-medium"
                  style={{ color: 'var(--positivus-black)' }}
                >
                  {link.label}
                </Link>
              ))}
              <Button asChild className="w-full rounded-none font-semibold mt-2" style={{ backgroundColor: 'var(--positivus-black)', color: 'var(--positivus-white)' }}>
                <Link href="/login">Sign in</Link>
              </Button>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}
