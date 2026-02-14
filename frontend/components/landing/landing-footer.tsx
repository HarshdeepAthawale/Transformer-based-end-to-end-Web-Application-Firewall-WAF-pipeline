import Link from 'next/link'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

const footerLinks = [
  { href: '#about', label: 'About' },
  { href: '#products', label: 'Products' },
  { href: '#solutions', label: 'Solutions' },
  { href: '#customers', label: 'Customers' },
  { href: '#pricing', label: 'Pricing' },
]

export function LandingFooter() {
  return (
    <footer
      className="py-16 lg:py-20"
      style={{ backgroundColor: 'var(--positivus-black)' }}
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8 mb-12 pb-12 border-b border-[var(--positivus-gray-dark)]/30">
          <Link href="/" className="flex items-center">
            <span
              className="text-2xl font-bold"
              style={{ color: 'var(--positivus-white)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              WAF
            </span>
          </Link>
          <ul className="flex flex-wrap gap-6 lg:gap-8">
            {footerLinks.map((link) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className="text-sm font-medium transition-colors hover:text-[var(--positivus-green)]"
                  style={{ color: 'var(--positivus-white)' }}
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
          <div className="flex gap-4">
            {['Twitter', 'LinkedIn', 'GitHub'].map((social) => (
              <a
                key={social}
                href="#"
                className="w-10 h-10 rounded-full border flex items-center justify-center transition-colors hover:border-[var(--positivus-green)] hover:text-[var(--positivus-green)]"
                style={{
                  borderColor: 'var(--positivus-gray-dark)',
                  color: 'var(--positivus-white)',
                }}
                aria-label={social}
              >
                {social[0]}
              </a>
            ))}
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold mb-4" style={{ color: 'var(--positivus-white)' }}>
              Contact
            </h4>
            <div className="space-y-2" style={{ color: 'var(--positivus-gray-dark)' }}>
              <p>Email: contact@waf.example</p>
              <p>Dashboard: <a href="/dashboard" className="text-[var(--positivus-green)] hover:underline">Get started</a></p>
            </div>
          </div>
          <div>
            <div className="flex flex-col sm:flex-row gap-3">
              <Input
                type="email"
                placeholder="email"
                className="rounded-none border-2 flex-1 max-w-xs"
                style={{
                  borderColor: 'var(--positivus-gray-dark)',
                  backgroundColor: 'transparent',
                  color: 'var(--positivus-white)',
                }}
              />
              <Button
                type="button"
                className="rounded-none font-semibold px-6 whitespace-nowrap"
                style={{
                  backgroundColor: 'var(--positivus-green)',
                  color: 'var(--positivus-black)',
                }}
              >
                Get product updates
              </Button>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
