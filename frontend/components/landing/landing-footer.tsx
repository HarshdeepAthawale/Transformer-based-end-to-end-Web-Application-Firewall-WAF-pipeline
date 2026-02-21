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
    <footer className="py-16 lg:py-20 bg-[#ffffff] border-t border-black/10">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8 mb-12 pb-12 border-b border-black/10">
          <Link href="/" className="flex items-center">
            <span
              className="text-2xl font-bold"
              style={{ color: '#000000', fontFamily: 'var(--font-space-grotesk)' }}
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
                  style={{ color: '#000000' }}
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
                className="w-10 h-10 rounded-full border-2 border-black/20 flex items-center justify-center transition-colors hover:border-[var(--positivus-green)] hover:text-[var(--positivus-green)]"
                style={{ color: '#000000' }}
                aria-label={social}
              >
                {social[0]}
              </a>
            ))}
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold mb-4" style={{ color: '#000000' }}>
              Contact
            </h4>
            <div className="space-y-2" style={{ color: '#000000' }}>
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
                  borderColor: '#000000',
                  backgroundColor: '#ffffff',
                  color: '#000000',
                }}
              />
              <Button
                type="button"
                className="rounded-none font-semibold px-6 whitespace-nowrap"
                style={{
                  backgroundColor: 'var(--positivus-green)',
                  color: '#000000',
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
