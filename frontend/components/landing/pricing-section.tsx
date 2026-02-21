import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Check } from 'lucide-react'

const plans = [
  {
    name: 'Free',
    price: '$0',
    priceNote: '/month',
    description: null,
    features: [
      '1 protected app',
      'Basic threat detection',
      '7-day analytics',
      'Community & email support',
    ],
    cta: { label: 'Get started', href: '/dashboard' },
    highlight: false,
    outline: false,
  },
  {
    name: 'Pro',
    price: '$15',
    priceNote: '/month',
    description: null,
    features: [
      'AI-powered threat detection',
      'Real-time analytics dashboard',
      'Bot protection',
      'Geo rules & IP management',
      'Unlimited requests',
      'Email support',
    ],
    cta: { label: 'Get started', href: '/dashboard' },
    highlight: true,
    outline: false,
  },
  {
    name: 'Business',
    price: '$180',
    priceNote: '/month',
    description: null,
    features: [
      'Everything in Pro',
      'SSO / SAML',
      'Priority support',
      'SLA & compliance',
    ],
    cta: { label: 'Get started', href: '/dashboard' },
    highlight: false,
    outline: false,
  },
  {
    name: 'Custom',
    price: 'Custom',
    priceNote: null,
    description: null,
    features: [
      'Everything in Business',
      'Dedicated success manager',
      'Custom rules & integrations',
      '24/7 phone & premium support',
    ],
    cta: { label: 'Contact sales', href: '#contact' },
    highlight: false,
    outline: true,
  },
]

export function PricingSection() {
  return (
    <section id="pricing" className="py-20 lg:py-28 bg-[#ffffff]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Simple pricing
          </h2>
          <p
            className="text-lg max-w-2xl mx-auto"
            style={{ color: '#000000' }}
          >
            Start free. Scale as you grow. No surprises.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`relative flex flex-col p-6 lg:p-8 border-2 transition-transform ${
                plan.highlight ? 'md:-mt-2 md:mb-2 md:scale-[1.02]' : ''
              }`}
              style={{
                backgroundColor: '#ffffff',
                borderColor: plan.highlight ? 'var(--positivus-green)' : '#000000',
              }}
            >
              {plan.highlight && (
                <span
                  className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 text-xs font-semibold uppercase tracking-wide"
                  style={{
                    backgroundColor: 'var(--positivus-green)',
                    color: '#000000',
                  }}
                >
                  Most popular
                </span>
              )}
              <h3
                className="text-2xl font-bold mb-2"
                style={{ color: '#000000', fontFamily: 'var(--font-space-grotesk)' }}
              >
                {plan.name}
              </h3>
              <p className="text-4xl font-bold mb-6" style={{ color: '#000000' }}>
                {plan.price}
                {plan.priceNote && (
                  <span className="text-xl font-normal text-[#000000]">{plan.priceNote}</span>
                )}
              </p>
              {plan.description && (
                <p className="text-sm mb-4" style={{ color: '#000000' }}>
                  {plan.description}
                </p>
              )}
              <ul className="space-y-3 mb-8 flex-1">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-3">
                    <Check className="w-5 h-5 flex-shrink-0" style={{ color: 'var(--positivus-green)' }} />
                    <span style={{ color: '#000000' }}>{feature}</span>
                  </li>
                ))}
              </ul>
              <Button
                asChild
                variant={plan.outline ? 'outline' : 'default'}
                className="w-full rounded-none font-semibold py-3 text-lg mt-auto"
                style={
                  plan.outline
                    ? {
                        borderColor: '#000000',
                        color: '#000000',
                        backgroundColor: 'transparent',
                      }
                    : {
                        backgroundColor: '#000000',
                        color: '#ffffff',
                      }
                }
              >
                <Link href={plan.cta.href}>{plan.cta.label}</Link>
              </Button>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
