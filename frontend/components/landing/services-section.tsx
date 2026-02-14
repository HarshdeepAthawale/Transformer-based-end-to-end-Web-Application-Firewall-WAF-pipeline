import Link from 'next/link'
import { ArrowRight, Shield, Bot, Globe, BarChart3, Zap, Lock } from 'lucide-react'

const services = [
  {
    title: 'AI Threat',
    title2: 'Detection',
    description: 'Transformer-based anomaly detection catches attacks signature-based WAFs miss',
    bgColor: 'var(--positivus-gray)',
    textColor: 'var(--positivus-green)',
    icon: Shield,
  },
  {
    title: 'Bot',
    title2: 'Protection',
    description: 'Identify and block malicious bots while allowing legitimate traffic',
    bgColor: 'var(--positivus-green-bg)',
    textColor: 'var(--positivus-green)',
    icon: Bot,
  },
  {
    title: 'Real-time',
    title2: 'Analytics',
    description: 'Live dashboards with threat metrics, traffic logs, and attack insights',
    bgColor: 'var(--positivus-black)',
    textColor: 'var(--positivus-green)',
    icon: BarChart3,
  },
  {
    title: 'Geo',
    title2: 'Rules',
    description: 'Block or allow traffic by region. Control access at the edge',
    bgColor: 'var(--positivus-gray)',
    textColor: 'var(--positivus-green)',
    icon: Globe,
  },
  {
    title: 'Zero-config',
    title2: 'Deployment',
    description: 'Deploy in front of your app in minutes. No code changes required',
    bgColor: 'var(--positivus-green-bg)',
    textColor: 'var(--positivus-green)',
    icon: Zap,
  },
  {
    title: 'IP Management',
    title2: '& Blocklists',
    description: 'Manage allowlists, blocklists, and rate limits per IP',
    bgColor: 'var(--positivus-black)',
    textColor: 'var(--positivus-green)',
    icon: Lock,
  },
]

export function ServicesSection() {
  return (
    <section id="services" className="py-20 lg:py-28 bg-[var(--positivus-white)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16" id="products">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Products
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: 'var(--positivus-gray-dark)' }}
          >
            Everything you need to protect your web applications. Deploy once, protect everywhere—with
            intelligent security that learns your traffic patterns.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service) => {
            const Icon = service.icon
            const isDark = service.bgColor === 'var(--positivus-black)'
            return (
              <Link
                key={service.title}
                href="#"
                className="group flex flex-col sm:flex-row p-6 lg:p-8 transition-transform hover:scale-[1.02]"
                style={{ backgroundColor: service.bgColor }}
              >
                <div className="flex-1">
                  <h3
                    className="text-xl lg:text-2xl font-bold mb-4"
                    style={{
                      color: isDark ? 'var(--positivus-white)' : 'var(--positivus-black)',
                      fontFamily: 'var(--font-space-grotesk)',
                    }}
                  >
                    <span style={{ color: service.textColor }}>{service.title} </span>
                    <span style={{ color: service.textColor }}>{service.title2}</span>
                  </h3>
                  <p
                    className="text-sm leading-relaxed mt-2"
                    style={{ color: isDark ? 'rgba(255,255,255,0.85)' : 'var(--positivus-gray-dark)' }}
                  >
                    {service.description}
                  </p>
                  <p
                    className="flex items-center gap-2 text-sm font-medium group-hover:gap-3 transition-all mt-4"
                    style={{ color: isDark ? 'var(--positivus-white)' : 'var(--positivus-black)' }}
                  >
                    <ArrowRight className="w-4 h-4" />
                    Learn more
                  </p>
                </div>
                <div className="mt-4 sm:mt-0 sm:ml-4">
                  <Icon
                    className="w-12 h-12 lg:w-16 lg:h-16 opacity-50"
                    style={{ color: isDark ? 'var(--positivus-white)' : 'var(--positivus-black)' }}
                  />
                </div>
              </Link>
            )
          })}
        </div>
      </div>
    </section>
  )
}
