'use client'

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { Plus } from 'lucide-react'
import { useState } from 'react'

const steps = [
  {
    number: '01',
    title: 'Sign up',
    description:
      'Create your account in seconds. No credit card required. You get instant access to the dashboard and can start protecting your first application right away.',
  },
  {
    number: '02',
    title: 'Add your application',
    description:
      'Point your traffic to our WAF—via reverse proxy, Docker, or direct integration. Works with Nginx, Apache, or any HTTP server. No code changes to your app.',
  },
  {
    number: '03',
    title: 'AI learns your traffic',
    description:
      'Our Transformer model learns the baseline of your legitimate traffic. It identifies anomalies and malicious patterns that signature-based WAFs cannot detect.',
  },
  {
    number: '04',
    title: 'Protection goes live',
    description:
      'Requests are evaluated in real time. Blocked attacks never reach your origin. Legitimate users experience zero added latency. All decisions are logged.',
  },
  {
    number: '05',
    title: 'Monitor and tune',
    description:
      'Use the real-time dashboard to view threats, traffic patterns, and detection metrics. Adjust thresholds, geo rules, and blocklists as needed.',
  },
  {
    number: '06',
    title: 'Scale with confidence',
    description:
      'As your traffic grows, our system scales automatically. Continuous learning keeps the model updated. Deploy to multiple apps from a single dashboard.',
  },
]

export function WorkingProcessSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(0)

  return (
    <section className="py-20 lg:py-28 bg-[var(--positivus-white)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            How it works
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: 'var(--positivus-gray-dark)' }}
          >
            Deploy in minutes. Protect in real time. Scale without limits.
          </p>
        </div>

        <div className="space-y-2">
          {steps.map((step, index) => (
            <Collapsible
              key={step.number}
              open={openIndex === index}
              onOpenChange={(open) => setOpenIndex(open ? index : null)}
            >
              <div
                className="border-2 transition-colors"
                style={{
                  borderColor: openIndex === index ? 'var(--positivus-green)' : 'var(--positivus-gray)',
                }}
              >
                <CollapsibleTrigger className="w-full flex items-center justify-between p-6 lg:p-8 text-left hover:bg-[var(--positivus-gray)]/30 transition-colors">
                  <div className="flex items-center gap-6">
                    <span
                      className="text-2xl font-bold"
                      style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
                    >
                      {step.number}
                    </span>
                    <p
                      className="text-xl font-semibold"
                      style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
                    >
                      {step.title}
                    </p>
                  </div>
                  <Plus
                    className={`w-6 h-6 flex-shrink-0 transition-transform ${
                      openIndex === index ? 'rotate-45' : ''
                    }`}
                    style={{ color: 'var(--positivus-black)' }}
                  />
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="px-6 lg:px-8 pb-6 lg:pb-8 pt-0">
                    <p
                      className="text-base max-w-3xl"
                      style={{ color: 'var(--positivus-gray-dark)' }}
                    >
                      {step.description}
                    </p>
                  </div>
                </CollapsibleContent>
              </div>
            </Collapsible>
          ))}
        </div>
      </div>
    </section>
  )
}
