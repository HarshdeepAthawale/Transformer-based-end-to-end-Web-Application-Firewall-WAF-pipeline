import { ArrowRight } from 'lucide-react'

const caseStudies = [
  {
    description:
      'A fintech startup blocked over 2.3M malicious requests in the first month. Our AI detection caught zero-day SQL injection attempts that legacy WAFs missed.',
    metric: '2.3M+ threats blocked',
  },
  {
    description:
      'An e-commerce platform reduced fraud and bot traffic by 94% while keeping checkout latency under 50ms. Real-time analytics gave full visibility into attack patterns.',
    metric: '94% bot traffic reduction',
  },
  {
    description:
      'A healthcare API provider achieved 97.6% threat detection accuracy with zero false positives on legitimate health record queries. Learned their traffic in days.',
    metric: '97.6% detection accuracy',
  },
]

export function CaseStudiesSection() {
  return (
    <section id="customers" className="py-20 lg:py-28 bg-[var(--positivus-white)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16" id="solutions">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Customers
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: 'var(--positivus-gray-dark)' }}
          >
            See how teams across industries secure their applications and stop attacks in real time
          </p>
        </div>

        <div
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 p-6 lg:p-8"
          style={{ backgroundColor: 'var(--positivus-black)' }}
        >
          {caseStudies.map((study, i) => (
            <div
              key={i}
              className="group p-6 lg:p-8 flex flex-col justify-between border border-[var(--positivus-gray-dark)]/30 hover:border-[var(--positivus-green)]/50 transition-colors"
            >
              <p
                className="text-base lg:text-lg mb-4"
                style={{ color: 'var(--positivus-white)' }}
              >
                {study.description}
              </p>
              <span
                className="text-sm font-semibold mb-4"
                style={{ color: 'var(--positivus-green)' }}
              >
                {study.metric}
              </span>
              <span className="flex items-center gap-2 text-sm font-medium text-[var(--positivus-green)] group-hover:gap-3 transition-all">
                <ArrowRight className="w-4 h-4" />
                Learn more
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
