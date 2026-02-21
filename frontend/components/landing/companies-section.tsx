import Link from 'next/link'

const companies = [
  'SaaS',
  'E‑commerce',
  'Fintech',
  'Healthcare',
  'API Platforms',
  'Startups',
]

export function CompaniesSection() {
  return (
    <section
      className="py-8 lg:py-10 bg-[#ffffff] border-y border-black/10"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <p
          className="text-center text-sm font-medium mb-6"
          style={{ color: '#000000' }}
        >
          Trusted by teams securing applications across industries.
        </p>
        <div className="flex flex-wrap justify-center items-center gap-6 lg:gap-10">
          {companies.map((company) => (
            <Link
              key={company}
              href="#products"
              className="text-base font-semibold transition-colors hover:text-[var(--positivus-green)]"
              style={{ color: '#000000', fontFamily: 'var(--font-space-grotesk)' }}
            >
              {company}
            </Link>
          ))}
        </div>
      </div>
    </section>
  )
}
