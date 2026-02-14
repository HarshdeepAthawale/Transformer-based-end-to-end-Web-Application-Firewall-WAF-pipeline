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
    <section className="py-12 lg:py-16 border-y border-[var(--positivus-gray)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <p
          className="text-center text-sm font-medium mb-8"
          style={{ color: 'var(--positivus-gray-dark)' }}
        >
          Trusted by teams securing applications across industries
        </p>
        <div className="flex flex-wrap justify-center items-center gap-8 lg:gap-16">
          {companies.map((company) => (
            <div
              key={company}
              className="text-lg font-semibold opacity-70 hover:opacity-100 transition-opacity"
              style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}
            >
              {company}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
