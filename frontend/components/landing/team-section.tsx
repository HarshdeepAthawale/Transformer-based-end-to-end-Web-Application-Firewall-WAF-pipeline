import { User } from 'lucide-react'

const team = [
  { name: 'Security-focused team', role: 'Core Engineering', bio: 'ML engineers and security researchers building the next generation of application protection' },
  { name: 'Platform team', role: 'Infrastructure', bio: 'Ensuring low latency, high availability, and seamless integration with your stack' },
  { name: 'Product team', role: 'Experience', bio: 'Making enterprise security accessible with intuitive dashboards and clear insights' },
]

export function TeamSection() {
  return (
    <section id="about" className="py-20 lg:py-28 bg-[var(--positivus-white)]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            About
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: 'var(--positivus-gray-dark)' }}
          >
            We build security that works at the speed of the internet. Our team combines ML expertise
            with deep application security knowledge to protect what matters.
          </p>
        </div>

        <div className="grid sm:grid-cols-3 gap-6">
          {team.map((member) => (
            <div
              key={member.name}
              className="border-2 p-6 transition-colors hover:border-[var(--positivus-green)]"
              style={{ borderColor: 'var(--positivus-gray)' }}
            >
              <div
                className="w-14 h-14 rounded-full flex items-center justify-center mb-4"
                style={{ backgroundColor: 'var(--positivus-green-bg)' }}
              >
                <User
                  className="w-7 h-7"
                  style={{ color: 'var(--positivus-green)' }}
                />
              </div>
              <h4 className="font-semibold mb-1" style={{ color: 'var(--positivus-black)', fontFamily: 'var(--font-space-grotesk)' }}>
                {member.name}
              </h4>
              <p className="text-sm mb-4" style={{ color: 'var(--positivus-gray-dark)' }}>
                {member.role}
              </p>
              <p className="text-sm" style={{ color: 'var(--positivus-gray-dark)' }}>
                {member.bio}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
