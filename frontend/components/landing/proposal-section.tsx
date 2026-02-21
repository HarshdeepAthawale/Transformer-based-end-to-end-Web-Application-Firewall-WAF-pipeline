import Link from 'next/link'
import { FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function ProposalSection() {
  return (
    <section
      id="solutions"
      className="py-20 lg:py-28 bg-[#ffffff]"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <h3
              className="text-2xl lg:text-3xl font-bold mb-4"
              style={{ color: '#000000', fontFamily: 'var(--font-space-grotesk)' }}
            >
              Stop attacks before they reach your apps
            </h3>
            <p
              className="text-lg mb-8 max-w-lg"
              style={{ color: '#000000' }}
            >
              Join teams who have secured their web applications with AI-powered protection.
              Get started in minutes—no credit card required.
            </p>
            <Button
              asChild
              size="lg"
              className="rounded-full font-semibold px-8 py-6"
              style={{
                backgroundColor: 'var(--positivus-green)',
                color: '#000000',
              }}
            >
              <Link href="/dashboard">Start protecting now</Link>
            </Button>
          </div>
          <div className="flex justify-center lg:justify-end">
            <div
              className="w-full max-w-sm aspect-square rounded-2xl flex items-center justify-center border-2 border-black/10"
              style={{ backgroundColor: '#E8F5B8' }}
            >
              <FileText
                className="w-32 h-32 lg:w-40 lg:h-40"
                style={{ color: 'var(--positivus-green)' }}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
