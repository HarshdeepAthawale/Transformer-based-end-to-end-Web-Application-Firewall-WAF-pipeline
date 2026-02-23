import {
  HeroSection,
  CompaniesSection,
  ServicesSection,
  ProposalSection,
  CaseStudiesSection,
  WorkingProcessSection,
  TeamSection,
  TestimonialsSection,
  PricingSection,
  ContactSection,
  LandingFooter,
} from '@/components/landing'
import { Header } from '@/components/ui/header-1'

export default function LandingPage() {
  return (
    <div
      className="min-h-screen bg-white pt-3 font-[family-name:var(--font-space-grotesk)]"
      style={{ fontFamily: 'var(--font-space-grotesk)' }}
    >
      <Header variant="light" />
      <main className="pt-14 bg-white">
        <HeroSection />
        <CompaniesSection />
        <ServicesSection />
        <ProposalSection />
        <CaseStudiesSection />
        <WorkingProcessSection />
        <TeamSection />
        <TestimonialsSection />
        <PricingSection />
        <ContactSection />
        <LandingFooter />
      </main>
    </div>
  )
}
