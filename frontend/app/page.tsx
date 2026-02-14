import {
  LandingHeader,
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

export default function LandingPage() {
  return (
    <div
      className="min-h-screen font-[family-name:var(--font-space-grotesk)]"
      style={{ fontFamily: 'var(--font-space-grotesk)' }}
    >
      <LandingHeader />
      <main>
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
