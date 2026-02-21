'use client'

import useEmblaCarousel from 'embla-carousel-react'
import { useCallback, useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Quote } from 'lucide-react'

const testimonials = [
  {
    quote:
      "We deployed in under an hour. The AI caught SQL injection attempts on day one that our previous WAF had missed for months. The dashboard gives us full visibility—exactly what we needed.",
    author: 'CTO, Fintech Startup',
  },
  {
    quote:
      "Latency stayed under 20ms. We block millions of bot requests monthly without impacting real users. The model learns our traffic so we spend less time tuning rules.",
    author: 'Head of Engineering, E‑commerce',
  },
  {
    quote:
      "Enterprise-grade protection without enterprise complexity. Our security team loves the real-time logs. Support actually understands application security.",
    author: 'Security Lead, SaaS Platform',
  },
]

export function TestimonialsSection() {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true })
  const [selectedIndex, setSelectedIndex] = useState(0)

  const scrollPrev = useCallback(() => emblaApi?.scrollPrev(), [emblaApi])
  const scrollNext = useCallback(() => emblaApi?.scrollNext(), [emblaApi])

  const onSelect = useCallback(() => {
    if (!emblaApi) return
    setSelectedIndex(emblaApi.selectedScrollSnap())
  }, [emblaApi])

  useEffect(() => {
    if (!emblaApi) return
    onSelect()
    emblaApi.on('select', onSelect)
  }, [emblaApi, onSelect])

  return (
    <section className="py-20 lg:py-28 bg-[#ffffff]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            What customers say
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: '#000000' }}
          >
            Teams securing their applications with us share their experience
          </p>
        </div>

        <div className="relative" ref={emblaRef}>
          <div className="overflow-hidden">
            <div className="flex -ml-4">
              {testimonials.map((t, i) => (
                <div
                  key={i}
                  className="flex-[0_0_100%] min-w-0 pl-4"
                >
                  <div
                    className="p-8 lg:p-12 max-w-3xl mx-auto relative border-2"
                    style={{
                      backgroundColor: '#ffffff',
                      borderColor: '#000000',
                    }}
                  >
                    <Quote
                      className="w-12 h-12 mb-6 opacity-30"
                      style={{ color: 'var(--positivus-green)' }}
                    />
                    <p
                      className="text-lg lg:text-xl leading-relaxed mb-8"
                      style={{ color: '#000000' }}
                    >
                      &quot;{t.quote}&quot;
                    </p>
                    <div>
                      <h5
                        className="font-semibold"
                        style={{ color: '#000000', fontFamily: 'var(--font-space-grotesk)' }}
                      >
                        {t.author}
                      </h5>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-center gap-4 mt-8">
            <button
              type="button"
              onClick={scrollPrev}
              className="p-2 transition-opacity hover:opacity-70"
              aria-label="Previous testimonial"
            >
              <ChevronLeft
                className="w-8 h-8"
                style={{ color: '#000000' }}
              />
            </button>
            <div className="flex gap-2">
              {testimonials.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => emblaApi?.scrollTo(i)}
                  className="w-2.5 h-2.5 rounded-full transition-all"
                  style={{
                    backgroundColor: selectedIndex === i ? 'var(--positivus-green)' : '#000000',
                    opacity: selectedIndex === i ? 1 : 0.5,
                  }}
                />
              ))}
            </div>
            <button
              type="button"
              onClick={scrollNext}
              className="p-2 transition-opacity hover:opacity-70"
              aria-label="Next testimonial"
            >
              <ChevronRight
                className="w-8 h-8"
                style={{ color: '#000000' }}
              />
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
