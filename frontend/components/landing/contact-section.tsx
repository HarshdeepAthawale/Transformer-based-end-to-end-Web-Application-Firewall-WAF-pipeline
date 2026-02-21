'use client'

import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { MessageSquare } from 'lucide-react'

export function ContactSection() {
  const [formType, setFormType] = useState<'hi' | 'quote'>('hi')

  return (
    <section id="contact" className="py-20 lg:py-28 bg-[#ffffff]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2
            className="text-3xl lg:text-4xl font-bold mb-4"
            style={{ color: 'var(--positivus-green)', fontFamily: 'var(--font-space-grotesk)' }}
          >
            Get in touch
          </h2>
          <p
            className="text-lg max-w-2xl"
            style={{ color: '#000000' }}
          >
            Questions about securing your applications? We&apos;re here to help.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-start">
          <form className="space-y-6">
            <div className="flex gap-8">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="formType"
                  checked={formType === 'hi'}
                  onChange={() => setFormType('hi')}
                  className="accent-[var(--positivus-green)]"
                />
                <span style={{ color: '#000000' }}>Say Hi</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="formType"
                  checked={formType === 'quote'}
                  onChange={() => setFormType('quote')}
                  className="accent-[var(--positivus-green)]"
                />
                <span style={{ color: '#000000' }}>Enterprise</span>
              </label>
            </div>

            <div className="space-y-4">
              <div>
                <label
                  htmlFor="name"
                  className="block text-sm font-medium mb-2"
                  style={{ color: '#000000' }}
                >
                  Name
                </label>
                <Input
                  id="name"
                  name="name"
                  placeholder="Name"
                  required
                  className="rounded-none border-2 w-full max-w-md"
                  style={{
                    borderColor: '#000000',
                    backgroundColor: '#ffffff',
                  }}
                />
              </div>
              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium mb-2"
                  style={{ color: '#000000' }}
                >
                  Email *
                </label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="Email"
                  required
                  className="rounded-none border-2 w-full max-w-md"
                  style={{
                    borderColor: '#000000',
                    backgroundColor: '#ffffff',
                  }}
                />
              </div>
              <div>
                <label
                  htmlFor="message"
                  className="block text-sm font-medium mb-2"
                  style={{ color: '#000000' }}
                >
                  Message *
                </label>
                <textarea
                  id="message"
                  name="message"
                  required
                  placeholder="Message"
                  rows={5}
                  className="w-full max-w-md rounded-none border-2 p-3 resize-none"
                  style={{
                    borderColor: '#000000',
                    backgroundColor: '#ffffff',
                  }}
                />
              </div>
            </div>

            <Button
              type="submit"
              className="rounded-none font-semibold px-8 py-3"
              style={{
                backgroundColor: '#000000',
                color: '#ffffff',
              }}
            >
              Send Message
            </Button>
          </form>

          <div className="flex justify-center lg:justify-end">
            <div
              className="w-full max-w-sm aspect-square rounded-2xl flex items-center justify-center border-2 border-black/10"
              style={{ backgroundColor: '#E8F5B8' }}
            >
              <MessageSquare
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
