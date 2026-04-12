import React from "react"
import type { Metadata } from 'next'
import Script from 'next/script'
import { Inter, JetBrains_Mono, Space_Grotesk } from 'next/font/google'
import { Analytics } from '@vercel/analytics/react'
import { ThemeProvider } from 'next-themes'
import { AuthSessionProvider } from '@/components/providers/session-provider'
import { TimezoneProvider } from '@/contexts/timezone-context'
import { DomainProvider } from '@/contexts/domain-context'
import './globals.css'

// Run before first paint so light/dark matches stored preference (avoids flash on settings open)
const themeInitScript = `
(function(){
  var k='waf-theme';
  var s=null;
  try{s=localStorage.getItem(k);}catch(e){}
  var p=s||'system';
  var r=p==='system'?(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):p;
  document.documentElement.classList.add(r);
})();
`

// Configure Inter font for clean, readable typography
const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
  preload: true,
})

// Configure JetBrains Mono for monospace text
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
  preload: true,
})

// Space Grotesk - Positivus landing page typography
const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
  preload: true,
})

export const metadata: Metadata = {
  title: 'WAF - Web Application Firewall | Secure Your Digital Presence',
  description: 'Professional Web Application Firewall with AI-powered threat detection. Protect your applications from attacks with real-time security monitoring and analytics',
  generator: 'v0.app',
  icons: {
    icon: '/icon.svg',
    apple: '/apple-icon.png',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} ${spaceGrotesk.variable}`} suppressHydrationWarning>
      <body className={`font-sans antialiased bg-background text-foreground`}>
        <Script id="theme-init" strategy="beforeInteractive" dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem storageKey="waf-theme">
          <TimezoneProvider>
            <AuthSessionProvider>
              <DomainProvider>
                {children}
              </DomainProvider>
            </AuthSessionProvider>
          </TimezoneProvider>
          <Analytics />
        </ThemeProvider>
      </body>
    </html>
  )
}
