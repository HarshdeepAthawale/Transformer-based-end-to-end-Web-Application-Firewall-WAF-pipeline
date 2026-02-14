import React from "react"
import type { Metadata } from 'next'
import { Inter, JetBrains_Mono, Space_Grotesk } from 'next/font/google'
import { Analytics } from '@vercel/analytics/react'
import { ThemeProvider } from 'next-themes'
import { AuthSessionProvider } from '@/components/providers/session-provider'
import './globals.css'

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
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem storageKey="waf-theme">
          <AuthSessionProvider>
            {children}
          </AuthSessionProvider>
          <Analytics />
        </ThemeProvider>
      </body>
    </html>
  )
}
