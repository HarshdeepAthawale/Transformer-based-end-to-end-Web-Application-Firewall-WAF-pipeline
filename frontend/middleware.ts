import { withAuth } from 'next-auth/middleware'
import { NextResponse } from 'next/server'

export default withAuth(
  function middleware(req) {
    const { pathname } = req.nextUrl
    const token = req.nextauth.token

    // Redirect authenticated users away from landing page and login page
    if ((pathname === '/' || pathname === '/login') && token) {
      return NextResponse.redirect(new URL('/dashboard', req.url))
    }

    // Admin-only routes: non-admin authenticated users get redirected to dashboard
    if (
      (pathname.startsWith('/admin') || pathname.startsWith('/users')) &&
      token?.role !== 'admin'
    ) {
      return NextResponse.redirect(new URL('/dashboard', req.url))
    }

    return NextResponse.next()
  },
  {
    callbacks: {
      authorized({ token, req }) {
        const { pathname } = req.nextUrl
        // Landing page and login are publicly accessible (middleware handles auth redirect above)
        if (pathname === '/' || pathname === '/login') return true
        // All other matched paths require an authenticated session
        return !!token
      },
    },
    pages: {
      signIn: '/login',
    },
  }
)

export const config = {
  matcher: [
    '/',
    '/login',
    '/dashboard/:path*',
    '/analytics/:path*',
    '/traffic/:path*',
    '/threats/:path*',
    '/dos-protection/:path*',
    '/upload-scanning/:path*',
    '/firewall-ai/:path*',
    '/credential-protection/:path*',
    '/ip-management/:path*',
    '/geo-rules/:path*',
    '/bot-detection/:path*',
    '/threat-intelligence/:path*',
    '/security-insights/:path*',
    '/emergency-rules/:path*',
    '/security-rules/:path*',
    '/managed-rules/:path*',
    '/users/:path*',
    '/audit-logs/:path*',
    '/settings/:path*',
    '/domains/:path*',
    '/copilot/:path*',
    '/campaigns/:path*',
    '/admin/:path*',
  ],
}
