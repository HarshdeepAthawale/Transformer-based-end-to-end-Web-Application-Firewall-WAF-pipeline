import { withAuth } from 'next-auth/middleware'

export default withAuth({
  pages: {
    signIn: '/login',
  },
  callbacks: {
    authorized({ token }) {
      // Only admin role can access /admin
      return token?.role === 'admin'
    },
  },
})

export const config = {
  matcher: ['/admin/:path*', '/users'],
}
