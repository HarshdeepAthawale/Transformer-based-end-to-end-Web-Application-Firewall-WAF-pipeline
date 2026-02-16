import type { NextAuthOptions } from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import CredentialsProvider from 'next-auth/providers/credentials'

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? '',
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? '',
    }),
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null

        // Server-side: use BACKEND_URL (Docker: http://backend:3001) or fallback for local dev
        const backendUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'
        try {
          const res = await fetch(`${backendUrl.replace(/\/$/, '')}/api/users/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              username: credentials.email,
              password: credentials.password,
            }),
          })
          if (!res.ok) return null
          const json = await res.json()
          const user = json?.data?.user
          const token = json?.data?.token
          if (!user) return null
          return {
            id: String(user.id),
            email: user.email ?? credentials.email,
            name: user.full_name ?? user.username ?? credentials.email,
            image: null,
            role: user.role ?? 'viewer',
            backendToken: token ?? undefined,
          }
        } catch {
          return null
        }
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      if (account?.provider === 'google') {
        // Only harshdeepathawale27@gmail.com gets admin role
        user.role = user.email === 'harshdeepathawale27@gmail.com' ? 'admin' : 'user'
      }
      return true
    },
    async jwt({ token, user }) {
      if (user) {
        token.role = (user as { role?: string }).role
        token.id = user.id
        token.backendToken = (user as { backendToken?: string }).backendToken
      }
      return token
    },
    async session({ session, token }) {
      if (session.user) {
        (session.user as { id?: string }).id = token.id as string
        (session.user as { role?: string }).role = token.role as string
        ;(session.user as { backendToken?: string }).backendToken = token.backendToken as string | undefined
      }
      return session
    },
  },
  pages: {
    signIn: '/login',
  },
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  secret: process.env.NEXTAUTH_SECRET,
}
