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

        // Only harshdeepathawale27@gmail.com has admin access (via credentials)
        const adminEmail = 'harshdeepathawale27@gmail.com'
        const adminPassword = process.env.ADMIN_PASSWORD ?? ''
        if (
          credentials.email === adminEmail &&
          credentials.password === adminPassword &&
          adminPassword
        ) {
          return {
            id: 'admin',
            email: adminEmail,
            name: 'Admin',
            image: null,
            role: 'admin',
          }
        }
        return null
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
      }
      return token
    },
    async session({ session, token }) {
      if (session.user) {
        (session.user as { id?: string }).id = token.id as string
        (session.user as { role?: string }).role = token.role as string
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
