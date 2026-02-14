/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  env: {
    // Empty API_URL = use proxy (rewrites below). Set to backend URL (e.g. http://localhost:3001) for direct calls.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? '',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001/ws/',
  },
  async rewrites() {
    // Proxy API requests to backend when API_URL is not set (local dev)
    if (!process.env.NEXT_PUBLIC_API_URL) {
      return [
        { source: '/api/:path*', destination: 'http://localhost:3001/api/:path*' },
      ]
    }
    return []
  },
}

export default nextConfig
