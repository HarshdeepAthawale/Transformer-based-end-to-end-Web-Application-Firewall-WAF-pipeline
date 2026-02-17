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
    // Always proxy /api/* to backend so Copilot and API work even when backend is only reachable via same host.
    // Server-side: BACKEND_URL (e.g. http://backend:3001 in Docker) or default localhost:3001.
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:3001'
    return [
      { source: '/api/:path*', destination: `${backendUrl.replace(/\/$/, '')}/api/:path*` },
    ]
  },
}

export default nextConfig
