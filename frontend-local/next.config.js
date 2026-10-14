/** @type {import('next').NextConfig} */
module.exports = {
  output: process.env.NEXT_OUTPUT === 'standalone' ? 'standalone' : undefined,
  async rewrites() {
    const apiUrl = process.env.API_URL || 'http://localhost:8000'
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ]
  },
}
