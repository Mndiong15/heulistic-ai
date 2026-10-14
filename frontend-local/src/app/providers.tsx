'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Suspense, useState } from 'react'
import AuthProvider from '../components/AuthProvider'
import PHProvider from '../components/PostHogProvider'
import PostHogPageview from '../components/PostHogPageview'

export default function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: {
      queries: { staleTime: 0, retry: 1 },
    },
  }))

  return (
    <PHProvider>
      <QueryClientProvider client={client}>
        <AuthProvider>
          <Suspense>
            <PostHogPageview />
          </Suspense>
          {children}
        </AuthProvider>
      </QueryClientProvider>
    </PHProvider>
  )
}
