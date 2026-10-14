'use client'

import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { getCurrentUser } from '../lib/auth'

interface AuthUser {
  userId: string
  email: string
}

interface AuthContextValue {
  user: AuthUser | null
  creditBalance: number | null
  creditDisplay: string
  isLoading: boolean
  isAdmin: boolean
  refreshUser: () => Promise<void>
  refreshCredits: () => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  creditBalance: null,
  creditDisplay: '$0.00',
  isLoading: true,
  isAdmin: false,
  refreshUser: async () => {},
  refreshCredits: async () => {},
  signOut: async () => {},
})

export function useAuth() {
  return useContext(AuthContext)
}

export default function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const refreshCredits = useCallback(async () => {}, [])

  const refreshUser = useCallback(async () => {
    try {
      const currentUser = await getCurrentUser()
      setUser(currentUser)
    } catch {
      setUser(null)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshUser()
  }, [refreshUser])

  return (
    <AuthContext.Provider value={{
      user,
      creditBalance: null,
      creditDisplay: '$0.00',
      isLoading,
      isAdmin: false,
      refreshUser,
      refreshCredits,
      signOut: async () => {},
    }}>
      {children}
    </AuthContext.Provider>
  )
}
