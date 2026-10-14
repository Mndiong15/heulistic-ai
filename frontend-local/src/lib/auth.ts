export async function getIdToken(): Promise<string> {
  return 'dev-localuser'
}

export async function getCurrentUser(): Promise<{ userId: string; email: string } | null> {
  return { userId: 'localuser', email: 'dev@local' }
}

export async function signOut(): Promise<void> {}

// Unused stubs — satisfy imports from components shared with the cloud frontend
export const confirmResetPassword = async () => {}
export const confirmSignUp = async () => {}
export const resetPassword = async () => {}
export const signIn = async () => ({ isSignedIn: false as const, nextStep: { signInStep: 'DONE' as const } })
export const signUp = async () => ({ isSignUpComplete: false as const, nextStep: { signUpStep: 'DONE' as const } })
