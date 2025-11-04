import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'vLLM Inference Server',
  description: 'Interact with your FastAPI vLLM server',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

