import { useQuery } from '@tanstack/react-query'
import { apiGet, type HealthResponse } from './api/client'

export default function App() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<HealthResponse>('/health'),
    retry: false,
  })

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-800 px-4 py-3 flex items-center justify-between">
        <h1 className="font-semibold">ragchat</h1>
        <span className="text-sm text-zinc-500">
          backend:{' '}
          {health.isPending && 'checking…'}
          {health.isError && <span className="text-red-500">unreachable</span>}
          {health.data && <span className="text-green-600">ok (v{health.data.version})</span>}
        </span>
      </header>
      <main className="flex-1 grid place-items-center p-4 text-zinc-500">
        Chat UI arrives in Milestone 7.
      </main>
    </div>
  )
}
