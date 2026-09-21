// Minimal typed fetch helper. Generated OpenAPI types (`npm run gen:api`) will
// land in ./schema.d.ts once the FastAPI backend exists (Milestone 7).

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`)
  if (!res.ok) throw new ApiError(res.status, await res.text())
  return (await res.json()) as T
}

export interface HealthResponse {
  status: string
  version: string
}
