import type { TaskType, TrainingState } from '../types';

export interface CreateRunResponse {
  run_id: string;
  status: 'running';
}

export interface BackendRunState extends TrainingState {
  run_id: string;
  result?: Record<string, unknown> | null;
}

interface CreateRunRequest {
  prompt: string;
  budget: number;
  task_type: TaskType;
  data_path?: string | null;
}

export function getApiBaseUrl(): string | null {
  const value = import.meta.env.VITE_API_BASE_URL?.trim();
  if (!value) return null;
  return value.replace(/\/$/, '');
}

export function isBackendApiConfigured(): boolean {
  return getApiBaseUrl() !== null;
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    throw new Error('Backend API is not configured.');
  }

  const headers = new Headers(options?.headers);
  headers.set('Content-Type', 'application/json');
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function createRun(request: CreateRunRequest): Promise<CreateRunResponse> {
  return requestJson<CreateRunResponse>('/runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export function getRun(runId: string): Promise<BackendRunState> {
  return requestJson<BackendRunState>(`/runs/${runId}`);
}
