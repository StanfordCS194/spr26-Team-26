import type { TaskType, TrainingState } from '../types';

export interface CreateRunResponse {
  run_id: string;
  status: 'running';
}

export interface BackendRunState extends TrainingState {
  run_id: string;
  dataPath?: string | null;
  result?: Record<string, unknown> | null;
}

interface CreateRunRequest {
  prompt: string;
  budget: number;
  task_type: TaskType;
  data_path?: string | null;
}

function envFlagEnabled(value: string | undefined): boolean {
  return ['1', 'true', 'yes', 'on'].includes(value?.trim().toLowerCase() ?? '');
}

export function isSimulationModeConfigured(): boolean {
  return envFlagEnabled(import.meta.env.VITE_USE_SIMULATION);
}

export function getApiBaseUrl(): string | null {
  if (isSimulationModeConfigured()) return null;

  const value = import.meta.env.VITE_API_BASE_URL?.trim();
  const configured = value || '/api';
  return configured.replace(/\/$/, '');
}

export function isBackendApiConfigured(): boolean {
  return !isSimulationModeConfigured();
}

export function resolveApiHrefFromBase(
  baseUrl: string | null,
  downloadPath?: string | null
): string | null {
  if (!baseUrl || !downloadPath) return null;

  if (/^https?:\/\//i.test(downloadPath)) {
    return downloadPath;
  }

  const path = downloadPath.startsWith('/') ? downloadPath : `/${downloadPath}`;
  if (path === baseUrl || path.startsWith(`${baseUrl}/`)) {
    return path;
  }

  try {
    const base = new URL(baseUrl);
    if (path === base.pathname || path.startsWith(`${base.pathname}/`)) {
      return `${base.origin}${path}`;
    }
  } catch {
    // Relative API base, handled below.
  }

  return `${baseUrl}${path}`;
}

export function resolveApiHref(downloadPath?: string | null): string | null {
  return resolveApiHrefFromBase(getApiBaseUrl(), downloadPath);
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const baseUrl = getApiBaseUrl();
  if (!baseUrl) {
    throw new Error('Backend API is not configured.');
  }

  const headers = new Headers(options?.headers);
  headers.set('Content-Type', 'application/json');
  const url = `${baseUrl}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch {
    throw new Error(
      `Manager API is not reachable at ${baseUrl}. Start the backend, set VITE_API_BASE_URL to its /api prefix, or set VITE_USE_SIMULATION=1 for a static demo.`
    );
  }
  if (!response.ok) {
    const body = await response.text();
    const detail = body.trim() && !body.trim().startsWith('<')
      ? body.trim()
      : `Manager API request failed with status ${response.status}.`;
    throw new Error(
      `${detail} Check that the Manager API is running and VITE_API_BASE_URL points to its /api prefix.`
    );
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

export function cancelRun(runId: string): Promise<BackendRunState> {
  return requestJson<BackendRunState>(`/runs/${runId}/cancel`, {
    method: 'POST',
  });
}
