import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelRun, createRun, getRun, type BackendRunState } from '../api/runs';
import type { StartTraining, TaskType, TrainingState } from '../types';

function makeInitialState(): TrainingState {
  return {
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    dataPath: null,
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: [],
    artifacts: null,
    result: null,
    error: null,
  };
}

function makeStartingState(
  prompt: string,
  budget: number,
  taskType: TaskType,
  dataPath?: string | null
): TrainingState {
  return {
    status: 'running',
    stage: 0,
    prompt,
    budget,
    taskType,
    dataPath: dataPath ?? null,
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: [
      { id: 0, label: 'Manager Init', status: 'in-progress' },
      { id: 1, label: 'Data Discovery', status: 'pending' },
      { id: 2, label: 'Model Selection', status: 'pending' },
      { id: 3, label: 'Training', status: 'pending' },
      { id: 4, label: 'AutoResearch', status: 'pending' },
      { id: 5, label: 'Finalization', status: 'pending' },
    ],
    artifacts: null,
    result: null,
    error: null,
  };
}

function isTerminalStatus(status: TrainingState['status']) {
  return status === 'complete' || status === 'failed' || status === 'cancelled';
}

function stateFromBackend(next: BackendRunState): TrainingState {
  return {
    status: next.status,
    stage: next.stage,
    prompt: next.prompt,
    budget: next.budget,
    taskType: next.taskType,
    dataPath: next.dataPath ?? null,
    costSpent: next.costSpent,
    metrics: next.metrics,
    iterations: next.iterations,
    logs: next.logs,
    stages: next.stages,
    artifacts: next.artifacts ?? null,
    result: next.result ?? null,
    error: next.error,
    runId: next.run_id,
  };
}

export function useTrainingRun() {
  const [state, setState] = useState<TrainingState>(makeInitialState);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const runIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async (runId: string) => {
    const next = await getRun(runId);
    setState(stateFromBackend(next));

    if (isTerminalStatus(next.status)) {
      stopPolling();
    }
  }, [stopPolling]);

  const start = useCallback<StartTraining>(async (prompt, budget, taskType, dataPath = null) => {
    const normalizedDataPath = dataPath?.trim() || null;
    stopPolling();
    setState(makeStartingState(prompt, budget, taskType, normalizedDataPath));

    try {
      const created = await createRun({
        prompt,
        budget,
        task_type: taskType,
        data_path: normalizedDataPath,
      });
      setState(prev => ({ ...prev, runId: created.run_id }));
      runIdRef.current = created.run_id;
      await poll(created.run_id);
      pollRef.current = setInterval(() => {
        void poll(created.run_id).catch((error: unknown) => {
          stopPolling();
          setState(prev => ({
            ...prev,
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
          }));
        });
      }, 2000);
    } catch (error) {
      setState(prev => ({
        ...prev,
        status: 'failed',
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [poll, stopPolling]);

  const reset = useCallback(() => {
    stopPolling();
    runIdRef.current = null;
    setState(makeInitialState());
  }, [stopPolling]);

  const cancel = useCallback(async () => {
    const runId = runIdRef.current ?? state.runId;
    if (!runId || isTerminalStatus(state.status)) {
      return;
    }

    setState(prev => ({ ...prev, status: 'cancelling' }));
    try {
      const next = await cancelRun(runId);
      setState(stateFromBackend(next));
      if (isTerminalStatus(next.status)) {
        stopPolling();
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        status: 'failed',
        error: error instanceof Error ? error.message : String(error),
      }));
      stopPolling();
    }
  }, [state.runId, state.status, stopPolling]);

  return { state, start, reset, cancel };
}
