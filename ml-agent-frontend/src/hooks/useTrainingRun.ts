import { useCallback, useEffect, useRef, useState } from 'react';
import { createRun, getRun } from '../api/runs';
import type { TaskType, TrainingState } from '../types';

function makeInitialState(): TrainingState {
  return {
    status: 'idle',
    stage: -1,
    prompt: '',
    budget: 50,
    taskType: 'classification',
    costSpent: 0,
    metrics: [],
    iterations: [],
    logs: [],
    stages: [],
    error: null,
  };
}

function makeStartingState(prompt: string, budget: number, taskType: TaskType): TrainingState {
  return {
    status: 'running',
    stage: 0,
    prompt,
    budget,
    taskType,
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
    error: null,
  };
}

export function useTrainingRun() {
  const [state, setState] = useState<TrainingState>(makeInitialState);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async (runId: string) => {
    const next = await getRun(runId);
    setState({
      status: next.status,
      stage: next.stage,
      prompt: next.prompt,
      budget: next.budget,
      taskType: next.taskType,
      costSpent: next.costSpent,
      metrics: next.metrics,
      iterations: next.iterations,
      logs: next.logs,
      stages: next.stages,
      error: next.error,
      runId: next.run_id,
    });

    if (next.status === 'complete' || next.status === 'failed') {
      stopPolling();
    }
  }, [stopPolling]);

  const start = useCallback(async (prompt: string, budget: number, taskType: TaskType) => {
    stopPolling();
    setState(makeStartingState(prompt, budget, taskType));

    try {
      const created = await createRun({
        prompt,
        budget,
        task_type: taskType,
        data_path: null,
      });
      setState(prev => ({ ...prev, runId: created.run_id }));
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
    setState(makeInitialState());
  }, [stopPolling]);

  return { state, start, reset };
}
