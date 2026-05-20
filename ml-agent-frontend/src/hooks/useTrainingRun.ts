import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelRun, createRun, getRun, type BackendRunState } from '../api/runs';
import type { PipelineStage, StartTraining, StageStatus, TaskType, TrainingState } from '../types';

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
    provenance: null,
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
    provenance: null,
    result: null,
    error: null,
  };
}

function isTerminalStatus(status: TrainingState['status']) {
  return status === 'complete' || status === 'failed' || status === 'cancelled';
}

function terminalStageStatus(status: TrainingState['status']): Extract<StageStatus, 'failed' | 'cancelled'> | null {
  if (status === 'failed' || status === 'cancelled') return status;
  return null;
}

function terminalStageIndex(stages: PipelineStage[], stage: number): number {
  if (stages.length === 0) return -1;

  const activeStage = stages.findIndex(item => item.status === 'in-progress');
  if (activeStage >= 0) return activeStage;

  return Math.max(0, Math.min(stage, stages.length - 1));
}

function markTerminalStage(
  stages: PipelineStage[],
  stage: number,
  status: Extract<StageStatus, 'failed' | 'cancelled'>
): PipelineStage[] {
  const activeStage = terminalStageIndex(stages, stage);
  if (activeStage < 0) return stages;

  return stages.map((item, index) => {
    if (index < activeStage) return { ...item, status: 'complete' };
    if (index === activeStage) return { ...item, status };
    return { ...item, status: 'pending' };
  });
}

function withTerminalStages(state: TrainingState): TrainingState {
  const status = terminalStageStatus(state.status);
  if (!status) return state;

  return {
    ...state,
    stages: markTerminalStage(state.stages, state.stage, status),
  };
}

function stateFromBackend(next: BackendRunState): TrainingState {
  return withTerminalStages({
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
    provenance: next.provenance ?? null,
    result: next.result ?? null,
    error: next.error,
    runId: next.run_id,
  });
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
            ...withTerminalStages({ ...prev, status: 'failed' }),
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
          }));
        });
      }, 2000);
    } catch (error) {
      setState(prev => ({
        ...withTerminalStages({ ...prev, status: 'failed' }),
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
        ...withTerminalStages({ ...prev, status: 'failed' }),
        status: 'failed',
        error: error instanceof Error ? error.message : String(error),
      }));
      stopPolling();
    }
  }, [state.runId, state.status, stopPolling]);

  return { state, start, reset, cancel };
}
