import { useTrainingSimulation } from './hooks/useTrainingSimulation';
import { useTrainingRun } from './hooks/useTrainingRun';
import { isBackendApiConfigured } from './api/runs';
import InputForm from './components/InputForm';
import Dashboard from './components/Dashboard';
import type { StartTraining } from './types';

export default function App() {
  const simulation = useTrainingSimulation();
  const backend = useTrainingRun();
  const controller = isBackendApiConfigured() ? backend : simulation;
  const { state, start, reset, cancel } = controller;

  const handleStart: StartTraining = (prompt, budget, taskType, dataPath = null) => {
    return start(prompt, budget, taskType, dataPath);
  };

  if (state.status === 'idle') {
    return <InputForm onStart={handleStart} />;
  }

  return <Dashboard state={state} onReset={reset} onCancel={cancel} />;
}
