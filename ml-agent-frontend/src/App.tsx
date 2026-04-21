import { useTrainingSimulation } from './hooks/useTrainingSimulation';
import InputForm from './components/InputForm';
import Dashboard from './components/Dashboard';
import type { TaskType } from './types';

export default function App() {
  const { state, start, reset } = useTrainingSimulation();

  const handleStart = (prompt: string, budget: number, taskType: TaskType) => {
    start(prompt, budget, taskType);
  };

  if (state.status === 'idle') {
    return <InputForm onStart={handleStart} />;
  }

  return <Dashboard state={state} onReset={reset} />;
}
