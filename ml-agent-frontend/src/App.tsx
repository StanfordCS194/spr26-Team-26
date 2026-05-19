import { useTrainingSimulation } from './hooks/useTrainingSimulation';
import InputForm from './components/InputForm';
import Dashboard from './components/Dashboard';
import type { TaskType, SkillLevel } from './types';

export default function App() {
  const { state, start, approve, reject, reset } = useTrainingSimulation();

  const handleStart = (prompt: string, budget: number, taskType: TaskType, skillLevel: SkillLevel) => {
    start(prompt, budget, taskType, skillLevel);
  };

  if (state.status === 'idle') {
    return <InputForm onStart={handleStart} />;
  }

  return <Dashboard state={state} onReset={reset} onApprove={approve} onReject={reject} />;
}
