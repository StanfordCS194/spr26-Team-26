import { useState } from 'react';
import { useTrainingSimulation } from './hooks/useTrainingSimulation';
import InputForm from './components/InputForm';
import Dashboard from './components/Dashboard';
import DiffViewer from './components/DiffViewer';
import type { TaskType, ExperienceLevel } from './types';

type View = 'dashboard' | 'diffs';

export default function App() {
  const { state, start, reset } = useTrainingSimulation();
  const [view, setView] = useState<View>('dashboard');

  const handleStart = (prompt: string, budget: number, taskType: TaskType, experience: ExperienceLevel) => {
    setView('dashboard');
    start(prompt, budget, taskType, experience);
  };

  const handleReset = () => {
    setView('dashboard');
    reset();
  };

  if (state.status === 'idle') {
    return <InputForm onStart={handleStart} />;
  }

  if (view === 'diffs') {
    return (
      <DiffViewer
        iterations={state.iterations}
        status={state.status}
        onBack={() => setView('dashboard')}
      />
    );
  }

  return (
    <Dashboard
      state={state}
      onReset={handleReset}
      onOpenDiffs={() => setView('diffs')}
    />
  );
}
