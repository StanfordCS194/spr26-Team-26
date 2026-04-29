import { useState } from 'react';
import type { TaskType, SkillLevel } from '../types';

interface Props {
  onStart: (prompt: string, budget: number, taskType: TaskType, skillLevel: SkillLevel) => void;
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '8rem 1rem',
    position: 'relative'
  },
  card: {
    background: 'var(--bg-surface)',
    padding: '2rem',
    borderRadius: '16px',
    boxShadow: '0px 0px 200px 15px rgba(59, 130, 246, 0.5)'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '3fr 2fr',
    gap: '1rem',
    width: '1250px',
  },
  header: {
    marginBottom: '1.5rem',
    textAlign: 'center'
  },
  logo: {
    fontSize: '12px',
    fontWeight: 500,
    letterSpacing: '0.12em',
    textTransform: 'uppercase' as const,
    color: 'var(--accent)',
    marginBottom: '0.5rem',
    position: 'absolute',
    top: '1.5rem',
    left: '1.5rem',
  },
  title: {
    fontSize: '50px',
    fontWeight: 500,
    color: 'var(--text-primary)',
    marginBottom: '0.25rem',
  },
  subtitle: {
    fontSize: '25px',
    color: 'var(--text-muted)',
    textAlign: 'center'
  },
  fieldset: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2rem',
  },
  field: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.375rem',
  },
  label: {
    fontSize: '35px',
    color: 'var(--text-muted)',
    fontWeight: 500,
  },
  smallLabel: {
    fontSize: '25px',
    color: 'var(--text-muted)',
    fontWeight: 500,
  },
  textarea: {
    background: 'var(--bg-elevated)',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '20px',
    padding: '0.625rem 0.75rem',
    resize: 'vertical' as const,
    minHeight: '80px',
    fontFamily: 'inherit',
    outline: 'none',
    transition: 'border-color 0.15s',
    width: '100%',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '0.75rem',
  },
  input: {
    background: 'var(--bg-elevated)',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '20px',
    padding: '1rem 1.25rem',
    width: '100%',
    fontFamily: 'inherit',
    outline: 'none',
    transition: 'border-color 0.15s',
  },
  select: {
    background: 'var(--bg-elevated)',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '20px',
    padding: '1rem 1.25rem',
    width: '100%',
    fontFamily: 'inherit',
    outline: 'none',
    cursor: 'pointer',
    appearance: 'none' as const,
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2364748b'/%3E%3C/svg%3E")`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 0.75rem center',
    paddingRight: '4rem',
    transition: 'border-color 0.15s',
  },
  error: {
    fontSize: '20px',
    color: 'var(--danger)',
    marginTop: '0.25rem',
  },
  actions: {
    display: 'flex',
    gap: '2rem',
    marginTop: '0.5rem',
    justifyContent: 'center'
  },
  btnPrimary: {
    padding: '0.625rem 1.25rem',
    background: 'var(--accent-dim)',
    border: '0.5px solid var(--accent)',
    borderRadius: '6px',
    color: 'var(--accent)',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'background 0.15s',
    fontFamily: 'inherit',
  },
  btnSecondary: {
    padding: '0.625rem 1rem',
    background: 'transparent',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-secondary)',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background 0.15s',
    fontFamily: 'inherit',
  },
  segGroup: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '6px',
  },
  segLabel: {
    fontSize: '20px',
    fontWeight: 600,
  },
  segHint: {
    fontSize: '15px',
    color: 'var(--text-muted)',
    fontWeight: 400,
    letterSpacing: '0.02em',
  },
  levelDesc: {
    fontSize: '20px',
    color: 'var(--text-muted)',
    marginTop: '0.375rem',
    lineHeight: 1.5,
  },
};


const SKILL_DESCRIPTIONS: Record<SkillLevel, string> = {
  beginner:
    'Fully autonomous: the agent runs from start to finish on its own',
  intermediate:
    'You approve the dataset and model type before training begins.',
  expert:
    'Full visibility: approval gates, raw config diffs, hyperparameter search space, unfiltered logs, token counts.',
};

export default function InputForm({ onStart }: Props) {
  const [prompt, setPrompt] = useState('');
  const [budget, setBudget] = useState(50);
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [skillLevel, setSkillLevel] = useState<SkillLevel>('intermediate');
  const [error, setError] = useState('');

  const handleSubmit = () => {
    if (prompt.trim().length < 10) {
      setError('Prompt must be at least 10 characters.');
      return;
    }
    if (budget < 10 || budget > 500) {
      setError('Budget must be between $10 and $500.');
      return;
    }
    setError('');
    onStart(prompt.trim(), budget, taskType, skillLevel);
  };

  const handleReset = () => {
    setPrompt('');
    setBudget(50);
    setTaskType('classification');
    setSkillLevel('intermediate');
    setError('');
  };
  return (
    <div style={styles.wrapper}>
      <p style={styles.logo}>CS194 · Team 26</p>
      <div style={styles.header}>
        <h1 style={styles.title}>AutoTrain Agent</h1>
        <p style={styles.subtitle}>Autonomous model training with cost guardrails</p>
      </div>

      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.fieldset}>
            <div style={styles.field}>
              <label style={styles.label} htmlFor="prompt">Training Objective</label>
              <textarea
                id="prompt"
                style={styles.textarea}
                placeholder="e.g. Classify sentiment in movie reviews"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                maxLength={500}
                aria-label="Training prompt"
              />
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'right' }}>
                {prompt.length}/500
              </span>
            </div>
            <div style={styles.actions}>
            <button className="btn-primary" onClick={handleSubmit} aria-label="Start training">
                Start Training
              </button>
              <button className="btn-secondary" onClick={handleReset} aria-label="Reset form">
                Reset
              </button>
            </div>
            {error && <p style={styles.error} role="alert">{error}</p>}
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.fieldset}>
            <div style={styles.row}>
              <div style={styles.field}>
                <label style={styles.smallLabel} htmlFor="budget">Budget Cap (USD)</label>
                <input
                  id="budget"
                  type="number"
                  style={styles.input}
                  value={budget}
                  min={10}
                  max={500}
                  step={5}
                  onChange={e => setBudget(Number(e.target.value))}
                  aria-label="Budget cap in dollars"
                />
              </div>
              <div style={styles.field}>
                <label style={styles.smallLabel} htmlFor="taskType">Task Type</label>
                <select
                  id="taskType"
                  style={styles.select}
                  value={taskType}
                  onChange={e => setTaskType(e.target.value as TaskType)}
                  aria-label="Task type"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="fine-tuning">Fine-tuning</option>
                </select>
              </div>
            </div>
            <div style={styles.field}>
              <label style={styles.smallLabel}>ML Experience Level</label>
              <div style={styles.segGroup} role="radiogroup" aria-label="ML experience level">
                {(['beginner', 'intermediate', 'expert'] as SkillLevel[]).map(lvl => {
                  const active = skillLevel === lvl;
                  return (
                    <button
                      key={lvl}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      className={`seg-btn ${active ? 'seg-btn--active' : ''}`}
                      onClick={() => setSkillLevel(lvl)}
                    >
                      <span style={styles.segLabel}>
                        {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                      </span>
                    </button>
                  );
                })}
              </div>
              <p style={styles.levelDesc}>{SKILL_DESCRIPTIONS[skillLevel]}</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  ); 
}

