import { useState } from 'react';
import type { TaskType, SkillLevel } from '../types';

interface Props {
  onStart: (prompt: string, budget: number, taskType: TaskType, skillLevel: SkillLevel) => void;
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem 1rem',
  },
  card: {
    background: 'var(--bg-surface)',
    border: '0.5px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '2rem',
    width: '100%',
    maxWidth: '560px',
  },
  header: {
    marginBottom: '1.5rem',
  },
  logo: {
    fontSize: '11px',
    fontWeight: 500,
    letterSpacing: '0.12em',
    textTransform: 'uppercase' as const,
    color: 'var(--accent)',
    marginBottom: '0.5rem',
  },
  title: {
    fontSize: '20px',
    fontWeight: 500,
    color: 'var(--text-primary)',
    marginBottom: '0.25rem',
  },
  subtitle: {
    fontSize: '13px',
    color: 'var(--text-muted)',
  },
  fieldset: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
  },
  field: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.375rem',
  },
  label: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    fontWeight: 500,
  },
  textarea: {
    background: 'var(--bg-elevated)',
    border: '0.5px solid var(--border)',
    borderRadius: '6px',
    color: 'var(--text-primary)',
    fontSize: '14px',
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
    fontSize: '14px',
    padding: '0.625rem 0.75rem',
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
    fontSize: '14px',
    padding: '0.625rem 0.75rem',
    width: '100%',
    fontFamily: 'inherit',
    outline: 'none',
    cursor: 'pointer',
    appearance: 'none' as const,
    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2364748b'/%3E%3C/svg%3E")`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'right 0.75rem center',
    paddingRight: '2rem',
    transition: 'border-color 0.15s',
  },
  error: {
    fontSize: '12px',
    color: 'var(--danger)',
    marginTop: '0.25rem',
  },
  actions: {
    display: 'flex',
    gap: '0.75rem',
    marginTop: '0.5rem',
  },
  btnPrimary: {
    flex: 1,
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
    fontSize: '13px',
    fontWeight: 600,
  },
  segHint: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    fontWeight: 400,
    letterSpacing: '0.02em',
  },
  levelDesc: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    marginTop: '0.375rem',
    lineHeight: 1.5,
  },
};

function segBtnStyle(active: boolean): React.CSSProperties {
  return {
    padding: '0.625rem 0.5rem',
    background: active ? 'var(--accent-dim)' : 'var(--bg-elevated)',
    border: active ? '0.5px solid var(--accent)' : '0.5px solid var(--border)',
    borderRadius: '6px',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    fontFamily: 'inherit',
    textAlign: 'center' as const,
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '2px',
  };
}

const SKILL_DESCRIPTIONS: Record<SkillLevel, string> = {
  beginner:
    'Fully autonomous — the agent handles every decision. Clean summaries, no interruptions. Best if you just want results.',
  intermediate:
    'You approve the dataset and model choices before training. Richer logs with component tags. Some technical detail.',
  expert:
    'Full visibility: approval gates, raw config diffs, hyperparameter search space, unfiltered logs, token counts. No training wheels.',
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
      <div style={styles.card}>
        <div style={styles.header}>
          <p style={styles.logo}>CS194 · Team 26</p>
          <h1 style={styles.title}>ML Training Agent</h1>
          <p style={styles.subtitle}>Autonomous model training with cost guardrails</p>
        </div>

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

          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label} htmlFor="budget">Budget Cap (USD)</label>
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
              <label style={styles.label} htmlFor="taskType">Task Type</label>
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
            <label style={styles.label}>ML Experience Level</label>
            <div style={styles.segGroup} role="radiogroup" aria-label="ML experience level">
              {(['beginner', 'intermediate', 'expert'] as SkillLevel[]).map(lvl => {
                const active = skillLevel === lvl;
                const hint = lvl === 'beginner' ? 'auto-pilot' : lvl === 'intermediate' ? 'some control' : 'full control';
                return (
                  <button
                    key={lvl}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    style={segBtnStyle(active)}
                    onClick={() => setSkillLevel(lvl)}
                  >
                    <span style={styles.segLabel}>
                      {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                    </span>
                    <span style={styles.segHint}>{hint}</span>
                  </button>
                );
              })}
            </div>
            <p style={styles.levelDesc}>{SKILL_DESCRIPTIONS[skillLevel]}</p>
          </div>

          {error && <p style={styles.error} role="alert">{error}</p>}

          <div style={styles.actions}>
            <button style={styles.btnPrimary} onClick={handleSubmit} aria-label="Start training">
              Start Training
            </button>
            <button style={styles.btnSecondary} onClick={handleReset} aria-label="Reset form">
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
