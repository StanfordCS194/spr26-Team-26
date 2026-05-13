import { useState } from 'react';
import type { TaskType } from '../types';

// ── Hardcoded prompt options ──────────────────────────────────────────────────
const DEMO_PROMPTS: Array<{ label: string; prompt: string; taskType: TaskType }> = [
  { label: 'Classify sentiment in movie reviews',                        prompt: 'Classify sentiment in movie reviews using the IMDB dataset',                            taskType: 'classification' },
  { label: 'Detect hate speech and offensive language in tweets',        prompt: 'Detect hate speech and offensive language in tweets',                                   taskType: 'classification' },
  { label: 'Classify news articles by topic (sports, tech, politics)',   prompt: 'Classify news articles into topics: sports, politics, technology, and world news',     taskType: 'classification' },
  { label: 'Identify emotions (joy, sadness, anger, fear) in text',      prompt: 'Identify emotions including joy, sadness, anger, and fear expressed in short texts',   taskType: 'classification' },
  { label: 'Route customer support tickets to the right department',     prompt: 'Route customer support tickets to the correct banking department using intent classification', taskType: 'classification' },
  { label: 'Detect spam and phishing emails',                            prompt: 'Detect spam and phishing emails using the Enron email dataset',                        taskType: 'classification' },
  { label: 'Identify duplicate and paraphrase sentence pairs',           prompt: 'Identify whether two sentences are paraphrases of each other using MRPC',              taskType: 'classification' },
  { label: 'Build an extractive question-answering model (SQuAD)',       prompt: 'Build an extractive question answering model trained on SQuAD passages',               taskType: 'classification' },
  { label: 'Train a natural language inference model (SNLI/MNLI)',       prompt: 'Train a natural language inference model to predict entailment, contradiction, or neutral', taskType: 'classification' },
  { label: 'Summarize news articles into concise bullet points',         prompt: 'Summarize news articles into concise summaries using CNN/DailyMail',                   taskType: 'fine-tuning' },
  { label: 'Summarize scientific papers and arXiv abstracts',            prompt: 'Summarize scientific research papers and arXiv abstracts',                             taskType: 'fine-tuning' },
  { label: 'Fine-tune an LLM to follow instructions (RLHF)',             prompt: 'Fine-tune an LLM to follow instructions using human feedback data from Anthropic hh-rlhf', taskType: 'fine-tuning' },
  { label: 'Build a conversational chatbot with OpenAssistant data',     prompt: 'Build a conversational chatbot fine-tuned on OpenAssistant oasst1 dialogue data',    taskType: 'fine-tuning' },
  { label: 'Extract named entities (people, places, organizations)',     prompt: 'Extract named entities including people, places, and organizations from text using CoNLL-2003', taskType: 'classification' },
  { label: 'Translate English to French',                                prompt: 'Translate English sentences to French using the Helsinki-NLP opus-100 dataset',       taskType: 'fine-tuning' },
  { label: 'Translate English to German',                                prompt: 'Translate English sentences to German using WMT14',                                   taskType: 'fine-tuning' },
];

interface Props {
  onStart: (prompt: string, budget: number, taskType: TaskType) => void;
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
    maxWidth: '520px',
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
};

export default function InputForm({ onStart }: Props) {
  const [prompt, setPrompt] = useState('');
  const [budget, setBudget] = useState(50);
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [error, setError] = useState('');

  const handlePromptSelect = (idx: string) => {
    if (idx === '') { setPrompt(''); return; }
    const option = DEMO_PROMPTS[Number(idx)];
    setPrompt(option.prompt);
    setTaskType(option.taskType);
    setError('');
  };

  const selectedIdx = DEMO_PROMPTS.findIndex(o => o.prompt === prompt);

  const handleSubmit = () => {
    if (prompt.trim().length < 10) {
      setError('Please select a training objective.');
      return;
    }
    if (budget < 10 || budget > 500) {
      setError('Budget must be between $10 and $500.');
      return;
    }
    setError('');
    onStart(prompt.trim(), budget, taskType);
  };

  const handleReset = () => {
    setPrompt('');
    setBudget(50);
    setTaskType('classification');
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
            <select
              id="prompt"
              style={styles.select}
              value={selectedIdx === -1 ? '' : String(selectedIdx)}
              onChange={e => handlePromptSelect(e.target.value)}
              aria-label="Training objective"
            >
              <option value="" disabled>Choose a training objective…</option>
              {DEMO_PROMPTS.map((o, i) => (
                <option key={i} value={String(i)}>{o.label}</option>
              ))}
            </select>
            {prompt && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>
                {prompt}
              </span>
            )}
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
