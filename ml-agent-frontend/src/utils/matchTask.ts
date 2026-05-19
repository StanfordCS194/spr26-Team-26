import { ALL_CONFIGS, DEFAULT_CONFIG, type TaskConfig } from '../data/taskConfigs';

/**
 * Scores a prompt against each task config's keyword list and returns
 * the best match. Falls back to DEFAULT_CONFIG if nothing scores.
 */
export function matchTask(prompt: string): TaskConfig {
  const lower = prompt.toLowerCase();
  // Tokenise: split on non-alphanumeric, also keep full lowercased string
  const words = new Set(lower.split(/\W+/).filter(Boolean));

  let bestScore = 0;
  let bestConfig = DEFAULT_CONFIG;

  for (const config of ALL_CONFIGS) {
    let score = 0;
    for (const kw of config.keywords) {
      if (lower.includes(kw)) {
        // Exact phrase match scores higher than substring
        score += kw.includes(' ') ? 3 : (words.has(kw) ? 2 : 1);
      }
    }
    if (score > bestScore) {
      bestScore = score;
      bestConfig = config;
    }
  }

  return bestConfig;
}
