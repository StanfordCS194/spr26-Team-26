import type { CapabilityProfile, ExperienceLevel } from '../types';

const PROFILES: Record<string, CapabilityProfile> = {
  Beginner: {
    comfort_level: 'BEGINNER',
    observability: {
      run_status: 'basic',
      metrics_visibility: 'summary',
      autoresearch_diary_access: 'none',
      cost_visibility: 'summary',
    },
    control: {
      can_edit_hyperparameters: false,
      hyperparameter_scope: 'none',
      can_edit_training_script: false,
      can_constrain_autoresearch_space: false,
      can_set_custom_stopping_criteria: false,
      strategy_hints_allowed: true,
    },
  },
  Intermediate: {
    comfort_level: 'INTERMEDIATE',
    observability: {
      run_status: 'detailed',
      metrics_visibility: 'summary',
      autoresearch_diary_access: 'summary',
      cost_visibility: 'summary',
    },
    control: {
      can_edit_hyperparameters: true,
      hyperparameter_scope: 'high_level',
      can_edit_training_script: false,
      can_constrain_autoresearch_space: true,
      can_set_custom_stopping_criteria: false,
      strategy_hints_allowed: true,
    },
  },
  Advanced: {
    comfort_level: 'ADVANCED',
    observability: {
      run_status: 'detailed',
      metrics_visibility: 'full',
      autoresearch_diary_access: 'full',
      cost_visibility: 'detailed',
    },
    control: {
      can_edit_hyperparameters: true,
      hyperparameter_scope: 'full',
      can_edit_training_script: true,
      can_constrain_autoresearch_space: true,
      can_set_custom_stopping_criteria: true,
      strategy_hints_allowed: true,
    },
  },
};

export function getCapabilities(level: ExperienceLevel): CapabilityProfile {
  return PROFILES[level];
}
