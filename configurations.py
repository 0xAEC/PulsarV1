# configurations.py

"""
Acts as a central repository for all default internal cognitive parameters.
This file isolates tuning parameters from core logic. It contains ONLY
configuration dictionaries and should have no class or function definitions.
"""

# ---------------------------------------------------------------------------
# New Perception, World Model, and Lifelong Learning Defaults (Directive Mu-Prime)
# ---------------------------------------------------------------------------
DEFAULT_VAE_CONFIG = {
    'IMG_SIZE': (64, 64),
    'LATENT_DIM': 32,
    'MODEL_PATH': 'visual_cortex.weights.h5',
}

DEFAULT_WORLD_MODEL_CONFIG = {
    # Use the filename that Keras 3+ prefers for whole-model saving.
    # The bootstrap script is updated to save in this format.
    'MODEL_PATH': 'world_model.weights.h5', # Keeping .h5 as per user's main.py logic
}

DEFAULT_LIFELONG_LEARNING_CONFIG = {
    'enabled': True,
    # The buffer stores experiences for retraining
    'replay_buffer_capacity': 5000,
    # These thresholds determine if an experience is "interesting" enough to store
    'experience_emotion_threshold': 0.7, # Store if abs(valence_mod) > this
    'experience_surprise_threshold': 2.5, # Store if prediction_error > this
    # Conditions to trigger a "sleep" (training) cycle
    'training_trigger_buffer_size': 256, # Train if the buffer has this many new items
    'training_trigger_idle_cycles': 50, # Train after this many cycles with no goal
    'cycles_since_idle_trigger': 0, # Internal counter
    'training_batch_size': 64,
    'training_epochs_per_cycle': 2, # How many epochs to train for during a sleep cycle
}

# ---------------------------------------------------------------------------
# Orch OR Emulator & System Defaults
# ---------------------------------------------------------------------------

DEFAULT_INTERNAL_PARAMS = {
    'curiosity': 0.5, 'goal_seeking_bias': 0.3,
    'preferred_logical_state': None,  # DEPRECATED, use preferred_state_handle. Retained for potential compatibility checks.
    'preferred_state_handle': None,   # Target state for problem-solving/goal-seeking, now a StateHandle
    'computation_length_preference': 3, # Avg number of ops agent tries to execute
    'attention_level': 1.0, 'cognitive_load': 0.0, 'mood': 0.0, 'frustration': 0.0,
    'exploration_mode_countdown': 0, # Cycles remaining in exploration mode
    'strategy_weights': {'memory': 0.25, 'problem_solve': 0.3, 'goal_seek': 0.25, 'curiosity': 0.2}, # How ops are chosen
    'sensor_input_noise_level': 0.01, # Chance for a bit in sensor input to flip
    'ltm_mutation_on_replay_rate': 0.02, # Chance to mutate LTM sequence on replay
    'ltm_mutation_on_store_rate': 0.01, # Chance to mutate LTM sequence metr`ics on store
    'temporal_feedback_valence_bias_strength': 0.1, # How much avg past valence delta affects strategy
    'temporal_feedback_entropy_bias_strength': 0.05, # How much avg past entropy shift affects strategy
    'smn_positive_valence_threshold': 0.6, # Valence above which SMN considers "good" for perturbation/stability
    'smn_negative_valence_threshold': -0.4, # Valence below which SMN considers "bad" for instability
    'smn_mutation_strength_decay': 0.99, # Factor to decay mutation strength on good valence
    'smn_mutation_strength_grow': 1.01, # Factor to grow mutation strength on bad valence
    'smn_perturbation_scale_factor': 0.05, # General scale of SMN param perturbations (multiplied by specific strength)
    'firewall_check_interval': 7, # How often to run firewall checks
    'firewall_cooldown_period': 15, # Cycles before firewall can trigger again
    'firewall_low_valence_threshold': -0.7, # Persistent low valence triggers firewall
    'firewall_low_valence_duration': 5,    # Num cycles of low valence to check
    'firewall_loop_detection_window': 8,   # Num recent states to check for loops
    'firewall_loop_min_repetitions': 3,    # How many times a state/op must repeat
    'firewall_premature_collapse_threshold_orp': 0.3, # ORP below which collapse is too early
    'firewall_premature_collapse_duration': 4, # Num cycles of premature collapse
    'interrupt_strong_consolidation_valence_threshold': 0.8,
    'interrupt_strong_consolidation_orp_surprise_factor': 1.5, # if orp > expected * factor
    'interrupt_reactive_ltm_valence_threshold': -0.6,
    'interrupt_cognitive_fork_valence_threshold': 0.85,
    'ltm_goal_context_match_bonus': 0.15, # Bonus to utility if LTM entry context matches active goal
    'ltm_initial_state_match_bonus': 0.10, # Bonus if LTM sequence starts from current agent state
    'ltm_input_context_match_bonus': 0.05, # Bonus if LTM sequence was learned for similar input context
    'concept_state_handle_map': {},  # E.g., {'HAPPY_PLACE': StateHandle('11'), 'TOOL_FOUND': StateHandle('01')}
    'ltm_active_concept_match_bonus': 0.12, # Bonus if LTM sequence active concepts match current concepts
    'clear_active_concepts_each_cycle': True, # If true, concepts are based purely on current collapsed state. If false, they persist until changed.
    'enable_counterfactual_simulation': True,
    'counterfactual_sim_reject_threshold': -0.1, # Reject plans if estimated valence is below this
    'enable_hierarchical_planning': True,
    'enable_analogical_planning': True,
    'analogical_planning_similarity_threshold': 0.75, # Min score to consider an LTM entry analogous
    # == NEW PARAMS FOR DIRECTIVE KAIZEN-PRIME (UNIFIED REASONING ENGINE) ==
    'planning_mental_trial_limit': 5,               # Max "thinking" attempts before acting randomly.
    'ltm_physical_plan_confidence_threshold': 0.3,  # Min confidence to recall a physical plan.
    'ltm_physical_plan_max_distance': 4.0,          # Max latent distance for a stored plan's start state.
    'ltm_physical_plan_distance_penalty_factor': 0.05,# Penalty per unit of latent distance.
        # == NEW PARAMS FOR SECE / PEML (Prediction Error Minimization Loop) ==
    'peml_active': True,                                 # Master switch for the intrinsic motivation system.
    'peml_uncertainty_threshold': 0.8,                   # How much variance in predictions triggers an INVESTIGATE goal.
    'peml_curiosity_requirement': 0.6,                   # Agent only investigates if its curiosity is already high.
}

DEFAULT_METACOGNITION_PARAMS = {
    'review_interval': 7, 'cycles_since_last_review': 0, # MODIFIED: Shorter review interval for demos
    'curiosity_adaptation_rate': 0.05, 'goal_bias_adaptation_rate': 0.05,
    'low_valence_threshold': -0.3, 'high_valence_threshold': 0.7, # For mood and param adaptation
    'exploration_threshold_entropy': 0.2, # Low diversity can trigger curiosity
    'frustration_threshold': 0.7, 'exploration_mode_duration': 5,
    'enable_threshold_adaptation': True, 'enable_decay_adaptation': True,
    'enable_compref_adaptation': True,
    # New: For Sophisticated Metacognition (Self-Modelling & Epistemic Uncertainty)
    'enable_self_model_adaptation': True,       # Master switch for using the self-model for adaptations
    'enable_epistemic_uncertainty_review': True,# Master switch for knowledge gap identification
    'epistemic_confidence_threshold': 0.35,     # LTM confidence below which is a "knowledge gap"
    'epistemic_curiosity_boost': 0.3,           # How much a knowledge gap boosts curiosity
    'self_model_stats': { # Data structure for storing the self-model
        'strategy_success_rates': {'memory':0.0, 'problem_solve':0.0, 'goal_seek':0.0, 'curiosity':0.0, 'default':0.0},
        'strategy_avg_valence': {'memory':0.0, 'problem_solve':0.0, 'goal_seek':0.0, 'curiosity':0.0, 'default':0.0},
        'strategy_total_uses': {'memory':0, 'problem_solve':0, 'goal_seek':0, 'curiosity':0, 'default':0},
        'strategy_success_count': {'memory':0, 'problem_solve':0, 'goal_seek':0, 'curiosity':0, 'default':0},
        'strategy_total_valence_accum': {'memory':0.0, 'problem_solve':0.0, 'goal_seek':0.0, 'curiosity':0.0, 'default':0.0},
        'total_reviews_for_model': 0
    }
}

DEFAULT_ORP_THRESHOLD_DYNAMICS = {
    'min': 0.4, 'max': 3.0, 'adapt_rate': 0.02, # adapt_rate is TRAINABLE
}

DEFAULT_ORP_DECAY_DYNAMICS = {
    'min': 0.0, 'max': 0.2, 'adapt_rate': 0.005, # adapt_rate is TRAINABLE
}

DEFAULT_TRAINABLE_PARAMS_CONFIG = {
    'sw_memory':        {'initial': 0.25, 'min': 0.01, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': 'internal_state_parameters', 'target_key': 'strategy_weights', 'target_subkey': 'memory'},
    'sw_problem_solve': {'initial': 0.30, 'min': 0.01, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': 'internal_state_parameters', 'target_key': 'strategy_weights', 'target_subkey': 'problem_solve'},
    'sw_goal_seek':     {'initial': 0.25, 'min': 0.01, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': 'internal_state_parameters', 'target_key': 'strategy_weights', 'target_subkey': 'goal_seek'},
    'sw_curiosity':     {'initial': 0.20, 'min': 0.01, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': 'internal_state_parameters', 'target_key': 'strategy_weights', 'target_subkey': 'curiosity'},
    'ltm_util_val':     {'initial': 0.6, 'min': 0.0, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': None, 'target_key': 'ltm_utility_weight_valence'},
    'ltm_util_eff':     {'initial': 0.4, 'min': 0.0, 'max': 1.0, 'perturb_scale': 0.05, 'target_dict_attr': None, 'target_key': 'ltm_utility_weight_efficiency'},
    'mc_cur_adapt_rate':{'initial': 0.05, 'min': 0.001, 'max': 0.2, 'perturb_scale': 0.005, 'target_dict_attr': 'metacognition_params', 'target_key': 'curiosity_adaptation_rate'},
    'mc_goal_adapt_rate':{'initial': 0.05, 'min': 0.001, 'max': 0.2, 'perturb_scale': 0.005, 'target_dict_attr': 'metacognition_params', 'target_key': 'goal_bias_adaptation_rate'},
    'orp_thresh_adapt_rate':{'initial': 0.02, 'min': 0.001, 'max': 0.1, 'perturb_scale': 0.002, 'target_dict_attr': 'orp_threshold_dynamics', 'target_key': 'adapt_rate'},
    'orp_decay_adapt_rate': {'initial': 0.005, 'min':0.0001,'max': 0.02, 'perturb_scale': 0.0005, 'target_dict_attr': 'orp_decay_dynamics', 'target_key': 'adapt_rate'},
}

DEFAULT_TEMPORAL_GRID_PARAMS = {
    'max_len': 10, # Stores past 10 cycles for feedback
    'low_valence_delta_threshold': -0.15, # Threshold for negative valence change to react to
    'high_entropy_shift_threshold': 0.25, # Threshold for high entropy increase to react to
    # How many recent cycles from the grid to consider for averaging (could be less than max_len)
    'feedback_window': 5, # Check up to the last 5 entries in the grid
}

DEFAULT_SMN_CONFIG = {
    'enabled': True,
    'mutation_trigger_min_valence_gain': 0.1, # Min positive valence change from previous cycle to trigger 'reinforcing' mutations
    
    # New: For SMN Graph-based Structural Plasticity
    'enable_influence_matrix': True, # Master switch for the graph part of SMN
    'smn_influence_matrix_initial_stddev': 0.05, # Initial randomness of influence weights
    'smn_influence_matrix_hebbian_learning_rate': 0.015, # Learning rate for good outcomes
    'smn_influence_matrix_weight_decay': 0.001, # General decay to prevent runaway weights
    'smn_influence_matrix_clip_min': -0.8, # Min influence weight
    'smn_influence_matrix_clip_max': 0.8,  # Max influence weight
    'smn_influence_propagation_threshold': 0.15, # Min absolute influence weight to trigger propagated mutation
    'smn_secondary_mutation_scale_factor': 0.4, # How much a primary mutation (scaled) on P_i affects P_j
    'smn_hebbian_orp_threshold_factor': 0.15, # e.g., 0.15 * E_OR_THRESHOLD as min ORP_at_collapse for Hebbian updates to occur
}

DEFAULT_SMN_CONTROLLED_PARAMS = {
    'sw_curiosity':       {'base_mutation_strength': 0.05, 'min_val':0.01, 'max_val':0.99, 'path': ('internal_state_parameters', 'strategy_weights', 'curiosity')},
    'mc_cur_adapt_rate':  {'base_mutation_strength': 0.005,'min_val':0.001,'max_val':0.2,  'path': ('metacognition_params', 'curiosity_adaptation_rate')},
    'computation_length_preference': {'base_mutation_strength': 0.2, 'min_val': 1, 'max_val': 8, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')}
}

DEFAULT_INTERRUPT_HANDLER_CONFIG = {
    'enabled': True,
    'consolidation_valence_abs_threshold': 0.7,
    'consolidation_orp_surprise_factor': 1.5,
    'consolidation_strength_bonus': 2.0,
    'reactive_ltm_valence_threshold': -0.5,
    'cognitive_fork_valence_threshold': 0.75,
    'cognitive_fork_goal_bias_boost': 0.2,
}

DEFAULT_COGNITIVE_FIREWALL_CONFIG = {
    'enabled': True,
    'check_interval': 5,       # How often to run firewall checks (remains 5)
    'cooldown_duration': 10,   # Cycles before firewall can trigger again (remains 10)
    'low_valence_threshold': -0.65, # MODIFIED as per suggestion for Demo 2
    'low_valence_streak_needed': 3, # MODIFIED - slightly more sensitive for demos
    'loop_detection_window': 7,   # Num recent states to check for loops
    'loop_detection_min_repeats': 3,    # How many times a state/op must repeat
    'premature_collapse_orp_max_ratio': 0.4, # ORP below which collapse is too early
    'premature_collapse_streak_needed': 3, # MODIFIED - slightly more sensitive for demos
    'intervention_exploration_boost_duration': 5,
    'intervention_orp_threshold_increase_factor': 1.2,
    'intervention_strategy_randomness_factor': 0.5,
    'clear_wm_on_intervention': True,
}

DEFAULT_GOAL_STATE_PARAMS = {
    'completion_valence_bonus': 0.5,
    'failure_valence_penalty': -0.5,
    'step_completion_valence_bonus': 0.1,
}

DEFAULT_LOT_CONFIG = {
    'enabled': True,
    'log_level_details': {
        'cycle_start': True, 'sensor_input': True, 'op_generation': False, 'op_execution': False,
        'collapse_event': True, 'valence_eval': True, 'ltm_update': False, 'internal_state_updates': False,
        'goal_tracking': True, 'firewall_action': True, 'smn_action': True, 'interrupt_action': True,
        'metacognitive_review': True, 'cycle_end': True,
        # New for SMN graph logging
        'smn_graph_propagation': False, 'smn_graph_hebbian': False,
        # New for Working Memory logging
        'workingmemory_ops': False, # General toggle for all WM operations below
        'workingmemory.push_goal_context': False, 
        'workingmemory.pop_goal_context': False,
        'workingmemory.push_intermediate': False, 
        'workingmemory.pop_intermediate': False,
        'workingmemory.peek': False, 
        'workingmemory.clear': False, 
        'workingmemory.full_discard': False, # Log if oldest item discarded due to max_len
    }
}