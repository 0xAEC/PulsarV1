import numpy as np
import copy
import time
import random
import collections # For deque
import traceback 
import math 

# ---------------------------------------------------------------------------
# Orch OR Emulator & System Defaults
# ---------------------------------------------------------------------------

# Default parameters for a single emulator's internal cognitive state
DEFAULT_INTERNAL_PARAMS = {
    'curiosity': 0.5, 'goal_seeking_bias': 0.3,
    'preferred_logical_state': None,  # Target state for problem-solving/goal-seeking
    'computation_length_preference': 3, # Avg number of ops agent tries to execute
    'attention_level': 1.0, 'cognitive_load': 0.0, 'mood': 0.0, 'frustration': 0.0,
    'exploration_mode_countdown': 0, # Cycles remaining in exploration mode
    'strategy_weights': {'memory': 0.25, 'problem_solve': 0.3, 'goal_seek': 0.25, 'curiosity': 0.2}, # How ops are chosen
    'sensor_input_noise_level': 0.01, # Chance for a bit in sensor input to flip
    'ltm_mutation_on_replay_rate': 0.02, # Chance to mutate LTM sequence on replay
    'ltm_mutation_on_store_rate': 0.01, # Chance to mutate LTM sequence metrics on store
    # New: for temporal feedback grid bias strength (Feature 2)
    'temporal_feedback_valence_bias_strength': 0.1, # How much avg past valence delta affects strategy
    'temporal_feedback_entropy_bias_strength': 0.05, # How much avg past entropy shift affects strategy
    # New: for SMN, parameters defining behavior of self-mutation
    'smn_positive_valence_threshold': 0.6, # Valence above which SMN considers "good" for perturbation/stability
    'smn_negative_valence_threshold': -0.4, # Valence below which SMN considers "bad" for instability
    'smn_mutation_strength_decay': 0.99, # Factor to decay mutation strength on good valence
    'smn_mutation_strength_grow': 1.01, # Factor to grow mutation strength on bad valence
    'smn_perturbation_scale_factor': 0.05, # General scale of SMN param perturbations (multiplied by specific strength)
    # New: For Cognitive Firewall
    'firewall_check_interval': 7, # How often to run firewall checks
    'firewall_cooldown_period': 15, # Cycles before firewall can trigger again
    'firewall_low_valence_threshold': -0.7, # Persistent low valence triggers firewall
    'firewall_low_valence_duration': 5,    # Num cycles of low valence to check
    'firewall_loop_detection_window': 8,   # Num recent states to check for loops
    'firewall_loop_min_repetitions': 3,    # How many times a state/op must repeat
    'firewall_premature_collapse_threshold_orp': 0.3, # ORP below which collapse is too early
    'firewall_premature_collapse_duration': 4, # Num cycles of premature collapse
    # New: For Interrupt Handlers
    'interrupt_strong_consolidation_valence_threshold': 0.8,
    'interrupt_strong_consolidation_orp_surprise_factor': 1.5, # if orp > expected * factor
    'interrupt_reactive_ltm_valence_threshold': -0.6,
    'interrupt_cognitive_fork_valence_threshold': 0.85,
}

# Default parameters for metacognitive processes
DEFAULT_METACOGNITION_PARAMS = {
    'review_interval': 10, 'cycles_since_last_review': 0,
    'curiosity_adaptation_rate': 0.05, 'goal_bias_adaptation_rate': 0.05,
    'low_valence_threshold': -0.3, 'high_valence_threshold': 0.7, # For mood and param adaptation
    'exploration_threshold_entropy': 0.2, # Low diversity can trigger curiosity
    'frustration_threshold': 0.7, 'exploration_mode_duration': 5,
    'enable_threshold_adaptation': True, 'enable_decay_adaptation': True,
    'enable_compref_adaptation': True,
}

# Default parameters for ORP threshold dynamics
DEFAULT_ORP_THRESHOLD_DYNAMICS = {
    'min': 0.4, 'max': 3.0, 'adapt_rate': 0.02, # adapt_rate is TRAINABLE
}

# Default parameters for ORP decay dynamics
DEFAULT_ORP_DECAY_DYNAMICS = {
    'min': 0.0, 'max': 0.2, 'adapt_rate': 0.005, # adapt_rate is TRAINABLE
}

# Configuration for trainable parameters (used by CognitiveAgentTrainer)
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
    # Potentially make TFG params trainable later if needed, e.g.
    # 'tfg_val_bias_str': {'initial':0.1, 'min':0.0, 'max':0.5, 'perturb_scale':0.02, 'target_dict_attr':'internal_state_parameters', 'target_key':'temporal_feedback_valence_bias_strength'}
}

# Default parameters for Temporal Feedback Grid (Feature 2 - NEWLY ADDED FEATURE)
DEFAULT_TEMPORAL_GRID_PARAMS = {
    'max_len': 10, # Stores past 10 cycles for feedback
    'low_valence_delta_threshold': -0.15, # Threshold for negative valence change to react to
    'high_entropy_shift_threshold': 0.25, # Threshold for high entropy increase to react to
    # How many recent cycles from the grid to consider for averaging (could be less than max_len)
    'feedback_window': 5, # Check up to the last 5 entries in the grid
}

# 🧬 Default parameters for Synaptic Mutation Network (SMN) (Feature 3 - Enhanced)
DEFAULT_SMN_CONFIG = { # General SMN behavior settings
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
# Defines which parameters are under SMN control and their specific mutation behavior
DEFAULT_SMN_CONTROLLED_PARAMS = { # These become the NODES of the SMN graph
    'sw_curiosity':       {'base_mutation_strength': 0.05, 'min_val':0.01, 'max_val':0.99, 'path': ('internal_state_parameters', 'strategy_weights', 'curiosity')},
    'mc_cur_adapt_rate':  {'base_mutation_strength': 0.005,'min_val':0.001,'max_val':0.2,  'path': ('metacognition_params', 'curiosity_adaptation_rate')},
    'computation_length_preference': {'base_mutation_strength': 0.2, 'min_val': 1, 'max_val': 8, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')}
    # Add more parameters here to expand the SMN's control and graph size.
    # E.g. 'smn_perturbation_scale_factor_itself': {'base_mutation_strength': 0.01, 'min_val':0.01, 'max_val':0.2, 'path':('internal_state_parameters', 'smn_perturbation_scale_factor')}
}


# Default parameters for Collapse-Triggered Interrupt Handlers (Feature 4)
DEFAULT_INTERRUPT_HANDLER_CONFIG = {
    'enabled': True,
    'consolidation_valence_abs_threshold': 0.7,
    'consolidation_orp_surprise_factor': 1.5,
    'consolidation_strength_bonus': 2.0,
    'reactive_ltm_valence_threshold': -0.5,
    'cognitive_fork_valence_threshold': 0.75,
    'cognitive_fork_goal_bias_boost': 0.2,
}

# Default parameters for Cognitive Firewall (Feature 6)
DEFAULT_COGNITIVE_FIREWALL_CONFIG = {
    'enabled': True,
    'check_interval': 5,
    'cooldown_duration': 10,
    'low_valence_threshold': -0.6,
    'low_valence_streak_needed': 4,
    'loop_detection_window': 7,
    'loop_detection_min_repeats': 3,
    'premature_collapse_orp_max_ratio': 0.4,
    'premature_collapse_streak_needed': 4,
    'intervention_exploration_boost_duration': 5,
    'intervention_orp_threshold_increase_factor': 1.2,
    'intervention_strategy_randomness_factor': 0.5
}

# Default parameters for GoalState (Feature 7)
DEFAULT_GOAL_STATE_PARAMS = {
    'completion_valence_bonus': 0.5,
    'failure_valence_penalty': -0.5,
    'step_completion_valence_bonus': 0.1,
}

# Internal Language of Thought Configuration (Feature 8)
DEFAULT_LOT_CONFIG = {
    'enabled': True,
    'log_level_details': {
        'cycle_start': True, 'sensor_input': True, 'op_generation': False, 'op_execution': False,
        'collapse_event': True, 'valence_eval': True, 'ltm_update': False, 'internal_state_updates': False,
        'goal_tracking': True, 'firewall_action': True, 'smn_action': True, 'interrupt_action': True,
        'metacognitive_review': True, 'cycle_end': True,
        # New for SMN graph logging
        'smn_graph_propagation': False, 'smn_graph_hebbian': False,
    }
}
# ---------------------------------------------------------------------------
# GoalState Structure (for Feature 7)
# ---------------------------------------------------------------------------
class GoalState:
    def __init__(self, current_goal, steps, error_tolerance=0.05, initial_progress=0.0):
        self.current_goal = current_goal
        self.steps = steps
        self.progress = initial_progress
        self.error_tolerance = error_tolerance
        self.current_step_index = 0
        self.status = "pending" # pending, active, completed, failed
        self.history = []

    def to_dict(self):
        # Basic serialization, callable criteria might need special handling if persisted
        serializable_steps = []
        for step in self.steps:
            s_copy = step.copy()
            if callable(s_copy.get("completion_criteria")):
                s_copy["completion_criteria"] = "callable_function_not_serialized"
            serializable_steps.append(s_copy)

        return {
            "current_goal": self.current_goal,
            "steps": serializable_steps,
            "progress": self.progress,
            "error_tolerance": self.error_tolerance,
            "current_step_index": self.current_step_index,
            "status": self.status,
            "history": self.history,
        }

    def __str__(self):
        step_name = "None"
        if 0 <= self.current_step_index < len(self.steps):
            step_name = self.steps[self.current_step_index].get("name", f"Step {self.current_step_index+1}")
        return f"Goal: '{self.current_goal}' (Step: '{step_name}', Progress: {self.progress*100:.1f}%, Status: {self.status})"


# ---------------------------------------------------------------------------
# Class Definition: SimplifiedOrchOREmulator
# ---------------------------------------------------------------------------
class SimplifiedOrchOREmulator:
    def __init__(self, agent_id="agent0", cycle_history_max_len=100,
                 initial_E_OR_THRESHOLD=1.0, initial_orp_decay_rate=0.01,
                 initial_internal_states=None, metacognition_config=None,
                 orp_threshold_dynamics_config=None, orp_decay_dynamics_config=None,
                 trainable_param_values=None,
                 temporal_grid_config=None,
                 smn_general_config=None, # Overall SMN behavior
                 smn_controlled_params_config=None, # Params SMN specifically manages
                 interrupt_handler_config=None,
                 cognitive_firewall_config=None,
                 goal_state_params = None,
                 lot_config=None,
                 shared_long_term_memory=None,
                 shared_attention_foci=None,
                 config_overrides=None,
                 verbose=0):

        self.agent_id = agent_id
        self.verbose = verbose # MOVED HERE - Fixed the AttributeError

        self.logical_superposition = {"00": 1.0 + 0j, "01":0j, "10":0j, "11":0j}
        self.collapsed_logical_state_str = "00"
        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate

        self.operation_costs = {'X': 0.1, 'Z': 0.1, 'H': 0.3, 'CNOT': 0.4, 'CZ': 0.4, 'ERROR_PENALTY': 0.05, 'PLANNING_BASE': 0.02}
        self.outcome_valence_map = {"00": 0.0, "01": 0.5, "10": -0.5, "11": 1.0} # Default, can be overridden by config_overrides
        self.last_cycle_valence_raw = 0.0
        self.last_cycle_valence_mod = 0.0
        self.current_orp_before_reset = 0.0

        # Initialize parameter dictionaries with defaults, then update with provided configs
        self.internal_state_parameters = copy.deepcopy(DEFAULT_INTERNAL_PARAMS)
        if initial_internal_states: self.internal_state_parameters.update(initial_internal_states)

        self.metacognition_params = copy.deepcopy(DEFAULT_METACOGNITION_PARAMS)
        if metacognition_config: self.metacognition_params.update(metacognition_config)

        self.orp_threshold_dynamics = copy.deepcopy(DEFAULT_ORP_THRESHOLD_DYNAMICS)
        if orp_threshold_dynamics_config: self.orp_threshold_dynamics.update(orp_threshold_dynamics_config)

        self.orp_decay_dynamics = copy.deepcopy(DEFAULT_ORP_DECAY_DYNAMICS)
        if orp_decay_dynamics_config: self.orp_decay_dynamics.update(orp_decay_dynamics_config)

        self.ltm_utility_weight_valence = 0.6
        self.ltm_utility_weight_efficiency = 0.4

        # Feature 2: Temporal Feedback Grid
        self.temporal_grid_params = copy.deepcopy(DEFAULT_TEMPORAL_GRID_PARAMS)
        if temporal_grid_config: self.temporal_grid_params.update(temporal_grid_config)
        self.temporal_feedback_grid = collections.deque(maxlen=self.temporal_grid_params['max_len'])
        self.last_cycle_entropy_for_delta = 0.0

        # 🧬 Feature 3: Synaptic Mutation Network (SMN) - Enhanced Graph Version
        self.smn_config = copy.deepcopy(DEFAULT_SMN_CONFIG) # General behavior
        if smn_general_config: self.smn_config.update(smn_general_config)

        self.smn_controlled_params_definitions = copy.deepcopy(DEFAULT_SMN_CONTROLLED_PARAMS) # Definitions of what SMN controls
        if smn_controlled_params_config: self.smn_controlled_params_definitions.update(smn_controlled_params_config)

        # These will be initialized by _initialize_smn_graph_structures
        self.smn_params_runtime_state = {} # Stores current_mutation_strength, min/max, path for each param node
        self.smn_param_indices = {}       # Maps param_smn_key to matrix index
        self.smn_param_names_from_indices = {} # Maps matrix index to param_smn_key
        self.smn_influence_matrix = np.array([]) # Adjacency matrix for influences
        self.smn_param_actual_changes_this_cycle = {} # Stores delta_values for Hebbian updates
        self._initialize_smn_graph_structures() # Sets up the above graph-related structures

        self.smn_internal_flags = {} # For other SMN related flags if needed outside graph

        # Feature 4: Collapse-Triggered Interrupt Handlers
        self.interrupt_handler_params = copy.deepcopy(DEFAULT_INTERRUPT_HANDLER_CONFIG)
        if interrupt_handler_config: self.interrupt_handler_params.update(interrupt_handler_config)

        # Feature 5: Co-Agents support
        self.long_term_memory = shared_long_term_memory if shared_long_term_memory is not None else {}
        self.shared_attention_foci = shared_attention_foci if shared_attention_foci is not None else collections.deque(maxlen=20)

        # Feature 6: Cognitive Firewall
        self.firewall_params = copy.deepcopy(DEFAULT_COGNITIVE_FIREWALL_CONFIG)
        if cognitive_firewall_config: self.firewall_params.update(cognitive_firewall_config)
        self.firewall_cooldown_remaining = 0
        self.firewall_cycles_since_last_check = 0

        # Feature 7: Goal-Oriented State Machines
        self.goal_state_config_params = copy.deepcopy(DEFAULT_GOAL_STATE_PARAMS)
        if goal_state_params: self.goal_state_config_params.update(goal_state_params)
        self.current_goal_state_obj = None

        # Feature 8: Internal Language Layer (LoT)
        self.lot_config_params = copy.deepcopy(DEFAULT_LOT_CONFIG)
        if lot_config: self.lot_config_params.update(lot_config)
        self.current_cycle_lot_stream = []

        if config_overrides:
            self._apply_config_overrides(config_overrides)

        if trainable_param_values:
            self.update_emulator_parameters(trainable_param_values)

        self.long_term_memory_capacity = 100
        self.successful_sequence_threshold_valence = 0.5

        self.cycle_history = collections.deque(maxlen=cycle_history_max_len)
        self.current_cycle_num = 0
        # self.verbose = verbose # Original position - MOVED EARLIER
        self.next_target_input_state = "00"

        if self.verbose >= 1:
            print(f"[{self.agent_id}] Orch-OR Emulator Initialized. Active Features: TemporalGrid, SMN (Graph Enabled: {self.smn_config.get('enable_influence_matrix', False)}), Interrupts, Firewall, Goals, LoT.")
            print(f"[{self.agent_id}] E_OR_THRESHOLD: {self.E_OR_THRESHOLD:.2f}, ORP Decay Rate: {self.orp_decay_rate:.3f}")
            if self.temporal_grid_params.get('max_len',0) > 0:
                print(f"[{self.agent_id}] Temporal Feedback Grid: Active (maxlen={self.temporal_grid_params['max_len']}, window={self.temporal_grid_params['feedback_window']})")
            if self.smn_config.get('enabled', False) and self.smn_config.get('enable_influence_matrix', False):
                 print(f"[{self.agent_id}] SMN Influence Matrix: Active ({len(self.smn_param_indices)} params, matrix_shape {self.smn_influence_matrix.shape})")


    def _apply_config_overrides(self, overrides):
        """Applies direct value overrides to emulator parameters, useful for co-agent setup."""
        if self.verbose >= 2: print(f"[{self.agent_id}] Applying config overrides: {overrides}")
        for path, value in overrides.items():
            try:
                current_obj = self
                is_direct_attr = True

                if isinstance(path, tuple) and len(path) > 1:
                    first_part_obj_candidate = getattr(current_obj, path[0], None)
                    if isinstance(first_part_obj_candidate, dict):
                        is_direct_attr = False
                        target_dict = first_part_obj_candidate
                        for i, key_segment in enumerate(path[1:-1]):
                            if key_segment not in target_dict or not isinstance(target_dict[key_segment], dict):
                                target_dict[key_segment] = {}
                            target_dict = target_dict[key_segment]
                        target_dict[path[-1]] = value
                        if self.verbose >= 3: print(f"  Override: self.{'.'.join(str(p) for p in path)} (dict) = {value}")

                if is_direct_attr:
                    obj_nav = self
                    for i, key in enumerate(path[:-1]):
                        obj_nav = getattr(obj_nav, key)
                    setattr(obj_nav, path[-1], value)
                    if self.verbose >= 3: print(f"  Override: self.{'.'.join(str(p) for p in path)} (attr) = {value}")

            except (AttributeError, KeyError, TypeError) as e:
                if self.verbose >= 1: print(f"  Warning: Override for path {path} with value {value} failed: {e}")


    def update_emulator_parameters(self, param_values_dict):
        if self.verbose >= 2: print(f"[{self.agent_id}] Updating emulator with trainable params: {param_values_dict}")
        normalized_strategy_weights = False
        for param_name, value in param_values_dict.items():
            if param_name not in DEFAULT_TRAINABLE_PARAMS_CONFIG:
                if self.verbose >= 1: print(f"  Warning: Param '{param_name}' not in DEFAULT_TRAINABLE_PARAMS_CONFIG. Skipping.")
                continue
            config = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name]
            dict_attr_name = config['target_dict_attr']
            key = config['target_key']
            subkey = config.get('target_subkey')

            try:
                if dict_attr_name is None:
                    setattr(self, key, value)
                    if self.verbose >= 3: print(f"      Set direct attr emulator.{key} = {value:.4f}")
                else:
                    target_dict_obj = getattr(self, dict_attr_name, None)
                    if target_dict_obj is None:
                        if self.verbose >=1: print(f"Warning: Target dict attribute object '{dict_attr_name}' not found for '{param_name}'.")
                        continue

                    if subkey:
                        if key not in target_dict_obj or not isinstance(target_dict_obj[key], dict):
                            target_dict_obj[key] = {}
                        target_dict_obj[key][subkey] = value
                        if key == 'strategy_weights': normalized_strategy_weights = True
                        if self.verbose >= 3: print(f"      Set emulator.{dict_attr_name}['{key}']['{subkey}'] = {value:.4f}")
                    else:
                        target_dict_obj[key] = value
                        if self.verbose >= 3: print(f"      Set emulator.{dict_attr_name}['{key}'] = {value:.4f}")
            except Exception as e:
                if self.verbose >=1: print(f"Error setting param {param_name}: {e}")

        if normalized_strategy_weights:
            sw = self.internal_state_parameters['strategy_weights']
            total_sw = sum(w for w in sw.values() if isinstance(w, (int,float)) and w > 0) # Sum positive numeric weights
            if total_sw > 1e-6 :
                for k_sw in sw:
                    if isinstance(sw[k_sw], (int,float)) : sw[k_sw] = max(0, sw[k_sw]/total_sw)
            else:
                if self.verbose >=1: print(f"Warning: All strategy weights became zero/negative. Resetting to uniform.")
                num_strats = len([k for k in sw if isinstance(sw[k], (int,float))]) if sw else 1
                uniform_weight = 1.0 / num_strats if num_strats > 0 else 1.0
                for k_sw in sw :
                    if isinstance(sw[k_sw], (int,float)) : sw[k_sw] = uniform_weight
                if not sw: self.internal_state_parameters['strategy_weights'] = {'curiosity': 1.0}


    # --- Feature 8: Internal Language Layer ---
    def _log_lot_event(self, event_type: str, details: dict):
        if not self.lot_config_params.get('enabled', False): return

        log_details_config = self.lot_config_params.get('log_level_details', {})
        event_category = event_type.split('.')[0]
        is_globally_enabled = log_details_config.get(event_category + ".*", False)
        if not is_globally_enabled and \
           not log_details_config.get(event_type, False) and \
           not log_details_config.get(event_category, False):
            return

        param_strs = []
        for k, v in details.items():
            if isinstance(v, float):
                param_strs.append(f"{k}:{v:.3f}")
            elif isinstance(v, (list, tuple)) and len(v) > 4:
                 param_strs.append(f"{k}:[...{len(v)}items...]")
            elif isinstance(v, dict) and len(v) > 2:
                 param_strs.append(f"{k}:{{...{len(v)}keys...}}")
            elif isinstance(v, np.ndarray):
                 param_strs.append(f"{k}:ndarray_shape{v.shape}")
            else:
                v_str = str(v)
                if len(v_str) > 35 : v_str = v_str[:32] + "..."
                param_strs.append(f"{k}:{v_str}")

        tag_name = event_type.upper().replace(".", "_").replace("_", "")
        self.current_cycle_lot_stream.append(f"#{tag_name}[{','.join(param_strs)}]")

    # --- Core Orch OR Mechanics (centralized, used primarily by Executive Layer) ---
    def _apply_logical_op_to_superposition(self, op_char, logical_arg, current_superposition, current_orp):
        new_superposition = collections.defaultdict(complex)
        new_orp = current_orp
        sqrt2_inv = 1 / np.sqrt(2)
        op_char_upper = op_char.upper()

        op_cost_val = self.operation_costs.get(op_char_upper, 0.05)
        if op_char_upper not in self.operation_costs:
            if self.verbose >= 1: print(f"Warning: Op '{op_char_upper}' not in operation_costs. Using default cost {op_cost_val}.")
        new_orp += op_cost_val

        self._log_lot_event("op_execution.attempt", {"op":op_char, "arg":logical_arg, "cost":op_cost_val, "cur_orp":current_orp})

        error_occurred = False
        for basis_state_str, amp in current_superposition.items():
            if abs(amp) < 1e-9: continue
            lq1_val, lq0_val = int(basis_state_str[0]), int(basis_state_str[1])

            new_basis_state_str = basis_state_str

            if op_char_upper == 'X':
                idx_to_flip = logical_arg
                if idx_to_flip == 0: new_basis_state_str = f"{lq1_val}{1-lq0_val}"
                elif idx_to_flip == 1: new_basis_state_str = f"{1-lq1_val}{lq0_val}"
                else: error_occurred = True
                new_superposition[new_basis_state_str] += amp
            elif op_char_upper == 'Z':
                idx_to_phase = logical_arg; phase_factor = 1
                if idx_to_phase == 0 and lq0_val == 1: phase_factor = -1
                elif idx_to_phase == 1 and lq1_val == 1: phase_factor = -1
                elif idx_to_phase not in [0,1]: error_occurred = True
                new_superposition[basis_state_str] += amp * phase_factor
            elif op_char_upper == 'H':
                idx_to_h = logical_arg
                if idx_to_h == 0:
                    s0_str, s1_str = f"{lq1_val}{0}", f"{lq1_val}{1}"
                    new_superposition[s0_str] += amp * sqrt2_inv * (+1 if lq0_val == 0 else +1)
                    new_superposition[s1_str] += amp * sqrt2_inv * (+1 if lq0_val == 0 else -1)
                elif idx_to_h == 1:
                    s0_str, s1_str = f"{0}{lq0_val}", f"{1}{lq0_val}"
                    new_superposition[s0_str] += amp * sqrt2_inv * (+1 if lq1_val == 0 else +1)
                    new_superposition[s1_str] += amp * sqrt2_inv * (+1 if lq1_val == 0 else -1)
                else: error_occurred = True; new_superposition[basis_state_str] += amp
            elif op_char_upper == 'CNOT':
                ctrl_idx, target_idx = logical_arg if isinstance(logical_arg, tuple) and len(logical_arg)==2 else (-1,-1)
                if ctrl_idx not in [0,1] or target_idx not in [0,1] or ctrl_idx == target_idx:
                    error_occurred = True; new_superposition[basis_state_str] += amp
                else:
                    control_is_active = (lq1_val == 1 if ctrl_idx == 1 else lq0_val == 1)
                    if control_is_active:
                        if target_idx == 0: new_basis_state_str = f"{lq1_val}{1-lq0_val}"
                        else: new_basis_state_str = f"{1-lq1_val}{lq0_val}"
                        new_superposition[new_basis_state_str] += amp
                    else:
                        new_superposition[basis_state_str] += amp
            elif op_char_upper == 'CZ':
                phase_factor = 1
                if lq0_val == 1 and lq1_val == 1: phase_factor = -1
                new_superposition[basis_state_str] += amp * phase_factor
            else:
                error_occurred = True; new_superposition[basis_state_str] += amp

            if error_occurred:
                 new_orp += self.operation_costs.get('ERROR_PENALTY',0.05)*0.2
                 self._log_lot_event("op_execution.logic_error", {"op":op_char, "arg":logical_arg, "state":basis_state_str})
                 error_occurred = False

        final_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        norm_sq = sum(abs(a)**2 for a in new_superposition.values())
        if norm_sq > 1e-12:
            norm = np.sqrt(norm_sq)
            for state_key, amp_val in new_superposition.items():
                final_superposition[state_key] = amp_val / norm
        else:
            if self.verbose >=1: print(f"[{self.agent_id}] CRITICAL Warning: Superposition norm zero after op '{op_char_upper}'. Resetting to |00>.")
            self._log_lot_event("op_execution.error", {"op":op_char, "error":"norm_zero_critical_reset_00"})
            final_superposition["00"] = 1.0 + 0j
            new_orp += 0.2
        return dict(final_superposition), new_orp


    def _calculate_superposition_entropy(self, superposition_dict=None):
        target_superposition = superposition_dict if superposition_dict is not None else self.logical_superposition
        probabilities = np.array([np.abs(amp)**2 for amp in target_superposition.values()])
        probabilities = probabilities[probabilities > 1e-9]
        if not probabilities.any(): return 0.0
        current_sum_probs = np.sum(probabilities)
        if not np.isclose(current_sum_probs, 1.0) and current_sum_probs > 1e-9:
            probabilities /= current_sum_probs
        return -np.sum(probabilities * np.log2(probabilities + 1e-12))

    def _executive_prepare_superposition(self, classical_input_str="00"):
        if self.verbose >= 2: print(f"  EXECUTIVE.Super_Prep: Target initial state |{classical_input_str}>")
        self._log_lot_event("executive.super_prep", {"target_state": classical_input_str})

        self.logical_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        if not (len(classical_input_str) == 2 and all(c in '01' for c in classical_input_str)):
            if self.verbose >= 1: print(f"    ERROR: Invalid classical_input_str '{classical_input_str}'. Defaulting to '00'.")
            self._log_lot_event("executive.super_prep.error", {"input": classical_input_str, "defaulted_to": "00"})
            classical_input_str = "00"
        self.logical_superposition[classical_input_str] = 1.0 + 0j
        self.objective_reduction_potential = 0.0

        if self.verbose >= 3:
            print(f"    Superposition prepared: {self.logical_superposition_str()}")
            print(f"    ORP after prep: {self.objective_reduction_potential:.3f}")
        self.objective_reduction_potential += 0.05
        return True

    def _executive_quantum_computation_phase(self, computation_sequence_ops):
        if self.verbose >= 2: print(f"  EXECUTIVE.Quantum_Comp: Evolving superposition.")
        self._log_lot_event("executive.quantum_comp.start", {"ops_planned_count": len(computation_sequence_ops or []), "orp_start":self.objective_reduction_potential, "decay_rate": self.orp_decay_rate})

        orp_before_decay = self.objective_reduction_potential
        decay_amount = self.objective_reduction_potential * self.orp_decay_rate
        self.objective_reduction_potential = max(0, self.objective_reduction_potential - decay_amount)
        if self.verbose >=3 and decay_amount > 1e-6:
            self._log_lot_event("executive.quantum_comp.orp_decay", {"before": orp_before_decay, "after": self.objective_reduction_potential, "amount": decay_amount})
            print(f"    ORP decay ({self.orp_decay_rate*100:.1f}%) applied: {orp_before_decay:.3f} -> {self.objective_reduction_potential:.3f}")

        or_triggered_early = False
        if not computation_sequence_ops:
            if self.verbose >= 3: print(f"    No computation operations this cycle.")
        else:
            if self.verbose >= 3: print(f"    Applying computation sequence: {computation_sequence_ops}")
            temp_superposition = copy.deepcopy(self.logical_superposition)
            temp_orp = self.objective_reduction_potential
            ops_executed_count = 0
            for i, (op_char, logical_arg) in enumerate(computation_sequence_ops):
                op_start_orp = temp_orp
                try:
                    temp_superposition, temp_orp = \
                        self._apply_logical_op_to_superposition(op_char, logical_arg, temp_superposition, temp_orp)
                    ops_executed_count += 1
                    self._log_lot_event("op_execution.success", {"op_idx":i, "op":op_char, "arg":logical_arg, "orp_change": temp_orp-op_start_orp})
                except ValueError as e:
                    if self.verbose >=1: print(f"    Error applying op ('{op_char}', {logical_arg}): {e}. Skipping.")
                    self._log_lot_event("executive.quantum_comp.op_error", {"op_idx":i, "op":op_char, "arg":logical_arg, "error":str(e)})
                    temp_orp += self.operation_costs.get('ERROR_PENALTY', 0.05)

                if self.verbose >= 3:
                    active_terms_str = ', '.join([f'{amp:.2f}|{s}>' for s, amp in temp_superposition.items() if abs(amp) > 1e-9])
                    print(f"      After op {i+1} ('{op_char}', {logical_arg}): [{active_terms_str}], ORP: {temp_orp:.3f}")

                if temp_orp >= self.E_OR_THRESHOLD:
                    if self.verbose >= 2:
                        print(f"      >>> OR THRESHOLD REACHED ({temp_orp:.3f} >= {self.E_OR_THRESHOLD:.3f}) after {ops_executed_count} ops. <<<")
                    self._log_lot_event("executive.quantum_comp.or_early", {"orp":temp_orp, "threshold":self.E_OR_THRESHOLD, "ops_done":ops_executed_count})
                    or_triggered_early = True
                    break
            self.logical_superposition = temp_superposition
            self.objective_reduction_potential = temp_orp

        if self.verbose >= 2:
            print(f"    Final superposition before OR: {self.logical_superposition_str()}, ORP: {self.objective_reduction_potential:.3f}")
        self._log_lot_event("executive.quantum_comp.end", {"ops_executed_count": len(computation_sequence_ops or []), "final_orp":self.objective_reduction_potential, "early_or":or_triggered_early})
        return True, or_triggered_early

    def _executive_trigger_objective_reduction(self):
        if self.verbose >= 2: print(f"  EXECUTIVE.Objective_Reduction: Collapsing superposition.")
        self._log_lot_event("executive.objective_reduction.start", {"orp_at_trigger": self.objective_reduction_potential, "superposition_str": self.logical_superposition_str()})

        basis_states = list(self.logical_superposition.keys())
        amplitudes = np.array([self.logical_superposition[s] for s in basis_states], dtype=complex)
        probabilities = np.abs(amplitudes)**2

        sum_probs = np.sum(probabilities)
        if sum_probs < 1e-9:
            if self.verbose >= 1: print("    ERROR: Superposition has near-zero norm before collapse. Defaulting to '00'.")
            self._log_lot_event("executive.objective_reduction.error", {"error":"norm_zero_collapse_00"})
            self.collapsed_logical_state_str = "00"
            self.logical_superposition = {"00":1.0+0j, "01":0.0j, "10":0.0j, "11":0.0j}
        elif not np.isclose(sum_probs, 1.0):
            if self.verbose >= 2: print(f"    Normalizing probabilities for collapse (sum was {sum_probs:.4f}).")
            probabilities /= sum_probs

        try:
            # Ensure probabilities is 1D array for np.random.choice
            if probabilities.ndim > 1: probabilities = probabilities.flatten()
            if not np.isclose(np.sum(probabilities), 1.0): # Final safety re-normalization
                 probabilities = probabilities / np.sum(probabilities)

            chosen_index = np.random.choice(len(basis_states), p=probabilities)
            self.collapsed_logical_state_str = basis_states[chosen_index]
        except ValueError as e:
            if self.verbose >=1: print(f"    Error during probabilistic collapse ({e}). Choosing max_prob or '00'. Probabilities: {probabilities}")
            self._log_lot_event("executive.objective_reduction.error", {"error": str(e), "probs_str":str(probabilities)})
            if probabilities.any() and not np.isnan(probabilities).any() and np.all(probabilities >= 0):
                # Ensure probabilities is 1D before argmax if it somehow became 2D
                if probabilities.ndim > 1: probabilities = probabilities.flatten()
                max_prob_idx = np.argmax(probabilities)
                self.collapsed_logical_state_str = basis_states[max_prob_idx]
            else:
                self.collapsed_logical_state_str = "00"
                self.logical_superposition = {"00":1.0+0j, "01":0.0j, "10":0.0j, "11":0.0j}

        if self.verbose >= 2:
            print(f"    OR Event: Collapsed to |{self.collapsed_logical_state_str}>")

        self.current_orp_before_reset = self.objective_reduction_potential
        self.objective_reduction_potential = 0.0
        for state_key in self.logical_superposition:
            self.logical_superposition[state_key] = 1.0 + 0j if state_key == self.collapsed_logical_state_str else 0.0j

        self._log_lot_event("executive.objective_reduction.end", {"collapsed_to": self.collapsed_logical_state_str, "orp_experienced":self.current_orp_before_reset})
        return self.collapsed_logical_state_str

    # --- Layer 1: Sensor Layer ---
    def _sensor_layer_process_input(self, target_classical_input_str: str) -> str:
        if self.verbose >= 2: print(f"  SENSOR_LAYER: Processing target input '{target_classical_input_str}'.")
        self._log_lot_event("sensor.process_input.start", {"target_input": target_classical_input_str})

        noise_level = self.internal_state_parameters.get('sensor_input_noise_level', 0.0)
        actual_classical_input_str = target_classical_input_str

        if noise_level > 0 and random.random() < 0.75 :
            mutated_input_list = list(target_classical_input_str)
            num_flips = 0
            for i in range(len(mutated_input_list)):
                if random.random() < noise_level:
                    mutated_input_list[i] = '1' if mutated_input_list[i] == '0' else '0'
                    num_flips +=1
            if num_flips > 0:
                actual_classical_input_str = "".join(mutated_input_list)
                if self.verbose >= 1: print(f"    SENSOR_LAYER: Input '{target_classical_input_str}' perceived as '{actual_classical_input_str}' due to noise.")
                self._log_lot_event("sensor.process_input.noise_applied", {"original": target_classical_input_str, "actual": actual_classical_input_str, "noise_level": noise_level, "flips":num_flips})

        self._log_lot_event("sensor.process_input.end", {"actual_input": actual_classical_input_str})
        return actual_classical_input_str

    # --- Layer 2: Associative Layer ---
    def _associative_layer_update_ltm(self, op_sequence, raw_valence, orp_cost, entropy_gen, consolidation_factor=1.0):
        if self.verbose >= 2: print(f"  ASSOCIATIVE_LAYER.LTM_Update: Seq {op_sequence if op_sequence else 'NoOps'}, Val={raw_valence:.2f}, ORP={orp_cost:.2f}, Ent={entropy_gen:.2f}, ConsolFactor={consolidation_factor:.2f}")
        self._log_lot_event("associative.ltm_update.start", {"op_seq_len":len(op_sequence or []), "raw_valence":raw_valence, "orp_cost": orp_cost, "consol_factor": consolidation_factor, "entropy":entropy_gen})

        if not op_sequence: return
        seq_tuple = tuple(tuple(op) for op in op_sequence)

        if raw_valence < self.successful_sequence_threshold_valence * 0.3 and consolidation_factor <= 1.0:
             if self.verbose >=3: print(f"    LTM_Update: Sequence {seq_tuple} not stored, raw_valence {raw_valence:.2f} too low (threshold factor 0.3).")
             self._log_lot_event("associative.ltm_update.skip_low_valence", {"seq_tuple":seq_tuple, "raw_valence":raw_valence})
             return

        entry = self.long_term_memory.get(seq_tuple)
        mutation_rate_store = self.internal_state_parameters.get('ltm_mutation_on_store_rate', 0.0)
        update_strength = int(math.ceil(consolidation_factor))

        if entry:
            entry['count'] += update_strength
            entry['total_valence'] += raw_valence * consolidation_factor
            entry['total_orp_cost'] += orp_cost
            entry['total_entropy_generated'] += entropy_gen

            if random.random() < mutation_rate_store:
                entry['total_valence'] *= (1 + random.uniform(-0.05, 0.05) * update_strength)
                entry['total_orp_cost'] *= (1 + random.uniform(-0.03, 0.03))
                self._log_lot_event("associative.ltm_update.metric_mutation", {"seq":seq_tuple})

            entry['avg_valence'] = entry['total_valence'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_orp_cost'] = entry['total_orp_cost'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_entropy'] = entry['total_entropy_generated'] / entry['count'] if entry['count'] > 0 else 0
        else:
            if len(self.long_term_memory) >= self.long_term_memory_capacity:
                if not self.long_term_memory: return
                min_utility_val = float('inf'); key_to_prune = None
                for k, v_data in self.long_term_memory.items():
                    temp_util = v_data.get('utility', self._associative_layer_calculate_ltm_entry_utility(v_data))
                    if temp_util < min_utility_val: min_utility_val = temp_util; key_to_prune = k

                if key_to_prune:
                    if self.verbose >=3: print(f"    LTM_Update: LTM full. Pruning {key_to_prune} (util {min_utility_val:.2f}).")
                    self._log_lot_event("associative.ltm_update.prune", {"pruned_seq_str":str(key_to_prune), "util":min_utility_val})
                    del self.long_term_memory[key_to_prune]
                elif self.verbose >=2: print("    LTM_Update: LTM full, but no suitable key to prune found.")

            if len(self.long_term_memory) < self.long_term_memory_capacity:
                current_raw_valence_store = raw_valence * consolidation_factor
                current_orp_cost_store = orp_cost
                current_entropy_store = entropy_gen

                if random.random() < mutation_rate_store:
                    current_raw_valence_store *= (1 + random.uniform(-0.05, 0.05) * update_strength)
                    current_orp_cost_store *= (1 + random.uniform(-0.03, 0.03))
                    self._log_lot_event("associative.ltm_update.new_metric_mutation", {"seq":seq_tuple})

                new_entry = {
                    'count': update_strength,
                    'total_valence': current_raw_valence_store, 'avg_valence': current_raw_valence_store / update_strength if update_strength else current_raw_valence_store,
                    'total_orp_cost': current_orp_cost_store * update_strength, 'avg_orp_cost': current_orp_cost_store,
                    'total_entropy_generated': current_entropy_store * update_strength, 'avg_entropy': current_entropy_store,
                    'first_cycle': self.current_cycle_num, 'last_cycle': self.current_cycle_num,
                }
                self.long_term_memory[seq_tuple] = new_entry
                if self.verbose >=3: print(f"    LTM_Update: Added new sequence {seq_tuple} with avg_valence {new_entry['avg_valence']:.2f}.")
                self._log_lot_event("associative.ltm_update.new_entry", {"seq_str":str(seq_tuple), "val":new_entry['avg_valence']})

        if seq_tuple in self.long_term_memory:
             self.long_term_memory[seq_tuple]['utility'] = self._associative_layer_calculate_ltm_entry_utility(self.long_term_memory[seq_tuple])
             self.long_term_memory[seq_tuple]['last_cycle'] = self.current_cycle_num

    def _associative_layer_calculate_ltm_entry_utility(self, seq_data):
        norm_orp_cost = seq_data['avg_orp_cost'] / (self.E_OR_THRESHOLD + 1e-6)
        utility = (self.ltm_utility_weight_valence * seq_data['avg_valence'] -
                   self.ltm_utility_weight_efficiency * norm_orp_cost +
                   0.05 * seq_data.get('avg_entropy', 0.0))
        return utility

    def _associative_layer_recall_from_ltm_strategy(self, current_orp_value, exec_thought_log):
        if not self.long_term_memory:
            exec_thought_log.append("LTM recall: LTM empty.")
            return None, current_orp_value

        candidate_sequences = []; weights = []
        min_utility_for_recall = 0.05

        for seq_tuple, data in self.long_term_memory.items():
            utility = data.get('utility', self._associative_layer_calculate_ltm_entry_utility(data))
            if utility > min_utility_for_recall:
                projected_cost = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in seq_tuple)
                if current_orp_value + projected_cost < self.E_OR_THRESHOLD * 1.2:
                    candidate_sequences.append(list(seq_tuple))
                    weights.append(utility**2.5)

        if not candidate_sequences:
            exec_thought_log.append(f"LTM recall: No sequences found with utility > {min_utility_for_recall} or all too costly from ORP {current_orp_value:.2f}.")
            return None, current_orp_value

        sum_weights = sum(weights)
        if sum_weights <= 1e-6:
             exec_thought_log.append("LTM recall: No LTM sequences with positive utility weights after filtering.")
             return None, current_orp_value

        normalized_weights = [w / sum_weights for w in weights]
        chosen_sequence_idx = random.choices(range(len(candidate_sequences)), weights=normalized_weights, k=1)[0]
        chosen_sequence_ops_orig = candidate_sequences[chosen_sequence_idx]
        chosen_sequence_ops = [list(op) for op in chosen_sequence_ops_orig]

        mutation_rate_replay = self.internal_state_parameters.get('ltm_mutation_on_replay_rate', 0.0)
        if chosen_sequence_ops and random.random() < mutation_rate_replay and len(chosen_sequence_ops) > 0:
            idx_to_mutate = random.randrange(len(chosen_sequence_ops))
            op_char, op_arg = chosen_sequence_ops[idx_to_mutate]
            original_op_tuple_str = f"('{op_char}', {op_arg})"

            mutation_type_rand = random.random()
            if mutation_type_rand < 0.35 and op_char in ['X', 'Z', 'H'] and isinstance(op_arg, int):
                chosen_sequence_ops[idx_to_mutate][1] = 1 - op_arg
            elif mutation_type_rand < 0.65:
                compatible_ops={'X':['Z','H'],'Z':['X','H'],'H':['X','Z'],'CNOT':['CZ'],'CZ':['CNOT']}
                new_op_char = random.choice(compatible_ops.get(op_char, ['X','Z','H']))
                new_op_arg = op_arg
                if new_op_char in ['X','Z','H']: new_op_arg = random.randint(0,1)
                elif new_op_char in ['CNOT', 'CZ']: new_op_arg = tuple(random.sample([0,1],2))
                chosen_sequence_ops[idx_to_mutate] = [new_op_char, new_op_arg]
            elif len(chosen_sequence_ops) > 1 and random.random() < 0.5 :
                del chosen_sequence_ops[idx_to_mutate]
            else:
                new_op_insert_char = random.choice(['X','Z','H'])
                new_op_insert_arg = random.randint(0,1) if new_op_insert_char in ['X','Z','H'] else tuple(random.sample([0,1],2))
                chosen_sequence_ops.insert(random.randint(0, len(chosen_sequence_ops)), [new_op_insert_char, new_op_insert_arg])

            exec_thought_log.append(f"LTM Replay MUTATION: Op {original_op_tuple_str} in {chosen_sequence_ops_orig} -> mutated to/around in {chosen_sequence_ops}.")
            self._log_lot_event("associative.ltm_recall.mutation", {"original_seq_str": str(chosen_sequence_ops_orig), "mutated_seq_str": str(chosen_sequence_ops)})


        projected_orp_increase_final = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in chosen_sequence_ops)
        if current_orp_value + projected_orp_increase_final >= self.E_OR_THRESHOLD * 1.1 and len(chosen_sequence_ops) > 0:
            exec_thought_log.append(f"LTM recall: Mutated/Chosen seq {chosen_sequence_ops} too costly. ORP would be {current_orp_value + projected_orp_increase_final:.2f}. Skipped.")
            return None, current_orp_value

        final_chosen_ops_as_tuples = [tuple(op) for op in chosen_sequence_ops]

        orig_data_key = tuple(tuple(op) for op in chosen_sequence_ops_orig) # Ensure hashable tuple of tuples for dict key
        orig_data = self.long_term_memory[orig_data_key]
        exec_thought_log.append(f"LTM recall: Replaying {final_chosen_ops_as_tuples} (orig_avg_V={orig_data['avg_valence']:.2f}, util={orig_data['utility']:.2f}). Cost {projected_orp_increase_final:.2f}")
        self._log_lot_event("associative.ltm_recall.chosen", {"seq_str":str(final_chosen_ops_as_tuples), "orig_util":orig_data['utility']})
        return final_chosen_ops_as_tuples, current_orp_value


    # --- Layer 3: Executive Layer (Decision Making, Planning, Conscious Experience) ---
    def _executive_evaluate_outcome_and_update_mood(self, logical_outcome_str, orp_at_collapse, entropy_at_collapse, num_ops_executed_this_cycle):
        if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.Outcome_Eval: |{logical_outcome_str}>, ORP={orp_at_collapse:.3f}, Ent={entropy_at_collapse:.2f}, Ops#={num_ops_executed_this_cycle}")
        acc_thoughts_log = []

        raw_valence = self.outcome_valence_map.get(logical_outcome_str, -0.15)
        mod_valence = raw_valence
        acc_thoughts_log.append(f"Raw val for |{logical_outcome_str}> is {raw_valence:.2f}.")

        orp_surprise_factor = 0.20
        if orp_at_collapse < self.E_OR_THRESHOLD * 0.35:
            penalty = orp_surprise_factor * (abs(raw_valence) if raw_valence != 0 else 0.25)
            mod_valence -= penalty
            acc_thoughts_log.append(f"Early OR collapse, val modified by {-penalty:.2f}.")
        elif orp_at_collapse > self.E_OR_THRESHOLD * 1.35 and num_ops_executed_this_cycle > 0:
            late_factor = -0.08 if raw_valence < 0 else 0.08
            mod_valence += late_factor
            acc_thoughts_log.append(f"Late OR collapse, val modified by {late_factor:.2f}.")

        current_preferred_state = self.internal_state_parameters.get('preferred_logical_state')
        if current_preferred_state is not None and current_preferred_state == logical_outcome_str:
            preference_bonus = 0.30 * (1.0 - abs(mod_valence))
            mod_valence += preference_bonus
            acc_thoughts_log.append(f"Preferred state |{current_preferred_state}> met, val boosted by {preference_bonus:.2f}.")
            if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
                self._executive_update_goal_progress(logical_outcome_str, None)

        mod_valence = np.clip(mod_valence, -1.0, 1.0)
        self.last_cycle_valence_raw = raw_valence
        self.last_cycle_valence_mod = mod_valence
        acc_thoughts_log.append(f"Final val (raw/mod): {raw_valence:.2f}/{mod_valence:.2f}.")
        self._log_lot_event("executive.outcome_eval.valence", {"raw":raw_valence, "mod":mod_valence, "outcome_state":logical_outcome_str, "orp_collapse": orp_at_collapse})

        current_mood = self.internal_state_parameters['mood']
        mood_inertia = 0.88
        valence_influence_on_mood = 0.28
        new_mood = current_mood * mood_inertia + mod_valence * valence_influence_on_mood
        self.internal_state_parameters['mood'] = np.clip(new_mood, -1.0, 1.0)
        acc_thoughts_log.append(f"Mood updated from {current_mood:.2f} to {self.internal_state_parameters['mood']:.2f}.")
        self._log_lot_event("executive.outcome_eval.mood", {"new_mood":self.internal_state_parameters['mood'], "old_mood": current_mood})

        current_frustration = self.internal_state_parameters['frustration']
        frustration_threshold = self.metacognition_params['frustration_threshold']
        if mod_valence < self.metacognition_params['low_valence_threshold'] * 0.7:
            current_frustration += 0.22
        else:
            current_frustration *= 0.82
        self.internal_state_parameters['frustration'] = np.clip(current_frustration, 0.0, 1.0)

        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            self.internal_state_parameters['exploration_mode_countdown'] -= 1
            if self.verbose >= 2 and self.internal_state_parameters['exploration_mode_countdown'] == 0:
                acc_thoughts_log.append("Exploration mode ended this cycle.")
                self._log_lot_event("executive.outcome_eval.exploration_end", {})

        if self.internal_state_parameters['frustration'] >= frustration_threshold and \
           self.internal_state_parameters['exploration_mode_countdown'] == 0:
            if self.verbose >= 1: print(f"[{self.agent_id}] High frustration ({self.internal_state_parameters['frustration']:.2f}) triggered exploration mode!")
            self._log_lot_event("executive.outcome_eval.exploration_start", {"frustration":self.internal_state_parameters['frustration'], "threshold": frustration_threshold})
            self.internal_state_parameters['exploration_mode_countdown'] = self.metacognition_params['exploration_mode_duration']
            self.internal_state_parameters['frustration'] = 0.25
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.35)

        acc_thoughts_log.append(f"Frustration: {self.internal_state_parameters['frustration']:.2f}, Exploration T-: {self.internal_state_parameters['exploration_mode_countdown']}.")

        return {
            'raw_valence':raw_valence, 'mod_valence':mod_valence,
            'mood':self.internal_state_parameters['mood'],
            'frustration':self.internal_state_parameters['frustration'],
            'exploration_countdown':self.internal_state_parameters['exploration_mode_countdown'],
            'thoughts_log': acc_thoughts_log
        }

    def _executive_generate_computation_sequence(self, ops_provided_externally=None):
        if ops_provided_externally is not None:
            if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.OpGen: Using externally provided ops: {ops_provided_externally}")
            self._log_lot_event("executive.opgen.external", {"ops_count": len(ops_provided_externally)})
            return ops_provided_externally, "StrategyProvidedExternal", ["Ops provided externally."]

        exec_thought_log = ["OpGen: Generating new computation sequence:"]
        self._log_lot_event("executive.opgen.start", {"orp_current":self.objective_reduction_potential, "threshold": self.E_OR_THRESHOLD})

        ops_sequence = []
        chosen_strategy_name = "NoOpsMethod"

        effective_attention = self.internal_state_parameters['attention_level']
        cognitive_load_factor = 1.0 - (self.internal_state_parameters['cognitive_load'] * 0.65)
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * cognitive_load_factor * effective_attention, scale=1.0)))
        num_ops_target = min(num_ops_target, 10)

        exec_thought_log.append(f"  Target ops: ~{num_ops_target} (base:{num_ops_target_base}, load_factor:{cognitive_load_factor:.2f}, attn:{effective_attention:.2f}). ORP start: {self.objective_reduction_potential:.3f}")

        current_strategy_weights = self.internal_state_parameters['strategy_weights'].copy()

        tfg_window = self.temporal_grid_params['feedback_window']
        grid_entries_to_consider = list(self.temporal_feedback_grid)[-tfg_window:]

        if grid_entries_to_consider:
            recent_valence_deltas = [g[1] for g in grid_entries_to_consider if g[1] is not None]
            recent_entropy_shifts = [g[2] for g in grid_entries_to_consider if g[2] is not None]

            avg_recent_valence_delta = np.mean(recent_valence_deltas) if recent_valence_deltas else 0.0
            avg_recent_entropy_shift = np.mean(recent_entropy_shifts) if recent_entropy_shifts else 0.0

            exec_thought_log.append(f"  TemporalGridInfo (last {len(grid_entries_to_consider)} entries): AvgValDelta={avg_recent_valence_delta:.2f}, AvgEntShift={avg_recent_entropy_shift:.2f}")

            valence_bias_strength = self.internal_state_parameters.get('temporal_feedback_valence_bias_strength', 0.1)
            entropy_bias_strength = self.internal_state_parameters.get('temporal_feedback_entropy_bias_strength', 0.05)

            if avg_recent_valence_delta < self.temporal_grid_params['low_valence_delta_threshold'] and len(recent_valence_deltas) > 0:
                exec_thought_log.append(f"    TFG Bias: Low avg valence delta ({avg_recent_valence_delta:.2f} < {self.temporal_grid_params['low_valence_delta_threshold']}). Increasing exploration/memory focus.")
                delta_v_bias_amount = abs(avg_recent_valence_delta) * valence_bias_strength
                current_strategy_weights['problem_solve'] = max(0.01, current_strategy_weights['problem_solve'] * (1 - delta_v_bias_amount))
                current_strategy_weights['goal_seek'] = max(0.01, current_strategy_weights['goal_seek'] * (1 - delta_v_bias_amount))
                current_strategy_weights['curiosity'] += delta_v_bias_amount * 0.6 + 0.03
                current_strategy_weights['memory'] += delta_v_bias_amount * 0.4 + 0.03
                self._log_lot_event("executive.opgen.temporal_bias.neg_val_delta", {
                    "val_delta": avg_recent_valence_delta, "bias_str": valence_bias_strength, "bias_eff": delta_v_bias_amount,
                    "old_ps_w": self.internal_state_parameters['strategy_weights']['problem_solve'], "new_ps_w": current_strategy_weights['problem_solve']
                })

            if avg_recent_entropy_shift > self.temporal_grid_params['high_entropy_shift_threshold'] and avg_recent_valence_delta < 0.05 and len(recent_entropy_shifts) > 0 :
                exec_thought_log.append(f"    TFG Bias: High avg entropy shift ({avg_recent_entropy_shift:.2f} > {self.temporal_grid_params['high_entropy_shift_threshold']}) with low/neutral valence. Increasing memory focus, reducing curiosity.")
                delta_e_bias_amount = avg_recent_entropy_shift * entropy_bias_strength
                current_strategy_weights['curiosity'] = max(0.01, current_strategy_weights['curiosity'] * (1 - delta_e_bias_amount))
                current_strategy_weights['memory'] += delta_e_bias_amount * 0.7 + 0.03
                self._log_lot_event("executive.opgen.temporal_bias.high_ent_shift", {
                    "ent_shift": avg_recent_entropy_shift, "bias_str": entropy_bias_strength, "bias_eff": delta_e_bias_amount,
                    "old_cur_w": self.internal_state_parameters['strategy_weights']['curiosity'], "new_cur_w": current_strategy_weights['curiosity']
                })
        else:
            exec_thought_log.append("  TemporalGridInfo: Grid empty or too few entries for feedback.")


        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            current_step_info = self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index]
            exec_thought_log.append(f"  Goal Active: '{current_step_info.get('name', 'UnnamedStep')}'. Boosting goal_seek/problem_solve.")
            current_strategy_weights['goal_seek'] *= 1.6
            current_strategy_weights['problem_solve'] *= 1.3
            if current_step_info.get("target_state"):
                self.internal_state_parameters['preferred_logical_state'] = current_step_info["target_state"]
                exec_thought_log.append(f"    Goal sets preferred state to |{current_step_info['target_state']}>")


        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            exec_thought_log.append("  Exploration mode active: Boosting curiosity, reducing goal/problem focus.")
            current_strategy_weights['curiosity'] = min(1.0, current_strategy_weights.get('curiosity',0.1)*2.8)
            current_strategy_weights['problem_solve'] *= 0.5
            current_strategy_weights['goal_seek'] *= 0.3
            self._log_lot_event("executive.opgen.exploration_bias", {"new_cur_weight": current_strategy_weights['curiosity']})


        if self.smn_internal_flags.get('force_ltm_reactive_op_next_cycle', False):
            exec_thought_log.append("  SMN/Interrupt Flag: Forcing LTM Reactive operation strategy.")
            current_strategy_weights = {'memory': 1.0, 'problem_solve': 0.001, 'goal_seek': 0.001, 'curiosity': 0.001}
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = False
            self._log_lot_event("executive.opgen.interrupt_bias.force_ltm", {})


        for key in DEFAULT_INTERNAL_PARAMS['strategy_weights']:
            if key not in current_strategy_weights: current_strategy_weights[key] = 0.001

        # Filter for valid numeric weights before sum and normalization
        valid_weights = {k: v for k, v in current_strategy_weights.items() if isinstance(v, (int, float))}
        total_weight = sum(w for w in valid_weights.values() if w > 0)

        if total_weight <= 1e-6:
            current_strategy_weights = {'curiosity': 1.0}; total_weight = 1.0
            valid_weights = {'curiosity': 1.0} # Ensure valid_weights reflects this fallback

        strategy_choices = []
        strategy_probs = []

        if not valid_weights: # If after all filtering, valid_weights is empty (e.g. all were non-numeric or non-positive)
             selected_strategy = 'curiosity' # Fallback
             strategy_choices = ['curiosity']
             strategy_probs = [1.0]
        else:
            for s_name, s_weight in valid_weights.items():
                strategy_choices.append(s_name)
                strategy_probs.append(max(0, s_weight) / total_weight if total_weight > 1e-6 else 1.0/len(valid_weights))

            try:
                selected_strategy = random.choices(strategy_choices, weights=strategy_probs, k=1)[0]
            except ValueError as e:
                selected_strategy = 'curiosity'
                exec_thought_log.append(f"  Error in strategy selection (probs sum to {sum(strategy_probs)}, choice from {strategy_choices}, error: {e}), defaulting to curiosity.")
                # Fix strategy_choices/probs if error happened (e.g. all probs became zero after an op)
                if not strategy_choices: strategy_choices = ['curiosity']
                if not strategy_probs or sum(strategy_probs) < 1e-6 : strategy_probs = [1.0/len(strategy_choices)]*len(strategy_choices)


        exec_thought_log.append(f"  Strategy weights (norm): { {s:f'{p:.3f}' for s,p in zip(strategy_choices, strategy_probs)} }")
        exec_thought_log.append(f"  Selected primary strategy: {selected_strategy}")
        self._log_lot_event("executive.opgen.strategy_selected", {"strategy":selected_strategy, "weights_str": str({s:f'{p:.2f}' for s,p in zip(strategy_choices, strategy_probs)}) })

        simulated_orp_accumulator = self.objective_reduction_potential

        if selected_strategy == 'memory':
            replay_ops, _ = self._associative_layer_recall_from_ltm_strategy(simulated_orp_accumulator, exec_thought_log)
            if replay_ops: ops_sequence = replay_ops; chosen_strategy_name = "StrategyLTMReplay"

        if not ops_sequence and selected_strategy == 'problem_solve':
            pref_state = self.internal_state_parameters['preferred_logical_state']
            if pref_state:
                exec_thought_log.append(f"  ProblemSolving towards |{pref_state}> from |{self.collapsed_logical_state_str}>")
                current_l1,current_l0=int(self.collapsed_logical_state_str[0]),int(self.collapsed_logical_state_str[1])
                target_l1,target_l0=int(pref_state[0]),int(pref_state[1])
                planned_problem_ops = []
                temp_plan_orp = simulated_orp_accumulator + self.operation_costs.get('PLANNING_BASE', 0.02)

                if abs((current_l0+current_l1) - (target_l0+target_l1)) >=2 and random.random() < 0.4 :
                    op_cost_h = self.operation_costs.get('H', 0.3)
                    if temp_plan_orp + op_cost_h < self.E_OR_THRESHOLD:
                         h_target_q = 0 if current_l0 != target_l0 else 1
                         planned_problem_ops.append(('H', h_target_q)); temp_plan_orp += op_cost_h
                         exec_thought_log.append(f"    ProblemSolving plan included H for |{pref_state}>.")

                if current_l0 != target_l0:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD: planned_problem_ops.append(('X',0)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    PS: Cannot apply ('X',0) to reach target |{pref_state}> due to ORP limit.")
                if current_l1 != target_l1:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD: planned_problem_ops.append(('X',1)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    PS: Cannot apply ('X',1) to reach target |{pref_state}> due to ORP limit.")

                if planned_problem_ops:
                    ops_sequence = planned_problem_ops; chosen_strategy_name = "StrategyProblemSolving"
                    exec_thought_log.append(f"    ProblemSolving plan: {ops_sequence}")
                elif pref_state == self.collapsed_logical_state_str:
                     exec_thought_log.append(f"    ProblemSolving: Already at preferred state |{pref_state}>.")
                else:
                     exec_thought_log.append(f"    ProblemSolving: No viable ops plan to |{pref_state}> (possibly ORP limited).")
            else:
                exec_thought_log.append("  ProblemSolving selected, but no preferred_logical_state is set. Falling through.")

        if not ops_sequence:
            if selected_strategy == 'goal_seek' and self.internal_state_parameters['preferred_logical_state']:
                chosen_strategy_name = "StrategyGoalSeekingLoop"
                exec_thought_log.append(f"  Executing GoalSeeking towards |{self.internal_state_parameters['preferred_logical_state']}>")
            else:
                chosen_strategy_name = "StrategyCuriosityDrivenLoop"
                exec_thought_log.append(f"  Executing CuriosityDriven op generation.")

            pref_s = self.internal_state_parameters['preferred_logical_state']
            c_l1,c_l0=int(self.collapsed_logical_state_str[0]),int(self.collapsed_logical_state_str[1])

            for op_count in range(num_ops_target):
                op_c, op_a = 'X', 0
                op_cost = self.operation_costs.get(op_c.upper(), 0.05)


                is_goal_seek_mode = (chosen_strategy_name == "StrategyGoalSeekingLoop" and pref_s)

                if is_goal_seek_mode:
                    t_l1,t_l0 = int(pref_s[0]), int(pref_s[1])
                    if c_l0 != t_l0 : op_c, op_a = 'X', 0
                    elif c_l1 != t_l1 : op_c, op_a = 'X', 1
                    elif random.random() < 0.4: op_c,op_a = ('H',random.randint(0,1))
                    else:
                        op_c = random.choice(['H','Z'] + (['CNOT','CZ'] if random.random() < 0.25 else []))
                        op_a = random.randint(0,1) if op_c in ['H','X','Z'] else tuple(random.sample([0,1],2))
                else:
                    op_choices = ['X','Z','H']
                    if random.random() < 0.4: op_choices.extend(['CNOT', 'CZ'])
                    op_c=random.choice(op_choices)
                    op_a = random.randint(0,1) if op_c in ['H','X','Z'] else tuple(random.sample([0,1],2))

                op_cost = self.operation_costs.get(op_c.upper(), 0.05)

                attention_lapse_prob = (self.internal_state_parameters['cognitive_load'] * 0.2) + \
                                      (1.0 - effective_attention) * 0.15
                if random.random() < attention_lapse_prob:
                    original_op_tuple = (op_c, op_a)
                    if op_c in ['X','Z','H'] and isinstance(op_a,int): op_a = 1 - op_a
                    elif op_c in ['CNOT','CZ'] and isinstance(op_a,tuple): op_a = (op_a[1],op_a[0])
                    else:
                        op_c = random.choice(['X','Z']) if op_c not in ['X','Z'] else 'H'

                    op_cost += self.operation_costs.get('ERROR_PENALTY',0.05) * 0.5
                    exec_thought_log.append(f"      ATTENTION LAPSE! Op {original_op_tuple} -> ({op_c},{op_a}), cost penalty. LapseProb={attention_lapse_prob:.2f}")
                    self._log_lot_event("executive.opgen.attention_lapse", {"original_op_str":str(original_op_tuple), "mutated_op_str":str((op_c,op_a)), "lapse_prob":attention_lapse_prob})


                if simulated_orp_accumulator + op_cost < self.E_OR_THRESHOLD * 0.98 :
                    ops_sequence.append((op_c,op_a))
                    simulated_orp_accumulator += op_cost
                    if op_c == 'X':
                        if op_a == 0: c_l0 = 1-c_l0
                        else: c_l1 = 1-c_l1
                else:
                    exec_thought_log.append(f"    OpGen loop ({op_count+1}/{num_ops_target}): Op ('{op_c}',{op_a}) cost {op_cost:.2f} would exceed ORP. Stopping. (SimORP {simulated_orp_accumulator:.2f} + {op_cost:.2f} vs E_OR {self.E_OR_THRESHOLD:.2f})")
                    break

        if not ops_sequence and chosen_strategy_name not in ["StrategyLTMReplay", "StrategyProblemSolving"]:
            chosen_strategy_name = "NoOpsGeneratedByLoop"
            exec_thought_log.append("  Final: No operations generated by dynamic loop (Curiosity/GoalSeek).")
        elif not ops_sequence:
             chosen_strategy_name = "NoOpsUltimately"
             exec_thought_log.append("  Final: No operations generated by any strategy.")

        self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy":chosen_strategy_name, "final_sim_orp":simulated_orp_accumulator})
        return ops_sequence, chosen_strategy_name, exec_thought_log


    def _executive_plan_next_target_input(self, current_outcome_str, executive_eval_results, exec_thought_log):
        exec_thought_log.append(f"PlanNextInput based on |{current_outcome_str}> (mood {executive_eval_results['mood']:.2f}):")

        base_next_input_map = {"00":"01", "01":"10", "10":"11", "11":"00"}
        next_input = base_next_input_map.get(current_outcome_str, "00")
        exec_thought_log.append(f"  Base heuristic next input: |{next_input}>.")

        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            current_step_idx = self.current_goal_state_obj.current_step_index
            if 0 <= current_step_idx < len(self.current_goal_state_obj.steps):
                step_info = self.current_goal_state_obj.steps[current_step_idx]
                if step_info.get("next_input_for_world"):
                    next_input = step_info["next_input_for_world"]
                    exec_thought_log.append(f"  GoalStep '{step_info.get('name')}' overrides next input to |{next_input}>.")
                    self._log_lot_event("executive.plannext.goal_override", {"next_input": next_input, "goal_step_name":step_info.get('name',"")})

        elif self.internal_state_parameters['preferred_logical_state'] and \
           self.internal_state_parameters['preferred_logical_state'] != next_input and \
           random.random() < self.internal_state_parameters['goal_seeking_bias'] * 0.75:
            next_input = self.internal_state_parameters['preferred_logical_state']
            exec_thought_log.append(f"  Overridden by PreferredStateBias (bias {self.internal_state_parameters['goal_seeking_bias']:.2f}): next input |{next_input}>.")
            self._log_lot_event("executive.plannext.preferred_state_override", {"next_input": next_input, "bias": self.internal_state_parameters['goal_seeking_bias']})

        elif executive_eval_results['exploration_countdown'] > 0 or \
             (executive_eval_results['mood'] < -0.65 and random.random() < 0.55):
            available_inputs = list(base_next_input_map.keys())
            if current_outcome_str in available_inputs: available_inputs.remove(str(current_outcome_str))
            if str(next_input) in available_inputs: available_inputs.remove(str(next_input))

            if available_inputs:
                next_input = random.choice(available_inputs)
                exec_thought_log.append(f"  Exploration/Mood (mood {executive_eval_results['mood']:.2f}, exp T-{executive_eval_results['exploration_countdown']}) override: next input |{next_input}>.")
                self._log_lot_event("executive.plannext.exploration_override", {"next_input": next_input, "mood":executive_eval_results['mood']})
            else:
                next_input = base_next_input_map.get(current_outcome_str, "00")
                exec_thought_log.append(f"  Exploration override failed (no other states avail), using default |{next_input}>.")

        elif executive_eval_results['mood'] > 0.75 and random.random() < 0.40 and self.cycle_history:
            last_actual_input = self.cycle_history[-1]['actual_input_state_used']
            if last_actual_input and last_actual_input != current_outcome_str :
                next_input = last_actual_input
                exec_thought_log.append(f"  Good mood ({executive_eval_results['mood']:.2f}), repeating last input context |{last_actual_input}>.")
                self._log_lot_event("executive.plannext.good_mood_repeat", {"next_input": next_input, "mood":executive_eval_results['mood']})

        final_next_input_str = str(next_input)
        exec_thought_log.append(f"  Final proposed next input: |{final_next_input_str}>.")
        self.next_target_input_state = final_next_input_str
        return final_next_input_str


    def _executive_update_goal_progress(self, collapsed_outcome_str, executed_ops):
        if not (self.current_goal_state_obj and self.current_goal_state_obj.status == "active"):
            return

        goal = self.current_goal_state_obj
        step_idx = goal.current_step_index
        if not (0 <= step_idx < len(goal.steps)):
            if self.verbose >= 1: print(f"[{self.agent_id}] Goal Error: Invalid step index {step_idx} for goal '{goal.current_goal}'")
            self._log_lot_event("executive.goalprogress.error", {"goal_name": goal.current_goal, "error": "invalid_step_idx", "step_idx":step_idx})
            goal.status = "failed"; goal.history.append({"cycle": self.current_cycle_num, "event": "error_invalid_step_idx"})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)
            return

        current_step = goal.steps[step_idx]
        step_name = current_step.get("name", f"Step {step_idx + 1}")
        self._log_lot_event("executive.goalprogress.check", {"goal_name": goal.current_goal, "step_name": step_name, "outcome_state":collapsed_outcome_str})

        achieved_step = False
        if current_step.get("target_state") and collapsed_outcome_str == current_step["target_state"]:
            achieved_step = True
            if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via target state |{collapsed_outcome_str}>.")
        elif callable(current_step.get("completion_criteria")):
            try:
                context = {'collapsed_state': collapsed_outcome_str, 'ops': executed_ops, 'agent_public_state': self.get_public_state_summary()}
                if current_step["completion_criteria"](context):
                    achieved_step = True
                    if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via custom criteria.")
            except Exception as e:
                if self.verbose >=1: print(f"[{self.agent_id}] Error in goal step completion_criteria for '{step_name}': {e}")
                self._log_lot_event("executive.goalprogress.criteria_error", {"step_name":step_name, "error_str":str(e)})

        if achieved_step:
            goal.history.append({"cycle": self.current_cycle_num, "event": f"step_completed", "step_name": step_name, "outcome_state":collapsed_outcome_str})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['step_completion_valence_bonus'], -1.0, 1.0)
            goal.current_step_index += 1
            num_steps = len(goal.steps)
            goal.progress = goal.current_step_index / num_steps if num_steps > 0 else 1.0
            self._log_lot_event("executive.goalprogress.step_complete", {"step_name": step_name, "new_progress": goal.progress, "valence_mod_bonus":self.goal_state_config_params['step_completion_valence_bonus']})

            if goal.current_step_index >= len(goal.steps):
                goal.status = "completed"; goal.progress = 1.0
                if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' COMPLETED!")
                self._log_lot_event("executive.goalprogress.goal_complete", {"goal_name": goal.current_goal, "valence_mod_bonus":self.goal_state_config_params['completion_valence_bonus']})
                self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['completion_valence_bonus'], -1.0, 1.0)
                if self.internal_state_parameters['preferred_logical_state'] == current_step.get("target_state"):
                    self.internal_state_parameters['preferred_logical_state'] = None
            else:
                 next_step_name = goal.steps[goal.current_step_index].get("name", f"Step {goal.current_step_index+1}")
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Advanced to step '{next_step_name}'.")
        else:
            goal.history.append({"cycle": self.current_cycle_num, "event": "step_no_progress", "step_name": step_name, "current_outcome": collapsed_outcome_str})
            max_cycles_on_step_val = current_step.get("max_cycles_on_step", float('inf'))

            cycles_on_this_step_count = sum(1 for h_entry in goal.history
                                            if h_entry.get("step_name")==step_name and \
                                               h_entry.get("current_step_index_at_event", goal.current_step_index) == goal.current_step_index and \
                                               h_entry.get("event") in ["step_no_progress", "step_try", "step_active"]) # Count attempts for *this specific instance* of the step
            if cycles_on_this_step_count >= max_cycles_on_step_val :
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' FAILED due to too many cycles ({cycles_on_this_step_count}) on step '{step_name}'.")
                 self._log_lot_event("executive.goalprogress.goal_fail", {"goal_name":goal.current_goal, "reason":f"max_cycles_on_step_{step_name}", "cycles_spent": cycles_on_this_step_count})
                 goal.status = "failed"
                 self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)


    # --- Feature 4: Collapse-Triggered Interrupt Handlers ---
    def _executive_handle_collapse_interrupts(self, orp_at_collapse, executed_ops_this_cycle, raw_valence_of_collapse):
        if not self.interrupt_handler_params.get('enabled', False): return

        self._log_lot_event("executive.interrupt_handler.check", {"orp_at_collapse":orp_at_collapse, "raw_valence_input":raw_valence_of_collapse, "ops_count":len(executed_ops_this_cycle or [])})

        num_ops = len(executed_ops_this_cycle or [])
        expected_orp = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in (executed_ops_this_cycle or [])) + (0.05 if num_ops > 0 else 0)
        orp_is_surprising = (orp_at_collapse > expected_orp * self.interrupt_handler_params['consolidation_orp_surprise_factor'] and expected_orp > 0.05 and num_ops > 0)

        valence_is_extreme = abs(raw_valence_of_collapse) >= self.interrupt_handler_params['consolidation_valence_abs_threshold']
        if valence_is_extreme or orp_is_surprising:
            consol_bonus = self.interrupt_handler_params['consolidation_strength_bonus']
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Strong LTM consolidation triggered (factor {consol_bonus:.1f}). Valence: {raw_valence_of_collapse:.2f}, ORPSurprise: {orp_is_surprising} (ORP {orp_at_collapse:.2f} vs Exp {expected_orp:.2f})")
            self._log_lot_event("executive.interrupt_handler.strong_consolidation", {"bonus_factor":consol_bonus, "reason_valence_extreme":valence_is_extreme, "reason_orp_surprise":orp_is_surprising})
            self.smn_internal_flags['ltm_consolidation_bonus_factor'] = consol_bonus

        if raw_valence_of_collapse < self.interrupt_handler_params['reactive_ltm_valence_threshold']:
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Reactive LTM flag set for next cycle due to low valence ({raw_valence_of_collapse:.2f} < {self.interrupt_handler_params['reactive_ltm_valence_threshold']}).")
            self._log_lot_event("executive.interrupt_handler.reactive_ltm_flag", {"valence":raw_valence_of_collapse})
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = True

        if raw_valence_of_collapse >= self.interrupt_handler_params['cognitive_fork_valence_threshold'] and \
           self.collapsed_logical_state_str != self.internal_state_parameters.get('preferred_logical_state'):
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Cognitive fork - marking |{self.collapsed_logical_state_str}> as new high-interest preferred state (Valence {raw_valence_of_collapse:.2f}).")
            self._log_lot_event("executive.interrupt_handler.cognitive_fork", {"new_preferred_state":self.collapsed_logical_state_str, "valence_trigger":raw_valence_of_collapse})
            self.internal_state_parameters['preferred_logical_state'] = self.collapsed_logical_state_str
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + self.interrupt_handler_params['cognitive_fork_goal_bias_boost'])


    # --- Layer 4: Meta Layer (Monitoring, Adaptation, Self-Reflection) ---
    def _meta_layer_update_cognitive_parameters(self, orp_at_collapse, num_ops_executed, executive_eval_results, entropy_at_collapse):
        if self.verbose >= 2: print(f"  META_LAYER.CognitiveParamUpdate (mood: {executive_eval_results['mood']:.2f}):")
        self._log_lot_event("meta.cog_param_update.start", {"mood_in":executive_eval_results['mood'], "frustration_in": executive_eval_results['frustration']})

        load_increase_factor = 0.18
        load_from_orp = (orp_at_collapse / (self.E_OR_THRESHOLD + 1e-6)) * load_increase_factor if num_ops_executed > 0 else 0
        load_from_ops = num_ops_executed * 0.025
        current_load = self.internal_state_parameters['cognitive_load']
        new_load = current_load * 0.88 + load_from_orp + load_from_ops
        self.internal_state_parameters['cognitive_load'] = np.clip(new_load, 0.0, 1.0)

        current_attention = self.internal_state_parameters['attention_level']
        attention_decay_due_to_load = self.internal_state_parameters['cognitive_load'] * 0.12
        attention_recovery_rate = 0.07
        mood_effect_on_attention = self.internal_state_parameters['mood'] * 0.09

        new_attention = current_attention * (1 - attention_decay_due_to_load) + \
                        (1.0 - current_attention) * attention_recovery_rate + \
                        mood_effect_on_attention
        self.internal_state_parameters['attention_level'] = np.clip(new_attention, 0.05, 1.0)

        if self.verbose >=3: print(f"    CogLoad: {self.internal_state_parameters['cognitive_load']:.2f} (from {current_load:.2f}), Attention: {self.internal_state_parameters['attention_level']:.2f} (from {current_attention:.2f})")

        mod_valence = executive_eval_results['mod_valence']
        curiosity_change = 0.0
        cur_base_rate = 0.035
        if mod_valence < -0.35: curiosity_change += cur_base_rate
        if entropy_at_collapse > 1.6 and mod_valence < 0.05 : curiosity_change += cur_base_rate * 0.7
        if self.internal_state_parameters['exploration_mode_countdown'] > 0 : curiosity_change += cur_base_rate * 1.5
        curiosity_change -= cur_base_rate * 0.4
        self.internal_state_parameters['curiosity'] = np.clip(self.internal_state_parameters['curiosity'] + curiosity_change, 0.01, 0.99)

        goal_bias_change = 0.0
        goal_base_rate = 0.045
        if self.internal_state_parameters['preferred_logical_state'] is not None:
            if mod_valence > 0.35: goal_bias_change += goal_base_rate
            else: goal_bias_change -=goal_base_rate*0.6
        else:
            goal_bias_change -= goal_base_rate*0.3
        self.internal_state_parameters['goal_seeking_bias'] = np.clip(self.internal_state_parameters['goal_seeking_bias'] + goal_bias_change, 0.01, 0.99)

        if self.verbose >=3: print(f"    Curiosity: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        self._log_lot_event("meta.cog_param_update.end", {"cog_load":self.internal_state_parameters['cognitive_load'], "attn": self.internal_state_parameters['attention_level'], "cur":self.internal_state_parameters['curiosity'], "goal_bias":self.internal_state_parameters['goal_seeking_bias']})


    def _meta_layer_adapt_preferred_state(self, collapsed_outcome_str, mod_valence):
        high_val_thresh = self.metacognition_params['high_valence_threshold']
        low_val_thresh = self.metacognition_params['low_valence_threshold']
        current_pref_state = self.internal_state_parameters['preferred_logical_state']
        pref_state_log_msg = ""

        if mod_valence >= high_val_thresh and current_pref_state != collapsed_outcome_str:
            self.internal_state_parameters['preferred_logical_state'] = collapsed_outcome_str
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.28)
            self.internal_state_parameters['frustration'] *= 0.55
            pref_state_log_msg = f"New preferred state |{collapsed_outcome_str}> set due to high valence ({mod_valence:.2f}). Goal bias up, frustration down."
        elif mod_valence <= low_val_thresh and current_pref_state == collapsed_outcome_str:
            self.internal_state_parameters['preferred_logical_state'] = None
            self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.22)
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.18)
            pref_state_log_msg = f"Preferred state |{collapsed_outcome_str}> cleared due to low valence ({mod_valence:.2f}). Goal bias down, curiosity up."
        elif current_pref_state == collapsed_outcome_str and low_val_thresh < mod_valence < (high_val_thresh * 0.5) and random.random() < 0.15:
            self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] * 0.9)
            pref_state_log_msg = f"Preferred state |{collapsed_outcome_str}> yielding mediocre results ({mod_valence:.2f}), slightly reduced goal_seeking_bias."
            if self.internal_state_parameters['goal_seeking_bias'] < 0.1:
                self.internal_state_parameters['preferred_logical_state'] = None
                pref_state_log_msg += " Preferred state cleared due to very low bias."

        if pref_state_log_msg:
            if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState: {pref_state_log_msg}")
            self._log_lot_event("meta.adapt_pref_state", {"message": pref_state_log_msg, "new_pref_state_str": str(self.internal_state_parameters['preferred_logical_state']), "mod_valence": mod_valence})


    def _meta_layer_perform_review(self):
        if self.verbose >= 1: print(f"[{self.agent_id}] --- META_LAYER.Review (Cycle {self.current_cycle_num}) ---")
        self._log_lot_event("meta.review.start", {"cycle": self.current_cycle_num, "review_interval": self.metacognition_params['review_interval']})

        history_span_for_review = min(len(self.cycle_history), self.metacognition_params['review_interval'] * 3)
        if history_span_for_review < self.metacognition_params['review_interval'] * 0.6 :
            if self.verbose >= 1: print(f"    META.Review: Insufficient history ({history_span_for_review} cycles) for meaningful review. Min required approx {self.metacognition_params['review_interval'] * 0.6:.0f}.")
            self._log_lot_event("meta.review.insufficient_history", {"history_len": history_span_for_review})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        recent_history_full = list(self.cycle_history)
        start_idx = max(0, len(recent_history_full) - history_span_for_review)
        recent_history_slice = recent_history_full[start_idx:]

        valid_cycles_for_review = [
            c for c in recent_history_slice if
            c.get('collapsed_to') != "N/A" and
            c.get('orp_at_collapse') is not None and c.get('orp_at_collapse') >= 0 and
            c.get('valence_mod_this_cycle') is not None and
            c.get('entropy_at_collapse') is not None and c.get('entropy_at_collapse') >= 0 and
            c.get('num_ops_executed') is not None and c.get('num_ops_executed') >= 0
        ]

        if not valid_cycles_for_review or len(valid_cycles_for_review) < self.metacognition_params['review_interval'] * 0.4:
            if self.verbose >= 1: print(f"    META.Review: Insufficient VALID cycles ({len(valid_cycles_for_review)}) in recent history for review. Min required approx {self.metacognition_params['review_interval'] * 0.4:.0f}.")
            self._log_lot_event("meta.review.no_valid_cycles", {"valid_cycles_count": len(valid_cycles_for_review)})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        avg_valence = np.mean([c['valence_mod_this_cycle'] for c in valid_cycles_for_review])
        avg_orp_at_collapse = np.mean([c['orp_at_collapse'] for c in valid_cycles_for_review])
        avg_entropy = np.mean([c['entropy_at_collapse'] for c in valid_cycles_for_review])
        avg_ops_per_cycle = np.mean([c['num_ops_executed'] for c in valid_cycles_for_review])
        outcome_diversity = len(set(c['collapsed_to'] for c in valid_cycles_for_review)) / len(valid_cycles_for_review) if valid_cycles_for_review else 0.0

        avg_metrics = {
            'avg_valence':avg_valence, 'avg_orp_at_collapse':avg_orp_at_collapse,
            'avg_entropy':avg_entropy, 'avg_ops_per_cycle':avg_ops_per_cycle,
            'outcome_diversity':outcome_diversity, 'num_valid_cycles_reviewed':len(valid_cycles_for_review)
        }
        if self.verbose >= 2: print(f"    META.Review Stats (over {len(valid_cycles_for_review)} cycles): AvgVal={avg_valence:.2f}, AvgORP={avg_orp_at_collapse:.3f}, AvgEnt={avg_entropy:.2f}, AvgOps={avg_ops_per_cycle:.1f}, Diversity={outcome_diversity:.2f}")
        self._log_lot_event("meta.review.stats", avg_metrics)

        cur_adapt_rate = self.metacognition_params.get('curiosity_adaptation_rate', DEFAULT_METACOGNITION_PARAMS['curiosity_adaptation_rate'])
        prev_cur = self.internal_state_parameters['curiosity']
        if avg_valence < self.metacognition_params['low_valence_threshold'] or \
           outcome_diversity < self.metacognition_params['exploration_threshold_entropy']:
            self.internal_state_parameters['curiosity'] = min(0.99, prev_cur + cur_adapt_rate)
            if self.verbose >= 2: print(f"      META.Adapt: Curiosity increased {prev_cur:.2f} -> {self.internal_state_parameters['curiosity']:.2f}")
        elif avg_valence > self.metacognition_params['high_valence_threshold'] and outcome_diversity > self.metacognition_params['exploration_threshold_entropy'] * 1.5:
            self.internal_state_parameters['curiosity'] = max(0.01, prev_cur - cur_adapt_rate * 0.6)
            if self.verbose >= 2: print(f"      META.Adapt: Curiosity reduced {prev_cur:.2f} -> {self.internal_state_parameters['curiosity']:.2f}")

        goal_adapt_rate = self.metacognition_params.get('goal_bias_adaptation_rate', DEFAULT_METACOGNITION_PARAMS['goal_bias_adaptation_rate'])
        prev_gb = self.internal_state_parameters['goal_seeking_bias']
        if avg_valence > self.metacognition_params['high_valence_threshold'] and (self.internal_state_parameters['preferred_logical_state'] is not None or (self.current_goal_state_obj and self.current_goal_state_obj.status == "active")):
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, prev_gb + goal_adapt_rate)
            if self.verbose >= 2: print(f"      META.Adapt: GoalBias increased {prev_gb:.2f} -> {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        elif avg_valence < self.metacognition_params['low_valence_threshold']:
            self.internal_state_parameters['goal_seeking_bias'] = max(0.01, prev_gb - goal_adapt_rate * 1.2)
            if self.verbose >= 2: print(f"      META.Adapt: GoalBias reduced {prev_gb:.2f} -> {self.internal_state_parameters['goal_seeking_bias']:.2f}")

        if self.metacognition_params.get('enable_threshold_adaptation', False):
            td = self.orp_threshold_dynamics; prev_eor = self.E_OR_THRESHOLD
            adapt_rate_thresh = td.get('adapt_rate', DEFAULT_ORP_THRESHOLD_DYNAMICS['adapt_rate'])
            if avg_orp_at_collapse < self.E_OR_THRESHOLD * 0.55 and avg_valence < 0.05:
                self.E_OR_THRESHOLD = min(td['max'], self.E_OR_THRESHOLD + adapt_rate_thresh * 1.1)
                if self.verbose >= 2: print(f"      META.Adapt: E_OR_THRESH increased {prev_eor:.3f} -> {self.E_OR_THRESHOLD:.3f} (low ORP/valence)")
            elif avg_orp_at_collapse > self.E_OR_THRESHOLD * 1.45 or \
                 (avg_ops_per_cycle > self.internal_state_parameters['computation_length_preference'] * 1.6 and avg_valence < 0.15):
                self.E_OR_THRESHOLD = max(td['min'], self.E_OR_THRESHOLD - adapt_rate_thresh)
                if self.verbose >= 2: print(f"      META.Adapt: E_OR_THRESH decreased {prev_eor:.3f} -> {self.E_OR_THRESHOLD:.3f} (high ORP or inefficient ops)")

        if self.metacognition_params.get('enable_decay_adaptation', False):
            dd = self.orp_decay_dynamics; prev_decay = self.orp_decay_rate
            adapt_rate_decay = dd.get('adapt_rate', DEFAULT_ORP_DECAY_DYNAMICS['adapt_rate'])
            if avg_valence < self.metacognition_params['low_valence_threshold'] * 0.85:
                self.orp_decay_rate = max(dd['min'], self.orp_decay_rate - adapt_rate_decay)
                if self.verbose >= 2: print(f"      META.Adapt: ORP_DECAY decreased {prev_decay:.4f} -> {self.orp_decay_rate:.4f}")
            elif avg_valence > self.metacognition_params['high_valence_threshold'] * 0.85:
                self.orp_decay_rate = min(dd['max'], self.orp_decay_rate + adapt_rate_decay * 0.4)
                if self.verbose >= 2: print(f"      META.Adapt: ORP_DECAY increased {prev_decay:.4f} -> {self.orp_decay_rate:.4f}")

        if self.metacognition_params.get('enable_compref_adaptation', False):
            prev_clp = self.internal_state_parameters['computation_length_preference']
            if avg_ops_per_cycle < prev_clp * 0.65 and avg_valence < -0.05:
                self.internal_state_parameters['computation_length_preference'] = min(10, prev_clp + 1)
                if self.verbose >= 2: print(f"      META.Adapt: COMP_LENGTH_PREF increased {prev_clp} -> {self.internal_state_parameters['computation_length_preference']}")
            elif avg_ops_per_cycle > prev_clp * 1.35 and avg_valence < -0.05:
                self.internal_state_parameters['computation_length_preference'] = max(1, prev_clp - 1)
                if self.verbose >= 2: print(f"      META.Adapt: COMP_LENGTH_PREF decreased {prev_clp} -> {self.internal_state_parameters['computation_length_preference']}")

        self.metacognition_params['cycles_since_last_review'] = 0
        if self.verbose >= 1: print(f"[{self.agent_id}] --- Metacognitive Review Complete ---")
        self._log_lot_event("meta.review.end", {"new_cur": self.internal_state_parameters['curiosity'], "new_gb":self.internal_state_parameters['goal_seeking_bias'], "new_eor":self.E_OR_THRESHOLD, "new_decay":self.orp_decay_rate, "new_clp":self.internal_state_parameters['computation_length_preference']})


    # --- 🧬 Feature 3: Synaptic Mutation Network (SMN) Methods (Enhanced Graph Version) ---
    def _initialize_smn_graph_structures(self):
        """Initializes SMN graph-related structures: param indices, influence matrix."""
        self.smn_param_indices = {name: i for i, name in enumerate(self.smn_controlled_params_definitions.keys())}
        self.smn_param_names_from_indices = {i: name for name, i in self.smn_param_indices.items()}

        num_smn_params = len(self.smn_controlled_params_definitions)
        self.smn_config['smn_influence_matrix_size'] = num_smn_params # Store for reference

        # Initialize runtime state for each controlled parameter
        self.smn_params_runtime_state = {}
        for smn_key, definition in self.smn_controlled_params_definitions.items():
            self.smn_params_runtime_state[smn_key] = {
                'current_mutation_strength': definition['base_mutation_strength'], # Starts at base, then adapts
                'base_mutation_strength_ref': definition['base_mutation_strength'], # Store for reference scaling
                'min_val': definition['min_val'],
                'max_val': definition['max_val'],
                'is_int': definition.get('is_int', False),
                'path': definition['path']
            }

        if num_smn_params > 0 and self.smn_config.get('enable_influence_matrix', False):
            initial_stddev = self.smn_config.get('smn_influence_matrix_initial_stddev', 0.05)
            self.smn_influence_matrix = np.random.normal(loc=0.0, scale=initial_stddev, size=(num_smn_params, num_smn_params))
            np.fill_diagonal(self.smn_influence_matrix, 0) # No self-propagation initially, can be learned
        else:
            self.smn_influence_matrix = np.array([]) # Empty if not enabled or no params

        self.smn_param_actual_changes_this_cycle = {}
        if self.verbose >=2 and self.smn_config.get('enable_influence_matrix', False) and num_smn_params > 0:
            print(f"    SMN Graph Structures Initialized: {num_smn_params} params. Influence Matrix shape: {self.smn_influence_matrix.shape}")

    def _smn_get_param_value(self, path_tuple):
        """Helper to get a parameter's value using its path tuple."""
        try:
            target_obj = self
            # Path: ('dict_name', 'key', 'subkey') or ('attr_name',)
            if len(path_tuple) == 1: # Direct attribute
                return getattr(target_obj, path_tuple[0])

            # Nested dictionary structure
            # path_tuple[0] must be an attribute of self that is a dict
            current_dict_or_obj = getattr(target_obj, path_tuple[0])
            for key_part_idx in range(1, len(path_tuple) -1): # Iterate up to second to last
                current_dict_or_obj = current_dict_or_obj[path_tuple[key_part_idx]]
            return current_dict_or_obj[path_tuple[-1]] # Get from final key
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            if self.verbose >= 1: print(f"    SMN_GET_PARAM_ERROR: Failed to get param at path {path_tuple}: {e}")
            self._log_lot_event("smn.error.get_param", {"path_str":str(path_tuple), "error":str(e)})
            # Fallback might be needed, or ensure paths are always valid via config.
            # For now, return a default that's unlikely to cause issues (e.g. 0 or None if appropriate)
            # However, this could mask problems. For robustness in SMN, path errors should be rare.
            # Attempting to find a default based on typical param types in SMN_CONTROLLED_PARAMS
            param_key_smn = next((k for k, v in self.smn_controlled_params_definitions.items() if v['path'] == path_tuple), None)
            if param_key_smn: return self.smn_controlled_params_definitions[param_key_smn].get('min_val', 0) # default to min_val
            return 0 # Generic fallback

    def _smn_set_param_value(self, path_tuple, value):
        """Helper to set a parameter's value using its path tuple."""
        try:
            target_obj = self
            if len(path_tuple) == 1:
                setattr(target_obj, path_tuple[0], value)
                return True

            current_dict_or_obj = getattr(target_obj, path_tuple[0])
            for key_part_idx in range(1, len(path_tuple) - 1):
                current_dict_or_obj = current_dict_or_obj[path_tuple[key_part_idx]]
            current_dict_or_obj[path_tuple[-1]] = value
            return True
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            if self.verbose >= 1: print(f"    SMN_SET_PARAM_ERROR: Failed to set param at path {path_tuple} to {value}: {e}")
            self._log_lot_event("smn.error.set_param", {"path_str":str(path_tuple), "value_set":value, "error":str(e)})
            return False


    def _smn_update_and_apply_mutations(self, valence_mod_this_cycle, valence_raw_this_cycle, prev_cycle_valence_mod, orp_at_collapse):
        if not self.smn_config.get('enabled', False) or not self.smn_param_indices: return # Not enabled or no params to control

        valence_gain = valence_mod_this_cycle - prev_cycle_valence_mod
        smn_pos_thresh = self.internal_state_parameters['smn_positive_valence_threshold']
        smn_neg_thresh = self.internal_state_parameters['smn_negative_valence_threshold']

        if self.verbose >= 2: print(f"  SMN Update & Mutate: ValenceMod={valence_mod_this_cycle:.2f}, PrevModVal={prev_cycle_valence_mod:.2f}, Gain={valence_gain:.2f}, ORP={orp_at_collapse:.3f}")
        self._log_lot_event("smn.update.start", {"val_mod_curr":valence_mod_this_cycle, "val_mod_prev": prev_cycle_valence_mod, "val_gain":valence_gain, "orp_col":orp_at_collapse})

        self.smn_param_actual_changes_this_cycle.clear() # Reset for current cycle calculations
        any_strategy_weights_mutated = False

        # --- Part 1: Primary mutations and updating individual mutation strengths ---
        primary_mutations_info = {} # Stores {param_smn_key: {'change': delta, 'original_perturb': raw_perturb}}

        for param_smn_key, runtime_state_info in self.smn_params_runtime_state.items():
            current_param_strength = runtime_state_info['current_mutation_strength']

            if valence_mod_this_cycle > smn_pos_thresh :
                current_param_strength *= self.internal_state_parameters['smn_mutation_strength_decay']
            elif valence_mod_this_cycle < smn_neg_thresh :
                current_param_strength *= self.internal_state_parameters['smn_mutation_strength_grow']
            runtime_state_info['current_mutation_strength'] = np.clip(current_param_strength, 0.0001, 0.8) # Wider bounds for individual param strength

            # Check for primary mutation trigger for this parameter
            # Mutate if gain is good AND overall valence is positive enough (don't reinforce slight improvements from terrible states)
            if valence_gain > self.smn_config.get('mutation_trigger_min_valence_gain', 0.1) and \
               valence_mod_this_cycle > (smn_pos_thresh * 0.2): # Small positive valence needed at least

                param_path = runtime_state_info['path']
                current_val = self._smn_get_param_value(param_path)

                # Primary perturbation based on this param's adaptive strength and a global scale factor
                perturbation = np.random.normal(0,
                                                runtime_state_info['current_mutation_strength'] * \
                                                self.internal_state_parameters['smn_perturbation_scale_factor'])

                new_val = current_val + perturbation
                if runtime_state_info['is_int']: new_val = int(round(new_val))
                new_val = np.clip(new_val, runtime_state_info['min_val'], runtime_state_info['max_val'])

                actual_change = new_val - current_val
                if abs(actual_change) > 1e-7: # If change actually happened after clipping etc.
                    if self._smn_set_param_value(param_path, new_val):
                        self.smn_param_actual_changes_this_cycle[param_smn_key] = actual_change
                        primary_mutations_info[param_smn_key] = {'change': actual_change, 'original_perturb': perturbation}
                        if self.verbose >= 2:
                            print(f"    SMN Primary Mutation: {param_smn_key} ('{'.'.join(str(p) for p in param_path)}') {current_val:.4f} -> {new_val:.4f} (strength:{runtime_state_info['current_mutation_strength']:.4f})")
                        self._log_lot_event("smn.update.mutation_applied", {"param_smn_key":param_smn_key, "path_str":str(param_path), "old_val":current_val, "new_val":new_val, "change":actual_change, "type":"primary"})
                        if param_path[0] == 'internal_state_parameters' and param_path[1] == 'strategy_weights':
                            any_strategy_weights_mutated = True


        # --- Part 2: Propagated mutations based on influence matrix (if enabled and primary mutations occurred) ---
        if self.smn_config.get('enable_influence_matrix', False) and primary_mutations_info and self.smn_influence_matrix.size > 0:
            propagated_perturb_accumulator = collections.defaultdict(float) # Accumulates influences on each target

            for source_param_smn_key, primary_mutation_data in primary_mutations_info.items():
                source_idx = self.smn_param_indices[source_param_smn_key]
                source_runtime_state = self.smn_params_runtime_state[source_param_smn_key]
                # Normalize primary perturbation relative to its reference scale to get a 'unit' of change
                source_ref_scale = source_runtime_state.get('base_mutation_strength_ref', 0.1) + 1e-6
                normalized_primary_perturb = primary_mutation_data['original_perturb'] / source_ref_scale

                for target_param_smn_key, target_idx in self.smn_param_indices.items():
                    if source_idx == target_idx: continue # No self-propagation via this mechanism now

                    influence_weight = self.smn_influence_matrix[source_idx, target_idx]
                    if abs(influence_weight) > self.smn_config['smn_influence_propagation_threshold']:
                        target_runtime_state = self.smn_params_runtime_state[target_param_smn_key]
                        target_ref_scale = target_runtime_state.get('base_mutation_strength_ref', 0.1)

                        # Calculate propagated perturbation for the target
                        propagated_perturb_on_target = influence_weight * \
                                                       normalized_primary_perturb * \
                                                       target_ref_scale * \
                                                       self.smn_config['smn_secondary_mutation_scale_factor']

                        propagated_perturb_accumulator[target_param_smn_key] += propagated_perturb_on_target
                        self._log_lot_event("smn_graph_propagation.attempt", {
                            "from":source_param_smn_key, "to":target_param_smn_key,
                            "weight":influence_weight, "prop_perturb":propagated_perturb_on_target
                        })

            # Apply accumulated propagated changes
            for target_param_smn_key, total_propagated_perturb in propagated_perturb_accumulator.items():
                if abs(total_propagated_perturb) < 1e-7: continue

                runtime_state_info = self.smn_params_runtime_state[target_param_smn_key]
                param_path = runtime_state_info['path']
                current_target_val = self._smn_get_param_value(param_path)

                new_target_val = current_target_val + total_propagated_perturb
                if runtime_state_info['is_int']: new_target_val = int(round(new_target_val))
                new_target_val = np.clip(new_target_val, runtime_state_info['min_val'], runtime_state_info['max_val'])

                actual_propagated_change = new_target_val - current_target_val
                if abs(actual_propagated_change) > 1e-7:
                    if self._smn_set_param_value(param_path, new_target_val):
                        # Add to smn_param_actual_changes_this_cycle, summing if primary mutation also occurred
                        self.smn_param_actual_changes_this_cycle[target_param_smn_key] = \
                            self.smn_param_actual_changes_this_cycle.get(target_param_smn_key, 0.0) + actual_propagated_change
                        if self.verbose >= 2:
                            print(f"    SMN Propagated Mutation: {target_param_smn_key} {current_target_val:.4f} -> {new_target_val:.4f} (total_prop_perturb:{total_propagated_perturb:.4f})")
                        self._log_lot_event("smn.update.mutation_applied", {"param_smn_key":target_param_smn_key, "old_val":current_target_val, "new_val":new_target_val, "change":actual_propagated_change, "type":"propagated"})
                        if param_path[0] == 'internal_state_parameters' and param_path[1] == 'strategy_weights':
                             any_strategy_weights_mutated = True


        # --- Part 3: Update smn_influence_matrix (Hebbian-like) if enabled ---
        if self.smn_config.get('enable_influence_matrix', False) and self.smn_influence_matrix.size > 0:
            hebbian_lr = self.smn_config['smn_influence_matrix_hebbian_learning_rate']
            effective_outcome_for_hebbian = 0.0

            # "Positive ORP" is interpreted as positive valence + significant computation
            min_orp_for_hebbian = self.E_OR_THRESHOLD * self.smn_config['smn_hebbian_orp_threshold_factor']

            if valence_mod_this_cycle > smn_pos_thresh and orp_at_collapse >= min_orp_for_hebbian:
                effective_outcome_for_hebbian = 1.0 # Reinforce connections
            elif valence_mod_this_cycle < smn_neg_thresh and orp_at_collapse >= min_orp_for_hebbian:
                effective_outcome_for_hebbian = -1.0 # Weaken/Anti-reinforce connections

            if abs(effective_outcome_for_hebbian) > 0 and self.smn_param_actual_changes_this_cycle:
                changed_param_keys_list = list(self.smn_param_actual_changes_this_cycle.keys())

                for i in range(len(changed_param_keys_list)):
                    for j in range(i, len(changed_param_keys_list)): # Iterate over unique pairs (P_A, P_B), including self A==B
                        param_key_A = changed_param_keys_list[i]
                        param_key_B = changed_param_keys_list[j]
                        idx_A = self.smn_param_indices[param_key_A]
                        idx_B = self.smn_param_indices[param_key_B]

                        change_A = self.smn_param_actual_changes_this_cycle[param_key_A]
                        change_B = self.smn_param_actual_changes_this_cycle[param_key_B]

                        # Normalize changes relative to their typical mutation scale (base strength serves as a proxy)
                        # Using tanh for a bounded normalization around 0.
                        scale_A = self.smn_params_runtime_state[param_key_A]['base_mutation_strength_ref'] + 1e-6
                        scale_B = self.smn_params_runtime_state[param_key_B]['base_mutation_strength_ref'] + 1e-6
                        norm_change_A = np.tanh(change_A / (scale_A * 2.0)) # Scaled by 2*base strength
                        norm_change_B = np.tanh(change_B / (scale_B * 2.0))

                        correlation_term = norm_change_A * norm_change_B
                        delta_weight = effective_outcome_for_hebbian * hebbian_lr * correlation_term

                        if abs(delta_weight) > 1e-7:
                            current_w_val = self.smn_influence_matrix[idx_A, idx_B] # For logging
                            if idx_A == idx_B: # Diagonal elements (self-influence strength)
                                self.smn_influence_matrix[idx_A, idx_A] += delta_weight
                            else: # Off-diagonal, assume symmetric update for now
                                self.smn_influence_matrix[idx_A, idx_B] += delta_weight
                                self.smn_influence_matrix[idx_B, idx_A] += delta_weight

                            self._log_lot_event("smn_graph_hebbian.update", {
                                "pA":param_key_A, "pB":param_key_B, "chA":change_A, "chB":change_B,
                                "corr":correlation_term, "eff_out":effective_outcome_for_hebbian, "dW":delta_weight,
                                "old_w": current_w_val, "new_w":self.smn_influence_matrix[idx_A, idx_B]
                            })

            # Apply global decay to influence matrix weights and clip them
            self.smn_influence_matrix *= (1.0 - self.smn_config['smn_influence_matrix_weight_decay'])
            np.clip(self.smn_influence_matrix,
                    self.smn_config['smn_influence_matrix_clip_min'],
                    self.smn_config['smn_influence_matrix_clip_max'],
                    out=self.smn_influence_matrix)

        # If SMN (primary or propagated) mutated any part of strategy_weights, they need re-normalization
        if any_strategy_weights_mutated:
            sw = self.internal_state_parameters['strategy_weights']
            valid_sw_values = [v for v in sw.values() if isinstance(v, (int, float))]
            total_sw = sum(v for v in valid_sw_values if v > 0)

            if total_sw > 1e-6 :
                for k_sw in sw:
                    if isinstance(sw[k_sw], (int,float)) : sw[k_sw] = max(0, sw[k_sw]/total_sw)
                if self.verbose >= 3: print(f"      SMN: Re-Normalized strategy_weights: { {k: f'{v:.2f}' for k,v in sw.items()} }")
            else: # Indentation of this block fixed
                if self.verbose >=1 : print(f"      SMN Warning: strategy_weights sum to near zero/negative after mutation. Resetting to uniform.")
                num_strats = len([k for k in sw if isinstance(sw[k], (int,float))]) if sw else 1
                uniform_weight = 1.0 / num_strats if num_strats > 0 else 1.0
                for k_sw in sw :
                    if isinstance(sw[k_sw], (int,float)) : sw[k_sw] = uniform_weight


    # --- Feature 6: Cognitive Firewall ---
    def _firewall_detect_and_correct_anomalies(self):
        if not self.firewall_params.get('enabled', False): return
        if self.firewall_cooldown_remaining > 0:
            self.firewall_cooldown_remaining -= 1
            self._log_lot_event("firewall.cooldown", {"remaining": self.firewall_cooldown_remaining})
            return

        self.firewall_cycles_since_last_check +=1
        if self.firewall_cycles_since_last_check < self.firewall_params['check_interval']:
            return

        self.firewall_cycles_since_last_check = 0
        intervention_made = False
        intervention_reason = "None"
        intervention_details = {}

        low_val_streak_needed = self.firewall_params['low_valence_streak_needed']
        if not intervention_made and len(self.cycle_history) >= low_val_streak_needed:
            recent_valences = [c['valence_mod_this_cycle'] for c in list(self.cycle_history)[-low_val_streak_needed:]]
            if all(v < self.firewall_params['low_valence_threshold'] for v in recent_valences):
                intervention_reason = f"Persistent Low Valence (avg {np.mean(recent_valences):.2f} < {self.firewall_params['low_valence_threshold']} for {low_val_streak_needed} cycles)"
                self.internal_state_parameters['exploration_mode_countdown'] = max(
                    self.internal_state_parameters['exploration_mode_countdown'],
                    self.firewall_params['intervention_exploration_boost_duration']
                )
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.33)
                intervention_made = True; intervention_details = {'avg_valence': np.mean(recent_valences), 'streak': low_val_streak_needed}

        loop_window = self.firewall_params['loop_detection_window']
        if not intervention_made and len(self.cycle_history) >= loop_window:
            history_slice = list(self.cycle_history)[-loop_window:]
            behavior_patterns = [(c['collapsed_to'], c['op_strategy']) for c in history_slice]
            counts = collections.Counter(behavior_patterns)
            for pattern_tuple, count in counts.items():
                if count >= self.firewall_params['loop_detection_min_repeats']:
                    loop_valences = [c['valence_mod_this_cycle'] for c, p_tuple in zip(history_slice, behavior_patterns) if p_tuple == pattern_tuple]
                    if np.mean(loop_valences) < self.firewall_params['low_valence_threshold'] * 0.75 :
                        intervention_reason = f"Behavioral Loop Detected (pattern {pattern_tuple} repeated {count}x with low avg_val {np.mean(loop_valences):.2f})"
                        sw = self.internal_state_parameters['strategy_weights']
                        rand_factor = self.firewall_params['intervention_strategy_randomness_factor']
                        for k in sw: sw[k] = sw[k] * (1-rand_factor) + random.random() * rand_factor * (1 + sw[k])

                        valid_sw_values = [v for v in sw.values() if isinstance(v, (int, float))]
                        total_sw = sum(v for v in valid_sw_values if v > 0)
                        if total_sw > 1e-6: [ sw.update({k:max(0,sw[k]/total_sw) if isinstance(sw[k],(int,float)) else sw[k] }) for k in sw.keys() ]
                        else: [ sw.update({k:1.0/len(valid_sw_values) if valid_sw_values else 1.0}) for k in sw.keys() if isinstance(sw[k], (int,float)) ]


                        self.internal_state_parameters['preferred_logical_state'] = None
                        intervention_made = True; intervention_details = {'pattern':str(pattern_tuple), 'count':count, 'avg_loop_val':np.mean(loop_valences)}
                        break

        prem_collapse_streak = self.firewall_params['premature_collapse_streak_needed']
        if not intervention_made and len(self.cycle_history) >= prem_collapse_streak:
            recent_collapse_data = list(self.cycle_history)[-prem_collapse_streak:]
            threshold_ratios = [c['orp_at_collapse'] / (c['E_OR_thresh_this_cycle']+1e-6) for c in recent_collapse_data if c.get('num_ops_executed',0) > 0]
            if threshold_ratios and all(ratio < self.firewall_params['premature_collapse_orp_max_ratio'] for ratio in threshold_ratios) and len(threshold_ratios) >= prem_collapse_streak *0.75:
                 intervention_reason = f"Persistent Premature ORP Collapse (avg ratio {np.mean(threshold_ratios):.2f} < {self.firewall_params['premature_collapse_orp_max_ratio']} for {len(threshold_ratios)} op-cycles)"
                 self.E_OR_THRESHOLD *= self.firewall_params['intervention_orp_threshold_increase_factor']
                 self.E_OR_THRESHOLD = min(self.E_OR_THRESHOLD, self.orp_threshold_dynamics['max'])
                 self.internal_state_parameters['computation_length_preference'] = min(8, max(self.internal_state_parameters['computation_length_preference'] + 1, 2))
                 intervention_made = True; intervention_details = {'avg_orp_ratio':np.mean(threshold_ratios), 'new_EOR':self.E_OR_THRESHOLD, 'new_comp_pref': self.internal_state_parameters['computation_length_preference']}

        if intervention_made:
            if self.verbose >= 1: print(f"[{self.agent_id}] COGNITIVE FIREWALL Activated: {intervention_reason}")
            self._log_lot_event("firewall.intervention", {"reason": intervention_reason, "details_str":str(intervention_details)})
            self.firewall_cooldown_remaining = self.firewall_params['cooldown_duration']
            self.internal_state_parameters['frustration'] *= 0.4
            if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
                if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL: Current goal '{self.current_goal_state_obj.current_goal}' status changed to 'pending' due to intervention.")
                self.current_goal_state_obj.status = "pending"
                self.current_goal_state_obj.history.append({"cycle": self.current_cycle_num, "event": "firewall_interrupted_goal", "reason": intervention_reason[:50]})


    # --- Main Cognitive Cycle Orchestration ---
    def run_full_cognitive_cycle(self, intended_classical_input_str:str, computation_sequence_ops=None):
        self.current_cycle_num += 1
        self.current_cycle_lot_stream = []
        start_time = time.time()

        self._log_lot_event("cycle_start", {"cycle_num":self.current_cycle_num, "intended_input_str": intended_classical_input_str, "agent_id":self.agent_id, "current_mood":self.internal_state_parameters['mood'], "current_orp":self.objective_reduction_potential})
        if self.verbose >= 1: print(f"\n[{self.agent_id}] ===== Cycle {self.current_cycle_num} | Input: |{intended_classical_input_str}> =====")

        actual_classical_input_str = self._sensor_layer_process_input(intended_classical_input_str)
        if self.verbose >= 2: print(f"  SensorLayer Out: Actual perceived input |{actual_classical_input_str}> (intended |{intended_classical_input_str}>)")

        self._executive_prepare_superposition(actual_classical_input_str)

        executed_sequence, chosen_op_strategy, op_gen_log_details = \
            self._executive_generate_computation_sequence(ops_provided_externally=computation_sequence_ops)

        if self.verbose >= 1 and executed_sequence:
             print(f"  ExecutiveLayer OpGen ({chosen_op_strategy}): {len(executed_sequence)} ops planned = {executed_sequence if len(executed_sequence)<5 else str(executed_sequence[:4])+'...'}")
        elif self.verbose >= 1 and not executed_sequence:
             print(f"  ExecutiveLayer OpGen ({chosen_op_strategy}): No ops planned.")
        if self.verbose >=3 and op_gen_log_details:
            for line_idx,line in enumerate(op_gen_log_details): print(f"    OpGenLog[{line_idx}]: {line}")

        _, or_triggered_early = self._executive_quantum_computation_phase(executed_sequence)

        entropy_at_collapse = self._calculate_superposition_entropy()
        num_superposition_terms = len([a for a in self.logical_superposition.values() if abs(a) > 1e-9])

        collapsed_outcome_str = self._executive_trigger_objective_reduction()
        orp_at_collapse = self.current_orp_before_reset # This is the ORP value *just before* it was reset post-collapse.

        if self.verbose >= 1: print(f"  ExecutiveLayer OR: Collapsed to |{collapsed_outcome_str}> (ORP experienced: {orp_at_collapse:.3f}, Early OR: {or_triggered_early}, Entropy: {entropy_at_collapse:.2f})")

        raw_valence_of_collapse = self.outcome_valence_map.get(collapsed_outcome_str, -0.15)
        self._executive_handle_collapse_interrupts(orp_at_collapse, executed_sequence, raw_valence_of_collapse)

        executive_eval_results = self._executive_evaluate_outcome_and_update_mood(
            collapsed_outcome_str, orp_at_collapse, entropy_at_collapse, len(executed_sequence or [])
        )
        if self.verbose >= 1: print(f"  ExecutiveLayer Eval: Val(raw/mod): {self.last_cycle_valence_raw:.2f}/{self.last_cycle_valence_mod:.2f}. Mood: {self.internal_state_parameters['mood']:.2f}, Frust: {self.internal_state_parameters['frustration']:.2f}")
        if self.verbose >=3 and executive_eval_results.get('thoughts_log'):
            for line_idx,line in enumerate(executive_eval_results['thoughts_log']): print(f"    AccEvalLog[{line_idx}]: {line}")

        if self.last_cycle_valence_mod > 0.65 and self.shared_attention_foci is not None:
            self.shared_attention_foci.append({'state': collapsed_outcome_str, 'op_seq': executed_sequence,
                                               'valence': self.last_cycle_valence_mod, 'agent_id': self.agent_id,
                                               'cycle': self.current_cycle_num})
            self._log_lot_event("coagent.attention_share", {"state":collapsed_outcome_str, "valence":self.last_cycle_valence_mod, "ops_count": len(executed_sequence or [])})

        consolidation_bonus = self.smn_internal_flags.pop('ltm_consolidation_bonus_factor', 1.0)
        self._associative_layer_update_ltm(executed_sequence, self.last_cycle_valence_raw, orp_at_collapse, entropy_at_collapse, consolidation_factor=consolidation_bonus)
        if self.verbose >=2 and consolidation_bonus > 1.0 : print(f"  AssociativeLayer LTM Update applied consolidation bonus: {consolidation_bonus:.1f}")

        self._meta_layer_update_cognitive_parameters(orp_at_collapse, len(executed_sequence or []), executive_eval_results, entropy_at_collapse)
        self._meta_layer_adapt_preferred_state(collapsed_outcome_str, self.last_cycle_valence_mod)
        if self.verbose >= 1: print(f"  MetaLayer State: Attn={self.internal_state_parameters['attention_level']:.2f},Cur={self.internal_state_parameters['curiosity']:.2f},PrefS=|{self.internal_state_parameters['preferred_logical_state']}>,Load={self.internal_state_parameters['cognitive_load']:.2f}")

        prev_mod_valence_for_smn = self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0.0
        self._smn_update_and_apply_mutations(self.last_cycle_valence_mod, self.last_cycle_valence_raw, prev_mod_valence_for_smn, orp_at_collapse) # Pass ORP for Hebbian logic

        self._firewall_detect_and_correct_anomalies()

        planning_log = []
        self._executive_plan_next_target_input(collapsed_outcome_str, executive_eval_results, planning_log)
        if self.verbose >= 1: print(f"  ExecutiveLayer PlanNext: Proposing |{self.next_target_input_state}> for next cycle.")
        if self.verbose >=3 :
            for line_idx, line in enumerate(planning_log): print(f"    PlanNextLog[{line_idx}]: {line}")

        self.metacognition_params['cycles_since_last_review'] += 1
        if self.metacognition_params['cycles_since_last_review'] >= self.metacognition_params['review_interval']:
            self._meta_layer_perform_review()

        prev_cycle_mod_valence_for_tfg = self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0.0
        valence_delta = self.last_cycle_valence_mod - prev_cycle_mod_valence_for_tfg
        entropy_shift = entropy_at_collapse - self.last_cycle_entropy_for_delta
        tfg_ops_entry = tuple(tuple(op) for op in (executed_sequence or []))
        self.temporal_feedback_grid.append( (tfg_ops_entry, valence_delta, entropy_shift) )
        self.last_cycle_entropy_for_delta = entropy_at_collapse
        if self.verbose >=2 : print(f"  TemporalGrid Appended: Ops({len(tfg_ops_entry)}), ValDelta({valence_delta:.2f}), EntShift({entropy_shift:.2f}). Grid size: {len(self.temporal_feedback_grid)}.")
        self._log_lot_event("temporal_feedback_grid.update", {"ops_count": len(tfg_ops_entry), "val_delta": valence_delta, "ent_shift": entropy_shift, "grid_len":len(self.temporal_feedback_grid)})

        current_cycle_data = {
            "cycle_num":self.current_cycle_num, "agent_id": self.agent_id,
            "intended_input_state":intended_classical_input_str, "actual_input_state_used":actual_classical_input_str,
            "ops_executed":executed_sequence, "op_strategy":chosen_op_strategy, "num_ops_executed":len(executed_sequence or []),
            "collapsed_to":collapsed_outcome_str, "orp_at_collapse":orp_at_collapse, "or_triggered_early": or_triggered_early,
            "num_terms_before_collapse":num_superposition_terms, "entropy_at_collapse":entropy_at_collapse,
            "valence_raw_this_cycle":self.last_cycle_valence_raw, "valence_mod_this_cycle":self.last_cycle_valence_mod,
            "mood_after_cycle":self.internal_state_parameters['mood'], "attention_after_cycle":self.internal_state_parameters['attention_level'],
            "cog_load_after_cycle":self.internal_state_parameters['cognitive_load'], "frustration_after_cycle":self.internal_state_parameters['frustration'],
            "curiosity_after_cycle":self.internal_state_parameters['curiosity'], "goal_bias_after_cycle":self.internal_state_parameters['goal_seeking_bias'],
            "preferred_state_after_cycle":str(self.internal_state_parameters['preferred_logical_state']),
            "E_OR_thresh_this_cycle":self.E_OR_THRESHOLD, "orp_decay_this_cycle":self.orp_decay_rate,
            "exploration_mode_countdown_after_cycle": self.internal_state_parameters['exploration_mode_countdown'],
            "smn_flags_in_cycle": copy.deepcopy(self.smn_internal_flags),
            "smn_influence_matrix_sample": self.smn_influence_matrix[:2,:2].tolist() if self.smn_influence_matrix.size > 0 else "N/A", # Sample for history
            "goal_state_at_cycle_end": self.current_goal_state_obj.to_dict() if self.current_goal_state_obj else None,
            "lot_stream_this_cycle": copy.deepcopy(self.current_cycle_lot_stream)
        }
        self.cycle_history.append(current_cycle_data)

        cycle_duration = time.time() - start_time
        self._log_lot_event("cycle_end", {"duration_ms": cycle_duration * 1000, "next_planned_input_str": self.next_target_input_state, "final_mood": self.internal_state_parameters['mood']})
        if self.verbose >= 1: print(f"[{self.agent_id}] ===== Cycle {self.current_cycle_num} End (Dur: {cycle_duration:.3f}s, Next: |{self.next_target_input_state}>) Mood:{self.internal_state_parameters['mood']:.2f} =====")

        return self.next_target_input_state

    # --- Helper & Utility ---
    def logical_superposition_str(self):
        active_terms = []
        sorted_states = sorted(self.logical_superposition.keys())
        for state in sorted_states:
            amp = self.logical_superposition[state]
            if abs(amp) > 1e-9:
                term_str = ""
                real_part_str = f"{amp.real:.3f}" if abs(amp.real) > 1e-9 else ""
                imag_part_str = f"{amp.imag:+.3f}j" if abs(amp.imag) > 1e-9 else ""

                if real_part_str and imag_part_str: term_str = f"({real_part_str}{imag_part_str})"
                elif real_part_str: term_str = real_part_str
                elif imag_part_str: term_str = imag_part_str.replace("+","")
                else: term_str = "0.000"

                active_terms.append(f"{term_str}|{state}>")
        return " + ".join(active_terms) if active_terms else "ZeroSuperposition"


    def set_goal_state(self, goal_state_obj: GoalState):
        if not isinstance(goal_state_obj, GoalState) and goal_state_obj is not None:
            raise ValueError("goal_state_obj must be an instance of GoalState or None.")
        self.current_goal_state_obj = goal_state_obj
        if self.current_goal_state_obj:
            self.current_goal_state_obj.status = "active"
            if self.verbose >= 1: print(f"[{self.agent_id}] New goal set: {self.current_goal_state_obj}")
            self._log_lot_event("goal.set", {"goal_name":self.current_goal_state_obj.current_goal, "num_steps":len(self.current_goal_state_obj.steps)})
        else:
            if self.verbose >= 1: print(f"[{self.agent_id}] Goal cleared.")
            self._log_lot_event("goal.cleared", {})


    def print_internal_state_summary(self, indent="  ", custom_logger=None):
        log_func = custom_logger if callable(custom_logger) else print

        log_func(f"{indent}--- Internal State Summary for Agent {self.agent_id} (Cycle {self.current_cycle_num}) ---")
        log_func(f"{indent}  State: Mood: {self.internal_state_parameters['mood']:.2f}, Attn: {self.internal_state_parameters['attention_level']:.2f}, Load: {self.internal_state_parameters['cognitive_load']:.2f}, Frust: {self.internal_state_parameters['frustration']:.2f}")
        log_func(f"{indent}  Cognition: Cur: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}, PrefState: |{self.internal_state_parameters['preferred_logical_state']}>, CompLenPref: {self.internal_state_parameters['computation_length_preference']}")
        log_func(f"{indent}  Exploration: Countdown: {self.internal_state_parameters['exploration_mode_countdown']}")
        log_func(f"{indent}  OrchOR: E_OR_THRESH: {self.E_OR_THRESHOLD:.3f} (AdaptRate: {self.orp_threshold_dynamics['adapt_rate']:.4f}), ORP_DECAY: {self.orp_decay_rate:.4f} (AdaptRate: {self.orp_decay_dynamics['adapt_rate']:.4f})")
        sw_str = ", ".join([f"{k.upper()[0]}:{v:.2f}" for k,v in self.internal_state_parameters['strategy_weights'].items()])
        log_func(f"{indent}  StrategyWeights: {sw_str}")
        log_func(f"{indent}  MetaCog: ReviewIn: {self.metacognition_params['review_interval']-self.metacognition_params['cycles_since_last_review']}, AdaptRates(Cur/Goal): {self.metacognition_params['curiosity_adaptation_rate']:.3f}/{self.metacognition_params['goal_bias_adaptation_rate']:.3f}")
        log_func(f"{indent}  LTM: {len(self.long_term_memory)}/{self.long_term_memory_capacity} entries. UtilWeights(V/E): {self.ltm_utility_weight_valence:.2f}/{self.ltm_utility_weight_efficiency:.2f}")
        log_func(f"{indent}  TemporalGrid: {len(self.temporal_feedback_grid)}/{self.temporal_grid_params['max_len']} (FeedbackWin: {self.temporal_grid_params['feedback_window']}). BiasStr(V/E): {self.internal_state_parameters['temporal_feedback_valence_bias_strength']:.2f}/{self.internal_state_parameters['temporal_feedback_entropy_bias_strength']:.2f}")

        smn_enabled = self.smn_config.get('enabled')
        smn_graph_enabled = self.smn_config.get('enable_influence_matrix')
        if smn_enabled:
            log_func(f"{indent}  SMN Active (Graph: {'On' if smn_graph_enabled else 'Off'}). Controlled params mut_strengths:")
            for p_name_smn, p_state_info in list(self.smn_params_runtime_state.items())[:3]:
                log_func(f"{indent}    {p_name_smn}: {p_state_info['current_mutation_strength']:.4f}")
            if smn_graph_enabled and self.smn_influence_matrix.size > 0:
                log_func(f"{indent}    Influence Matrix sample (top-left 2x2, rounded):")
                matrix_sample = np.round(self.smn_influence_matrix[:2, :2], 3).tolist()
                for row_idx, row in enumerate(matrix_sample):
                     p_name_row = self.smn_param_names_from_indices.get(row_idx, f"P{row_idx}")[:7]
                     row_str = ", ".join([f"{x:.3f}" for x in row])
                     log_func(f"{indent}      {p_name_row}: [{row_str}]")

        if self.firewall_params.get('enabled'):
            log_func(f"{indent}  Firewall Active. CooldownLeft: {self.firewall_cooldown_remaining}, NextCheckIn: {max(0, self.firewall_params['check_interval'] - self.firewall_cycles_since_last_check)}")

        if self.current_goal_state_obj: log_func(f"{indent}  Current Goal: {self.current_goal_state_obj}")
        else: log_func(f"{indent}  Current Goal: None")
        log_func(f"{indent}--- End of Summary ---")

    def get_public_state_summary(self):
        return {
            "agent_id": self.agent_id, "current_cycle_num": self.current_cycle_num,
            "mood": self.internal_state_parameters['mood'], "attention": self.internal_state_parameters['attention_level'],
            "frustration": self.internal_state_parameters['frustration'], "curiosity": self.internal_state_parameters['curiosity'],
            "collapsed_state": self.collapsed_logical_state_str, "preferred_state": self.internal_state_parameters['preferred_logical_state'],
            "E_OR_THRESHOLD": self.E_OR_THRESHOLD,
            "active_goal_name": self.current_goal_state_obj.current_goal if self.current_goal_state_obj else None,
            "active_goal_progress": self.current_goal_state_obj.progress if self.current_goal_state_obj else None,
        }

    def run_chained_cognitive_cycles(self, initial_input_str, num_cycles, computation_sequence_ops_template=None):
        if self.verbose >= 0: print(f"\n\n[{self.agent_id}] %%%%% STARTING CHAINED CYCLES (Num: {num_cycles}, Init Input: |{initial_input_str}>) %%%%%")
        self.next_target_input_state = initial_input_str

        for i in range(num_cycles):
            current_input_str_for_cycle = self.next_target_input_state

            if self.verbose >= 1:
                pref_str = f"|{self.internal_state_parameters['preferred_logical_state']}>" if self.internal_state_parameters['preferred_logical_state'] else "None"
                goal_summary = "No Goal"
                if self.current_goal_state_obj:
                    step_name = "N/A"
                    if self.current_goal_state_obj.steps and 0 <= self.current_goal_state_obj.current_step_index < len(self.current_goal_state_obj.steps):
                         step_name = self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index].get('name', 'UnnamedStep')
                    goal_summary = f"Goal: '{self.current_goal_state_obj.current_goal}' Step: '{step_name}' ({self.current_goal_state_obj.status})"

                print(f"\n>>>> Chained Cycle {i+1}/{num_cycles} for {self.agent_id} <<<< Input: |{current_input_str_for_cycle}>; Mood:{self.internal_state_parameters['mood']:.2f}; Pref:{pref_str}; {goal_summary}")

            current_comp_ops = None
            if isinstance(computation_sequence_ops_template, list) and computation_sequence_ops_template:
                current_comp_ops = computation_sequence_ops_template[i % len(computation_sequence_ops_template)] if len(computation_sequence_ops_template) > 0 else None
            elif callable(computation_sequence_ops_template):
                 current_comp_ops = computation_sequence_ops_template(self, i)

            try:
                self.run_full_cognitive_cycle(current_input_str_for_cycle, current_comp_ops)

            except Exception as e:
                critical_error_msg = f"CRITICAL EXCEPTION in cycle {self.current_cycle_num} (idx {i+1}) for agent {self.agent_id}: {type(e).__name__} - {e}.AGENT STOPPING."
                print(critical_error_msg)
                traceback.print_exc()
                self._log_lot_event("error.critical_exception", {"cycle_num_runtime":i+1, "error_type":type(e).__name__, "error_msg":str(e), "traceback_str":traceback.format_exc()[:500]})
                break

        if self.verbose >= 0: print(f"\n[{self.agent_id}] %%%%% CHAINED CYCLES COMPLETED ({self.current_cycle_num} total cycles for this agent instance) %%%%%");
        if self.verbose >= 1 : self.print_internal_state_summary(indent=f"  Final State ({self.agent_id}) ")


# ---------------------------------------------------------------------------
# CoAgentManager (Feature 5: Parallel Cognitive Threads)
# ---------------------------------------------------------------------------
class CoAgentManager:
    def __init__(self, num_agents, base_emulator_config_template, agent_config_variations_list=None, verbose=0):
        self.num_agents = num_agents
        self.base_config = base_emulator_config_template
        self.agent_variations = agent_config_variations_list if agent_config_variations_list else []
        self.verbose = verbose

        self.shared_long_term_memory = {}
        self.shared_attention_foci = collections.deque(maxlen=50 * max(1, num_agents // 2))
        self.agents = []
        self.agent_performance_history = collections.defaultdict(lambda: collections.deque(maxlen=30))
        self.system_cycle_num = 0

        self._initialize_agents()
        if self.verbose >= 0: print(f"CoAgentManager Initialized with {self.num_agents} agents sharing LTM and AttentionFoci.")

    def _initialize_agents(self):
        for i in range(self.num_agents):
            agent_id = f"agent{i:02d}"
            agent_specific_config = copy.deepcopy(self.base_config)

            agent_overrides = {}
            agent_trainable_params_init = {}
            if i < len(self.agent_variations):
                agent_custom_settings = self.agent_variations[i]
                agent_overrides = agent_custom_settings.get('config_overrides', {})
                agent_trainable_params_init = agent_custom_settings.get('trainable_param_values', {})

            agent_specific_config['agent_id'] = agent_id
            agent_specific_config['shared_long_term_memory'] = self.shared_long_term_memory
            agent_specific_config['shared_attention_foci'] = self.shared_attention_foci
            agent_specific_config['config_overrides'] = agent_overrides # Pass potentially already existing overrides + new ones
            if agent_trainable_params_init:
                 agent_specific_config['trainable_param_values'] = agent_trainable_params_init

            final_agent_verbose = agent_overrides.get(('verbose',), self.verbose -1 if self.verbose > 0 else 0) # verbose can be overridden
            agent_specific_config['verbose'] = final_agent_verbose


            try:
                emulator = SimplifiedOrchOREmulator(**agent_specific_config)
                self.agents.append(emulator)
                if self.verbose >= 1: print(f"  Initialized {agent_id}. Verbose: {final_agent_verbose}.")
                if self.verbose >=2 and (agent_overrides or agent_trainable_params_init):
                    print(f"    {agent_id} Initial Overrides: {agent_overrides}")
                    print(f"    {agent_id} Initial Trainable Params: {agent_trainable_params_init}")
                    emulator.print_internal_state_summary(indent="      ")
            except Exception as e:
                print(f"CRITICAL ERROR Initializing {agent_id}: {type(e).__name__} - {e}")
                traceback.print_exc()

    def run_system_cycles(self, num_system_cycles, initial_input_per_agent_list=None):
        if self.verbose >= 0: print(f"\n\n========= CoAgentManager: Starting {num_system_cycles} System Cycles =========")

        for i_sys_cycle in range(num_system_cycles):
            self.system_cycle_num += 1
            if self.verbose >=0: print(f"\n------- System Cycle {self.system_cycle_num}/{num_system_cycles} (Manager Cycle {i_sys_cycle+1}) -------")

            agent_threads = []

            for agent_idx, agent in enumerate(self.agents):
                current_agent_input = agent.next_target_input_state
                if initial_input_per_agent_list and self.system_cycle_num == 1 and agent_idx < len(initial_input_per_agent_list):
                    current_agent_input = initial_input_per_agent_list[agent_idx]
                    agent.next_target_input_state = current_agent_input

                if self.verbose >=1: print(f"  Running {agent.agent_id} (Cycle {agent.current_cycle_num + 1}) with intended input |{current_agent_input}>")
                try:
                    agent.run_full_cognitive_cycle(current_agent_input)
                    if agent.cycle_history:
                        self.agent_performance_history[agent.agent_id].append(agent.cycle_history[-1]['valence_mod_this_cycle'])
                except Exception as e:
                     print(f"  ERROR during cognitive cycle for {agent.agent_id}: {type(e).__name__} - {e}. Agent may become unstable.")
                     traceback.print_exc()

            interaction_interval = max(3, min(10, num_system_cycles // 5))
            if self.system_cycle_num > 0 and self.system_cycle_num % interaction_interval == 0:
                self._perform_inter_agent_learning()

        if self.verbose >= 0: print(f"\n========= CoAgentManager: System Cycles Completed ({self.system_cycle_num} total) =========")
        self.print_system_summary()

    def _perform_inter_agent_learning(self):
        if len(self.agents) < 2: return

        if self.verbose >= 1: print(f"\n  --- CoAgentManager: Performing Inter-Agent Learning/Alignment (System Cycle {self.system_cycle_num}) ---")

        avg_performances = []
        for agent in self.agents:
            if self.agent_performance_history[agent.agent_id]:
                hist = list(self.agent_performance_history[agent.agent_id])
                weights = np.linspace(0.5, 1.5, len(hist)) if len(hist) > 1 else [1.0]
                weighted_avg_perf = np.average(hist, weights=weights) if hist else -float('inf')
                avg_performances.append({'agent_id': agent.agent_id, 'perf': weighted_avg_perf, 'agent_obj': agent})
            else:
                avg_performances.append({'agent_id': agent.agent_id, 'perf': -float('inf'), 'agent_obj': agent})

        if not avg_performances: return
        avg_performances.sort(key=lambda x: x['perf'], reverse=True)

        if self.verbose >= 2:
            print(f"    Agent Performance Ranking (weighted_avg_recent_valence):")
            for p_data in avg_performances: print(f"      {p_data['agent_id']}: {p_data['perf']:.3f}")

        num_learners = min(len(self.agents) // 3 + 1, 4)
        teacher_data = avg_performances[0]

        copied_count = 0
        for i in range(num_learners):
            learner_idx_from_bottom = i
            if len(avg_performances) - 1 - learner_idx_from_bottom <= 0 : break

            learner_data = avg_performances[len(avg_performances) - 1 - learner_idx_from_bottom]
            if teacher_data['agent_id'] == learner_data['agent_id']: continue

            performance_gap_threshold_abs = 0.25
            if teacher_data['perf'] > learner_data['perf'] + performance_gap_threshold_abs and learner_data['perf'] < 0.1:
                learner_agent = learner_data['agent_obj']
                teacher_agent = teacher_data['agent_obj']
                if self.verbose >= 1: print(f"    {learner_agent.agent_id} (perf {learner_data['perf']:.2f}) learning from {teacher_agent.agent_id} (perf {teacher_data['perf']:.2f})")

                params_to_align = DEFAULT_TRAINABLE_PARAMS_CONFIG.keys()
                alignment_factor = random.uniform(0.1, 0.25)

                learner_current_params_for_update = {}
                teacher_params_for_reference = {}

                for param_name in params_to_align:
                    config = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name]
                    dict_attr, key, subkey = config['target_dict_attr'], config['target_key'], config.get('target_subkey')
                    try:
                        t_obj = teacher_agent
                        if dict_attr: t_obj = getattr(t_obj, dict_attr)
                        teacher_val = t_obj[key]
                        if subkey: teacher_val = teacher_val[subkey]
                        teacher_params_for_reference[param_name] = teacher_val
                    except (AttributeError, KeyError):
                        if self.verbose >=2: print(f"      Skipping {param_name} for teacher {teacher_agent.agent_id}, not found.")
                        continue

                for param_name, teacher_val in teacher_params_for_reference.items():
                    config = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name]
                    dict_attr, key, subkey = config['target_dict_attr'], config['target_key'], config.get('target_subkey')
                    try:
                        l_obj = learner_agent
                        if dict_attr: l_obj = getattr(l_obj, dict_attr)

                        current_learner_val_container = l_obj
                        if subkey: current_learner_val = current_learner_val_container[key][subkey]
                        elif dict_attr : current_learner_val = current_learner_val_container[key]
                        else: current_learner_val = getattr(learner_agent, key)

                        nudged_val = current_learner_val * (1 - alignment_factor) + teacher_val * alignment_factor
                        nudged_val += np.random.normal(0, config['perturb_scale'] * 0.05)
                        nudged_val = np.clip(nudged_val, config['min'], config['max'])

                        learner_current_params_for_update[param_name] = nudged_val
                        copied_count +=1
                    except (AttributeError, KeyError, TypeError) as e:
                         if self.verbose >=1: print(f"      Error aligning {param_name} for {learner_agent.agent_id} (key: {key}, subkey: {subkey}): {type(e).__name__} {e}")

                if learner_current_params_for_update:
                    learner_agent.update_emulator_parameters(learner_current_params_for_update)
                    if self.verbose >=2 : print(f"      {learner_agent.agent_id} applied {len(learner_current_params_for_update)} param updates from {teacher_agent.agent_id}.")
                    learner_agent._log_lot_event("coagent.learn_from_peer", {"teacher_id": teacher_agent.agent_id, "learner_perf":learner_data['perf'], "teacher_perf":teacher_data['perf'], "num_params_aligned": len(learner_current_params_for_update)})

        if copied_count == 0 and self.verbose >=1: print("    No significant performance gaps or conditions met for agent parameter alignment this step.")
        if self.verbose >=1: print("  --- CoAgentManager: Inter-Agent Learning/Alignment Complete ---")


    def print_system_summary(self):
        print(f"\n--- CoAgentManager System Summary (After {self.system_cycle_num} system cycles) ---")
        print(f"  Number of Agents: {self.num_agents}")
        print(f"  Shared LTM Size: {len(self.shared_long_term_memory)}")
        print(f"  Shared Attention Foci Queue Size: {len(self.shared_attention_foci)} / {self.shared_attention_foci.maxlen}")
        if self.shared_attention_foci:
             last_focus = self.shared_attention_foci[-1]
             print(f"    Last shared focus by {last_focus['agent_id']}: state |{last_focus['state']}>, valence {last_focus['valence']:.2f}, cycle {last_focus['cycle']}")

        print("\n  Final Individual Agent Summaries (or current state if mid-run):")
        for agent_idx, agent in enumerate(self.agents):
            print(f"\n  Agent {agent_idx}: {agent.agent_id}")
            agent.print_internal_state_summary(indent="    ", custom_logger=print)
            if agent_idx < len(self.agents) -1: print("    " + "-" * 30)


# ---------------------------------------------------------------------------
# Cognitive Agent Trainer
# ---------------------------------------------------------------------------
class CognitiveAgentTrainer:
    def __init__(self, trainable_params_config, base_emulator_config=None, verbose=0):
        self.trainable_params_config = copy.deepcopy(trainable_params_config)
        self.current_params = {name: config['initial'] for name, config in trainable_params_config.items()}
        self.base_emulator_config = base_emulator_config if base_emulator_config else {}
        self.verbose = verbose
        self.best_params = copy.deepcopy(self.current_params)
        self.best_reward = -float('inf')
        self.training_log = []

    def _get_emulator_init_args(self, current_run_params_for_episode):
        init_args = copy.deepcopy(self.base_emulator_config)
        init_args['trainable_param_values'] = current_run_params_for_episode
        init_args['verbose'] = self.base_emulator_config.get('verbose_emulator_episodes', self.verbose - 2 if self.verbose > 1 else 0)

        base_overrides = self.base_emulator_config.get('config_overrides', {})
        init_args['config_overrides'] = {**base_overrides, **init_args.get('config_overrides',{})}

        return init_args

    def run_episode(self, episode_params_to_test, num_cycles, initial_input="00", task_goal_state_obj=None):
        emulator_kwargs = self._get_emulator_init_args(episode_params_to_test)
        emulator_kwargs['agent_id'] = emulator_kwargs.get('agent_id', "trainer_ep_agent")

        emulator = SimplifiedOrchOREmulator(**emulator_kwargs)

        # This direct setting of outcome_valence_map might become redundant if
        # it's handled correctly by config_overrides, but kept for safety unless specified otherwise
        # Update: Per prompt, this line should no longer trigger as desired due to fix.
        if 'outcome_valence_map' in self.base_emulator_config and not emulator_kwargs.get('config_overrides', {}).get(('outcome_valence_map',)):
             emulator.outcome_valence_map = copy.deepcopy(self.base_emulator_config['outcome_valence_map'])

        task_pref_state = self.base_emulator_config.get('initial_internal_states', {}).get('preferred_logical_state')
        if task_pref_state:
            emulator.internal_state_parameters['preferred_logical_state'] = task_pref_state

        if task_goal_state_obj:
            emulator.set_goal_state(task_goal_state_obj)

        emulator.run_chained_cognitive_cycles(initial_input_str=initial_input, num_cycles=num_cycles)

        if not emulator.cycle_history: return -1.0, [], None

        avg_mod_valence_episode = np.mean([c['valence_mod_this_cycle'] for c in emulator.cycle_history if c.get('valence_mod_this_cycle') is not None] or [-1.0])
        avg_mood_episode = np.mean([c['mood_after_cycle'] for c in emulator.cycle_history if c.get('mood_after_cycle') is not None] or [-1.0])

        primary_reward_metric = (avg_mod_valence_episode * 0.4 + avg_mood_episode * 0.6)
        reward = primary_reward_metric

        goal_completion_bonus = 0.0
        goal_progress_bonus = 0.0
        final_goal_status_str = "N/A"
        final_goal_progress_val = 0.0

        if task_goal_state_obj and emulator.current_goal_state_obj:
            final_goal_obj = emulator.current_goal_state_obj
            final_goal_status_str = final_goal_obj.status
            final_goal_progress_val = final_goal_obj.progress

            if final_goal_obj.status == 'completed':
                goal_completion_bonus = self.base_emulator_config.get('trainer_goal_completion_reward', 0.8)
            elif final_goal_obj.status == 'failed':
                goal_completion_bonus = self.base_emulator_config.get('trainer_goal_failure_penalty', -0.5)

            goal_progress_bonus = final_goal_obj.progress * self.base_emulator_config.get('trainer_goal_progress_reward_factor', 0.3)
            reward += goal_completion_bonus + goal_progress_bonus

        reward = np.clip(reward, -2.0, 2.0)

        episode_outcome_details = {
            "avg_valence": avg_mod_valence_episode, "avg_mood": avg_mood_episode,
            "goal_status": final_goal_status_str, "goal_progress": final_goal_progress_val,
            "final_reward_calculated": reward
        }
        return reward, emulator.cycle_history, episode_outcome_details


    def train(self, num_training_episodes, cycles_per_episode, initial_input="00",
              learning_rate_decay=0.995, training_goal_state_template_dict=None):
        if self.verbose >= 0: print(f"\n--- Starting Training ({num_training_episodes} episodes, {cycles_per_episode} cycles/ep) ---")

        current_perturb_scales = {name: config['perturb_scale'] for name, config in self.trainable_params_config.items()}

        for episode_num in range(num_training_episodes):
            candidate_params = copy.deepcopy(self.best_params)
            num_params_to_perturb = random.randint(1, max(1, len(candidate_params) // 2))
            params_to_perturb_this_ep = random.sample(list(candidate_params.keys()), num_params_to_perturb)

            for name in params_to_perturb_this_ep:
                config = self.trainable_params_config[name]
                if config.get('fixed', False): continue
                perturb = np.random.normal(0, current_perturb_scales[name])
                candidate_params[name] += perturb
                candidate_params[name] = np.clip(candidate_params[name], config['min'], config['max'])

            if self.verbose >= 1: print(f"\nEp {episode_num + 1}/{num_training_episodes} | Best Reward So Far: {self.best_reward:.4f} | Perturbing: {params_to_perturb_this_ep}")
            if self.verbose >= 2:
                print("  Candidate params for this episode:")
                for pname, pval in candidate_params.items(): print(f"    {pname}: {pval:.4f}")

            current_episode_goal_obj = None
            if training_goal_state_template_dict:
                 current_episode_goal_obj = GoalState(**copy.deepcopy(training_goal_state_template_dict))


            reward, ep_history, ep_outcome_details = self.run_episode(
                candidate_params, cycles_per_episode, initial_input,
                task_goal_state_obj=current_episode_goal_obj
            )

            episode_log_entry = {
                'episode': episode_num + 1,
                'reward_achieved': reward,
                'best_reward_at_start_of_ep': self.best_reward,
                'parameters_tried': copy.deepcopy(candidate_params),
                'outcome_details': ep_outcome_details,
                'perturb_scales_used': copy.deepcopy(current_perturb_scales)
            }

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_params = copy.deepcopy(candidate_params)
                episode_log_entry['result_action'] = "NEW_BEST_PARAMS_FOUND"
                if self.verbose >= 1:
                    print(f"  Ep {episode_num+1}: * New Best Reward: {reward:.4f} * | Goal: {ep_outcome_details['goal_status']} ({ep_outcome_details['goal_progress']*100:.0f}%)")
                    self.print_best_params(prefix=f"    NewBest Ep{episode_num+1} ")
            else:
                episode_log_entry['result_action'] = "KEPT_PREVIOUS_BEST_PARAMS"
                if self.verbose >= 1:
                    print(f"  Ep {episode_num+1}: Reward {reward:.4f}. No improvement. | Goal: {ep_outcome_details['goal_status']} ({ep_outcome_details['goal_progress']*100:.0f}%)")

            for name in current_perturb_scales:
                 if not self.trainable_params_config[name].get('fixed', False):
                    current_perturb_scales[name] *= learning_rate_decay
                    current_perturb_scales[name] = max(current_perturb_scales[name], 0.00005)

            self.training_log.append(episode_log_entry)

        if self.verbose >= 0:
            print(f"\n--- Training Complete ({num_training_episodes} episodes) ---")
            self.print_best_params(prefix="Final Best ")
            print(f"  Final best reward achieved during training: {self.best_reward:.4f}")
        return self.best_params, self.best_reward, self.training_log

    def print_best_params(self, prefix=""):
        if not self.best_params:
            if self.verbose >=1: print(f"{prefix}No best parameters recorded yet.")
            return

        if self.verbose >= 1:
            print(f"{prefix}Parameters (Associated Reward: {self.best_reward:.4f}):")
            param_details_list = []
            for name, value in self.best_params.items():
                config = self.trainable_params_config.get(name, {})
                param_display_name = name
                val_str = f"{int(value)}" if config.get('is_int') else f"{value:.5f}"
                param_details_list.append(f"{param_display_name}: {val_str}")
            num_columns = 3
            col_width = 35
            for i in range(0, len(param_details_list), num_columns):
                row_items = [item.ljust(col_width) for item in param_details_list[i:i+num_columns]]
                print(f"  {prefix}  {''.join(row_items)}")


# ---------------------------------------------------------------------------
# Main Execution Block (Demos - Significantly Adapted)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    seed_val = int(time.time() * 1000 + random.randint(0,10000)) % (2**32)
    np.random.seed(seed_val)
    random.seed(seed_val)
    print(f"Global random seed set to: {seed_val}")

    MASTER_VERBOSE_LEVEL = 1

    # --- DEMO 0: Cortical Layers & Internal Language (Feature 1 & 8) ---
    print("\n\n--- DEMO 0: Layered Architecture & LoT Logging (Features 1 & 8) ---")
    demo0_lot_config_details = {k: True for k in DEFAULT_LOT_CONFIG['log_level_details'].keys()}
    demo0_lot_config_details['op_generation'] = True
    demo0_lot_config_details['op_execution'] = True
    demo0_lot_config_details['smn_graph_propagation'] = True # Ensure these are true for LoT output
    demo0_lot_config_details['smn_graph_hebbian'] = True

    demo0_config = {
        'verbose': MASTER_VERBOSE_LEVEL, 'cycle_history_max_len': 6,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate': 0.02,
        'lot_config': {'enabled': True, 'log_level_details': demo0_lot_config_details},
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': False},
        'temporal_grid_config': {'max_len':3},
    }
    emulator_demo0 = SimplifiedOrchOREmulator(agent_id="agent_demo0_LayersLoT", **demo0_config)
    emulator_demo0.internal_state_parameters['preferred_logical_state'] = "11"
    emulator_demo0.internal_state_parameters['curiosity'] = 0.8
    print(f"Running {emulator_demo0.agent_id} for 3 cycles to demonstrate layered processing and comprehensive LoT output.")
    emulator_demo0.run_chained_cognitive_cycles("00", 3)
    if emulator_demo0.cycle_history:
        last_cycle_log = emulator_demo0.cycle_history[-1]
        print(f"\n{emulator_demo0.agent_id} LoT for last cycle (Cycle {last_cycle_log['cycle_num']}):")
        for lot_idx, lot_entry in enumerate(last_cycle_log['lot_stream_this_cycle']):
            print(f"  LoT[{lot_idx:02d}]: {lot_entry}")
    else:
        print(f"\n{emulator_demo0.agent_id} - No cycle history found for LoT demo.")
    emulator_demo0.print_internal_state_summary(indent="  [Demo0 Summary] ")


    # --- DEMO 1: Temporal Feedback Grid & Enhanced SMN Graph (Feature 2 & 3) ---
    print("\n\n--- DEMO 1: Temporal Feedback Grid & SMN Graph in Action (Features 2 & 3) ---")
    demo1_lot_config = {
        'enabled':True,
        'log_level_details':{
            'executive.opgen.temporal_bias':True, 'smn.update.mutation_applied':True,
            'smn_graph_propagation': True, 'smn_graph_hebbian':True, # Enable SMN graph LoT
            'executive.opgen.strategy_selected':True, 'cycle_start':True, 'valence_eval':True
        }
    }
    demo1_smn_general_config = { # SMN settings for responsiveness & graph dynamics
            'enabled': True,
            'mutation_trigger_min_valence_gain': 0.08,
            'enable_influence_matrix': True, # Critically, enable the graph part
            'smn_influence_matrix_hebbian_learning_rate': 0.02, # Slightly higher LR for demo
            'smn_influence_propagation_threshold': 0.1, # Lower threshold to see propagation
            'smn_secondary_mutation_scale_factor': 0.6, # Stronger propagation effect
    }
    demo1_smn_controlled_params = { # SMN controls these, forming graph nodes
             'computation_length_preference': {'base_mutation_strength': 0.5, 'min_val': 1, 'max_val': 6, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')},
             'mc_cur_adapt_rate': {'base_mutation_strength': 0.01, 'min_val':0.005,'max_val':0.15, 'path': ('metacognition_params', 'curiosity_adaptation_rate')},
             # Added for more graph interaction potential:
             'sw_curiosity': {'base_mutation_strength': 0.08, 'min_val':0.01, 'max_val':0.99, 'path': ('internal_state_parameters', 'strategy_weights', 'curiosity')},
    }

    demo1_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate':0.015,
        'temporal_grid_config': {
            'max_len': 6, 'feedback_window': 4,
            'low_valence_delta_threshold': -0.1, 'high_entropy_shift_threshold': 0.3,
        },
        'initial_internal_states': {   # <<< CHANGED HERE
            'temporal_feedback_valence_bias_strength': 0.25,
            'temporal_feedback_entropy_bias_strength': 0.15,
            'computation_length_preference': 2,
        },
        'smn_general_config': demo1_smn_general_config, # Use the detailed SMN config from above
        'smn_controlled_params_config': demo1_smn_controlled_params, # And controlled params
        'cognitive_firewall_config': {'enabled': False},
        'lot_config': demo1_lot_config,
    }
    emulator_demo1 = SimplifiedOrchOREmulator(agent_id="agent_demo1_TFG_SMN_Graph", **demo1_config)

    # Capture initial SMN-controlled values for comparison
    initial_smn_vals_d1 = {
        key: emulator_demo1._smn_get_param_value(info['path'])
        for key, info in emulator_demo1.smn_controlled_params_definitions.items()
    }
    print(f"{emulator_demo1.agent_id} starting with initial SMN values: {initial_smn_vals_d1}. SMN Graph enabled. Running 18 cycles.")
    emulator_demo1.outcome_valence_map = {"00": -0.6, "01": 0.75, "10": -0.3, "11": 0.4}
    emulator_demo1.run_chained_cognitive_cycles("00", 18) # Increased cycles to see more evolution

    print(f"\n{emulator_demo1.agent_id} Final SMN-controlled parameter values:")
    for key, init_val in initial_smn_vals_d1.items():
        final_val = emulator_demo1._smn_get_param_value(emulator_demo1.smn_controlled_params_definitions[key]['path'])
        val_format = ".0f" if emulator_demo1.smn_controlled_params_definitions[key].get('is_int') else ".3f"
        print(f"  {key}: {final_val:{val_format}} (was {init_val:{val_format}})")

    if emulator_demo1.smn_config.get('enable_influence_matrix',False) and emulator_demo1.smn_influence_matrix.size > 0 :
        print(f"{emulator_demo1.agent_id} Final SMN Influence Matrix (sample, rounded to 3dp):")
        num_p = emulator_demo1.smn_influence_matrix.shape[0]
        sample_size = min(num_p, 3) # Show up to 3x3
        for i in range(sample_size):
            row_str = ", ".join([f"{x:.3f}" for x in emulator_demo1.smn_influence_matrix[i, :sample_size]])
            param_name_i = emulator_demo1.smn_param_names_from_indices.get(i, f"P{i}")[:10].ljust(10)
            print(f"  {param_name_i}: [{row_str}]")

    emulator_demo1.print_internal_state_summary(indent="  [Demo1 Summary] ")


    # --- DEMO 2: Interrupt Handlers & Cognitive Firewall (Features 4 & 6) ---
    print("\n\n--- DEMO 2: Interrupt Handlers & Cognitive Firewall (Features 4 & 6) ---")
    demo2_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.45,
        'initial_orp_decay_rate': 0.005,
        'interrupt_handler_config': {'enabled': True, 'reactive_ltm_valence_threshold': -0.85, 'consolidation_valence_abs_threshold':0.8},
        'cognitive_firewall_config': {
            'enabled': True, 'check_interval': 3, 'cooldown_duration': 4,
            'low_valence_threshold': -0.65, 'low_valence_streak_needed': 2,
            'loop_detection_window': 5, 'loop_detection_min_repeats': 2,
            'premature_collapse_orp_max_ratio':0.3, 'premature_collapse_streak_needed':2
        },
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False}, # Also ensure enable_influence_matrix is False if smn is disabled
        # 'outcome_valence_map': {"00": -0.9, "01": -0.8, "10": -0.85, "11": 0.2}, # MOVED
         'lot_config': {'enabled': True, 'log_level_details':{'firewall.intervention':True, 'executive.interrupt_handler':True, 'cycle_start':True, 'executive.opgen.interrupt_bias.force_ltm':True}},
        'config_overrides': { # ADDED THIS
            ('outcome_valence_map',): {"00": -0.9, "01": -0.8, "10": -0.85, "11": 0.2}
        }
    }
    emulator_demo2 = SimplifiedOrchOREmulator(agent_id="agent_demo2_IntFW", **demo2_config)
    fix_it_seq_demo2 = (('H',0),('X',1),('H',1))
    emulator_demo2.long_term_memory[fix_it_seq_demo2] = {
        'count':10, 'total_valence': 8.0, 'avg_valence':0.8, 'total_orp_cost':1.0, 'avg_orp_cost':0.1,
        'total_entropy_generated':2.0, 'avg_entropy':0.2, 'utility':0.75, 'last_cycle':0
    }
    print(f"{emulator_demo2.agent_id} starting with punishing valence map. Expecting Firewall/Interrupt activity. Running 12 cycles.")
    emulator_demo2.run_chained_cognitive_cycles("00", 12)
    emulator_demo2.print_internal_state_summary(indent="  [Demo2 Summary] ")


    # --- DEMO 3: Goal-Oriented State Machine (Feature 7) ---
    print("\n\n--- DEMO 3: Goal-Oriented State Machine (Feature 7) ---")
    demo3_lot_details = {'goal_tracking': True, 'executive.opgen.strategy_selected':True, 'executive.plannext.goal_override':True, 'executive.outcome_eval.valence':True, 'cycle_start':True}
    demo3_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'lot_config': {'enabled': True, 'log_level_details': demo3_lot_details },
        'initial_internal_states': {'goal_seeking_bias': 0.6, 'computation_length_preference':4}
    }
    emulator_demo3 = SimplifiedOrchOREmulator(agent_id="agent_demo3_Goal", **demo3_config)

    goal_steps_demo3 = [
        {"name": "Reach state 01", "target_state": "01"},
        {"name": "From 01, reach state 10", "target_state": "10", "next_input_for_world":"01"},
        {"name": "From 10, reach state 11 (final)", "target_state": "11", "next_input_for_world":"10"}
    ]
    task_goal_demo3 = GoalState(current_goal="Multi-step Sequence Demo Task", steps=goal_steps_demo3, error_tolerance=0.1)
    emulator_demo3.set_goal_state(task_goal_demo3)

    print(f"{emulator_demo3.agent_id} attempting goal: '{emulator_demo3.current_goal_state_obj.current_goal}'. Running up to 25 cycles.")
    op_template_d3 = [[('H',0)], [('X',1)], [('H',1)], [('CNOT',(0,1))], []]
    emulator_demo3.run_chained_cognitive_cycles("00", 25, computation_sequence_ops_template=lambda agent, i: op_template_d3[i % len(op_template_d3)])

    print(f"\nFinal Goal Status for {emulator_demo3.agent_id}:")
    if emulator_demo3.current_goal_state_obj:
        print(f"  {emulator_demo3.current_goal_state_obj}")
        if emulator_demo3.current_goal_state_obj.history:
            print(f"  Recent Goal History (last 5):")
            for hist_entry_idx, hist_entry in enumerate(emulator_demo3.current_goal_state_obj.history[-5:]):
                 print(f"    [{hist_entry_idx}]: {hist_entry}")

    emulator_demo3.print_internal_state_summary(indent="  [Demo3 Summary] ")


    # --- DEMO 4: Co-Agent Manager (Feature 5) ---
    print("\n\n--- DEMO 4: Co-Agent Manager & Inter-Agent Learning (Feature 5) ---")
    coagent_base_conf_demo4 = {
        'cycle_history_max_len': 25,
        'verbose': MASTER_VERBOSE_LEVEL -1 if MASTER_VERBOSE_LEVEL > 0 else 0,
        'smn_general_config': {'enabled': True, 'mutation_trigger_min_valence_gain': 0.12, 'enable_influence_matrix':True},
        'cognitive_firewall_config': {'enabled': True, 'check_interval': 5, 'cooldown_duration': 7, 'low_valence_streak_needed':3},
        'temporal_grid_config': {'max_len':8, 'feedback_window':4},
        'lot_config': {'enabled': False } # Keep LoT off for co-agents to reduce console spam
    }
    coagent_variations_demo4 = [
        {'config_overrides': {('internal_state_parameters', 'curiosity'): 0.85, ('E_OR_THRESHOLD',): 0.6, ('outcome_valence_map',): {"00":0.1,"01":0.9,"10":-0.3,"11":0.3}}},
        {'config_overrides': {('internal_state_parameters', 'goal_seeking_bias'): 0.75, ('orp_decay_rate',): 0.008, ('outcome_valence_map',): {"00":-0.2,"01":0.7,"10":-0.7,"11":0.5}}},
        {'config_overrides': {('internal_state_parameters', 'strategy_weights', 'memory'): 0.7, ('E_OR_THRESHOLD',): 1.3, ('outcome_valence_map',): {"00":0.0,"01":0.3,"10":-0.9,"11":0.8}}},
    ]
    manager_demo4 = CoAgentManager(num_agents=3,
                             base_emulator_config_template=coagent_base_conf_demo4,
                             agent_config_variations_list=coagent_variations_demo4,
                             verbose=MASTER_VERBOSE_LEVEL)
    print(f"CoAgentManager demo with {manager_demo4.num_agents} agents. Running 15 system cycles. Expect inter-agent learning attempts.")
    manager_demo4.run_system_cycles(num_system_cycles=15, initial_input_per_agent_list=["00", "01", "10"])


    # --- DEMO 5: Cognitive Agent Trainer ---
    print("\n\n--- DEMO 5: Cognitive Agent Trainer ---")
    trainer_base_emulator_config_d5 = {
        'cycle_history_max_len': 12,
        'initial_E_OR_THRESHOLD': 0.9,
        'initial_orp_decay_rate': 0.025,
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': True, 'check_interval':8, 'cooldown_duration':12},
        'temporal_grid_config': {'max_len':5, 'feedback_window':3},
        # 'outcome_valence_map': {"00": -0.2, "01": 1.0, "10": -0.6, "11": 0.1}, # MOVED
        'initial_internal_states': {'preferred_logical_state': "01"},
        'verbose_emulator_episodes': MASTER_VERBOSE_LEVEL - 2 if MASTER_VERBOSE_LEVEL > 1 else 0,
        'trainer_goal_completion_reward': 1.0, 'trainer_goal_failure_penalty': -0.6, 'trainer_goal_progress_reward_factor': 0.4,
        'config_overrides': { # ADDED THIS
            ('outcome_valence_map',): {"00": -0.2, "01": 1.0, "10": -0.6, "11": 0.1}
        }
    }

    trainer_goal_template_dict_d5 = {"current_goal": "Trainer Task: Reach 01",
                                 "steps": [{"name": "Achieve state 01", "target_state": "01"}],
                                 "error_tolerance": 0.05}

    agent_trainer_d5 = CognitiveAgentTrainer(
        trainable_params_config=DEFAULT_TRAINABLE_PARAMS_CONFIG,
        base_emulator_config=trainer_base_emulator_config_d5,
        verbose=MASTER_VERBOSE_LEVEL
    )
    num_train_eps_d5 = 20
    cycles_per_ep_d5 = 10
    print(f"Trainer starting: {num_train_eps_d5} episodes, {cycles_per_ep_d5} cycles/ep. Training to achieve state '01'.")

    best_trained_params_d5, best_reward_d5, training_history_d5 = agent_trainer_d5.train(
        num_training_episodes=num_train_eps_d5,
        cycles_per_episode=cycles_per_ep_d5,
        initial_input="00",
        learning_rate_decay=0.99,
        training_goal_state_template_dict=trainer_goal_template_dict_d5
    )
    print("\n--- Trainer Demo 5 Summary ---")
    print(f"Best reward achieved after training: {best_reward_d5:.4f}")          
    agent_trainer_d5.print_best_params("Final Trained ")

    print("\n--- Running test with best parameters from Trainer Demo 5 ---")
    final_test_config_d5 = agent_trainer_d5._get_emulator_init_args(best_trained_params_d5)
    final_test_config_d5['verbose'] = MASTER_VERBOSE_LEVEL
    final_test_config_d5['agent_id'] = "agent_trained_D5"
    # SMN graph would remain off as per trainer_base_emulator_config_d5 unless explicitly enabled here
    # For this test, let's enable it with the *trained* base parameters, allowing SMN graph to function based on them
    final_test_config_d5['smn_general_config'] = {'enabled': True, 'enable_influence_matrix': True}

    trained_emulator_d5 = SimplifiedOrchOREmulator(**final_test_config_d5)
    # Since outcome_valence_map is in config_overrides now within trainer_base_emulator_config_d5,
    # and _get_emulator_init_args merges config_overrides, this direct set is likely not needed if
    # final_test_config_d5 correctly inherited it. However, to be absolutely safe, let's consider
    # if the specific instance 'trained_emulator_d5' should explicitly have it IF it wasn't deeply merged or was reset.
    # Based on current logic of _get_emulator_init_args, it should be set.
    # The _apply_config_overrides in emulator init should handle it.

    # if 'outcome_valence_map' in trainer_base_emulator_config_d5.get('config_overrides', {}): # Check if it's in the source overrides
    #      # This isn't the right way to access it now that it's a tuple key
    #      pass
    # TO-DO: COMPLETED -- ISSUE SOLVED

    task_pref_state_d5 = trainer_base_emulator_config_d5.get('initial_internal_states', {}).get('preferred_logical_state')
    if task_pref_state_d5:
         trained_emulator_d5.internal_state_parameters['preferred_logical_state'] = task_pref_state_d5
    if trainer_goal_template_dict_d5:
        trained_emulator_d5.set_goal_state(GoalState(**copy.deepcopy(trainer_goal_template_dict_d5)))

    trained_emulator_d5.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=15)

    print("\n\n--- ALL DEMOS COMPLETED ---")
