import numpy as np
import copy
import time
import random
import collections # For deque
import traceback # For printing stack traces on errors
import math # For log2

# ---------------------------------------------------------------------------
# Orch OR Emulator & System Defaults (Enhanced & Reorganized)
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
    # New: for temporal feedback grid bias strength
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
}

# Default parameters for Temporal Feedback Grid (Feature 2)
DEFAULT_TEMPORAL_GRID_PARAMS = {
    'max_len': 10, # Stores past 10 cycles
    'low_valence_delta_threshold': -0.2, # Threshold for negative valence change to react to
    'high_entropy_shift_threshold': 0.3, # Threshold for high entropy increase to react to
}

# Default parameters for Synaptic Mutation Network (SMN) (Feature 3)
DEFAULT_SMN_CONFIG = { # General SMN behavior settings
    'enabled': True,
    'mutation_trigger_min_valence_gain': 0.1, # Min positive valence change from previous cycle to trigger 'reinforcing' mutations
                                              # If just overall valence high, it's more about stability, actual mutations need 'gain'
}
# Defines which parameters are under SMN control and their specific mutation behavior
DEFAULT_SMN_CONTROLLED_PARAMS = {
    # param_name_in_trainable_config: {'base_mutation_strength': 0.1, 'min_val':0.01, 'max_val':0.99}
    'sw_curiosity':       {'base_mutation_strength': 0.05, 'min_val':0.01, 'max_val':0.99, 'path': ('internal_state_parameters', 'strategy_weights', 'curiosity')},
    'mc_cur_adapt_rate':  {'base_mutation_strength': 0.005,'min_val':0.001,'max_val':0.2,  'path': ('metacognition_params', 'curiosity_adaptation_rate')},
    'computation_length_preference': {'base_mutation_strength': 0.2, 'min_val': 1, 'max_val': 8, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')}
    # Add more parameters here like 'E_OR_THRESHOLD', 'orp_decay_rate' etc. if SMN should control them.
    # Path indicates where to find the param: (dict_attr, key, subkey (optional))
    # If param is a direct attribute, path: (None, attr_name)
}

# Default parameters for Collapse-Triggered Interrupt Handlers (Feature 4)
DEFAULT_INTERRUPT_HANDLER_CONFIG = {
    'enabled': True,
    'consolidation_valence_abs_threshold': 0.7, # Absolute valence to trigger strong consolidation
    'consolidation_orp_surprise_factor': 1.5, # If actual ORP > expected ORP * factor
    'consolidation_strength_bonus': 2.0, # Multiplier for LTM update strength
    'reactive_ltm_valence_threshold': -0.5, # Valence below which reactive LTM might trigger
    'cognitive_fork_valence_threshold': 0.75, # High valence to mark a state as interesting/preferred
    'cognitive_fork_goal_bias_boost': 0.2,
}

# Default parameters for Cognitive Firewall (Feature 6)
DEFAULT_COGNITIVE_FIREWALL_CONFIG = {
    'enabled': True,
    'check_interval': 5, # How often to run firewall checks
    'cooldown_duration': 10, # Cycles before firewall can trigger again after an intervention
    'low_valence_threshold': -0.6,
    'low_valence_streak_needed': 4, # Number of consecutive cycles with low valence
    'loop_detection_window': 7, # Check last N states for loops
    'loop_detection_min_repeats': 3, # Minimum repetitions to be considered a loop
    'premature_collapse_orp_max_ratio': 0.4, # if orp_at_collapse < E_OR_THRESHOLD * ratio
    'premature_collapse_streak_needed': 4,
    'intervention_exploration_boost_duration': 5, # Duration for forced exploration
    'intervention_orp_threshold_increase_factor': 1.2, # Temporary increase to E_OR_THRESHOLD
    'intervention_strategy_randomness_factor': 0.5 # Factor to shuffle strategy weights by
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
    'log_level_details': { # Which events to log symbols for
        'cycle_start': True, 'sensor_input': True, 'op_generation': False, 'op_execution': False,
        'collapse_event': True, 'valence_eval': True, 'ltm_update': False, 'internal_state_updates': False,
        'goal_tracking': True, 'firewall_action': True, 'smn_action': True, 'interrupt_action': True,
        'metacognitive_review': True, 'cycle_end': True
    }
}
# ---------------------------------------------------------------------------
# GoalState Structure (for Feature 7)
# ---------------------------------------------------------------------------
class GoalState:
    def __init__(self, current_goal, steps, error_tolerance=0.05, initial_progress=0.0):
        self.current_goal = current_goal # string: overall goal description
        self.steps = steps # list of dicts, e.g., [{"name": "expand polynomial", "target_state": "01", "completion_criteria": callable}, ...]
        self.progress = initial_progress # float: 0.0 to 1.0
        self.error_tolerance = error_tolerance # float
        self.current_step_index = 0
        self.status = "pending" # pending, active, completed, failed
        self.history = [] # Log of actions related to this goal

    def to_dict(self):
        return {
            "current_goal": self.current_goal,
            "steps": self.steps, # Steps might contain non-serializable elements like callables if not careful
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
                 trainable_param_values=None, # From trainer or co-agent manager
                 temporal_grid_config=None, # Feature 2
                 smn_general_config=None, # Feature 3
                 smn_controlled_params_config=None, # Feature 3
                 interrupt_handler_config=None, # Feature 4
                 cognitive_firewall_config=None, # Feature 6
                 goal_state_params = None, # Feature 7
                 lot_config=None, # Feature 8
                 shared_long_term_memory=None, # For Feature 5 (Co-Agents)
                 shared_attention_foci=None, # For Feature 5 (Co-Agents)
                 config_overrides=None, # For Co-Agent diversity setup
                 verbose=0):

        self.agent_id = agent_id
        self.logical_superposition = {"00": 1.0 + 0j}
        self.collapsed_logical_state_str = "00"
        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate

        self.operation_costs = {'X': 0.1, 'Z': 0.1, 'H': 0.3, 'CNOT': 0.4, 'CZ': 0.4, 'ERROR_PENALTY': 0.05, 'PLANNING_BASE': 0.02}
        self.outcome_valence_map = {"00": 0.0, "01": 0.5, "10": -0.5, "11": 1.0}
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
        
        self.ltm_utility_weight_valence = 0.6 # Default, can be trained
        self.ltm_utility_weight_efficiency = 0.4 # Default, can be trained

        # Feature 2: Temporal Feedback Grid
        self.temporal_grid_params = copy.deepcopy(DEFAULT_TEMPORAL_GRID_PARAMS)
        if temporal_grid_config: self.temporal_grid_params.update(temporal_grid_config)
        self.temporal_feedback_grid = collections.deque(maxlen=self.temporal_grid_params['max_len'])
        self.last_cycle_entropy_for_delta = 0.0 # For calculating entropy_shift

        # Feature 3: Synaptic Mutation Network (SMN)
        self.smn_config = copy.deepcopy(DEFAULT_SMN_CONFIG)
        if smn_general_config: self.smn_config.update(smn_general_config)
        self.smn_controlled_params_config = copy.deepcopy(DEFAULT_SMN_CONTROLLED_PARAMS)
        if smn_controlled_params_config: self.smn_controlled_params_config.update(smn_controlled_params_config)
        self.smn_params_state = self._initialize_smn_params_state()
        self.smn_internal_flags = {} # For SMN effects, e.g. forcing LTM replay

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
        self.current_goal_state_obj = None # Will hold a GoalState object

        # Feature 8: Internal Language Layer (LoT)
        self.lot_config_params = copy.deepcopy(DEFAULT_LOT_CONFIG)
        if lot_config: self.lot_config_params.update(lot_config)
        self.current_cycle_lot_stream = []


        # Config overrides for co-agent diversity (applied after defaults but before trainable_param_values)
        if config_overrides:
            self._apply_config_overrides(config_overrides)

        if trainable_param_values:
            self.update_emulator_parameters(trainable_param_values) # Apply trained values

        self.long_term_memory_capacity = 100 # Can be configured
        self.successful_sequence_threshold_valence = 0.5 # Min avg valence for LTM to be considered "successful"

        self.cycle_history = collections.deque(maxlen=cycle_history_max_len)
        self.current_cycle_num = 0
        self.verbose = verbose
        self.next_target_input_state = "00" # Planned input for next cycle
        
        if self.verbose >= 1:
            print(f"[{self.agent_id}] Orch-OR Emulator Initialized. Layers: Sensor, Associative, Executive, Meta.")
            print(f"[{self.agent_id}] E_OR_THRESHOLD: {self.E_OR_THRESHOLD:.2f}, ORP Decay Rate: {self.orp_decay_rate:.3f}")

    def _apply_config_overrides(self, overrides):
        """Applies direct value overrides to emulator parameters, useful for co-agent setup."""
        if self.verbose >= 2: print(f"[{self.agent_id}] Applying config overrides: {overrides}")
        for path, value in overrides.items():
            # Path is a tuple e.g., ('internal_state_parameters', 'curiosity')
            # or ('E_OR_THRESHOLD',) for direct attributes
            try:
                if len(path) == 1: # Direct attribute
                    setattr(self, path[0], value)
                    if self.verbose >= 3: print(f"  Override: self.{path[0]} = {value}")
                elif len(path) > 1:
                    obj = self
                    for i, key in enumerate(path[:-1]):
                        obj = getattr(obj, key)
                    obj[path[-1]] = value
                    if self.verbose >= 3: print(f"  Override: self.{'.'.join(path)} = {value}")
            except (AttributeError, KeyError) as e:
                if self.verbose >= 1: print(f"  Warning: Override for path {path} failed: {e}")


    def update_emulator_parameters(self, param_values_dict):
        """Updates emulator's parameters based on a dictionary (e.g., from trainer)."""
        if self.verbose >= 2: print(f"[{self.agent_id}] Updating emulator with trainable params: {param_values_dict}")
        normalized_strategy_weights = False
        for param_name, value in param_values_dict.items():
            if param_name not in DEFAULT_TRAINABLE_PARAMS_CONFIG:
                if self.verbose >= 1: print(f"  Warning: Param '{param_name}' not in DEFAULT_TRAINABLE_PARAMS_CONFIG. Skipping.")
                continue
            config = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name]
            # ... (rest of the logic from existing update_emulator_parameters)
            dict_attr_name = config['target_dict_attr']
            key = config['target_key']
            subkey = config.get('target_subkey') 

            if dict_attr_name is None: 
                setattr(self, key, value)
                if self.verbose >= 3: print(f"      Set direct attr emulator.{key} = {value:.4f}")
            else:
                target_dict = getattr(self, dict_attr_name, None)
                if target_dict is None:
                    if self.verbose >=1: print(f"Warning: Target dict attribute '{dict_attr_name}' not found in emulator for '{param_name}'.")
                    continue
                
                if subkey:
                    if key not in target_dict or not isinstance(target_dict[key], dict):
                        target_dict[key] = {} 
                    target_dict[key][subkey] = value
                    if key == 'strategy_weights': normalized_strategy_weights = True
                    if self.verbose >= 3: print(f"      Set emulator.{dict_attr_name}['{key}']['{subkey}'] = {value:.4f}")
                else:
                    target_dict[key] = value
                    if self.verbose >= 3: print(f"      Set emulator.{dict_attr_name}['{key}'] = {value:.4f}")
        
        if normalized_strategy_weights:
            sw = self.internal_state_parameters['strategy_weights']
            total_sw = sum(sw.values())
            if total_sw > 1e-6 : 
                for k_sw in sw: sw[k_sw] /= total_sw
                if self.verbose >= 3: print(f"      Normalized strategy_weights: {sw}")
            else: 
                if self.verbose >=1: print(f"Warning: All strategy weights became zero. Resetting to uniform.")
                num_strats = len(sw) if sw else 1
                uniform_weight = 1.0 / num_strats if num_strats > 0 else 1.0
                for k_sw in sw : sw[k_sw] = uniform_weight
                if not sw: self.internal_state_parameters['strategy_weights'] = {'curiosity': 1.0}


    # --- Feature 8: Internal Language Layer ---
    def _log_lot_event(self, event_type: str, details: dict):
        if not self.lot_config_params.get('enabled', False): return
        if not self.lot_config_params.get('log_level_details', {}).get(event_type, False) and not self.lot_config_params.get('log_level_details', {}).get(event_type.split('.')[0], False): # Check specific and then main category
            return

        # Sanitize details for LoT string: format floats, handle complex structures simply
        param_strs = []
        for k, v in details.items():
            if isinstance(v, float):
                param_strs.append(f"{k}:{v:.2f}")
            elif isinstance(v, (list, tuple)) and len(v) > 5:
                 param_strs.append(f"{k}:[...{len(v)}items...]")
            elif isinstance(v, dict) and len(v) > 3:
                 param_strs.append(f"{k}:{{...{len(v)}keys...}}")
            else:
                v_str = str(v)
                if len(v_str) > 30 : v_str = v_str[:27] + "..."
                param_strs.append(f"{k}:{v_str}")
        
        tag_name = event_type.upper().replace("_", "")
        self.current_cycle_lot_stream.append(f"#{tag_name}[{','.join(param_strs)}]")

    # --- Core Orch OR Mechanics (centralized, used primarily by Executive Layer) ---
    def _apply_logical_op_to_superposition(self, op_char, logical_arg, current_superposition, current_orp):
        # (largely unchanged from original, but could log LoT events for ops)
        new_superposition = collections.defaultdict(complex)
        new_orp = current_orp
        sqrt2_inv = 1 / np.sqrt(2)
        op_char_upper = op_char.upper()

        if op_char_upper not in self.operation_costs:
            if self.verbose >= 1: print(f"Warning: Op '{op_char_upper}' not in operation_costs. Using default cost 0.05.")
            new_orp += 0.05
        else:
            new_orp += self.operation_costs[op_char_upper]

        self._log_lot_event("op_execution.attempt", {"op":op_char, "arg":logical_arg, "cost":self.operation_costs.get(op_char_upper,0.05)})

        for basis_state_str, amp in current_superposition.items():
            if abs(amp) < 1e-9: continue
            lq1_val, lq0_val = int(basis_state_str[0]), int(basis_state_str[1])
            if op_char_upper == 'X':
                idx_to_flip = logical_arg
                if idx_to_flip == 0: new_basis_state_str = f"{lq1_val}{1-lq0_val}"
                elif idx_to_flip == 1: new_basis_state_str = f"{1-lq1_val}{lq0_val}"
                else: new_superposition[basis_state_str] += amp; new_orp += 0.01 # Error case, keep state add penalty
                new_superposition[new_basis_state_str] += amp
            elif op_char_upper == 'Z':
                idx_to_phase = logical_arg; phase = 1
                if idx_to_phase == 0 and lq0_val == 1: phase = -1
                elif idx_to_phase == 1 and lq1_val == 1: phase = -1
                elif idx_to_phase not in [0,1]: pass # Error, no phase change
                new_superposition[basis_state_str] += amp * phase
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
                else: new_superposition[basis_state_str] += amp; new_orp += 0.01
            elif op_char_upper == 'CNOT':
                ctrl_idx, target_idx = logical_arg
                if ctrl_idx not in [0,1] or target_idx not in [0,1] or ctrl_idx == target_idx: new_superposition[basis_state_str] += amp; new_orp += 0.01
                else:
                    control_active = (lq0_val == 1 if ctrl_idx == 0 else lq1_val == 1)
                    if control_active:
                        flipped_lq0 = 1 - lq0_val if target_idx == 0 else lq0_val
                        flipped_lq1 = 1 - lq1_val if target_idx == 1 else lq1_val
                        new_basis_state_str = f"{flipped_lq1}{flipped_lq0}"
                        new_superposition[new_basis_state_str] += amp
                    else: new_superposition[basis_state_str] += amp
            elif op_char_upper == 'CZ':
                if not isinstance(logical_arg, tuple) or not ({logical_arg[0], logical_arg[1]} == {0,1}): pass # Invalid arg, treat as identity for this state
                phase = 1
                if lq0_val == 1 and lq1_val == 1: phase = -1
                new_superposition[basis_state_str] += amp * phase
            else: # Unknown op
                new_superposition[basis_state_str] += amp
        
        final_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j} # Ensure all keys exist
        norm_sq = sum(abs(a)**2 for a in new_superposition.values())
        if norm_sq > 1e-12:
            norm = np.sqrt(norm_sq)
            for state_key, amp_val in new_superposition.items():
                final_superposition[state_key] = amp_val / norm
        else: # Superposition vanished
            if self.verbose >=1: print(f"[{self.agent_id}] Warning: Superposition norm near zero after op '{op_char_upper}'. Resetting to |00>.")
            self._log_lot_event("op_execution.error", {"op":op_char, "error":"norm_zero_reset_00"})
            final_superposition["00"] = 1.0 + 0j
            new_orp += 0.1 # Penalty for losing superposition
        return dict(final_superposition), new_orp

    def _calculate_superposition_entropy(self, superposition_dict=None):
        # (largely unchanged)
        target_superposition = superposition_dict if superposition_dict is not None else self.logical_superposition
        probabilities = np.array([np.abs(amp)**2 for amp in target_superposition.values()])
        probabilities = probabilities[probabilities > 1e-9] # Filter out tiny probabilities
        if not probabilities.any(): return 0.0
        current_sum_probs = np.sum(probabilities)
        if not np.isclose(current_sum_probs, 1.0) and current_sum_probs > 1e-9:
            probabilities /= current_sum_probs # Normalize if not already (should be close)
        return -np.sum(probabilities * np.log2(probabilities + 1e-12)) # Add epsilon for log(0)

    def _executive_prepare_superposition(self, classical_input_str="00"):
        """Part of Executive Layer: Prepares the initial superposition for the conscious moment."""
        if self.verbose >= 2: print(f"  EXECUTIVE.Super_Prep: Target initial state |{classical_input_str}>")
        self._log_lot_event("executive.super_prep", {"target_state": classical_input_str})
        
        self.logical_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j} # Ensure all keys
        if not (len(classical_input_str) == 2 and all(c in '01' for c in classical_input_str)):
            if self.verbose >= 1: print(f"    ERROR: Invalid classical_input_str '{classical_input_str}'. Defaulting to '00'.")
            self._log_lot_event("executive.super_prep.error", {"input": classical_input_str, "defaulted_to": "00"})
            classical_input_str = "00"
        self.logical_superposition[classical_input_str] = 1.0 + 0j
        self.objective_reduction_potential = 0.0 # Reset ORP for the new quantum evolution
        
        if self.verbose >= 3:
            print(f"    Superposition prepared: {self.logical_superposition_str()}")
            print(f"    ORP after prep: {self.objective_reduction_potential:.3f}")
        self.objective_reduction_potential += 0.05 # Base cost for state prep
        return True

    def _executive_quantum_computation_phase(self, computation_sequence_ops):
        """Part of Executive Layer: Evolves the superposition according to selected operations."""
        if self.verbose >= 2: print(f"  EXECUTIVE.Quantum_Comp: Evolving superposition.")
        self._log_lot_event("executive.quantum_comp.start", {"ops_planned": len(computation_sequence_ops), "orp_start":self.objective_reduction_potential, "decay_rate": self.orp_decay_rate})

        # Apply ORP decay
        orp_before_decay = self.objective_reduction_potential
        decay_amount = self.objective_reduction_potential * self.orp_decay_rate
        self.objective_reduction_potential = max(0, self.objective_reduction_potential - decay_amount)
        if self.verbose >=3 and decay_amount > 1e-6:
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
                try:
                    temp_superposition, temp_orp = \
                        self._apply_logical_op_to_superposition(op_char, logical_arg, temp_superposition, temp_orp)
                    ops_executed_count += 1
                except ValueError as e:
                    if self.verbose >=1: print(f"    Error applying op ('{op_char}', {logical_arg}): {e}. Skipping.")
                    self._log_lot_event("executive.quantum_comp.op_error", {"op":op_char, "arg":logical_arg, "error":str(e)})
                    temp_orp += self.operation_costs.get('ERROR_PENALTY', 0.05) # Penalize for error

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
        return True, or_triggered_early

    def _executive_trigger_objective_reduction(self):
        """Part of Executive Layer: The "conscious moment" of collapse."""
        if self.verbose >= 2: print(f"  EXECUTIVE.Objective_Reduction: Collapsing superposition.")
        self._log_lot_event("executive.objective_reduction.start", {"orp_at_trigger": self.objective_reduction_potential, "superposition": self.logical_superposition_str()})

        basis_states = list(self.logical_superposition.keys())
        amplitudes = np.array([self.logical_superposition[s] for s in basis_states], dtype=complex)
        probabilities = np.abs(amplitudes)**2
        
        sum_probs = np.sum(probabilities)
        if sum_probs < 1e-9: # Should ideally not happen if _apply_logical_op handles it
            if self.verbose >= 1: print("    ERROR: Superposition has near-zero norm before collapse. Defaulting to '00'.")
            self._log_lot_event("executive.objective_reduction.error", {"error":"norm_zero_collapse_00"})
            self.collapsed_logical_state_str = "00"
            # Reset superposition to a valid state
            self.logical_superposition = {"00":1.0+0j, "01":0.0j, "10":0.0j, "11":0.0j}
        elif not np.isclose(sum_probs, 1.0):
            if self.verbose >= 2: print(f"    Normalizing probabilities for collapse (sum was {sum_probs:.4f}).")
            probabilities /= sum_probs
        
        # Perform probabilistic choice
        try:
            chosen_index = np.random.choice(len(basis_states), p=probabilities)
            self.collapsed_logical_state_str = basis_states[chosen_index]
        except ValueError as e: # Catch issues with probabilities (e.g., sum not 1, NaNs)
            if self.verbose >=1: print(f"    Error during probabilistic collapse ({e}). Choosing max_prob or '00'.")
            self._log_lot_event("executive.objective_reduction.error", {"error": str(e)})
            if probabilities.any() and not np.isnan(probabilities).any():
                max_prob_idx = np.argmax(probabilities)
                self.collapsed_logical_state_str = basis_states[max_prob_idx]
            else: # Fallback if probabilities are completely corrupt
                self.collapsed_logical_state_str = "00"
                self.logical_superposition = {"00":1.0+0j, "01":0.0j, "10":0.0j, "11":0.0j}

        if self.verbose >= 2:
            print(f"    OR Event: Collapsed to |{self.collapsed_logical_state_str}>")
        
        # Store ORP at moment of collapse, then reset superposition and ORP for next evolution
        self.current_orp_before_reset = self.objective_reduction_potential 
        self.objective_reduction_potential = 0.0 
        for state_key in self.logical_superposition: # Solidify superposition to collapsed state
            self.logical_superposition[state_key] = 1.0 + 0j if state_key == self.collapsed_logical_state_str else 0.0j
        
        self._log_lot_event("executive.objective_reduction.end", {"collapsed_to": self.collapsed_logical_state_str, "orp_experienced":self.current_orp_before_reset})
        return self.collapsed_logical_state_str

    # --- Layer 1: Sensor Layer ---
    def _sensor_layer_process_input(self, target_classical_input_str: str) -> str:
        """Encodes symbolic/numeric input. Fast, handles noise."""
        if self.verbose >= 2: print(f"  SENSOR_LAYER: Processing target input '{target_classical_input_str}'.")
        self._log_lot_event("sensor.process_input.start", {"target_input": target_classical_input_str})
        
        noise_level = self.internal_state_parameters.get('sensor_input_noise_level', 0.0)
        actual_classical_input_str = target_classical_input_str
        
        if noise_level > 0 and random.random() < 0.5 : # Only apply noise sometimes for variability
            mutated_input_list = list(target_classical_input_str)
            num_flips = 0
            for i in range(len(mutated_input_list)):
                if random.random() < noise_level:
                    mutated_input_list[i] = '1' if mutated_input_list[i] == '0' else '0'
                    num_flips +=1
            if num_flips > 0:
                actual_classical_input_str = "".join(mutated_input_list)
                if self.verbose >= 1: print(f"    SENSOR_LAYER: Input '{target_classical_input_str}' perceived as '{actual_classical_input_str}' due to noise.")
                self._log_lot_event("sensor.process_input.noise_applied", {"original": target_classical_input_str, "actual": actual_classical_input_str, "noise_level": noise_level})
        
        self._log_lot_event("sensor.process_input.end", {"actual_input": actual_classical_input_str})
        return actual_classical_input_str

    # --- Layer 2: Associative Layer ---
    def _associative_layer_update_ltm(self, op_sequence, raw_valence, orp_cost, entropy_gen, consolidation_factor=1.0):
        """Links similar patterns, triggers LTM recall (recall is in Executive). This part handles LTM storage/updates."""
        if self.verbose >= 2: print(f"  ASSOCIATIVE_LAYER.LTM_Update: Seq {op_sequence if op_sequence else 'NoOps'}, Val={raw_valence:.2f}, ORP={orp_cost:.2f}, Ent={entropy_gen:.2f}, ConsolFactor={consolidation_factor:.2f}")
        self._log_lot_event("associative.ltm_update.start", {"op_seq_len":len(op_sequence) if op_sequence else 0, "raw_valence":raw_valence, "orp_cost": orp_cost, "consol_factor": consolidation_factor})

        if not op_sequence: return
        seq_tuple = tuple(op_sequence) # Sequences must be hashable

        # Don't store if valence too low, unless strong consolidation event bypasses this
        if raw_valence < self.successful_sequence_threshold_valence * 0.5 and consolidation_factor <= 1.0:
             if self.verbose >=3: print(f"    LTM_Update: Sequence {seq_tuple} not stored, raw_valence {raw_valence:.2f} too low.")
             return

        entry = self.long_term_memory.get(seq_tuple)
        mutation_rate_store = self.internal_state_parameters.get('ltm_mutation_on_store_rate', 0.0)
        
        # Calculate effective count/strength of this update
        # High consolidation factor means this experience is "stronger"
        update_strength = int(math.ceil(consolidation_factor)) 

        if entry:
            entry['count'] += update_strength
            entry['total_valence'] += raw_valence * consolidation_factor
            entry['total_orp_cost'] += orp_cost * consolidation_factor # Cost is less "good", so factor might be 1 here
            entry['total_entropy_generated'] += entropy_gen * consolidation_factor # Or maybe less for consolidation?

            if random.random() < mutation_rate_store:
                # Small metric perturbation
                entry['total_valence'] *= (1 + random.uniform(-0.05, 0.05) * consolidation_factor)
                entry['total_orp_cost'] *= (1 + random.uniform(-0.05, 0.05)) # Orp cost less affected by good valence
                self._log_lot_event("associative.ltm_update.metric_mutation", {"seq":seq_tuple})
            
            # Update averages
            entry['avg_valence'] = entry['total_valence'] / entry['count']
            entry['avg_orp_cost'] = entry['total_orp_cost'] / entry['count']
            entry['avg_entropy'] = entry['total_entropy_generated'] / entry['count']
        else:
            # Prune LTM if full
            if len(self.long_term_memory) >= self.long_term_memory_capacity:
                if not self.long_term_memory: return # Should not happen if capacity > 0
                # Find least useful entry to prune (simple version: lowest utility)
                # This utility calculation is temporary for pruning, LTM entries will store their own computed utility
                min_utility_val = float('inf'); key_to_prune = None
                for k, v_data in self.long_term_memory.items():
                    # temp_util = (self.ltm_utility_weight_valence * v_data['avg_valence'] - self.ltm_utility_weight_efficiency * (v_data['avg_orp_cost'] / (self.E_OR_THRESHOLD + 1e-6)))
                    temp_util = v_data.get('utility', self._associative_layer_calculate_ltm_entry_utility(v_data))
                    if temp_util < min_utility_val: min_utility_val = temp_util; key_to_prune = k
                
                if key_to_prune:
                    if self.verbose >=3: print(f"    LTM_Update: LTM full. Pruning {key_to_prune} (util {min_utility_val:.2f}).")
                    self._log_lot_event("associative.ltm_update.prune", {"pruned_seq":key_to_prune, "util":min_utility_val})
                    del self.long_term_memory[key_to_prune]
                elif self.verbose >=2: print("    LTM_Update: LTM full, but no suitable key to prune (e.g., all high utility or error).")

            # Add new entry if space available
            if len(self.long_term_memory) < self.long_term_memory_capacity:
                current_raw_valence = raw_valence * consolidation_factor
                current_orp_cost = orp_cost # ORP cost is factual, not scaled by positive consolidation

                if random.random() < mutation_rate_store:
                    current_raw_valence *= (1 + random.uniform(-0.05, 0.05))
                    current_orp_cost *= (1 + random.uniform(-0.05, 0.05))
                    self._log_lot_event("associative.ltm_update.new_metric_mutation", {"seq":seq_tuple})

                new_entry = {
                    'count': update_strength, 
                    'total_valence': current_raw_valence, 'avg_valence': current_raw_valence / update_strength if update_strength else current_raw_valence,
                    'total_orp_cost': current_orp_cost * update_strength, 'avg_orp_cost': current_orp_cost, # Avg orp cost is per instance
                    'total_entropy_generated': entropy_gen * update_strength, 'avg_entropy': entropy_gen,
                    'first_cycle': self.current_cycle_num, 'last_cycle': self.current_cycle_num,
                }
                self.long_term_memory[seq_tuple] = new_entry
                if self.verbose >=3: print(f"    LTM_Update: Added new sequence {seq_tuple}.")
                self._log_lot_event("associative.ltm_update.new_entry", {"seq":seq_tuple, "val":new_entry['avg_valence']})

        # Update utility for the entry (either new or existing)
        if seq_tuple in self.long_term_memory:
             self.long_term_memory[seq_tuple]['utility'] = self._associative_layer_calculate_ltm_entry_utility(self.long_term_memory[seq_tuple])
             self.long_term_memory[seq_tuple]['last_cycle'] = self.current_cycle_num
    
    def _associative_layer_calculate_ltm_entry_utility(self, seq_data):
        """Calculates utility of an LTM entry."""
        # Normalize ORP cost against current threshold (higher threshold makes same cost less "efficient")
        norm_orp_cost = seq_data['avg_orp_cost'] / (self.E_OR_THRESHOLD + 1e-6)
        # Utility: high valence good, high cost bad, moderate entropy can be good (novelty)
        utility = (self.ltm_utility_weight_valence * seq_data['avg_valence'] -
                   self.ltm_utility_weight_efficiency * norm_orp_cost +
                   0.05 * seq_data['avg_entropy']) # Small bonus for entropy/novelty
        # Recency bonus (slight): older memories are slightly less utility unless very good
        # cycles_ago = self.current_cycle_num - seq_data.get('last_cycle', self.current_cycle_num)
        # utility -= 0.001 * cycles_ago
        return utility

    def _associative_layer_recall_from_ltm_strategy(self, current_orp_value, exec_thought_log): # Part of Executive decision
        """Strategy for Executive: recall a sequence from LTM."""
        if not self.long_term_memory:
            exec_thought_log.append("LTM recall: LTM empty.")
            return None, current_orp_value

        candidate_sequences = []; weights = []
        min_utility_for_recall = 0.1 # Don't recall very low utility sequences

        for seq_tuple, data in self.long_term_memory.items():
            utility = data.get('utility', self._associative_layer_calculate_ltm_entry_utility(data)) # ensure utility exists
            if utility > min_utility_for_recall:
                # Bias towards sequences that don't immediately exceed ORP threshold from current ORP
                projected_cost = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in seq_tuple)
                if current_orp_value + projected_cost < self.E_OR_THRESHOLD * 1.1: # Allow slight overrun
                    candidate_sequences.append(list(seq_tuple))
                    weights.append(utility**2) # Square utility to emphasize better ones

        if not candidate_sequences:
            exec_thought_log.append("LTM recall: No suitable high-utility sequences found or all too costly.")
            return None, current_orp_value

        sum_weights = sum(weights)
        if sum_weights <= 0: # Should not happen if min_utility_for_recall > 0
             exec_thought_log.append("LTM recall: No LTM sequences with positive utility weights after filtering.")
             return None, current_orp_value
        
        normalized_weights = [w / sum_weights for w in weights]
        chosen_sequence_ops_orig_idx = random.choices(range(len(candidate_sequences)), weights=normalized_weights, k=1)[0]
        chosen_sequence_ops_orig = candidate_sequences[chosen_sequence_ops_orig_idx]
        chosen_sequence_ops = list(chosen_sequence_ops_orig) # Work with a copy

        # Potential mutation on replay
        mutation_rate_replay = self.internal_state_parameters.get('ltm_mutation_on_replay_rate', 0.0)
        if chosen_sequence_ops and random.random() < mutation_rate_replay and len(chosen_sequence_ops) > 0:
            # Apply a small mutation to the chosen sequence (change op, arg, add/remove)
            idx_to_mutate = random.randrange(len(chosen_sequence_ops))
            op_char, op_arg = chosen_sequence_ops[idx_to_mutate]
            original_op_tuple_str = f"('{op_char}', {op_arg})"
            
            mutation_type_rand = random.random()
            if mutation_type_rand < 0.4 and op_char in ['X', 'Z', 'H'] and isinstance(op_arg, int): # Flip arg
                chosen_sequence_ops[idx_to_mutate] = (op_char, 1 - op_arg)
            elif mutation_type_rand < 0.7: # Change op type
                compatible_ops={'X':['Z','H'],'Z':['X','H'],'H':['X','Z'],'CNOT':['CZ'],'CZ':['CNOT']}
                new_op_char = random.choice(compatible_ops.get(op_char, ['X','Z','H'])) # Fallback to simple ops
                new_op_arg = op_arg # Keep arg if compatible (CNOT/CZ)
                if new_op_char in ['X','Z','H']: new_op_arg = random.randint(0,1)
                chosen_sequence_ops[idx_to_mutate] = (new_op_char, new_op_arg)
            else: # Add or remove op near mutated one
                if len(chosen_sequence_ops) > 1 and random.random() < 0.5 : # Delete
                    del chosen_sequence_ops[idx_to_mutate]
                else: # Insert new random op
                    new_op_insert = (random.choice(['X','Z','H']), random.randint(0,1))
                    chosen_sequence_ops.insert(random.randint(0, len(chosen_sequence_ops)), new_op_insert)
            
            exec_thought_log.append(f"LTM Replay MUTATION: Op {original_op_tuple_str} in {chosen_sequence_ops_orig} -> mutated to/around in {chosen_sequence_ops}.")
            self._log_lot_event("associative.ltm_recall.mutation", {"original_seq": chosen_sequence_ops_orig, "mutated_seq": chosen_sequence_ops})


        # Final check on cost after potential mutation
        projected_orp_increase_final = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in chosen_sequence_ops)
        if current_orp_value + projected_orp_increase_final >= self.E_OR_THRESHOLD and len(chosen_sequence_ops) > 1:
            exec_thought_log.append(f"LTM recall: Mutated seq {chosen_sequence_ops} too costly. Skipped.")
            return None, current_orp_value # Skip if mutated version became too costly
        
        orig_data = self.long_term_memory[tuple(chosen_sequence_ops_orig)] 
        exec_thought_log.append(f"LTM recall: Replaying {chosen_sequence_ops} (orig_avg_V={orig_data['avg_valence']:.2f}, util={orig_data['utility']:.2f}).")
        self._log_lot_event("associative.ltm_recall.chosen", {"seq":chosen_sequence_ops, "orig_util":orig_data['utility']})
        return chosen_sequence_ops, current_orp_value


    # --- Layer 3: Executive Layer (Decision Making, Planning, Conscious Experience) ---
    # Includes ACC-like outcome evaluation
    def _executive_evaluate_outcome_and_update_mood(self, logical_outcome_str, orp_at_collapse, entropy_at_collapse, num_ops_executed_this_cycle):
        """Evaluates collapse outcome, updates mood, frustration. This is ACC-like function."""
        if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.Outcome_Eval: |{logical_outcome_str}>, ORP={orp_at_collapse:.3f}, Ent={entropy_at_collapse:.2f}")
        acc_thoughts_log = [] # For detailed logging within this function
        
        raw_valence = self.outcome_valence_map.get(logical_outcome_str, -0.1) # Default small neg valence for unknown states
        mod_valence = raw_valence
        acc_thoughts_log.append(f"Raw val for |{logical_outcome_str}> is {raw_valence:.2f}.")

        # Valence modulation based on ORP surprise
        orp_surprise_factor = 0.15
        expected_orp_mid = self.E_OR_THRESHOLD * 0.75 # sweet spot
        if orp_at_collapse < self.E_OR_THRESHOLD * 0.4: # Collapsed too early
            mod_valence -= orp_surprise_factor * abs(raw_valence if raw_valence!=0 else 0.2) # Penalty even if raw_valence is 0
            acc_thoughts_log.append(f"Early OR collapse, val modified by {-orp_surprise_factor * abs(raw_valence if raw_valence!=0 else 0.2):.2f}.")
        elif orp_at_collapse > self.E_OR_THRESHOLD * 1.3 and num_ops_executed_this_cycle > 0: # Collapsed late (worked hard)
            # Small penalty if valence is negative, small bonus if positive
            late_factor = -0.05 if raw_valence < 0 else 0.05
            mod_valence += late_factor
            acc_thoughts_log.append(f"Late OR collapse, val modified by {late_factor:.2f}.")

        # Valence modulation by preferred state achievement
        current_preferred_state = self.internal_state_parameters.get('preferred_logical_state')
        if current_preferred_state is not None and current_preferred_state == logical_outcome_str:
            preference_bonus = 0.25 * (1.0 - abs(mod_valence)) # Add more bonus if current valence isn't already extreme
            mod_valence += preference_bonus
            acc_thoughts_log.append(f"Preferred state |{current_preferred_state}> met, val boosted by {preference_bonus:.2f}.")
            # Check if this preferred state was part of a goal
            if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
                self._executive_update_goal_progress(logical_outcome_str, None) # Ops_applied is None here, but can be passed

        mod_valence = np.clip(mod_valence, -1.0, 1.0)
        self.last_cycle_valence_raw = raw_valence
        self.last_cycle_valence_mod = mod_valence
        acc_thoughts_log.append(f"Final val (raw/mod): {raw_valence:.2f}/{mod_valence:.2f}.")
        self._log_lot_event("executive.outcome_eval.valence", {"raw":raw_valence, "mod":mod_valence, "outcome_state":logical_outcome_str})

        # Update mood based on modified valence
        current_mood = self.internal_state_parameters['mood']
        mood_inertia = 0.9 # Mood is somewhat stable
        valence_influence_on_mood = 0.25 
        new_mood = current_mood * mood_inertia + mod_valence * valence_influence_on_mood
        self.internal_state_parameters['mood'] = np.clip(new_mood, -1.0, 1.0)
        acc_thoughts_log.append(f"Mood updated to {self.internal_state_parameters['mood']:.2f}.")
        self._log_lot_event("executive.outcome_eval.mood", {"new_mood":self.internal_state_parameters['mood']})


        # Update frustration and exploration mode based on modified valence
        current_frustration = self.internal_state_parameters['frustration']
        frustration_threshold = self.metacognition_params['frustration_threshold']
        if mod_valence < self.metacognition_params['low_valence_threshold'] * 0.8: # Significantly low valence increases frustration
            current_frustration += 0.20
        else: # Otherwise, frustration naturally decays
            current_frustration *= 0.85
        self.internal_state_parameters['frustration'] = np.clip(current_frustration, 0.0, 1.0)

        # Exploration mode countdown
        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            self.internal_state_parameters['exploration_mode_countdown'] -= 1
            if self.verbose >= 2 and self.internal_state_parameters['exploration_mode_countdown'] == 0:
                acc_thoughts_log.append("Exploration mode ended.")
                self._log_lot_event("executive.outcome_eval.exploration_end", {})


        # Trigger exploration if frustration is high and not already exploring
        if self.internal_state_parameters['frustration'] >= frustration_threshold and \
           self.internal_state_parameters['exploration_mode_countdown'] == 0:
            if self.verbose >= 1: print(f"[{self.agent_id}] High frustration ({self.internal_state_parameters['frustration']:.2f}) triggered exploration mode!")
            self._log_lot_event("executive.outcome_eval.exploration_start", {"frustration":self.internal_state_parameters['frustration']})
            self.internal_state_parameters['exploration_mode_countdown'] = self.metacognition_params['exploration_mode_duration']
            self.internal_state_parameters['frustration'] = 0.3 # Partially alleviate frustration
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.3) # Boost curiosity
            # Could also trigger firewall-like gentle interventions like strategy shuffle here if desired
        
        acc_thoughts_log.append(f"Frustration: {self.internal_state_parameters['frustration']:.2f}, Exploration T-: {self.internal_state_parameters['exploration_mode_countdown']}.")
        
        return { # Return a dict of key outcomes for logging and other layers
            'raw_valence':raw_valence, 'mod_valence':mod_valence, 
            'mood':self.internal_state_parameters['mood'],
            'frustration':self.internal_state_parameters['frustration'], 
            'exploration_countdown':self.internal_state_parameters['exploration_mode_countdown'],
            'thoughts_log': acc_thoughts_log # For detailed debugging if needed
        }

    def _executive_generate_computation_sequence(self, ops_provided_externally=None):
        """Part of Executive Layer: Decides what operations to perform (PFC-like action selection)."""
        if ops_provided_externally is not None:
            if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.OpGen: Using externally provided ops: {ops_provided_externally}")
            self._log_lot_event("executive.opgen.external", {"ops_count": len(ops_provided_externally)})
            return ops_provided_externally, "StrategyProvidedExternal", ["Ops provided externally."]

        exec_thought_log = ["OpGen: Generating new computation sequence:"] # For detailed logging within this function
        self._log_lot_event("executive.opgen.start", {"orp_current":self.objective_reduction_potential, "threshold": self.E_OR_THRESHOLD})

        ops_sequence = []
        chosen_strategy_name = "NoOpsMethod" # Fallback name

        effective_attention = self.internal_state_parameters['attention_level']
        # Cognitive load reduces willingness to make long computation sequences
        cognitive_load_factor = 1.0 - (self.internal_state_parameters['cognitive_load'] * 0.6) 
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        # Fewer ops if load is high, or if attention is low
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * cognitive_load_factor * effective_attention, scale=0.8)))
        
        exec_thought_log.append(f"  Target ops: ~{num_ops_target} (base:{num_ops_target_base}, load_factor:{cognitive_load_factor:.2f}, attn:{effective_attention:.2f}). ORP start: {self.objective_reduction_potential:.3f}")

        current_strategy_weights = self.internal_state_parameters['strategy_weights'].copy()

        # --- Influence from Temporal Feedback Grid (Feature 2) ---
        if self.temporal_feedback_grid:
            avg_recent_valence_delta = np.mean([g[1] for g in self.temporal_feedback_grid if g[1] is not None] or [0])
            avg_recent_entropy_shift = np.mean([g[2] for g in self.temporal_feedback_grid if g[2] is not None] or [0])
            exec_thought_log.append(f"  TemporalGrid: AvgValDelta={avg_recent_valence_delta:.2f}, AvgEntShift={avg_recent_entropy_shift:.2f}")

            valence_bias_strength = self.internal_state_parameters.get('temporal_feedback_valence_bias_strength',0.1)
            entropy_bias_strength = self.internal_state_parameters.get('temporal_feedback_entropy_bias_strength',0.05)

            if avg_recent_valence_delta < self.temporal_grid_params['low_valence_delta_threshold']:
                exec_thought_log.append("    Low recent valence delta, boosting curiosity/memory, reducing goal/problem focus slightly.")
                current_strategy_weights['problem_solve'] *= (1 - valence_bias_strength)
                current_strategy_weights['goal_seek'] *= (1 - valence_bias_strength)
                current_strategy_weights['curiosity'] += current_strategy_weights['problem_solve'] * valence_bias_strength * 0.5 + 0.05 # Ensure some boost
                current_strategy_weights['memory'] += current_strategy_weights['goal_seek'] * valence_bias_strength * 0.5 + 0.05 # Ensure some boost
                self._log_lot_event("executive.opgen.temporal_bias.neg_val_delta", {"val_delta": avg_recent_valence_delta})


            if avg_recent_entropy_shift > self.temporal_grid_params['high_entropy_shift_threshold'] and avg_recent_valence_delta < 0.1: # High chaos, not good results
                exec_thought_log.append("    High recent entropy shift with non-positive valence, boosting memory, reducing curiosity.")
                current_strategy_weights['curiosity'] *= (1 - entropy_bias_strength)
                current_strategy_weights['memory'] += current_strategy_weights['curiosity'] * entropy_bias_strength + 0.05
                self._log_lot_event("executive.opgen.temporal_bias.high_ent_shift", {"ent_shift": avg_recent_entropy_shift})


        # --- Influence from current goal state (Feature 7) ---
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            exec_thought_log.append(f"  Goal Active: {self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index].get('name', 'UnnamedStep')}")
            current_strategy_weights['goal_seek'] *= 1.5 # Boost goal seeking significantly
            current_strategy_weights['problem_solve'] *= 1.2 
            # If current step has a target state, set it as preferred state strongly
            current_step_details = self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index]
            if current_step_details.get("target_state"):
                self.internal_state_parameters['preferred_logical_state'] = current_step_details["target_state"]
                exec_thought_log.append(f"    Goal sets preferred state to |{current_step_details['target_state']}>")


        # --- Influence from Exploration Mode ---
        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            exec_thought_log.append("  Exploration mode active: Boosting curiosity strategy.")
            current_strategy_weights['curiosity'] = min(1.0, current_strategy_weights.get('curiosity',0.1)*2.5) # Strong boost
            current_strategy_weights['problem_solve'] *= 0.6
            current_strategy_weights['goal_seek'] *= 0.4 # Less goal-focused during exploration
            self._log_lot_event("executive.opgen.exploration_bias", {})

        # --- SMN Influence: Check for forced LTM replay (Feature 4 via SMN flag)
        if self.smn_internal_flags.get('force_ltm_reactive_op_next_cycle', False):
            exec_thought_log.append("  SMN/Interrupt Flag: Forcing LTM Reactive operation.")
            current_strategy_weights = {'memory': 1.0, 'problem_solve': 0.01, 'goal_seek': 0.01, 'curiosity': 0.01} # Strongly bias to memory
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = False # Consume flag
            self._log_lot_event("executive.opgen.interrupt_bias.force_ltm", {})


        # Normalize strategy weights
        # Ensure all expected keys exist for normalization
        for key in DEFAULT_INTERNAL_PARAMS['strategy_weights']:
            if key not in current_strategy_weights: current_strategy_weights[key] = 0.001 # small floor value
        total_weight = sum(current_strategy_weights.values())
        if total_weight <= 1e-6: current_strategy_weights = {'curiosity': 1.0}; total_weight = 1.0 # Fallback
        strategy_choices = list(current_strategy_weights.keys())
        strategy_probs = [w / total_weight for w in current_strategy_weights.values()]
        
        exec_thought_log.append(f"  Strategy weights (norm): { {s:f'{p:.2f}' for s,p in zip(strategy_choices, strategy_probs)} }")

        # Select a primary strategy for this op generation block
        # This is a "collapse" of strategy choice
        try:
            selected_strategy = random.choices(strategy_choices, weights=strategy_probs, k=1)[0]
        except ValueError: # Handle empty or invalid weights
             selected_strategy = 'curiosity' # Fallback
             exec_thought_log.append("  Error in strategy selection weights, defaulting to curiosity.")
        exec_thought_log.append(f"  Selected primary strategy: {selected_strategy}")
        self._log_lot_event("executive.opgen.strategy_selected", {"strategy":selected_strategy, "weights": {s:f'{p:.2f}' for s,p in zip(strategy_choices, strategy_probs)}})


        # --- Implement chosen strategy to generate ops ---
        simulated_orp_accumulator = self.objective_reduction_potential # For planning ops without exceeding threshold

        if selected_strategy == 'memory':
            replay_ops, _ = self._associative_layer_recall_from_ltm_strategy(simulated_orp_accumulator, exec_thought_log)
            if replay_ops: ops_sequence = replay_ops; chosen_strategy_name = "StrategyLTMReplay"
        
        # Problem Solving: Tries to reach preferred_logical_state with simple ops
        # This is a basic version; could be expanded to more complex problem decomposition
        if not ops_sequence and selected_strategy == 'problem_solve':
            pref_state = self.internal_state_parameters['preferred_logical_state']
            if pref_state:
                exec_thought_log.append(f"  ProblemSolving towards |{pref_state}> from |{self.collapsed_logical_state_str}>")
                # Simple plan: flip bits if different. More complex planning could involve H for superposition.
                # This is a simplified placeholder for more advanced symbolic reasoning for "problem_solve"
                # For now, just checks if already there or plans simple flips.
                # This logic would be much more elaborate in a system solving, e.g., algebra.
                current_l1,current_l0=int(self.collapsed_logical_state_str[0]),int(self.collapsed_logical_state_str[1])
                target_l1,target_l0=int(pref_state[0]),int(pref_state[1])
                
                planned_problem_ops = []
                temp_plan_orp = simulated_orp_accumulator + self.operation_costs.get('PLANNING_BASE', 0.02)

                if current_l0 != target_l0:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD:
                        planned_problem_ops.append(('X',0)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    Cannot apply ('X',0) to reach target due to ORP limit.")
                if current_l1 != target_l1:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD:
                        planned_problem_ops.append(('X',1)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    Cannot apply ('X',1) to reach target due to ORP limit.")
                
                if planned_problem_ops:
                    ops_sequence = planned_problem_ops
                    chosen_strategy_name = "StrategyProblemSolving"
                    exec_thought_log.append(f"    ProblemSolving plan: {ops_sequence}")
                elif pref_state == self.collapsed_logical_state_str:
                     exec_thought_log.append(f"    ProblemSolving: Already at preferred state |{pref_state}>.")
                else:
                     exec_thought_log.append(f"    ProblemSolving: No simple ops plan to |{pref_state}> or ORP limited.")
            else:
                exec_thought_log.append("  ProblemSolving: No preferred_logical_state set.")


        # Goal Seeking or Curiosity: Generates ops more dynamically, up to num_ops_target
        if not ops_sequence: # Fallback if LTM/ProblemSolving failed or wasn't chosen
            if selected_strategy == 'goal_seek' and self.internal_state_parameters['preferred_logical_state']:
                chosen_strategy_name = "StrategyGoalSeekingLoop"
                exec_thought_log.append(f"  Executing GoalSeeking towards |{self.internal_state_parameters['preferred_logical_state']}>")
            else: # Default to curiosity or if goal_seek conditions not met
                chosen_strategy_name = "StrategyCuriosityDrivenLoop"
                exec_thought_log.append(f"  Executing CuriosityDriven op generation.")

            pref_s = self.internal_state_parameters['preferred_logical_state']
            c_l1,c_l0=int(self.collapsed_logical_state_str[0]),int(self.collapsed_logical_state_str[1])

            for op_count in range(num_ops_target):
                chosen_op_details = None; op_cost = 0
                attention_lapse_this_op = (random.random() > effective_attention * (1-self.internal_state_parameters['cognitive_load']*0.3) )
                
                op_c, op_a = 'X', 0 # Fallback op

                if chosen_strategy_name == "StrategyGoalSeekingLoop" and pref_s: # Try to make progress on goal state
                    t_l1,t_l0=int(pref_s[0]),int(pref_s[1])
                    if c_l0 != t_l0 : op_c,op_a = ('X',0) 
                    elif c_l1 != t_l1 : op_c,op_a = ('X',1)
                    # If current state matches preferred, goal seeking might try to make it "more interesting" via H
                    elif random.random() < 0.3: op_c,op_a = ('H',random.randint(0,1))
                    else: # No obvious bit flip, do a curiosity op
                        op_c=random.choice(['H','Z'] + (['CNOT','CZ'] if random.random() < 0.3 else [])) # Bias to single-qubit
                        op_a=random.randint(0,1) if op_c in ['H','X','Z'] else tuple(random.sample([0,1],2)) if op_c in ['CNOT','CZ'] else 0
                else: # Curiosity driven
                    op_c=random.choice(['X','Z','H','CNOT','CZ'])
                    if op_c in ['X','Z','H']: op_a = random.randint(0,1)
                    else: op_a = tuple(random.sample([0,1],2)) # CNOT/CZ target
                
                op_cost = self.operation_costs.get(op_c, 0.05)

                if attention_lapse_this_op:
                    original_op_arg = op_a
                    if op_c in ['X','Z','H'] and isinstance(op_a,int): op_a = 1 - op_a # Flip target qubit
                    elif op_c in ['CNOT','CZ'] and isinstance(op_a,tuple): op_a = (op_a[1],op_a[0]) # Swap ctrl/target
                    op_cost += self.operation_costs.get('ERROR_PENALTY',0.05) # Cost of attention lapse
                    exec_thought_log.append(f"      ATTENTION LAPSE! Op ({op_c},{original_op_arg}) -> ({op_c},{op_a}), cost penalty.")
                    self._log_lot_event("executive.opgen.attention_lapse", {"original_op":(op_c, original_op_arg), "mutated_op":(op_c, op_a)})


                if simulated_orp_accumulator + op_cost < self.E_OR_THRESHOLD:
                    ops_sequence.append((op_c,op_a))
                    simulated_orp_accumulator += op_cost
                    # Update current state bits if X op applied, for multi-step goal seeking logic
                    if op_c == 'X': 
                        if op_a == 0: c_l0 = 1-c_l0
                        else: c_l1 = 1-c_l1
                else:
                    exec_thought_log.append(f"    OpGen loop ({op_count+1}): Op ('{op_c}',{op_a}) would exceed ORP threshold. Stopping sequence generation.")
                    break # Stop if next op exceeds threshold
        
        if not ops_sequence:
            chosen_strategy_name = "NoOpsGenerated"
            exec_thought_log.append("  Final: No operations generated by any strategy.")
        
        self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy":chosen_strategy_name, "final_sim_orp":simulated_orp_accumulator})
        return ops_sequence, chosen_strategy_name, exec_thought_log

    def _executive_plan_next_target_input(self, current_outcome_str, executive_eval_results, exec_thought_log):
        """Part of Executive Layer: Plans the "intended" input for the next cycle (PFC-like)."""
        exec_thought_log.append(f"PlanNextInput based on |{current_outcome_str}> and eval:")
        
        # Default heuristic: cycle through states
        base_next_input_map = {"00":"01", "01":"10", "10":"11", "11":"00"}
        next_input = base_next_input_map.get(current_outcome_str, "00") # Default to "00" if outcome unknown
        exec_thought_log.append(f"  Base heuristic next input: |{next_input}>.")

        # Goal-directed override (Feature 7)
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            # If the current goal step specifies a next target input (e.g. for sequence tasks)
            current_step_idx = self.current_goal_state_obj.current_step_index
            if 0 <= current_step_idx < len(self.current_goal_state_obj.steps):
                step_info = self.current_goal_state_obj.steps[current_step_idx]
                if step_info.get("next_input_for_world"): # If goal step defines next external input
                    next_input = step_info["next_input_for_world"]
                    exec_thought_log.append(f"  GoalStep '{step_info.get('name')}' overrides next input to |{next_input}>.")
                    self._log_lot_event("executive.plannext.goal_override", {"next_input": next_input, "goal_step":step_info.get('name')})
        
        # Influence from preferred logical state (if not goal-overridden)
        elif self.internal_state_parameters['preferred_logical_state'] and \
           self.internal_state_parameters['preferred_logical_state'] != next_input and \
           random.random() < self.internal_state_parameters['goal_seeking_bias'] * 0.8:
            next_input = self.internal_state_parameters['preferred_logical_state']
            exec_thought_log.append(f"  Overridden by PreferredStateBias: next input |{next_input}>.")
            self._log_lot_event("executive.plannext.preferred_state_override", {"next_input": next_input})

        # Influence from exploration mode or very negative mood
        elif executive_eval_results['exploration_countdown'] > 0 or \
             (executive_eval_results['mood'] < -0.7 and random.random() < 0.5): # Bad mood prompts change
            available_inputs = list(base_next_input_map.keys())
            if current_outcome_str in available_inputs: available_inputs.remove(str(current_outcome_str)) # Don't repeat current
            if str(next_input) in available_inputs: available_inputs.remove(str(next_input)) # Don't pick base heuristic if exploring
            
            if available_inputs:
                next_input = random.choice(available_inputs)
                exec_thought_log.append(f"  Exploration/Mood override: next input |{next_input}>.")
                self._log_lot_event("executive.plannext.exploration_override", {"next_input": next_input})
            else: # Fallback if all other states removed
                next_input = base_next_input_map.get(current_outcome_str, "00") # Back to default
                exec_thought_log.append(f"  Exploration override failed to find different state, using default |{next_input}>.")

        # Good mood repeat tendency: if good outcome, might want to try same input context again
        elif executive_eval_results['mood'] > 0.7 and random.random() < 0.35 and self.cycle_history:
            last_actual_input = self.cycle_history[-1]['actual_input_state_used']
            if last_actual_input:
                next_input = last_actual_input
                exec_thought_log.append(f"  Good mood, repeating last input context |{last_actual_input}>.")
                self._log_lot_event("executive.plannext.good_mood_repeat", {"next_input": next_input})
        
        final_next_input_str = str(next_input) # Ensure it's a string
        exec_thought_log.append(f"  Final proposed next input: |{final_next_input_str}>.")
        self.next_target_input_state = final_next_input_str # Store for next cycle run
        return final_next_input_str

    def _executive_update_goal_progress(self, collapsed_outcome_str, executed_ops):
        """Part of Executive Layer: Updates progress on the current goal (Feature 7)."""
        if not (self.current_goal_state_obj and self.current_goal_state_obj.status == "active"):
            return

        goal = self.current_goal_state_obj
        step_idx = goal.current_step_index
        if not (0 <= step_idx < len(goal.steps)):
            if self.verbose >= 1: print(f"[{self.agent_id}] Goal Error: Invalid step index {step_idx} for goal '{goal.current_goal}'")
            self._log_lot_event("executive.goalprogress.error", {"goal": goal.current_goal, "error": "invalid_step_idx"})
            goal.status = "failed"
            goal.history.append({"cycle": self.current_cycle_num, "event": "error_invalid_step_idx"})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)
            return

        current_step = goal.steps[step_idx]
        step_name = current_step.get("name", f"Step {step_idx + 1}")
        self._log_lot_event("executive.goalprogress.check", {"goal": goal.current_goal, "step": step_name})

        achieved_step = False
        # How to check achievement?
        # 1. Target state match
        if current_step.get("target_state") and collapsed_outcome_str == current_step["target_state"]:
            achieved_step = True
            if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via target state |{collapsed_outcome_str}>.")
        # 2. Callable criteria (advanced)
        elif callable(current_step.get("completion_criteria")):
            try:
                # Criteria might need more context like agent state, full LTM access etc.
                # For now, pass basic info.
                context = {'collapsed_state': collapsed_outcome_str, 'ops': executed_ops, 'agent': self}
                if current_step["completion_criteria"](context):
                    achieved_step = True
                    if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via custom criteria.")
            except Exception as e:
                if self.verbose >=1: print(f"[{self.agent_id}] Error in goal step completion_criteria for '{step_name}': {e}")
                self._log_lot_event("executive.goalprogress.criteria_error", {"step":step_name, "error":str(e)})


        # 3. Heuristic: If LTM recall was used and the LTM was tagged for this step (very advanced)
        #   This requires ops_gen to tag its output or LTM to have goal-step tags.

        if achieved_step:
            goal.history.append({"cycle": self.current_cycle_num, "event": f"step_completed", "step_name": step_name})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['step_completion_valence_bonus'], -1.0, 1.0)
            goal.current_step_index += 1
            num_steps = len(goal.steps)
            goal.progress = goal.current_step_index / num_steps if num_steps > 0 else 1.0
            self._log_lot_event("executive.goalprogress.step_complete", {"step": step_name, "new_progress": goal.progress})


            if goal.current_step_index >= len(goal.steps):
                goal.status = "completed"
                goal.progress = 1.0
                if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' COMPLETED!")
                self._log_lot_event("executive.goalprogress.goal_complete", {"goal": goal.current_goal})
                self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['completion_valence_bonus'], -1.0, 1.0)
                # Maybe clear preferred_logical_state if it was set by this goal
                if self.internal_state_parameters['preferred_logical_state'] == current_step.get("target_state"):
                    self.internal_state_parameters['preferred_logical_state'] = None
            else: # Log next step
                 next_step_name = goal.steps[goal.current_step_index].get("name", f"Step {goal.current_step_index+1}")
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Advanced to step '{next_step_name}'.")
        else: # Step not achieved this cycle
            # Check for goal failure conditions (e.g., too many cycles on one step, error tolerance exceeded)
            # This is a placeholder for more sophisticated failure detection
            goal.history.append({"cycle": self.current_cycle_num, "event": "step_no_progress", "step_name": step_name})
            # Potentially increase frustration or trigger replanning if stuck on a step for too long.
            # Example: If a step has a 'max_cycles' and it's exceeded, fail the goal.
            if current_step.get("max_cycles_on_step", float('inf')) <= sum(1 for h_entry in goal.history if h_entry.get("step_name")==step_name and h_entry.get("event")=="step_no_progress"):
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' FAILED due to too many cycles on step '{step_name}'.")
                 self._log_lot_event("executive.goalprogress.goal_fail", {"goal":goal.current_goal, "reason":f"max_cycles_on_step_{step_name}"})
                 goal.status = "failed"
                 self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)

    # --- Feature 4: Collapse-Triggered Interrupt Handlers ---
    def _executive_handle_collapse_interrupts(self, orp_at_collapse, executed_ops_this_cycle, raw_valence_of_collapse):
        """Checks for conditions to trigger special interrupt handling routines after OR event."""
        if not self.interrupt_handler_params.get('enabled', False): return

        self._log_lot_event("executive.interrupt_handler.check", {"orp":orp_at_collapse, "raw_valence":raw_valence_of_collapse})

        # 1. Stronger Memory Consolidation
        # If valence is extremely high/low, or ORP was surprisingly high for ops done
        expected_orp = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in executed_ops_this_cycle) + 0.05 # base prep cost
        orp_surprise = (orp_at_collapse > expected_orp * self.interrupt_handler_params['consolidation_orp_surprise_factor'] and expected_orp > 0.1)
        
        if abs(raw_valence_of_collapse) >= self.interrupt_handler_params['consolidation_valence_abs_threshold'] or orp_surprise:
            consol_bonus = self.interrupt_handler_params['consolidation_strength_bonus']
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Strong LTM consolidation triggered (factor {consol_bonus:.1f}). Valence: {raw_valence_of_collapse:.2f}, ORPSurprise: {orp_surprise}")
            self._log_lot_event("executive.interrupt_handler.strong_consolidation", {"bonus":consol_bonus, "reason_val":abs(raw_valence_of_collapse), "reason_orp_surprise":orp_surprise})
            # The actual call to LTM update with bonus happens later in main cycle flow, this just notes the factor for now.
            # We store it in smn_internal_flags which can be picked by the LTM update func
            self.smn_internal_flags['ltm_consolidation_bonus_factor'] = consol_bonus


        # 2. Reactive LTM Ops Trigger
        # If valence is very low, flag to prioritize LTM search for "fix-it" sequence next cycle
        if raw_valence_of_collapse < self.interrupt_handler_params['reactive_ltm_valence_threshold']:
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Reactive LTM flag set due to low valence ({raw_valence_of_collapse:.2f}).")
            self._log_lot_event("executive.interrupt_handler.reactive_ltm_flag", {"valence":raw_valence_of_collapse})
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = True # Checked by op_gen

        # 3. Cognitive State Forking (Simplified: Mark as interesting preferred state)
        # If a very positive/interesting collapse happened.
        if raw_valence_of_collapse >= self.interrupt_handler_params['cognitive_fork_valence_threshold']:
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Cognitive fork - marking |{self.collapsed_logical_state_str}> as high-interest preferred state.")
            self._log_lot_event("executive.interrupt_handler.cognitive_fork", {"state":self.collapsed_logical_state_str, "valence":raw_valence_of_collapse})
            self.internal_state_parameters['preferred_logical_state'] = self.collapsed_logical_state_str
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + self.interrupt_handler_params['cognitive_fork_goal_bias_boost'])
            # Could also boost attention or curiosity towards exploring this state's consequences.


    # --- Layer 4: Meta Layer (Monitoring, Adaptation, Self-Reflection) ---
    def _meta_layer_update_cognitive_parameters(self, orp_at_collapse, num_ops_executed, executive_eval_results, entropy_at_collapse):
        """Monitors mood, attention, valence; adjusts thresholds (some here, some in review)."""
        if self.verbose >= 2: print(f"  META_LAYER.CognitiveParamUpdate:")
        self._log_lot_event("meta.cog_param_update.start", {})

        # Update Cognitive Load and Attention
        load_increase_factor = 0.15 # How much ORP/ops contribute to load
        load_from_orp = (orp_at_collapse / (self.E_OR_THRESHOLD + 1e-6)) * load_increase_factor
        load_from_ops = num_ops_executed * 0.020 # Each op adds a bit of load
        current_load = self.internal_state_parameters['cognitive_load']
        # Load decays, increases with activity
        new_load = current_load * 0.90 + load_from_orp + load_from_ops 
        self.internal_state_parameters['cognitive_load'] = np.clip(new_load, 0.0, 1.0)

        current_attention = self.internal_state_parameters['attention_level']
        # Attention drops with load, recovers over time, boosted by positive mood
        attention_decay_due_to_load = self.internal_state_parameters['cognitive_load'] * 0.10
        attention_recovery_rate = 0.06 # Natural recovery if not overloaded
        mood_effect_on_attention = self.internal_state_parameters['mood'] * 0.08 # Good mood = better attention
        
        new_attention = current_attention * (1 - attention_decay_due_to_load) + \
                        (1.0 - current_attention) * attention_recovery_rate + \
                        mood_effect_on_attention
        self.internal_state_parameters['attention_level'] = np.clip(new_attention, 0.1, 1.0) # Attention has a floor

        if self.verbose >=3: print(f"    CogLoad: {self.internal_state_parameters['cognitive_load']:.2f}, Attention: {self.internal_state_parameters['attention_level']:.2f}")

        # Dynamic update of Curiosity and Goal-Seeking Bias based on recent experience
        mod_valence = executive_eval_results['mod_valence']
        
        # Curiosity changes based on valence, entropy, and exploration mode
        # Negative valence or high entropy (if not fruitful) might boost curiosity to find alternatives.
        # Exploration mode directly boosts it.
        curiosity_change = 0
        if mod_valence < -0.3: curiosity_change += 0.03 # Seek novelty if unhappy
        if entropy_at_collapse > 1.5 and mod_valence < 0.1 : curiosity_change += 0.02 # High entropy but no reward -> explore
        if self.internal_state_parameters['exploration_mode_countdown'] > 0 : curiosity_change += 0.05 # Exploration itself sustains curiosity
        curiosity_change -= 0.01 # Natural decay of curiosity if nothing stimulates it
        self.internal_state_parameters['curiosity'] = np.clip(self.internal_state_parameters['curiosity'] + curiosity_change, 0.01, 0.99)

        # Goal-seeking bias changes based on valence, especially if a preferred state is set
        goal_bias_change = 0
        if self.internal_state_parameters['preferred_logical_state'] is not None:
            if mod_valence > 0.3: goal_bias_change += 0.04 # Reinforce goal seeking if yielding results
            else: goal_bias_change -=0.02 # Slightly reduce if not working
        else: # No specific goal, bias decays slowly
            goal_bias_change -= 0.01
        self.internal_state_parameters['goal_seeking_bias'] = np.clip(self.internal_state_parameters['goal_seeking_bias'] + goal_bias_change, 0.01, 0.99)

        if self.verbose >=3: print(f"    Curiosity: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        self._log_lot_event("meta.cog_param_update.end", {"cog_load":self.internal_state_parameters['cognitive_load'], "attn": self.internal_state_parameters['attention_level'], "cur":self.internal_state_parameters['curiosity'], "goal_bias":self.internal_state_parameters['goal_seeking_bias']})


    def _meta_layer_adapt_preferred_state(self, collapsed_outcome_str, mod_valence):
        """Part of Meta Layer: Adapts the agent's preferred logical state based on outcomes."""
        # This could be more sophisticated, e.g., based on surprise, utility, goal relevance.
        high_val_thresh = self.metacognition_params['high_valence_threshold']
        low_val_thresh = self.metacognition_params['low_valence_threshold']
        current_pref_state = self.internal_state_parameters['preferred_logical_state']
        
        pref_state_log_msg = ""

        # If current outcome is highly positive and different from current preference (or no preference), set it.
        if mod_valence >= high_val_thresh and current_pref_state != collapsed_outcome_str:
            self.internal_state_parameters['preferred_logical_state'] = collapsed_outcome_str
            # Boost goal seeking bias when a new promising state is found
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.25)
            # Reduce frustration if a good new preference is found
            self.internal_state_parameters['frustration'] *= 0.6 
            pref_state_log_msg = f"New preferred state |{collapsed_outcome_str}> set due to high valence ({mod_valence:.2f})."
        # If current outcome matches preference but valence is low, clear preference.
        elif mod_valence <= low_val_thresh and current_pref_state == collapsed_outcome_str:
            self.internal_state_parameters['preferred_logical_state'] = None
            # Reduce goal seeking bias if preferred state becomes aversive
            self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.2)
            # Increase curiosity if preferred state is disappointing
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.15)
            pref_state_log_msg = f"Preferred state |{collapsed_outcome_str}> cleared due to low valence ({mod_valence:.2f})."
        
        if pref_state_log_msg:
            if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState: {pref_state_log_msg}")
            self._log_lot_event("meta.adapt_pref_state", {"message": pref_state_log_msg, "new_pref_state": self.internal_state_parameters['preferred_logical_state']})


    def _meta_layer_perform_review(self):
        """Part of Meta Layer: Longer-term review and adaptation of core parameters (thresholds, decay, biases)."""
        if self.verbose >= 1: print(f"[{self.agent_id}] --- META_LAYER.Review (Cycle {self.current_cycle_num}) ---")
        self._log_lot_event("meta.review.start", {"cycle": self.current_cycle_num})

        # Use a longer span of history for more stable averages
        history_span_for_review = min(len(self.cycle_history), self.metacognition_params['review_interval'] * 2)
        if history_span_for_review < self.metacognition_params['review_interval'] / 2:
            if self.verbose >= 1: print("    META.Review: Insufficient history for meaningful review.")
            self._log_lot_event("meta.review.insufficient_history", {})
            self.metacognition_params['cycles_since_last_review'] = 0 # Reset counter
            return

        recent_history = list(self.cycle_history)[-history_span_for_review:]
        # Filter for valid cycles that have all necessary metrics
        valid_cycles_for_review = [
            c for c in recent_history if 
            c.get('collapsed_to') != "N/A" and 
            c.get('orp_at_collapse') is not None and
            c.get('valence_mod_this_cycle') is not None and
            c.get('entropy_at_collapse') is not None and
            c.get('num_ops_executed') is not None
        ]

        if not valid_cycles_for_review:
            if self.verbose >= 1: print("    META.Review: No valid cycles in recent history for review.")
            self._log_lot_event("meta.review.no_valid_cycles", {})
            self.metacognition_params['cycles_since_last_review'] = 0
            return
            
        avg_valence = np.mean([c['valence_mod_this_cycle'] for c in valid_cycles_for_review])
        avg_orp_at_collapse = np.mean([c['orp_at_collapse'] for c in valid_cycles_for_review])
        avg_entropy = np.mean([c['entropy_at_collapse'] for c in valid_cycles_for_review])
        avg_ops_per_cycle = np.mean([c['num_ops_executed'] for c in valid_cycles_for_review])
        # Calculate diversity of outcomes (how many unique states visited)
        outcome_diversity = len(set(c['collapsed_to'] for c in valid_cycles_for_review)) / len(valid_cycles_for_review) if valid_cycles_for_review else 0.0
        
        avg_metrics = {
            'avg_valence':avg_valence, 'avg_orp_at_collapse':avg_orp_at_collapse, 
            'avg_entropy':avg_entropy, 'avg_ops_per_cycle':avg_ops_per_cycle, 
            'outcome_diversity':outcome_diversity
        }
        if self.verbose >= 2: print(f"    META.Review Stats: AvgVal={avg_valence:.2f}, AvgORP={avg_orp_at_collapse:.3f}, AvgEnt={avg_entropy:.2f}, AvgOps={avg_ops_per_cycle:.1f}, Diversity={outcome_diversity:.2f}")
        self._log_lot_event("meta.review.stats", avg_metrics)


        # Adapt Curiosity and Goal Bias based on review
        # These use TRAINABLE adaptation rates from metacognition_params
        cur_adapt_rate = self.metacognition_params.get('curiosity_adaptation_rate', DEFAULT_METACOGNITION_PARAMS['curiosity_adaptation_rate'])
        if avg_valence < self.metacognition_params['low_valence_threshold'] or \
           outcome_diversity < self.metacognition_params['exploration_threshold_entropy']: # If stagnant or unhappy
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + cur_adapt_rate)
            if self.verbose >= 2: print(f"      META.Adapt: Curiosity increased to {self.internal_state_parameters['curiosity']:.2f}")
        elif avg_valence > self.metacognition_params['high_valence_threshold']: # If consistently successful, can reduce base curiosity
            self.internal_state_parameters['curiosity'] = max(0.01, self.internal_state_parameters['curiosity'] - cur_adapt_rate * 0.7)
            if self.verbose >= 2: print(f"      META.Adapt: Curiosity reduced to {self.internal_state_parameters['curiosity']:.2f}")

        goal_adapt_rate = self.metacognition_params.get('goal_bias_adaptation_rate', DEFAULT_METACOGNITION_PARAMS['goal_bias_adaptation_rate'])
        if avg_valence > self.metacognition_params['high_valence_threshold'] and self.internal_state_parameters['preferred_logical_state'] is not None:
            # If doing well and has a goal, reinforce goal-seeking behavior
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + goal_adapt_rate)
            if self.verbose >= 2: print(f"      META.Adapt: GoalBias increased to {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        elif avg_valence < self.metacognition_params['low_valence_threshold']:
            # If doing poorly, reduce stubbornness of goal seeking to allow exploration
            self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - goal_adapt_rate)
            if self.verbose >= 2: print(f"      META.Adapt: GoalBias reduced to {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        
        # Adapt E_OR_THRESHOLD, ORP_DECAY_RATE, COMPUTATION_LENGTH_PREFERENCE
        # These use TRAINABLE adaptation rates from their respective dynamic configs
        # E_OR_THRESHOLD Adaptation
        if self.metacognition_params.get('enable_threshold_adaptation', False):
            td = self.orp_threshold_dynamics 
            adapt_rate_thresh = td.get('adapt_rate', DEFAULT_ORP_THRESHOLD_DYNAMICS['adapt_rate'])
            # If collapsing too early (low ORP) and not achieving high valence -> need more "thought" -> increase threshold
            if avg_orp_at_collapse < self.E_OR_THRESHOLD * 0.6 and avg_valence < 0.1:
                self.E_OR_THRESHOLD = min(td['max'], self.E_OR_THRESHOLD + adapt_rate_thresh)
                if self.verbose >= 2: print(f"      META.Adapt: E_OR_THRESH increased to {self.E_OR_THRESHOLD:.3f}")
            # If collapsing too late (high ORP) consistently, or too many ops for little gain -> decrease threshold
            elif avg_orp_at_collapse > self.E_OR_THRESHOLD * 1.4 or (avg_ops_per_cycle > self.internal_state_parameters['computation_length_preference'] * 1.5 and avg_valence < 0.2):
                self.E_OR_THRESHOLD = max(td['min'], self.E_OR_THRESHOLD - adapt_rate_thresh)
                if self.verbose >= 2: print(f"      META.Adapt: E_OR_THRESH decreased to {self.E_OR_THRESHOLD:.3f}")

        # ORP_DECAY_RATE Adaptation
        if self.metacognition_params.get('enable_decay_adaptation', False):
            dd = self.orp_decay_dynamics
            adapt_rate_decay = dd.get('adapt_rate', DEFAULT_ORP_DECAY_DYNAMICS['adapt_rate'])
            # If consistently low valence -> less decay (allow ORP to build easier)
            if avg_valence < self.metacognition_params['low_valence_threshold'] * 0.8:
                self.orp_decay_rate = max(dd['min'], self.orp_decay_rate - adapt_rate_decay)
                if self.verbose >= 2: print(f"      META.Adapt: ORP_DECAY decreased to {self.orp_decay_rate:.4f}")
            # If consistently high valence -> can afford slightly more decay (prevents trivial ORP build-up)
            elif avg_valence > self.metacognition_params['high_valence_threshold'] * 0.8:
                self.orp_decay_rate = min(dd['max'], self.orp_decay_rate + adapt_rate_decay * 0.5) # Slower increase
                if self.verbose >= 2: print(f"      META.Adapt: ORP_DECAY increased to {self.orp_decay_rate:.4f}")
        
        # Computation Length Preference Adaptation
        if self.metacognition_params.get('enable_compref_adaptation', False):
            clp = self.internal_state_parameters['computation_length_preference']
            # If avg ops much lower than preference AND valence is bad -> agent isn't trying enough ops -> increase preference
            if avg_ops_per_cycle < clp * 0.7 and avg_valence < 0.0:
                self.internal_state_parameters['computation_length_preference'] = min(8, clp + 1) # Max preference of 8 ops
                if self.verbose >= 2: print(f"      META.Adapt: COMP_LENGTH_PREF increased to {clp + 1}")
            # If avg ops much higher than preference AND valence is bad -> agent is doing too much for no gain -> reduce preference
            elif avg_ops_per_cycle > clp * 1.3 and avg_valence < 0.0:
                self.internal_state_parameters['computation_length_preference'] = max(1, clp - 1) # Min preference of 1 op
                if self.verbose >= 2: print(f"      META.Adapt: COMP_LENGTH_PREF decreased to {clp - 1}")
        
        self.metacognition_params['cycles_since_last_review'] = 0
        if self.verbose >= 1: print(f"[{self.agent_id}] --- Metacognitive Review Complete ---")
        self._log_lot_event("meta.review.end", {})


    # --- Feature 3: Synaptic Mutation Network (SMN) Methods ---
    def _initialize_smn_params_state(self):
        """Initializes mutation strengths for SMN-controlled parameters."""
        smn_state = {}
        for param_key_in_trainable, smn_config in self.smn_controlled_params_config.items():
            smn_state[param_key_in_trainable] = {
                'current_mutation_strength': smn_config['base_mutation_strength'],
                'min_val': smn_config['min_val'],
                'max_val': smn_config['max_val'],
                'is_int': smn_config.get('is_int', False),
                'path': smn_config['path'] # e.g., ('internal_state_parameters', 'strategy_weights', 'curiosity')
            }
        return smn_state

    def _smn_update_and_apply_mutations(self, valence_mod_this_cycle, valence_raw_this_cycle, prev_cycle_valence_mod):
        """Applies Hebbian-like updates and mutations to SMN-controlled parameters."""
        if not self.smn_config.get('enabled', False): return

        valence_gain = valence_mod_this_cycle - prev_cycle_valence_mod
        smn_pos_thresh = self.internal_state_parameters['smn_positive_valence_threshold']
        smn_neg_thresh = self.internal_state_parameters['smn_negative_valence_threshold']
        
        if self.verbose >= 2: print(f"  SMN: ValenceMod={valence_mod_this_cycle:.2f}, Gain={valence_gain:.2f}")
        self._log_lot_event("smn.update.start", {"val_mod":valence_mod_this_cycle, "val_gain":valence_gain})


        for param_name, state in self.smn_params_state.items():
            current_strength = state['current_mutation_strength']
            
            # Adapt mutation strength
            if valence_mod_this_cycle > smn_pos_thresh : # Good outcome, stabilize (reduce mutation)
                current_strength *= self.internal_state_parameters['smn_mutation_strength_decay']
            elif valence_mod_this_cycle < smn_neg_thresh : # Bad outcome, destabilize (increase mutation)
                current_strength *= self.internal_state_parameters['smn_mutation_strength_grow']
            current_strength = np.clip(current_strength, 0.001, 0.5) # Bound strength
            state['current_mutation_strength'] = current_strength

            # Apply mutation if conditions met (e.g., significant positive valence gain implies current params are good to explore around)
            if valence_gain > self.smn_config.get('mutation_trigger_min_valence_gain', 0.1) and valence_mod_this_cycle > 0 : # Only mutate if there was improvement and current state is positive
                path = state['path']
                try:
                    # Get current value
                    target_obj = self
                    for key_part in path[:-1]: # Navigate to the dict containing the param
                        if key_part is None: continue # For direct attributes of self
                        target_obj = getattr(target_obj, key_part) if isinstance(key_part, str) else target_obj[key_part]
                    
                    param_key_final = path[-1]
                    current_val = target_obj[param_key_final] if isinstance(target_obj, dict) else getattr(target_obj, param_key_final)

                    # Perturbation logic
                    perturbation_magnitude = np.random.normal(0, current_strength * self.internal_state_parameters['smn_perturbation_scale_factor'])
                    # For strategy weights, perturbation should be relative to its typical scale (0-1)
                    # For other params, like comp_len_pref, perturbation scale might differ
                    # A universal smn_perturbation_scale_factor might need adjustment per parameter type or use param-specific scales
                    
                    new_val = current_val + perturbation_magnitude
                    
                    if state['is_int']: new_val = int(round(new_val))
                    new_val = np.clip(new_val, state['min_val'], state['max_val'])

                    # Set new value
                    if isinstance(target_obj, dict): target_obj[param_key_final] = new_val
                    else: setattr(target_obj, param_key_final, new_val)
                    
                    if self.verbose >= 2: print(f"    SMN Mutation: {param_name} to {new_val:.3f} (strength:{current_strength:.4f})")
                    self._log_lot_event("smn.update.mutation_applied", {"param":param_name, "new_val":new_val, "strength":current_strength})


                except (AttributeError, KeyError, IndexError) as e:
                    if self.verbose >=1: print(f"    SMN Error: Could not access/mutate param {param_name} at path {path}: {e}")
                    self._log_lot_event("smn.update.error", {"param":param_name, "path":path, "error":str(e)})
            
             # If param is a strategy weight, normalization will be needed after all SMN updates.
             # This is handled by update_emulator_parameters if SMN params are also trainable, or needs a specific call here.
             # For simplicity, assume normalization of strategy_weights happens if they are modified by other means,
             # or we add a call to normalize them after SMN loop if any sw_ was changed.
             # Currently, SMN uses paths from DEFAULT_SMN_CONTROLLED_PARAMS - if these affect strategy_weights, normalization happens
             # when the main training loop calls update_emulator_parameters, or must be called explicitly.
             # The provided DEFAULT_SMN_CONTROLLED_PARAMS includes 'sw_curiosity'. The logic in update_emulator_parameters handles this.
             # If SMN is enabled, consider re-normalizing strategy weights more directly after this block.
             # For now, let's assume periodic re-normalization or accept slight de-sync until next full update.

    # --- Feature 6: Cognitive Firewall ---
    def _firewall_detect_and_correct_anomalies(self):
        if not self.firewall_params.get('enabled', False): return
        if self.firewall_cooldown_remaining > 0:
            self.firewall_cooldown_remaining -= 1
            return
        
        self.firewall_cycles_since_last_check +=1
        if self.firewall_cycles_since_last_check < self.firewall_params['check_interval']:
            return

        self.firewall_cycles_since_last_check = 0
        intervention_made = False
        intervention_reason = "None"

        # 1. Persistent Low Valence
        if len(self.cycle_history) >= self.firewall_params['low_valence_streak_needed']:
            recent_valences = [c['valence_mod_this_cycle'] for c in list(self.cycle_history)[-self.firewall_params['low_valence_streak_needed']:]]
            if all(v < self.firewall_params['low_valence_threshold'] for v in recent_valences):
                intervention_reason = f"Persistent Low Valence (avg {np.mean(recent_valences):.2f} over {self.firewall_params['low_valence_streak_needed']} cycles)"
                self.internal_state_parameters['exploration_mode_countdown'] = max(
                    self.internal_state_parameters['exploration_mode_countdown'],
                    self.firewall_params['intervention_exploration_boost_duration']
                )
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.3)
                intervention_made = True

        # 2. State-Op Looping with Low Valence
        if not intervention_made and len(self.cycle_history) >= self.firewall_params['loop_detection_window']:
            history_slice = list(self.cycle_history)[-self.firewall_params['loop_detection_window']:]
            # (input_state, ops_tuple, outcome_state)
            state_op_outcome_sequences = [(c['actual_input_state_used'], tuple(c['ops_executed'] or []), c['collapsed_to']) for c in history_slice]
            counts = collections.Counter(state_op_outcome_sequences)
            for seq, count in counts.items():
                if count >= self.firewall_params['loop_detection_min_repeats']:
                    # Check valence during these specific loop cycles
                    loop_valences = [c['valence_mod_this_cycle'] for c, s_o_o in zip(history_slice, state_op_outcome_sequences) if s_o_o == seq]
                    if np.mean(loop_valences) < self.firewall_params['low_valence_threshold'] / 2 : # Even stricter for loops
                        intervention_reason = f"State-Op Loop Detected (sequence {seq} repeated {count} times with low valence)"
                        # More drastic: randomize strategy weights
                        sw = self.internal_state_parameters['strategy_weights']
                        rand_factor = self.firewall_params['intervention_strategy_randomness_factor']
                        for k in sw: sw[k] = sw[k] * (1-rand_factor) + random.random() * rand_factor
                        # Re-normalize (simple normalization, can be part of update_emulator_parameters call)
                        total_sw = sum(sw.values()); [ (sw.update({k:v/total_sw}) if total_sw > 1e-6 else sw.update({k:1.0/len(sw)})) for k,v in sw.items() ]
                        self.internal_state_parameters['preferred_logical_state'] = None
                        intervention_made = True
                        break
        
        # 3. Persistent Premature Collapse
        if not intervention_made and len(self.cycle_history) >= self.firewall_params['premature_collapse_streak_needed']:
            recent_orps = [c['orp_at_collapse'] for c in list(self.cycle_history)[-self.firewall_params['premature_collapse_streak_needed']:]]
            threshold_ratios = [orp / (c['E_OR_thresh_this_cycle']+1e-6) for orp, c in zip(recent_orps, list(self.cycle_history)[-self.firewall_params['premature_collapse_streak_needed']:])]
            if all(ratio < self.firewall_params['premature_collapse_orp_max_ratio'] for ratio in threshold_ratios):
                 intervention_reason = f"Persistent Premature ORP Collapse (avg ratio {np.mean(threshold_ratios):.2f} for {self.firewall_params['premature_collapse_streak_needed']} cycles)"
                 self.E_OR_THRESHOLD *= self.firewall_params['intervention_orp_threshold_increase_factor']
                 self.E_OR_THRESHOLD = min(self.E_OR_THRESHOLD, self.orp_threshold_dynamics['max']) # Cap at max
                 self.internal_state_parameters['computation_length_preference'] = max(self.internal_state_parameters['computation_length_preference'] + 1, 1)
                 intervention_made = True

        if intervention_made:
            if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL Activated: {intervention_reason}")
            self._log_lot_event("firewall.intervention", {"reason": intervention_reason})
            self.firewall_cooldown_remaining = self.firewall_params['cooldown_duration']
            # Frustration is often a root cause or result; alleviate it slightly.
            self.internal_state_parameters['frustration'] *= 0.5 
            # Any intervention might warrant clearing an immediate goal to re-evaluate
            if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
                if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL: Current goal '{self.current_goal_state_obj.current_goal}' put on hold.")
                self.current_goal_state_obj.status = "pending" # Or 'failed' if firewall is severe
                self.current_goal_state_obj.history.append({"cycle": self.current_cycle_num, "event": "firewall_interrupted_goal"})


    # --- Main Cognitive Cycle Orchestration ---
    def run_full_cognitive_cycle(self, intended_classical_input_str:str, computation_sequence_ops=None): # Added computation_sequence_ops override
        self.current_cycle_num += 1
        self.current_cycle_lot_stream = [] # Reset LoT stream for the new cycle
        start_time = time.time()
        
        self._log_lot_event("cycle_start", {"cycle_num":self.current_cycle_num, "intended_input": intended_classical_input_str, "agent_id":self.agent_id})
        if self.verbose >= 1: print(f"\n[{self.agent_id}] ===== Cycle {self.current_cycle_num} | Input: |{intended_classical_input_str}> =====")

        # --- 1. Sensor Layer ---
        actual_classical_input_str = self._sensor_layer_process_input(intended_classical_input_str)
        if self.verbose >= 2: print(f"  SensorLayer Out: Actual perceived input |{actual_classical_input_str}>")

        # --- Initial Executive Prep for Quantum Phase ---
        self._executive_prepare_superposition(actual_classical_input_str)

        # --- 2. Executive Layer: Op Generation ---
        # Ops can be provided externally (e.g., for testing, or from a higher controller)
        executed_sequence, chosen_op_strategy, op_gen_log_details = \
            self._executive_generate_computation_sequence(ops_provided_externally=computation_sequence_ops)
        
        if self.verbose >= 1 and computation_sequence_ops is None and executed_sequence: # Log if internally generated
             print(f"  ExecutiveLayer OpGen ({chosen_op_strategy}): {executed_sequence if executed_sequence else 'No ops'}")
        if self.verbose >=3 and op_gen_log_details: # Print detailed thought log for op_gen
            for line in op_gen_log_details: print(f"    OpGenLog: {line}")
        
        # --- 3. Executive Layer: Quantum Computation and Objective Reduction ---
        _, or_triggered_early = self._executive_quantum_computation_phase(executed_sequence)
        
        entropy_at_collapse = self._calculate_superposition_entropy() # Calculate entropy before actual collapse changes superposition
        num_superposition_terms = len([a for a in self.logical_superposition.values() if abs(a) > 1e-9]) # Before collapse
        
        collapsed_outcome_str = self._executive_trigger_objective_reduction()
        orp_at_collapse = self.current_orp_before_reset # Value of ORP right before it was reset by collapse
        
        if self.verbose >= 1: print(f"  ExecutiveLayer OR: Collapsed to |{collapsed_outcome_str}> (ORP experienced: {orp_at_collapse:.3f})")
        
        # --- Feature 4: Executive Layer: Collapse-Triggered Interrupt Handling ---
        # Use raw valence of outcome for immediate interrupt checks
        raw_valence_of_collapse = self.outcome_valence_map.get(collapsed_outcome_str, -0.1)
        self._executive_handle_collapse_interrupts(orp_at_collapse, executed_sequence, raw_valence_of_collapse)
        
        # --- 4. Executive Layer: Outcome Evaluation (ACC-like) & Mood Update ---
        # This uses/updates self.last_cycle_valence_raw and self.last_cycle_valence_mod
        executive_eval_results = self._executive_evaluate_outcome_and_update_mood(
            collapsed_outcome_str, orp_at_collapse, entropy_at_collapse, len(executed_sequence or [])
        )
        if self.verbose >= 1: print(f"  ExecutiveLayer Eval: Val(raw/mod): {self.last_cycle_valence_raw:.2f}/{self.last_cycle_valence_mod:.2f}. Mood: {self.internal_state_parameters['mood']:.2f}")
        if self.verbose >=3 and executive_eval_results.get('thoughts_log'):
            for line in executive_eval_results['thoughts_log']: print(f"    AccEvalLog: {line}")


        # --- Shared Attention Foci update (for Co-Agents, Feature 5) ---
        if self.last_cycle_valence_mod > 0.6 and self.shared_attention_foci is not None: # Threshold for "interesting enough to share"
            self.shared_attention_foci.append({'state': collapsed_outcome_str, 'op_seq': executed_sequence, 
                                               'valence': self.last_cycle_valence_mod, 'agent_id': self.agent_id,
                                               'cycle': self.current_cycle_num})
            self._log_lot_event("coagent.attention_share", {"state":collapsed_outcome_str, "valence":self.last_cycle_valence_mod})


        # --- 5. Associative Layer: LTM Update ---
        # Use consolidation factor from interrupts if set
        consolidation_bonus = self.smn_internal_flags.pop('ltm_consolidation_bonus_factor', 1.0)
        self._associative_layer_update_ltm(executed_sequence, self.last_cycle_valence_raw, orp_at_collapse, entropy_at_collapse, consolidation_factor=consolidation_bonus)
        if self.verbose >=2 and consolidation_bonus > 1.0 : print(f"  AssociativeLayer LTM Update used consolidation bonus: {consolidation_bonus:.1f}")


        # --- 6. Meta Layer: Cognitive Parameter Updates & Preferred State Adaptation ---
        self._meta_layer_update_cognitive_parameters(orp_at_collapse, len(executed_sequence or []), executive_eval_results, entropy_at_collapse)
        self._meta_layer_adapt_preferred_state(collapsed_outcome_str, self.last_cycle_valence_mod)
        if self.verbose >= 1: print(f"  MetaLayer State: Attn={self.internal_state_parameters['attention_level']:.2f},Cur={self.internal_state_parameters['curiosity']:.2f},PrefS=|{self.internal_state_parameters['preferred_logical_state']}>")
        
        # --- 7. Synaptic Mutation Network (SMN - Feature 3) ---
        # SMN requires previous cycle's valence to calculate gain
        prev_mod_valence = self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0.0
        self._smn_update_and_apply_mutations(self.last_cycle_valence_mod, self.last_cycle_valence_raw, prev_mod_valence)

        # --- 8. Cognitive Firewall (Feature 6) ---
        self._firewall_detect_and_correct_anomalies()
        
        # --- 9. Executive Layer: Plan Next Input (stored in self.next_target_input_state) ---
        planning_log = []
        self._executive_plan_next_target_input(collapsed_outcome_str, executive_eval_results, planning_log)
        if self.verbose >= 1: print(f"  ExecutiveLayer PlanNext: Proposing |{self.next_target_input_state}> for next cycle.")
        if self.verbose >=3 : 
            for line in planning_log: print(f"    PlanNextLog: {line}")

        # --- Meta Layer: Metacognitive Review (Periodic) ---
        self.metacognition_params['cycles_since_last_review'] += 1
        if self.metacognition_params['cycles_since_last_review'] >= self.metacognition_params['review_interval']:
            self._meta_layer_perform_review()

        # --- Temporal Feedback Grid Update (Feature 2) ---
        valence_delta = self.last_cycle_valence_mod - (self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0)
        entropy_shift = entropy_at_collapse - self.last_cycle_entropy_for_delta
        self.temporal_feedback_grid.append( (tuple(executed_sequence or []), valence_delta, entropy_shift) )
        self.last_cycle_entropy_for_delta = entropy_at_collapse # Store for next cycle's delta calculation

        # Store cycle results
        self.cycle_history.append({
            "cycle_num":self.current_cycle_num, "agent_id": self.agent_id,
            "intended_input_state":intended_classical_input_str, "actual_input_state_used":actual_classical_input_str,
            "ops_executed":executed_sequence, "op_strategy":chosen_op_strategy, "num_ops_executed":len(executed_sequence or []),
            "collapsed_to":collapsed_outcome_str, "orp_at_collapse":orp_at_collapse, "or_triggered_early": or_triggered_early,
            "num_terms_before_collapse":num_superposition_terms, "entropy_at_collapse":entropy_at_collapse,
            "valence_raw_this_cycle":self.last_cycle_valence_raw, "valence_mod_this_cycle":self.last_cycle_valence_mod,
            "mood_after_cycle":self.internal_state_parameters['mood'], "attention_after_cycle":self.internal_state_parameters['attention_level'],
            "cog_load_after_cycle":self.internal_state_parameters['cognitive_load'], "frustration_after_cycle":self.internal_state_parameters['frustration'],
            "curiosity_after_cycle":self.internal_state_parameters['curiosity'], "goal_bias_after_cycle":self.internal_state_parameters['goal_seeking_bias'],
            "preferred_state_after_cycle":self.internal_state_parameters['preferred_logical_state'], 
            "E_OR_thresh_this_cycle":self.E_OR_THRESHOLD, "orp_decay_this_cycle":self.orp_decay_rate,
            "exploration_mode_countdown_after_cycle": self.internal_state_parameters['exploration_mode_countdown'],
            "smn_flags_in_cycle": copy.deepcopy(self.smn_internal_flags), # Log SMN effects for debug
            "goal_state_at_cycle_end": self.current_goal_state_obj.to_dict() if self.current_goal_state_obj else None,
            "lot_stream_this_cycle": copy.deepcopy(self.current_cycle_lot_stream) # Feature 8
        })
        
        cycle_duration = time.time() - start_time
        self._log_lot_event("cycle_end", {"duration_ms": cycle_duration * 1000, "next_planned_input": self.next_target_input_state})
        if self.verbose >= 1: print(f"[{self.agent_id}] ===== Cycle {self.current_cycle_num} End (Dur: {cycle_duration:.3f}s, Next: |{self.next_target_input_state}>) =====")
        
        return self.next_target_input_state # Return the planned input for the next cycle

    # --- Helper & Utility ---
    def logical_superposition_str(self):
        active_terms = []
        for state, amp in self.logical_superposition.items():
            if abs(amp) > 1e-9:
                term_str = ""
                if abs(amp.real) > 1e-9: term_str += f"{amp.real:.2f}"
                if abs(amp.imag) > 1e-9: term_str += f"{amp.imag:+.2f}j" if amp.real !=0 or amp.imag < 0 else f"{amp.imag:.2f}j"
                if not term_str: term_str="0" # handles pure zero if somehow passes abs(amp) > 1e-9
                active_terms.append(f"{term_str} |{state}>")
        return ", ".join(active_terms) if active_terms else "ZeroSuperposition"

    def set_goal_state(self, goal_state_obj: GoalState):
        """Externally set a high-level goal for the agent (Feature 7)."""
        if not isinstance(goal_state_obj, GoalState) and goal_state_obj is not None:
            raise ValueError("goal_state_obj must be an instance of GoalState or None.")
        self.current_goal_state_obj = goal_state_obj
        if self.current_goal_state_obj:
            self.current_goal_state_obj.status = "active" # Ensure it's active
            if self.verbose >= 1: print(f"[{self.agent_id}] New goal set: {self.current_goal_state_obj}")
            self._log_lot_event("goal.set", {"goal":self.current_goal_state_obj.current_goal, "num_steps":len(self.current_goal_state_obj.steps)})


    def print_internal_state_summary(self, indent="  ", custom_logger=None):
        log_func = custom_logger if custom_logger else print
        
        log_func(f"{indent}--- Internal State Summary for Agent {self.agent_id} (Cycle {self.current_cycle_num}) ---")
        # Core emotional/attentional states
        log_func(f"{indent}  Mood: {self.internal_state_parameters['mood']:.2f}, Attention: {self.internal_state_parameters['attention_level']:.2f}, CogLoad: {self.internal_state_parameters['cognitive_load']:.2f}")
        log_func(f"{indent}  Frustration: {self.internal_state_parameters['frustration']:.2f}, ExplorationCountdown: {self.internal_state_parameters['exploration_mode_countdown']}")
        # Key cognitive biases and preferences
        log_func(f"{indent}  Curiosity: {self.internal_state_parameters['curiosity']:.2f}, GoalSeekingBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        log_func(f"{indent}  PreferredState: |{self.internal_state_parameters['preferred_logical_state']}>, CompLengthPref: {self.internal_state_parameters['computation_length_preference']}")
        # Orch OR dynamics
        log_func(f"{indent}  E_OR_THRESH: {self.E_OR_THRESHOLD:.3f} (AdaptRate: {self.orp_threshold_dynamics['adapt_rate']:.3f})")
        log_func(f"{indent}  ORP_DECAY: {self.orp_decay_rate:.4f} (AdaptRate: {self.orp_decay_dynamics['adapt_rate']:.4f})")
        # Metacognition params
        log_func(f"{indent}  MC ReviewInterval: {self.metacognition_params['review_interval']}, CyclesSinceReview: {self.metacognition_params['cycles_since_last_review']}")
        log_func(f"{indent}  MC AdaptRates (Cur/Goal): {self.metacognition_params['curiosity_adaptation_rate']:.3f}/{self.metacognition_params['goal_bias_adaptation_rate']:.3f}")
        # LTM state
        log_func(f"{indent}  LTM entries: {len(self.long_term_memory)}/{self.long_term_memory_capacity}, UtilWeights(V/E): {self.ltm_utility_weight_valence:.2f}/{self.ltm_utility_weight_efficiency:.2f}")
        # Temporal Grid
        log_func(f"{indent}  TemporalGrid size: {len(self.temporal_feedback_grid)}/{self.temporal_grid_params['max_len']}")
        # SMN Controlled Params (example, could be many)
        if self.smn_config.get('enabled'):
            log_func(f"{indent}  SMN Active. Example controlled param strengths:")
            for p_name, p_state in list(self.smn_params_state.items())[:2]: # Print first 2 for brevity
                log_func(f"{indent}    {p_name}: strength {p_state['current_mutation_strength']:.4f}")
        # Firewall state
        if self.firewall_params.get('enabled'):
            log_func(f"{indent}  Firewall Active. Cooldown: {self.firewall_cooldown_remaining}, CheckIn: {self.firewall_params['check_interval'] - self.firewall_cycles_since_last_check}")
        # Goal State
        if self.current_goal_state_obj:
            log_func(f"{indent}  Current Goal: {self.current_goal_state_obj}")
        else:
            log_func(f"{indent}  Current Goal: None")
        log_func(f"{indent}--- End of Summary for Agent {self.agent_id} ---")


    def run_chained_cognitive_cycles(self, initial_input_str, num_cycles, computation_sequence_ops_template=None):
        if self.verbose >= 0: print(f"\n\n[{self.agent_id}] %%%%% STARTING CHAINED CYCLES ({num_cycles}) %%%%%")
        current_input_str = initial_input_str
        self.next_target_input_state = initial_input_str # Initialize next target

        for i in range(num_cycles):
            # Use self.next_target_input_state as the intended input, which was planned by the previous cycle
            current_input_str_for_cycle = self.next_target_input_state
            
            if self.verbose >= 1:
                pref_str = f"|{self.internal_state_parameters['preferred_logical_state']}>" if self.internal_state_parameters['preferred_logical_state'] else "None"
                goal_str = f"Goal: '{self.current_goal_state_obj.current_goal}'" if self.current_goal_state_obj else "No Goal"
                print(f"\n>>>> Chained Cycle {i+1}/{num_cycles} for {self.agent_id} <<<< Intended: |{current_input_str_for_cycle}>; Pref:{pref_str}; {goal_str}")
            
            current_comp_ops = None # By default, let agent generate ops
            if isinstance(computation_sequence_ops_template, list) and computation_sequence_ops_template:
                current_comp_ops = computation_sequence_ops_template[i % len(computation_sequence_ops_template)]
            elif callable(computation_sequence_ops_template): # For dynamic op provision
                 current_comp_ops = computation_sequence_ops_template(self) # Pass agent instance to callable

            try:
                # run_full_cognitive_cycle now returns the *next* planned input
                # and uses current_input_str_for_cycle as its current input
                next_planned_input = self.run_full_cognitive_cycle(current_input_str_for_cycle, current_comp_ops)
                # self.next_target_input_state is already updated internally by run_full_cognitive_cycle
                
                if next_planned_input is None: # Should not happen with new design
                    print(f"CRITICAL ERROR: Cycle returned None for next input! Stopping agent {self.agent_id}.")
                    self._log_lot_event("error.critical", {"message":"cycle_returned_none_next_input"})
                    break
            except Exception as e:
                print(f"CRITICAL EXCEPTION in cycle {i+1} for agent {self.agent_id}: {e}. Stopping.")
                traceback.print_exc()
                self._log_lot_event("error.critical_exception", {"cycle":i+1, "error":str(e), "traceback":traceback.format_exc()})
                break
        
        if self.verbose >= 0: print(f"\n[{self.agent_id}] %%%%% CHAINED CYCLES COMPLETED ({self.current_cycle_num} total cycles for this agent run) %%%%%"); self.print_internal_state_summary(indent=f"  Final State ({self.agent_id}) ")


# ---------------------------------------------------------------------------
# CoAgentManager (Feature 5: Parallel Cognitive Threads)
# ---------------------------------------------------------------------------
class CoAgentManager:
    def __init__(self, num_agents, base_emulator_config_template, agent_config_variations_list=None, verbose=0):
        self.num_agents = num_agents
        self.base_config = base_emulator_config_template # Template for agent config
        self.agent_variations = agent_config_variations_list or [] # List of dicts for overrides per agent
        self.verbose = verbose

        self.shared_long_term_memory = {} # Common LTM for all agents
        self.shared_attention_foci = collections.deque(maxlen=50) # Tuples of (state, op_seq, valence, agent_id) from interesting collapses

        self.agents = []
        self.agent_performance_history = collections.defaultdict(lambda: collections.deque(maxlen=20)) # agent_id -> deque of recent valences

        self.system_cycle_num = 0

        self._initialize_agents()

        if self.verbose >= 0: print(f"CoAgentManager Initialized with {self.num_agents} agents.")

    def _initialize_agents(self):
        for i in range(self.num_agents):
            agent_id = f"agent{i}"
            # Start with a deep copy of the base config
            agent_specific_config = copy.deepcopy(self.base_config)
            
            # Apply variations if provided for this agent index
            agent_overrides = {}
            if i < len(self.agent_variations):
                agent_custom_settings = self.agent_variations[i] # This should be a dict of {'path_tuple': value, ...} for _apply_config_overrides
                agent_overrides = agent_custom_settings.get('config_overrides', {})
                # Also merge/override dicts like initial_internal_states, metacognition_config if specified directly in variations
                # For simplicity, current _apply_config_overrides handles direct path based overrides best.

            # Pass shared resources to the agent constructor
            agent_specific_config['agent_id'] = agent_id
            agent_specific_config['shared_long_term_memory'] = self.shared_long_term_memory
            agent_specific_config['shared_attention_foci'] = self.shared_attention_foci # Agents can read this
            agent_specific_config['config_overrides'] = agent_overrides
            agent_specific_config['verbose'] = agent_specific_config.get('verbose', self.verbose-1 if self.verbose > 0 else 0) # Agents usually less verbose than manager
            
            try:
                emulator = SimplifiedOrchOREmulator(**agent_specific_config)
                self.agents.append(emulator)
                if self.verbose >= 1: print(f"  Initialized {agent_id} with specific config variations.")
                if self.verbose >=2 and agent_overrides: emulator.print_internal_state_summary()
            except Exception as e:
                print(f"ERROR Initializing {agent_id}: {e}")
                traceback.print_exc()


    def run_system_cycles(self, num_system_cycles, initial_input_per_agent_list=None):
        if self.verbose >= 0: print(f"\n\n========= CoAgentManager: Starting {num_system_cycles} System Cycles =========")
        
        for i in range(num_system_cycles):
            self.system_cycle_num += 1
            if self.verbose >=0: print(f"\n------- System Cycle {self.system_cycle_num}/{num_system_cycles} -------")

            # Each agent runs one cognitive cycle
            for agent_idx, agent in enumerate(self.agents):
                # Determine input for this agent in this system cycle
                # Agents plan their own next input internally via self.next_target_input_state
                # which is set at the end of their previous full_cognitive_cycle.
                current_agent_input = agent.next_target_input_state
                if initial_input_per_agent_list and self.system_cycle_num == 1 and agent_idx < len(initial_input_per_agent_list):
                    current_agent_input = initial_input_per_agent_list[agent_idx]
                    agent.next_target_input_state = current_agent_input # Prime its first input
                
                if self.verbose >=1: print(f"  Running {agent.agent_id} with intended input |{current_agent_input}>")
                try:
                    # Agent runs its cycle, its next_target_input_state will be updated internally
                    agent.run_full_cognitive_cycle(current_agent_input)
                    
                    # Store performance for inter-agent learning
                    if agent.cycle_history:
                        self.agent_performance_history[agent.agent_id].append(agent.cycle_history[-1]['valence_mod_this_cycle'])
                except Exception as e:
                     print(f"  ERROR during cycle for {agent.agent_id}: {e}. Agent may be unstable.")
                     traceback.print_exc()
                     # Optionally remove or reset problematic agent
            
            # Perform inter-agent interactions (e.g., learning, competition - simplified for now)
            if self.system_cycle_num % 5 == 0: # Perform interaction every 5 system cycles
                self._perform_inter_agent_learning()
        
        if self.verbose >= 0: print(f"\n========= CoAgentManager: System Cycles Completed ({self.system_cycle_num} total) =========")
        self.print_system_summary()

    def _perform_inter_agent_learning(self):
        """Facilitates learning or strategy copying between agents."""
        if len(self.agents) < 2: return # Needs at least two agents to interact

        if self.verbose >= 1: print(f"\n  --- CoAgentManager: Performing Inter-Agent Learning (System Cycle {self.system_cycle_num}) ---")

        avg_performances = []
        for agent in self.agents:
            if self.agent_performance_history[agent.agent_id]:
                avg_perf = np.mean(list(self.agent_performance_history[agent.agent_id]))
                avg_performances.append({'agent_id': agent.agent_id, 'perf': avg_perf, 'agent_obj': agent})
            else:
                avg_performances.append({'agent_id': agent.agent_id, 'perf': -float('inf'), 'agent_obj': agent}) # No history, low perf

        if not avg_performances: return

        avg_performances.sort(key=lambda x: x['perf'], reverse=True) # Sort best to worst
        
        best_agent_data = avg_performances[0]
        worst_agent_data = avg_performances[-1]

        if self.verbose >= 2:
            print(f"    Agent Performance Ranking (avg recent valence):")
            for p_data in avg_performances: print(f"      {p_data['agent_id']}: {p_data['perf']:.3f}")

        # Simple learning: worst N agents try to copy some parameters from the best agent
        # For now, just one worst agent copies one best agent if performance gap is significant
        copy_threshold_factor = 0.3 # e.g. if worst is 30% less than best
        num_agents_to_copy = min(len(self.agents) // 3, 3) # up to 1/3rd of agents or max 3 copy

        copied_something = False
        for i in range(num_agents_to_copy):
            idx_to_copy = len(avg_performances) -1 - i # get from the worst end
            if idx_to_copy <= 0 : break # don't let best agent copy itself or worse copy better
            
            learner_data = avg_performances[idx_to_copy]
            teacher_data = best_agent_data # Always learn from the current best

            if teacher_data['agent_id'] == learner_data['agent_id']: continue # Don't copy self

            # Copy if performance significantly different
            if teacher_data['perf'] > learner_data['perf'] + abs(teacher_data['perf'] * copy_threshold_factor) : # Teacher needs to be substantially better
                learner_agent = learner_data['agent_obj']
                teacher_agent = teacher_data['agent_obj']

                # Parameters to potentially copy/align (e.g., strategy weights, curiosity)
                params_to_align = {
                    'strategy_weights': ('internal_state_parameters','strategy_weights'), # path to dict
                    'curiosity': ('internal_state_parameters','curiosity'), # path to key
                    'goal_seeking_bias': ('internal_state_parameters', 'goal_seeking_bias')
                }
                alignment_factor = 0.15 # How much to nudge towards teacher's params (0 to 1)

                if self.verbose >= 1: print(f"    {learner_agent.agent_id} (perf {learner_data['perf']:.2f}) learning from {teacher_agent.agent_id} (perf {teacher_data['perf']:.2f})")

                for param_key, path_tuple in params_to_align.items():
                    try:
                        # Get teacher's value
                        teacher_val_obj = teacher_agent
                        for p_item in path_tuple[:-1]: teacher_val_obj = getattr(teacher_val_obj, p_item)
                        teacher_param_val = teacher_val_obj[path_tuple[-1]] if isinstance(teacher_val_obj, dict) else getattr(teacher_val_obj, path_tuple[-1])
                        
                        # Get learner's value and update it
                        learner_val_obj = learner_agent
                        for p_item in path_tuple[:-1]: learner_val_obj = getattr(learner_val_obj, p_item)
                        
                        if isinstance(teacher_param_val, dict): # e.g. strategy_weights
                            current_learner_dict = learner_val_obj[path_tuple[-1]]
                            for sub_key, t_val in teacher_param_val.items():
                                l_val = current_learner_dict.get(sub_key, t_val) # Default to teacher if learner missing
                                current_learner_dict[sub_key] = l_val * (1-alignment_factor) + t_val * alignment_factor
                            # Needs normalization if strategy_weights
                            if path_tuple[-1] == 'strategy_weights':
                                sw = current_learner_dict; total_sw = sum(sw.values())
                                [ (sw.update({k_sw:v_sw/total_sw}) if total_sw > 1e-6 else sw.update({k_sw:1.0/len(sw)})) for k_sw,v_sw in sw.items() ]
                        elif isinstance(teacher_param_val, (float, int)): # e.g. curiosity
                            learner_curr_val = learner_val_obj[path_tuple[-1]] if isinstance(learner_val_obj, dict) else getattr(learner_val_obj, path_tuple[-1])
                            new_learner_val = learner_curr_val * (1-alignment_factor) + teacher_param_val * alignment_factor
                            if isinstance(learner_val_obj, dict): learner_val_obj[path_tuple[-1]] = new_learner_val
                            else: setattr(learner_val_obj, path_tuple[-1], new_learner_val)
                        
                        copied_something = True
                        if self.verbose >= 2: print(f"      {learner_agent.agent_id} aligned {param_key} towards {teacher_agent.agent_id}'s value.")

                    except (AttributeError, KeyError) as e:
                        if self.verbose >=1: print(f"      Error aligning {param_key} for {learner_agent.agent_id}: {e}")
                
                if copied_something:
                     learner_agent._log_lot_event("coagent.learn_from_peer", {"teacher_id": teacher_agent.agent_id, "learner_perf":learner_data['perf'], "teacher_perf":teacher_data['perf']})


        if not copied_something and self.verbose >=1: print("    No significant performance gaps for agent learning this interaction step.")
        if self.verbose >=1: print("  --- CoAgentManager: Inter-Agent Learning Complete ---")

    def print_system_summary(self):
        print(f"\n--- CoAgentManager System Summary (After {self.system_cycle_num} system cycles) ---")
        print(f"  Number of Agents: {self.num_agents}")
        print(f"  Shared LTM Size: {len(self.shared_long_term_memory)}")
        print(f"  Shared Attention Foci Queue Size: {len(self.shared_attention_foci)}")
        if self.shared_attention_foci:
             print(f"    Last shared focus: {self.shared_attention_foci[-1]}")
        
        print("\n  Individual Agent Summaries:")
        for agent in self.agents:
            agent.print_internal_state_summary(indent="    ", custom_logger=print) # use manager's print
            print("-" * 20)


# ---------------------------------------------------------------------------
# Cognitive Agent Trainer (largely unchanged but interacts with more complex agent)
# ---------------------------------------------------------------------------
class CognitiveAgentTrainer:
    def __init__(self, trainable_params_config, base_emulator_config=None, verbose=0):
        self.trainable_params_config = copy.deepcopy(trainable_params_config)
        # Initialize current_params from the 'initial' values in the config
        self.current_params = {name: config['initial'] for name, config in trainable_params_config.items()}
        self.base_emulator_config = base_emulator_config if base_emulator_config else {}
        self.verbose = verbose
        self.best_params = copy.deepcopy(self.current_params)
        self.best_reward = -float('inf')
        self.training_log = []

    def _get_emulator_init_args(self, current_run_params):
        """ Prepares kwargs for emulator instantiation based on current_run_params. """
        init_args = copy.deepcopy(self.base_emulator_config)
        
        # The SimpliedOrchOREmulator's __init__ now has a `trainable_param_values` argument
        # which internally uses DEFAULT_TRAINABLE_PARAMS_CONFIG for mapping.
        # So, we can directly pass current_run_params to it.
        init_args['trainable_param_values'] = current_run_params
        
        # Verbosity for emulator during training runs can be set here if needed
        init_args['verbose'] = self.base_emulator_config.get('verbose_emulator_episodes', self.verbose -1 if self.verbose > 0 else 0)
        
        # Specific task-related setup (e.g., outcome_valence_map, initial preferred_state for training)
        # These are often part of base_emulator_config and handled by SimplifiedOrchOREmulator __init__ or _apply_config_overrides
        if 'outcome_valence_map' in self.base_emulator_config:
            # This will be handled by emulator if passed in base_config, e.g. through 'config_overrides'
            # or directly if constructor accepts it. Let's ensure constructor uses outcome_valence_map if passed in initial_internal_states or equivalent.
            # Current design: outcome_valence_map is an attribute. A bit of a hack:
            # Store it for after init for now. A cleaner way is path override in config.
            pass


        return init_args

    def run_episode(self, episode_params, num_cycles, initial_input="00", task_goal_state=None):
        """Runs one episode with the emulator using a given set of parameters."""
        emulator_kwargs = self._get_emulator_init_args(episode_params)
        emulator = SimplifiedOrchOREmulator(**emulator_kwargs)

        # Apply outcome_valence_map if it was in base_config (temporary hack)
        if 'outcome_valence_map' in self.base_emulator_config:
             emulator.outcome_valence_map = copy.deepcopy(self.base_emulator_config['outcome_valence_map'])
        if 'preferred_logical_state' in self.base_emulator_config.get('initial_internal_states', {}): # set if defined for training task
            emulator.internal_state_parameters['preferred_logical_state'] = self.base_emulator_config['initial_internal_states']['preferred_logical_state']

        if task_goal_state:
            emulator.set_goal_state(copy.deepcopy(task_goal_state)) # Give agent the training goal

        # Run the cycles
        emulator.run_chained_cognitive_cycles(initial_input_str=initial_input, num_cycles=num_cycles)
        
        # Calculate reward
        if not emulator.cycle_history: return 0.0, emulator.cycle_history, None
        
        avg_mod_valence = np.mean([c['valence_mod_this_cycle'] for c in emulator.cycle_history if c.get('valence_mod_this_cycle') is not None] or [0])
        reward = avg_mod_valence

        # Goal-based reward for trainer
        goal_reward_bonus = 0.0
        if task_goal_state and emulator.current_goal_state_obj:
            final_goal_state = emulator.current_goal_state_obj
            if final_goal_state.status == 'completed':
                goal_reward_bonus = 0.5  # Significant bonus for completing goal
            elif final_goal_state.status == 'failed':
                goal_reward_bonus = -0.3 # Penalty for failing
            reward += goal_reward_bonus * (1 + final_goal_state.progress) # Weight bonus by progress
        
        return reward, emulator.cycle_history, emulator.current_goal_state_obj


    def train(self, num_training_episodes, cycles_per_episode, initial_input="00",
              learning_rate_decay=0.99, training_goal_state_template=None):
        if self.verbose >= 0: print(f"\n--- Starting Training ({num_training_episodes} episodes, {cycles_per_episode} cycles/ep) ---")

        current_perturb_scales = {name: config['perturb_scale'] for name, config in self.trainable_params_config.items()}

        for episode_num in range(num_training_episodes):
            candidate_params = copy.deepcopy(self.best_params)
            for name, config in self.trainable_params_config.items():
                if config.get('fixed', False): continue 
                perturb = np.random.normal(0, current_perturb_scales[name])
                candidate_params[name] += perturb
                candidate_params[name] = np.clip(candidate_params[name], config['min'], config['max'])
            
            if self.verbose >= 1: print(f"\nEp {episode_num + 1}/{num_training_episodes} with candidate params...")
            if self.verbose >= 2:
                print("  Candidate params for this episode:")
                for pname, pval in candidate_params.items(): print(f"    {pname}: {pval:.4f}")
            
            current_episode_goal = None
            if training_goal_state_template: # Create a fresh goal for each episode
                 current_episode_goal = GoalState(**training_goal_state_template)


            reward, history, final_goal = self.run_episode(candidate_params, cycles_per_episode, initial_input, task_goal_state=current_episode_goal)

            log_entry = {
                'episode': episode_num + 1, 'params_tried': copy.deepcopy(candidate_params),
                'reward': reward, 'best_reward_so_far': self.best_reward,
                'goal_status': final_goal.status if final_goal else "N/A",
                'goal_progress': final_goal.progress if final_goal else 0.0
            }

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_params = copy.deepcopy(candidate_params)
                log_entry['action'] = "New best found"
                if self.verbose >= 1: print(f"  Ep {episode_num+1}: New best reward! {reward:.4f}. Goal: {final_goal.status if final_goal else 'N/A'} ({final_goal.progress*100 if final_goal else 0:.0f}%)")
            else:
                log_entry['action'] = "Kept previous best"
                if self.verbose >= 1: print(f"  Ep {episode_num+1}: Reward {reward:.4f}. No improvement (Best: {self.best_reward:.4f}). Goal: {final_goal.status if final_goal else 'N/A'} ({final_goal.progress*100 if final_goal else 0:.0f}%)")
            
            for name in current_perturb_scales:
                 if not self.trainable_params_config[name].get('fixed', False):
                    current_perturb_scales[name] *= learning_rate_decay
                    current_perturb_scales[name] = max(current_perturb_scales[name], 0.0001)

            log_entry['current_best_params'] = copy.deepcopy(self.best_params)
            self.training_log.append(log_entry)
            if self.verbose >=1 : self.print_best_params(prefix=f"  Ep {episode_num+1} Best ")

        if self.verbose >= 0:
            print(f"\n--- Training Complete ---")
            self.print_best_params(prefix="Final Best ")
            print(f"  Achieved best reward: {self.best_reward:.4f}")
        return self.best_params, self.best_reward, self.training_log

    def print_best_params(self, prefix=""):
        if self.verbose >= 1:
            print(f"{prefix}Parameters (Reward: {self.best_reward:.4f}):")
            for name, value in self.best_params.items():
                config = self.trainable_params_config.get(name, {})
                val_str = f"{int(value)}" if config.get('is_int') else f"{value:.4f}"
                print(f"  {prefix}  {name}: {val_str}")

# ---------------------------------------------------------------------------
# Main Execution Block (Demos - Significantly Adapted)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(int(time.time()) % (2**32 -1) ) # Use time based seed for more variability if re-run
    random.seed(int(time.time()*1000) % (2**32 -1))   
    
    MASTER_VERBOSE_LEVEL = 1 # 0: quiet, 1: basic, 2: detail, 3: very detail

    print("\n\n--- DEMO 0: Cortical Layers & Internal Language (Feature 1 & 8) ---")
    # This demo focuses on showing the structure and the LoT output for a few cycles.
    demo0_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'cycle_history_max_len': 5,
        'initial_E_OR_THRESHOLD': 0.6,
        'lot_config': {'enabled': True, 'log_level_details': { # Enable most LoT logs for this demo
            'cycle_start': True, 'sensor_input': True, 'op_generation': True, 'executive.opgen.strategy_selected':True, 'executive.opgen.temporal_bias':True,
            'op_execution': True, 'collapse_event': True, 'valence_eval': True, 'ltm_update': True, 'internal_state_updates': True,
            'goal_tracking': True, 'firewall_action': True, 'smn_action': True, 'interrupt_action': True,
            'meta.cog_param_update': True, 'meta.review': True, 'cycle_end': True, 'coagent': True, 'error':True,
            'executive.plannext': True, 'executive.objective_reduction': True, 'executive.quantum_comp':True
            }
        },
        'smn_config': {'enabled': False}, # Keep SMN off for simple demo
        'cognitive_firewall_config': {'enabled': False}, # Firewall off
    }
    emulator_demo0 = SimplifiedOrchOREmulator(agent_id="agent_f1_f8", **demo0_config)
    emulator_demo0.internal_state_parameters['preferred_logical_state'] = "11"
    print(f"Running agent {emulator_demo0.agent_id} for 3 cycles to show Layered processing and LoT.")
    emulator_demo0.run_chained_cognitive_cycles("00", 3)
    print(f"\n{emulator_demo0.agent_id} LoT for last cycle (Cycle {emulator_demo0.current_cycle_num}):")
    if emulator_demo0.cycle_history:
        for lot_entry in emulator_demo0.cycle_history[-1]['lot_stream_this_cycle']:
            print(f"  LoT: {lot_entry}")
    emulator_demo0.print_internal_state_summary()


    print("\n\n--- DEMO 1: Temporal Feedback Grid & SMN (Feature 2 & 3) ---")
    demo1_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.8,
        'temporal_grid_config': {'max_len': 5}, # Short grid for demo
        'smn_config': {'enabled': True, 'mutation_trigger_min_valence_gain': 0.05},
        # Let SMN control computation_length_preference to see it change
        'smn_controlled_params_config': {
             'computation_length_preference': {'base_mutation_strength': 0.5, 'min_val': 1, 'max_val': 5, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')}
        },
        'cognitive_firewall_config': {'enabled': False}, # Keep other advanced features simpler
    }
    emulator_demo1 = SimplifiedOrchOREmulator(agent_id="agent_f2_f3", **demo1_config)
    initial_comp_len = emulator_demo1.internal_state_parameters['computation_length_preference']
    print(f"Agent {emulator_demo1.agent_id} starting with CompLengthPref: {initial_comp_len}. Running 15 cycles.")
    # Use a valence map that will likely cause fluctuations for SMN/TemporalGrid to react
    emulator_demo1.outcome_valence_map = {"00": -0.5, "01": 0.7, "10": -0.2, "11": 0.3} 
    emulator_demo1.run_chained_cognitive_cycles("00", 15)
    print(f"\n{emulator_demo1.agent_id} final CompLengthPref: {emulator_demo1.internal_state_parameters['computation_length_preference']}")
    print(f"{emulator_demo1.agent_id} Temporal Feedback Grid content (last few):")
    for item in list(emulator_demo1.temporal_feedback_grid)[-3:]:
        print(f"  Grid Entry: Ops={item[0]}, ValDelta={item[1]:.2f}, EntShift={item[2]:.2f}")
    emulator_demo1.print_internal_state_summary()


    print("\n\n--- DEMO 2: Interrupt Handlers & Cognitive Firewall (Feature 4 & 6) ---")
    demo2_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.5, # Lower threshold to make anomalies more likely
        'interrupt_handler_config': {'enabled': True, 'reactive_ltm_valence_threshold': -0.8},
        'cognitive_firewall_config': {'enabled': True, 'check_interval': 3, 'low_valence_threshold': -0.7, 'low_valence_streak_needed': 2, 'cooldown_duration': 5},
        'smn_config': {'enabled': False},
        'outcome_valence_map': {"00": -0.9, "01": -0.8, "10": -0.9, "11": 0.1}, # Very punishing map
         'lot_config': {'enabled': True, 'log_level_details':{'firewall_action':True, 'interrupt_action':True, 'cycle_start':True}}
    }
    emulator_demo2 = SimplifiedOrchOREmulator(agent_id="agent_f4_f6", **demo2_config)
    # Add a "fix-it" sequence to LTM for the reactive interrupt
    fix_it_seq = (('H',0),('H',1))
    emulator_demo2.long_term_memory[fix_it_seq] = {
        'count':5, 'total_valence': 3.5, 'avg_valence':0.7, 'total_orp_cost':0.5, 'avg_orp_cost':0.1,
        'total_entropy_generated':1.0, 'avg_entropy':0.2, 'utility':0.65, 'last_cycle':0
    }
    print(f"Agent {emulator_demo2.agent_id} running 12 cycles with punishing valence. Expect Firewall/Interrupts.")
    emulator_demo2.run_chained_cognitive_cycles("00", 12)
    emulator_demo2.print_internal_state_summary()


    print("\n\n--- DEMO 3: Goal-Oriented State Machine (Feature 7) ---")
    demo3_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'lot_config': {'enabled': True, 'log_level_details': {'goal_tracking': True, 'executive.opgen.strategy_selected':True}}
    }
    emulator_demo3 = SimplifiedOrchOREmulator(agent_id="agent_f7", **demo3_config)
    # Define a simple goal
    goal_steps = [
        {"name": "Achieve state 01", "target_state": "01"},
        {"name": "Achieve state 10 from 01", "target_state": "10", "next_input_for_world":"01"}, # Hint to agent about next world state for planning
        {"name": "Achieve state 11 (final)", "target_state": "11"}
    ]
    training_task_goal = GoalState(current_goal="Demo Sequence Task", steps=goal_steps, error_tolerance=0.1)
    emulator_demo3.set_goal_state(training_task_goal)
    
    print(f"Agent {emulator_demo3.agent_id} attempting goal: {emulator_demo3.current_goal_state_obj.current_goal}. Running up to 20 cycles.")
    emulator_demo3.run_chained_cognitive_cycles("00", 20) # Give enough cycles
    
    print(f"\nFinal Goal Status for {emulator_demo3.agent_id}:")
    if emulator_demo3.current_goal_state_obj:
        print(f"  {emulator_demo3.current_goal_state_obj}")
        print(f"  History: {emulator_demo3.current_goal_state_obj.history[-5:]}") # Last 5 history events
    emulator_demo3.print_internal_state_summary()


    print("\n\n--- DEMO 4: Co-Agent Manager (Feature 5) ---")
    # Base config for agents in the manager
    coagent_base_conf = {
        'cycle_history_max_len': 20, 'verbose': MASTER_VERBOSE_LEVEL -1 if MASTER_VERBOSE_LEVEL > 0 else 0,
        'smn_config': {'enabled': True}, # Allow agents to self-mutate
        'cognitive_firewall_config': {'enabled': True, 'check_interval': 4, 'cooldown_duration': 6},
        'lot_config': {'enabled': False } # Keep LoT off for agent sub-cycles in manager demo for brevity
    }
    # Variations for each agent
    coagent_variations = [
        {'config_overrides': {('internal_state_parameters', 'curiosity'): 0.8, ('E_OR_THRESHOLD',): 0.7, ('outcome_valence_map', '01'):0.8,('outcome_valence_map', '10'):-0.2 }}, # Agent 0: Curious, low ORT
        {'config_overrides': {('internal_state_parameters', 'goal_seeking_bias'): 0.7, ('orp_decay_rate',): 0.005, ('outcome_valence_map', '01'):0.6, ('outcome_valence_map', '10'):-0.6 }}, # Agent 1: Goal-focused, low decay
        {'config_overrides': {('internal_state_parameters', 'strategy_weights', 'memory'): 0.6, ('E_OR_THRESHOLD',): 1.2, ('outcome_valence_map', '01'):0.2, ('outcome_valence_map', '10'):-0.8 }}, # Agent 2: Memory-based, high ORT
    ]
    manager = CoAgentManager(num_agents=3, 
                             base_emulator_config_template=coagent_base_conf,
                             agent_config_variations_list=coagent_variations,
                             verbose=MASTER_VERBOSE_LEVEL)
    print(f"CoAgentManager running with {manager.num_agents} agents for 10 system cycles.")
    manager.run_system_cycles(num_system_cycles=10, initial_input_per_agent_list=["00", "01", "10"])
    # manager.print_system_summary() # This is called at the end of run_system_cycles


    print("\n\n--- DEMO 5: Cognitive Agent Trainer (with more complex agent) ---")
    # Trainer will train parameters that are part of DEFAULT_TRAINABLE_PARAMS_CONFIG
    # Other features like SMN, Firewall, Goals operate with their own default/set configs during training runs.
    trainer_base_emulator_config = {
        'cycle_history_max_len': 15, # Shorter history for faster training episodes
        'initial_E_OR_THRESHOLD': 1.0, # This might be overridden if E_OR_THRESHOLD is trainable via SMN etc.
        'initial_orp_decay_rate': 0.02,
        'smn_config': {'enabled': False}, # Turn SMN off during supervised training of base params for stability
        'cognitive_firewall_config': {'enabled': True, 'check_interval':10}, # Keep firewall on lightly
        'outcome_valence_map': {"00": -0.1, "01": 1.0, "10": -0.5, "11": 0.2}, # Task: achieve '01'
        'initial_internal_states': {'preferred_logical_state': "01"},
        'verbose_emulator_episodes': MASTER_VERBOSE_LEVEL -2 if MASTER_VERBOSE_LEVEL > 1 else 0 # Very quiet episodes
    }
    
    # Optional: Define a simple goal for the trainer's episodes
    # trainer_goal_template = {"current_goal": "Reach 01", "steps": [{"name": "Get to 01", "target_state": "01"}], "error_tolerance": 0.1}


    agent_trainer = CognitiveAgentTrainer(
        trainable_params_config=DEFAULT_TRAINABLE_PARAMS_CONFIG,
        base_emulator_config=trainer_base_emulator_config,
        verbose=MASTER_VERBOSE_LEVEL 
    )
    num_train_eps = 15  # Reduced episodes for faster demo
    cycles_per_ep = 10   # Reduced cycles

    print(f"Trainer: Starting training...")
    best_trained_params, best_reward, training_history = agent_trainer.train(
        num_training_episodes=num_train_eps,
        cycles_per_episode=cycles_per_ep,
        initial_input="00",
        # training_goal_state_template=trainer_goal_template # Uncomment to train with a goal
    )
    print("\n--- Training Summary ---")
    print(f"Best reward achieved after training: {best_reward:.4f}")
    agent_trainer.print_best_params("Final Best ")

    # Test with best parameters
    print("\n--- Running test with best trained parameters ---")
    final_test_config = agent_trainer._get_emulator_init_args(best_trained_params) # Get kwargs
    final_test_config['verbose'] = MASTER_VERBOSE_LEVEL # Make test run verbose
    if 'outcome_valence_map' in trainer_base_emulator_config: # Reuse training task valence map
         final_test_config['config_overrides'] = {('outcome_valence_map',): trainer_base_emulator_config['outcome_valence_map']} # Hack to set it
         # Better: trainer_base_emulator_config['config_overrides'] = {('outcome_valence_map',): trainer_base_emulator_config['outcome_valence_map']}
         # and then merge this into init_args['config_overrides'] in _get_emulator_init_args
    
    trained_emulator = SimplifiedOrchOREmulator(agent_id="trained_agent", **final_test_config)
    # Re-apply critical elements from trainer_base_config not handled by trainable_params
    if 'outcome_valence_map' in trainer_base_emulator_config:
         trained_emulator.outcome_valence_map = copy.deepcopy(trainer_base_emulator_config['outcome_valence_map'])
    if 'preferred_logical_state' in trainer_base_emulator_config.get('initial_internal_states', {}):
         trained_emulator.internal_state_parameters['preferred_logical_state'] = trainer_base_emulator_config['initial_internal_states']['preferred_logical_state']

    trained_emulator.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=15)
    #trained_emulator.print_internal_state_summary("  Final state of trained emulator ") # printed by run_chained...


    print("\n\n--- ALL NEW FEATURE DEMOS COMPLETED ---")


# one could say i put quite the grind into this 
