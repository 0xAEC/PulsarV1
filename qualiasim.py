import numpy as np
import copy
import time
import random
import collections # For deque
import traceback # For printing stack traces on errors
import math # For log2

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
    'ltm_mutation_on_store_rate': 0.01, # Chance to mutate LTM sequence metr`ics on store
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
    'review_interval': 7, 'cycles_since_last_review': 0, # MODIFIED: Shorter review interval for demos
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

# Default parameters for Temporal Feedback Grid (Feature 2 - NEWLY ADDED FEATURE) ---- team 2: Todo --- u must adapt the TFG on the class b4 LIS
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
# ---------------------------------------------------------------------------
# GoalState Structure (for Feature 7)
# ---------------------------------------------------------------------------
class GoalState:
    def __init__(self, current_goal, steps, error_tolerance=0.05, initial_progress=0.0):
        self.current_goal = current_goal
        self.steps = steps # List of step dictionaries
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
            # If a step itself is a GoalState, recursively serialize
            if isinstance(s_copy.get("sub_goal"), GoalState):
                 s_copy["sub_goal"] = s_copy["sub_goal"].to_dict()
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
            sub_goal_obj = self.steps[self.current_step_index].get("sub_goal")
            if isinstance(sub_goal_obj, GoalState) and sub_goal_obj.status == "active":
                step_name += f" -> (SubGoal: {sub_goal_obj.current_goal} - {sub_goal_obj.steps[sub_goal_obj.current_step_index].get('name')})"
        return f"Goal: '{self.current_goal}' (Step: '{step_name}', Progress: {self.progress*100:.1f}%, Status: {self.status})"


# ---------------------------------------------------------------------------
# Working Memory Components (NEW FEATURE) | added on 6/3/2025 6:39 PM for coordination
# ---------------------------------------------------------------------------
class WorkingMemoryItem:
    def __init__(self, type: str, data: dict, description: str = ""):
        self.type = type  # e.g., "goal_step_context", "intermediate_result", "backtrack_point"
        self.data = data  # specific data for the item type
        self.description = description
        self.timestamp = time.time()

    def __str__(self):
        data_preview = str(list(self.data.keys()))
        if len(data_preview) > 50: data_preview = data_preview[:47] + "..."
        return f"WMItem(type='{self.type}', desc='{self.description[:30]}...', data_keys={data_preview})"

class WorkingMemoryStack:
    def __init__(self, max_depth=20):
        self.stack = collections.deque(maxlen=max_depth)
        # self.internal_log = [] # Basic list for standalone debugging, emulator will use its own LoT.

    def push(self, item: WorkingMemoryItem) -> bool:
        """Pushes item, returns True if successful, False if max_depth would be exceeded (though deque handles this)."""
        # Deque handles maxlen by discarding from the other end, so push always "succeeds"
        # but we can signal if an item was implicitly discarded.
        item_discarded_to_make_space = False
        if len(self.stack) == self.stack.maxlen and self.stack.maxlen > 0:
            # self.internal_log.append(f"WM full, oldest item '{self.stack[0].type}' discarded due to push.")
            item_discarded_to_make_space = True # The deque will discard stack[0]
        self.stack.append(item)
        # self.internal_log.append(f"Pushed: {item}")
        return item_discarded_to_make_space # Returns True if an old item was discarded to make space

    def pop(self) -> WorkingMemoryItem | None:
        if not self.is_empty():
            item = self.stack.pop()
            # self.internal_log.append(f"Popped: {item}")
            return item
        return None

    def peek(self) -> WorkingMemoryItem | None:
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def clear(self):
        self.stack.clear()
        # self.internal_log.append("Cleared WM")

    def __len__(self):
        return len(self.stack)

    def to_dict_summary(self):
        top_item_summary = None
        if not self.is_empty():
            peeked_item = self.peek()
            top_item_summary = {
                "type": peeked_item.type,
                "description": peeked_item.description[:50] + "..." if len(peeked_item.description) > 50 else peeked_item.description
            }
        return {
            "current_depth": len(self.stack),
            "max_depth": self.stack.maxlen,
            "top_item_summary": top_item_summary,
        }


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
                 smn_general_config=None,
                 smn_controlled_params_config=None,
                 interrupt_handler_config=None,
                 cognitive_firewall_config=None,
                 goal_state_params = None,
                 lot_config=None,
                 shared_long_term_memory=None,
                 shared_attention_foci=None,
                 working_memory_max_depth=20, # NEW PARAM
                 config_overrides=None,
                 verbose=0):

        self.agent_id = agent_id
        self.verbose = verbose

        self.logical_superposition = {"00": 1.0 + 0j, "01":0j, "10":0j, "11":0j}
        self.collapsed_logical_state_str = "00"
        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate

        self.operation_costs = {'X': 0.1, 'Z': 0.1, 'H': 0.3, 'CNOT': 0.4, 'CZ': 0.4, 'ERROR_PENALTY': 0.05, 'PLANNING_BASE': 0.02}
        self.outcome_valence_map = {"00": 0.0, "01": 0.5, "10": -0.5, "11": 1.0}
        self.last_cycle_valence_raw = 0.0
        self.last_cycle_valence_mod = 0.0
        self.current_orp_before_reset = 0.0

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

        self.temporal_grid_params = copy.deepcopy(DEFAULT_TEMPORAL_GRID_PARAMS)
        if temporal_grid_config: self.temporal_grid_params.update(temporal_grid_config)
        self.temporal_feedback_grid = collections.deque(maxlen=self.temporal_grid_params['max_len'])
        self.last_cycle_entropy_for_delta = 0.0

        self.smn_config = copy.deepcopy(DEFAULT_SMN_CONFIG)
        if smn_general_config: self.smn_config.update(smn_general_config)
        self.smn_controlled_params_definitions = copy.deepcopy(DEFAULT_SMN_CONTROLLED_PARAMS)
        if smn_controlled_params_config: self.smn_controlled_params_definitions.update(smn_controlled_params_config)
        self.smn_params_runtime_state = {}
        self.smn_param_indices = {}
        self.smn_param_names_from_indices = {}
        self.smn_influence_matrix = np.array([])
        self.smn_param_actual_changes_this_cycle = {}
        self._initialize_smn_graph_structures()
        self.smn_internal_flags = {}

        self.interrupt_handler_params = copy.deepcopy(DEFAULT_INTERRUPT_HANDLER_CONFIG)
        if interrupt_handler_config: self.interrupt_handler_params.update(interrupt_handler_config)

        self.long_term_memory = shared_long_term_memory if shared_long_term_memory is not None else {}
        self.shared_attention_foci = shared_attention_foci if shared_attention_foci is not None else collections.deque(maxlen=20)

        self.firewall_params = copy.deepcopy(DEFAULT_COGNITIVE_FIREWALL_CONFIG)
        if cognitive_firewall_config: self.firewall_params.update(cognitive_firewall_config)
        self.firewall_cooldown_remaining = 0
        self.firewall_cycles_since_last_check = 0

        self.goal_state_config_params = copy.deepcopy(DEFAULT_GOAL_STATE_PARAMS)
        if goal_state_params: self.goal_state_config_params.update(goal_state_params)
        self.current_goal_state_obj = None

        # NEW: Working Memory Initialization
        self.working_memory = WorkingMemoryStack(max_depth=working_memory_max_depth)

        self.lot_config_params = copy.deepcopy(DEFAULT_LOT_CONFIG)
        if lot_config: self.lot_config_params.update(lot_config)
        self.current_cycle_lot_stream = []

        # MODIFIED for Demo 5, Fix 3: Initialize post-goal valence lock state
        self.post_goal_valence_lock_cycles_remaining = 0
        self.post_goal_valence_lock_value = 0.2
        self.post_goal_valence_lock_duration = 3


        if config_overrides:
            self._apply_config_overrides(config_overrides)

        if trainable_param_values:
            self.update_emulator_parameters(trainable_param_values)

        self.long_term_memory_capacity = 100
        self.successful_sequence_threshold_valence = 0.5 # Default, can be overridden by config_overrides

        self.cycle_history = collections.deque(maxlen=cycle_history_max_len)
        self.current_cycle_num = 0
        self.next_target_input_state = "00"

        if self.verbose >= 1:
            active_features_list = ["TemporalGrid", f"SMN(Graph:{self.smn_config.get('enable_influence_matrix', False)})", "Interrupts", "Firewall", "Goals", "LoT", "WorkingMemory"]
            print(f"[{self.agent_id}] Orch-OR Emulator Initialized. Active Features: {', '.join(active_features_list)}.")
            print(f"[{self.agent_id}] E_OR_THRESHOLD: {self.E_OR_THRESHOLD:.2f}, ORP Decay Rate: {self.orp_decay_rate:.3f}, WM Depth: {self.working_memory.stack.maxlen}")
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
        
        event_type_parts = event_type.split('.')
        event_category_for_ops_check = event_type_parts[0] + "_ops" # e.g. workingmemory.push -> workingmemory_ops
        event_category_for_general_check = event_type_parts[0] # e.g. workingmemory.push -> workingmemory
        event_category_for_wildcard_check = event_category_for_general_check + ".*" # e.g. workingmemory.*

        should_log_this_event = False
        if log_details_config.get(event_category_for_ops_check, False): # Check "category_ops" toggle
            should_log_this_event = True
        elif log_details_config.get(event_type, False): # Check specific event toggle (e.g., "workingmemory.push_goal_context")
            should_log_this_event = True
        elif log_details_config.get(event_category_for_wildcard_check, False): # Check "category.*" toggle
            should_log_this_event = True
        elif log_details_config.get(event_category_for_general_check, False): # Check "category" toggle (e.g., "workingmemory")
            should_log_this_event = True
        
        if not should_log_this_event:
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

        tag_name = event_type.upper().replace(".", "_") 
        self.current_cycle_lot_stream.append(f"#{tag_name}[{','.join(param_strs)}]")

    # --- Working Memory Logging Helper (NEW) --- perhaps for 
    def _log_wm_op(self, op_type: str, item: WorkingMemoryItem = None, details: dict = None):
        """Helper to log working memory operations via LoT."""
        log_data = details if details is not None else {}
        if item:
            log_data['item_type'] = item.type
            log_data['item_desc'] = item.description[:30] # Shorter for log
            log_data['item_data_keys_count'] = len(item.data.keys())
        
        log_data['wm_depth_after_op'] = len(self.working_memory)
        log_data['wm_max_depth'] = self.working_memory.stack.maxlen

        # For "push", indicate if an item was discarded from the bottom
        if op_type == "push" and 'item_discarded_on_push' in log_data and log_data['item_discarded_on_push']:
            self._log_lot_event("workingmemory.full_discard", {'reason': f'implicit_discard_for_push_of_{item.type if item else "item"}'})
        
        self._log_lot_event(f"workingmemory.{op_type}", log_data)


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
        mod_valence = raw_valence # Initialize mod_valence with raw_valence

        # --- MODIFIED for Demo 5, Fix 3: Stabilize Post-Goal Valence (Check Lock) ---
        if self.post_goal_valence_lock_cycles_remaining > 0:
            original_raw_valence_before_lock = raw_valence
            original_mod_valence_before_lock = mod_valence
            
            raw_valence = self.post_goal_valence_lock_value # Optionally lock raw_valence too
            mod_valence = self.post_goal_valence_lock_value
            self.post_goal_valence_lock_cycles_remaining -= 1
            
            lock_msg = f"Post-goal valence lock ACTIVE. Valences (raw/mod) set to {mod_valence:.2f}. Cycles remaining: {self.post_goal_valence_lock_cycles_remaining}."
            acc_thoughts_log.append(lock_msg)
            if self.verbose >= 1: print(f"    EXECUTIVE.Outcome_Eval: {lock_msg}")
            self._log_lot_event("executive.outcome_eval.post_goal_lock_active", {
                "locked_valence": mod_valence, 
                "cycles_left": self.post_goal_valence_lock_cycles_remaining,
                "original_raw_val_b4_lock": original_raw_valence_before_lock,
                "original_mod_val_b4_lock": original_mod_valence_before_lock
            })
        else: # Normal valence calculation if lock is not active
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
                preference_bonus = 0.30 * (1.0 - abs(mod_valence)) # Mod valence here should be before this bonus
                mod_valence += preference_bonus
                acc_thoughts_log.append(f"Preferred state |{current_preferred_state}> met, val boosted by {preference_bonus:.2f}.")
        # --- End of valence calculation block (lock or normal) ---

        # Update goal progress based on the outcome BEFORE mood is finalized,
        # (mod_valence might be further adjusted by goal bonuses within _executive_update_goal_progress)
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
             # _executive_update_goal_progress can also trigger the post-goal valence lock
             self._executive_update_goal_progress(logical_outcome_str, None) 

        # If post-goal lock was JUST activated by _executive_update_goal_progress, it might override current mod_valence.
        # This check re-applies the lock value if it became active during _executive_update_goal_progress and wasn't active at the start of this function.
        if self.post_goal_valence_lock_cycles_remaining == self.post_goal_valence_lock_duration: # Check if it was just set
             if mod_valence != self.post_goal_valence_lock_value :
                acc_thoughts_log.append(f"Re-applying post-goal valence lock ({self.post_goal_valence_lock_value:.2f}) immediately after goal completion logic altered valence.")
                mod_valence = self.post_goal_valence_lock_value
        
        mod_valence = np.clip(mod_valence, -1.0, 1.0) 
        self.last_cycle_valence_raw = raw_valence # This raw_valence could be locked or natural
        self.last_cycle_valence_mod = mod_valence
        acc_thoughts_log.append(f"Final val (raw/mod): {self.last_cycle_valence_raw:.2f}/{self.last_cycle_valence_mod:.2f}.") # Updated log string to use the member variables
        self._log_lot_event("executive.outcome_eval.valence", {"raw":self.last_cycle_valence_raw, "mod":self.last_cycle_valence_mod, "outcome_state":logical_outcome_str, "orp_collapse": orp_at_collapse, "post_goal_lock_active_this_eval": self.post_goal_valence_lock_cycles_remaining > 0})

        current_mood = self.internal_state_parameters['mood']
        mood_inertia = 0.88
        valence_influence_on_mood = 0.28
        new_mood = current_mood * mood_inertia + self.last_cycle_valence_mod * valence_influence_on_mood
        self.internal_state_parameters['mood'] = np.clip(new_mood, -1.0, 1.0)
        acc_thoughts_log.append(f"Mood updated from {current_mood:.2f} to {self.internal_state_parameters['mood']:.2f}.")
        self._log_lot_event("executive.outcome_eval.mood", {"new_mood":self.internal_state_parameters['mood'], "old_mood": current_mood})

        current_frustration = self.internal_state_parameters['frustration']
        frustration_threshold = self.metacognition_params['frustration_threshold']
        if self.last_cycle_valence_mod < self.metacognition_params['low_valence_threshold'] * 0.7:
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
            'raw_valence':self.last_cycle_valence_raw, 'mod_valence':self.last_cycle_valence_mod,
            'mood':self.internal_state_parameters['mood'],
            'frustration':self.internal_state_parameters['frustration'],
            'exploration_countdown':self.internal_state_parameters['exploration_mode_countdown'],
            'thoughts_log': acc_thoughts_log
        }

# --- NEW CODE BLOCK (_executive_generate_computation_sequence - FULL FUNCTION) ---
    def _executive_generate_computation_sequence(self, ops_provided_externally=None):
        if ops_provided_externally is not None:
            if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.OpGen: Using externally provided ops: {ops_provided_externally}")
            self._log_lot_event("executive.opgen.external", {"ops_count": len(ops_provided_externally)})
            return ops_provided_externally, "StrategyProvidedExternal", ["Ops provided externally."]

        exec_thought_log = ["OpGen: Generating new computation sequence:"]
        self._log_lot_event("executive.opgen.start", {"orp_current":self.objective_reduction_potential, "threshold": self.E_OR_THRESHOLD})

        ops_sequence = []
        chosen_strategy_name = "NoOpsMethod" # Default, will be updated

        effective_attention = self.internal_state_parameters['attention_level']
        cognitive_load_factor = 1.0 - (self.internal_state_parameters['cognitive_load'] * 0.65)
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * cognitive_load_factor * effective_attention, scale=1.0)))
        num_ops_target = min(num_ops_target, 10)

        exec_thought_log.append(f"  Target ops: ~{num_ops_target} (base:{num_ops_target_base}, load_factor:{cognitive_load_factor:.2f}, attn:{effective_attention:.2f}). ORP start: {self.objective_reduction_potential:.3f}")

        current_strategy_weights = self.internal_state_parameters['strategy_weights'].copy()
        ops_from_goal_hint = None # To store ops if a hint is directly used

        # --- WORKING MEMORY & GOAL STATE INFLUENCE ---
        active_goal_step_info = None
        active_goal_step_name = "None"
        is_goal_context_from_wm = False

        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            goal_obj = self.current_goal_state_obj
            if 0 <= goal_obj.current_step_index < len(goal_obj.steps):
                active_goal_step_info = goal_obj.steps[goal_obj.current_step_index]
                active_goal_step_name = active_goal_step_info.get('name', f'Step{goal_obj.current_step_index}')

                if not self.working_memory.is_empty():
                    wm_top_item = self.working_memory.peek()
                    if wm_top_item.type == "goal_step_context" and \
                       wm_top_item.data.get("goal_name") == goal_obj.current_goal and \
                       wm_top_item.data.get("step_index") == goal_obj.current_step_index and \
                       wm_top_item.data.get("goal_step_name") == active_goal_step_name:
                        is_goal_context_from_wm = True
                        exec_thought_log.append(f"  WM Active Context: Matched Goal '{goal_obj.current_goal}' - Step '{active_goal_step_name}'.")
                        self._log_lot_event("executive.opgen.wm_match_goal", {"goal":goal_obj.current_goal, "step":active_goal_step_name})
        
        if active_goal_step_info: # A goal step is active
            self._log_lot_event("executive.opgen.goal_influence.check", {"step_name": active_goal_step_name, "is_wm_ctx":is_goal_context_from_wm})
            exec_thought_log.append(f"  Goal Active ('{active_goal_step_name}', WM_Ctx: {is_goal_context_from_wm}): Applying influence.")

            step_target_state = active_goal_step_info.get("target_state")
            if step_target_state:
                if self.internal_state_parameters['preferred_logical_state'] != step_target_state:
                    self.internal_state_parameters['preferred_logical_state'] = step_target_state
                    exec_thought_log.append(f"    Goal ('{active_goal_step_name}') mandates preferred_state to |{step_target_state}>.")
                    self._log_lot_event("executive.opgen.goal_influence.pref_state_set", {"new_pref_state": step_target_state, "source_step": active_goal_step_name})
            
            goal_seek_boost = 0.35 if is_goal_context_from_wm else 0.25 # Stronger boost if WM context active
            current_strategy_weights['goal_seek'] = min(1.0, current_strategy_weights.get('goal_seek',0.1) * (1 + goal_seek_boost) + goal_seek_boost)
            current_strategy_weights['problem_solve'] = min(1.0, current_strategy_weights.get('problem_solve',0.1) * (1.2 + (0.2 * is_goal_context_from_wm)) + (0.05 + 0.05*is_goal_context_from_wm) )
            exec_thought_log.append(f"    Goal ('{active_goal_step_name}') boosts goal_seek (~{goal_seek_boost*100:.0f}%) & problem_solve.")
            self._log_lot_event("executive.opgen.goal_influence.strategy_boost", {"boost": goal_seek_boost, "source_step": active_goal_step_name})

            ops_hint_from_step = active_goal_step_info.get("next_ops_hint")
            if ops_hint_from_step and isinstance(ops_hint_from_step, list) and ops_hint_from_step:
                exec_thought_log.append(f"    Goal ('{active_goal_step_name}') provides ops_hint: {ops_hint_from_step}")
                self._log_lot_event("executive.opgen.goal_ops_hint.available", {"hint_str": str(ops_hint_from_step), "step_name": active_goal_step_name})
                
                use_hint_roll = random.random()
                use_hint_threshold = 0.65 if is_goal_context_from_wm else 0.45 # Higher chance to use hint if WM reinforces it
                
                if use_hint_roll < use_hint_threshold:
                    projected_hint_cost = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in ops_hint_from_step)
                    max_allowable_hint_orp = self.E_OR_THRESHOLD * 0.95 # Allow hint if it's not too close to threshold
                    
                    if self.objective_reduction_potential + projected_hint_cost < max_allowable_hint_orp:
                        ops_from_goal_hint = [list(op) for op in ops_hint_from_step] # Use a mutable copy
                        exec_thought_log.append(f"    Attempting to use ops_hint directly (cost {projected_hint_cost:.2f} < ORP budget {max_allowable_hint_orp:.2f}).")
                        self._log_lot_event("executive.opgen.goal_ops_hint.attempt_use", {"hint_ops_str":str(ops_hint_from_step), "cost":projected_hint_cost, "max_orp":max_allowable_hint_orp})
                    else:
                        exec_thought_log.append(f"    Ops_hint from goal was too costly (cost {projected_hint_cost:.2f}). ORP budget {max_allowable_hint_orp:.2f}. Standard generation will proceed.")
                        self._log_lot_event("executive.opgen.goal_ops_hint.too_costly", {"hint_ops_str":str(ops_hint_from_step), "cost":projected_hint_cost})
                else:
                    exec_thought_log.append(f"    Goal ops_hint available, but random roll ({use_hint_roll:.2f} >= {use_hint_threshold:.2f}) means not using it this time.")
                    self._log_lot_event("executive.opgen.goal_ops_hint.roll_failed", {"roll":use_hint_roll, "threshold": use_hint_threshold})
        
        if ops_from_goal_hint:
            ops_sequence = ops_from_goal_hint
            chosen_strategy_name = f"StrategyGoalStepHint({active_goal_step_name})"
            exec_thought_log.append(f"  OpGen Result: Using ops sequence from goal hint: {ops_sequence}")
            self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy":chosen_strategy_name, "final_sim_orp":"N/A_HintUsed"})
            return ops_sequence, chosen_strategy_name, exec_thought_log
        # --- END WORKING MEMORY & GOAL STATE INFLUENCE ---


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
                current_strategy_weights['goal_seek'] = max(0.01, current_strategy_weights['goal_seek'] * (1 - delta_v_bias_amount)) # Dampen goal seek if recent trend bad
                current_strategy_weights['curiosity'] = min(0.99, current_strategy_weights.get('curiosity',0.1) + delta_v_bias_amount * 0.6 + 0.03)
                current_strategy_weights['memory'] = min(0.99, current_strategy_weights.get('memory',0.1) + delta_v_bias_amount * 0.4 + 0.03)
                self._log_lot_event("executive.opgen.temporal_bias.neg_val_delta", {
                    "val_delta": avg_recent_valence_delta, "bias_str": valence_bias_strength, "bias_eff": delta_v_bias_amount
                })

            if avg_recent_entropy_shift > self.temporal_grid_params['high_entropy_shift_threshold'] and avg_recent_valence_delta < 0.05 and len(recent_entropy_shifts) > 0 :
                exec_thought_log.append(f"    TFG Bias: High avg entropy shift ({avg_recent_entropy_shift:.2f} > {self.temporal_grid_params['high_entropy_shift_threshold']}) with low/neutral valence. Increasing memory focus, reducing curiosity.")
                delta_e_bias_amount = avg_recent_entropy_shift * entropy_bias_strength
                current_strategy_weights['curiosity'] = max(0.01, current_strategy_weights.get('curiosity',0.1) * (1 - delta_e_bias_amount))
                current_strategy_weights['memory'] = min(0.99, current_strategy_weights.get('memory',0.1) + delta_e_bias_amount * 0.7 + 0.03)
                self._log_lot_event("executive.opgen.temporal_bias.high_ent_shift", {
                    "ent_shift": avg_recent_entropy_shift, "bias_str": entropy_bias_strength, "bias_eff": delta_e_bias_amount
                })
        else:
            exec_thought_log.append("  TemporalGridInfo: Grid empty or too few entries for feedback.")

        # General Goal State influence (if active_goal_step_info was NOT processed above, or as a minor fallback)
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active" and not active_goal_step_info:
             # This condition means a goal is active, but its specific step info wasn't primary driver (e.g. hint wasn't taken)
            goal_obj = self.current_goal_state_obj
            if 0 <= goal_obj.current_step_index < len(goal_obj.steps):
                fallback_step_info = goal_obj.steps[goal_obj.current_step_index]
                fallback_step_name = fallback_step_info.get('name', f'Step{goal_obj.current_step_index}')
                exec_thought_log.append(f"  General Goal Influence ('{fallback_step_name}' still active): Applying moderate strategy boosts.")
                current_strategy_weights['goal_seek'] = min(1.0, current_strategy_weights.get('goal_seek',0.1) * 1.15 + 0.1) # Moderate general boost
                current_strategy_weights['problem_solve'] = min(1.0, current_strategy_weights.get('problem_solve',0.1) * 1.1 + 0.03)

                fallback_target_state = fallback_step_info.get("target_state")
                if fallback_target_state and self.internal_state_parameters['preferred_logical_state'] != fallback_target_state:
                    self.internal_state_parameters['preferred_logical_state'] = fallback_target_state
                    exec_thought_log.append(f"    Fallback Goal Logic sets preferred_state to |{fallback_target_state}> for step '{fallback_step_name}'.")
                    self._log_lot_event("executive.opgen.fallback_goal_influence.pref_state", {"new_pref_state":fallback_target_state, "source_step":fallback_step_name})


        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            exec_thought_log.append("  Exploration mode active: Boosting curiosity, reducing goal/problem focus.")
            current_strategy_weights['curiosity'] = min(1.0, current_strategy_weights.get('curiosity',0.1)*2.8)
            current_strategy_weights['problem_solve'] *= 0.5 # Dampen specific problem solving
            current_strategy_weights['goal_seek'] *= 0.3   # Significantly dampen direct goal seeking
            self._log_lot_event("executive.opgen.exploration_bias", {"new_cur_weight": current_strategy_weights['curiosity']})


        if self.smn_internal_flags.get('force_ltm_reactive_op_next_cycle', False):
            exec_thought_log.append("  SMN/Interrupt Flag: Forcing LTM Reactive operation strategy.")
            # Override all other weights to ensure LTM recall is chosen
            current_strategy_weights = {'memory': 1.0, 'problem_solve': 0.001, 'goal_seek': 0.001, 'curiosity': 0.001}
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = False # Consume flag
            self._log_lot_event("executive.opgen.interrupt_bias.force_ltm", {})

        # Normalize strategy weights
        for key in DEFAULT_INTERNAL_PARAMS['strategy_weights']: # Ensure all keys exist
            if key not in current_strategy_weights: current_strategy_weights[key] = 0.001 # Small default if missing

        valid_weights = {k: v for k, v in current_strategy_weights.items() if isinstance(v, (int, float))}
        total_weight = sum(w for w in valid_weights.values() if w > 0)

        if total_weight <= 1e-6: # Prevent division by zero, default to curiosity
            exec_thought_log.append("  Warning: Strategy weights sum to near zero. Defaulting to curiosity.")
            current_strategy_weights = {k: 0.0 for k in DEFAULT_INTERNAL_PARAMS['strategy_weights']}
            current_strategy_weights['curiosity'] = 1.0
            valid_weights = {'curiosity': 1.0}
            total_weight = 1.0

        strategy_choices = []
        strategy_probs = []
        for s_name, s_weight in valid_weights.items():
            strategy_choices.append(s_name)
            # Ensure probability is non-negative
            strategy_probs.append(max(0, s_weight) / total_weight if total_weight > 1e-6 else 1.0/len(valid_weights if valid_weights else [1]))

        try:
            selected_strategy = random.choices(strategy_choices, weights=strategy_probs, k=1)[0]
        except ValueError as e: # Handle cases where all weights might have become zero due to aggressive scaling
            selected_strategy = 'curiosity'
            exec_thought_log.append(f"  Error in strategy selection (probs sum to {sum(strategy_probs)}, choices {strategy_choices}, error: {e}). Defaulting to curiosity.")
            # Ensure valid probabilities for logging / subsequent use if error occurs
            if not strategy_choices: strategy_choices = ['curiosity']
            strategy_probs = [1.0/len(strategy_choices)] * len(strategy_choices)


        exec_thought_log.append(f"  Strategy weights (norm): { {s:f'{p:.3f}' for s,p in zip(strategy_choices, strategy_probs)} }")
        exec_thought_log.append(f"  Selected primary strategy: {selected_strategy}")
        self._log_lot_event("executive.opgen.strategy_selected", {"strategy":selected_strategy, "weights_str": str({s:f'{p:.2f}' for s,p in zip(strategy_choices, strategy_probs)}) })

        simulated_orp_accumulator = self.objective_reduction_potential

        if selected_strategy == 'memory':
            replay_ops, _ = self._associative_layer_recall_from_ltm_strategy(simulated_orp_accumulator, exec_thought_log)
            if replay_ops:
                ops_sequence = replay_ops
                chosen_strategy_name = "StrategyLTMReplay"

        if not ops_sequence and selected_strategy == 'problem_solve':
            pref_state = self.internal_state_parameters['preferred_logical_state']
            if pref_state:
                exec_thought_log.append(f"  ProblemSolving towards |{pref_state}> from |{self.collapsed_logical_state_str}>")
                current_l1,current_l0=int(self.collapsed_logical_state_str[0]),int(self.collapsed_logical_state_str[1])
                target_l1,target_l0=int(pref_state[0]),int(pref_state[1])
                planned_problem_ops = []
                temp_plan_orp = simulated_orp_accumulator + self.operation_costs.get('PLANNING_BASE', 0.02)

                # Heuristic: If states are very different (e.g. "00" to "11"), consider Hadamard
                if abs((current_l0+current_l1) - (target_l0+target_l1)) >=2 and random.random() < 0.4 :
                    op_cost_h = self.operation_costs.get('H', 0.3)
                    if temp_plan_orp + op_cost_h < self.E_OR_THRESHOLD:
                         h_target_q = 0 if current_l0 != target_l0 else 1 # Heuristic which qubit to H
                         planned_problem_ops.append(('H', h_target_q)); temp_plan_orp += op_cost_h
                         exec_thought_log.append(f"    ProblemSolving plan included H for |{pref_state}>.")

                # Flip bits to match target_state
                if current_l0 != target_l0:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD: planned_problem_ops.append(('X',0)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    PS: Cannot apply ('X',0) to reach target |{pref_state}> due to ORP limit.")
                if current_l1 != target_l1:
                    op_cost = self.operation_costs.get('X',0.1)
                    if temp_plan_orp + op_cost < self.E_OR_THRESHOLD: planned_problem_ops.append(('X',1)); temp_plan_orp += op_cost
                    else: exec_thought_log.append(f"    PS: Cannot apply ('X',1) to reach target |{pref_state}> due to ORP limit.")
                
                # Further refinements for ProblemSolve can be added, e.g. CNOTs if bits need to be correlated for goal

                if planned_problem_ops:
                    ops_sequence = planned_problem_ops
                    chosen_strategy_name = "StrategyProblemSolving"
                    exec_thought_log.append(f"    ProblemSolving plan: {ops_sequence}")
                elif pref_state == self.collapsed_logical_state_str:
                     exec_thought_log.append(f"    ProblemSolving: Already at preferred state |{pref_state}>.")
                else: # No ops planned but target not met
                     exec_thought_log.append(f"    ProblemSolving: No viable ops plan generated to |{pref_state}> (possibly ORP limited or simple flips not enough).")
            else: # No preferred_state
                exec_thought_log.append("  ProblemSolving selected, but no preferred_logical_state is set. Falling through to general loop.")
                selected_strategy = 'curiosity' # Fallback if PS fails due to no target

        # This loop runs if:
        # 1. LTM replay didn't yield ops (or wasn't chosen)
        # 2. ProblemSolving didn't yield ops (or wasn't chosen, or had no target)
        # OR if selected_strategy was 'goal_seek' (with preferred_state) or 'curiosity' from the start.
        if not ops_sequence:
            pref_s_for_loop = self.internal_state_parameters['preferred_logical_state']
            
            # Determine mode for the loop
            is_goal_seek_mode_loop = False
            if selected_strategy == 'goal_seek' and pref_s_for_loop:
                chosen_strategy_name = "StrategyGoalSeekingLoop"
                exec_thought_log.append(f"  Executing GoalSeeking op generation towards |{pref_s_for_loop}>")
                is_goal_seek_mode_loop = True
            else: # Includes 'curiosity', or 'problem_solve'/'goal_seek' that fell through due to no target/ops
                chosen_strategy_name = "StrategyCuriosityDrivenLoop"
                exec_thought_log.append(f"  Executing CuriosityDriven (or fallback) op generation.")
            
            # Get current "simulated" state if ops were applied in thought
            # For simplicity, use the last actual collapsed state as the basis for random op gen.
            c_l1, c_l0 = int(self.collapsed_logical_state_str[0]), int(self.collapsed_logical_state_str[1])

            for op_count in range(num_ops_target):
                op_c, op_a = 'X', 0 # Default op if choices fail
                
                if is_goal_seek_mode_loop:
                    # Simple directed ops if goal seeking
                    t_l1, t_l0 = int(pref_s_for_loop[0]), int(pref_s_for_loop[1])
                    if c_l0 != t_l0 : op_c, op_a = 'X', 0
                    elif c_l1 != t_l1 : op_c, op_a = 'X', 1
                    elif random.random() < 0.5: op_c,op_a = ('H',random.randint(0,1)) # Try Hadamard if at target
                    else: # If bits match, try more complex or random ops
                        op_c = random.choice(['H','Z'] + (['CNOT','CZ'] if random.random() < 0.35 else []))
                        op_a = random.randint(0,1) if op_c in ['H','X','Z'] else tuple(random.sample([0,1],2))
                else: # Curiosity-driven or other fallbacks
                    op_choices = ['X','Z','H']
                    if random.random() < 0.45: op_choices.extend(['CNOT', 'CZ']) # More 2-qubit ops
                    op_c = random.choice(op_choices)
                    op_a = random.randint(0,1) if op_c in ['X','Z','H'] else tuple(random.sample([0,1],2))

                op_cost = self.operation_costs.get(op_c.upper(), 0.05)

                attention_lapse_prob = (self.internal_state_parameters['cognitive_load'] * 0.2) + \
                                      (1.0 - effective_attention) * 0.15
                if random.random() < attention_lapse_prob:
                    original_op_tuple = (op_c, op_a)
                    if op_c in ['X','Z','H'] and isinstance(op_a,int): op_a = 1 - op_a # Flip target
                    elif op_c in ['CNOT','CZ'] and isinstance(op_a,tuple): op_a = (op_a[1],op_a[0]) # Swap ctrl/target
                    else: # Change op type
                        op_c = random.choice(['X','Z']) if op_c not in ['X','Z'] else 'H'

                    op_cost += self.operation_costs.get('ERROR_PENALTY',0.05) * 0.5
                    exec_thought_log.append(f"      ATTENTION LAPSE! Op {original_op_tuple} -> ({op_c},{op_a}), cost penalty. LapseProb={attention_lapse_prob:.2f}")
                    self._log_lot_event("executive.opgen.attention_lapse", {"original_op_str":str(original_op_tuple), "mutated_op_str":str((op_c,op_a)), "lapse_prob":attention_lapse_prob})

                if simulated_orp_accumulator + op_cost < self.E_OR_THRESHOLD * 0.98 : # Safety margin
                    ops_sequence.append((op_c,op_a))
                    simulated_orp_accumulator += op_cost
                    # Update simulated c_l0, c_l1 if 'X' op was applied for next iter of goal seeking loop
                    if is_goal_seek_mode_loop and op_c == 'X':
                        if op_a == 0: c_l0 = 1 - c_l0
                        else: c_l1 = 1 - c_l1
                else:
                    exec_thought_log.append(f"    OpGen loop ({op_count+1}/{num_ops_target}): Op ('{op_c}',{op_a}) cost {op_cost:.2f} would exceed ORP. Stopping. (SimORP {simulated_orp_accumulator:.2f} + {op_cost:.2f} vs E_OR {self.E_OR_THRESHOLD:.2f})")
                    break # Stop adding ops if ORP budget is tight

        # Final check on chosen_strategy_name if ops_sequence is still empty after all attempts
        if not ops_sequence:
            if chosen_strategy_name not in ["StrategyLTMReplay", "StrategyProblemSolving", "StrategyGoalStepHint"]: # If it wasn't one of these that failed
                chosen_strategy_name = "NoOpsGeneratedByAnyMethod"
            exec_thought_log.append("  Final Result: No operations generated by any strategy this cycle.")
        elif chosen_strategy_name == "NoOpsMethod": # If ops_sequence got filled but strategy name wasn't updated
             chosen_strategy_name = "StrategyUnknownSourceOrLoopPopulated" # A more indicative fallback name


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
        step_idx = goal.current_step_index # This is the step *before* potential advance

        if not (0 <= step_idx < len(goal.steps)):
            if self.verbose >= 1: print(f"[{self.agent_id}] Goal Error: Invalid step index {step_idx} for goal '{goal.current_goal}' (NumSteps: {len(goal.steps)})")
            self._log_lot_event("executive.goalprogress.error", {"goal_name": goal.current_goal, "error": "invalid_step_idx", "step_idx":step_idx, "num_steps_in_goal": len(goal.steps)})
            goal.status = "failed"; goal.history.append({"cycle": self.current_cycle_num, "event": "error_invalid_step_idx", "step_index_at_event": step_idx})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)
            
            if not self.working_memory.is_empty():
                top_item = self.working_memory.peek()
                current_step_name_for_error_pop = goal.steps[step_idx].get("name", f"Step {step_idx + 1}") if 0 <= step_idx < len(goal.steps) else f"InvalidStep{step_idx}"
                if top_item.type == "goal_step_context" and \
                   top_item.data.get("goal_name") == goal.current_goal and \
                   top_item.data.get("goal_step_name") == current_step_name_for_error_pop and \
                   top_item.data.get("step_index") == step_idx:
                    popped_item = self.working_memory.pop()
                    self._log_wm_op("pop_goal_context", item=popped_item, details={"reason": "goal_error_invalid_step_idx"})
            return

        current_step = goal.steps[step_idx]
        step_name = current_step.get("name", f"Step {step_idx + 1}")
        self._log_lot_event("executive.goalprogress.check", {"goal_name": goal.current_goal, "step_name": step_name, "step_idx":step_idx, "outcome_state":collapsed_outcome_str})

        # --- MODIFIED WM PUSH LOGIC for Demo 3, Fix 2 ---
        if current_step.get("requires_explicit_wm_context_push", False):
            needs_to_push_specific_context_for_this_step = True 
            if not self.working_memory.is_empty():
                top_item = self.working_memory.peek()
                if top_item.type == "goal_step_context" and \
                   top_item.data.get("goal_name") == goal.current_goal and \
                   top_item.data.get("goal_step_name") == step_name and \
                   top_item.data.get("step_index") == step_idx:
                    # If the exact context for THIS step is already on top, no need to re-push.
                    # This prevents WM bloat for a step that's active for multiple cycles.
                    needs_to_push_specific_context_for_this_step = False
                    if self.verbose >= 3: print(f"    WM Push Skipped for '{step_name}': Context already on top.")
                    self._log_lot_event("workingmemory.push_goal_context.skip_duplicate_top", {
                        "goal_name": goal.current_goal, "step_name": step_name, "step_idx": step_idx
                    })
            
            if needs_to_push_specific_context_for_this_step:
                wm_item_data = {
                    "goal_name": goal.current_goal, 
                    "goal_step_name": step_name, 
                    "step_index": step_idx,
                    "collapsed_state_at_eval_time": self.collapsed_logical_state_str,
                    # "cycle_num_pushed_for_eval": self.current_cycle_num # Optional: keep for detailed debug if needed
                }
                item_to_push = WorkingMemoryItem(type="goal_step_context", data=wm_item_data, 
                                                 description=f"CtxPush: {goal.current_goal} - {step_name}") # Changed desc slightly for clarity
                item_discarded = self.working_memory.push(item_to_push)
                self._log_wm_op("push_goal_context", item=item_to_push, 
                                details={'reason': 'ensure_active_step_context', 
                                         'item_discarded_on_push': item_discarded})
        # --- END MODIFIED WM PUSH LOGIC ---

        achieved_step = False
        sub_goal_obj = current_step.get("sub_goal")
        if isinstance(sub_goal_obj, GoalState):
            if sub_goal_obj.status == "pending":
                if self.verbose >= 1: print(f"[{self.agent_id}] Activating SubGoal '{sub_goal_obj.current_goal}' for step '{step_name}' of '{goal.current_goal}'")
                self._log_lot_event("executive.goalprogress.subgoal_activate", {"parent_goal": goal.current_goal, "sub_goal":sub_goal_obj.current_goal})
                sub_goal_obj.status = "active"
            
            if sub_goal_obj.status == "active":
                pass 
            
            if sub_goal_obj.status == "completed": 
                achieved_step = True
                if self.verbose >=1: print(f"[{self.agent_id}] SubGoal '{sub_goal_obj.current_goal}' previously completed for step '{step_name}'. Step for parent achieved.")
                self._log_lot_event("executive.goalprogress.subgoal_eval_complete", {"parent_goal": goal.current_goal, "sub_goal":sub_goal_obj.current_goal})
            elif sub_goal_obj.status == "failed":
                 if self.verbose >=1: print(f"[{self.agent_id}] SubGoal '{sub_goal_obj.current_goal}' previously failed for step '{step_name}'. Step fails for parent.")
                 goal.history.append({"cycle": self.current_cycle_num, "event": f"step_failed_due_to_subgoal", "step_name": step_name, "subgoal_name": sub_goal_obj.current_goal, "current_step_index_at_event": goal.current_step_index})

        if not achieved_step and current_step.get("target_state") and collapsed_outcome_str == current_step["target_state"]:
            achieved_step = True
            if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via target state |{collapsed_outcome_str}>.")
        elif not achieved_step and callable(current_step.get("completion_criteria")):
            try:
                context_for_callable = {
                    'collapsed_state': collapsed_outcome_str, 
                    'ops': executed_ops, 
                    'agent_public_state': self.get_public_state_summary(), 
                    'working_memory': self.working_memory, 
                    'current_goal_obj': goal, 
                    'current_step_obj': current_step, 
                }
                if current_step["completion_criteria"](context_for_callable):
                    achieved_step = True
                    if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via custom criteria.")
            except Exception as e:
                if self.verbose >=1: print(f"[{self.agent_id}] Error in goal step completion_criteria for '{step_name}': {e}")
                self._log_lot_event("executive.goalprogress.criteria_error", {"step_name":step_name, "error_str":str(e)})
        
        pop_reason_details = ""
        if achieved_step:
            goal.history.append({"cycle": self.current_cycle_num, "event": f"step_completed", "step_name": step_name, "outcome_state":collapsed_outcome_str, "current_step_index_at_event": goal.current_step_index})
            self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['step_completion_valence_bonus'], -1.0, 1.0)
            goal.current_step_index += 1 
            num_steps = len(goal.steps)
            goal.progress = goal.current_step_index / num_steps if num_steps > 0 else 1.0
            self._log_lot_event("executive.goalprogress.step_complete", {"step_name": step_name, "new_progress": goal.progress, "valence_mod_bonus":self.goal_state_config_params['step_completion_valence_bonus']})
            pop_reason_details = "step_achieved"

            if goal.current_step_index >= len(goal.steps):
                goal.status = "completed"; goal.progress = 1.0
                if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' COMPLETED!")
                self._log_lot_event("executive.goalprogress.goal_complete", {"goal_name": goal.current_goal, "valence_mod_bonus":self.goal_state_config_params['completion_valence_bonus']})
                self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['completion_valence_bonus'], -1.0, 1.0)
                
                if self.post_goal_valence_lock_cycles_remaining <= 0: 
                    self.post_goal_valence_lock_cycles_remaining = self.post_goal_valence_lock_duration
                    if self.verbose >=1: print(f"    EXECUTIVE.GoalProgress: Post-goal valence lock initiated for {self.post_goal_valence_lock_duration} cycles at value {self.post_goal_valence_lock_value:.2f}.")
                    self._log_lot_event("executive.goalprogress.post_goal_lock_set", {
                        "duration": self.post_goal_valence_lock_duration, 
                        "lock_value": self.post_goal_valence_lock_value
                    })

                if self.internal_state_parameters['preferred_logical_state'] == current_step.get("target_state"): 
                    self.internal_state_parameters['preferred_logical_state'] = None 
            else: 
                 next_step_index_after_advance = goal.current_step_index
                 next_step_name = goal.steps[next_step_index_after_advance].get("name", f"Step {next_step_index_after_advance+1}")
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Advanced to step '{next_step_name}'.")
        else: 
            goal.history.append({"cycle": self.current_cycle_num, "event": "step_no_progress", "step_name": step_name, "current_outcome": collapsed_outcome_str, "current_step_index_at_event": goal.current_step_index})
            max_cycles_on_step_val = current_step.get("max_cycles_on_step", float('inf'))
            
            cycles_on_this_step_count = 0
            for hist_entry in reversed(goal.history):
                if hist_entry.get("current_step_index_at_event") != goal.current_step_index :
                    break 
                if hist_entry.get("step_name") == step_name:
                    cycles_on_this_step_count +=1
            
            if cycles_on_this_step_count >= max_cycles_on_step_val :
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}' FAILED due to too many cycles ({cycles_on_this_step_count}) on step '{step_name}'. Max was {max_cycles_on_step_val}") # Added max_cycles_on_step_val to log
                 self._log_lot_event("executive.goalprogress.goal_fail", {"goal_name":goal.current_goal, "reason":f"max_cycles_on_step_{step_name}", "cycles_spent": cycles_on_this_step_count, "max_allowed":max_cycles_on_step_val})
                 goal.status = "failed"
                 self.last_cycle_valence_mod = np.clip(self.last_cycle_valence_mod + self.goal_state_config_params['failure_valence_penalty'], -1.0, 1.0)
                 pop_reason_details = "step_failed_max_cycles"

        if pop_reason_details: 
            if not self.working_memory.is_empty():
                top_item = self.working_memory.peek()
                if top_item.type == "goal_step_context" and \
                   top_item.data.get("goal_name") == goal.current_goal and \
                   top_item.data.get("goal_step_name") == step_name and \
                   top_item.data.get("step_index") == step_idx: 
                    popped_item = self.working_memory.pop()
                    self._log_wm_op("pop_goal_context", item=popped_item, details={"reason": pop_reason_details})
        
        if goal.status in ["completed", "failed"]:
            final_step_details = current_step 
            if self.internal_state_parameters['preferred_logical_state'] == final_step_details.get("target_state"):
                 self.internal_state_parameters['preferred_logical_state'] = None
                 self._log_lot_event("executive.goalprogress.clear_pref_state_on_goal_end", {"goal_status": goal.status, "related_target_state": final_step_details.get("target_state"), "step_name_involved": final_step_details.get("name")})



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
        curiosity_change -= cur_base_rate * 0.4 # Natural decay/usage of curiosity
        self.internal_state_parameters['curiosity'] = np.clip(self.internal_state_parameters['curiosity'] + curiosity_change, 0.01, 0.99)

        goal_bias_change = 0.0
        goal_base_rate = 0.045
        # If a goal is active OR a preferred state exists (which could be from a goal step)
        is_goal_oriented_context = (self.internal_state_parameters['preferred_logical_state'] is not None) or \
                                  (self.current_goal_state_obj and self.current_goal_state_obj.status == "active")
        
        if is_goal_oriented_context:
            if mod_valence > 0.35: goal_bias_change += goal_base_rate # Reinforce if successful
            else: goal_bias_change -=goal_base_rate*0.6 # Slightly decrease if not meeting goals in goal context
        else: # No active goal or preferred state
            goal_bias_change -= goal_base_rate*0.3 # Gradual decay if no goals
        self.internal_state_parameters['goal_seeking_bias'] = np.clip(self.internal_state_parameters['goal_seeking_bias'] + goal_bias_change, 0.01, 0.99)

        if self.verbose >=3: print(f"    Curiosity: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        self._log_lot_event("meta.cog_param_update.end", {"cog_load":self.internal_state_parameters['cognitive_load'], "attn": self.internal_state_parameters['attention_level'], "cur": self.internal_state_parameters['curiosity'], "goal_bias":self.internal_state_parameters['goal_seeking_bias']})


    def _meta_layer_adapt_preferred_state(self, collapsed_outcome_str, mod_valence):
        high_val_thresh = self.metacognition_params['high_valence_threshold']
        # MODIFIED for Demo 3, Fix 1: Specific threshold for goal alignment
        goal_alignment_valence_threshold = 0.8 
        low_val_thresh = self.metacognition_params['low_valence_threshold']
        current_pref_state = self.internal_state_parameters['preferred_logical_state']
        pref_state_log_msg = ""
        action_taken_this_adaptation = False

        # --- Demo 3, Fix 1 START: Align Preferred State with Goal on high valence ---
        active_goal_target_state_for_alignment = None
        goal_step_name_for_alignment_log = "N/A"
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            active_goal = self.current_goal_state_obj
            if 0 <= active_goal.current_step_index < len(active_goal.steps):
                current_active_step_for_align = active_goal.steps[active_goal.current_step_index]
                active_goal_target_state_for_alignment = current_active_step_for_align.get("target_state")
                goal_step_name_for_alignment_log = current_active_step_for_align.get("name", f"Step {active_goal.current_step_index}")

        if mod_valence >= goal_alignment_valence_threshold and active_goal_target_state_for_alignment:
            if current_pref_state != active_goal_target_state_for_alignment:
                self.internal_state_parameters['preferred_logical_state'] = active_goal_target_state_for_alignment
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.35) # Stronger alignment boost
                pref_state_log_msg += f"High valence ({mod_valence:.2f} >= {goal_alignment_valence_threshold}) AND active goal step '{goal_step_name_for_alignment_log}' target. Aligned preferred state to |{active_goal_target_state_for_alignment}>. Goal bias strongly boosted."
                action_taken_this_adaptation = True
                if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState (GoalAlign): {pref_state_log_msg}")
                self._log_lot_event("meta.adapt_pref_state.goal_align_high_valence", {
                    "message": pref_state_log_msg,
                    "new_pref_state_str": str(self.internal_state_parameters['preferred_logical_state']),
                    "mod_valence": mod_valence,
                    "goal_target_state": active_goal_target_state_for_alignment
                })
                return # Exit after this specific alignment logic for Demo 3, Fix 1
        # --- Demo 3, Fix 1 END ---


        # Original logic continues if the above high-valence goal alignment didn't trigger / return
        is_pref_state_goal_dictated = False
        goal_driven_pref_state_source = "None"
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            active_goal = self.current_goal_state_obj
            if 0 <= active_goal.current_step_index < len(active_goal.steps):
                current_active_step = active_goal.steps[active_goal.current_step_index]
                step_target_state = current_active_step.get("target_state")
                if step_target_state == current_pref_state: # Check if current_pref_state *is already* the goal's target
                    is_pref_state_goal_dictated = True
                    goal_driven_pref_state_source = f"GoalStep:'{current_active_step.get('name', '')}'"
                    if not self.working_memory.is_empty():
                        top_item = self.working_memory.peek()
                        if top_item.type == "goal_step_context":
                            wm_data = top_item.data
                            if wm_data.get("goal_name") == active_goal.current_goal and \
                               wm_data.get("step_index") == active_goal.current_step_index and \
                               wm_data.get("goal_step_name") == current_active_step.get("name"):
                                goal_driven_pref_state_source += "+WM_Match"

        if is_pref_state_goal_dictated:
            pref_state_log_msg += f"Preferred state |{current_pref_state}> currently aligned with active {goal_driven_pref_state_source}. Adaptation highly constrained. "
            if mod_valence <= low_val_thresh * 0.8 and current_pref_state == collapsed_outcome_str :
                if self.current_goal_state_obj.status != "active": 
                    self.internal_state_parameters['preferred_logical_state'] = None
                    self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.3) 
                    pref_state_log_msg += f"Goal no longer active AND low valence for |{collapsed_outcome_str}> ({mod_valence:.2f}), cleared preferred state."
                    action_taken_this_adaptation = True
                else: 
                    pref_state_log_msg += f"Low valence ({mod_valence:.2f}) for goal-driven preferred state |{collapsed_outcome_str}>, but goal is active. No change to pref state here. Frustration may increase."
            elif mod_valence >= high_val_thresh and current_pref_state == collapsed_outcome_str:
                pref_state_log_msg += f"High valence ({mod_valence:.2f}) for achieving goal-driven preferred state. Reinforced."
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.1) 
                action_taken_this_adaptation = True
        else: 
            if mod_valence >= high_val_thresh and current_pref_state != collapsed_outcome_str:
                self.internal_state_parameters['preferred_logical_state'] = collapsed_outcome_str
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.28)
                self.internal_state_parameters['frustration'] *= 0.55
                pref_state_log_msg += f"New (non-goal-driven) preferred state |{collapsed_outcome_str}> set due to high valence ({mod_valence:.2f}). Goal bias up, frustration down."
                action_taken_this_adaptation = True
            elif mod_valence <= low_val_thresh and current_pref_state == collapsed_outcome_str: 
                self.internal_state_parameters['preferred_logical_state'] = None
                self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.22)
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.18)
                pref_state_log_msg += f"Non-goal-driven preferred state |{collapsed_outcome_str}> cleared due to low valence ({mod_valence:.2f}). Goal bias down, curiosity up."
                action_taken_this_adaptation = True
            elif current_pref_state == collapsed_outcome_str and low_val_thresh < mod_valence < (high_val_thresh * 0.5) and random.random() < 0.15: 
                self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] * 0.9)
                pref_state_log_msg += f"Non-goal-driven preferred state |{collapsed_outcome_str}> yielding mediocre results ({mod_valence:.2f}), slightly reduced goal_seeking_bias."
                if self.internal_state_parameters['goal_seeking_bias'] < 0.1:
                    self.internal_state_parameters['preferred_logical_state'] = None
                    pref_state_log_msg += " Preferred state cleared due to very low bias."
                action_taken_this_adaptation = True

        if action_taken_this_adaptation or (self.verbose >=2 and is_pref_state_goal_dictated and "No change to pref state here" not in pref_state_log_msg):
            if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState: {pref_state_log_msg}")
            self._log_lot_event("meta.adapt_pref_state", {"message": pref_state_log_msg,
                                                           "new_pref_state_str": str(self.internal_state_parameters['preferred_logical_state']),
                                                           "mod_valence": mod_valence,
                                                           "is_goal_dictated": is_pref_state_goal_dictated,
                                                           "source_if_goal_dictated": goal_driven_pref_state_source if is_pref_state_goal_dictated else "N/A"})
        elif is_pref_state_goal_dictated and "No change to pref state here" in pref_state_log_msg and self.verbose >= 2 : 
             self._log_lot_event("meta.adapt_pref_state.skipped_active_goal", {"current_pref_state": str(current_pref_state), "mod_valence": mod_valence, "reason_msg": pref_state_log_msg})


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


    # ---  Feature 3: Synaptic Mutation Network (SMN) Methods (Enhanced Graph Version) ---
    def _initialize_smn_graph_structures(self):
        """Initializes SMN graph-related structures: param indices, influence matrix."""
        self.smn_param_indices = {name: i for i, name in enumerate(self.smn_controlled_params_definitions.keys())}
        self.smn_param_names_from_indices = {i: name for name, i in self.smn_param_indices.items()}

        num_smn_params = len(self.smn_controlled_params_definitions)
        self.smn_config['smn_influence_matrix_size'] = num_smn_params # Store for reference

        self.smn_params_runtime_state = {}
        for smn_key, definition in self.smn_controlled_params_definitions.items():
            self.smn_params_runtime_state[smn_key] = {
                'current_mutation_strength': definition['base_mutation_strength'], 
                'base_mutation_strength_ref': definition['base_mutation_strength'], 
                'min_val': definition['min_val'],
                'max_val': definition['max_val'],
                'is_int': definition.get('is_int', False),
                'path': definition['path']
            }

        if num_smn_params > 0 and self.smn_config.get('enable_influence_matrix', False):
            initial_stddev = self.smn_config.get('smn_influence_matrix_initial_stddev', 0.05)
            self.smn_influence_matrix = np.random.normal(loc=0.0, scale=initial_stddev, size=(num_smn_params, num_smn_params))
            np.fill_diagonal(self.smn_influence_matrix, 0) 
        else:
            self.smn_influence_matrix = np.array([]) 

        self.smn_param_actual_changes_this_cycle = {}
        if self.verbose >=2 and self.smn_config.get('enable_influence_matrix', False) and num_smn_params > 0:
            print(f"    SMN Graph Structures Initialized: {num_smn_params} params. Influence Matrix shape: {self.smn_influence_matrix.shape}")

    def _smn_get_param_value(self, path_tuple):
        """Helper to get a parameter's value using its path tuple."""
        try:
            target_obj = self
            if len(path_tuple) == 1: 
                return getattr(target_obj, path_tuple[0])
            current_dict_or_obj = getattr(target_obj, path_tuple[0])
            for key_part_idx in range(1, len(path_tuple) -1): 
                current_dict_or_obj = current_dict_or_obj[path_tuple[key_part_idx]]
            return current_dict_or_obj[path_tuple[-1]] 
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            if self.verbose >= 1: print(f"    SMN_GET_PARAM_ERROR: Failed to get param at path {path_tuple}: {e}")
            self._log_lot_event("smn.error.get_param", {"path_str":str(path_tuple), "error":str(e)})
            param_key_smn = next((k for k, v in self.smn_controlled_params_definitions.items() if v['path'] == path_tuple), None)
            if param_key_smn: return self.smn_controlled_params_definitions[param_key_smn].get('min_val', 0) 
            return 0 

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
        if not self.smn_config.get('enabled', False) or not self.smn_param_indices: return 

        valence_gain = valence_mod_this_cycle - prev_cycle_valence_mod
        smn_pos_thresh = self.internal_state_parameters['smn_positive_valence_threshold']
        smn_neg_thresh = self.internal_state_parameters['smn_negative_valence_threshold']

        if self.verbose >= 2: print(f"  SMN Update & Mutate: ValenceMod={valence_mod_this_cycle:.2f}, PrevModVal={prev_cycle_valence_mod:.2f}, Gain={valence_gain:.2f}, ORP={orp_at_collapse:.3f}")
        self._log_lot_event("smn.update.start", {"val_mod_curr":valence_mod_this_cycle, "val_mod_prev": prev_cycle_valence_mod, "val_gain":valence_gain, "orp_col":orp_at_collapse})

        self.smn_param_actual_changes_this_cycle.clear() 
        any_strategy_weights_mutated = False
        primary_mutations_info = {} 

        for param_smn_key, runtime_state_info in self.smn_params_runtime_state.items():
            current_param_strength = runtime_state_info['current_mutation_strength']

            if valence_mod_this_cycle > smn_pos_thresh :
                current_param_strength *= self.internal_state_parameters['smn_mutation_strength_decay']
            elif valence_mod_this_cycle < smn_neg_thresh :
                current_param_strength *= self.internal_state_parameters['smn_mutation_strength_grow']
            runtime_state_info['current_mutation_strength'] = np.clip(current_param_strength, 0.0001, 0.8) 

            if valence_gain > self.smn_config.get('mutation_trigger_min_valence_gain', 0.1) and \
               valence_mod_this_cycle > (smn_pos_thresh * 0.2): 

                param_path = runtime_state_info['path']
                current_val = self._smn_get_param_value(param_path)
                perturbation = np.random.normal(0,
                                                runtime_state_info['current_mutation_strength'] * \
                                                self.internal_state_parameters['smn_perturbation_scale_factor'])
                new_val = current_val + perturbation
                if runtime_state_info['is_int']: new_val = int(round(new_val))
                new_val = np.clip(new_val, runtime_state_info['min_val'], runtime_state_info['max_val'])
                actual_change = new_val - current_val
                if abs(actual_change) > 1e-7: 
                    if self._smn_set_param_value(param_path, new_val):
                        self.smn_param_actual_changes_this_cycle[param_smn_key] = actual_change
                        primary_mutations_info[param_smn_key] = {'change': actual_change, 'original_perturb': perturbation}
                        if self.verbose >= 2:
                            print(f"    SMN Primary Mutation: {param_smn_key} ('{'.'.join(str(p) for p in param_path)}') {current_val:.4f} -> {new_val:.4f} (strength:{runtime_state_info['current_mutation_strength']:.4f})")
                        self._log_lot_event("smn.update.mutation_applied", {"param_smn_key":param_smn_key, "path_str":str(param_path), "old_val":current_val, "new_val":new_val, "change":actual_change, "type":"primary"})
                        if param_path[0] == 'internal_state_parameters' and param_path[1] == 'strategy_weights':
                            any_strategy_weights_mutated = True

        if self.smn_config.get('enable_influence_matrix', False) and primary_mutations_info and self.smn_influence_matrix.size > 0:
            propagated_perturb_accumulator = collections.defaultdict(float) 

            for source_param_smn_key, primary_mutation_data in primary_mutations_info.items():
                source_idx = self.smn_param_indices[source_param_smn_key]
                source_runtime_state = self.smn_params_runtime_state[source_param_smn_key]
                source_ref_scale = source_runtime_state.get('base_mutation_strength_ref', 0.1) + 1e-6
                normalized_primary_perturb = primary_mutation_data['original_perturb'] / source_ref_scale

                for target_param_smn_key, target_idx in self.smn_param_indices.items():
                    if source_idx == target_idx: continue 

                    influence_weight = self.smn_influence_matrix[source_idx, target_idx]
                    if abs(influence_weight) > self.smn_config['smn_influence_propagation_threshold']:
                        target_runtime_state = self.smn_params_runtime_state[target_param_smn_key]
                        target_ref_scale = target_runtime_state.get('base_mutation_strength_ref', 0.1)
                        propagated_perturb_on_target = influence_weight * \
                                                       normalized_primary_perturb * \
                                                       target_ref_scale * \
                                                       self.smn_config['smn_secondary_mutation_scale_factor']
                        propagated_perturb_accumulator[target_param_smn_key] += propagated_perturb_on_target
                        self._log_lot_event("smn_graph_propagation.attempt", {
                            "from":source_param_smn_key, "to":target_param_smn_key,
                            "weight":influence_weight, "prop_perturb":propagated_perturb_on_target
                        })
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
                        self.smn_param_actual_changes_this_cycle[target_param_smn_key] = \
                            self.smn_param_actual_changes_this_cycle.get(target_param_smn_key, 0.0) + actual_propagated_change
                        if self.verbose >= 2:
                            print(f"    SMN Propagated Mutation: {target_param_smn_key} {current_target_val:.4f} -> {new_target_val:.4f} (total_prop_perturb:{total_propagated_perturb:.4f})")
                        self._log_lot_event("smn.update.mutation_applied", {"param_smn_key":target_param_smn_key, "old_val":current_target_val, "new_val":new_target_val, "change":actual_propagated_change, "type":"propagated"})
                        if param_path[0] == 'internal_state_parameters' and param_path[1] == 'strategy_weights':
                             any_strategy_weights_mutated = True

        if self.smn_config.get('enable_influence_matrix', False) and self.smn_influence_matrix.size > 0:
            hebbian_lr = self.smn_config['smn_influence_matrix_hebbian_learning_rate']
            effective_outcome_for_hebbian = 0.0
            min_orp_for_hebbian = self.E_OR_THRESHOLD * self.smn_config['smn_hebbian_orp_threshold_factor']

            if valence_mod_this_cycle > smn_pos_thresh and orp_at_collapse >= min_orp_for_hebbian:
                effective_outcome_for_hebbian = 1.0 
            elif valence_mod_this_cycle < smn_neg_thresh and orp_at_collapse >= min_orp_for_hebbian:
                effective_outcome_for_hebbian = -1.0 

            if abs(effective_outcome_for_hebbian) > 0 and self.smn_param_actual_changes_this_cycle:
                changed_param_keys_list = list(self.smn_param_actual_changes_this_cycle.keys())
                for i in range(len(changed_param_keys_list)):
                    for j in range(i, len(changed_param_keys_list)): 
                        param_key_A = changed_param_keys_list[i]
                        param_key_B = changed_param_keys_list[j]
                        idx_A = self.smn_param_indices[param_key_A]
                        idx_B = self.smn_param_indices[param_key_B]
                        change_A = self.smn_param_actual_changes_this_cycle[param_key_A]
                        change_B = self.smn_param_actual_changes_this_cycle[param_key_B]
                        scale_A = self.smn_params_runtime_state[param_key_A]['base_mutation_strength_ref'] + 1e-6
                        scale_B = self.smn_params_runtime_state[param_key_B]['base_mutation_strength_ref'] + 1e-6
                        norm_change_A = np.tanh(change_A / (scale_A * 2.0)) 
                        norm_change_B = np.tanh(change_B / (scale_B * 2.0))
                        correlation_term = norm_change_A * norm_change_B
                        delta_weight = effective_outcome_for_hebbian * hebbian_lr * correlation_term

                        if abs(delta_weight) > 1e-7:
                            current_w_val = self.smn_influence_matrix[idx_A, idx_B] 
                            if idx_A == idx_B: 
                                self.smn_influence_matrix[idx_A, idx_A] += delta_weight
                            else: 
                                self.smn_influence_matrix[idx_A, idx_B] += delta_weight
                                self.smn_influence_matrix[idx_B, idx_A] += delta_weight
                            self._log_lot_event("smn_graph_hebbian.update", {
                                "pA":param_key_A, "pB":param_key_B, "chA":change_A, "chB":change_B,
                                "corr":correlation_term, "eff_out":effective_outcome_for_hebbian, "dW":delta_weight,
                                "old_w": current_w_val, "new_w":self.smn_influence_matrix[idx_A, idx_B]
                            })
            self.smn_influence_matrix *= (1.0 - self.smn_config['smn_influence_matrix_weight_decay'])
            np.clip(self.smn_influence_matrix,
                    self.smn_config['smn_influence_matrix_clip_min'],
                    self.smn_config['smn_influence_matrix_clip_max'],
                    out=self.smn_influence_matrix)

        if any_strategy_weights_mutated:
            sw = self.internal_state_parameters['strategy_weights']
            valid_sw_values = [v for v in sw.values() if isinstance(v, (int, float))]
            total_sw = sum(v for v in valid_sw_values if v > 0)
            if total_sw > 1e-6 :
                for k_sw in sw:
                    if isinstance(sw[k_sw], (int,float)) : sw[k_sw] = max(0, sw[k_sw]/total_sw)
                if self.verbose >= 3: print(f"      SMN: Re-Normalized strategy_weights: { {k: f'{v:.2f}' for k,v in sw.items()} }")
            else:
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
            
            # MODIFIED for Demo 2, Fix 3: Boost Attention Recovery
            attention_boost_after_firewall_wm_clear = 0.15 
            wm_cleared_by_firewall_this_time = False

            if not self.working_memory.is_empty() and self.firewall_params.get('clear_wm_on_intervention', True):
                self.working_memory.clear()
                self._log_wm_op("clear", details={"reason": "firewall_intervention"})
                if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL: Cleared Working Memory due to intervention.")
                wm_cleared_by_firewall_this_time = True
            
            if wm_cleared_by_firewall_this_time:
                old_attn = self.internal_state_parameters['attention_level']
                self.internal_state_parameters['attention_level'] = min(1.0, old_attn + attention_boost_after_firewall_wm_clear)
                if self.verbose >=1: print(f"[{self.agent_id}] FIREWALL: Attention boosted by {attention_boost_after_firewall_wm_clear:.2f} to {self.internal_state_parameters['attention_level']:.2f} after WM clear.")
                self._log_lot_event("firewall.intervention.attention_boost", {"old_attn": old_attn, "boost": attention_boost_after_firewall_wm_clear, "new_attn": self.internal_state_parameters['attention_level']})


            if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
                if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL: Current goal '{self.current_goal_state_obj.current_goal}' status changed to 'pending' due to intervention.")
                self.current_goal_state_obj.status = "pending" # Interrupt the goal, force re-evaluation
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

        # Logic related to goal state might influence op_gen by setting preferred_state,
        # which happens in _executive_generate_computation_sequence now, using WM context.
        
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
        orp_at_collapse = self.current_orp_before_reset 

        if self.verbose >= 1: print(f"  ExecutiveLayer OR: Collapsed to |{collapsed_outcome_str}> (ORP experienced: {orp_at_collapse:.3f}, Early OR: {or_triggered_early}, Entropy: {entropy_at_collapse:.2f})")

        raw_valence_of_collapse = self.outcome_valence_map.get(collapsed_outcome_str, -0.15)
        self._executive_handle_collapse_interrupts(orp_at_collapse, executed_sequence, raw_valence_of_collapse)

        # Goal progress and WM context management for the current step happens within _executive_evaluate_outcome_and_update_mood
        # specifically when it calls _executive_update_goal_progress
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
        self._meta_layer_adapt_preferred_state(collapsed_outcome_str, self.last_cycle_valence_mod) # May be influenced by WM context presence
        if self.verbose >= 1: print(f"  MetaLayer State: Attn={self.internal_state_parameters['attention_level']:.2f},Cur={self.internal_state_parameters['curiosity']:.2f},PrefS=|{str(self.internal_state_parameters['preferred_logical_state'])}>,Load={self.internal_state_parameters['cognitive_load']:.2f}")


        prev_mod_valence_for_smn = self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0.0
        self._smn_update_and_apply_mutations(self.last_cycle_valence_mod, self.last_cycle_valence_raw, prev_mod_valence_for_smn, orp_at_collapse)

        self._firewall_detect_and_correct_anomalies() # Firewall might clear WM or alter goal status

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
            "smn_influence_matrix_sample": self.smn_influence_matrix[:2,:2].tolist() if self.smn_influence_matrix.size > 0 else "N/A", 
            "goal_state_at_cycle_end": self.current_goal_state_obj.to_dict() if self.current_goal_state_obj else None,
            "working_memory_summary_at_cycle_end": self.working_memory.to_dict_summary() if self.working_memory else None, 
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
                elif imag_part_str: term_str = imag_part_str.replace("+","") # remove leading + if only imag
                else: term_str = "0.000" # Should not happen if abs(amp) > 1e-9

                active_terms.append(f"{term_str}|{state}>")
        return " + ".join(active_terms) if active_terms else "ZeroSuperposition"


    def set_goal_state(self, goal_state_obj: GoalState):
        if not isinstance(goal_state_obj, GoalState) and goal_state_obj is not None:
            raise ValueError("goal_state_obj must be an instance of GoalState or None.")
        
        old_goal_name = None
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            old_goal_name = self.current_goal_state_obj.current_goal
            # Pop context if the top WM item is specifically for the goal being replaced/cleared.
            if not self.working_memory.is_empty():
                top_item = self.working_memory.peek()
                if top_item.type == "goal_step_context" and top_item.data.get("goal_name") == old_goal_name:
                    # More refined: check if it matches the specific current step being replaced.
                    # For simplicity, if *any* context for the old goal is on top, pop it.
                    popped_item = self.working_memory.pop()
                    self._log_wm_op("pop_goal_context", item=popped_item, details={"reason": f"goal_replaced_or_cleared (was: {old_goal_name})"})
                    if self.verbose >= 1: print(f"[{self.agent_id}] Popped WM context for '{old_goal_name}' as goal is being replaced/cleared by '{goal_state_obj.current_goal if goal_state_obj else 'None'}'.")

        self.current_goal_state_obj = goal_state_obj
        if self.current_goal_state_obj:
            self.current_goal_state_obj.status = "active"
            self.current_goal_state_obj.current_step_index = 0
            self.current_goal_state_obj.progress = 0.0
            # History might be reset for a truly "new" goal, or preserved if it's a reactivation.
            # self.current_goal_state_obj.history = [] # Uncomment if new goal means fresh history
            
            # If the first step of the new goal requires explicit WM context push.
            if self.current_goal_state_obj.steps and \
               0 <= self.current_goal_state_obj.current_step_index < len(self.current_goal_state_obj.steps) and \
               self.current_goal_state_obj.steps[0].get("requires_explicit_wm_context_push", False):
                
                first_step = self.current_goal_state_obj.steps[0]
                step_name = first_step.get("name", "Step 1 (of new goal)")
                wm_item_data = {
                    "goal_name": self.current_goal_state_obj.current_goal, 
                    "goal_step_name": step_name, 
                    "step_index": 0,
                    "collapsed_state_at_eval_time": self.collapsed_logical_state_str, # State at the time of setting this initial context
                    "cycle_num_pushed_for_eval": self.current_cycle_num # Pushed when goal is set / step 0 becomes active
                }
                item_to_push = WorkingMemoryItem(type="goal_step_context", data=wm_item_data, 
                                                 description=f"Initial Ctx: {self.current_goal_state_obj.current_goal} - {step_name}")
                item_discarded_on_set_goal_push = self.working_memory.push(item_to_push)
                self._log_wm_op("push_goal_context", item=item_to_push, 
                                details={'reason':'new_goal_set_step0_req_push', 
                                         'item_discarded_on_push': item_discarded_on_set_goal_push})

            if self.verbose >= 1: print(f"[{self.agent_id}] New goal set and activated: {self.current_goal_state_obj}")
            self._log_lot_event("goal.set", {"goal_name":self.current_goal_state_obj.current_goal, "num_steps":len(self.current_goal_state_obj.steps)})
        else: # Goal is being cleared (set to None)
            if self.verbose >= 1: print(f"[{self.agent_id}] Goal cleared (was: {old_goal_name if old_goal_name else 'None'}).")
            self._log_lot_event("goal.cleared", {"previous_goal_name": old_goal_name if old_goal_name else "None"})


    def print_internal_state_summary(self, indent="  ", custom_logger=None):
        log_func = custom_logger if callable(custom_logger) else print

        log_func(f"{indent}--- Internal State Summary for Agent {self.agent_id} (Cycle {self.current_cycle_num}) ---")
        log_func(f"{indent}  State: Mood: {self.internal_state_parameters['mood']:.2f}, Attn: {self.internal_state_parameters['attention_level']:.2f}, Load: {self.internal_state_parameters['cognitive_load']:.2f}, Frust: {self.internal_state_parameters['frustration']:.2f}")
        log_func(f"{indent}  Cognition: Cur: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}, PrefState: |{str(self.internal_state_parameters['preferred_logical_state'])}>, CompLenPref: {self.internal_state_parameters['computation_length_preference']}")
        log_func(f"{indent}  Exploration: Countdown: {self.internal_state_parameters['exploration_mode_countdown']}")
        log_func(f"{indent}  OrchOR: E_OR_THRESH: {self.E_OR_THRESHOLD:.3f} (AdaptRate: {self.orp_threshold_dynamics['adapt_rate']:.4f}), ORP_DECAY: {self.orp_decay_rate:.4f} (AdaptRate: {self.orp_decay_dynamics['adapt_rate']:.4f})")
        sw_str = ", ".join([f"{k.upper()[0]}:{v:.2f}" for k,v in self.internal_state_parameters['strategy_weights'].items()])
        log_func(f"{indent}  StrategyWeights: {sw_str}")
        log_func(f"{indent}  MetaCog: ReviewIn: {self.metacognition_params['review_interval']-self.metacognition_params['cycles_since_last_review']}, AdaptRates(Cur/Goal): {self.metacognition_params['curiosity_adaptation_rate']:.3f}/{self.metacognition_params['goal_bias_adaptation_rate']:.3f}")
        log_func(f"{indent}  LTM: {len(self.long_term_memory)}/{self.long_term_memory_capacity} entries. UtilWeights(V/E): {self.ltm_utility_weight_valence:.2f}/{self.ltm_utility_weight_efficiency:.2f}")
        log_func(f"{indent}  TemporalGrid: {len(self.temporal_feedback_grid)}/{self.temporal_grid_params['max_len']} (FeedbackWin: {self.temporal_grid_params['feedback_window']}). BiasStr(V/E): {self.internal_state_parameters['temporal_feedback_valence_bias_strength']:.2f}/{self.internal_state_parameters['temporal_feedback_entropy_bias_strength']:.2f}")
        
        wm_summary = self.working_memory.to_dict_summary()
        top_item_str = "Empty"
        if wm_summary['top_item_summary']:
            top_item_str = f"{wm_summary['top_item_summary']['type']} ('{wm_summary['top_item_summary']['description'][:20]}...')"
        log_func(f"{indent}  WorkingMemory: Depth: {wm_summary['current_depth']}/{wm_summary['max_depth']}, Top: {top_item_str}")


        smn_enabled = self.smn_config.get('enabled')
        smn_graph_enabled = self.smn_config.get('enable_influence_matrix')
        if smn_enabled:
            log_func(f"{indent}  SMN Active (Graph: {'On' if smn_graph_enabled else 'Off'}). Controlled params mut_strengths:")
            for p_name_smn, p_state_info in list(self.smn_params_runtime_state.items())[:3]: # Print sample
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
            "active_goal_current_step_name": self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index].get("name", f"Step {self.current_goal_state_obj.current_step_index+1}") if self.current_goal_state_obj and self.current_goal_state_obj.steps and 0 <= self.current_goal_state_obj.current_step_index < len(self.current_goal_state_obj.steps) else None,
            "working_memory_depth": len(self.working_memory) if self.working_memory else 0,
            "verbose": self.verbose, # To allow completion_criteria to check agent's verbose level
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
                wm_depth_info = f"WMd:{len(self.working_memory)}"

                print(f"\n>>>> Chained Cycle {i+1}/{num_cycles} for {self.agent_id} <<<< Input: |{current_input_str_for_cycle}>; Mood:{self.internal_state_parameters['mood']:.2f}; Pref:{pref_str}; {goal_summary}; {wm_depth_info}")

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
            # Ensure 'config_overrides' in agent_specific_config is updated, not just overwritten
            existing_overrides = agent_specific_config.get('config_overrides', {})
            agent_specific_config['config_overrides'] = {**existing_overrides, **agent_overrides}

            if agent_trainable_params_init:
                 agent_specific_config['trainable_param_values'] = agent_trainable_params_init

            final_agent_verbose = agent_specific_config['config_overrides'].get(('verbose',), self.verbose -1 if self.verbose > 0 else 0)
            agent_specific_config['verbose'] = final_agent_verbose


            try:
                emulator = SimplifiedOrchOREmulator(**agent_specific_config)
                self.agents.append(emulator)
                if self.verbose >= 1: print(f"  Initialized {agent_id}. Verbose: {final_agent_verbose}.")
                if self.verbose >=2 and (agent_specific_config['config_overrides'] or agent_trainable_params_init): # Check effective overrides
                    print(f"    {agent_id} Initial Overrides Applied: {agent_specific_config['config_overrides']}")
                    if agent_trainable_params_init: print(f"    {agent_id} Initial Trainable Params: {agent_trainable_params_init}")
                    emulator.print_internal_state_summary(indent="      ")
            except Exception as e:
                print(f"CRITICAL ERROR Initializing {agent_id}: {type(e).__name__} - {e}")
                traceback.print_exc()

    def run_system_cycles(self, num_system_cycles, initial_input_per_agent_list=None):
        if self.verbose >= 0: print(f"\n\n========= CoAgentManager: Starting {num_system_cycles} System Cycles =========")

        for i_sys_cycle in range(num_system_cycles):
            self.system_cycle_num += 1
            if self.verbose >=0: print(f"\n------- System Cycle {self.system_cycle_num}/{num_system_cycles} (Manager Cycle {i_sys_cycle+1}) -------")

            agent_threads = [] # For future threading if needed, now sequential

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

            interaction_interval = max(3, min(10, num_system_cycles // 5 if num_system_cycles // 5 > 0 else 3))
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
        
        # Determine a consensus preferred state from top performers (agent00, agent01) if available
        # This is for Demo 4, Fix 2
        consensus_pref_state_from_top_agents = None
        top_agent_pref_states = []
        for top_idx in range(min(2, len(avg_performances))): # Look at top 2 agents
            top_agent_obj = avg_performances[top_idx]['agent_obj']
            if top_agent_obj.internal_state_parameters['preferred_logical_state'] is not None:
                top_agent_pref_states.append(top_agent_obj.internal_state_parameters['preferred_logical_state'])
        
        if top_agent_pref_states:
            counts = collections.Counter(top_agent_pref_states)
            most_common_pref_state, _ = counts.most_common(1)[0]
            consensus_pref_state_from_top_agents = most_common_pref_state
            if self.verbose >=2 : print(f"    CoAgentManager: Consensus preferred_logical_state from top agents: |{consensus_pref_state_from_top_agents}>")


        for i in range(num_learners):
            learner_idx_from_bottom = i
            learner_data_idx_in_sorted_list = len(avg_performances) - 1 - learner_idx_from_bottom
            
            if learner_data_idx_in_sorted_list <= 0: break 
            learner_data = avg_performances[learner_data_idx_in_sorted_list]
            if teacher_data['agent_id'] == learner_data['agent_id']: continue

            # MODIFIED for Demo 4, Fix 1: Lower Learning Threshold
            performance_gap_threshold_abs = 0.15 
            
            learner_agent = learner_data['agent_obj']
            teacher_agent = teacher_data['agent_obj']

            # Condition for any learning intervention for this learner
            should_intervene_learner = teacher_data['perf'] > learner_data['perf'] + performance_gap_threshold_abs and learner_data['perf'] < 0.15 # Slightly wider perf condition to intervene

            if should_intervene_learner:
                if self.verbose >= 1: print(f"    {learner_agent.agent_id} (perf {learner_data['perf']:.2f}) learning from {teacher_agent.agent_id} (perf {teacher_data['perf']:.2f})")

                # Align trainable parameters (original logic)
                params_to_align = list(DEFAULT_TRAINABLE_PARAMS_CONFIG.keys()) 
                alignment_factor = random.uniform(0.1, 0.25)
                teacher_params_for_reference = {}
                for param_name_for_teacher in params_to_align:
                    config_teacher = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name_for_teacher]
                    dict_attr_teacher = config_teacher['target_dict_attr']
                    key_teacher = config_teacher['target_key']
                    subkey_teacher = config_teacher.get('target_subkey')
                    try:
                        val = None
                        if dict_attr_teacher:
                            target_dict_on_teacher = getattr(teacher_agent, dict_attr_teacher)
                            if subkey_teacher:
                                val = target_dict_on_teacher[key_teacher][subkey_teacher]
                            else:
                                val = target_dict_on_teacher[key_teacher]
                        else:
                            val = getattr(teacher_agent, key_teacher)
                        teacher_params_for_reference[param_name_for_teacher] = val
                    except (AttributeError, KeyError, TypeError) as e:
                        if self.verbose >= 2: print(f"      Skipping param retrieval for {param_name_for_teacher} from teacher {teacher_agent.agent_id} (Path: {dict_attr_teacher}, {key_teacher}, {subkey_teacher}): {type(e).__name__} - {e}")
                        continue
                
                learner_current_params_for_update = {}
                for param_name, teacher_val in teacher_params_for_reference.items():
                    config = DEFAULT_TRAINABLE_PARAMS_CONFIG[param_name]
                    dict_attr = config['target_dict_attr']
                    key = config['target_key']
                    subkey = config.get('target_subkey')
                    try:
                        current_learner_val = None
                        if dict_attr: 
                            target_learner_dict = getattr(learner_agent, dict_attr)
                            if subkey: 
                                current_learner_val = target_learner_dict[key][subkey]
                            else: 
                                current_learner_val = target_learner_dict[key]
                        else: 
                            current_learner_val = getattr(learner_agent, key)
                        nudged_val = current_learner_val * (1 - alignment_factor) + teacher_val * alignment_factor
                        nudged_val += np.random.normal(0, config['perturb_scale'] * 0.05)
                        nudged_val = np.clip(nudged_val, config['min'], config['max'])
                        learner_current_params_for_update[param_name] = nudged_val
                        copied_count +=1
                    except (AttributeError, KeyError, TypeError) as e:
                         if self.verbose >=1: print(f"      Error nudging/retrieving {param_name} for learner {learner_agent.agent_id} (Path: {dict_attr}, {key}, {subkey}): {type(e).__name__} {e}")

                if learner_current_params_for_update:
                    learner_agent.update_emulator_parameters(learner_current_params_for_update)
                    if self.verbose >=2 : print(f"      {learner_agent.agent_id} applied {len(learner_current_params_for_update)} trainable param updates from {teacher_agent.agent_id}.")
                    learner_agent._log_lot_event("coagent.learn_from_peer.params", {"teacher_id": teacher_agent.agent_id, "learner_perf":learner_data['perf'], "teacher_perf":teacher_data['perf'], "num_params_aligned": len(learner_current_params_for_update)})
                
                # MODIFIED for Demo 4, Fix 2: Align Agent02’s Preferred State (applied to current underperforming learner_agent if consensus_pref_state exists)
                # This applies specifically if this learner is agent02, or more generally to underperformers.
                # Assuming fix meant for "agent02" is prototypical of an underperformer to align.
                if consensus_pref_state_from_top_agents and learner_agent.internal_state_parameters['preferred_logical_state'] != consensus_pref_state_from_top_agents:
                    old_pref_state_learner = learner_agent.internal_state_parameters['preferred_logical_state']
                    learner_agent.internal_state_parameters['preferred_logical_state'] = consensus_pref_state_from_top_agents
                    if self.verbose >= 1: print(f"      {learner_agent.agent_id} preferred_logical_state aligned to consensus |{consensus_pref_state_from_top_agents}> (was |{old_pref_state_learner}>).")
                    learner_agent._log_lot_event("coagent.learn_from_peer.pref_state", {"teacher_id": "consensus_top_agents", "old_pref_state": old_pref_state_learner, "new_pref_state": consensus_pref_state_from_top_agents})
                    copied_count +=1 # Count this as a learning action

                # MODIFIED for Demo 4, Fix 3: Increase SMN Mutations for underperformers
                if learner_agent.smn_config.get('enabled', False):
                    old_smn_scale = learner_agent.internal_state_parameters['smn_perturbation_scale_factor']
                    new_smn_scale = min(old_smn_scale * 1.20, 0.2) # Increase by 20%, capped at 0.2
                    if new_smn_scale > old_smn_scale + 1e-4 : # Avoid tiny changes triggering log
                        learner_agent.internal_state_parameters['smn_perturbation_scale_factor'] = new_smn_scale
                        if self.verbose >= 1: print(f"      {learner_agent.agent_id} SMN perturbation_scale_factor increased to {new_smn_scale:.4f} (was {old_smn_scale:.4f}).")
                        learner_agent._log_lot_event("coagent.learn_from_peer.smn_boost", {"old_smn_scale":old_smn_scale, "new_smn_scale":new_smn_scale})
                        copied_count +=1


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
        
        trainer_specific_keys_to_remove = [
            'verbose_emulator_episodes', 
            'trainer_goal_completion_reward',
            'trainer_goal_failure_penalty',
            'trainer_goal_progress_reward_factor'
        ]
        for key_to_remove in trainer_specific_keys_to_remove:
            if key_to_remove in init_args:
                del init_args[key_to_remove]
        
        base_overrides = self.base_emulator_config.get('config_overrides', {})
        existing_init_overrides = init_args.get('config_overrides', {})
        if not isinstance(existing_init_overrides, dict): existing_init_overrides = {}
        init_args['config_overrides'] = {**base_overrides, **existing_init_overrides}
        
        return init_args

    def run_episode(self, episode_params_to_test, num_cycles, initial_input="00", task_goal_state_obj=None):
        emulator_kwargs = self._get_emulator_init_args(episode_params_to_test)
        emulator_kwargs['agent_id'] = emulator_kwargs.get('agent_id', "trainer_ep_agent")
        emulator = SimplifiedOrchOREmulator(**emulator_kwargs)

        task_pref_state = self.base_emulator_config.get('initial_internal_states', {}).get('preferred_logical_state')
        if task_pref_state:
            emulator.internal_state_parameters['preferred_logical_state'] = task_pref_state

        if task_goal_state_obj: # Make a fresh copy for the episode
            emulator.set_goal_state(copy.deepcopy(task_goal_state_obj))

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

        if task_goal_state_obj and emulator.current_goal_state_obj: # Check against the one used by emulator
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
        
        # Create the goal object template once if provided
        goal_obj_template_for_episode = None
        if training_goal_state_template_dict:
            goal_obj_template_for_episode = GoalState(**copy.deepcopy(training_goal_state_template_dict))


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
            
            # Use a fresh copy of the goal template for each episode
            current_episode_goal_obj = copy.deepcopy(goal_obj_template_for_episode) if goal_obj_template_for_episode else None


            reward, ep_history, ep_outcome_details = self.run_episode(
                candidate_params, cycles_per_episode, initial_input,
                task_goal_state_obj=current_episode_goal_obj # Pass the copy
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
                    current_perturb_scales[name] = max(current_perturb_scales[name], 0.00005) # Prevent scales from becoming too small

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
    demo0_lot_config_details['smn_graph_propagation'] = True 
    demo0_lot_config_details['smn_graph_hebbian'] = True
    demo0_lot_config_details['workingmemory_ops'] = True # Enable general WM logging
    demo0_lot_config_details['workingmemory.push_goal_context'] = True 
    demo0_lot_config_details['workingmemory.pop_goal_context'] = True

    demo0_config = {
        'verbose': MASTER_VERBOSE_LEVEL, 'cycle_history_max_len': 6,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate': 0.02,
        'lot_config': {'enabled': True, 'log_level_details': demo0_lot_config_details},
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': False},
        'temporal_grid_config': {'max_len':3},
        'working_memory_max_depth': 5, 
    }
    emulator_demo0 = SimplifiedOrchOREmulator(agent_id="agent_demo0_LayersLoT", **demo0_config)
    emulator_demo0.internal_state_parameters['preferred_logical_state'] = "11"
    emulator_demo0.internal_state_parameters['curiosity'] = 0.8
    print(f"Running {emulator_demo0.agent_id} for 3 cycles to demonstrate layered processing and comprehensive LoT output (including WM).")
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
            'smn_graph_propagation': True, 'smn_graph_hebbian':True, 
            'executive.opgen.strategy_selected':True, 'cycle_start':True, 'valence_eval':True,
            'workingmemory_ops': False # Keep WM quiet here to focus on SMN/TFG
        }
    }
    demo1_smn_general_config = { 
            'enabled': True,
            'mutation_trigger_min_valence_gain': 0.08,
            'enable_influence_matrix': True, 
            'smn_influence_matrix_hebbian_learning_rate': 0.02, 
            'smn_influence_propagation_threshold': 0.1, 
            'smn_secondary_mutation_scale_factor': 0.6, 
    }
    demo1_smn_controlled_params = { 
             'computation_length_preference': {'base_mutation_strength': 0.5, 'min_val': 1, 'max_val': 6, 'is_int': True, 'path': ('internal_state_parameters', 'computation_length_preference')},
             'mc_cur_adapt_rate': {'base_mutation_strength': 0.01, 'min_val':0.005,'max_val':0.15, 'path': ('metacognition_params', 'curiosity_adaptation_rate')},
             'sw_curiosity': {'base_mutation_strength': 0.08, 'min_val':0.01, 'max_val':0.99, 'path': ('internal_state_parameters', 'strategy_weights', 'curiosity')},
    }

    demo1_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate':0.015,
        'temporal_grid_config': {
            'max_len': 6, 'feedback_window': 4,
            'low_valence_delta_threshold': -0.1, 'high_entropy_shift_threshold': 0.3,
        },
        'initial_internal_states': { 
            'temporal_feedback_valence_bias_strength': 0.25,
            'temporal_feedback_entropy_bias_strength': 0.15,
            'computation_length_preference': 2,
        },
        'smn_general_config': demo1_smn_general_config, 
        'smn_controlled_params_config': demo1_smn_controlled_params, 
        'cognitive_firewall_config': {'enabled': False},
        'lot_config': demo1_lot_config,
        'working_memory_max_depth': 3, 
    }
    emulator_demo1 = SimplifiedOrchOREmulator(agent_id="agent_demo1_TFG_SMN_Graph", **demo1_config)

    initial_smn_vals_d1 = {
        key: emulator_demo1._smn_get_param_value(info['path'])
        for key, info in emulator_demo1.smn_controlled_params_definitions.items()
    }
    print(f"{emulator_demo1.agent_id} starting with initial SMN values: {initial_smn_vals_d1}. SMN Graph enabled. Running 18 cycles.")
    emulator_demo1.outcome_valence_map = {"00": -0.6, "01": 0.75, "10": -0.3, "11": 0.4}
    emulator_demo1.run_chained_cognitive_cycles("00", 18) 

    print(f"\n{emulator_demo1.agent_id} Final SMN-controlled parameter values:")
    for key, init_val in initial_smn_vals_d1.items():
        final_val = emulator_demo1._smn_get_param_value(emulator_demo1.smn_controlled_params_definitions[key]['path'])
        val_format = ".0f" if emulator_demo1.smn_controlled_params_definitions[key].get('is_int') else ".3f"
        print(f"  {key}: {final_val:{val_format}} (was {init_val:{val_format}})")

    if emulator_demo1.smn_config.get('enable_influence_matrix',False) and emulator_demo1.smn_influence_matrix.size > 0 :
        print(f"{emulator_demo1.agent_id} Final SMN Influence Matrix (sample, rounded to 3dp):")
        num_p = emulator_demo1.smn_influence_matrix.shape[0]
        sample_size = min(num_p, 3) 
        for i in range(sample_size):
            row_str = ", ".join([f"{x:.3f}" for x in emulator_demo1.smn_influence_matrix[i, :sample_size]])
            param_name_i = emulator_demo1.smn_param_names_from_indices.get(i, f"P{i}")[:10].ljust(10)
            print(f"  {param_name_i}: [{row_str}]")

    emulator_demo1.print_internal_state_summary(indent="  [Demo1 Summary] ")


    # --- DEMO 2: Interrupt Handlers & Cognitive Firewall (Features 4 & 6) ---
    print("\n\n--- DEMO 2: Interrupt Handlers & Cognitive Firewall (Features 4 & 6) ---")
    demo2_lot_config = {
        'enabled': True,
        'log_level_details':{
            'firewall.intervention':True, 'executive.interrupt_handler':True,
            'cycle_start':True, 'executive.opgen.interrupt_bias.force_ltm':True,
            'workingmemory_ops': True,
            'workingmemory.clear': True,
            'goal_tracking': True, # To see goal activity
            }
    }
    demo2_firewall_config_override = { # Using specific overrides for demo
            'enabled': True, 'check_interval': 4, 'cooldown_duration': 6, # Slightly less aggressive check
            'low_valence_threshold': -0.65, # As suggested: low_valence_threshold for FW more tolerant
            'low_valence_streak_needed': 3,
            'loop_detection_window': 6, 'loop_detection_min_repeats': 3,
            'premature_collapse_orp_max_ratio':0.35, 'premature_collapse_streak_needed':3,
            'clear_wm_on_intervention': True
    }
    demo2_interrupt_config_override = {
        'enabled': True, 'reactive_ltm_valence_threshold': -0.85,
        'consolidation_valence_abs_threshold':0.8,
        'consolidation_strength_bonus': 1.5 # Dampen LTM consolidation bonus a bit
    }

    demo2_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.5, # Slightly higher threshold
        'initial_orp_decay_rate': 0.01,
        'interrupt_handler_config': demo2_interrupt_config_override,
        'cognitive_firewall_config': demo2_firewall_config_override,
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'lot_config': demo2_lot_config,
        'config_overrides': {
            # MODIFIED for Demo 2, Fix 1: Soften the Valence Map
            ('outcome_valence_map',): {"00": -0.5, "01": -0.4, "10": -0.5, "11": 0.3} 
        },
        'working_memory_max_depth': 5,
        'metacognition_config': {'review_interval': 5} # Override for this specific demo's agent
    }
    emulator_demo2 = SimplifiedOrchOREmulator(agent_id="agent_demo2_IntFW", **demo2_config)

    # Introduce a simple goal for Demo 2
    goal_steps_demo2 = [
        {"name": "Survive Initial Punishment", "target_state": "11", "requires_explicit_wm_context_push": True, "max_cycles_on_step": 7},
        {"name": "Explore a bit", "target_state": "01", "requires_explicit_wm_context_push": False, "max_cycles_on_step": 7} # A follow up
    ]
    task_goal_demo2 = GoalState(current_goal="Demo2 Survival and Exploration", steps=goal_steps_demo2)
    emulator_demo2.set_goal_state(task_goal_demo2)

    fix_it_seq_demo2 = (('H',0),('X',1),('H',1)) # Represents a "good" sequence to recall
    emulator_demo2.long_term_memory[fix_it_seq_demo2] = {
        'count':10, 'total_valence': 7.0, 'avg_valence':0.7, 'total_orp_cost':1.0, 'avg_orp_cost':0.1, # Slightly less super-valuable
        'total_entropy_generated':2.0, 'avg_entropy':0.2, 'utility':0.65, 'last_cycle':0
    }
    print(f"{emulator_demo2.agent_id} starting with punishing valence map AND A GOAL. Expecting Firewall/Interrupt activity. Running 15 cycles.")

    # Push something to WM to see if firewall clears it (or goal context handles it).
    wm_test_item = WorkingMemoryItem(type="dummy_context", data={"info":"test_firewall_clear"}, description="Pre-firewall item for Demo2")
    discarded = emulator_demo2.working_memory.push(wm_test_item)
    emulator_demo2._log_wm_op("push", item=wm_test_item, details={"reason": "manual_test_push_demo2", "item_discarded_on_push":discarded})

    emulator_demo2.run_chained_cognitive_cycles("00", 15) # Increased cycles for goal
    emulator_demo2.print_internal_state_summary(indent="  [Demo2 Summary] ")


    # --- DEMO 3: Goal-Oriented State Machine with Working Memory (Feature 7 & NEW WM) ---
    print("\n\n--- DEMO 3: Goal-Oriented State Machine with Working Memory (Feature 7 & NEW WM) ---")
    demo3_lot_details = {
        'goal_tracking': True, 'executive.opgen.strategy_selected':True, 
        'executive.plannext.goal_override':True, 'executive.outcome_eval.valence':True, 
        'cycle_start':True, 
        'workingmemory_ops': True, 
        'executive.opgen.wm_peek': True, 'executive.opgen.wm_ops_hint_available':True,
        'workingmemory.push_goal_context': True, 
        'workingmemory.pop_goal_context': True,
        'meta.adapt_pref_state':True 
    }
    demo3_config = {
        'verbose': MASTER_VERBOSE_LEVEL,
        'lot_config': {'enabled': True, 'log_level_details': demo3_lot_details },
        'initial_internal_states': {'goal_seeking_bias': 0.6, 'computation_length_preference':3}, 
        'working_memory_max_depth': 7,
        # MODIFIED for Demo 3, Fix 3: Reduce WM Clear Frequency (by disabling firewall auto-clear for this agent)
        # Note: The "every 10 cycles" clear would be a new mechanism not present in the base code.
        # This fix prevents firewall from clearing WM, thus reducing clear frequency for this agent.
        'cognitive_firewall_config': {
            'enabled': True, # Assuming firewall should be on as per default. If not, this can be False.
            'clear_wm_on_intervention': False 
        }
    }
    emulator_demo3 = SimplifiedOrchOREmulator(agent_id="agent_demo3_Goal_WM", **demo3_config)

    def demo3_step2_callable_criterion(context_dict):
        """
        Completion criterion for "From 01, reach state 10".
        Requires:
        1. Collapsed state to be the target_state of this step ("10").
        2. The working memory's top item should be the goal_step_context FOR THIS VERY STEP,
           indicating WM is actively managing this step.
        """
        current_step_obj = context_dict['current_step_obj']
        target_state_for_this_step = current_step_obj.get("target_state")
        step_name_for_this_step = current_step_obj.get("name")

        state_achieved = context_dict['collapsed_state'] == target_state_for_this_step
        
        wm_context_matches = False
        wm = context_dict['working_memory']
        if not wm.is_empty():
            top_item = wm.peek()
            if top_item.type == "goal_step_context" and \
               top_item.data.get("goal_name") == context_dict['current_goal_obj'].current_goal and \
               top_item.data.get("goal_step_name") == step_name_for_this_step and \
               top_item.data.get("step_index") == context_dict['current_goal_obj'].current_step_index:
                wm_context_matches = True
        
        agent_state = context_dict['agent_public_state']
        if agent_state.get('verbose',0) >=2: # Check agent's verbosity
             print(f"    DEMO3 Criterion Check ('{step_name_for_this_step}'): State Achieved ({context_dict['collapsed_state']} vs {target_state_for_this_step}): {state_achieved}, WM Context Matches Current Step: {wm_context_matches}")

        return state_achieved and wm_context_matches


    goal_steps_demo3 = [
        {"name": "Reach state 01", "target_state": "01", "requires_explicit_wm_context_push": True, "next_ops_hint": [('X',0),('H',1)], "max_cycles_on_step": 4},
        {"name": "From 01, reach state 10", 
         "target_state": "10", 
         "completion_criteria": demo3_step2_callable_criterion, # Uses target_state AND WM check
         "next_input_for_world":"01", "requires_explicit_wm_context_push": True, 
         # MODIFIED for Demo 3, Fix 2: Extend Step Timeout
         "max_cycles_on_step": 8},
        {"name": "From 10, reach state 11 (final)", "target_state": "11", "next_input_for_world":"10", "requires_explicit_wm_context_push": True, "max_cycles_on_step": 4}
    ]
    task_goal_demo3 = GoalState(current_goal="WM-Enhanced Multi-step Task", steps=goal_steps_demo3, error_tolerance=0.1)
    emulator_demo3.set_goal_state(task_goal_demo3)

    print(f"{emulator_demo3.agent_id} attempting WM-enhanced goal: '{emulator_demo3.current_goal_state_obj.current_goal}'. Running up to 25 cycles.")
    # MODIFIED for Demo 3 Fix: op_template_d3[1] changed to achieve "10" from "01"
    # op_template_d3[1] is used in Agent's Cycle 2, when Step 2 ("From 01, reach state 10") becomes active.
    # Input |01> --X(1)--> |11> --X(0)--> |10>
    op_template_d3 = [
        [('X',0)],              # Cycle 1 (i=0): For Step 0 (|00> -> |01>)
        [('X',1), ('X',0)],     # Cycle 2 (i=1): For Step 1 (|01> -> |10>) - MODIFIED
        [('X',0)], # Cycle 3 (i=2): For Step 2 (final step |10> -> |11>) - MODIFIED
        [('CNOT',(0,1))],       # Cycle 4 (i=3)
        [],                     # Cycle 5 (i=4)
        [('Z',0),('H',0)],      # Cycle 6 (i=5)
        [('X',0)]               # Cycle 7 (i=6)
    ] # more variation
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
        'cognitive_firewall_config': {'enabled': True, 'check_interval': 5, 'cooldown_duration': 7, 'low_valence_streak_needed':3, 'clear_wm_on_intervention': True},
        'temporal_grid_config': {'max_len':8, 'feedback_window':4},
        'lot_config': {'enabled': False }, # Keep LoT off for co-agent summary readability by default
        'working_memory_max_depth': 10 
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
        'initial_internal_states': {
            'preferred_logical_state': "01",
            # MODIFIED for Demo 5, Fix 2: Add Noise Mitigation (cut default 0.01 by 50%)
            'sensor_input_noise_level': 0.005 
            },
        'verbose_emulator_episodes': MASTER_VERBOSE_LEVEL - 2 if MASTER_VERBOSE_LEVEL > 1 else 0,
        'trainer_goal_completion_reward': 1.0, 'trainer_goal_failure_penalty': -0.6, 'trainer_goal_progress_reward_factor': 0.4,
        'config_overrides': { 
            ('outcome_valence_map',): {"00": -0.2, "01": 1.0, "10": -0.6, "11": 0.1},
            ('successful_sequence_threshold_valence',): 0.4
        },
        'working_memory_max_depth': 8
    }

    trainer_goal_template_dict_d5 = {"current_goal": "Trainer Task: Reach 01",
                                 "steps": [{"name": "Achieve state 01", "target_state": "01", "requires_explicit_wm_context_push":True}], 
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
    final_test_config_d5['working_memory_max_depth'] = trainer_base_emulator_config_d5.get('working_memory_max_depth', 20)
    final_test_config_d5['smn_general_config'] = {'enabled': True, 'enable_influence_matrix': True}
    final_test_config_d5['lot_config'] = {'enabled':True, 
                                          'log_level_details': {
                                              'workingmemory_ops':True, 'goal_tracking':True, 'cycle_start':True, 
                                              'executive.opgen.wm_peek': True, 'workingmemory.push_goal_context': True,
                                              'workingmemory.pop_goal_context':True
                                            }
                                          }


    trained_emulator_d5 = SimplifiedOrchOREmulator(**final_test_config_d5)
    
    task_pref_state_d5 = trainer_base_emulator_config_d5.get('initial_internal_states', {}).get('preferred_logical_state')
    if task_pref_state_d5:
         trained_emulator_d5.internal_state_parameters['preferred_logical_state'] = task_pref_state_d5
    if trainer_goal_template_dict_d5:
        trained_emulator_d5.set_goal_state(GoalState(**copy.deepcopy(trainer_goal_template_dict_d5)))

    trained_emulator_d5.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=15)
    print(f"\nFinal Working Memory for {trained_emulator_d5.agent_id}: Depth {len(trained_emulator_d5.working_memory)}")
    if not trained_emulator_d5.working_memory.is_empty():
        print(f"  Top WM Item: {trained_emulator_d5.working_memory.peek()}")
    
    print(f"  Final Goal Status for {trained_emulator_d5.agent_id}: {trained_emulator_d5.current_goal_state_obj}")


    print("\n\n--- ALL DEMOS COMPLETED ---")

