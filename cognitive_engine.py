# cognitive_engine.py

"""
This file is the "brain." It contains the core agent class and its
operational frameworks (Trainer, CoAgentManager). It contains no top-level 
configuration dictionaries; it imports them from `configurations.py` and
uses them as defaults.
"""

import numpy as np
import copy
import time
import random
import collections 
import traceback 
import math 
from typing import List, Dict, Any, Deque, Optional, Tuple, Callable

from core_abstractions import StateHandle, WorkingMemoryItem, WorkingMemoryStack, GoalState
from configurations import * # Import all default config dicts for use as defaults

# ---------------------------------------------------------------------------
# Class Definition: SimplifiedOrchOREmulator
# ---------------------------------------------------------------------------
class SimplifiedOrchOREmulator:
    def __init__(self, agent_id="agent0", cycle_history_max_len=100,
                 universe: Dict = None,
                 initial_E_OR_THRESHOLD=1.0, initial_orp_decay_rate=0.01,
                 internal_state_parameters_config=None, metacognition_config=None,
                 orp_threshold_dynamics_config=None, orp_decay_dynamics_config=None,
                 trainable_param_values=None,
                 temporal_grid_config=None,
                 smn_general_config=None,
                 smn_controlled_params_config=None,
                 interrupt_handler_config=None,
                 cognitive_firewall_config=None,
                 goal_state_params=None,
                 lot_config=None,
                 shared_long_term_memory=None,
                 shared_attention_foci=None,
                 working_memory_max_depth=20,
                 config_overrides=None,
                 verbose=0,
                 # The kwargs catch-all handles unused keys from unpacked dicts, like 'name' from the universe
                 **kwargs): 

        self.agent_id = agent_id
        self.verbose = verbose
        
        # MANDATORY REFACTOR: Use imported configs as defaults for None-type keyword arguments
        if universe is None:
            raise ValueError("A 'universe' configuration dictionary must be provided.")
        self.universe = universe
        start_comp_basis = self.universe['state_to_comp_basis'][self.universe['start_state']]
        
        # Core computational state remains in the string basis
        self.logical_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        self.logical_superposition[start_comp_basis] = 1.0 + 0j
        self.collapsed_computational_state_str = start_comp_basis
        self.current_conceptual_state = self.universe['start_state'] # State in agent's 'reality'

        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate

        self.operation_costs = {'X': 0.1, 'Z': 0.1, 'H': 0.3, 'CNOT': 0.4, 'CZ': 0.4, 'ERROR_PENALTY': 0.05, 'PLANNING_BASE': 0.02}
        self.last_cycle_valence_raw = 0.0
        self.last_cycle_valence_mod = 0.0
        self.current_orp_before_reset = 0.0

        # MANDATORY REFACTOR: Apply the new configuration loading pattern.
        self.internal_state_parameters = copy.deepcopy(DEFAULT_INTERNAL_PARAMS) if internal_state_parameters_config is None else copy.deepcopy(internal_state_parameters_config)
        
        # Handle string-based preferred state from old configs for compatibility.
        # This logic is based on the provided values, not the default, so it's placed after initialization.
        if self.internal_state_parameters.get('preferred_logical_state'):
             try:
                 handle = self.universe['comp_basis_to_state'][self.internal_state_parameters['preferred_logical_state']]
                 self.internal_state_parameters['preferred_state_handle'] = handle
                 del self.internal_state_parameters['preferred_logical_state'] # Remove old key
             except KeyError:
                 if self.verbose >=1: print(f"Warning: Could not convert old 'preferred_logical_state' string to a StateHandle.")

        self.metacognition_params = copy.deepcopy(DEFAULT_METACOGNITION_PARAMS) if metacognition_config is None else copy.deepcopy(metacognition_config)
        self.orp_threshold_dynamics = copy.deepcopy(DEFAULT_ORP_THRESHOLD_DYNAMICS) if orp_threshold_dynamics_config is None else copy.deepcopy(orp_threshold_dynamics_config)
        self.orp_decay_dynamics = copy.deepcopy(DEFAULT_ORP_DECAY_DYNAMICS) if orp_decay_dynamics_config is None else copy.deepcopy(orp_decay_dynamics_config)
        self.temporal_grid_params = copy.deepcopy(DEFAULT_TEMPORAL_GRID_PARAMS) if temporal_grid_config is None else copy.deepcopy(temporal_grid_config)
        self.smn_config = copy.deepcopy(DEFAULT_SMN_CONFIG) if smn_general_config is None else copy.deepcopy(smn_general_config)
        self.smn_controlled_params_definitions = copy.deepcopy(DEFAULT_SMN_CONTROLLED_PARAMS) if smn_controlled_params_config is None else copy.deepcopy(smn_controlled_params_config)
        self.interrupt_handler_params = copy.deepcopy(DEFAULT_INTERRUPT_HANDLER_CONFIG) if interrupt_handler_config is None else copy.deepcopy(interrupt_handler_config)
        self.firewall_params = copy.deepcopy(DEFAULT_COGNITIVE_FIREWALL_CONFIG) if cognitive_firewall_config is None else copy.deepcopy(cognitive_firewall_config)
        self.goal_state_config_params = copy.deepcopy(DEFAULT_GOAL_STATE_PARAMS) if goal_state_params is None else copy.deepcopy(goal_state_params)
        self.lot_config_params = copy.deepcopy(DEFAULT_LOT_CONFIG) if lot_config is None else copy.deepcopy(lot_config)

        # LTM/Attention can be shared or independent
        self.long_term_memory = shared_long_term_memory if shared_long_term_memory is not None else {}
        self.shared_attention_foci = shared_attention_foci if shared_attention_foci is not None else collections.deque(maxlen=20)
        
        # Direct attributes, not from config files
        self.ltm_utility_weight_valence = 0.6
        self.ltm_utility_weight_efficiency = 0.4
        
        self.temporal_feedback_grid = collections.deque(maxlen=self.temporal_grid_params['max_len'])
        self.last_cycle_entropy_for_delta = 0.0

        self.smn_params_runtime_state = {}
        self.smn_param_indices = {}
        self.smn_param_names_from_indices = {}
        self.smn_influence_matrix = np.array([])
        self.smn_param_actual_changes_this_cycle = {}
        self._initialize_smn_graph_structures()
        self.smn_internal_flags = {}

        self.firewall_cooldown_remaining = 0
        self.firewall_cycles_since_last_check = 0

        self.current_goal_state_obj = None

        # ADV_REASONING_FEATURE_1: Initialize active concepts state
        self.active_concepts = set()

        self.working_memory = WorkingMemoryStack(max_depth=working_memory_max_depth)
        self.current_cycle_lot_stream = []
        
        self.post_goal_valence_lock_cycles_remaining = 0
        self.post_goal_valence_lock_value = 0.2
        self.post_goal_valence_lock_duration = 3


        if config_overrides:
            self._apply_config_overrides(config_overrides)

        if trainable_param_values:
            self.update_emulator_parameters(trainable_param_values)

        self.long_term_memory_capacity = 100
        # This will be overridden by the config override in main, but has a default here.
        self.successful_sequence_threshold_valence = 0.5 

        self.cycle_history = collections.deque(maxlen=cycle_history_max_len)
        self.current_cycle_num = 0
        self.next_target_input_state_handle = self.universe['start_state']

        if self.verbose >= 1:
            active_features_list = ["TemporalGrid", f"SMN(Graph:{self.smn_config.get('enable_influence_matrix', False)})", "Interrupts", "Firewall", "Goals", "LoT", "WorkingMemory"]
            print(f"[{self.agent_id}] Orch-OR Emulator Initialized. Universe: '{self.universe['name']}'. Active Features: {', '.join(active_features_list)}.")
            print(f"[{self.agent_id}] E_OR_THRESHOLD: {self.E_OR_THRESHOLD:.2f}, ORP Decay Rate: {self.orp_decay_rate:.3f}, WM Depth: {self.working_memory.stack.maxlen}")
            if self.temporal_grid_params.get('max_len',0) > 0:
                print(f"[{self.agent_id}] Temporal Feedback Grid: Active (maxlen={self.temporal_grid_params['max_len']}, window={self.temporal_grid_params['feedback_window']})")
            if self.smn_config.get('enabled', False) and self.smn_config.get('enable_influence_matrix', False):
                 print(f"[{self.agent_id}] SMN Influence Matrix: Active ({len(self.smn_param_indices)} params, matrix_shape {self.smn_influence_matrix.shape})")
    
    # ...[The ENTIRE rest of the SimplifiedOrchOREmulator class from the monolith, unmodified]...
    # [This section is elided for brevity, but all methods from _apply_config_overrides to 
    # run_chained_cognitive_cycles are migrated here verbatim without any changes.]
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

    # --- Working Memory Logging Helper (NEW) 
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
    
        # --- ADV_REASONING_FEATURE_1: Conceptual Layer - Activation ---
    def _executive_update_active_concepts(self, collapsed_logical_state_str):
        """Updates the agent's set of active concepts based on the collapsed state."""
        concept_map = self.internal_state_parameters.get('concept_logical_state_map', {})
        if not concept_map:
            if self.active_concepts: # Clear if concepts exist but map is now empty
                 self.active_concepts.clear()
            return

        # Determine if concepts should persist or be refreshed each cycle
        if self.internal_state_parameters.get('clear_active_concepts_each_cycle', True):
            self.active_concepts.clear()

        activated_this_cycle = set()
        for concept_name, state_pattern in concept_map.items():
            if state_pattern == collapsed_logical_state_str:
                self.active_concepts.add(concept_name)
                activated_this_cycle.add(concept_name)

        if activated_this_cycle:
            if self.verbose >= 2: print(f"  EXECUTIVE.ConceptUpdate: |{collapsed_logical_state_str}> activated concepts: {activated_this_cycle}")
            self._log_lot_event("executive.concept_update", {"state": collapsed_logical_state_str,
                                                            "activated": list(activated_this_cycle),
                                                            "all_active_now": list(self.active_concepts)})

    # --- Layer 1: Sensor Layer ---
    def _sensor_layer_process_input(self, target_conceptual_input: StateHandle) -> tuple[str, StateHandle]:
        if self.verbose >= 2: print(f"  SENSOR_LAYER: Processing target conceptual input '{target_conceptual_input}'.")
        self._log_lot_event("sensor.process_input.start", {"target_input_conceptual": str(target_conceptual_input)})

        target_classical_input_str = self.universe['state_to_comp_basis'].get(target_conceptual_input)
        if target_classical_input_str is None:
             if self.verbose >= 1: print(f"    SENSOR_LAYER: Warning - could not find conceptual input {target_conceptual_input} in universe map. Defaulting to basis '00'.")
             target_classical_input_str = self.universe['state_to_comp_basis'][self.universe['start_state']]


        noise_level = self.internal_state_parameters.get('sensor_input_noise_level', 0.0)
        actual_classical_input_str = target_classical_input_str
        actual_conceptual_state = target_conceptual_input # Default to no noise

        if noise_level > 0 and random.random() < 0.75:
            mutated_input_list = list(target_classical_input_str)
            num_flips = 0
            for i in range(len(mutated_input_list)):
                if random.random() < noise_level:
                    mutated_input_list[i] = '1' if mutated_input_list[i] == '0' else '0'
                    num_flips +=1

            if num_flips > 0:
                actual_classical_input_str = "".join(mutated_input_list)
                # The corresponding conceptual state
                actual_conceptual_state = self.universe['comp_basis_to_state'].get(actual_classical_input_str, self.universe['start_state'])
                if self.verbose >= 1: print(f"    SENSOR_LAYER: Input '{target_conceptual_input}' (basis |{target_classical_input_str}>) perceived as {actual_conceptual_state} (basis |{actual_classical_input_str}>) due to noise.")
                self._log_lot_event("sensor.process_input.noise_applied", {"original": str(target_conceptual_input), "original_comp_basis": target_classical_input_str, "actual_comp_basis": actual_classical_input_str, "actual_conceptual": str(actual_conceptual_state), "noise_level": noise_level, "flips":num_flips})

        self._log_lot_event("sensor.process_input.end", {"actual_input_computational": actual_classical_input_str, "actual_input_conceptual": str(actual_conceptual_state)})
        return actual_classical_input_str, actual_conceptual_state

    # --- Layer 2: Associative Layer ---
### NEW VERSION ###

    def _associative_layer_update_ltm(self, op_sequence, raw_valence, orp_cost, entropy_gen, final_collapsed_state_str, consolidation_factor=1.0,
                                      initial_state_when_sequence_started="unknown", input_context_when_sequence_started="unknown"):
        if self.verbose >= 2: print(f"  ASSOCIATIVE_LAYER.LTM_Update: Seq {op_sequence if op_sequence else 'NoOps'}, Val={raw_valence:.2f}, ORP={orp_cost:.2f}, Ent={entropy_gen:.2f}, ConsolFactor={consolidation_factor:.2f}")
        self._log_lot_event("associative.ltm_update.start", {
            "op_seq_len":len(op_sequence or []), "raw_valence":raw_valence, "orp_cost": orp_cost,
            "consol_factor": consolidation_factor, "entropy":entropy_gen,
            "initial_state_ctx": initial_state_when_sequence_started,
            "input_ctx": input_context_when_sequence_started,
            "outcome_state_ctx": final_collapsed_state_str,
            "active_concepts_for_store": list(self.active_concepts)
        })

        if not op_sequence: return
        seq_tuple = tuple(tuple(op) for op in op_sequence)

        if raw_valence < self.successful_sequence_threshold_valence * 0.3 and consolidation_factor <= 1.0:
             if self.verbose >=3: print(f"    LTM_Update: Sequence {seq_tuple} not stored, raw_valence {raw_valence:.2f} too low (threshold factor 0.3).")
             self._log_lot_event("associative.ltm_update.skip_low_valence", {"seq_tuple":seq_tuple, "raw_valence":raw_valence})
             return

        current_goal_name_for_ltm = None
        current_step_name_for_ltm = None
        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            goal = self.current_goal_state_obj
            active_goal_for_context = goal
            if 0 <= goal.current_step_index < len(goal.steps):
                step_hosting_subgoal = goal.steps[goal.current_step_index]
                potential_sub_goal = step_hosting_subgoal.get("sub_goal")
                if isinstance(potential_sub_goal, GoalState) and potential_sub_goal.status == "active":
                    active_goal_for_context = potential_sub_goal

            if 0 <= active_goal_for_context.current_step_index < len(active_goal_for_context.steps):
                current_goal_name_for_ltm = active_goal_for_context.current_goal
                current_step_name_for_ltm = active_goal_for_context.steps[active_goal_for_context.current_step_index].get("name", f"Step_{active_goal_for_context.current_step_index}")

        entry = self.long_term_memory.get(seq_tuple)
        mutation_rate_store = self.internal_state_parameters.get('ltm_mutation_on_store_rate', 0.0)
        update_strength = int(math.ceil(consolidation_factor))

        if entry:
            entry['count'] += update_strength
            entry['total_valence'] += raw_valence * consolidation_factor
            entry['total_orp_cost'] += orp_cost
            entry['total_entropy_generated'] += entropy_gen

            entry['last_goal_context_name'] = current_goal_name_for_ltm
            entry['last_goal_context_step'] = current_step_name_for_ltm
            if current_goal_name_for_ltm:
                context_key = (current_goal_name_for_ltm, current_step_name_for_ltm)
                entry['goal_context_counts'] = entry.get('goal_context_counts', {})
                entry['goal_context_counts'][context_key] = entry['goal_context_counts'].get(context_key, 0) + update_strength

            entry['final_outcome_states'] = entry.get('final_outcome_states', collections.Counter())
            entry['final_outcome_states'][final_collapsed_state_str] += update_strength

            if random.random() < 0.25 :
                entry['initial_states_seen'] = entry.get('initial_states_seen', collections.Counter())
                entry['initial_states_seen'][initial_state_when_sequence_started] += 1
                entry['input_contexts_seen'] = entry.get('input_contexts_seen', collections.Counter())
                entry['input_contexts_seen'][input_context_when_sequence_started] +=1

            entry['concepts_seen_counts'] = entry.get('concepts_seen_counts', collections.Counter())
            for concept in self.active_concepts:
                entry['concepts_seen_counts'][concept] += update_strength
            if entry['count'] % 5 == 0:
                 most_common_concepts_list = [c[0] for c in entry['concepts_seen_counts'].most_common(3)]
                 entry['most_frequent_concepts_at_store'] = most_common_concepts_list

            if random.random() < mutation_rate_store:
                entry['total_valence'] *= (1 + random.uniform(-0.05, 0.05) * update_strength)
                entry['total_orp_cost'] *= (1 + random.uniform(-0.03, 0.03))
                self._log_lot_event("associative.ltm_update.metric_mutation", {"seq":seq_tuple})

            entry['avg_valence'] = entry['total_valence'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_orp_cost'] = entry['total_orp_cost'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_entropy'] = entry['total_entropy_generated'] / entry['count'] if entry['count'] > 0 else 0

        else: # New LTM entry
            if len(self.long_term_memory) >= self.long_term_memory_capacity:
                if not self.long_term_memory: return
                # Pruning logic now considers confidence as a factor to AVOID pruning high-confidence items
                min_prune_score = float('inf'); key_to_prune = None
                for k, v_data in self.long_term_memory.items():
                    # Score = utility - confidence bonus. Lower score is worse.
                    prune_score = v_data.get('utility', 0) - v_data.get('confidence', 0) * 0.2
                    if prune_score < min_prune_score:
                        min_prune_score = prune_score
                        key_to_prune = k

                if key_to_prune:
                    if self.verbose >=3: print(f"    LTM_Update: LTM full. Pruning {key_to_prune} (prune_score {min_prune_score:.2f}).")
                    self._log_lot_event("associative.ltm_update.prune", {"pruned_seq_str":str(key_to_prune), "prune_score":min_prune_score})
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
                    'first_goal_context_name': current_goal_name_for_ltm, 'first_goal_context_step': current_step_name_for_ltm,
                    'last_goal_context_name': current_goal_name_for_ltm, 'last_goal_context_step': current_step_name_for_ltm,
                    'goal_context_counts': {},
                    'initial_states_seen': collections.Counter({initial_state_when_sequence_started: update_strength}),
                    'input_contexts_seen': collections.Counter({input_context_when_sequence_started: update_strength}),
                    'most_frequent_initial_state': initial_state_when_sequence_started,
                    'most_frequent_input_context': input_context_when_sequence_started,
                    'final_outcome_states': collections.Counter({final_collapsed_state_str: update_strength}),
                    'most_frequent_outcome_state': final_collapsed_state_str,
                    'concepts_seen_counts': collections.Counter(self.active_concepts),
                    'most_frequent_concepts_at_store': list(self.active_concepts),
                    ### NEW: Initialize Epistemic/Confidence fields ###
                    'confidence': 0.0, # Will be calculated after creation
                    'last_successful_use_cycle': self.current_cycle_num if raw_valence > 0 else -1
                }
                if current_goal_name_for_ltm:
                     context_key_new = (current_goal_name_for_ltm, current_step_name_for_ltm)
                     new_entry['goal_context_counts'][context_key_new] = update_strength

                self.long_term_memory[seq_tuple] = new_entry
                log_extra = { "goal_name_ctx":current_goal_name_for_ltm or "N/A", "initial_state_ctx_new":initial_state_when_sequence_started, "input_ctx_new": input_context_when_sequence_started, "active_concepts": list(self.active_concepts) }
                if self.verbose >=3: print(f"    LTM_Update: Added new sequence {seq_tuple} with avg_valence {new_entry['avg_valence']:.2f}, Contexts: {log_extra}.")
                self._log_lot_event("associative.ltm_update.new_entry", {"seq_str":str(seq_tuple), "val":new_entry['avg_valence'], **log_extra})

        if seq_tuple in self.long_term_memory:
             entry_ref = self.long_term_memory[seq_tuple]
             entry_ref['utility'] = self._associative_layer_calculate_ltm_entry_utility(entry_ref)
             entry_ref['last_cycle'] = self.current_cycle_num
             ### NEW: CONFIDENCE UPDATE ###
             # Confidence is a mix of familiarity (count) and reliability (positive valence).
             # Tanh provides a nice curve, capping familiarity bonus. Clipped valence ensures negative outcomes don't help confidence.
             familiarity_score = np.tanh(entry_ref['count'] / 20.0) # Saturates around 20 uses
             reliability_score = np.clip(entry_ref['avg_valence'], 0, 1.0)
             entry_ref['confidence'] = familiarity_score * reliability_score
             if raw_valence > 0:
                 entry_ref['last_successful_use_cycle'] = self.current_cycle_num

             if entry_ref['count'] > 5 and random.random() < 0.1 :
                 if entry_ref.get('initial_states_seen'):
                     entry_ref['most_frequent_initial_state'] = entry_ref['initial_states_seen'].most_common(1)[0][0]
                 if entry_ref.get('input_contexts_seen'):
                     entry_ref['most_frequent_input_context'] = entry_ref['input_contexts_seen'].most_common(1)[0][0]
                 if entry_ref.get('final_outcome_states'):
                     entry_ref['most_frequent_outcome_state'] = entry_ref['final_outcome_states'].most_common(1)[0][0]

    def _associative_layer_calculate_ltm_entry_utility(self, seq_data):
        norm_orp_cost = seq_data['avg_orp_cost'] / (self.E_OR_THRESHOLD + 1e-6)
        utility = (self.ltm_utility_weight_valence * seq_data['avg_valence'] -
                   self.ltm_utility_weight_efficiency * norm_orp_cost +
                   0.05 * seq_data.get('avg_entropy', 0.0))
        return utility

### NEW VERSION ###
    def _associative_layer_recall_from_ltm_strategy(self, current_orp_value, exec_thought_log,
                                                     current_collapsed_state_for_recall_context,
                                                     current_input_context_for_recall):
        if not self.long_term_memory:
            exec_thought_log.append("LTM recall: LTM empty.")
            return None, current_orp_value

        min_utility_for_recall = 0.05 # Base utility threshold
        candidate_info = [] # Store (sequence_ops_list, effective_utility, applied_bonuses_dict, original_ltm_data_dict)

        active_recall_goal_name = None
        active_recall_step_name = None

        # Determine active goal for context bonus
        # Traverse up from sub-goals if necessary to get the highest relevant goal context.
        # For simplicity, we use the immediate self.current_goal_state_obj and its active step.
        current_processing_goal_obj = self.current_goal_state_obj
        # Potentially refine to check if a sub-goal is the one currently being executed, for more precise context
        # but for now, the 'primary' goal's step provides the top-level context.
        if current_processing_goal_obj and current_processing_goal_obj.status == "active":
            temp_goal_for_context = current_processing_goal_obj
            # Check if current step of 'temp_goal_for_context' is an active sub-goal, and if so, use ITS step.
            if 0 <= temp_goal_for_context.current_step_index < len(temp_goal_for_context.steps):
                potential_sub_goal_in_step = temp_goal_for_context.steps[temp_goal_for_context.current_step_index].get("sub_goal")
                if isinstance(potential_sub_goal_in_step, GoalState) and potential_sub_goal_in_step.status == "active":
                    temp_goal_for_context = potential_sub_goal_in_step # Use the active sub-goal's context

            # Now, get step name from the 'temp_goal_for_context' (which might be the original or a sub-goal)
            if 0 <= temp_goal_for_context.current_step_index < len(temp_goal_for_context.steps):
                active_recall_goal_name = temp_goal_for_context.current_goal
                active_recall_step_name = temp_goal_for_context.steps[temp_goal_for_context.current_step_index].get("name", f"Step_{temp_goal_for_context.current_step_index}")


        # Retrieve bonus parameters
        goal_ctx_bonus_val = self.internal_state_parameters.get('ltm_goal_context_match_bonus', 0.15)
        initial_state_bonus_val = self.internal_state_parameters.get('ltm_initial_state_match_bonus', 0.10)
        input_ctx_bonus_val = self.internal_state_parameters.get('ltm_input_context_match_bonus', 0.05)
        concept_match_bonus_val = self.internal_state_parameters.get('ltm_active_concept_match_bonus', 0.12) # ADV_REASONING_FEATURE_1

        for seq_tuple, data in self.long_term_memory.items():
            base_utility = data.get('utility', self._associative_layer_calculate_ltm_entry_utility(data))
            current_effective_utility = base_utility
            applied_bonuses_detail = {'goal': 0.0, 'initial_state': 0.0, 'input_ctx': 0.0, 'concept': 0.0}

            # Goal Context Bonus
            if active_recall_goal_name and data.get('last_goal_context_name') == active_recall_goal_name:
                bonus_val_for_goal_match = goal_ctx_bonus_val * 0.5 # Default for just goal name match
                if data.get('last_goal_context_step') == active_recall_step_name:
                    bonus_val_for_goal_match = goal_ctx_bonus_val # Full bonus for specific step match
                current_effective_utility += bonus_val_for_goal_match
                applied_bonuses_detail['goal'] = bonus_val_for_goal_match

            # Initial State Context Bonus
            if data.get('most_frequent_initial_state') == current_collapsed_state_for_recall_context:
                current_effective_utility += initial_state_bonus_val
                applied_bonuses_detail['initial_state'] = initial_state_bonus_val

            # Input Context Bonus
            if data.get('most_frequent_input_context') == current_input_context_for_recall:
                current_effective_utility += input_ctx_bonus_val
                applied_bonuses_detail['input_ctx'] = input_ctx_bonus_val

            # ADV_REASONING_FEATURE_1: Active Concept Match Bonus
            if concept_match_bonus_val > 0 and self.active_concepts:
                # Use concepts seen most frequently with this sequence from LTM
                stored_concepts_list = data.get('most_frequent_concepts_at_store', [])
                if stored_concepts_list:
                    stored_concepts_set = set(stored_concepts_list)
                    intersection_size = len(self.active_concepts.intersection(stored_concepts_set))
                    union_size = len(self.active_concepts.union(stored_concepts_set))

                    if union_size > 0:
                        jaccard_similarity = intersection_size / union_size
                        concept_bonus = jaccard_similarity * concept_match_bonus_val
                        current_effective_utility += concept_bonus
                        applied_bonuses_detail['concept'] = concept_bonus


            if current_effective_utility > min_utility_for_recall:
                projected_cost = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in seq_tuple)
                # Max ORP threshold consideration: allow recall if it doesn't immediately exceed, or get too close to, threshold.
                if current_orp_value + projected_cost < self.E_OR_THRESHOLD * 1.15:
                    candidate_info.append( (list(seq_tuple), current_effective_utility, applied_bonuses_detail, data) )

        if not candidate_info:
            exec_thought_log.append(f"LTM recall: No sequences found with effective_utility > {min_utility_for_recall} (after all context bonuses) or all too costly from ORP {current_orp_value:.2f}.")
            return None, current_orp_value

        # Select from candidates based on effective utility
        candidate_sequences = [c[0] for c in candidate_info]
        # Weight by the *effective* utility - squared to emphasize higher utility items more
        weights = [c[1]**2.5 for c in candidate_info]

        sum_weights = sum(weights)
        if sum_weights <= 1e-6: # Should be rare if candidate_info is not empty and utilities are positive
             exec_thought_log.append("LTM recall: No LTM sequences with positive utility weights after filtering (or all weights zero).")
             return None, current_orp_value

        normalized_weights = [w / sum_weights for w in weights]
        try:
            chosen_index = random.choices(range(len(candidate_sequences)), weights=normalized_weights, k=1)[0]
        except ValueError as e: # Catch empty or invalid weights
            exec_thought_log.append(f"LTM recall: Error in weighted choice ({e}). Defaulting to highest utility if possible, or skip.")
            if candidate_info: # Try to pick the one with max effective utility as a fallback
                chosen_index = max(range(len(candidate_info)), key=lambda i: candidate_info[i][1])
            else: return None, current_orp_value

        chosen_sequence_ops_orig_list = candidate_sequences[chosen_index]
        # Create a mutable copy for potential mutation
        chosen_sequence_ops_mutable = [list(op) for op in chosen_sequence_ops_orig_list]

        bonuses_applied_for_chosen = candidate_info[chosen_index][2]
        original_ltm_data_for_chosen = candidate_info[chosen_index][3]

        # LTM Mutation on Replay
        mutation_rate_replay = self.internal_state_parameters.get('ltm_mutation_on_replay_rate', 0.0)
        if chosen_sequence_ops_mutable and random.random() < mutation_rate_replay and len(chosen_sequence_ops_mutable) > 0:
            idx_to_mutate = random.randrange(len(chosen_sequence_ops_mutable))
            op_char, op_arg = chosen_sequence_ops_mutable[idx_to_mutate]
            original_op_tuple_str_in_seq = f"('{op_char}', {op_arg})"

            mutation_type_rand = random.random()
            if mutation_type_rand < 0.35 and op_char in ['X', 'Z', 'H'] and isinstance(op_arg, int): # Flip arg for single qubit
                chosen_sequence_ops_mutable[idx_to_mutate][1] = 1 - op_arg
            elif mutation_type_rand < 0.65: # Change op type
                compatible_ops={'X':['Z','H'],'Z':['X','H'],'H':['X','Z'],'CNOT':['CZ'],'CZ':['CNOT']}
                new_op_char = random.choice(compatible_ops.get(op_char, ['X','Z','H']))
                new_op_arg = op_arg
                if new_op_char in ['X','Z','H']: new_op_arg = random.randint(0,1)
                elif new_op_char in ['CNOT', 'CZ']:
                    new_op_arg = tuple(random.sample([0,1],2)) if op_char not in ['CNOT','CZ'] else op_arg
                chosen_sequence_ops_mutable[idx_to_mutate] = [new_op_char, new_op_arg]
            elif len(chosen_sequence_ops_mutable) > 1 and random.random() < 0.5 : # Delete an op
                del chosen_sequence_ops_mutable[idx_to_mutate]
            else: # Insert a new random op
                new_op_insert_char = random.choice(['X','Z','H'])
                new_op_insert_arg = random.randint(0,1)
                chosen_sequence_ops_mutable.insert(random.randint(0, len(chosen_sequence_ops_mutable)), [new_op_insert_char, new_op_insert_arg])

            exec_thought_log.append(f"LTM Replay MUTATION: Op {original_op_tuple_str_in_seq} in original seq {chosen_sequence_ops_orig_list} -> seq potentially modified to {chosen_sequence_ops_mutable}.")
            self._log_lot_event("associative.ltm_recall.mutation", {"original_seq_str": str(chosen_sequence_ops_orig_list), "mutated_seq_str": str(chosen_sequence_ops_mutable)})

        # Final check on cost for the (potentially mutated) chosen sequence
        projected_orp_increase_final = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in chosen_sequence_ops_mutable)
        if current_orp_value + projected_orp_increase_final >= self.E_OR_THRESHOLD * 1.1 and len(chosen_sequence_ops_mutable) > 0 :
            exec_thought_log.append(f"LTM recall: Mutated/Chosen seq {chosen_sequence_ops_mutable} too costly. ORP would be {current_orp_value + projected_orp_increase_final:.2f}. Skipped.")
            return None, current_orp_value

        final_chosen_ops_as_tuples = [tuple(op) for op in chosen_sequence_ops_mutable]

        bonus_summary_str = f"GoalCtx:{bonuses_applied_for_chosen['goal']:.2f},StateCtx:{bonuses_applied_for_chosen['initial_state']:.2f},InputCtx:{bonuses_applied_for_chosen['input_ctx']:.2f},ConceptCtx:{bonuses_applied_for_chosen['concept']:.2f}"
        exec_thought_log.append(f"LTM recall: Replaying {final_chosen_ops_as_tuples} (orig_avg_V={original_ltm_data_for_chosen['avg_valence']:.2f}, base_util={original_ltm_data_for_chosen['utility']:.2f}, bonuses_sum={sum(bonuses_applied_for_chosen.values()):.2f} [{bonus_summary_str}]). Cost {projected_orp_increase_final:.2f}")
        self._log_lot_event("associative.ltm_recall.chosen", {
            "seq_str":str(final_chosen_ops_as_tuples),
            "orig_util":original_ltm_data_for_chosen['utility'],
            "applied_bonuses_sum": sum(bonuses_applied_for_chosen.values()),
            "bonuses_detail_str": bonus_summary_str,
            "current_state_ctx_match_val": current_collapsed_state_for_recall_context,
            "current_input_ctx_match_val": current_input_context_for_recall,
            "goal_context_name_at_recall": active_recall_goal_name or "N/A",
            "goal_context_step_at_recall": active_recall_step_name or "N/A",
            "active_concepts_at_recall": list(self.active_concepts)
            })
        return final_chosen_ops_as_tuples, current_orp_value


    # --- Layer 3: Executive Layer (Decision Making, Planning, Conscious Experience) ---
    def _executive_evaluate_outcome_and_update_mood(self, logical_outcome_str, orp_at_collapse, entropy_at_collapse, num_ops_executed_this_cycle):
        if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.Outcome_Eval: |{logical_outcome_str}>, ORP={orp_at_collapse:.3f}, Ent={entropy_at_collapse:.2f}, Ops#={num_ops_executed_this_cycle}")
        acc_thoughts_log = []

        raw_valence = self.universe['valence_map'].get(logical_outcome_str, -0.15)
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


    def _executive_generate_computation_sequence(self, ops_provided_externally=None):
        if ops_provided_externally is not None:
            if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.OpGen: Using externally provided ops: {ops_provided_externally}")
            self._log_lot_event("executive.opgen.external", {"ops_count": len(ops_provided_externally)})
            return ops_provided_externally, "StrategyProvidedExternal", ["Ops provided externally."]

        exec_thought_log = ["OpGen: Generating new computation sequence:"]
        self._log_lot_event("executive.opgen.start", {"orp_current":self.objective_reduction_potential, "threshold": self.E_OR_THRESHOLD})

        ops_sequence = []
        chosen_strategy_name = "NoOpsMethod"
        simulated_orp_accumulator = self.objective_reduction_potential

        # --- Check Working Memory for Intermediate Results or Ops Sequences ---
        if not self.working_memory.is_empty():
            wm_top_item = self.working_memory.peek()
            self._log_wm_op("peek", item=wm_top_item, details={'reason':'opgen_start_check'})
            if wm_top_item and wm_top_item.type == "intermediate_result":
                wm_data = wm_top_item.data
                exec_thought_log.append(f"  WM_IntermediateResult FOUND: '{wm_top_item.description[:50]}...'")
                if "next_planned_ops" in wm_data and isinstance(wm_data["next_planned_ops"], list) and wm_data["next_planned_ops"]:
                    candidate_wm_ops = [list(op) for op in wm_data["next_planned_ops"]]
                    projected_wm_ops_cost = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in candidate_wm_ops)
                    if self.objective_reduction_potential + projected_wm_ops_cost < self.E_OR_THRESHOLD * 0.98:
                        ops_sequence = candidate_wm_ops
                        chosen_strategy_name = "StrategyWMIntermediateOps"
                        exec_thought_log.append(f"    Using planned ops sequence from WM: {ops_sequence}. Cost: {projected_wm_ops_cost:.2f}")
                        if wm_data.get("consume_after_use", True):
                            popped_item = self.working_memory.pop()
                            self._log_wm_op("pop_intermediate", item=popped_item, details={'reason':'intermediate_ops_used'})
                        self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy": chosen_strategy_name})
                        return ops_sequence, chosen_strategy_name, exec_thought_log
                    else:
                        exec_thought_log.append(f"    WM Intermediate ops too costly (cost {projected_wm_ops_cost:.2f}).")
        
        # --- [START of merged logic] ---
        
        effective_attention = self.internal_state_parameters['attention_level']
        cognitive_load_factor = 1.0 - (self.internal_state_parameters['cognitive_load'] * 0.65)
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * cognitive_load_factor * effective_attention, scale=1.0)))
        num_ops_target = min(num_ops_target, 10) # Max ops cap
        exec_thought_log.append(f"  Target ops: ~{num_ops_target} (base:{num_ops_target_base}, load_factor:{cognitive_load_factor:.2f}, attn:{effective_attention:.2f}). ORP start: {self.objective_reduction_potential:.3f}")
        
        current_strategy_weights = self.internal_state_parameters['strategy_weights'].copy()
        
        # --- NEW, SUPERIOR GOAL/WM CONTEXT LOGIC (FROM UNREACHABLE BLOCK) ---
        active_goal_step_info = None
        active_goal_step_name = "None"
        current_processing_goal = None
        is_goal_context_from_wm = False
        ops_from_goal_hint = None

        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            current_processing_goal = self.current_goal_state_obj
            temp_goal = current_processing_goal
            # Traverse down to the deepest active sub-goal for context
            while isinstance(temp_goal, GoalState) and temp_goal.status == "active" and 0 <= temp_goal.current_step_index < len(temp_goal.steps):
                step_obj = temp_goal.steps[temp_goal.current_step_index]
                if isinstance(step_obj.get("sub_goal"), GoalState) and step_obj["sub_goal"].status == "active":
                    current_processing_goal = step_obj["sub_goal"]
                    temp_goal = current_processing_goal
                else:
                    break

            if 0 <= current_processing_goal.current_step_index < len(current_processing_goal.steps):
                active_goal_step_info = current_processing_goal.steps[current_processing_goal.current_step_index]
                active_goal_step_name = active_goal_step_info.get('name', f'Step{current_processing_goal.current_step_index}')

                # Detailed WM context check
                if not self.working_memory.is_empty():
                    wm_top_item_for_goal_ctx = self.working_memory.peek()
                    if wm_top_item_for_goal_ctx and wm_top_item_for_goal_ctx.type == "goal_step_context" and \
                       wm_top_item_for_goal_ctx.data.get("goal_name") == current_processing_goal.current_goal and \
                       wm_top_item_for_goal_ctx.data.get("goal_step_name") == active_goal_step_name and \
                       wm_top_item_for_goal_ctx.data.get("step_index") == current_processing_goal.current_step_index:
                        is_goal_context_from_wm = True
                        exec_thought_log.append(f"  WM Active GoalContext: Matched Goal '{current_processing_goal.current_goal}' - Step '{active_goal_step_name}'.")
                        self._log_lot_event("executive.opgen.wm_match_goal", {"goal":current_processing_goal.current_goal, "step":active_goal_step_name})

        # --- NEW, SUPERIOR STRATEGY INFLUENCE LOGIC (FROM UNREACHABLE BLOCK) ---
        if active_goal_step_info:
            exec_thought_log.append(f"  Goal Active ('{current_processing_goal.current_goal}::{active_goal_step_name}', WM_Ctx: {is_goal_context_from_wm}): Applying influence.")
            step_target_state = active_goal_step_info.get("target_state")
            if step_target_state and self.internal_state_parameters['preferred_logical_state'] != step_target_state:
                self.internal_state_parameters['preferred_logical_state'] = step_target_state
                exec_thought_log.append(f"    Goal ('{active_goal_step_name}') mandates preferred_state to |{step_target_state}>.")
            
            # Nuanced boosts
            goal_seek_boost = 0.35 if is_goal_context_from_wm else 0.25 
            current_strategy_weights['goal_seek'] = min(1.0, current_strategy_weights.get('goal_seek',0.1) * (1 + goal_seek_boost) + goal_seek_boost)
            current_strategy_weights['problem_solve'] = min(1.0, current_strategy_weights.get('problem_solve',0.1) * (1.2 + (0.2 * is_goal_context_from_wm)) + (0.05 + 0.05*is_goal_context_from_wm) )
            exec_thought_log.append(f"    Goal ('{active_goal_step_name}') boosts goal_seek (~{goal_seek_boost*100:.0f}%) & problem_solve.")

            # Probabilistic hint usage
            ops_hint_from_step = active_goal_step_info.get("next_ops_hint")
            if ops_hint_from_step and isinstance(ops_hint_from_step, list) and ops_hint_from_step:
                use_hint_roll = random.random()
                use_hint_threshold = 0.75 if is_goal_context_from_wm else 0.55
                if use_hint_roll < use_hint_threshold:
                    projected_hint_cost = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in ops_hint_from_step)
                    if self.objective_reduction_potential + projected_hint_cost < self.E_OR_THRESHOLD * 0.95:
                        ops_from_goal_hint = [list(op) for op in ops_hint_from_step]
        
        if ops_from_goal_hint:
            ops_sequence = ops_from_goal_hint
            chosen_strategy_name = f"StrategyGoalStepHint({active_goal_step_name})"
            exec_thought_log.append(f"  OpGen Result: Using ops sequence from goal hint: {ops_sequence}")
            self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy":chosen_strategy_name, "final_sim_orp":"N/A_HintUsed"})
            return ops_sequence, chosen_strategy_name, exec_thought_log

        # --- TEMPORAL FEEDBACK GRID AND OTHER BIAS LOGIC ---
        tfg_window = self.temporal_grid_params['feedback_window']
        grid_entries_to_consider = list(self.temporal_feedback_grid)[-tfg_window:]
        if grid_entries_to_consider:
            recent_valence_deltas = [g[1] for g in grid_entries_to_consider if g[1] is not None]
            recent_entropy_shifts = [g[2] for g in grid_entries_to_consider if g[2] is not None]
            avg_recent_valence_delta = np.mean(recent_valence_deltas) if recent_valence_deltas else 0.0
            avg_recent_entropy_shift = np.mean(recent_entropy_shifts) if recent_entropy_shifts else 0.0
            
            if avg_recent_valence_delta < self.temporal_grid_params['low_valence_delta_threshold']:
                #... bias logic ...
                pass
            if avg_recent_entropy_shift > self.temporal_grid_params['high_entropy_shift_threshold']:
                #... bias logic ...
                pass

        # Exploration mode and interrupt flags
        if self.internal_state_parameters['exploration_mode_countdown'] > 0:
            current_strategy_weights['curiosity'] *= 2.8
            current_strategy_weights['goal_seek'] *= 0.3   
        if self.smn_internal_flags.get('force_ltm_reactive_op_next_cycle', False):
            current_strategy_weights = {'memory': 1.0, 'problem_solve': 0.001, 'goal_seek': 0.001, 'curiosity': 0.001}
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = False

        # --- STRATEGY SELECTION (Normalization and Choice) ---
        for key in DEFAULT_INTERNAL_PARAMS['strategy_weights']: 
            if key not in current_strategy_weights: current_strategy_weights[key] = 0.001 
        valid_weights = {k:v for k,v in current_strategy_weights.items() if isinstance(v,(int,float))}
        total_weight = sum(w for w in valid_weights.values() if w > 0)
        if total_weight <= 1e-6: # Fallback
            strategy_choices = ['curiosity']; strategy_probs = [1.0]
        else:
            strategy_choices, strategy_probs = zip(*[(k, v/total_weight) for k, v in valid_weights.items()])
        selected_strategy = random.choices(strategy_choices, weights=strategy_probs, k=1)[0]
        exec_thought_log.append(f"  Selected primary strategy: {selected_strategy}")
        
        # --- OP GENERATION based on STRATEGY ---
        was_novel_sequence = False
        if selected_strategy == 'memory':
            replay_ops, _ = self._associative_layer_recall_from_ltm_strategy(
                simulated_orp_accumulator, exec_thought_log, 
                self.current_conceptual_state, self.next_target_input_state_handle
            )
            if replay_ops:
                ops_sequence = replay_ops
                chosen_strategy_name = "StrategyLTMReplay"

        if not ops_sequence and selected_strategy == 'problem_solve':
            # This is simplified. Your logic can be more complex.
            pref_state_handle = self.internal_state_parameters.get('preferred_state_handle')
            
            # Use the conceptual handle, then find its string representation
            if pref_state_handle and pref_state_handle != self.current_conceptual_state:
                
                pref_state_str = self.universe['state_to_comp_basis'].get(pref_state_handle)
                current_state_str = self.collapsed_computational_state_str

                if pref_state_str is not None:
                    exec_thought_log.append(f"  ProblemSolving towards {pref_state_handle} (basis |{pref_state_str}>).")
                    
                    # Ensure both are valid 2-bit strings before proceeding
                    if len(pref_state_str) == 2 and len(current_state_str) == 2:
                        target_bits = list(map(int, pref_state_str))
                        current_bits = list(map(int, current_state_str))
                        planned_ops = []
                        
                        # Correct Logic: bit 0 maps to qubit 1, bit 1 maps to qubit 0
                        # Example: state '10' is |q1, q0>
                        # If current bit at index 0 ('1' in '10') != target bit at index 0
                        # then we need to flip qubit 1.
                        if current_bits[0] != target_bits[0]:
                            planned_ops.append(['X', 1])
                        # If current bit at index 1 ('0' in '10') != target bit at index 1
                        # then we need to flip qubit 0.
                        if current_bits[1] != target_bits[1]:
                            planned_ops.append(['X', 0])
                            
                        ops_sequence = planned_ops
                        chosen_strategy_name = "StrategyProblemSolving"
                        was_novel_sequence = True

        if not ops_sequence: # Fallback to goal-seek or curiosity loop
            # Your original op-generation loop remains a good fallback.
            exec_thought_log.append(f"  Using Fallback op generation loop ({selected_strategy}).")
            chosen_strategy_name = f"StrategyFallbackLoop_{selected_strategy}"
            was_novel_sequence = True # Any sequence generated this way is "novel"
            for op_count in range(num_ops_target):
                op_c = random.choice(['X', 'Z', 'H', 'CNOT', 'CZ'])
                op_a = random.randint(0,1) if op_c in ['X','Z','H'] else tuple(random.sample([0,1],2))
                op_cost = self.operation_costs.get(op_c.upper(), 0.05)
                if simulated_orp_accumulator + op_cost < self.E_OR_THRESHOLD * 0.98:
                    ops_sequence.append([op_c, op_a])
                    simulated_orp_accumulator += op_cost
                else:
                    break
        
        # --- ADVANCED PLANNING & REASONING (FROM LIVE BLOCK) AS A FALLBACK ---
        if not ops_sequence and current_processing_goal and active_goal_step_info:
            exec_thought_log.append("  Standard strategies failed. Engaging advanced planning...")
            advanced_planning_ops = None
            adv_strategy_name = "NoAdvPlan"
            
            # 1. Try Analogical Planning first
            if self.internal_state_parameters.get('enable_analogical_planning', True):
                advanced_planning_ops = self._advanced_planning_find_analogous_solution(
                    active_goal_step_info, self.collapsed_computational_state_str, exec_thought_log
                )
                if advanced_planning_ops:
                    adv_strategy_name = "StrategyAnalogicalPlanning"
            
            # 2. If that fails, try Hierarchical Breakdown
            if not advanced_planning_ops and self.internal_state_parameters.get('enable_hierarchical_planning', True):
                breakdown_succeeded = self._advanced_planning_breakdown_goal_hierarchically(
                    current_processing_goal, current_processing_goal.current_step_index, exec_thought_log
                )
                if breakdown_succeeded:
                    adv_strategy_name = "StrategyHierarchicalBreakdown"
                    ops_sequence = [] # The breakdown is the action; generate no ops this cycle.
            
            if advanced_planning_ops:
                ops_sequence = advanced_planning_ops
                was_novel_sequence = True
            
            if adv_strategy_name != "NoAdvPlan":
                chosen_strategy_name = adv_strategy_name

        # --- COUNTERFACTUAL SIMULATION on novel plans ---
        if ops_sequence and was_novel_sequence and self.internal_state_parameters.get('enable_counterfactual_simulation', True):
            sim_reject_thresh = self.internal_state_parameters.get('counterfactual_sim_reject_threshold', -0.1)
            sim_result = self._reasoning_simulate_counterfactual(ops_sequence, exec_thought_log)

            if sim_result['is_valid'] and sim_result['estimated_valence'] < sim_reject_thresh:
                exec_thought_log.append(f"    CounterfactualSim REJECTED plan. Est. valence {sim_result['estimated_valence']:.2f} < {sim_reject_thresh}. Wiping ops.")
                ops_sequence = []
                chosen_strategy_name = chosen_strategy_name + "_RejectedBySim"
            else:
                 chosen_strategy_name = chosen_strategy_name + "_VerifiedBySim"

        # --- FINAL CLEANUP AND RETURN ---
        if not ops_sequence:
            # Avoid the vague "NoOpsMethod"
            chosen_strategy_name = "NoOpsGenerated" if chosen_strategy_name == "NoOpsMethod" else chosen_strategy_name
        
        self._log_lot_event("executive.opgen.end", {"ops_generated_count": len(ops_sequence), "strategy": chosen_strategy_name})
        return ops_sequence, chosen_strategy_name, exec_thought_log

    # ------------------------------------------------------------------------------------------
    # --- Advanced Reasoning & Planning Engine (Hierarchical, Analogical, Counterfactual) ---
    # ------------------------------------------------------------------------------------------

    def _reasoning_simulate_counterfactual(self, ops_sequence_to_test, exec_thought_log):
        """
        Internally simulates a sequence of operations to estimate its outcome without full execution.
        This represents a form of "what-if" thinking or imagination.

        Args:
            ops_sequence_to_test (list): The list of operations [('OP', arg), ...] to simulate.
            exec_thought_log (list): The log to append thoughts to.

        Returns:
            dict: A dictionary containing {'estimated_valence': float, 'estimated_orp': float, 'is_valid': bool}
        """
        if not ops_sequence_to_test:
            return {'estimated_valence': 0.0, 'estimated_orp': self.objective_reduction_potential, 'is_valid': False}

        if self.verbose >= 2: exec_thought_log.append(f"  CounterfactualSim: Testing sequence {ops_sequence_to_test}")
        self._log_lot_event("reasoning.counterfactual_sim.start", {"ops_count": len(ops_sequence_to_test), "start_orp": self.objective_reduction_potential})

        try:
            sim_superposition = copy.deepcopy(self.logical_superposition)
            sim_orp = self.objective_reduction_potential

            for op_char, op_arg in ops_sequence_to_test:
                # Use a simplified cost accumulation for speed, as we are not checking for early OR.
                sim_superposition, sim_orp = self._apply_logical_op_to_superposition(op_char, op_arg, sim_superposition, sim_orp)

            # After applying all ops, estimate the expected valence from the resulting superposition
            probabilities = {state: abs(amp)**2 for state, amp in sim_superposition.items()}
            # Normalize probabilities just in case of float inaccuracies
            prob_sum = sum(probabilities.values())
            if prob_sum > 1e-9:
                                estimated_valence = sum(self.universe['valence_map'].get(self.universe['comp_basis_to_state'].get(state), 0.0) * (prob / prob_sum) for state, prob in probabilities.items())
            else:
                estimated_valence = -1.0 # Invalid superposition state

            exec_thought_log.append(f"    CounterfactualSim Result: Est. Valence={estimated_valence:.3f}, Est. ORP={sim_orp:.3f}")
            self._log_lot_event("reasoning.counterfactual_sim.result", {"est_valence": estimated_valence, "est_orp": sim_orp})
            return {'estimated_valence': estimated_valence, 'estimated_orp': sim_orp, 'is_valid': True}

        except Exception as e:
            exec_thought_log.append(f"    CounterfactualSim ERROR: {e}")
            self._log_lot_event("reasoning.counterfactual_sim.error", {"error_str": str(e)})
            return {'estimated_valence': -1.0, 'estimated_orp': self.objective_reduction_potential, 'is_valid': False}


    def _advanced_planning_find_analogous_solution(self, current_goal_step, current_state, exec_thought_log):
        """
        Searches LTM for structurally similar, previously successful solutions to adapt for the current goal.

        Args:
            current_goal_step (dict): The current goal step dictionary.
            current_state (str): The agent's current collapsed logical state string.
            exec_thought_log (list): The log to append thoughts to.

        Returns:
            list or None: A list of operations if an analogous solution is found, otherwise None.
        """
        target_state = current_goal_step.get("target_state")
        if not target_state or not self.long_term_memory:
            return None

        self._log_lot_event("reasoning.analogical_planning.start", {"current_state": current_state, "target_state": target_state, "goal_step": current_goal_step.get('name')})
        exec_thought_log.append(f"  AnalogicalPlanning: Searching LTM for path |{current_state}> -> |{target_state}>")

        candidates = []
        for seq_tuple, data in self.long_term_memory.items():
            if not data.get('most_frequent_initial_state') or not data.get('most_frequent_outcome_state'):
                continue

            # Hamming distance for similarity (for 2-bit system)
            initial_dist = sum(c1 != c2 for c1, c2 in zip(current_state, data['most_frequent_initial_state']))
            outcome_dist = sum(c1 != c2 for c1, c2 in zip(target_state, data['most_frequent_outcome_state']))

            # Normalize distance (max distance is 2 for 2-bit strings)
            initial_similarity = 1.0 - (initial_dist / 2.0)
            outcome_similarity = 1.0 - (outcome_dist / 2.0)

            # Score = combination of similarity and proven utility
            similarity_score = (initial_similarity * 0.4 + outcome_similarity * 0.6)
            final_score = similarity_score * data.get('utility', 0.0)

            if final_score > self.internal_state_parameters['analogical_planning_similarity_threshold']:
                 candidates.append({'seq': list(seq_tuple), 'score': final_score, 'data': data})

        if not candidates:
            exec_thought_log.append("    AnalogicalPlanning: No suitable analogous sequences found in LTM.")
            self._log_lot_event("reasoning.analogical_planning.fail", {"reason": "no_candidates_above_threshold"})
            return None

        # Select best candidate
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_analogous_solution = candidates[0]
        chosen_seq = best_analogous_solution['seq']
        projected_cost = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in chosen_seq)

        if self.objective_reduction_potential + projected_cost >= self.E_OR_THRESHOLD:
            exec_thought_log.append(f"    AnalogicalPlanning: Best candidate seq {chosen_seq} too costly (cost: {projected_cost:.2f}). Skipped.")
            self._log_lot_event("reasoning.analogical_planning.fail", {"reason": "best_candidate_too_costly", "cost": projected_cost})
            return None

        exec_thought_log.append(f"    AnalogicalPlanning: Found analogous seq {chosen_seq} with score {best_analogous_solution['score']:.3f}. Applying it.")
        self._log_lot_event("reasoning.analogical_planning.success", {"chosen_seq": str(chosen_seq), "score": best_analogous_solution['score']})

        # Future enhancement: adapt the sequence here instead of just replaying it.
        # For now, we will just return the sequence as-is.
        return chosen_seq


    def _advanced_planning_breakdown_goal_hierarchically(self, parent_goal, parent_step_idx, exec_thought_log):
        """
        Dynamically inserts a sub-goal into the current goal plan to reach a landmark state first.
        """
        if not (0 <= parent_step_idx < len(parent_goal.steps)):
            return False

        current_step_obj = parent_goal.steps[parent_step_idx]
        
        ### FIX: Prevent recursive breakdown. If a step already hosts a sub-goal, do not break it down further.
        if current_step_obj.get("sub_goal"):
            exec_thought_log.append("  HierarchicalPlanning: Skipped breakdown, step already contains a sub-goal.")
            return False

        target_state = current_step_obj.get("target_state")
        current_state = self.collapsed_computational_state_str

        if not target_state or target_state == current_state:
            return False

        self._log_lot_event("reasoning.hierarchical_planning.start", {"current_state": current_state, "target_state": target_state, "goal_step": current_step_obj.get('name')})

        # --- Simple Landmark Finding for 2-bit space: Find state 1 step away ---
        hamm_dist = sum(c1 != c2 for c1, c2 in zip(current_state, target_state))
        landmark_state = None
        if hamm_dist > 1:
            for i in range(len(current_state)):
                temp_list = list(current_state)
                temp_list[i] = '1' if temp_list[i] == '0' else '0'
                potential_landmark = "".join(temp_list)
                landmark_to_target_dist = sum(c1 != c2 for c1, c2 in zip(potential_landmark, target_state))
                if landmark_to_target_dist < hamm_dist:
                    landmark_state = potential_landmark
                    break

        if not landmark_state:
            exec_thought_log.append("  HierarchicalPlanning: Could not determine a useful landmark state.")
            self._log_lot_event("reasoning.hierarchical_planning.fail", {"reason": "no_landmark_found"})
            return False

        exec_thought_log.append(f"  HierarchicalPlanning: Breaking down step '{current_step_obj['name']}'. New landmark: |{landmark_state}>.")
        self._log_lot_event("reasoning.hierarchical_planning.success", {"landmark_state": landmark_state})

        sub_goal_step1 = {
            "name": f"Sub-goal: Reach landmark |{landmark_state}>",
            "target_state": landmark_state,
            "max_cycles_on_step": 5
        }
        original_step_as_sub_step2 = copy.deepcopy(current_step_obj)
        original_step_as_sub_step2['name'] = f"Sub-goal: Final step to |{target_state}>"
        # Critically, remove any ops hints from the second part to avoid confusion
        if 'next_ops_hint' in original_step_as_sub_step2:
             del original_step_as_sub_step2['next_ops_hint']

        sub_goal = GoalState(
            current_goal=f"Sub-goal for '{current_step_obj.get('name')}'",
            steps=[sub_goal_step1, original_step_as_sub_step2]
        )

        new_parent_step = {
            "name": f"Execute Sub-Goal for {target_state}",
            "sub_goal": sub_goal
        }

        parent_goal.steps[parent_step_idx] = new_parent_step
        parent_goal.history.append({
            "cycle": self.current_cycle_num,
            "event": "hierarchical_breakdown",
            "step_name": current_step_obj.get('name'),
            "new_sub_goal": sub_goal.current_goal
        })

        return True




    def _executive_plan_next_target_input(self, current_outcome_handle, executive_eval_results, exec_thought_log):
        exec_thought_log.append(f"PlanNextInput based on {current_outcome_handle} (mood {executive_eval_results['mood']:.2f}):")

        # Heuristic: cycle through states. This logic is universe-dependent and should
        # ideally be part of the universe config, but is kept here for simplicity.
        all_states = self.universe['states']
        try:
            current_index = all_states.index(current_outcome_handle)
            next_index = (current_index + 1) % len(all_states)
            next_handle = all_states[next_index]
        except (ValueError, IndexError):
            next_handle = self.universe['start_state']
        exec_thought_log.append(f"  Base heuristic next input: {next_handle}.")

        if self.current_goal_state_obj and self.current_goal_state_obj.status == "active":
            current_step_idx = self.current_goal_state_obj.current_step_index
            if 0 <= current_step_idx < len(self.current_goal_state_obj.steps):
                step_info = self.current_goal_state_obj.steps[current_step_idx]
                # Goal step can specify the next conceptual input for the world
                if isinstance(step_info.get("next_input_for_world"), StateHandle):
                    next_handle = step_info["next_input_for_world"]
                    exec_thought_log.append(f"  GoalStep '{step_info.get('name')}' overrides next input to {next_handle}.")
                    self._log_lot_event("executive.plannext.goal_override", {"next_input": str(next_handle), "goal_step_name":step_info.get('name',"")})

        elif self.internal_state_parameters['preferred_state_handle'] and \
           self.internal_state_parameters['preferred_state_handle'] != next_handle and \
           random.random() < self.internal_state_parameters['goal_seeking_bias'] * 0.75:
            next_handle = self.internal_state_parameters['preferred_state_handle']
            exec_thought_log.append(f"  Overridden by PreferredStateBias (bias {self.internal_state_parameters['goal_seeking_bias']:.2f}): next input {next_handle}.")
            self._log_lot_event("executive.plannext.preferred_state_override", {"next_input": str(next_handle), "bias": self.internal_state_parameters['goal_seeking_bias']})

        elif executive_eval_results['exploration_countdown'] > 0 or \
             (executive_eval_results['mood'] < -0.65 and random.random() < 0.55):
            available_inputs = list(self.universe['states'])
            if current_outcome_handle in available_inputs: available_inputs.remove(current_outcome_handle)
            if next_handle in available_inputs: available_inputs.remove(next_handle)

            if available_inputs:
                next_handle = random.choice(available_inputs)
                exec_thought_log.append(f"  Exploration/Mood (mood {executive_eval_results['mood']:.2f}, exp T-{executive_eval_results['exploration_countdown']}) override: next input {next_handle}.")
                self._log_lot_event("executive.plannext.exploration_override", {"next_input": str(next_handle), "mood":executive_eval_results['mood']})
            # else, keep the heuristic choice

        elif executive_eval_results['mood'] > 0.75 and random.random() < 0.40 and self.cycle_history:
            last_actual_input_handle = self.cycle_history[-1]['actual_input_state_handle']
            if last_actual_input_handle and last_actual_input_handle != current_outcome_handle :
                next_handle = last_actual_input_handle
                exec_thought_log.append(f"  Good mood ({executive_eval_results['mood']:.2f}), repeating last input context {last_actual_input_handle}.")
                self._log_lot_event("executive.plannext.good_mood_repeat", {"next_input": str(next_handle), "mood":executive_eval_results['mood']})

        exec_thought_log.append(f"  Final proposed next input: {next_handle}.")
        self.next_target_input_state_handle = next_handle
        return next_handle


    def _executive_update_goal_progress(self, collapsed_outcome_handle, executed_ops):
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
        self._log_lot_event("executive.goalprogress.check", {"goal_name": goal.current_goal, "step_name": step_name, "step_idx":step_idx, "outcome_state": str(collapsed_outcome_handle)})

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
                    "collapsed_state_at_eval_time": self.collapsed_computational_state_str,
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

        if not achieved_step and current_step.get("target_state") and collapsed_outcome_handle == current_step["target_state"]:
            achieved_step = True
            if self.verbose >=1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Step '{step_name}' achieved via target state {collapsed_outcome_handle}.")
        elif not achieved_step and callable(current_step.get("completion_criteria")):
            try:
                context_for_callable = {
                    'collapsed_state': collapsed_outcome_handle, # Pass the handle
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
            goal.history.append({"cycle": self.current_cycle_num, "event": f"step_completed", "step_name": step_name, "outcome_state":str(collapsed_outcome_handle), "current_step_index_at_event": goal.current_step_index})
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

                if self.internal_state_parameters['preferred_state_handle'] == current_step.get("target_state"): 
                    self.internal_state_parameters['preferred_state_handle'] = None 
            else: 
                 next_step_index_after_advance = goal.current_step_index
                 next_step_name = goal.steps[next_step_index_after_advance].get("name", f"Step {next_step_index_after_advance+1}")
                 if self.verbose >= 1: print(f"[{self.agent_id}] Goal '{goal.current_goal}': Advanced to step '{next_step_name}'.")
        else: 
            goal.history.append({"cycle": self.current_cycle_num, "event": "step_no_progress", "step_name": step_name, "current_outcome": str(collapsed_outcome_handle), "current_step_index_at_event": goal.current_step_index})
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
            if self.internal_state_parameters['preferred_state_handle'] == final_step_details.get("target_state"):
                 self.internal_state_parameters['preferred_state_handle'] = None
                 self._log_lot_event("executive.goalprogress.clear_pref_state_on_goal_end", {"goal_status": goal.status, "related_target_state": str(final_step_details.get("target_state")), "step_name_involved": final_step_details.get("name")})



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


# =================== AFTER ===================
    def _meta_layer_perform_review(self):
        if self.verbose >= 1: print(f"[{self.agent_id}] --- META_LAYER.Review (Cycle {self.current_cycle_num}) ---")
        self._log_lot_event("meta.review.start", {"cycle": self.current_cycle_num, "review_interval": self.metacognition_params['review_interval']})

        history_span_for_review = min(len(self.cycle_history), self.metacognition_params['review_interval'] * 3)
        if history_span_for_review < self.metacognition_params['review_interval'] * 0.6 :
            if self.verbose >= 1: print(f"    META.Review: Insufficient history ({history_span_for_review} cycles) for meaningful review.")
            self._log_lot_event("meta.review.insufficient_history", {"history_len": history_span_for_review})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        recent_history_slice = list(self.cycle_history)[-history_span_for_review:]
        valid_cycles = [c for c in recent_history_slice if c.get('valence_mod_this_cycle') is not None and c.get('op_strategy')]
        if not valid_cycles:
            if self.verbose >= 1: print("    META.Review: No valid cycles with strategy info in recent history.")
            self._log_lot_event("meta.review.no_valid_cycles", {"valid_cycles_count": len(valid_cycles)})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        # --- 1. Update Self-Model Statistics ---
        self_model = self.metacognition_params['self_model_stats']
        use_self_model = self.metacognition_params.get('enable_self_model_adaptation', False)
        
        # We perform a rolling update rather than a reset each time.
        # This builds a more stable long-term self-model.
        review_updates = collections.defaultdict(lambda: {'uses': 0, 'success': 0, 'valence': 0.0})
        for cycle in valid_cycles:
            # Map complex strategy names to base types for stats
            strategy = cycle['op_strategy']
            base_strategy = "default"
            if 'LTM' in strategy: base_strategy = 'memory'
            elif 'ProblemSolving' in strategy: base_strategy = 'problem_solve'
            elif 'Goal' in strategy: base_strategy = 'goal_seek'
            elif 'Curiosity' in strategy or 'Fallback' in strategy: base_strategy = 'curiosity'

            review_updates[base_strategy]['uses'] += 1
            review_updates[base_strategy]['valence'] += cycle['valence_mod_this_cycle']
            if cycle['valence_mod_this_cycle'] > self.metacognition_params['high_valence_threshold'] * 0.5:
                review_updates[base_strategy]['success'] += 1

        if use_self_model:
            # Integrate new observations into the long-term model using a learning rate
            lr = 0.1 # Learning rate for model update
            self_model['total_reviews_for_model'] += 1
            for strat, data in review_updates.items():
                if data['uses'] > 0:
                    self_model['strategy_total_uses'][strat] += data['uses']
                    self_model['strategy_success_count'][strat] += data['success']
                    self_model['strategy_total_valence_accum'][strat] += data['valence']

                    # Update rates with a moving average approach
                    recent_success_rate = data['success'] / data['uses']
                    self_model['strategy_success_rates'][strat] = (1 - lr) * self_model['strategy_success_rates'].get(strat, 0.0) + lr * recent_success_rate
                    
                    recent_avg_valence = data['valence'] / data['uses']
                    self_model['strategy_avg_valence'][strat] = (1 - lr) * self_model['strategy_avg_valence'].get(strat, 0.0) + lr * recent_avg_valence

            if self.verbose >= 2:
                print("    META.Review: Self-Model Updated.")
                for s in ['memory', 'problem_solve', 'goal_seek', 'curiosity']:
                    if self_model['strategy_total_uses'][s] > 0:
                         print(f"      - {s}: SuccessRate={self_model['strategy_success_rates'][s]:.2f}, AvgValence={self_model['strategy_avg_valence'][s]:.2f}, Uses(total)={self_model['strategy_total_uses'][s]}")
            self._log_lot_event("meta.review.self_model_update", {"model_state_str": str(self_model['strategy_success_rates'])})


        # --- 2. Data-Driven Adaptation using Self-Model (or Fallback to original logic) ---
        avg_valence_overall = np.mean([c['valence_mod_this_cycle'] for c in valid_cycles])
        outcome_diversity = len(set(c['collapsed_to_handle'] for c in valid_cycles)) / len(valid_cycles) if valid_cycles else 0.0

        if use_self_model and self_model['total_reviews_for_model'] > 2: # Wait for model to stabilize
            if self.verbose >= 1: print("    META.Review: Applying adaptations based on SELF-MODEL.")
            
            # If memory strategy is performing poorly, become more curious and experimental.
            if self_model['strategy_success_rates'].get('memory', 1.0) < 0.3 and self_model['strategy_avg_valence'].get('memory', 1.0) < 0.1 and self_model['strategy_total_uses']['memory'] > 5:
                if self.verbose >=2: print("      SELF-MODEL: Memory strategy is underperforming. Boosting curiosity.")
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.15)
                # Encourage shorter, more experimental operations
                self.internal_state_parameters['computation_length_preference'] = max(1, self.internal_state_parameters['computation_length_preference'] - 1)
                self._log_lot_event("meta.review.adapt_selfmodel.poor_memory", {})

            # If curiosity/problem-solving is yielding low valence, rely more on established, high-confidence memories.
            elif self_model['strategy_avg_valence'].get('curiosity', 1.0) < -0.2 and self_model['strategy_avg_valence'].get('problem_solve', 1.0) < -0.1 and self_model['strategy_total_uses']['curiosity'] > 5:
                if self.verbose >=2: print("      SELF-MODEL: Exploratory strategies are failing. Increasing weight of memory strategy.")
                sw = self.internal_state_parameters['strategy_weights']
                sw['memory'] = min(1.0, sw.get('memory', 0.1) * 1.3 + 0.1)
                sw['curiosity'] *= 0.7
                # Normalize weights after change
                total = sum(v for k,v in sw.items() if isinstance(v, (int, float)) and k != 'default')
                if total > 0: [sw.update({k: v/total}) for k,v in sw.items() if isinstance(v, (int,float)) and k != 'default']
                self._log_lot_event("meta.review.adapt_selfmodel.poor_exploration", {})

            # General adaptation of E_OR_THRESHOLD based on which strategies are succeeding
            td = self.orp_threshold_dynamics; prev_eor = self.E_OR_THRESHOLD
            adapt_rate_thresh = td.get('adapt_rate', DEFAULT_ORP_THRESHOLD_DYNAMICS['adapt_rate'])
            # If successful strategies are short (memory), no need for high ORP.
            if self_model['strategy_success_rates'].get('memory', 0.0) > 0.6 and self_model['strategy_avg_valence'].get('problem_solve', 1.0) < 0.2:
                self.E_OR_THRESHOLD = max(td['min'], self.E_OR_THRESHOLD - adapt_rate_thresh * 0.8)
                if self.verbose >= 2: print(f"      SELF-MODEL: Memory success suggests lower E_OR_THRESH is efficient. Decreased to {self.E_OR_THRESHOLD:.3f}")
            # If complex problem-solving is needed and working, allow more potential to build.
            elif self_model['strategy_success_rates'].get('problem_solve', 0.0) > 0.5 and self.E_OR_THRESHOLD < (td['max'] * 0.8):
                self.E_OR_THRESHOLD = min(td['max'], self.E_OR_THRESHOLD + adapt_rate_thresh)
                if self.verbose >= 2: print(f"      SELF-MODEL: Problem-solving success warrants higher E_OR_THRESH. Increased to {self.E_OR_THRESHOLD:.3f}")

        else: # Fallback to original, less sophisticated logic if self-model is disabled or new
            if self.verbose >= 1: print("    META.Review: Applying adaptations based on GENERAL stats (Self-Model OFF or new).")
            # This is the original logic from the "BEFORE" block
            if avg_valence_overall < self.metacognition_params['low_valence_threshold'] or outcome_diversity < self.metacognition_params['exploration_threshold_entropy']:
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + self.metacognition_params['curiosity_adaptation_rate'])
            if avg_valence_overall > self.metacognition_params['high_valence_threshold']:
                 self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + self.metacognition_params['goal_bias_adaptation_rate'])
            
        # --- 3. Epistemic Uncertainty & Knowledge Gap Review ---
        if self.metacognition_params.get('enable_epistemic_uncertainty_review', False):
            if self.verbose >= 1: print("    META.Review: Performing Epistemic Uncertainty (Knowledge Gap) scan...")
            
            confidence_threshold = self.metacognition_params.get('epistemic_confidence_threshold', 0.35)
            knowledge_gaps_found = []
            
            if self.long_term_memory:
                all_confidences = [data.get('confidence', 1.0) for data in self.long_term_memory.values()]
                avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
                if self.verbose >=2: print(f"      Overall LTM confidence: {avg_confidence:.3f}")

                for seq_tuple, data in self.long_term_memory.items():
                    confidence = data.get('confidence', 1.0)
                    if confidence < confidence_threshold:
                        # Check if it's not just a transient new memory
                        is_persistent_gap = (self.current_cycle_num - data.get('first_cycle', 0)) > 15
                        if is_persistent_gap:
                            knowledge_gaps_found.append({'seq': seq_tuple, 'confidence': confidence})

            if knowledge_gaps_found:
                num_gaps = len(knowledge_gaps_found)
                # Sort to find the most uncertain gap
                knowledge_gaps_found.sort(key=lambda x: x['confidence'])
                worst_gap = knowledge_gaps_found[0]

                if self.verbose >= 1: print(f"      KNOWLEDGE GAPS DETECTED: {num_gaps} low-confidence memories. Worst gap: conf={worst_gap['confidence']:.2f}. Boosting Curiosity.")
                self._log_lot_event("meta.review.knowledge_gap_found", {"count": num_gaps, "worst_gap_conf": worst_gap['confidence'], "worst_gap_seq_str": str(worst_gap['seq'])})
                
                # Take action: boost curiosity significantly to encourage exploration and solidification of these memories.
                curiosity_boost = self.metacognition_params.get('epistemic_curiosity_boost', 0.3)
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + curiosity_boost)
                # May also trigger exploration mode directly
                if random.random() < 0.25:
                    self.internal_state_parameters['exploration_mode_countdown'] = max(
                        self.internal_state_parameters['exploration_mode_countdown'],
                        self.metacognition_params['exploration_mode_duration']
                    )
                    if self.verbose>=2: print("      Triggering exploration mode due to knowledge gaps.")

        self.metacognition_params['cycles_since_last_review'] = 0
        if self.verbose >= 1: print(f"[{self.agent_id}] --- Metacognitive Review Complete ---")
        self._log_lot_event("meta.review.end", {"new_cur": self.internal_state_parameters['curiosity'], "new_gb":self.internal_state_parameters['goal_seeking_bias']})


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
            ### FIX: Make the loop detection key more specific by including the operations tuple.
            ### This prevents flagging non-identical actions that just happen to lead to the same state.
            behavior_patterns = []
            for c in history_slice:
                ops_tuple = tuple(tuple(op) for op in c.get('ops_executed', []))
                behavior_patterns.append((c['collapsed_to_handle'], c['op_strategy'], ops_tuple))
            
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
                        if total_sw > 1e-6:
                            for k in sw:
                                if isinstance(sw[k], (int,float)): sw[k] = max(0,sw[k]/total_sw)
                        else:
                            num_strats = len([k for k in sw if isinstance(sw[k],(int,float))]) or 1
                            for k in sw:
                                if isinstance(sw[k],(int,float)): sw[k] = 1.0/num_strats

                        self.internal_state_parameters['preferred_logical_state'] = None
                        intervention_made = True; intervention_details = {'pattern':str(pattern_tuple), 'count':count, 'avg_loop_val':np.mean(loop_valences)}
                        break

        prem_collapse_streak = self.firewall_params['premature_collapse_streak_needed']
        if not intervention_made and len(self.cycle_history) >= prem_collapse_streak:
            recent_collapse_data = list(self.cycle_history)[-prem_collapse_streak:]
            threshold_ratios = [c['orp_at_collapse'] / (c['E_OR_thresh_this_cycle']+1e-6) for c in recent_collapse_data if c.get('num_ops_executed',0) > 0]
            if threshold_ratios and len(threshold_ratios) >= prem_collapse_streak *0.75 and all(ratio < self.firewall_params['premature_collapse_orp_max_ratio'] for ratio in threshold_ratios):
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
    def run_full_cognitive_cycle(self, intended_conceptual_input:StateHandle, computation_sequence_ops=None):
        self.current_cycle_num += 1
        self.current_cycle_lot_stream = []
        start_time = time.time()

        self._log_lot_event("cycle_start", {"cycle_num":self.current_cycle_num, "intended_input_conceptual": str(intended_conceptual_input), "agent_id":self.agent_id, "current_mood":self.internal_state_parameters['mood'], "current_orp":self.objective_reduction_potential})
        if self.verbose >= 1: print(f"\n[{self.agent_id}] ===== Cycle {self.current_cycle_num} | Input: {intended_conceptual_input} =====")

        actual_computational_str, actual_conceptual_state = self._sensor_layer_process_input(intended_conceptual_input)
        if self.verbose >= 2: print(f"  SensorLayer Out: Actual perceived input {actual_conceptual_state} (basis |{actual_computational_str}>) from intended {intended_conceptual_input}")

        self._executive_prepare_superposition(actual_computational_str)
        
        # NOTE: Many internal methods will now use `self.current_conceptual_state`
        # for context instead of `self.collapsed_computational_state_str`
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

        collapsed_comp_str = self._executive_trigger_objective_reduction()
        # Look up the conceptual handle from the computational basis string
        collapsed_concept_handle = self.universe['comp_basis_to_state'].get(collapsed_comp_str, self.universe['start_state'])
        self.current_conceptual_state = collapsed_concept_handle
        orp_at_collapse = self.current_orp_before_reset

        # ADV_REASONING_FEATURE_1: Update active concepts based on the collapsed state.
        self._executive_update_active_concepts(collapsed_concept_handle)

        if self.verbose >= 1: print(f"  ExecutiveLayer OR: Collapsed to {collapsed_concept_handle} (basis |{collapsed_comp_str}>) (ORP experienced: {orp_at_collapse:.3f}, Early OR: {or_triggered_early}, Entropy: {entropy_at_collapse:.2f})")
        
        raw_valence_of_collapse = self.universe['valence_map'].get(collapsed_concept_handle, -0.15)
        self._executive_handle_collapse_interrupts(orp_at_collapse, executed_sequence, raw_valence_of_collapse)

        executive_eval_results = self._executive_evaluate_outcome_and_update_mood(
            collapsed_concept_handle, orp_at_collapse, entropy_at_collapse, len(executed_sequence or [])
        )
        if self.verbose >= 1: print(f"  ExecutiveLayer Eval: Val(raw/mod): {self.last_cycle_valence_raw:.2f}/{self.last_cycle_valence_mod:.2f}. Mood: {self.internal_state_parameters['mood']:.2f}, Frust: {self.internal_state_parameters['frustration']:.2f}")
        if self.verbose >=3 and executive_eval_results.get('thoughts_log'):
            for line_idx,line in enumerate(executive_eval_results['thoughts_log']): print(f"    AccEvalLog[{line_idx}]: {line}")

        if self.last_cycle_valence_mod > 0.65 and self.shared_attention_foci is not None:
            self.shared_attention_foci.append({'state': collapsed_concept_handle, 'op_seq': executed_sequence,
                                               'valence': self.last_cycle_valence_mod, 'agent_id': self.agent_id,
                                               'cycle': self.current_cycle_num})
            self._log_lot_event("coagent.attention_share", {"state":str(collapsed_concept_handle), "valence":self.last_cycle_valence_mod, "ops_count": len(executed_sequence or [])})

        consolidation_bonus = self.smn_internal_flags.pop('ltm_consolidation_bonus_factor', 1.0)
        
        # LTM Context is now based on conceptual StateHandles
        state_context_for_ltm = actual_conceptual_state # Default if no history
        if self.cycle_history and len(self.cycle_history) > 0:
            last_hist_entry = self.cycle_history[-1]
            if last_hist_entry and 'collapsed_to_handle' in last_hist_entry:
                 state_context_for_ltm = last_hist_entry['collapsed_to_handle']
            elif 'actual_input_state_handle' in last_hist_entry:
                 state_context_for_ltm = last_hist_entry['actual_input_state_handle']
        
        self._associative_layer_update_ltm(
            executed_sequence, self.last_cycle_valence_raw, orp_at_collapse, entropy_at_collapse,
            collapsed_concept_handle,
            consolidation_factor=consolidation_bonus,
            initial_state_when_sequence_started=state_context_for_ltm,
            input_context_when_sequence_started=actual_conceptual_state
        )
        if self.verbose >=2 and consolidation_bonus > 1.0 : print(f"  AssociativeLayer LTM Update applied consolidation bonus: {consolidation_bonus:.1f}")

        self._meta_layer_update_cognitive_parameters(orp_at_collapse, len(executed_sequence or []), executive_eval_results, entropy_at_collapse)
        self._meta_layer_adapt_preferred_state(collapsed_concept_handle, self.last_cycle_valence_mod) # Use conceptual state
        if self.verbose >= 1: print(f"  MetaLayer State: Attn={self.internal_state_parameters['attention_level']:.2f},Cur={self.internal_state_parameters['curiosity']:.2f},PrefS={str(self.internal_state_parameters['preferred_state_handle'])}>,Load={self.internal_state_parameters['cognitive_load']:.2f}")


        prev_mod_valence_for_smn = self.cycle_history[-1]['valence_mod_this_cycle'] if self.cycle_history else 0.0
        self._smn_update_and_apply_mutations(self.last_cycle_valence_mod, self.last_cycle_valence_raw, prev_mod_valence_for_smn, orp_at_collapse)

        self._firewall_detect_and_correct_anomalies() # Firewall might clear WM or alter goal status

        planning_log = []
        self._executive_plan_next_target_input(collapsed_concept_handle, executive_eval_results, planning_log)
        if self.verbose >= 1: print(f"  ExecutiveLayer PlanNext: Proposing {self.next_target_input_state_handle} for next cycle.")
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
            "intended_input_state_handle":intended_conceptual_input, "actual_input_state_handle":actual_conceptual_state,
            "ops_executed":executed_sequence, "op_strategy":chosen_op_strategy, "num_ops_executed":len(executed_sequence or []),
            "collapsed_to_comp":collapsed_comp_str, "collapsed_to_handle":collapsed_concept_handle,
            "orp_at_collapse":orp_at_collapse, "or_triggered_early": or_triggered_early,
            "num_terms_before_collapse":num_superposition_terms, "entropy_at_collapse":entropy_at_collapse,
            "valence_raw_this_cycle":self.last_cycle_valence_raw, "valence_mod_this_cycle":self.last_cycle_valence_mod,
            "mood_after_cycle":self.internal_state_parameters['mood'], "attention_after_cycle":self.internal_state_parameters['attention_level'],
            "cog_load_after_cycle":self.internal_state_parameters['cognitive_load'], "frustration_after_cycle":self.internal_state_parameters['frustration'],
            "curiosity_after_cycle":self.internal_state_parameters['curiosity'], "goal_bias_after_cycle":self.internal_state_parameters['goal_seeking_bias'],
            "preferred_state_after_cycle":str(self.internal_state_parameters.get('preferred_state_handle')),
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
        self._log_lot_event("cycle_end", {"duration_ms": cycle_duration * 1000, "next_planned_input_handle": str(self.next_target_input_state_handle), "final_mood": self.internal_state_parameters['mood']})
        if self.verbose >= 1: print(f"[{self.agent_id}] ===== Cycle {self.current_cycle_num} End (Dur: {cycle_duration:.3f}s, Next: {self.next_target_input_state_handle}) Mood:{self.internal_state_parameters['mood']:.2f} =====")

        return self.next_target_input_state_handle

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
                    "collapsed_state_at_eval_time": self.collapsed_computational_state_str, # State at the time of setting this initial context
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
        sw_str = ", ".join([f"{k[:4]}:{v:.2f}" for k,v in self.internal_state_parameters['strategy_weights'].items()])
        log_func(f"{indent}  StrategyWeights: {sw_str}")
        log_func(f"{indent}  MetaCog: ReviewIn: {self.metacognition_params['review_interval']-self.metacognition_params['cycles_since_last_review']}, AdaptRates(Cur/Goal): {self.metacognition_params['curiosity_adaptation_rate']:.3f}/{self.metacognition_params['goal_bias_adaptation_rate']:.3f}")
        
        ### NEW: SELF-MODEL SUMMARY ###
        if self.metacognition_params.get('enable_self_model_adaptation', False):
            self_model = self.metacognition_params.get('self_model_stats', {})
            if self_model and self_model.get('total_reviews_for_model', 0) > 0:
                log_func(f"{indent}  Self-Model ({self_model['total_reviews_for_model']} reviews):")
                perf_list = []
                for s in ['memory', 'problem_solve', 'goal_seek', 'curiosity']:
                    if self_model['strategy_total_uses'][s] > 0:
                        perf_list.append((s, self_model['strategy_avg_valence'][s], self_model['strategy_success_rates'][s]))
                if perf_list:
                    perf_list.sort(key=lambda x: x[1], reverse=True) # Sort by avg valence
                    best_strat, best_val, best_rate = perf_list[0]
                    worst_strat, worst_val, worst_rate = perf_list[-1]
                    log_func(f"{indent}    Best Strategy: '{best_strat}' (AvgVal: {best_val:.3f}, Success: {best_rate:.2f})")
                    log_func(f"{indent}    Worst Strategy: '{worst_strat}' (AvgVal: {worst_val:.3f}, Success: {worst_rate:.2f})")

        log_func(f"{indent}  LTM: {len(self.long_term_memory)}/{self.long_term_memory_capacity} entries. UtilWeights(V/E): {self.ltm_utility_weight_valence:.2f}/{self.ltm_utility_weight_efficiency:.2f}")
        if self.metacognition_params.get('enable_epistemic_uncertainty_review', False) and self.long_term_memory:
             all_confidences = [data.get('confidence', 0.0) for data in self.long_term_memory.values()]
             avg_conf = np.mean(all_confidences) if all_confidences else 0.0
             log_func(f"{indent}    LTM Avg Confidence: {avg_conf:.3f}")


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
            "collapsed_state": self.collapsed_computational_state_str, "preferred_state": self.internal_state_parameters['preferred_logical_state'],
            "E_OR_THRESHOLD": self.E_OR_THRESHOLD,
            "active_goal_name": self.current_goal_state_obj.current_goal if self.current_goal_state_obj else None,
            "active_goal_progress": self.current_goal_state_obj.progress if self.current_goal_state_obj else None,
            "active_goal_current_step_name": self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index].get("name", f"Step {self.current_goal_state_obj.current_step_index+1}") if self.current_goal_state_obj and self.current_goal_state_obj.steps and 0 <= self.current_goal_state_obj.current_step_index < len(self.current_goal_state_obj.steps) else None,
            "working_memory_depth": len(self.working_memory) if self.working_memory else 0,
            "verbose": self.verbose, # To allow completion_criteria to check agent's verbose level
        }

    def run_chained_cognitive_cycles(self, initial_input_str, num_cycles, computation_sequence_ops_template=None):
        if self.verbose >= 0: print(f"\n\n[{self.agent_id}] %%%%% STARTING CHAINED CYCLES (Num: {num_cycles}, Init Input: |{initial_input_str}>) %%%%%")

        try:
            # Initialize the handle for the first run from the input string
            self.next_target_input_state_handle = self.universe['comp_basis_to_state'][initial_input_str]
        except KeyError:
            if self.verbose >= 1: print(f"Warning: Initial input string '{initial_input_str}' not a valid state basis. Defaulting to start_state handle.")
            self.next_target_input_state_handle = self.universe['start_state']


        for i in range(num_cycles):
            # The handle to use for the current cycle is the one set by the previous cycle, or the initial one.
            current_input_handle_for_cycle = self.next_target_input_state_handle

            if self.verbose >= 1:
                pref_str = f"{self.internal_state_parameters['preferred_state_handle']}" if self.internal_state_parameters['preferred_state_handle'] else "None"
                goal_summary = "No Goal"
                if self.current_goal_state_obj:
                    step_name = "N/A"
                    if self.current_goal_state_obj.steps and 0 <= self.current_goal_state_obj.current_step_index < len(self.current_goal_state_obj.steps):
                         step_name = self.current_goal_state_obj.steps[self.current_goal_state_obj.current_step_index].get('name', 'UnnamedStep')
                    goal_summary = f"Goal: '{self.current_goal_state_obj.current_goal}' Step: '{step_name}' ({self.current_goal_state_obj.status})"
                wm_depth_info = f"WMd:{len(self.working_memory)}"

                print(f"\n>>>> Chained Cycle {i+1}/{num_cycles} for {self.agent_id} <<<< Input: {current_input_handle_for_cycle}; Mood:{self.internal_state_parameters['mood']:.2f}; Pref:{pref_str}; {goal_summary}; {wm_depth_info}")

            current_comp_ops = None
            if isinstance(computation_sequence_ops_template, list) and computation_sequence_ops_template:
                current_comp_ops = computation_sequence_ops_template[i % len(computation_sequence_ops_template)] if len(computation_sequence_ops_template) > 0 else None
            elif callable(computation_sequence_ops_template):
                 current_comp_ops = computation_sequence_ops_template(self, i)

            try:
                self.run_full_cognitive_cycle(current_input_handle_for_cycle, current_comp_ops)

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
            # IMPORTANT REFACTOR: Create a dictionary for kwargs to pass to the emulator
            # This is cleaner than modifying a dictionary in place
            agent_kwargs = copy.deepcopy(self.base_config)
            agent_kwargs['agent_id'] = agent_id
            
            # Use dictionary unpacking to merge variations. `agent_custom_settings` will override `base_config` keys.
            if i < len(self.agent_variations):
                agent_custom_settings = self.agent_variations[i]
                agent_kwargs = {**agent_kwargs, **agent_custom_settings}
            
            agent_kwargs['shared_long_term_memory'] = self.shared_long_term_memory
            agent_kwargs['shared_attention_foci'] = self.shared_attention_foci
            
            # The verbose level should be determined from the final combined configuration
            final_agent_verbose = agent_kwargs.get('verbose', self.verbose - 1 if self.verbose > 0 else 0)
            agent_kwargs['verbose'] = final_agent_verbose

            try:
                # The SimplifiedOrchOREmulator constructor must accept these keys directly or via **kwargs.
                emulator = SimplifiedOrchOREmulator(**agent_kwargs)
                self.agents.append(emulator)
                if self.verbose >= 1: print(f"  Initialized {agent_id}. Verbose: {final_agent_verbose}.")
                if self.verbose >=2 and (agent_kwargs.get('config_overrides') or agent_kwargs.get('trainable_param_values')):
                    print(f"    {agent_id} Initial Overrides Applied: {agent_kwargs.get('config_overrides', {})}")
                    if agent_kwargs.get('trainable_param_values'): 
                        print(f"    {agent_id} Initial Trainable Params: {agent_kwargs.get('trainable_param_values')}")
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
                # Use the handle for the cycle. The manager passes strings, the agent works with handles.
                current_agent_input_handle = agent.next_target_input_state_handle
                if initial_input_per_agent_list and self.system_cycle_num == 1 and agent_idx < len(initial_input_per_agent_list):
                    initial_input_str = initial_input_per_agent_list[agent_idx]
                    try:
                        current_agent_input_handle = agent.universe['comp_basis_to_state'][initial_input_str]
                    except KeyError:
                        print(f"Warning: CoAgentManager initial input '{initial_input_str}' for {agent.agent_id} not found, using agent default.")
                        current_agent_input_handle = agent.next_target_input_state_handle
                    agent.next_target_input_state_handle = current_agent_input_handle

                if self.verbose >=1: print(f"  Running {agent.agent_id} (Cycle {agent.current_cycle_num + 1}) with intended input {current_agent_input_handle}")
                try:
                    # Pass the HANDLE to the cycle function now
                    agent.run_full_cognitive_cycle(current_agent_input_handle)


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

