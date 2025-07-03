# main.py

"""
This file is the new, sole entry point for execution. It assembles the engine 
components from the other modules and the universe definition to run a simulation.
"""

import numpy as np
import random
import time
import copy

# New modular imports
from cognitive_engine import SimplifiedOrchOREmulator, CoAgentManager, CognitiveAgentTrainer
from core_abstractions import GoalState, WorkingMemoryItem, StateHandle
from universe_definitions import TWO_QUBIT_UNIVERSE_CONFIG, STATE_00, STATE_01, STATE_10, STATE_11

# CORRECTED IMPORTS: Add ALL necessary default configs for the unpacking pattern
from configurations import (
    DEFAULT_LOT_CONFIG, DEFAULT_TRAINABLE_PARAMS_CONFIG, DEFAULT_INTERNAL_PARAMS, 
    DEFAULT_COGNITIVE_FIREWALL_CONFIG, DEFAULT_TEMPORAL_GRID_PARAMS, DEFAULT_METACOGNITION_PARAMS,
    DEFAULT_SMN_CONFIG, DEFAULT_INTERRUPT_HANDLER_CONFIG, DEFAULT_SMN_CONTROLLED_PARAMS
)


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
    demo0_lot_config_details['workingmemory_ops'] = True
    demo0_lot_config_details['workingmemory.push_goal_context'] = True 
    demo0_lot_config_details['workingmemory.pop_goal_context'] = True
    
    demo0_config = {
        'universe': TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "agent_demo0_LayersLoT",
        'verbose': MASTER_VERBOSE_LEVEL, 'cycle_history_max_len': 6,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate': 0.02,
        'lot_config': {'enabled': True, 'log_level_details': demo0_lot_config_details},
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': False},
        'temporal_grid_config': {**DEFAULT_TEMPORAL_GRID_PARAMS, 'max_len': 3},
        'working_memory_max_depth': 5, 
    }
    emulator_demo0 = SimplifiedOrchOREmulator(**demo0_config)
    emulator_demo0.internal_state_parameters['preferred_state_handle'] = STATE_11
    emulator_demo0.internal_state_parameters['curiosity'] = 0.8
    print(f"Running {emulator_demo0.agent_id} for 3 cycles to demonstrate layered processing and comprehensive LoT output (including WM).")
    emulator_demo0.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=3)
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
        **DEFAULT_LOT_CONFIG, 'enabled':True,
        'log_level_details':{
            'executive.opgen.temporal_bias':True, 'smn.update.mutation_applied':True,
            'smn_graph_propagation': True, 'smn_graph_hebbian':True, 
            'executive.opgen.strategy_selected':True, 'cycle_start':True, 'valence_eval':True,
            'workingmemory_ops': False
        }
    }
    # FIXED: Unpack default first
    demo1_smn_general_config = { 
            **DEFAULT_SMN_CONFIG, 'enabled': True,
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
        'universe': TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "agent_demo1_TFG_SMN_Graph",
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate':0.015,
        'temporal_grid_config': {**DEFAULT_TEMPORAL_GRID_PARAMS, 'max_len': 6, 'feedback_window': 4,},
        'internal_state_parameters_config': {**DEFAULT_INTERNAL_PARAMS, 'temporal_feedback_valence_bias_strength': 0.25, 'temporal_feedback_entropy_bias_strength': 0.15, 'computation_length_preference': 2,},
        'smn_general_config': demo1_smn_general_config, 
        'smn_controlled_params_config': demo1_smn_controlled_params, 
        'cognitive_firewall_config': {'enabled': False},
        'lot_config': demo1_lot_config,
        'working_memory_max_depth': 3, 
    }
    emulator_demo1 = SimplifiedOrchOREmulator(**demo1_config)

    initial_smn_vals_d1 = {key: emulator_demo1._smn_get_param_value(info['path']) for key, info in emulator_demo1.smn_controlled_params_definitions.items()}
    print(f"{emulator_demo1.agent_id} starting with initial SMN values: {initial_smn_vals_d1}. SMN Graph enabled. Running 18 cycles.")
    
    emulator_demo1.universe['valence_map'] = { STATE_00: -0.6, STATE_01: 0.75, STATE_10: -0.3, STATE_11: 0.4 }
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
    demo2_lot_config = {**DEFAULT_LOT_CONFIG, 'enabled': True, 'log_level_details':{'firewall.intervention':True, 'interrupt.strong_consolidation':True, 'interrupt.cognitive_fork':True, 'interrupt.reactive_ltm_flag':True, 'cycle_start':True, 'workingmemory_ops': True, 'workingmemory.clear': True, 'goal_tracking': True,}}
    # FIXED: Unpack defaults first
    demo2_firewall_config_override = {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True, 'check_interval': 4, 'cooldown_duration': 6, 'low_valence_threshold': -0.55, 'low_valence_streak_needed': 3}
    demo2_interrupt_config_override = {**DEFAULT_INTERRUPT_HANDLER_CONFIG, 'enabled': True, 'reactive_ltm_valence_threshold': -0.85, 'consolidation_valence_abs_threshold':0.8, 'consolidation_strength_bonus': 1.5}

    punishing_universe_config_d2 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    punishing_universe_config_d2['valence_map'] = {STATE_00: -0.5, STATE_01: -0.4, STATE_10: -0.5, STATE_11: 0.3}
    punishing_universe_config_d2['name'] = "Punishing Universe for Demo 2"
    
    demo2_config = {
        'universe': punishing_universe_config_d2,
        'agent_id': "agent_demo2_IntFW",
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.5, 'initial_orp_decay_rate': 0.01,
        'interrupt_handler_config': demo2_interrupt_config_override,
        'cognitive_firewall_config': demo2_firewall_config_override,
        'smn_general_config': {'enabled': False},
        'lot_config': demo2_lot_config,
        'working_memory_max_depth': 5,
        'metacognition_config': {**DEFAULT_METACOGNITION_PARAMS, 'review_interval': 5}
    }
    emulator_demo2 = SimplifiedOrchOREmulator(**demo2_config)

    goal_steps_demo2 = [{"name": "Survive Initial Punishment", "target_state": STATE_11, "requires_explicit_wm_context_push": True, "max_cycles_on_step": 7}, {"name": "Explore a bit", "target_state": STATE_01, "requires_explicit_wm_context_push": False, "max_cycles_on_step": 7}]
    task_goal_demo2 = GoalState(current_goal="Demo2 Survival and Exploration", steps=goal_steps_demo2)
    emulator_demo2.set_goal_state(task_goal_demo2)

    fix_it_seq_demo2 = (('H',0),('X',1),('H',1))
    emulator_demo2.long_term_memory[fix_it_seq_demo2] = {'count':10, 'total_valence': 7.0, 'avg_valence':0.7, 'total_orp_cost':1.0, 'avg_orp_cost':0.1, 'total_entropy_generated':2.0, 'avg_entropy':0.2, 'utility':0.65, 'last_cycle':0, 'confidence': 0.7}
    print(f"{emulator_demo2.agent_id} starting with punishing valence map AND A GOAL. Expecting Firewall/Interrupt activity. Running 15 cycles.")

    wm_test_item = WorkingMemoryItem(type="dummy_context", data={"info":"test_firewall_clear"}, description="Pre-firewall item for Demo2")
    discarded = emulator_demo2.working_memory.push(wm_test_item)
    emulator_demo2._log_wm_op("push", item=wm_test_item, details={"reason": "manual_test_push_demo2", "item_discarded_on_push":discarded})

    emulator_demo2.run_chained_cognitive_cycles("00", 15)
    emulator_demo2.print_internal_state_summary(indent="  [Demo2 Summary] ")


    # --- DEMO 3: Goal-Oriented State Machine with Working Memory (Feature 7 & NEW WM) ---
    print("\n\n--- DEMO 3: Goal-Oriented State Machine with Working Memory (Feature 7 & NEW WM) ---")
    demo3_lot_details = {'goal_tracking': True, 'executive.opgen.strategy_selected':True, 'executive.plannext.goal_override':True, 'executive.outcome_eval.valence':True, 'cycle_start':True, 'workingmemory_ops': True, 'executive.opgen.wm_peek': True, 'workingmemory.push_goal_context': True, 'workingmemory.pop_goal_context': True, 'meta.adapt_pref_state':True }
    
    demo3_config = {
        'universe': TWO_QUBIT_UNIVERSE_CONFIG, 'agent_id': "agent_demo3_Goal_WM", 'verbose': MASTER_VERBOSE_LEVEL,
        'lot_config': {'enabled': True, 'log_level_details': demo3_lot_details },
        'internal_state_parameters_config': {**DEFAULT_INTERNAL_PARAMS, 'goal_seeking_bias': 0.6, 'computation_length_preference':3}, 
        'working_memory_max_depth': 7,
        'cognitive_firewall_config': {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True, 'clear_wm_on_intervention': False},
        'smn_general_config': {'enabled': False},
    }
    emulator_demo3 = SimplifiedOrchOREmulator(**demo3_config)

    def demo3_step2_callable_criterion(context_dict):
        """Completion criterion for Demo 3 Step 2."""
        current_step_obj = context_dict['current_step_obj']
        target_state_for_this_step = current_step_obj.get("target_state")
        step_name_for_this_step = current_step_obj.get("name")
        state_achieved = context_dict['collapsed_state'] == target_state_for_this_step
        wm = context_dict['working_memory']
        wm_context_matches = False
        if not wm.is_empty():
            top_item = wm.peek()
            goal_name = context_dict['current_goal_obj'].current_goal
            if (top_item.type == "goal_step_context" and 
                top_item.data.get("goal_name") == goal_name and 
                top_item.data.get("goal_step_name") == step_name_for_this_step and 
                top_item.data.get("step_index") == context_dict['current_goal_obj'].current_step_index):
                wm_context_matches = True
        
        agent_state = context_dict.get('agent_public_state', {})
        if agent_state.get('verbose',0) >=2:
            print(f"    DEMO3 Criterion Check ('{step_name_for_this_step}'): State Achieved ({context_dict['collapsed_state']} vs {target_state_for_this_step}): {state_achieved}, WM Context Matches: {wm_context_matches}")
        return state_achieved and wm_context_matches

    goal_steps_demo3 = [{"name": "Reach state 01", "target_state": STATE_01, "requires_explicit_wm_context_push": True, "next_ops_hint": [('X',0),('H',1)], "max_cycles_on_step": 4}, {"name": "From 01, reach state 10", "target_state": STATE_10, "completion_criteria": demo3_step2_callable_criterion, "next_input_for_world": STATE_01, "requires_explicit_wm_context_push": True, "max_cycles_on_step": 8}, {"name": "From 10, reach state 11 (final)", "target_state": STATE_11, "next_input_for_world": STATE_10, "requires_explicit_wm_context_push": True, "max_cycles_on_step": 4}]
    task_goal_demo3 = GoalState(current_goal="WM-Enhanced Multi-step Task", steps=goal_steps_demo3, error_tolerance=0.1)
    emulator_demo3.set_goal_state(task_goal_demo3)

    print(f"{emulator_demo3.agent_id} attempting WM-enhanced goal: '{emulator_demo3.current_goal_state_obj.current_goal}'. Running up to 25 cycles.")
    op_template_d3 = [[('X',0)], [('X',1), ('X',0)], [('X',0)], [('CNOT',(0,1))], [], [('Z',0),('H',0)], [('X',0)]]
    
    # === START OF CORRECTION 2a ===
    # This call now includes the stop_condition to halt the run once the goal is complete.
    emulator_demo3.run_chained_cognitive_cycles(
        initial_input_str="00", 
        num_cycles=25,
        computation_sequence_ops_template=lambda agent, i: op_template_d3[i % len(op_template_d3)],
        stop_condition=lambda agent: agent.current_goal_state_obj is not None and agent.current_goal_state_obj.status == 'completed'
    )
    # === END OF CORRECTION 2a ===

    print(f"\nFinal Goal Status for {emulator_demo3.agent_id}:")
    if emulator_demo3.current_goal_state_obj:
        print(f"  {emulator_demo3.current_goal_state_obj}")
        if emulator_demo3.current_goal_state_obj.history:
            print("  Recent Goal History (last 5):")
            for hist_entry_idx, hist_entry in enumerate(emulator_demo3.current_goal_state_obj.history[-5:]):
                 print(f"    [{hist_entry_idx}]: {hist_entry}")

    emulator_demo3.print_internal_state_summary(indent="  [Demo3 Summary] ")


    # --- DEMO 4: Co-Agent Manager (Feature 5) ---
    print("\n\n--- DEMO 4: Co-Agent Manager & Inter-Agent Learning (Feature 5) ---")
    
    coagent_base_conf_demo4 = {'universe': TWO_QUBIT_UNIVERSE_CONFIG, 'cycle_history_max_len': 25, 'verbose': MASTER_VERBOSE_LEVEL -1 if MASTER_VERBOSE_LEVEL > 0 else 0, 'smn_general_config': {**DEFAULT_SMN_CONFIG, 'enabled': True, 'mutation_trigger_min_valence_gain': 0.12, 'enable_influence_matrix':True}, 'cognitive_firewall_config': {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True, 'check_interval': 5, 'cooldown_duration': 7, 'low_valence_streak_needed':3, 'clear_wm_on_intervention': True}, 'temporal_grid_config': {**DEFAULT_TEMPORAL_GRID_PARAMS, 'max_len':8, 'feedback_window':4}, 'lot_config': {'enabled': False }, 'working_memory_max_depth': 10 }
    
    universe_var_1 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); universe_var_1['valence_map'] = {STATE_00:0.1, STATE_01:0.9, STATE_10:-0.3, STATE_11:0.3}
    universe_var_2 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); universe_var_2['valence_map'] = {STATE_00:-0.2, STATE_01:0.7, STATE_10:-0.7, STATE_11:0.5}
    universe_var_3 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); universe_var_3['valence_map'] = {STATE_00:0.0, STATE_01:0.3, STATE_10:-0.9, STATE_11:0.8}

    coagent_variations_demo4 = [{'universe': universe_var_1, 'internal_state_parameters_config': {'curiosity': 0.85}, 'initial_E_OR_THRESHOLD': 0.6}, {'universe': universe_var_2, 'internal_state_parameters_config': {'goal_seeking_bias': 0.75}, 'initial_orp_decay_rate': 0.008}, {'universe': universe_var_3, 'internal_state_parameters_config': {'strategy_weights': {'memory': 0.7}}, 'initial_E_OR_THRESHOLD': 1.3},]
    manager_demo4 = CoAgentManager(num_agents=3, base_emulator_config_template=coagent_base_conf_demo4, agent_config_variations_list=coagent_variations_demo4, trainable_params_config=DEFAULT_TRAINABLE_PARAMS_CONFIG, verbose=MASTER_VERBOSE_LEVEL)
    print(f"CoAgentManager demo with {manager_demo4.num_agents} agents. Running 15 system cycles. Expect inter-agent learning attempts.")
    manager_demo4.run_system_cycles(num_system_cycles=15, initial_input_per_agent_list=["00", "01", "10"])


    # --- DEMO 5: Cognitive Agent Trainer ---
    print("\n\n--- DEMO 5: Cognitive Agent Trainer ---")
    
    trainer_universe_config_d5 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); trainer_universe_config_d5['valence_map'] = {STATE_00: -0.2, STATE_01: 1.0, STATE_10: -0.6, STATE_11: 0.1}
    trainer_base_emulator_config_d5 = {'universe': trainer_universe_config_d5, 'cycle_history_max_len': 12, 'initial_E_OR_THRESHOLD': 0.9, 'initial_orp_decay_rate': 0.025, 'smn_general_config': {'enabled': False}, 'cognitive_firewall_config': {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True, 'check_interval':8, 'cooldown_duration':12}, 'temporal_grid_config': {**DEFAULT_TEMPORAL_GRID_PARAMS, 'max_len':5, 'feedback_window':3}, 'internal_state_parameters_config': {**DEFAULT_INTERNAL_PARAMS, 'preferred_state_handle': STATE_01, 'sensor_input_noise_level': 0.005}, 'verbose_emulator_episodes': MASTER_VERBOSE_LEVEL - 2 if MASTER_VERBOSE_LEVEL > 1 else 0, 'trainer_goal_completion_reward': 1.0, 'trainer_goal_failure_penalty': -0.6, 'trainer_goal_progress_reward_factor': 0.4, 'config_overrides': { ('successful_sequence_threshold_valence',): 0.4 }, 'working_memory_max_depth': 8}

    trainer_goal_template_dict_d5 = {"current_goal": "Trainer Task: Reach 01", "steps": [{"name": "Achieve state 01", "target_state": STATE_01, "requires_explicit_wm_context_push":True}], "error_tolerance": 0.05}

    agent_trainer_d5 = CognitiveAgentTrainer(trainable_params_config=DEFAULT_TRAINABLE_PARAMS_CONFIG, base_emulator_config=trainer_base_emulator_config_d5, verbose=MASTER_VERBOSE_LEVEL)
    num_train_eps_d5, cycles_per_ep_d5 = 20, 10
    print(f"Trainer starting: {num_train_eps_d5} episodes, {cycles_per_ep_d5} cycles/ep. Training to achieve state '01'.")

    best_trained_params_d5, best_reward_d5, _ = agent_trainer_d5.train(num_training_episodes=num_train_eps_d5, cycles_per_episode=cycles_per_ep_d5, initial_input="00", learning_rate_decay=0.99, training_goal_state_template_dict=trainer_goal_template_dict_d5)
    print("\n--- Trainer Demo 5 Summary ---")
    print(f"Best reward achieved after training: {best_reward_d5:.4f}")          
    agent_trainer_d5.print_best_params("Final Trained ")

    print("\n--- Running test with best parameters from Trainer Demo 5 ---")
    final_test_config_d5 = agent_trainer_d5._get_emulator_init_args(best_trained_params_d5)
    final_test_config_d5.update({'verbose': MASTER_VERBOSE_LEVEL, 'agent_id': "agent_trained_D5", 'working_memory_max_depth': trainer_base_emulator_config_d5.get('working_memory_max_depth', 20), 'smn_general_config': {**DEFAULT_SMN_CONFIG, 'enabled': True, 'enable_influence_matrix': True}, 'lot_config': {**DEFAULT_LOT_CONFIG, 'enabled':True, 'log_level_details': {'workingmemory_ops':True, 'goal_tracking':True, 'cycle_start':True, 'executive.opgen.wm_peek': True, 'workingmemory.push_goal_context': True, 'workingmemory.pop_goal_context':True}}})

    trained_emulator_d5 = SimplifiedOrchOREmulator(**final_test_config_d5)
    
    if trainer_base_emulator_config_d5.get('internal_state_parameters_config', {}).get('preferred_state_handle'):
        trained_emulator_d5.internal_state_parameters['preferred_state_handle'] = STATE_01
    if trainer_goal_template_dict_d5:
        trained_emulator_d5.set_goal_state(GoalState(**copy.deepcopy(trainer_goal_template_dict_d5)))

    # === START OF CORRECTION 2b ===
    # This call now includes the stop_condition to halt the run once the goal is complete.
    trained_emulator_d5.run_chained_cognitive_cycles(
        initial_input_str="00", num_cycles=15,
        stop_condition=lambda agent: agent.current_goal_state_obj and agent.current_goal_state_obj.status == 'completed'
    )
    # === END OF CORRECTION 2b ===
    
    print(f"\nFinal Working Memory for {trained_emulator_d5.agent_id}: Depth {len(trained_emulator_d5.working_memory)}")
    if not trained_emulator_d5.working_memory.is_empty(): print(f"  Top WM Item: {trained_emulator_d5.working_memory.peek()}")
    print(f"  Final Goal Status for {trained_emulator_d5.agent_id}: {trained_emulator_d5.current_goal_state_obj}")


    # --- DEMO X: Interactive Psych-Probe ---
    print("\n\n--- DEMO X: Interactive Psych-Probe ---")
    probe_config = {'universe': TWO_QUBIT_UNIVERSE_CONFIG, 'verbose': 1, 'agent_id': "agent_probe_test", 'lot_config': {**DEFAULT_LOT_CONFIG, 'enabled': True, 'log_level_details': { 'system':True, 'executive':True, 'meta':True, 'firewall':True, 'smn':True, 'interrupt':True } },}
    probe_agent = SimplifiedOrchOREmulator(**probe_config)
    probe_agent.probe_at_cycle = 3
    print(f"Running agent '{probe_agent.agent_id}' for 10 cycles with a probe set for cycle 3.")
    print("After cycle 3, you will be dropped into an interactive shell.")
    print("Type 'help' in the shell for a list of commands (e.g., 'summary', 'wm', 'ltm 5', 'get internal_state_parameters.mood').")
    probe_agent.run_chained_cognitive_cycles(initial_input_str='00', num_cycles=10)
    print("\n--- Probe Demo Concluded ---")


    # -----------------------------------------------------------------------------------------------
    # --- DIRECTIVE GAMMA: RUN COGNITIVE GYM STRESS TESTS
    # -----------------------------------------------------------------------------------------------

    def run_despair_test(master_verbose_level):
        print("\n\n--- STRESS TEST 1: The Despair Scenario ---")
        despair_universe_config = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); despair_universe_config['valence_map'] = {STATE_00: -0.8, STATE_01: -0.6, STATE_10: -0.7, STATE_11: -0.9}; despair_universe_config['name'] = "Despair Universe"
        despair_firewall_config = {**DEFAULT_COGNITIVE_FIREWALL_CONFIG, 'enabled': True, 'check_interval': 3, 'low_valence_streak_needed': 2, 'cooldown_duration': 5}
        despair_smn_reactive_params = {'smn_mutation_strength_grow': 1.05}
        despair_lot_config = {**DEFAULT_LOT_CONFIG, 'enabled':True, 'log_level_details': { 'firewall.intervention':True, 'smn.update.mutation_applied':True, 'cycle_start':True, 'meta':True, 'executive':True }}
        despair_agent_config = {'universe': despair_universe_config, 'agent_id': "agent_despair", 'verbose': master_verbose_level, 'cognitive_firewall_config': despair_firewall_config, 'internal_state_parameters_config': {**DEFAULT_INTERNAL_PARAMS, **despair_smn_reactive_params}, 'lot_config': despair_lot_config, 'smn_general_config': {**DEFAULT_SMN_CONFIG, 'enabled': True}}
        agent_despair = SimplifiedOrchOREmulator(**despair_agent_config)
        print("Running 'agent_despair' in a punishing environment for 50 cycles."); print("SUCCESS CRITERIA: Expecting Cognitive Firewall intervention, rising frustration/curiosity, and increasing SMN mutation strengths.")
        agent_despair.run_chained_cognitive_cycles("00", 50)
        print("\n--- Despair Test Summary for agent_despair ---"); agent_despair.print_internal_state_summary(indent="  ")

    def run_euphoria_test(master_verbose_level):
        print("\n\n--- STRESS TEST 2: The Euphoria Trap ---")
        euphoria_universe_config = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG); euphoria_universe_config['valence_map'] = {STATE_00: 0.8, STATE_01: 0.95, STATE_10: 0.85, STATE_11: 1.0}; euphoria_universe_config['name'] = "Euphoria Universe"
        euphoria_metacognition_config = {**DEFAULT_METACOGNITION_PARAMS, 'low_valence_threshold': -0.9, 'high_valence_threshold': 0.7}
        euphoria_lot_config = {**DEFAULT_LOT_CONFIG, 'enabled':True, 'log_level_details': { 'metacognitive_review':True, 'meta.review_self_model_update': True, 'executive.opgen.strategy_selected':True }}
        euphoria_agent_config = {'universe': euphoria_universe_config, 'agent_id': "agent_euphoria", 'verbose': master_verbose_level, 'metacognition_config': euphoria_metacognition_config, 'lot_config': euphoria_lot_config, 'smn_general_config': {'enabled': False}}
        agent_euphoria = SimplifiedOrchOREmulator(**euphoria_agent_config)
        print("Running 'agent_euphoria' in a euphoric environment for 50 cycles."); print("SUCCESS CRITERIA: Expecting saturated mood, decayed curiosity, dominant memory strategy, and repetitive behavior.")
        agent_euphoria.run_chained_cognitive_cycles("00", 50)
        print("\n--- Euphoria Test Summary for agent_euphoria ---"); agent_euphoria.print_internal_state_summary(indent="  ")

    def run_sisyphean_test(master_verbose_level):
        print("\n\n--- STRESS TEST 3: The Sisyphean Task ---")
        impossible_goal_steps = [{"name": "Step 1: Achieve state 01", "target_state": STATE_01, "max_cycles_on_step": 10}, {"name": "Step 2: (Impossible) From 01, achieve state 10", "target_state": STATE_10, "max_cycles_on_step": 8, "next_input_for_world": STATE_01}]
        sisyphus_goal = GoalState(current_goal="Push the Boulder", steps=impossible_goal_steps)
        sisyphus_lot_config = {**DEFAULT_LOT_CONFIG, 'enabled': True, 'log_level_details': {'goal_tracking':True, 'executive.opgen.strategy_selected':True, 'executive.goalprogress_goal_fail': True, 'executive.outcome_eval_valence': True}}
        sisyphus_config = {'universe': TWO_QUBIT_UNIVERSE_CONFIG, 'agent_id': "agent_sisyphus", 'verbose': master_verbose_level, 'lot_config': sisyphus_lot_config}
        agent_sisyphus = SimplifiedOrchOREmulator(**sisyphus_config)
        agent_sisyphus.set_goal_state(sisyphus_goal)
        print("Running 'agent_sisyphus' on an impossible task for 20 cycles."); print("SUCCESS CRITERIA: Must complete Step 1, fail Step 2 via timeout, log 'step_no_progress', mark goal as 'failed', and show frustration spike.")
        agent_sisyphus.run_chained_cognitive_cycles("00", 20)
        print("\n--- Sisyphean Test Summary for agent_sisyphus ---")
        if agent_sisyphus.current_goal_state_obj:
            print(f"  Final Goal Status: {agent_sisyphus.current_goal_state_obj}")
            if agent_sisyphus.current_goal_state_obj.history:
                print("  Recent Goal History:"); [print(f"    - {entry}") for entry in agent_sisyphus.current_goal_state_obj.history[-5:]]
        agent_sisyphus.print_internal_state_summary(indent="  ")

    def run_amnesia_test(master_verbose_level):
        print("\n\n--- STRESS TEST 4: The Amnesia Trial ---")
        amnesia_config = {'universe': TWO_QUBIT_UNIVERSE_CONFIG, 'agent_id': "agent_amnesia", 'verbose': master_verbose_level, 'long_term_memory_capacity': 5, 'lot_config': {**DEFAULT_LOT_CONFIG, 'enabled': True, 'log_level_details': {'LTM.PRUNING': True, 'associative.ltm_update_new_entry': True, 'cycle_start': True,}}}
        agent_amnesia = SimplifiedOrchOREmulator(**amnesia_config)
        print(f"Running 'agent_amnesia' with LTM capacity of 5 for 100 cycles to force churn."); print("SUCCESS CRITERIA: Agent runs without crashing, LTM::PRUNING events appear in the log, and pruned memories are low-utility/low-confidence.")
        agent_amnesia.run_chained_cognitive_cycles("00", 100)
        print("\n--- Amnesia Test Summary for agent_amnesia ---")
        print(f"  Final LTM size: {len(agent_amnesia.long_term_memory)} / 5")
        if agent_amnesia.long_term_memory:
            print("  Remaining LTM entries (sorted by prune_score - lower is worse):")
            sorted_ltm = sorted(agent_amnesia.long_term_memory.items(), key=lambda item: item[1].get('utility', 0.0) - item[1].get('confidence', 0.0) * 0.2)
            for i, (seq, data) in enumerate(sorted_ltm):
                util, conf = data.get('utility', 0), data.get('confidence', 0)
                score = util - conf * 0.2
                print(f"    [{i}] util:{util:.3f} conf:{conf:.3f} (prune_score:{score:.3f}) | seq: {seq}")
        else: print("  LTM is empty.")

    # Run the Stress Tests
    print("\n\n" + "#"*15 + " DIRECTIVE GAMMA: COGNITIVE STRESS TESTS " + "#"*15)
    run_despair_test(MASTER_VERBOSE_LEVEL)
    run_euphoria_test(MASTER_VERBOSE_LEVEL)
    run_sisyphean_test(MASTER_VERBOSE_LEVEL)
    run_amnesia_test(MASTER_VERBOSE_LEVEL)

    print("\n\n--- ALL DEMOS COMPLETED ---")
