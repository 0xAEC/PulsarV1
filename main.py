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
from configurations import DEFAULT_LOT_CONFIG, DEFAULT_TRAINABLE_PARAMS_CONFIG, DEFAULT_INTERNAL_PARAMS, DEFAULT_COGNITIVE_FIREWALL_CONFIG

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
    
    # CRITICAL MODIFICATION: Unpack the universe into the config dict
    demo0_config = {
        **TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "agent_demo0_LayersLoT",
        'verbose': MASTER_VERBOSE_LEVEL, 'cycle_history_max_len': 6,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate': 0.02,
        'lot_config': {'enabled': True, 'log_level_details': demo0_lot_config_details},
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': False},
        'temporal_grid_config': {'max_len':3},
        'working_memory_max_depth': 5, 
    }
    emulator_demo0 = SimplifiedOrchOREmulator(**demo0_config)
    # CRITICAL MODIFICATION: Use imported state handle constants
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
    
    # CRITICAL MODIFICATION: Unpack the universe into the config dict
    demo1_config = {
        **TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "agent_demo1_TFG_SMN_Graph",
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.7, 'initial_orp_decay_rate':0.015,
        'temporal_grid_config': {
            'max_len': 6, 'feedback_window': 4,
            'low_valence_delta_threshold': -0.1, 'high_entropy_shift_threshold': 0.3,
        },
        'internal_state_parameters_config': { 
            **DEFAULT_INTERNAL_PARAMS, # Start with defaults
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
    emulator_demo1 = SimplifiedOrchOREmulator(**demo1_config)

    initial_smn_vals_d1 = {
        key: emulator_demo1._smn_get_param_value(info['path'])
        for key, info in emulator_demo1.smn_controlled_params_definitions.items()
    }
    print(f"{emulator_demo1.agent_id} starting with initial SMN values: {initial_smn_vals_d1}. SMN Graph enabled. Running 18 cycles.")
    
    # Modify the agent's universe directly for the demo
    # CRITICAL MODIFICATION: Use imported state handle constants
    emulator_demo1.universe['valence_map'] = {
        STATE_00: -0.6, STATE_01: 0.75,
        STATE_10: -0.3, STATE_11: 0.4
    }
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
    demo2_firewall_config_override = {
            'enabled': True, 'check_interval': 4, 'cooldown_duration': 6,
            'low_valence_threshold': -0.55,
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

    # CRITICAL MODIFICATION: Use a modified universe for this demo
    punishing_universe_config_d2 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    punishing_universe_config_d2['valence_map'] = {
        STATE_00: -0.5, STATE_01: -0.4, STATE_10: -0.5, STATE_11: 0.3
    }
    punishing_universe_config_d2['name'] = "Punishing Universe for Demo 2"
    
    demo2_config = {
        **punishing_universe_config_d2,
        'agent_id': "agent_demo2_IntFW",
        'verbose': MASTER_VERBOSE_LEVEL,
        'initial_E_OR_THRESHOLD': 0.5, 
        'initial_orp_decay_rate': 0.01,
        'interrupt_handler_config': demo2_interrupt_config_override,
        'cognitive_firewall_config': demo2_firewall_config_override,
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'lot_config': demo2_lot_config,
        'working_memory_max_depth': 5,
        'metacognition_config': {'review_interval': 5}
    }
    emulator_demo2 = SimplifiedOrchOREmulator(**demo2_config)

    # CRITICAL MODIFICATION: Use imported state handle constants for Goal definition
    goal_steps_demo2 = [
        {"name": "Survive Initial Punishment", "target_state": STATE_11, "next_ops_hint": [('H',0), ('H',1)], "requires_explicit_wm_context_push": True, "max_cycles_on_step": 7},
        {"name": "Explore a bit", "target_state": STATE_01, "requires_explicit_wm_context_push": False, "max_cycles_on_step": 7}
    ]
    task_goal_demo2 = GoalState(current_goal="Demo2 Survival and Exploration", steps=goal_steps_demo2)
    emulator_demo2.set_goal_state(task_goal_demo2)

    fix_it_seq_demo2 = (('H',0),('X',1),('H',1))
    emulator_demo2.long_term_memory[fix_it_seq_demo2] = {
        'count':10, 'total_valence': 7.0, 'avg_valence':0.7, 'total_orp_cost':1.0, 'avg_orp_cost':0.1,
        'total_entropy_generated':2.0, 'avg_entropy':0.2, 'utility':0.65, 'last_cycle':0, 'confidence': 0.7
    }
    print(f"{emulator_demo2.agent_id} starting with punishing valence map AND A GOAL. Expecting Firewall/Interrupt activity. Running 15 cycles.")

    wm_test_item = WorkingMemoryItem(type="dummy_context", data={"info":"test_firewall_clear"}, description="Pre-firewall item for Demo2")
    discarded = emulator_demo2.working_memory.push(wm_test_item)
    emulator_demo2._log_wm_op("push", item=wm_test_item, details={"reason": "manual_test_push_demo2", "item_discarded_on_push":discarded})

    emulator_demo2.run_chained_cognitive_cycles("00", 15)
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
        **TWO_QUBIT_UNIVERSE_CONFIG,
        'agent_id': "agent_demo3_Goal_WM",
        'verbose': MASTER_VERBOSE_LEVEL,
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
        
        wm_context_matches = False
        wm = context_dict['working_memory']
        if not wm.is_empty():
            top_item = wm.peek()
            goal_name = context_dict['current_goal_obj'].current_goal
            if top_item.type == "goal_step_context" and \
               top_item.data.get("goal_name") == goal_name and \
               top_item.data.get("goal_step_name") == step_name_for_this_step and \
               top_item.data.get("step_index") == context_dict['current_goal_obj'].current_step_index:
                wm_context_matches = True
        
        agent_state = context_dict['agent_public_state']
        if agent_state.get('verbose',0) >=2:
             print(f"    DEMO3 Criterion Check ('{step_name_for_this_step}'): State Achieved ({context_dict['collapsed_state']} vs {target_state_for_this_step}): {state_achieved}, WM Context Matches: {wm_context_matches}")

        return state_achieved and wm_context_matches
    
    # CRITICAL MODIFICATION: Use imported state handle constants for Goal definition
    goal_steps_demo3 = [
        {"name": "Reach state 01", "target_state": STATE_01, "requires_explicit_wm_context_push": True, "next_ops_hint": [('X',0),('H',1)], "max_cycles_on_step": 4},
        {"name": "From 01, reach state 10", 
         "target_state": STATE_10, 
         "completion_criteria": demo3_step2_callable_criterion,
         "next_input_for_world": STATE_01, "requires_explicit_wm_context_push": True, 
         "max_cycles_on_step": 8},
        {"name": "From 10, reach state 11 (final)", "target_state": STATE_11, "next_input_for_world": STATE_10, "requires_explicit_wm_context_push": True, "max_cycles_on_step": 4}
    ]
    task_goal_demo3 = GoalState(current_goal="WM-Enhanced Multi-step Task", steps=goal_steps_demo3, error_tolerance=0.1)
    emulator_demo3.set_goal_state(task_goal_demo3)

    print(f"{emulator_demo3.agent_id} attempting WM-enhanced goal: '{emulator_demo3.current_goal_state_obj.current_goal}'. Running up to 25 cycles.")
    op_template_d3 = [
        [('X',0)],              # For |00> -> |01>
        [('X',1), ('X',0)],     # For |01> -> |10>
        [('X',0)],              # For |10> -> |11>
        [('CNOT',(0,1))],       
        [],                     
        [('Z',0),('H',0)],      
        [('X',0)]               
    ]
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
    
    # CRITICAL MODIFICATION: Base config must include the universe
    coagent_base_conf_demo4 = {
        **TWO_QUBIT_UNIVERSE_CONFIG,
        'cycle_history_max_len': 25,
        'verbose': MASTER_VERBOSE_LEVEL -1 if MASTER_VERBOSE_LEVEL > 0 else 0,
        'smn_general_config': {'enabled': True, 'mutation_trigger_min_valence_gain': 0.12, 'enable_influence_matrix':True},
        'cognitive_firewall_config': {'enabled': True, 'check_interval': 5, 'cooldown_duration': 7, 'low_valence_streak_needed':3, 'clear_wm_on_intervention': True},
        'temporal_grid_config': {'max_len':8, 'feedback_window':4},
        'lot_config': {'enabled': False }, 
        'working_memory_max_depth': 10 
    }
    
    # CRITICAL MODIFICATION: Variations modify the `valence_map` within a universe dict
    # We now modify the 'universe' key for each agent variation.
    # The CoAgentManager init must handle this merge correctly.
    universe_var_1 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    universe_var_1['valence_map'] = {STATE_00:0.1, STATE_01:0.9, STATE_10:-0.3, STATE_11:0.3}
    
    universe_var_2 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    universe_var_2['valence_map'] = {STATE_00:-0.2, STATE_01:0.7, STATE_10:-0.7, STATE_11:0.5}

    universe_var_3 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    universe_var_3['valence_map'] = {STATE_00:0.0, STATE_01:0.3, STATE_10:-0.9, STATE_11:0.8}

    # Agent config variations now correctly modify nested dictionaries or specific parameters.
    # This structure is designed to be merged into the base config.
    coagent_variations_demo4 = [
        {'universe': universe_var_1, 'internal_state_parameters_config': {'curiosity': 0.85}, 'initial_E_OR_THRESHOLD': 0.6},
        {'universe': universe_var_2, 'internal_state_parameters_config': {'goal_seeking_bias': 0.75}, 'initial_orp_decay_rate': 0.008},
        {'universe': universe_var_3, 'internal_state_parameters_config': {'strategy_weights': {'memory': 0.7}}, 'initial_E_OR_THRESHOLD': 1.3},
    ]

    manager_demo4 = CoAgentManager(num_agents=3,
                             base_emulator_config_template=coagent_base_conf_demo4,
                             agent_config_variations_list=coagent_variations_demo4,
                             trainable_params_config=DEFAULT_TRAINABLE_PARAMS_CONFIG,
                             verbose=MASTER_VERBOSE_LEVEL)
                             
    print(f"CoAgentManager demo with {manager_demo4.num_agents} agents. Running 15 system cycles. Expect inter-agent learning attempts.")
    manager_demo4.run_system_cycles(num_system_cycles=15, initial_input_per_agent_list=["00", "01", "10"])


    # --- DEMO 5: Cognitive Agent Trainer ---
    print("\n\n--- DEMO 5: Cognitive Agent Trainer ---")
    
    # CRITICAL MODIFICATION: Base trainer config unpacks the universe
    trainer_universe_config_d5 = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG)
    trainer_universe_config_d5['valence_map'] = {STATE_00: -0.2, STATE_01: 1.0, STATE_10: -0.6, STATE_11: 0.1}
    
    trainer_base_emulator_config_d5 = {
        **trainer_universe_config_d5,
        'cycle_history_max_len': 12,
        'initial_E_OR_THRESHOLD': 0.9,
        'initial_orp_decay_rate': 0.025,
        'smn_general_config': {'enabled': False, 'enable_influence_matrix': False},
        'cognitive_firewall_config': {'enabled': True, 'check_interval':8, 'cooldown_duration':12},
        'temporal_grid_config': {'max_len':5, 'feedback_window':3},
        'internal_state_parameters_config': {**DEFAULT_INTERNAL_PARAMS, 'preferred_state_handle': STATE_01, 'sensor_input_noise_level': 0.005},
        'verbose_emulator_episodes': MASTER_VERBOSE_LEVEL - 2 if MASTER_VERBOSE_LEVEL > 1 else 0,
        'trainer_goal_completion_reward': 1.0, 'trainer_goal_failure_penalty': -0.6, 'trainer_goal_progress_reward_factor': 0.4,
        'config_overrides': { ('successful_sequence_threshold_valence',): 0.4 },
        'working_memory_max_depth': 8
    }

    # CRITICAL MODIFICATION: Use imported state handle constants for Goal template
    trainer_goal_template_dict_d5 = {"current_goal": "Trainer Task: Reach 01",
                             "steps": [{"name": "Achieve state 01", "target_state": STATE_01, "requires_explicit_wm_context_push":True}], 
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
    
    # Re-apply goal and preferred state to the final test agent
    # CRITICAL MODIFICATION: Use imported state handle constants
    if trainer_base_emulator_config_d5.get('internal_state_parameters_config', {}).get('preferred_state_handle'):
        trained_emulator_d5.internal_state_parameters['preferred_state_handle'] = STATE_01
    if trainer_goal_template_dict_d5:
        trained_emulator_d5.set_goal_state(GoalState(**copy.deepcopy(trainer_goal_template_dict_d5)))

    trained_emulator_d5.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=15)
    print(f"\nFinal Working Memory for {trained_emulator_d5.agent_id}: Depth {len(trained_emulator_d5.working_memory)}")
    if not trained_emulator_d5.working_memory.is_empty():
        print(f"  Top WM Item: {trained_emulator_d5.working_memory.peek()}")
    
    print(f"  Final Goal Status for {trained_emulator_d5.agent_id}: {trained_emulator_d5.current_goal_state_obj}")


    # --- DEMO X: Interactive Psych-Probe ---
    print("\n\n--- DEMO X: Interactive Psych-Probe ---")
    probe_config = {
        'verbose': 1,
        'agent_id': "agent_probe_test",
        **TWO_QUBIT_UNIVERSE_CONFIG,
        'lot_config': {'enabled': True, 'log_level_details': { 'system':True, 'executive':True, 'meta':True, 'firewall':True, 'smn':True } },
    }
    probe_agent = SimplifiedOrchOREmulator(**probe_config)

    # We will run 10 cycles, but set a breakpoint to trigger the probe after cycle 3
    probe_agent.probe_at_cycle = 3

    print(f"Running agent '{probe_agent.agent_id}' for 10 cycles with a probe set for cycle 3.")
    print("After cycle 3, you will be dropped into an interactive shell.")
    print("Type 'help' in the shell for a list of commands (e.g., 'summary', 'wm', 'ltm 5', 'get internal_state_parameters.mood').")

    probe_agent.run_chained_cognitive_cycles(initial_input_str='00', num_cycles=10)

    print(f"\n--- Probe Demo Concluded ---")

    print("\n\n--- ALL DEMOS COMPLETED ---")
