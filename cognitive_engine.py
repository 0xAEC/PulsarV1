# cognitive_engine.py

"""
This file is the "brain." It contains the core agent class and its
operational frameworks (Trainer, CoAgentManager). It contains no top-level 
configuration dictionaries; it imports them from `configurations.py` and
uses them as defaults.
"""
# CORRECTED and MINIMAL imports for this file
import tensorflow as tf

from perception import VisualCortexVAE, PredictiveWorldModel 
import numpy as np
import copy
import time
import random
import collections 
import traceback 
import heapq
import math
import gymnasium as gym
import pickle 
import os


from typing import List, Dict, Any, Deque, Optional, Tuple
# PerceptionSystem is no longer directly used by the agent, but may be used by other helpers. Keep for now.
from environment import PerceptionSystem 

from perception import SymbolicPerceptionCore
from core_abstractions import ActiveConceptNetGraph, LanguageOfThought, Program, Function

from core_abstractions import StateHandle, WorkingMemoryItem, WorkingMemoryStack, GoalState, LogEntry
from universe_definitions import TWO_QUBIT_UNIVERSE_CONFIG
from configurations import (
    DEFAULT_INTERNAL_PARAMS,
    DEFAULT_METACOGNITION_PARAMS,
    DEFAULT_ORP_THRESHOLD_DYNAMICS,
    DEFAULT_ORP_DECAY_DYNAMICS,
    DEFAULT_TRAINABLE_PARAMS_CONFIG,
    DEFAULT_TEMPORAL_GRID_PARAMS,
    DEFAULT_SMN_CONFIG,
    DEFAULT_SMN_CONTROLLED_PARAMS,
    DEFAULT_INTERRUPT_HANDLER_CONFIG,
    DEFAULT_COGNITIVE_FIREWALL_CONFIG,
    DEFAULT_GOAL_STATE_PARAMS,
    DEFAULT_LOT_CONFIG,
    DEFAULT_VAE_CONFIG,
    DEFAULT_WORLD_MODEL_CONFIG,
    DEFAULT_LIFELONG_LEARNING_CONFIG
)
# ---------------------------------------------------------------------------
# Helper Class: ExperienceReplayBuffer
# ---------------------------------------------------------------------------
class ExperienceReplayBuffer:
    """A 'smart' replay buffer that prioritizes storing cognitively significant events."""
    def __init__(self, capacity, emotion_threshold, surprise_threshold):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.emotion_threshold = emotion_threshold
        self.surprise_threshold = surprise_threshold

    def add(self, experience: Dict, valence_mod: float, prediction_error: Optional[float]):
        """
        Adds an experience to the buffer only if it's emotionally charged or surprising.
        
        Args:
            experience (Dict): A dict containing {'state_img', 'action', 'next_state_img'}.
            valence_mod (float): The modified valence from the cognitive cycle.
            prediction_error (float): The error from the world model's last prediction.
        
        Returns:
            bool: True if the experience was added, False otherwise.
        """
        is_emotional = abs(valence_mod) > self.emotion_threshold
        is_surprising = (prediction_error is not None) and (prediction_error > self.surprise_threshold)

        if is_emotional or is_surprising:
            self.buffer.append(experience)
            return True
        return False
        
    def sample(self, batch_size: int) -> List[Dict]:
        """Samples a random batch of experiences from the buffer."""
        return random.sample(list(self.buffer), min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)
    


# ---------------------------------------------------------------------------
# Class Definition: GoalManager
# ---------------------------------------------------------------------------
class GoalManager:
    """
    The agent's "prefrontal cortex" for life management.
    Arbitrates between multiple competing goals, selecting the most urgent
    one for the current cognitive cycle.
    """
    def __init__(self, agent_ref):
        self.agent: 'SimplifiedOrchOREmulator' = agent_ref
        self.goals: List[GoalState] = []
        self._survival_goal_active = False

    def update_and_arbitrate(self) -> Optional[GoalState]:
        """
        Primary interface. Updates goal list based on world state and
        then selects the most urgent goal to focus on.
        """
        self._update_goals_from_world_state()
        return self._arbitrate_active_goal()

    def add_goal(self, goal: GoalState):
        """Adds a new goal to the list, preventing duplicates."""
        if not any(g.current_goal == goal.current_goal for g in self.goals):
            goal.status = 'active'
            self.goals.append(goal)
            self.agent._log_lot_event("goal_manager", "goal_added", {"goal": str(goal)})

    def set_main_goal(self, goal: GoalState):
        """Sets the primary, long-term goal, removing any previous main goal."""
        # Remove any existing main quests
        self.goals = [g for g in self.goals if g.goal_type != 'MAIN_QUEST']
        self.add_goal(goal)
        
    def _update_goals_from_world_state(self):
        """Creates/removes goals based on world state AND internal drives."""
        # --- Plan Formulation Drive ---
        main_quest = next((g for g in self.goals if g.goal_type == 'MAIN_QUEST'), None)
        plan_formulation_goal_exists = any(g.goal_type == 'PLAN_FORMULATION' for g in self.goals)
        
        if main_quest and main_quest.status == 'active' and not main_quest.steps and not plan_formulation_goal_exists:
            self.add_goal(GoalState(
                current_goal="Formulate Plan for Main Quest", goal_type="PLAN_FORMULATION",
                base_priority=0.9, 
                steps=[{"name": "Decompose Main Quest", "cognitive_action": "decompose_task"}],
                evaluation_criteria="GOAL_COMPLETION" 
            ))
            
        # --- Standard Reactive Goal Management (Survival, Opportunity) ---
        obs = self.agent.last_full_obs
        if obs:
            agent_energy = obs.get('agent_energy', np.array([100.0]))[0]
            survival_goal_exists = any(g.goal_type == 'SURVIVAL' for g in self.goals)
            if agent_energy < 80.0 and not survival_goal_exists:
                self.add_goal(GoalState(
                    current_goal="Maintain Energy", goal_type="SURVIVAL", base_priority=0.8,
                    steps=[{"name": "Recharge", "target_concept": "charging_pad"}]
                ))
            elif agent_energy >= 99.0 and survival_goal_exists:
                self.goals = [g for g in self.goals if g.goal_type != 'SURVIVAL']

            bonus_loc = obs.get('bonus_target_loc')
            opportunity_goal_exists = any(g.goal_type == 'OPPORTUNITY' for g in self.goals)
            if bonus_loc is not None and bonus_loc[0] != -1 and not opportunity_goal_exists:
                self.add_goal(GoalState(
                    current_goal="Seize Opportunity", goal_type="OPPORTUNITY", base_priority=0.6,
                    steps=[{"name": "Collect Bonus", "target_concept": "bonus_target"}]
                ))
            elif (bonus_loc is None or bonus_loc[0] == -1) and opportunity_goal_exists:
                self.goals = [g for g in self.goals if g.goal_type != 'OPPORTUNITY']
        
        # --- SECE Intrinsic Motivation (The fast, efficient version) ---
        p = self.agent.internal_state_parameters
        # This version gets the future_latents from the attribute set in run_cycle
        future_latents = getattr(self.agent, '_future_latents_this_cycle', None)
        
        if p['peml_active'] and p['curiosity'] > p['peml_curiosity_requirement'] and future_latents:
            
            investigate_goal_exists = any(g.goal_type == 'INVESTIGATE' for g in self.goals)

            if len(future_latents) > 1 and not investigate_goal_exists:
                all_latents = np.array(list(future_latents.values()))
                variance = np.mean(np.var(all_latents, axis=0))

                if variance > p['peml_uncertainty_threshold']:
                    mean_latent = np.mean(all_latents, axis=0)
                    distances = {action: np.linalg.norm(latent - mean_latent) for action, latent in future_latents.items()}
                    most_interesting_action = max(distances, key=distances.get)
                    
                    self.add_goal(GoalState(
                        current_goal=f"Investigate Uncertainty (Action {most_interesting_action})",
                        goal_type="INVESTIGATE", base_priority=0.4,
                        steps=[{"name": f"Perform uncertain action {most_interesting_action}", "action_to_take": most_interesting_action}],
                        evaluation_criteria="GOAL_COMPLETION" 
                    ))
                    self.agent._log_lot_event("peml", "investigate_goal_created", {"variance": variance, "action": most_interesting_action})


    def _arbitrate_active_goal(self) -> Optional[GoalState]:
        """Calculates urgency for all active goals and returns the winner."""
        if not self.goals:
            return None

        urgency_scores = {}
        # --- FIX #1: THE CRITICAL LOGIC FIX ---
        # Only consider goals that are NOT completed or failed for arbitration.
        # This prevents the agent from getting stuck on a finished task.
        for goal in self.goals:
            if goal.status in ['active', 'pending']:
                urgency = self._calculate_urgency(goal)
                urgency_scores[goal.current_goal] = (urgency, goal)
        
        if not urgency_scores:
            # This can happen if all available goals are completed/failed.
            return None

        # Select the goal with the highest urgency
        top_goal_name = max(urgency_scores, key=lambda k: urgency_scores[k][0])
        arbitrated_goal = urgency_scores[top_goal_name][1]

        self.agent._log_lot_event("goal_manager", "arbitration", {
            "winner": f"{arbitrated_goal.goal_type}:{arbitrated_goal.current_goal}",
            "winner_urgency": urgency_scores[top_goal_name][0],
            "contenders": {k: f"{v[0]:.3f}" for k, v in urgency_scores.items()}
        })
        return arbitrated_goal


    def _calculate_urgency(self, goal: GoalState) -> float:
        """The core heuristic for calculating goal urgency."""
        urgency = 0.0
        obs = self.agent.last_full_obs
        if obs is None:
            return goal.base_priority # Fallback

        # --- THE FINAL FIX: PRIORITIZING PLANNING ---
        # --- THE FINAL FIX: PRIORITIZING PLANNING ---
        if goal.goal_type == "PLAN_FORMULATION":
            # Not having a plan is the most urgent crisis. Give it a near-infinite urgency
            # to ensure it becomes the focus, allowing the agent to think before acting.
            return 100.0

        elif goal.goal_type == "SURVIVAL":
            energy = obs.get('agent_energy', np.array([100.0]))[0]
            max_energy = 100.0
            # Urgency skyrockets as energy approaches zero.
            urgency = goal.base_priority + ((1.0 - (energy / max_energy))**3 * 5.0)
            
        elif goal.goal_type == "OPPORTUNITY":
            timer = obs.get('bonus_target_timer', np.array([0]))[0]
            max_lifespan = 50.0 # Match env definition
            # Urgency increases as the opportunity is about to expire.
            time_pressure = (1.0 - (timer / max_lifespan))**2 * 1.5
            curiosity_bonus = self.agent.internal_state_parameters['curiosity'] * 0.5
            urgency = goal.base_priority + time_pressure + curiosity_bonus
            
        elif goal.goal_type == "MAIN_QUEST":
            goal_bias = self.agent.internal_state_parameters['goal_seeking_bias']
            frustration_penalty = self.agent.internal_state_parameters['frustration'] * 0.5
            
            impatience_bonus = 0.0
            if self.agent.last_arbitrated_goal and self.agent.last_arbitrated_goal.current_goal == goal.current_goal:
                 impatience_bonus = self.agent.cycles_stuck_on_step * 0.015 

            urgency = goal.base_priority + goal_bias - frustration_penalty + impatience_bonus

        else: # Default case
            urgency = goal.base_priority

        return max(0, urgency) # Urgency cannot be negative
    

class LambdaLearner:
    """The engine of intellectual growth."""
    def __init__(self, lot_library_ref: LanguageOfThought):
        self.lot_library = lot_library_ref
        # MODIFIED: program_history now stores a dictionary for richer context for analogy.
        self.program_history: Dict[str, Dict[str, Any]] = {}
        self.subroutine_counter = collections.Counter()
        self.verbose = 0
        self._log_lot_event = lambda *args, **kwargs: None

    # MODIFIED: Method signature changed to accept the diff for analogical reasoning.
    def log_successful_program(self, task_id: str, program: Program, diff: Dict):
        """Logs a successful program and its generating diff for future analogy."""
        self.program_history[task_id] = {'program': program, 'diff': diff}

        if not program or not isinstance(program, list) or 'function' not in program[0]:
            return
        serializable_program = tuple(step['function'] for step in program)
        if len(serializable_program) < 2: return

        for i in range(len(serializable_program) - 1):
            self.subroutine_counter[serializable_program[i:i+2]] += 1


    def consolidate_knowledge(self):
        """Triggers abstraction, creating new functions from redundant subroutines."""
        if self.verbose >= 1: print("  LAMBDA_LEARNER: Consolidating knowledge...")
        for subroutine_tuple, count in self.subroutine_counter.items():
            if count > 2:
                new_func_name = "Learned_" + "_then_".join(subroutine_tuple)
                if self.lot_library.get_function(new_func_name): continue
                if self.verbose >= 1: print(f"    Found redundant subroutine: {subroutine_tuple}. Abstracting to '{new_func_name}'")
                learned_body = [{'function': fn, 'args': []} for fn in subroutine_tuple]
                self.lot_library.add_learned_function(Function(name=new_func_name, body=None, is_learned=True, learned_body=learned_body))
                self._log_lot_event("lambda_learner", "function_learned", {"name": new_func_name})

# ---------------------------------------------------------------------------
# >>> NEW: Pillar 3 - The Multi-Strategy Scientific Reasoning Engine
# ---------------------------------------------------------------------------
class MultiStrategyScientificReasoningEngine:
    """
    Implements the scientific method. It searches for a program in the agent's LoT
    that explains the data, using multiple, cascading strategies.
    """
    def __init__(self, agent_ref: 'SimplifiedOrchOREmulator'):
        self.agent = agent_ref
        self.verbose = agent_ref.verbose
        self._log_lot_event = agent_ref._log_lot_event

    def generate_solution_program(self, input_graph: ActiveConceptNetGraph, output_graph: ActiveConceptNetGraph, diff: Dict, input_grid: np.ndarray) -> Program | None:
        """Main entry point. Tries strategies from simplest to most complex."""
        self._log_lot_event("reasoning_engine", "start_search", {"diff_keys": list(d for d in diff if diff[d])})
        
        # --- THE ULTIMATE COGNITIVE ORDERING ---
        # 0. NEW: Before anything else, see if a high-level learned skill solves the problem.
        program = self._strategy_use_learned_abstraction(input_graph, output_graph, input_grid)
        if program:
            self._log_lot_event("reasoning_engine", "solution_found", {"strategy": 0, "method": "Learned Abstraction", "program": [p['function'] for p in program]})
            return program
            
        # 1. If no learned skill applies, check LTM for an analogous problem to adapt.
        program = self._strategy_analogical_synthesis(diff, input_graph, output_graph)
        if program:
            self._log_lot_event("reasoning_engine", "solution_found", {"strategy": 1, "method": "Analogical Synthesis", "program": [p['function'] for p in program]})
            return program

        # 2. If no memory exists, solve from scratch using simple, direct observations.
        program = self._strategy_empirical_correspondence(input_graph, output_graph, diff)
        if program:
            self._log_lot_event("reasoning_engine", "solution_found", {"strategy": 2, "method": "Empirical Correspondence", "program": [p['function'] for p in program]})
            return program
        
        # 3. If direct observation fails, form a more abstract, overarching theory.
        program = self._strategy_inductive_generation(input_graph, output_graph, diff)
        if program:
            self._log_lot_event("reasoning_engine", "solution_found", {"strategy": 3, "method": "Inductive Generation", "program": [p['function'] for p in program]})
            return program

        self._log_lot_event("reasoning_engine", "search_failed", {"reason": "all_strategies_exhausted"})
        return None
    

    def _strategy_use_learned_abstraction(self, in_graph: ActiveConceptNetGraph, out_graph: ActiveConceptNetGraph, in_grid: np.ndarray) -> Program | None:
        """Strategy 0: Test high-level learned skills against the problem."""
        if self.verbose >= 1: print("    REASONING: Trying Strategy 0 (Use Learned Abstraction)...")
        
        learned_functions = self.agent.language_of_thought.learned_functions
        if not learned_functions:
            return None # No skills have been learned yet.

        for func_name, func_obj in learned_functions.items():
            # To test a learned function, we must simulate its effect.
            # A learned function is just a program. We can run it on the input grid.
            if self.verbose >= 2: print(f"      Testing learned skill: '{func_name}'")
            
            # We need to re-ground the arguments for the simulation.
            # This is a simplified but effective way for copy-move.
            adapted_program = self.agent.reasoning_engine._strategy_analogical_synthesis(
                ActiveConceptNetGraph.graph_diff(in_graph, out_graph),
                in_graph,
                out_graph
            )

            if adapted_program:
                 # Check if the body of the learned function matches the adapted program's functions
                learned_body_funcs = tuple(s['function'] for s in func_obj.learned_body)
                adapted_program_funcs = tuple(s['function'] for s in adapted_program)

                if learned_body_funcs == adapted_program_funcs:
                    if self.verbose >= 1: print(f"      SUCCESS: Learned skill '{func_name}' is a perfect match for this problem's required operations.")
                    # We can now use the high-level function call, but we must pass the adapted arguments.
                    # For a simple program like [Copy, Move], the args for the learned function
                    # would be the combined args of its body. This can get complex.
                    # A simpler, powerful approach: return the adapted program but LOG that a learned skill was the key.
                    # For now, let's just use the learned function name directly if it requires no arguments.
                    # A more robust solution would be to make the learned function callable with the necessary args.
                    # Let's return the adapted program, but acknowledge the learned skill was the gateway.
                    # This is a good compromise for this stage.
                    return adapted_program # The key is that we *found* it via the learned skill.

        return None


    def _strategy_empirical_correspondence(self, in_graph, out_graph, diff):
        """Strategy 1: Finds simple 1-to-1 mappings. Good for Move/Recolor/etc."""
        if self.verbose >= 1: print("    REASONING: Trying Strategy 1 (Empirical Correspondence)...")
    
        nodes_added = diff.get('nodes_added', [])
        
        # === NEW COMPLEX LOGIC: Handle object "growth" as a copy/move ===
        # This fires when an object changes shape/size, but no new object is created. This is common
        # for adjacent copies that get merged into a single object by the perception system.
        num_in_objects = len([n for n in in_graph.nodes.values() if n.type == 'object'])
        num_out_objects = len([n for n in out_graph.nodes.values() if n.type == 'object'])

        if not nodes_added and num_in_objects == 1 and num_out_objects == 1:
            in_obj = next((n for n in in_graph.nodes.values() if n.type == 'object'), None)
            out_obj = next((n for n in out_graph.nodes.values() if n.type == 'object'), None)
    
            if in_obj and out_obj:
                in_coords = in_obj.attributes.get('Coordinates', [])
                out_coords = out_obj.attributes.get('Coordinates', [])
                
                # Check for same color and a size increase
                if in_obj.attributes.get('Color') == out_obj.attributes.get('Color') and len(out_coords) > len(in_coords):
                    # Find the pixels that were added to form the new object
                    added_pixels = sorted(list(set(tuple(c) for c in out_coords) - set(tuple(c) for c in in_coords)))
    
                    # Check if the added pixels form a shape identical to the original object
                    if len(added_pixels) == len(in_coords):
                        original_shape_hash = in_obj.attributes.get('ShapeHash')
    
                        # Calculate the ShapeHash of the newly added pixels to see if they match the original's shape
                        if added_pixels:
                            min_r_new = min(r for r, c in added_pixels)
                            min_c_new = min(c for r, c in added_pixels)
                            new_shape_hash = tuple(sorted([(r - min_r_new, c - min_c_new) for r, c in added_pixels]))
    
                            if new_shape_hash == original_shape_hash:
                                # It's a match! We've deduced a copy-move. Now find the translation vector.
                                old_pos = in_obj.attributes.get('Position', (0, 0)) # top-left of original
                                new_pos_of_copy = (min_r_new, min_c_new) # top-left of the new part
                                
                                # Movement is (dy, dx). Remember grid is (row, col) so dy is change in row, dx is change in col.
                                dy = new_pos_of_copy[0] - old_pos[0]
                                dx = new_pos_of_copy[1] - old_pos[1]
                                
                                if self.verbose >= 1: print("      Strategy 1 Insight: Object 'growth' detected as identical copy-move.")
                                # The selector must refer to the original object's ID from the input graph.
                                return [
                                    self.agent.language_of_thought.get_function('Copy').body(f"object_with_id({in_obj.id})"),
                                    self.agent.language_of_thought.get_function('Move').body("last_created_object", dx, dy)
                                ]

        # === ORIGINAL LOGIC: Handle a distinct, non-adjacent new object being added ===
        if len(nodes_added) == 1:
            added_attrs = nodes_added[0]
            # This logic is looking for an object in the input that has the same intrinsic properties (shape, color)
            # as the newly added object. It's the classic, non-adjacent copy detection.
            for node in in_graph.nodes.values():
                if node.type == 'object' and \
                   node.attributes.get('ShapeHash') == added_attrs.get('ShapeHash') and \
                   node.attributes.get('Color') == added_attrs.get('Color'):
                    old_pos = node.attributes.get('Position', (0, 0))
                    new_pos = added_attrs.get('Position', (0, 0))
                    dx, dy = new_pos[1] - old_pos[1], new_pos[0] - old_pos[0]
                    return [
                        self.agent.language_of_thought.get_function('Copy').body(f"object_with_id({node.id})"),
                        self.agent.language_of_thought.get_function('Move').body("last_created_object", dx, dy)
                    ]
        return None

    def _strategy_analogical_synthesis(self, diff: Dict, in_graph: ActiveConceptNetGraph, out_graph: ActiveConceptNetGraph) -> Program | None:
        """Strategy 2: Queries LTM for problems with a similar transformation signature and re-grounds the parameters."""
        if self.verbose >= 1: print("    REASONING: Trying Strategy 2 (Analogical Synthesis)...")
        current_diff_sig = tuple(sorted(key for key, val in diff.items() if val))
        if not current_diff_sig: return None

        for task_id, memory in self.agent.lambda_learner.program_history.items():
            past_diff = memory.get('diff', {})
            past_diff_sig = tuple(sorted(key for key, val in past_diff.items() if val))

            if current_diff_sig == past_diff_sig:
                if self.verbose >= 1: print(f"      Found analogous solved problem: {task_id}. Adapting its program.")
                
                recalled_program = memory['program']
                
                # --- RE-GROUNDING LOGIC ---
                # Check if the recalled program needs its parameters updated for the new context.
                # This is a targeted fix for the 'Move' function. A more advanced system
                # would have a general way to tag parameters that need re-grounding.
                new_program = []
                is_regrounded = False
                for step in recalled_program:
                    if step.get('function') == 'Move':
                        # The 'Move' arguments are likely stale. Recalculate them.
                        # This logic is intentionally duplicated from _strategy_empirical_correspondence
                        # to show the re-grounding process. A future refactor could create a shared helper function.
                        in_obj = next((n for n in in_graph.nodes.values() if n.type == 'object'), None)
                        out_obj = next((n for n in out_graph.nodes.values() if n.type == 'object'), None)

                        if in_obj and out_obj:
                            in_coords = in_obj.attributes.get('Coordinates', [])
                            out_coords = out_obj.attributes.get('Coordinates', [])
                            added_pixels = sorted(list(set(tuple(c) for c in out_coords) - set(tuple(c) for c in in_coords)))
                            
                            if added_pixels:
                                old_pos = in_obj.attributes.get('Position', (0, 0))
                                new_pos_of_copy = (min(r for r, c in added_pixels), min(c for r, c in added_pixels))
                                dy = new_pos_of_copy[0] - old_pos[0]
                                dx = new_pos_of_copy[1] - old_pos[1]
                                
                                # Create a new, re-grounded 'Move' step
                                new_move_step = self.agent.language_of_thought.get_function('Move').body("last_created_object", dx, dy)
                                new_program.append(new_move_step)
                                is_regrounded = True
                                if self.verbose >= 1: print(f"        Re-grounded 'Move' with new parameters: dx={dx}, dy={dy}")
                            else:
                                # Could not re-ground, append the old step and hope for the best
                                new_program.append(step)
                        else:
                            new_program.append(step)
                    else:
                        # This step doesn't need re-grounding, so just copy it.
                        new_program.append(step)

                return new_program if is_regrounded else recalled_program

        return None

    def _strategy_inductive_generation(self, in_graph: ActiveConceptNetGraph, out_graph: ActiveConceptNetGraph, diff: Dict) -> Program | None:
        """
        [DEFINITIVE FIX] Generates theories about abstract rules governing the transformation.
        """
        if self.verbose >= 1: print("    REASONING: Trying Strategy 3 (Inductive/Abductive Generation)...")

        # First, safely get the grid node objects from both input and output graphs.
        in_grid_node = in_graph.nodes.get('grid')
        out_grid_node = out_graph.nodes.get('grid')
        if not in_grid_node or not out_grid_node:
            return None # Cannot reason without grid-level information.

        # Correctly access the .attributes dictionaries
        in_globals = in_grid_node.attributes
        out_globals = out_grid_node.attributes
        
        # --- Find the most common object color in the input ---
        in_colors = [obj.attributes['Color'] for obj in in_graph.nodes.values() if obj.type == 'object']
        if not in_colors: return None
        most_common_in_color = collections.Counter(in_colors).most_common(1)[0][0]

        # --- THEORY 1: Is the output a solid grid of the most common input color? ---
        # A "solid fill" means the output grid has no foreground objects (empty color palette)
        # and its background is the color we hypothesized.
        out_palette = out_globals.get('ColorPalette', [])
        is_solid_fill = (len(out_palette) == 0 and out_globals.get('BackgroundColor') == most_common_in_color)
        
        if is_solid_fill:
            if self.verbose >= 1: print("      Strategy 3 Insight: Transformation is a SOLID FILL with the most common object color.")
            # The argument 'MostCommonColor(InputObjects)' is a placeholder that the executor will resolve.
            return [{'function': 'FillGrid', 'args': ['MostCommonColor(InputObjects)']}]
            
        # --- THEORY 2 (Fallback): Is JUST the background color changing, leaving objects intact? ---
        if out_globals.get('BackgroundColor') == most_common_in_color:
            if self.verbose >= 1: print("      Strategy 3 Insight: Background color changed to most common object color (objects preserved).")
            return [{'function': 'SetGridProperty', 'args': ['BackgroundColor', 'MostCommonColor(InputObjects)']}]
                
        return None


# ---------------------------------------------------------------------------
# Class Definition: SimplifiedOrchOREmulator
# ---------------------------------------------------------------------------
class SimplifiedOrchOREmulator:
    """The complete cognitive agent architecture, integrating all features."""

    def __init__(self, agent_id="agent0", cycle_history_max_len=100,
                 # DEPRECATED PARAMS (kept for signature compatibility if old scripts call it)
                 universe: Optional[Dict] = None,
                 perception_system=None, 
                 # NEW: Deep Learning model configs
                 vae_config: Optional[Dict] = None,
                 world_model_config: Optional[Dict] = None,
                 lifelong_learning_config: Optional[Dict] = None,
                 # Standard Params
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
                 action_space: gym.spaces.Discrete = None,
                 **kwargs): 

        self.agent_id = agent_id
        self.verbose = verbose
        
        # --- Universal Reasoning and Perception Components ---
        self.symbolic_perception_core = SymbolicPerceptionCore()
        self.language_of_thought = LanguageOfThought()
        self.lambda_learner = LambdaLearner(self.language_of_thought)
        self.lambda_learner._log_lot_event = self._log_lot_event
        self.lambda_learner.verbose = self.verbose
        
        # >>> NEW: Pillar 3 - Instantiate the Reasoning Engine
        self.reasoning_engine = MultiStrategyScientificReasoningEngine(self)
        
        # MODIFIED: Share the program history directly with the lambda learner for analogy
        self.solved_arc_programs = self.lambda_learner.program_history

        # --- Initialize Embodied Attributes to None ---
        self.action_space = action_space
        self.visual_cortex = None
        self.world_model = None
        self.experience_replay_buffer = None
        self.vae_params = None
        self.world_model_params = None
        self.ll_params = None
        self.current_action_plan: Deque[int] = collections.deque()
        self.last_perceived_state: Optional[StateHandle] = None 
        self.last_full_obs: Optional[Dict] = None
        self.next_target_input_state_handle: Optional[StateHandle] = None
        self.last_prediction_error: Optional[float] = None
        self.last_imagined_next_state: Optional[np.ndarray] = None
        self.known_goal_latent_vectors: Dict[str, np.ndarray] = {}
        self.known_concept_vectors: Dict[str, np.ndarray] = {}
        self.cycles_stuck_on_step = 0
        

        if vae_config and world_model_config:
            if self.action_space is None:
                raise ValueError("An embodied agent requires an 'action_space' with VAE/WorldModel configs.")
            
            if self.verbose >= 1: print(f"[{self.agent_id}] Initializing Embodied Perception Systems...")
            
            self.vae_params = copy.deepcopy(vae_config)
            self.world_model_params = copy.deepcopy(world_model_config)
            self.ll_params = copy.deepcopy(lifelong_learning_config or DEFAULT_LIFELONG_LEARNING_CONFIG)
            
            # --- FIX #3: THE GIFT OF INSTINCT ---
            # Pre-populate the agent's mind with innate "ideas" of what critical objects look like.
            # We use a random vector as a placeholder. The agent will wander until its perception
            # is "close enough" to this innate idea, at which point it will ground the concept properly.
            # This is the bootstrap for survival and opportunism.
            self.known_concept_vectors['charging_pad'] = np.random.randn(self.vae_params['LATENT_DIM']) * 0.1 
            self.known_concept_vectors['bonus_target'] = np.random.randn(self.vae_params['LATENT_DIM']) * 0.1
            self.known_concept_vectors['target'] = np.random.randn(self.vae_params['LATENT_DIM']) * 0.1
            # --- END OF FIX #3 ---

            self.visual_cortex = VisualCortexVAE(original_dim=(*self.vae_params['IMG_SIZE'], 3), latent_dim=self.vae_params['LATENT_DIM'])
            try:
                self.visual_cortex.build(input_shape=(None, *self.vae_params['IMG_SIZE'], 3))
                self.visual_cortex.load_weights(self.vae_params['MODEL_PATH'])
            except (FileNotFoundError, OSError, ValueError) as e:
                if self.verbose >= 0: print(f"[{self.agent_id}] WARNING: Could not load VAE weights. Error: {e}")
            
            self.world_model = PredictiveWorldModel(latent_dim=self.vae_params['LATENT_DIM'], num_actions=self.action_space.n)
            try:
                self.world_model.build(input_shape=[(None, self.vae_params['LATENT_DIM']), (None, self.action_space.n)])
                self.world_model.load_weights(self.world_model_params['MODEL_PATH'])
            except (FileNotFoundError, OSError, ValueError) as e:
                 if self.verbose >= 0: print(f"[{self.agent_id}] WARNING: Could not load World Model weights. Error: {e}")

            self.experience_replay_buffer = ExperienceReplayBuffer(self.ll_params['replay_buffer_capacity'], self.ll_params['experience_emotion_threshold'], self.ll_params['experience_surprise_threshold'])
        
        # --- Universal Cognitive Components ---
        self.goal_manager = GoalManager(self)
        self.last_arbitrated_goal: Optional[GoalState] = None
        self.universe = copy.deepcopy(TWO_QUBIT_UNIVERSE_CONFIG) if universe is None else universe
        self.state_handle_by_id = {handle.id: handle for handle in self.universe['states']}
        start_comp_basis = self.universe['state_to_comp_basis'][self.universe['start_state']]
        self.logical_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        self.logical_superposition[start_comp_basis] = 1.0 + 0j
        self.collapsed_computational_state_str = start_comp_basis
        self.current_conceptual_state = self.universe['start_state']
        self.internal_thought_result = None

        # --- OrchOR and Valence Mechanics ---
        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate
        self.operation_costs = {'X': 0.1, 'Z': 0.1, 'H': 0.3, 'CNOT': 0.4, 'CZ': 0.4, 'ERROR_PENALTY': 0.05, 'PLANNING_BASE': 0.02}
        self.last_cycle_valence_raw = 0.0
        self.last_cycle_valence_mod = 0.0
        self.current_orp_before_reset = 0.0

        # --- Loading ALL Original Parameter Configurations ---
        self.internal_state_parameters = copy.deepcopy(DEFAULT_INTERNAL_PARAMS) if internal_state_parameters_config is None else copy.deepcopy(internal_state_parameters_config)
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

        # --- Shared and Long-Term Memory ---
        self.long_term_memory = shared_long_term_memory if shared_long_term_memory is not None else {}
        self.shared_attention_foci = shared_attention_foci if shared_attention_foci is not None else collections.deque(maxlen=20)
        self.ltm_utility_weight_valence = 0.6
        self.ltm_utility_weight_efficiency = 0.4
        self.long_term_memory_capacity = 100
        self.successful_sequence_threshold_valence = 0.5 
        
        # --- Internal State Tracking and Temporal Features ---
        self.temporal_feedback_grid = collections.deque(maxlen=self.temporal_grid_params['max_len'])
        self.last_cycle_entropy_for_delta = 0.0
        self.post_goal_valence_lock_cycles_remaining = 0
        self.post_goal_valence_lock_value = 0.2
        self.post_goal_valence_lock_duration = 3

        # --- SMN (Graph) Initialization ---
        self.smn_params_runtime_state = {}
        self.smn_param_indices = {}
        self.smn_param_names_from_indices = {}
        self.smn_influence_matrix = np.array([])
        self.smn_param_actual_changes_this_cycle = {}
        self._initialize_smn_graph_structures()
        self.smn_internal_flags = {}

        # --- Firewall State ---
        self.firewall_cooldown_remaining = 0
        self.firewall_cycles_since_last_check = 0
        
        # --- Working Memory and LoT Stream ---
        self.working_memory = WorkingMemoryStack(max_depth=working_memory_max_depth)
        self.current_cycle_lot_stream = []

        # --- Final Configuration Overrides and Parameter Application ---
        if config_overrides: self._apply_config_overrides(config_overrides)
        if trainable_param_values: self.update_emulator_parameters(trainable_param_values)
            
        if 'strategy_weights' in self.internal_state_parameters:
            for key, default_value in DEFAULT_INTERNAL_PARAMS['strategy_weights'].items():
                if key not in self.internal_state_parameters['strategy_weights']:
                    self.internal_state_parameters['strategy_weights'][key] = default_value
        else:
            self.internal_state_parameters['strategy_weights'] = copy.deepcopy(DEFAULT_INTERNAL_PARAMS['strategy_weights'])
        
        self.cycle_history = collections.deque(maxlen=cycle_history_max_len)
        self.current_cycle_num = 0

    def solve_arc_task(self, task_id: str, task_data: Dict):
        """[DEFINITIVE] Orchestrates perception, reasoning, and execution for ARC."""
        if self.verbose >= 1:
            print(f"\n[{self.agent_id}]--- Starting ARC task: {task_id} ---")

        solution_program: Program | None = None

        # We only need one good training pair to form a theory.
        # Assuming at least one training pair exists
        train_pair = task_data['train'][0]
        input_grid, output_grid = np.array(train_pair['input']), np.array(train_pair['output'])

        # Pillar 1: Perceive the world.
        input_graph = self.symbolic_perception_core.parse(input_grid)
        output_graph = self.symbolic_perception_core.parse(output_grid)
        abstract_diff = ActiveConceptNetGraph.graph_diff(input_graph, output_graph)

        # Pillar 3: Reason to find a program. Pass the raw grid for simulation.
        solution_program = self.reasoning_engine.generate_solution_program(input_graph, output_graph, abstract_diff, input_grid)

        if not solution_program:
            if self.verbose >= 1:
                print("  REASONING FAILED: No program could be synthesized.")
            # Use a list for the program to be consistent with Program type
            solution_program = [{'function': 'UnknownTransform'}]
        else:
            if self.verbose >= 1:
                program_str = ', '.join([p.get('function', 'N/A') for p in solution_program])
                print(f"\n  [Pillar 3] Synthesized Program: [{program_str}]")

        # Post-hoc: Log the confirmed program and its causal diff for future analogical reasoning.
        self.lambda_learner.log_successful_program(
            task_id=task_id,
            program=solution_program,
            diff=abstract_diff
        )

        # Trigger knowledge consolidation periodically.
        if len(self.lambda_learner.program_history) % 3 == 0:
            self.lambda_learner.consolidate_knowledge()

        # Apply the final program to the test case(s).
        test_solutions = [self._execution_engine_run_program(solution_program, np.array(tp['input'])) for tp in task_data.get('test', [])]

        if self.verbose >= 1:
            print(f"--- ARC task finished. ---")
        return test_solutions

    # --- [NEW] Pillar 4: The LoT Execution Engine ---
    def _execution_engine_run_program(self, program: Program, input_grid: np.ndarray) -> np.ndarray:
        """Executes a LoT program on a given input grid to produce an output grid."""
        if self.verbose >= 1:
            program_str = ', '.join([p.get('function', 'N/A') for p in program])
            print(f"  [Pillar 4: Execution] Running program [{program_str}] on test input...")

        # Use a deque as a command queue to handle learned functions
        command_queue = collections.deque(program)
        current_grid = np.copy(input_grid)
        context = {'last_created_object_coords': None, 'background_color': 0}

        if not command_queue or command_queue[0].get('function') == 'UnknownTransform':
            if self.verbose >= 1: print("    EXECUTION SKIPPED: Program is 'UnknownTransform'. Returning input as is.")
            return current_grid

        # Main execution loop using the queue
        while command_queue:
            step = command_queue.popleft()
            func_name = step.get('function', 'NoFunction')
            args = step.get('args', [])

            # Check if this is a learned function
            learned_func = self.language_of_thought.get_function(func_name)
            if learned_func and learned_func.is_learned:
                if self.verbose >= 1: print(f"    Executing learned function '{func_name}' by expanding its body.")
                # Prepend the body of the learned function to the front of the queue
                # Note: This is a simple expansion. A more advanced system would handle argument mapping.
                for learned_step in reversed(learned_func.learned_body):
                    command_queue.appendleft(learned_step)
                continue # Go to the next iteration to execute the first step of the expanded body

            # --- Execute innate functions ---
            try:
                if func_name == 'Copy':
                    current_grid, context = self._execute_copy(current_grid, context, *args)
                elif func_name == 'Move':
                    current_grid, context = self._execute_move(current_grid, context, *args)
                elif func_name == 'SetGridProperty':
                    current_grid, context = self._execute_set_grid_property(current_grid, input_grid, context, *args)
                elif func_name == 'FillGrid':
                    current_grid, context = self._execute_fill_grid(current_grid, input_grid, context, *args)
                else:
                    if self.verbose >= 1: print(f"    EXECUTION WARNING: Unknown function '{func_name}'. Step skipped.")
            except Exception as e:
                if self.verbose >= 1: print(f"    EXECUTION ERROR in function '{func_name}' with args {args}: {e}")
                traceback.print_exc()
                return current_grid

        if self.verbose >= 1:
            print("  [Pillar 4: Execution] Program finished.")
        return current_grid

    def _resolve_object_selector(self, grid: np.ndarray, context: Dict, selector: str) -> Optional[List[Tuple[int, int]]]:
        """Helper to resolve an object selector string into pixel coordinates."""
        if selector == "last_created_object":
            return context.get('last_created_object_coords')

        if selector and "object_with_id" in selector:
            try:
                # Corrected string parsing
                obj_id_str = selector.split('(')[1].split(')')[0]
                perceived_graph = self.symbolic_perception_core.parse(grid)
                target_node = perceived_graph.nodes.get(obj_id_str)
                if target_node:
                    return target_node.attributes.get('Coordinates')
            except Exception as e:
                if self.verbose >= 1:
                    print(f"      ERROR resolving object selector '{selector}': {e}")
                return None
        if self.verbose >= 1:
            print(f"      WARNING could not resolve object selector '{selector}'")
        return None

    def _execute_copy(self, grid: np.ndarray, context: Dict, object_selector: str) -> Tuple[np.ndarray, Dict]:
        """Executor for the 'Copy' function."""
        object_coords = self._resolve_object_selector(grid, context, object_selector)
        if not object_coords:
            if self.verbose >= 1:
                print(f"    EXECUTE_COPY: Could not find object for selector '{object_selector}'.")
            return grid, context

        context['last_created_object_coords'] = object_coords

        if self.verbose >= 2:
            print(f"    EXECUTE_COPY: Object at {object_coords} selected for duplication.")
        return grid, context

    def _execute_move(self, grid: np.ndarray, context: Dict, object_selector: str, dx: int, dy: int) -> Tuple[np.ndarray, Dict]:
        """Executor for the 'Move' function."""
        coords_to_move = self._resolve_object_selector(grid, context, object_selector)
        if not coords_to_move:
            if self.verbose >= 1:
                print(f"    EXECUTE_MOVE: Could not find object for selector '{object_selector}'.")
            return grid, context

        # Assuming coords_to_move is not empty and all pixels have the same color
        color = int(grid[coords_to_move[0]])
        new_coords = []
        is_placing_copy = (object_selector == 'last_created_object')

        if not is_placing_copy:
            for r, c in coords_to_move:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    grid[r, c] = context.get('background_color', 0)

        for r, c in coords_to_move:
            new_r, new_c = r + dy, c + dx
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                grid[new_r, new_c] = color
            new_coords.append((new_r, new_c))

        context['last_created_object_coords'] = new_coords
        if self.verbose >= 2 and coords_to_move and new_coords:
            print(f"    EXECUTE_MOVE: Moved object from {coords_to_move} to {new_coords}.")

        return grid, context

    def _execute_set_grid_property(self, grid: np.ndarray, original_input_grid: np.ndarray, context: Dict, prop_name: str, value_selector: str) -> Tuple[np.ndarray, Dict]:
        """Executor for 'SetGridProperty'."""
        if self.verbose >= 2:
            print(f"    EXECUTE_SET_GRID_PROP: Setting '{prop_name}' to '{value_selector}'.")

        if prop_name == 'BackgroundColor':
            if value_selector == 'MostCommonColor(InputObjects)':
                graph = self.symbolic_perception_core.parse(original_input_grid)
                if not graph: # handle case where parsing fails
                    return grid, context
                colors = [obj.attributes['Color'] for obj in graph.nodes.values() if obj.type == 'object']
                if not colors:
                    if self.verbose >= 1:
                        print("    EXECUTION_WARNING: Cannot set background, no objects in original input to get color from.")
                    return grid, context
                # collections.Counter returns a list of (element, count) tuples
                most_common_color = collections.Counter(colors).most_common(1)[0][0]

                # --- START OF FIX ---
                # 1. Create a new grid filled entirely with the target background color.
                #    We use original_input_grid.shape to handle cases where the grid might have been resized by other operations.
                new_grid = np.full(original_input_grid.shape, most_common_color, dtype=original_input_grid.dtype)

                # 2. Identify the pixels that are NOT background in the *original* input grid.
                #    The standard ARC convention is that color 0 is the background.
                object_mask = (original_input_grid != 0)

                # 3. Use the boolean mask to copy the object pixels from the original grid onto our new grid.
                #    This preserves the foreground objects on the new background.
                new_grid[object_mask] = original_input_grid[object_mask]
                
                # Also update the context's background color for future operations in this program
                context['background_color'] = most_common_color

                return new_grid, context
                # --- END OF FIX ---
            else:
                if self.verbose >= 1:
                    print(f"    EXECUTION_WARNING: Unknown value selector for BackgroundColor: '{value_selector}'.")
        return grid, context
    


    def _execute_fill_grid(self, grid: np.ndarray, original_input_grid: np.ndarray, context: Dict, value_selector: str) -> Tuple[np.ndarray, Dict]:
        """Executor for 'FillGrid' - completely fills the grid with a color."""
        if self.verbose >= 2:
            print(f"    EXECUTE_FILL_GRID: Filling with '{value_selector}'.")
        
        target_color = None
        if value_selector == 'MostCommonColor(InputObjects)':
            # Resolve the symbolic value into a concrete color from the input
            graph = self.symbolic_perception_core.parse(original_input_grid)
            colors = [obj.attributes['Color'] for obj in graph.nodes.values() if obj.type == 'object']
            if not colors:
                if self.verbose >= 1: print("    EXECUTION_WARNING: Cannot fill grid, no objects in original input to determine color.")
                return grid, context
            target_color = collections.Counter(colors).most_common(1)[0][0]
        else:
            # This allows for future value selectors, e.g., FillGrid('5')
            try:
                target_color = int(value_selector)
            except (ValueError, TypeError):
                if self.verbose >= 1: print(f"    EXECUTION_WARNING: Unknown value selector for FillGrid: '{value_selector}'.")
                return grid, context

        if target_color is not None:
            # Create a new grid completely filled with the target color
            new_grid = np.full(grid.shape, target_color, dtype=grid.dtype)
            context['background_color'] = target_color
            return new_grid, context
            
        return grid, context


    
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
    def _log_lot_event(self, event_source: str, event_type: str, details: dict):
        # Configuration check remains similar
        if not self.lot_config_params.get('enabled', False): return
        log_details_config = self.lot_config_params.get('log_level_details', {})
        
        # Check against lowercase keys from config
        source_key_lower = event_source.lower()
        full_key_lower = f"{source_key_lower}.{event_type.lower()}"

        # Split source key if it contains dots for checking, e.g., "executive.opgen"
        source_main_category = source_key_lower.split('.')[0]
        
        # New, simpler check logic based on the command and old config files' structure
        # Log if a specific key `source.type` exists and is true, or if the main category `source` exists and is true
        if not (log_details_config.get(full_key_lower, False) or 
                log_details_config.get(source_key_lower, False) or
                log_details_config.get(source_main_category, False) or
                log_details_config.get(source_main_category+"_ops", False) # Maintain compatibility with keys like `workingmemory_ops`
               ):
             return
        
        # Create and append the structured LogEntry object
        # Sanitize details for logging to avoid excessive length
        sanitized_details = {}
        for k, v in details.items():
            if isinstance(v, (np.ndarray, list, tuple, dict)) and len(v) > 5:
                sanitized_details[k] = f"<{type(v).__name__} of len {len(v)}>"
            elif isinstance(v, str) and len(v) > 70:
                sanitized_details[k] = v[:67] + "..."
            else:
                sanitized_details[k] = v

        log_entry = LogEntry(
            event_source=event_source.upper(),
            event_type=event_type.upper(),
            details=sanitized_details
        )
        self.current_cycle_lot_stream.append(log_entry)

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
            self._log_lot_event("workingmemory", "full_discard", {'reason': f'implicit_discard_for_push_of_{item.type if item else "item"}'})
        
        self._log_lot_event("workingmemory", op_type, log_data)


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

        self._log_lot_event("op_execution", "attempt", {"op":op_char, "arg":logical_arg, "cost":op_cost_val, "cur_orp":current_orp})

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
                 self._log_lot_event("op_execution", "logic_error", {"op":op_char, "arg":logical_arg, "state":basis_state_str})
                 error_occurred = False

        final_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        norm_sq = sum(abs(a)**2 for a in new_superposition.values())
        if norm_sq > 1e-12:
            norm = np.sqrt(norm_sq)
            for state_key, amp_val in new_superposition.items():
                final_superposition[state_key] = amp_val / norm
        else:
            if self.verbose >=1: print(f"[{self.agent_id}] CRITICAL Warning: Superposition norm zero after op '{op_char_upper}'. Resetting to |00>.")
            self._log_lot_event("op_execution", "error", {"op":op_char, "error":"norm_zero_critical_reset_00"})
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
        self._log_lot_event("executive", "super_prep", {"target_state": classical_input_str})

        self.logical_superposition = {"00": 0j, "01": 0j, "10": 0j, "11": 0j}
        if not (len(classical_input_str) == 2 and all(c in '01' for c in classical_input_str)):
            if self.verbose >= 1: print(f"    ERROR: Invalid classical_input_str '{classical_input_str}'. Defaulting to '00'.")
            self._log_lot_event("executive", "super_prep_error", {"input": classical_input_str, "defaulted_to": "00"})
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
        self._log_lot_event("executive", "quantum_comp_start", {"ops_planned_count": len(computation_sequence_ops or []), "orp_start":self.objective_reduction_potential, "decay_rate": self.orp_decay_rate})

        orp_before_decay = self.objective_reduction_potential
        decay_amount = self.objective_reduction_potential * self.orp_decay_rate
        self.objective_reduction_potential = max(0, self.objective_reduction_potential - decay_amount)
        if self.verbose >=3 and decay_amount > 1e-6:
            self._log_lot_event("executive", "quantum_comp_orp_decay", {"before": orp_before_decay, "after": self.objective_reduction_potential, "amount": decay_amount})
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
                    self._log_lot_event("op_execution", "success", {"op_idx":i, "op":op_char, "arg":logical_arg, "orp_change": temp_orp-op_start_orp})
                except ValueError as e:
                    if self.verbose >=1: print(f"    Error applying op ('{op_char}', {logical_arg}): {e}. Skipping.")
                    self._log_lot_event("executive", "quantum_comp_op_error", {"op_idx":i, "op":op_char, "arg":logical_arg, "error":str(e)})
                    temp_orp += self.operation_costs.get('ERROR_PENALTY', 0.05)

                if self.verbose >= 3:
                    active_terms_str = ', '.join([f'{amp:.2f}|{s}>' for s, amp in temp_superposition.items() if abs(amp) > 1e-9])
                    print(f"      After op {i+1} ('{op_char}', {logical_arg}): [{active_terms_str}], ORP: {temp_orp:.3f}")

                if temp_orp >= self.E_OR_THRESHOLD:
                    if self.verbose >= 2:
                        print(f"      >>> OR THRESHOLD REACHED ({temp_orp:.3f} >= {self.E_OR_THRESHOLD:.3f}) after {ops_executed_count} ops. <<<")
                    self._log_lot_event("executive", "quantum_comp_or_early", {"orp":temp_orp, "threshold":self.E_OR_THRESHOLD, "ops_done":ops_executed_count})
                    or_triggered_early = True
                    break
            self.logical_superposition = temp_superposition
            self.objective_reduction_potential = temp_orp

        if self.verbose >= 2:
            print(f"    Final superposition before OR: {self.logical_superposition_str()}, ORP: {self.objective_reduction_potential:.3f}")
        self._log_lot_event("executive", "quantum_comp_end", {"ops_executed_count": len(computation_sequence_ops or []), "final_orp":self.objective_reduction_potential, "early_or":or_triggered_early})
        return True, or_triggered_early

    def _executive_trigger_objective_reduction(self):
        if self.verbose >= 2: print(f"  EXECUTIVE.Objective_Reduction: Collapsing superposition.")
        self._log_lot_event("executive", "objective_reduction_start", {"orp_at_trigger": self.objective_reduction_potential, "superposition_str": self.logical_superposition_str()})

        basis_states = list(self.logical_superposition.keys())
        amplitudes = np.array([self.logical_superposition[s] for s in basis_states], dtype=complex)
        probabilities = np.abs(amplitudes)**2

        sum_probs = np.sum(probabilities)
        if sum_probs < 1e-9:
            if self.verbose >= 1: print("    ERROR: Superposition has near-zero norm before collapse. Defaulting to '00'.")
            self._log_lot_event("executive", "objective_reduction_error", {"error":"norm_zero_collapse_00"})
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
            self._log_lot_event("executive", "objective_reduction_error", {"error": str(e), "probs_str":str(probabilities)})
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

        self._log_lot_event("executive", "objective_reduction_end", {"collapsed_to": self.collapsed_logical_state_str, "orp_experienced":self.current_orp_before_reset})
        return self.collapsed_logical_state_str
    
        # --- ADV_REASONING_FEATURE_1: Conceptual Layer - Activation ---
    def _executive_update_active_concepts(self, collapsed_conceptual_state):
        """Updates the agent's set of active concepts based on the collapsed state."""
        concept_map = self.internal_state_parameters.get('concept_state_handle_map', {})
        if not concept_map:
            if self.active_concepts: # Clear if concepts exist but map is now empty
                 self.active_concepts.clear()
            return

        # Determine if concepts should persist or be refreshed each cycle
        if self.internal_state_parameters.get('clear_active_concepts_each_cycle', True):
            self.active_concepts.clear()

        activated_this_cycle = set()
        for concept_name, state_handle in concept_map.items():
            if state_handle == collapsed_conceptual_state:
                self.active_concepts.add(concept_name)
                activated_this_cycle.add(concept_name)

        if activated_this_cycle:
            if self.verbose >= 2: print(f"  EXECUTIVE.ConceptUpdate: {collapsed_conceptual_state} activated concepts: {activated_this_cycle}")
            self._log_lot_event("executive", "concept_update", {"state": str(collapsed_conceptual_state),
                                                            "activated": list(activated_this_cycle),
                                                            "all_active_now": list(self.active_concepts)})

    # --- Layer 1: Sensor Layer ---
    def _sensor_layer_process_input(self, target_conceptual_input: StateHandle) -> tuple[str, StateHandle]:
        if self.verbose >= 2: print(f"  SENSOR_LAYER: Processing target conceptual input '{target_conceptual_input}'.")
        self._log_lot_event("sensor", "process_input_start", {"target_input_conceptual": str(target_conceptual_input)})

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
                self._log_lot_event("sensor", "process_input_noise_applied", {"original": str(target_conceptual_input), "original_comp_basis": target_classical_input_str, "actual_comp_basis": actual_classical_input_str, "actual_conceptual": str(actual_conceptual_state), "noise_level": noise_level, "flips":num_flips})

        self._log_lot_event("sensor", "process_input_end", {"actual_input_computational": actual_classical_input_str, "actual_input_conceptual": str(actual_conceptual_state)})
        return actual_classical_input_str, actual_conceptual_state

    # --- Layer 2: Associative Layer ---
### NEW VERSION ###

    def _associative_layer_update_ltm(self, op_sequence, raw_valence, orp_cost, entropy_gen, final_collapsed_state, consolidation_factor=1.0,
                                      initial_state_when_sequence_started=None, input_context_when_sequence_started=None):
        if self.verbose >= 2: print(f"  ASSOCIATIVE_LAYER.LTM_Update: Seq {op_sequence if op_sequence else 'NoOps'}, Val={raw_valence:.2f}, ORP={orp_cost:.2f}, Ent={entropy_gen:.2f}, ConsolFactor={consolidation_factor:.2f}")
        self._log_lot_event("associative", "ltm_update_start", {
            "op_seq_len":len(op_sequence or []), "raw_valence":raw_valence, "orp_cost": orp_cost,
            "consol_factor": consolidation_factor, "entropy":entropy_gen,
            "initial_state_ctx": str(initial_state_when_sequence_started),
            "input_ctx": str(input_context_when_sequence_started),
            "outcome_state_ctx": str(final_collapsed_state),
            "active_concepts_for_store": list(self.active_concepts)
        })

        if not op_sequence: return
        seq_tuple = tuple(tuple(op) for op in op_sequence)
        
        final_collapsed_state_str = final_collapsed_state.id
        initial_state_str = initial_state_when_sequence_started.id if initial_state_when_sequence_started else "unknown"
        input_context_str = input_context_when_sequence_started.id if input_context_when_sequence_started else "unknown"

        if raw_valence < self.successful_sequence_threshold_valence * 0.3 and consolidation_factor <= 1.0:
             if self.verbose >=3: print(f"    LTM_Update: Sequence {seq_tuple} not stored, raw_valence {raw_valence:.2f} too low (threshold factor 0.3).")
             self._log_lot_event("associative", "ltm_update_skip_low_valence", {"seq_tuple":seq_tuple, "raw_valence":raw_valence})
             return

        current_goal_name_for_ltm = None
        current_step_name_for_ltm = None
        
        # --- ROBUST CONTEXT FETCHING ---
        # The goal context for abstract thought should come from the currently active arbitrated goal, if any.
        goal = self.last_arbitrated_goal
        if goal: # It's okay if this is None during abstract thought
            active_goal_for_context = goal
            if 0 <= goal.current_step_index < len(goal.steps):
                step_hosting_subgoal = goal.steps[goal.current_step_index]
                potential_sub_goal = step_hosting_subgoal.get("sub_goal")
                if isinstance(potential_sub_goal, GoalState) and potential_sub_goal.status == "active":
                    active_goal_for_context = potential_sub_goal

            if 0 <= active_goal_for_context.current_step_index < len(active_goal_for_context.steps):
                current_goal_name_for_ltm = active_goal_for_context.current_goal if not isinstance(active_goal_for_context.current_goal, StateHandle) else str(active_goal_for_context.current_goal)
                current_step_name_for_ltm = active_goal_for_context.steps[active_goal_for_context.current_step_index].get("name", f"Step_{active_goal_for_context.current_step_index}")
        # --- END ROBUST CONTEXT ---

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
                entry['initial_states_seen'][initial_state_str] += 1
                entry['input_contexts_seen'] = entry.get('input_contexts_seen', collections.Counter())
                entry['input_contexts_seen'][input_context_str] +=1

            entry['concepts_seen_counts'] = entry.get('concepts_seen_counts', collections.Counter())
            for concept in self.active_concepts:
                entry['concepts_seen_counts'][concept] += update_strength
            if entry['count'] % 5 == 0:
                 most_common_concepts_list = [c[0] for c in entry['concepts_seen_counts'].most_common(3)]
                 entry['most_frequent_concepts_at_store'] = most_common_concepts_list

            if random.random() < mutation_rate_store:
                entry['total_valence'] *= (1 + random.uniform(-0.05, 0.05) * update_strength)
                entry['total_orp_cost'] *= (1 + random.uniform(-0.03, 0.03))
                self._log_lot_event("associative", "ltm_update_metric_mutation", {"seq":seq_tuple})

            entry['avg_valence'] = entry['total_valence'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_orp_cost'] = entry['total_orp_cost'] / entry['count'] if entry['count'] > 0 else 0
            entry['avg_entropy'] = entry['total_entropy_generated'] / entry['count'] if entry['count'] > 0 else 0

        else: # New LTM entry
            if len(self.long_term_memory) >= self.long_term_memory_capacity:
                if not self.long_term_memory: return
                min_prune_score = float('inf'); key_to_prune = None
                for k, v_data in self.long_term_memory.items():
                    prune_score = v_data.get('utility', 0) - v_data.get('confidence', 0) * 0.2
                    if prune_score < min_prune_score:
                        min_prune_score = prune_score
                        key_to_prune = k
                if key_to_prune:
                    if self.verbose >=3: print(f"    LTM_Update: LTM full. Pruning {key_to_prune} (prune_score {min_prune_score:.2f}).")
                    self._log_lot_event("associative", "ltm_update_prune", {"pruned_seq_str":str(key_to_prune), "prune_score":min_prune_score})
                    del self.long_term_memory[key_to_prune]
                elif self.verbose >=2: print("    LTM_Update: LTM full, but no suitable key to prune found.")

            if len(self.long_term_memory) < self.long_term_memory_capacity:
                current_raw_valence_store = raw_valence * consolidation_factor
                current_orp_cost_store = orp_cost
                current_entropy_store = entropy_gen

                if random.random() < mutation_rate_store:
                    current_raw_valence_store *= (1 + random.uniform(-0.05, 0.05) * update_strength)
                    current_orp_cost_store *= (1 + random.uniform(-0.03, 0.03))
                    self._log_lot_event("associative", "ltm_update_new_metric_mutation", {"seq":seq_tuple})

                new_entry = {
                    'type': 'abstract_op_sequence',
                    'count': update_strength,
                    'total_valence': current_raw_valence_store, 'avg_valence': current_raw_valence_store / update_strength if update_strength else current_raw_valence_store,
                    'total_orp_cost': current_orp_cost_store * update_strength, 'avg_orp_cost': current_orp_cost_store,
                    'total_entropy_generated': current_entropy_store * update_strength, 'avg_entropy': current_entropy_store,
                    'first_cycle': self.current_cycle_num, 'last_cycle': self.current_cycle_num,
                    'first_goal_context_name': current_goal_name_for_ltm, 'first_goal_context_step': current_step_name_for_ltm,
                    'last_goal_context_name': current_goal_name_for_ltm, 'last_goal_context_step': current_step_name_for_ltm,
                    'goal_context_counts': {},
                    'initial_states_seen': collections.Counter({initial_state_str: update_strength}),
                    'input_contexts_seen': collections.Counter({input_context_str: update_strength}),
                    'most_frequent_initial_state': initial_state_str,
                    'most_frequent_input_context': input_context_str,
                    'final_outcome_states': collections.Counter({final_collapsed_state_str: update_strength}),
                    'most_frequent_outcome_state': final_collapsed_state_str,
                    'concepts_seen_counts': collections.Counter(self.active_concepts),
                    'most_frequent_concepts_at_store': list(self.active_concepts),
                    'confidence': 0.0,
                    'last_successful_use_cycle': self.current_cycle_num if raw_valence > 0 else -1
                }
                if current_goal_name_for_ltm:
                     context_key_new = (current_goal_name_for_ltm, current_step_name_for_ltm)
                     new_entry['goal_context_counts'][context_key_new] = update_strength

                self.long_term_memory[seq_tuple] = new_entry
                log_extra = { "goal_name_ctx":current_goal_name_for_ltm or "N/A", "initial_state_ctx_new":initial_state_str, "input_ctx_new": input_context_str, "active_concepts": list(self.active_concepts) }
                if self.verbose >=3: print(f"    LTM_Update: Added new sequence {seq_tuple} with avg_valence {new_entry['avg_valence']:.2f}, Contexts: {log_extra}.")
                self._log_lot_event("associative", "ltm_update_new_entry", {"seq_str":str(seq_tuple), "val":new_entry['avg_valence'], **log_extra})

        if seq_tuple in self.long_term_memory:
             entry_ref = self.long_term_memory[seq_tuple]
             entry_ref['utility'] = self._associative_layer_calculate_ltm_entry_utility(entry_ref)
             entry_ref['last_cycle'] = self.current_cycle_num
             familiarity_score = np.tanh(entry_ref['count'] / 20.0)
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
                                                     current_conceptual_state_for_recall_context,
                                                     current_input_conceptual_context_for_recall):
        if not self.long_term_memory:
            exec_thought_log.append("LTM recall: LTM empty.")
            return None, current_orp_value

        current_state_str = current_conceptual_state_for_recall_context.id
        current_input_str = current_input_conceptual_context_for_recall.id

        min_utility_for_recall = 0.05
        candidate_info = []

        active_recall_goal_name = None
        active_recall_step_name = None

        # --- THIS IS THE CORRECT, ROBUST LOGIC ---
        # It's okay for the agent to be "daydreaming" without an active goal.
        # We only try to get goal context if a goal is actively being pursued.
        goal_for_context = self.last_arbitrated_goal
        if goal_for_context and goal_for_context.status == "active":
            temp_goal_for_context = goal_for_context
            if 0 <= temp_goal_for_context.current_step_index < len(temp_goal_for_context.steps):
                potential_sub_goal_in_step = temp_goal_for_context.steps[temp_goal_for_context.current_step_index].get("sub_goal")
                if isinstance(potential_sub_goal_in_step, GoalState) and potential_sub_goal_in_step.status == "active":
                    temp_goal_for_context = potential_sub_goal_in_step

            if 0 <= temp_goal_for_context.current_step_index < len(temp_goal_for_context.steps):
                active_recall_goal_name = temp_goal_for_context.current_goal if not isinstance(temp_goal_for_context.current_goal, StateHandle) else str(temp_goal_for_context.current_goal)
                active_recall_step_name = temp_goal_for_context.steps[temp_goal_for_context.current_step_index].get("name", f"Step_{temp_goal_for_context.current_step_index}")
        # --- END OF CORRECTED LOGIC ---

        goal_ctx_bonus_val = self.internal_state_parameters.get('ltm_goal_context_match_bonus', 0.15)
        initial_state_bonus_val = self.internal_state_parameters.get('ltm_initial_state_match_bonus', 0.10)
        input_ctx_bonus_val = self.internal_state_parameters.get('ltm_input_context_match_bonus', 0.05)
        concept_match_bonus_val = self.internal_state_parameters.get('ltm_active_concept_match_bonus', 0.12)

        for seq_tuple, data in self.long_term_memory.items():
            if data.get('type') != 'abstract_op_sequence':
                continue

            base_utility = data.get('utility', self._associative_layer_calculate_ltm_entry_utility(data))
            current_effective_utility = base_utility
            applied_bonuses_detail = {'goal': 0.0, 'initial_state': 0.0, 'input_ctx': 0.0, 'concept': 0.0}

            # This check now safely handles the case where there is no active goal name
            if active_recall_goal_name and data.get('last_goal_context_name') == active_recall_goal_name:
                bonus_val_for_goal_match = goal_ctx_bonus_val * 0.5
                if data.get('last_goal_context_step') == active_recall_step_name:
                    bonus_val_for_goal_match = goal_ctx_bonus_val
                current_effective_utility += bonus_val_for_goal_match
                applied_bonuses_detail['goal'] = bonus_val_for_goal_match

            if data.get('most_frequent_initial_state') == current_state_str:
                current_effective_utility += initial_state_bonus_val
                applied_bonuses_detail['initial_state'] = initial_state_bonus_val

            if data.get('most_frequent_input_context') == current_input_str:
                current_effective_utility += input_ctx_bonus_val
                applied_bonuses_detail['input_ctx'] = input_ctx_bonus_val

            if concept_match_bonus_val > 0 and self.active_concepts:
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
                if current_orp_value + projected_cost < self.E_OR_THRESHOLD * 1.15:
                    candidate_info.append( (list(seq_tuple), current_effective_utility, applied_bonuses_detail, data) )

        if not candidate_info:
            exec_thought_log.append(f"LTM recall: No sequences found with effective_utility > {min_utility_for_recall} (after context bonuses) or all too costly from ORP {current_orp_value:.2f}.")
            return None, current_orp_value

        candidate_sequences = [c[0] for c in candidate_info]
        weights = [c[1]**2.5 for c in candidate_info]

        sum_weights = sum(weights)
        if sum_weights <= 1e-6:
             exec_thought_log.append("LTM recall: No LTM sequences with positive utility weights after filtering.")
             return None, current_orp_value

        normalized_weights = [w / sum_weights for w in weights]
        try:
            chosen_index = random.choices(range(len(candidate_sequences)), weights=normalized_weights, k=1)[0]
        except ValueError as e:
            exec_thought_log.append(f"LTM recall: Error in weighted choice ({e}). Defaulting to highest utility.")
            if candidate_info:
                chosen_index = max(range(len(candidate_info)), key=lambda i: candidate_info[i][1])
            else: return None, current_orp_value

        chosen_sequence_ops_orig_list = candidate_info[chosen_index][0]
        chosen_sequence_ops_mutable = [list(op) for op in chosen_sequence_ops_orig_list]
        
        bonuses_applied_for_chosen = candidate_info[chosen_index][2]
        original_ltm_data_for_chosen = candidate_info[chosen_index][3]
        
        mutation_rate_replay = self.internal_state_parameters.get('ltm_mutation_on_replay_rate', 0.0)
        if chosen_sequence_ops_mutable and random.random() < mutation_rate_replay:
            idx_to_mutate = random.randrange(len(chosen_sequence_ops_mutable))
            op_char, op_arg = chosen_sequence_ops_mutable[idx_to_mutate]
            original_op_tuple_str_in_seq = f"('{op_char}', {op_arg})"

            mutation_type_rand = random.random()
            if mutation_type_rand < 0.35 and op_char in ['X', 'Z', 'H'] and isinstance(op_arg, int):
                chosen_sequence_ops_mutable[idx_to_mutate][1] = 1 - op_arg
            elif mutation_type_rand < 0.65:
                compatible_ops={'X':['Z','H'],'Z':['X','H'],'H':['X','Z'],'CNOT':['CZ'],'CZ':['CNOT']}
                new_op_char = random.choice(compatible_ops.get(op_char, ['X','Z','H']))
                new_op_arg = op_arg
                if new_op_char in ['X','Z','H']: new_op_arg = random.randint(0,1)
                elif new_op_char in ['CNOT', 'CZ']:
                    new_op_arg = tuple(random.sample([0,1],2)) if op_char not in ['CNOT','CZ'] else op_arg
                chosen_sequence_ops_mutable[idx_to_mutate] = [new_op_char, new_op_arg]
            elif len(chosen_sequence_ops_mutable) > 1 and random.random() < 0.5 :
                del chosen_sequence_ops_mutable[idx_to_mutate]
            else:
                new_op_insert_char = random.choice(['X','Z','H'])
                new_op_insert_arg = random.randint(0,1)
                chosen_sequence_ops_mutable.insert(random.randint(0, len(chosen_sequence_ops_mutable)), [new_op_insert_char, new_op_insert_arg])

            exec_thought_log.append(f"LTM Replay MUTATION: Op {original_op_tuple_str_in_seq} in original seq {chosen_sequence_ops_orig_list} -> seq potentially modified to {chosen_sequence_ops_mutable}.")
            self._log_lot_event("associative", "ltm_recall_mutation", {"original_seq_str": str(chosen_sequence_ops_orig_list), "mutated_seq_str": str(chosen_sequence_ops_mutable)})

        projected_orp_increase_final = sum(self.operation_costs.get(op_data[0].upper(), 0.05) for op_data in chosen_sequence_ops_mutable)
        if current_orp_value + projected_orp_increase_final >= self.E_OR_THRESHOLD * 1.1 and len(chosen_sequence_ops_mutable) > 0 :
            exec_thought_log.append(f"LTM recall: Mutated/Chosen seq {chosen_sequence_ops_mutable} too costly. ORP would be {current_orp_value + projected_orp_increase_final:.2f}. Skipped.")
            return None, current_orp_value

        final_chosen_ops_as_tuples = [tuple(op) for op in chosen_sequence_ops_mutable]

        bonus_summary_str = f"GoalCtx:{bonuses_applied_for_chosen['goal']:.2f},StateCtx:{bonuses_applied_for_chosen['initial_state']:.2f},InputCtx:{bonuses_applied_for_chosen['input_ctx']:.2f},ConceptCtx:{bonuses_applied_for_chosen['concept']:.2f}"
        exec_thought_log.append(f"LTM recall: Replaying {final_chosen_ops_as_tuples} (orig_avg_V={original_ltm_data_for_chosen['avg_valence']:.2f}, base_util={original_ltm_data_for_chosen['utility']:.2f}, bonuses_sum={sum(bonuses_applied_for_chosen.values()):.2f} [{bonus_summary_str}]). Cost {projected_orp_increase_final:.2f}")
        self._log_lot_event("associative", "ltm_recall_chosen", {
            "seq_str":str(final_chosen_ops_as_tuples),
            "orig_util":original_ltm_data_for_chosen['utility'],
            "applied_bonuses_sum": sum(bonuses_applied_for_chosen.values()),
            "bonuses_detail_str": bonus_summary_str,
            "current_state_ctx_match_val": current_state_str,
            "current_input_ctx_match_val": current_input_str,
            "goal_context_name_at_recall": active_recall_goal_name or "N/A",
            "goal_context_step_at_recall": active_recall_step_name or "N/A",
            "active_concepts_at_recall": list(self.active_concepts)
            })
        return final_chosen_ops_as_tuples, current_orp_value


    # --- Layer 3: Executive Layer (Decision Making, Planning, Conscious Experience) ---
    def _executive_evaluate_outcome_and_update_mood(self, logical_outcome, orp_at_collapse, entropy_at_collapse, num_ops_executed_this_cycle):
        if self.verbose >= 2: print(f"  EXECUTIVE_LAYER.Outcome_Eval: {logical_outcome}, ORP={orp_at_collapse:.3f}, Ent={entropy_at_collapse:.2f}, Ops#={num_ops_executed_this_cycle}")
        acc_thoughts_log = []

        raw_valence = self.universe['valence_map'].get(logical_outcome, -0.15)
        mod_valence = raw_valence 

        if self.post_goal_valence_lock_cycles_remaining > 0:
            original_raw_valence_before_lock = raw_valence
            original_mod_valence_before_lock = mod_valence
            
            raw_valence = self.post_goal_valence_lock_value 
            mod_valence = self.post_goal_valence_lock_value
            self.post_goal_valence_lock_cycles_remaining -= 1
            
            lock_msg = f"Post-goal valence lock ACTIVE. Valences (raw/mod) set to {mod_valence:.2f}. Cycles remaining: {self.post_goal_valence_lock_cycles_remaining}."
            acc_thoughts_log.append(lock_msg)
            if self.verbose >= 1: print(f"    EXECUTIVE.Outcome_Eval: {lock_msg}")
            self._log_lot_event("executive", "outcome_eval_post_goal_lock_active", {
                "locked_valence": mod_valence, 
                "cycles_left": self.post_goal_valence_lock_cycles_remaining,
                "original_raw_val_b4_lock": original_raw_valence_before_lock,
                "original_mod_val_b4_lock": original_mod_valence_before_lock
            })
        else: 
            acc_thoughts_log.append(f"Raw val for {logical_outcome} is {raw_valence:.2f}.")
            orp_surprise_factor = 0.20
            if orp_at_collapse < self.E_OR_THRESHOLD * 0.35:
                penalty = orp_surprise_factor * (abs(raw_valence) if raw_valence != 0 else 0.25)
                mod_valence -= penalty
                acc_thoughts_log.append(f"Early OR collapse, val modified by {-penalty:.2f}.")
            elif orp_at_collapse > self.E_OR_THRESHOLD * 1.35 and num_ops_executed_this_cycle > 0:
                late_factor = -0.08 if raw_valence < 0 else 0.08
                mod_valence += late_factor
                acc_thoughts_log.append(f"Late OR collapse, val modified by {late_factor:.2f}.")

            current_preferred_state = self.internal_state_parameters.get('preferred_state_handle')
            if current_preferred_state is not None and current_preferred_state == logical_outcome:
                preference_bonus = 0.30 * (1.0 - abs(mod_valence)) 
                mod_valence += preference_bonus
                acc_thoughts_log.append(f"Preferred state {current_preferred_state} met, val boosted by {preference_bonus:.2f}.")

        # --- CORRECTED GOAL PROGRESS LOGIC ---
        active_goal_for_eval = self.last_arbitrated_goal
        if active_goal_for_eval and active_goal_for_eval.status == "active":
             # During an abstract cycle, the `logical_outcome` is the current conceptual state.
             self._executive_update_goal_progress(active_goal_for_eval, logical_outcome)

        if self.post_goal_valence_lock_cycles_remaining == self.post_goal_valence_lock_duration: 
             if mod_valence != self.post_goal_valence_lock_value :
                acc_thoughts_log.append(f"Re-applying post-goal valence lock ({self.post_goal_valence_lock_value:.2f}) immediately after goal completion logic altered valence.")
                mod_valence = self.post_goal_valence_lock_value
        
        mod_valence = np.clip(mod_valence, -1.0, 1.0) 
        self.last_cycle_valence_raw = raw_valence
        self.last_cycle_valence_mod = mod_valence
        acc_thoughts_log.append(f"Final val (raw/mod): {self.last_cycle_valence_raw:.2f}/{self.last_cycle_valence_mod:.2f}.")
        self._log_lot_event("executive", "outcome_eval_valence", {"raw":self.last_cycle_valence_raw, "mod":self.last_cycle_valence_mod, "outcome_state":str(logical_outcome), "orp_collapse": orp_at_collapse, "post_goal_lock_active_this_eval": self.post_goal_valence_lock_cycles_remaining > 0})

        current_mood = self.internal_state_parameters['mood']
        mood_inertia = 0.88
        valence_influence_on_mood = 0.28
        new_mood = current_mood * mood_inertia + self.last_cycle_valence_mod * valence_influence_on_mood
        self.internal_state_parameters['mood'] = np.clip(new_mood, -1.0, 1.0)
        acc_thoughts_log.append(f"Mood updated from {current_mood:.2f} to {self.internal_state_parameters['mood']:.2f}.")
        self._log_lot_event("executive", "outcome_eval_mood", {"new_mood":self.internal_state_parameters['mood'], "old_mood": current_mood})

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
                self._log_lot_event("executive", "outcome_eval_exploration_end", {})

        if self.internal_state_parameters['frustration'] >= frustration_threshold and \
           self.internal_state_parameters['exploration_mode_countdown'] == 0:
            if self.verbose >= 1: print(f"[{self.agent_id}] High frustration ({self.internal_state_parameters['frustration']:.2f}) triggered exploration mode!")
            self._log_lot_event("executive", "outcome_eval_exploration_start", {"frustration":self.internal_state_parameters['frustration'], "threshold": frustration_threshold})
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
            if self.verbose >= 2:
                print(f"  EXECUTIVE_LAYER.OpGen: Using externally provided ops: {ops_provided_externally}")
            self._log_lot_event("executive", "opgen_external", {"ops_count": len(ops_provided_externally)})
            return ops_provided_externally, "StrategyProvidedExternal", ["Ops provided externally."]

        exec_thought_log = ["OpGen: Generating new computation sequence:"]
        self._log_lot_event("executive", "opgen_start", {"orp_current": self.objective_reduction_potential, "threshold": self.E_OR_THRESHOLD})

        ops_sequence = []
        chosen_strategy_name = "NoOpsMethod"
        simulated_orp_accumulator = self.objective_reduction_potential

        effective_attention = self.internal_state_parameters['attention_level']
        cognitive_load_factor = 1.0 - (self.internal_state_parameters['cognitive_load'] * 0.65)
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * cognitive_load_factor * effective_attention, scale=1.0)))
        num_ops_target = min(num_ops_target, 10)
        exec_thought_log.append(f"  Target ops: ~{num_ops_target} (base:{num_ops_target_base}, load_factor:{cognitive_load_factor:.2f}, attn:{effective_attention:.2f}).")

        current_strategy_weights = self.internal_state_parameters['strategy_weights'].copy()

        # --- NEW ROBUST GOAL HANDLING ---
        # The abstract cycle uses the same `last_arbitrated_goal` as the physical one.
        # It's okay if this is None; the agent is simply "daydreaming" without a specific focus.
        is_creative_generation_mode = False
        goal_for_context = self.last_arbitrated_goal

        if goal_for_context and (goal_for_context.reasoning_heuristic == "CREATIVE_GENERATION" or goal_for_context.evaluation_criteria == "NOVELTY"):
            is_creative_generation_mode = True
            chosen_strategy_name = "StrategyCreativeGeneration"
            exec_thought_log.append("  OpGen Mode: CREATIVE_GENERATION primed by GoalState.")
            self._log_lot_event("executive", "opgen_creative_mode", {"heuristic": goal_for_context.reasoning_heuristic, "eval": goal_for_context.evaluation_criteria})

            if goal_for_context.focus_concepts:
                exec_thought_log.append(f"    Focusing creative thought on concepts: {[str(c) for c in goal_for_context.focus_concepts]}")
                for concept_handle in goal_for_context.focus_concepts:
                    concept_bit_str = self.universe['state_to_comp_basis'].get(concept_handle)
                    if not concept_bit_str:
                        continue
                    if len(concept_bit_str) == 2:
                        ops_sequence.append(['H', 1])
                        ops_sequence.append(['H', 0])
                if len(goal_for_context.focus_concepts) > 1 and len(ops_sequence) > 0:
                    ops_sequence.append(['CNOT', (1, 0)])
            else:
                for _ in range(num_ops_target):
                    op_c = random.choice(['H', 'H', 'Z', 'CNOT', 'CZ'])
                    op_a = random.randint(0, 1) if op_c in ['Z', 'H'] else tuple(random.sample([0, 1], 2))
                    ops_sequence.append([op_c, op_a])

            was_novel_sequence = True

        # --- Standard, non-creative thought process ---
        if not is_creative_generation_mode:
            for key in DEFAULT_INTERNAL_PARAMS['strategy_weights']:
                if key not in current_strategy_weights:
                    current_strategy_weights[key] = 0.001
            valid_weights = {k: v for k, v in current_strategy_weights.items() if isinstance(v, (int, float))}
            total_weight = sum(w for w in valid_weights.values() if w > 0)
            
            if total_weight <= 1e-6:
                strategy_choices, strategy_probs = ['curiosity'], [1.0]
            else:
                strategy_choices, strategy_probs = zip(*[(k, v / total_weight) for k, v in valid_weights.items()])
            
            selected_strategy = random.choices(strategy_choices, weights=strategy_probs, k=1)[0]
            exec_thought_log.append(f"  Selected primary strategy: {selected_strategy}")

            was_novel_sequence = False
            if selected_strategy == 'memory':
                replay_ops, _ = self._associative_layer_recall_from_ltm_strategy(
                    simulated_orp_accumulator, exec_thought_log,
                    self.current_conceptual_state, self.current_conceptual_state
                )
                if replay_ops:
                    ops_sequence = replay_ops
                    chosen_strategy_name = "StrategyLTMReplay"
            
            if not ops_sequence:
                exec_thought_log.append(f"  Using Fallback op generation loop ({selected_strategy}).")
                chosen_strategy_name = f"StrategyFallbackLoop_{selected_strategy}"
                was_novel_sequence = True
                for _ in range(num_ops_target):
                    op_c = random.choice(['X', 'Z', 'H', 'CNOT', 'CZ'])
                    op_a = random.randint(0, 1) if op_c in ['X', 'Z', 'H'] else tuple(random.sample([0, 1], 2))
                    op_cost = self.operation_costs.get(op_c.upper(), 0.05)
                    if simulated_orp_accumulator + op_cost < self.E_OR_THRESHOLD * 0.98:
                        ops_sequence.append([op_c, op_a])
                        simulated_orp_accumulator += op_cost
                    else:
                        break

        # --- Final Checks (Counterfactual Sim, etc.) ---
        if ops_sequence and was_novel_sequence and self.internal_state_parameters.get('enable_counterfactual_simulation', True):
            sim_reject_thresh = self.internal_state_parameters.get('counterfactual_sim_reject_threshold', -0.1)
            sim_result = self._reasoning_simulate_counterfactual(ops_sequence, exec_thought_log)

            effective_reject_thresh = sim_reject_thresh if not is_creative_generation_mode else sim_reject_thresh * 2.0

            if sim_result['is_valid'] and sim_result['estimated_valence'] < effective_reject_thresh:
                exec_thought_log.append(f"    CounterfactualSim REJECTED plan. Est. valence {sim_result['estimated_valence']:.2f} < {effective_reject_thresh:.2f}. Wiping ops.")
                ops_sequence = []
                chosen_strategy_name += "_RejectedBySim"
            else:
                chosen_strategy_name += "_VerifiedBySim"

        if not ops_sequence and chosen_strategy_name == "NoOpsMethod":
            chosen_strategy_name = "NoOpsGenerated"

        self._log_lot_event("executive", "opgen_end", {"ops_generated_count": len(ops_sequence), "strategy": chosen_strategy_name})
        return ops_sequence, chosen_strategy_name, exec_thought_log

    # ------------------------------------------------------------------------------------------
    # --- Advanced Reasoning & Planning Engine (Hierarchical, Analogical, Counterfactual) ---
    # ------------------------------------------------------------------------------------------

    def _reasoning_simulate_counterfactual(self, ops_sequence_to_test, exec_thought_log):
        if not ops_sequence_to_test:
            return {'estimated_valence': 0.0, 'estimated_orp': self.objective_reduction_potential, 'is_valid': False}

        if self.verbose >= 2: exec_thought_log.append(f"  CounterfactualSim: Testing sequence {ops_sequence_to_test}")
        self._log_lot_event("reasoning", "counterfactual_sim_start", {"ops_count": len(ops_sequence_to_test), "start_orp": self.objective_reduction_potential})

        try:
            sim_superposition = copy.deepcopy(self.logical_superposition)
            sim_orp = self.objective_reduction_potential

            for op_char, op_arg in ops_sequence_to_test:
                sim_superposition, sim_orp = self._apply_logical_op_to_superposition(op_char, op_arg, sim_superposition, sim_orp)

            probabilities = {state: abs(amp)**2 for state, amp in sim_superposition.items()}
            prob_sum = sum(probabilities.values())
            if prob_sum > 1e-9:
                estimated_valence = sum(self.universe['valence_map'].get(self.universe['comp_basis_to_state'].get(state, self.universe['start_state']), 0.0) * (prob / prob_sum) for state, prob in probabilities.items())
            else:
                estimated_valence = -1.0 

            exec_thought_log.append(f"    CounterfactualSim Result: Est. Valence={estimated_valence:.3f}, Est. ORP={sim_orp:.3f}")
            self._log_lot_event("reasoning", "counterfactual_sim_result", {"est_valence": estimated_valence, "est_orp": sim_orp})
            return {'estimated_valence': estimated_valence, 'estimated_orp': sim_orp, 'is_valid': True}

        except Exception as e:
            exec_thought_log.append(f"    CounterfactualSim ERROR: {e}")
            self._log_lot_event("reasoning", "counterfactual_sim_error", {"error_str": str(e)})
            return {'estimated_valence': -1.0, 'estimated_orp': self.objective_reduction_potential, 'is_valid': False}


    def _advanced_planning_find_analogous_solution(self, current_goal_step, current_state_handle, exec_thought_log):
        target_state_handle = current_goal_step.get("target_state")
        if not target_state_handle or not self.long_term_memory:
            return None

        current_state_str = current_state_handle.id
        target_state_str = target_state_handle.id

        self._log_lot_event("reasoning", "analogical_planning_start", {"current_state": current_state_str, "target_state": target_state_str, "goal_step": current_goal_step.get('name')})
        exec_thought_log.append(f"  AnalogicalPlanning: Searching LTM for path |{current_state_str}> -> |{target_state_str}>")

        candidates = []
        for seq_tuple, data in self.long_term_memory.items():
            initial_state_seen = data.get('most_frequent_initial_state')
            outcome_state_seen = data.get('most_frequent_outcome_state')
            if not initial_state_seen or not outcome_state_seen:
                continue

            initial_dist = sum(c1 != c2 for c1, c2 in zip(current_state_str, initial_state_seen))
            outcome_dist = sum(c1 != c2 for c1, c2 in zip(target_state_str, outcome_state_seen))

            initial_similarity = 1.0 - (initial_dist / 2.0)
            outcome_similarity = 1.0 - (outcome_dist / 2.0)

            similarity_score = (initial_similarity * 0.4 + outcome_similarity * 0.6)
            final_score = similarity_score * data.get('utility', 0.0)

            if final_score > self.internal_state_parameters['analogical_planning_similarity_threshold']:
                 candidates.append({'seq': list(seq_tuple), 'score': final_score, 'data': data})

        if not candidates:
            exec_thought_log.append("    AnalogicalPlanning: No suitable analogous sequences found in LTM.")
            self._log_lot_event("reasoning", "analogical_planning_fail", {"reason": "no_candidates_above_threshold"})
            return None

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_analogous_solution = candidates[0]
        chosen_seq = best_analogous_solution['seq']
        projected_cost = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in chosen_seq)

        if self.objective_reduction_potential + projected_cost >= self.E_OR_THRESHOLD:
            exec_thought_log.append(f"    AnalogicalPlanning: Best candidate seq {chosen_seq} too costly (cost: {projected_cost:.2f}). Skipped.")
            self._log_lot_event("reasoning", "analogical_planning_fail", {"reason": "best_candidate_too_costly", "cost": projected_cost})
            return None

        exec_thought_log.append(f"    AnalogicalPlanning: Found analogous seq {chosen_seq} with score {best_analogous_solution['score']:.3f}. Applying it.")
        self._log_lot_event("reasoning", "analogical_planning_success", {"chosen_seq": str(chosen_seq), "score": best_analogous_solution['score']})
        return chosen_seq


    def _advanced_planning_breakdown_goal_hierarchically(self, parent_goal, parent_step_idx, exec_thought_log):
        if not (0 <= parent_step_idx < len(parent_goal.steps)):
            return False

        current_step_obj = parent_goal.steps[parent_step_idx]
        
        if current_step_obj.get("sub_goal"):
            exec_thought_log.append("  HierarchicalPlanning: Skipped breakdown, step already contains a sub-goal.")
            return False

        target_state_handle = current_step_obj.get("target_state")
        current_state_handle = self.current_conceptual_state
        if not target_state_handle or target_state_handle == current_state_handle:
            return False

        current_state_str = current_state_handle.id
        target_state_str = target_state_handle.id

        self._log_lot_event("reasoning", "hierarchical_planning_start", {"current_state": current_state_str, "target_state": target_state_str, "goal_step": current_step_obj.get('name')})

        hamm_dist = sum(c1 != c2 for c1, c2 in zip(current_state_str, target_state_str))
        landmark_state_handle = None
        if hamm_dist > 1:
            for i in range(len(current_state_str)):
                temp_list = list(current_state_str)
                temp_list[i] = '1' if temp_list[i] == '0' else '0'
                potential_landmark_str = "".join(temp_list)
                landmark_to_target_dist = sum(c1 != c2 for c1, c2 in zip(potential_landmark_str, target_state_str))
                if landmark_to_target_dist < hamm_dist:
                    landmark_state_handle = self.universe['comp_basis_to_state'].get(potential_landmark_str)
                    break

        if not landmark_state_handle:
            exec_thought_log.append("  HierarchicalPlanning: Could not determine a useful landmark state.")
            self._log_lot_event("reasoning", "hierarchical_planning_fail", {"reason": "no_landmark_found"})
            return False

        exec_thought_log.append(f"  HierarchicalPlanning: Breaking down step '{current_step_obj['name']}'. New landmark: {landmark_state_handle}.")
        self._log_lot_event("reasoning", "hierarchical_planning_success", {"landmark_state": str(landmark_state_handle)})

        sub_goal_step1 = {
            "name": f"Sub-goal: Reach landmark {landmark_state_handle}",
            "target_state": landmark_state_handle,
            "max_cycles_on_step": 5
        }
        original_step_as_sub_step2 = copy.deepcopy(current_step_obj)
        original_step_as_sub_step2['name'] = f"Sub-goal: Final step to {target_state_handle}"
        if 'next_ops_hint' in original_step_as_sub_step2:
             del original_step_as_sub_step2['next_ops_hint']

        sub_goal = GoalState(
            current_goal=f"Sub-goal for '{current_step_obj.get('name')}'",
            steps=[sub_goal_step1, original_step_as_sub_step2]
        )

        new_parent_step = {
            "name": f"Execute Sub-Goal for {target_state_handle}",
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

        all_states = self.universe['states']
        try:
            current_index = all_states.index(current_outcome_handle)
            next_index = (current_index + 1) % len(all_states)
            next_handle = all_states[next_index]
        except (ValueError, IndexError):
            next_handle = self.universe['start_state']
        exec_thought_log.append(f"  Base heuristic next input: {next_handle}.")

        # --- CORRECTED TO USE LAST ARBITRATED GOAL ---
        active_goal = self.last_arbitrated_goal
        if active_goal and active_goal.status == "active":
            current_step_idx = active_goal.current_step_index
            if 0 <= current_step_idx < len(active_goal.steps):
                step_info = active_goal.steps[current_step_idx]
                if isinstance(step_info.get("next_input_for_world"), StateHandle):
                    next_handle = step_info["next_input_for_world"]
                    exec_thought_log.append(f"  GoalStep '{step_info.get('name')}' overrides next input to {next_handle}.")
                    self._log_lot_event("executive", "plannext_goal_override", {"next_input": str(next_handle), "goal_step_name":step_info.get('name',"")})

        elif self.internal_state_parameters['preferred_state_handle'] and \
           self.internal_state_parameters['preferred_state_handle'] != next_handle and \
           random.random() < self.internal_state_parameters['goal_seeking_bias'] * 0.75:
            next_handle = self.internal_state_parameters['preferred_state_handle']
            exec_thought_log.append(f"  Overridden by PreferredStateBias (bias {self.internal_state_parameters['goal_seeking_bias']:.2f}): next input {next_handle}.")
            self._log_lot_event("executive", "plannext_preferred_state_override", {"next_input": str(next_handle), "bias": self.internal_state_parameters['goal_seeking_bias']})

        elif executive_eval_results['exploration_countdown'] > 0 or \
             (executive_eval_results['mood'] < -0.65 and random.random() < 0.55):
            available_inputs = list(self.universe['states'])
            if current_outcome_handle in available_inputs: available_inputs.remove(current_outcome_handle)
            if next_handle in available_inputs: available_inputs.remove(next_handle)

            if available_inputs:
                next_handle = random.choice(available_inputs)
                exec_thought_log.append(f"  Exploration/Mood (mood {executive_eval_results['mood']:.2f}, exp T-{executive_eval_results['exploration_countdown']}) override: next input {next_handle}.")
                self._log_lot_event("executive", "plannext_exploration_override", {"next_input": str(next_handle), "mood":executive_eval_results['mood']})

        elif executive_eval_results['mood'] > 0.75 and random.random() < 0.40 and self.cycle_history:
            last_actual_input_handle = self.cycle_history[-1]['actual_input_state_handle']
            if last_actual_input_handle and last_actual_input_handle != current_outcome_handle :
                next_handle = last_actual_input_handle
                exec_thought_log.append(f"  Good mood ({executive_eval_results['mood']:.2f}), repeating last input context {last_actual_input_handle}.")
                self._log_lot_event("executive", "plannext_good_mood_repeat", {"next_input": str(next_handle), "mood":executive_eval_results['mood']})

        exec_thought_log.append(f"  Final proposed next input: {next_handle}.")
        self.next_target_input_state_handle = next_handle
        return next_handle
    
        # ----------------------------------------------------------------------------------
    # --- NEW: MIND-BODY INTERFACE & EMBODIED COGNITIVE CYCLE COMPONENTS ---
    # ----------------------------------------------------------------------------------

    def _ingest_sensory_data(self, image_observation: np.ndarray) -> Optional[float]:
        """
        Sensory Cortex: Uses the VAE to translate a raw pixel image into
        a conceptual latent vector, creating a StateHandle. It also calculates
        the "surprise" by comparing this real observation to its last prediction.

        Returns:
            The prediction error (surprise) from the previous time step, or None.
        """
        # Preprocess the image to match VAE input requirements
        img_tensor = tf.convert_to_tensor(image_observation, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, self.vae_params['IMG_SIZE']) / 255.0
        if len(img_tensor.shape) == 2: # Handle grayscale if it occurs
            img_tensor = tf.stack([img_tensor]*3, axis=-1)

        # 1. Perception: Convert the image to a latent vector
        latent_vector_tensor = self.visual_cortex.observe_to_latent_vector(img_tensor)
        latent_vector = latent_vector_tensor.numpy().flatten()
        
        # 2. Surprise Calculation: Compare reality to imagination
        prediction_error = None
        if self.last_imagined_next_state is not None:
            # Euclidean distance between what was imagined and what actually happened
            prediction_error = np.linalg.norm(latent_vector - self.last_imagined_next_state)
            self.last_prediction_error = prediction_error # Store for logging/history
        
        # 3. Create the StateHandle for this perception
        # The 'id' is a hash of the vector to make it usable in dicts.
        state_id = str(hash(latent_vector.tobytes()))
        # Any discrete, human-readable properties can still be added if available,
        # but the core logic will use the latent vector.
        perceived_state = StateHandle(
            id=state_id,
            latent_vector=latent_vector,
            properties={'timestamp': time.time()} # Example property
        )
        self.last_perceived_state = perceived_state
        self._log_lot_event("perception", "ingest", {"state_hash": state_id[:8], "surprise": prediction_error})
        
        return prediction_error
    

    def _update_internal_world_model(self, current_full_obs: Dict):
        """
        Updates the agent's internal, non-visual understanding of the world,
        like its energy levels or transient environmental states.
        """
        if current_full_obs is None:
            return
            
        self.last_full_obs = current_full_obs
        # Here we could update more sophisticated internal models.
        # For now, it just ensures the latest obs is available for the GoalManager.
        self._log_lot_event("perception", "internal_model_update", {"keys_received": list(current_full_obs.keys())})



    def _process_environmental_feedback(self, reward: float, terminated: bool, info: Dict):
        """
        Consequence Module: The heart of causal and conceptual learning.
        Processes rewards, learns concepts, and learns causal links between
        preconditions, actions, and outcomes.
        """
        # --- Update Mood and Valence (standard procedure) ---
        mood_inertia = 0.9
        reward_influence = 0.3
        self.internal_state_parameters['mood'] = np.clip(
            self.internal_state_parameters['mood'] * mood_inertia + reward * reward_influence, -1.0, 1.0)
        self.last_cycle_valence_raw = reward
        self.last_cycle_valence_mod = reward

        # --- Step 1: Concept Grounding ---
        # If the environment reports a significant interaction, learn the "look" of it.
        interacted_concept = info.get('interacted_with')
        if interacted_concept and self.last_perceived_state:
            # We average the vector over time to get a more stable representation
            vec = self.last_perceived_state.latent_vector
            if interacted_concept not in self.known_concept_vectors:
                self.known_concept_vectors[interacted_concept] = vec
                if self.verbose >= 1: print(f"  CONCEPT LEARNING: Grounded new concept '{interacted_concept}'.")
                self._log_lot_event("learning", "concept_grounded", {"concept": interacted_concept})
            else:
                # Update existing concept with a moving average
                self.known_concept_vectors[interacted_concept] = (self.known_concept_vectors[interacted_concept] * 0.9) + (vec * 0.1)

        # --- Step 2: Causal Learning (Storing experiences in LTM with context) ---
        # We learn from both success AND failure to understand preconditions.
        last_history_entry = self.cycle_history[-1] if self.cycle_history else None
        if last_history_entry and last_history_entry.get('action_taken') is not None:
            # We are learning the consequence of the *last* action
            action_taken = last_history_entry['action_taken']
            preconditions = info.get('preconditions', {}) # e.g., {'has_key': False}
            outcome = {
                "reward": reward,
                "concept_interacted": interacted_concept,
                "terminated": terminated
            }
            
            # The key for a causal memory is the action and the state of relevant preconditions
            precondition_tuple = tuple(sorted(preconditions.items()))
            causal_key = (action_taken, precondition_tuple)

            entry = self.long_term_memory.get(causal_key, {
                'type': 'causal_link',
                'outcomes': collections.Counter(),
                'count': 0
            })
            entry['count'] += 1
            # We store the *outcome* of taking this action under these preconditions
            outcome_tuple_for_counter = (outcome['reward'] > 0, outcome['concept_interacted'], outcome['terminated'])
            entry['outcomes'][outcome_tuple_for_counter] += 1
            
            self.long_term_memory[causal_key] = entry
            self._log_lot_event("learning", "ltm_store_causal_link", {"key": str(causal_key), "outcome": str(outcome_tuple_for_counter)})


        # --- Existing Goal Vector Learning ---
        # This remains important for the shooting planner.
        # CORRECTED: Use the last_arbitrated_goal which led to this successful state.
        if terminated and reward > 0 and self.last_arbitrated_goal:
            goal_desc = self.last_arbitrated_goal.current_goal
            success_vector = self.last_perceived_state.latent_vector
            self.known_goal_latent_vectors[goal_desc] = success_vector
            self._log_lot_event("learning", "goal_vector_memorized", {"goal": goal_desc})


    
    def _execute_abstract_computation(self):
        """Encapsulates a full cycle of internal, non-physical thought."""
        if self.verbose >= 1: print(f"  -- Starting Abstract Computation Cycle --")
        self._log_lot_event("abstract_cycle", "start", {})
        
        # Use the current perceived state to seed the computation if possible
        # For simplicity, we map any non-computational basis state to a default '00'
        seed_state_str = "00"
        
        self._executive_prepare_superposition(seed_state_str)
        
        executed_sequence, chosen_op_strategy, _ = self._executive_generate_computation_sequence()
        
        self._executive_quantum_computation_phase(executed_sequence)
        
        entropy = self._calculate_superposition_entropy()
        collapsed_comp_str = self._executive_trigger_objective_reduction()
        collapsed_concept_handle = self.universe['comp_basis_to_state'].get(collapsed_comp_str, self.universe['start_state'])
        
        # This thought cycle impacts internal state (mood, etc.) but produces no physical action
        # It's like daydreaming or problem-solving.
        self.internal_thought_result = {
            'outcome': collapsed_concept_handle,
            'orp_cost': self.current_orp_before_reset,
            'entropy': entropy
        }

        # The thought itself has a valence which affects mood
        self._executive_evaluate_outcome_and_update_mood(
            collapsed_concept_handle,
            self.current_orp_before_reset,
            entropy,
            len(executed_sequence or [])
        )

        if self.verbose >= 1:
            print(f"  -- Abstract Computation Cycle End. Outcome: {collapsed_concept_handle}, Mood now: {self.internal_state_parameters['mood']:.2f} --")
        self._log_lot_event("abstract_cycle", "end", {"outcome": str(collapsed_concept_handle), "mood": self.internal_state_parameters['mood']})


    def _executive_update_goal_progress(self, goal_to_evaluate: Optional[GoalState], current_state_handle: StateHandle):
        """
        Updates a specific goal's progress based on the current state.
        This version uses EXCLUSIVE criteria for each goal type to prevent logical flaws.
        """
        if not (goal_to_evaluate and goal_to_evaluate.status == "active" and goal_to_evaluate.steps):
            return

        goal = goal_to_evaluate
        if goal.current_step_index >= len(goal.steps):
            return

        current_step = goal.steps[goal.current_step_index]
        step_completed = False
        log_details = {}
        
        # --- THE FINAL EXECUTIVE FUNCTION FIX ---
        # Use an exclusive if/elif structure to ensure the correct completion criterion is used for each goal type.
        
        if goal.goal_type == 'SURVIVAL':
            # Survival is ONLY complete when energy is full. Proximity is irrelevant for completion.
            energy = self.last_full_obs.get('agent_energy', np.array([0.0]))[0]
            if energy >= 99.0:
                step_completed = True
                log_details = {"goal": goal.current_goal, "step_name": current_step.get('name'), "reason": f"Energy full ({energy:.1f})"}
                if self.verbose >= 1: print(f"  STEP COMPLETED (by energy): [{goal.current_goal}] -> '{current_step.get('name')}'")
        
        # For all other goal types, use the generic proximity or state-matching logic.
        else:
            target_concept_name = current_step.get("target_concept")
            target_state_handle = current_step.get("target_state")

            if target_state_handle and isinstance(target_state_handle, StateHandle):
                if current_state_handle == target_state_handle:
                    step_completed = True
                    log_details = {"goal": goal.current_goal, "step_name": current_step.get('name'), "match": str(target_state_handle)}

            elif target_concept_name and target_concept_name in self.known_concept_vectors:
                target_latent_vector = self.known_concept_vectors[target_concept_name]
                current_latent_vector = current_state_handle.latent_vector
                distance = np.linalg.norm(current_latent_vector - target_latent_vector)
                completion_threshold = 1.8 
                
                if distance < completion_threshold:
                    step_completed = True
                    log_details = {"goal": goal.current_goal, "step_name": current_step.get('name'), "concept": target_concept_name, "dist": distance}
                    if self.verbose >= 1: print(f"  STEP COMPLETED (by proximity): [{goal.current_goal}] -> '{current_step.get('name')}'")
        
        if step_completed:
            self._log_lot_event("goal_tracking", "step_completed", log_details)
            self.cycles_stuck_on_step = 0
            goal.current_step_index += 1
            goal.progress = goal.current_step_index / len(goal.steps) if goal.steps else 1.0

            if goal.current_step_index >= len(goal.steps):
                goal.status = "completed"
                if self.verbose >= 1: print(f"  GOAL COMPLETED: '{goal.current_goal}' - All steps finished.")
                self._log_lot_event("goal_tracking", "goal_completed_all_steps", {"goal": goal.current_goal})
                
                # For non-persistent goals, remove them from the manager's list upon completion.
                if goal.goal_type in ['OPPORTUNITY', 'SURVIVAL']: 
                    self.goal_manager.goals = [g for g in self.goal_manager.goals if g is not goal]
            else:
                # This branch is for multi-step goals, like the main quest.
                new_step_info = goal.steps[goal.current_step_index]
                if self.verbose >= 1: print(f"  ADVANCING GOAL: '{goal.current_goal}' to step '{new_step_info.get('name')}'")
                self._log_lot_event("goal_tracking", "step_set", {"name": new_step_info.get("name"), "index": goal.current_step_index})



    # --- Feature 4: Collapse-Triggered Interrupt Handlers ---
    def _executive_handle_collapse_interrupts(self, orp_at_collapse, executed_ops_this_cycle, raw_valence_of_collapse):
        if not self.interrupt_handler_params.get('enabled', False): return

        self._log_lot_event("interrupt", "check", {"orp_at_collapse":orp_at_collapse, "raw_valence_input":raw_valence_of_collapse, "ops_count":len(executed_ops_this_cycle or [])})

        num_ops = len(executed_ops_this_cycle or [])
        expected_orp = sum(self.operation_costs.get(op[0].upper(), 0.05) for op in (executed_ops_this_cycle or [])) + (0.05 if num_ops > 0 else 0)
        orp_is_surprising = (orp_at_collapse > expected_orp * self.interrupt_handler_params['consolidation_orp_surprise_factor'] and expected_orp > 0.05 and num_ops > 0)

        valence_is_extreme = abs(raw_valence_of_collapse) >= self.interrupt_handler_params['consolidation_valence_abs_threshold']
        if valence_is_extreme or orp_is_surprising:
            consol_bonus = self.interrupt_handler_params['consolidation_strength_bonus']
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Strong LTM consolidation triggered (factor {consol_bonus:.1f}). Valence: {raw_valence_of_collapse:.2f}, ORPSurprise: {orp_is_surprising} (ORP {orp_at_collapse:.2f} vs Exp {expected_orp:.2f})")
            self._log_lot_event("interrupt", "strong_consolidation", {"bonus_factor":consol_bonus, "reason_valence_extreme":valence_is_extreme, "reason_orp_surprise":orp_is_surprising})
            self.smn_internal_flags['ltm_consolidation_bonus_factor'] = consol_bonus

        if raw_valence_of_collapse < self.interrupt_handler_params['reactive_ltm_valence_threshold']:
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Reactive LTM flag set for next cycle due to low valence ({raw_valence_of_collapse:.2f} < {self.interrupt_handler_params['reactive_ltm_valence_threshold']}).")
            self._log_lot_event("interrupt", "reactive_ltm_flag", {"valence":raw_valence_of_collapse})
            self.smn_internal_flags['force_ltm_reactive_op_next_cycle'] = True

        if raw_valence_of_collapse >= self.interrupt_handler_params['cognitive_fork_valence_threshold'] and \
           self.current_conceptual_state != self.internal_state_parameters.get('preferred_state_handle'):
            if self.verbose >= 1: print(f"[{self.agent_id}] INTERRUPT: Cognitive fork - marking {self.current_conceptual_state} as new high-interest preferred state (Valence {raw_valence_of_collapse:.2f}).")
            self._log_lot_event("interrupt", "cognitive_fork", {"new_preferred_state":str(self.current_conceptual_state), "valence_trigger":raw_valence_of_collapse})
            self.internal_state_parameters['preferred_state_handle'] = self.current_conceptual_state
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + self.interrupt_handler_params['cognitive_fork_goal_bias_boost'])


    # --- Layer 4: Meta Layer (Monitoring, Adaptation, Self-Reflection) ---
    def _meta_layer_update_cognitive_parameters(self, orp_at_collapse, num_ops_executed, executive_eval_results, entropy_at_collapse):
        if self.verbose >= 2: print(f"  META_LAYER.CognitiveParamUpdate (mood: {executive_eval_results['mood']:.2f}):")
        self._log_lot_event("meta", "cog_param_update_start", {"mood_in":executive_eval_results['mood'], "frustration_in": executive_eval_results['frustration']})

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
        is_goal_oriented_context = (self.internal_state_parameters['preferred_state_handle'] is not None) or \
                                  (self.current_goal_state_obj and self.current_goal_state_obj.status == "active")
        
        if is_goal_oriented_context:
            if mod_valence > 0.35: goal_bias_change += goal_base_rate
            else: goal_bias_change -=goal_base_rate*0.6
        else:
            goal_bias_change -= goal_base_rate*0.3
        self.internal_state_parameters['goal_seeking_bias'] = np.clip(self.internal_state_parameters['goal_seeking_bias'] + goal_bias_change, 0.01, 0.99)

        if self.verbose >=3: print(f"    Curiosity: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}")
        self._log_lot_event("meta", "cog_param_update_end", {"cog_load":self.internal_state_parameters['cognitive_load'], "attn": self.internal_state_parameters['attention_level'], "cur": self.internal_state_parameters['curiosity'], "goal_bias":self.internal_state_parameters['goal_seeking_bias']})


    def _meta_layer_adapt_preferred_state(self, collapsed_outcome_handle, mod_valence):
        high_val_thresh = self.metacognition_params['high_valence_threshold']
        goal_alignment_valence_threshold = 0.8 
        low_val_thresh = self.metacognition_params['low_valence_threshold']
        current_pref_state = self.internal_state_parameters['preferred_state_handle']
        pref_state_log_msg = ""
        action_taken_this_adaptation = False

        # --- CORRECTED TO USE LAST ARBITRATED GOAL ---
        active_goal = self.last_arbitrated_goal
        active_goal_target_state_for_alignment = None
        goal_step_name_for_alignment_log = "N/A"
        
        if active_goal and active_goal.status == "active":
            if 0 <= active_goal.current_step_index < len(active_goal.steps):
                current_active_step_for_align = active_goal.steps[active_goal.current_step_index]
                active_goal_target_state_for_alignment = current_active_step_for_align.get("target_state")
                goal_step_name_for_alignment_log = current_active_step_for_align.get("name", f"Step {active_goal.current_step_index}")

        if mod_valence >= goal_alignment_valence_threshold and active_goal_target_state_for_alignment:
            if current_pref_state != active_goal_target_state_for_alignment:
                self.internal_state_parameters['preferred_state_handle'] = active_goal_target_state_for_alignment
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.35)
                pref_state_log_msg += f"High valence ({mod_valence:.2f} >= {goal_alignment_valence_threshold}) AND active goal step '{goal_step_name_for_alignment_log}' target. Aligned preferred state to {active_goal_target_state_for_alignment}. Goal bias strongly boosted."
                action_taken_this_adaptation = True
                if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState (GoalAlign): {pref_state_log_msg}")
                self._log_lot_event("meta", "adapt_pref_state_goal_align_high_valence", {
                    "message": pref_state_log_msg,
                    "new_pref_state_str": str(self.internal_state_parameters['preferred_state_handle']),
                    "mod_valence": mod_valence,
                    "goal_target_state": str(active_goal_target_state_for_alignment)
                })
                return

        is_pref_state_goal_dictated = False
        goal_driven_pref_state_source = "None"
        if active_goal and active_goal.status == "active":
            if 0 <= active_goal.current_step_index < len(active_goal.steps):
                current_active_step = active_goal.steps[active_goal.current_step_index]
                step_target_state = current_active_step.get("target_state")
                if step_target_state == current_pref_state:
                    is_pref_state_goal_dictated = True
                    goal_driven_pref_state_source = f"GoalStep:'{current_active_step.get('name', '')}'"
                    if not self.working_memory.is_empty():
                        top_item = self.working_memory.peek()
                        if top_item.type == "goal_step_context":
                            wm_data = top_item.data
                            goal_name = active_goal.current_goal if not isinstance(active_goal.current_goal, StateHandle) else str(active_goal.current_goal)
                            if wm_data.get("goal_name") == goal_name and \
                               wm_data.get("step_index") == active_goal.current_step_index and \
                               wm_data.get("goal_step_name") == current_active_step.get("name"):
                                goal_driven_pref_state_source += "+WM_Match"

        if is_pref_state_goal_dictated:
            pref_state_log_msg += f"Preferred state {current_pref_state} currently aligned with active {goal_driven_pref_state_source}. Adaptation highly constrained. "
            if mod_valence <= low_val_thresh * 0.8 and current_pref_state == collapsed_outcome_handle :
                if active_goal.status != "active": 
                    self.internal_state_parameters['preferred_state_handle'] = None
                    self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.3) 
                    pref_state_log_msg += f"Goal no longer active AND low valence for {collapsed_outcome_handle} ({mod_valence:.2f}), cleared preferred state."
                    action_taken_this_adaptation = True
                else: 
                    pref_state_log_msg += f"Low valence ({mod_valence:.2f}) for goal-driven preferred state {collapsed_outcome_handle}, but goal is active. No change to pref state here. Frustration may increase."
            elif mod_valence >= high_val_thresh and current_pref_state == collapsed_outcome_handle:
                pref_state_log_msg += f"High valence ({mod_valence:.2f}) for achieving goal-driven preferred state. Reinforced."
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.1) 
                action_taken_this_adaptation = True
        else: 
            if mod_valence >= high_val_thresh and current_pref_state != collapsed_outcome_handle:
                self.internal_state_parameters['preferred_state_handle'] = collapsed_outcome_handle
                self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.28)
                self.internal_state_parameters['frustration'] *= 0.55
                pref_state_log_msg += f"New (non-goal-driven) preferred state {collapsed_outcome_handle} set due to high valence ({mod_valence:.2f}). Goal bias up, frustration down."
                action_taken_this_adaptation = True
            elif mod_valence <= low_val_thresh and current_pref_state == collapsed_outcome_handle: 
                self.internal_state_parameters['preferred_state_handle'] = None
                self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.22)
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.18)
                pref_state_log_msg += f"Non-goal-driven preferred state {collapsed_outcome_handle} cleared due to low valence ({mod_valence:.2f}). Goal bias down, curiosity up."
                action_taken_this_adaptation = True
            elif current_pref_state == collapsed_outcome_handle and low_val_thresh < mod_valence < (high_val_thresh * 0.5) and random.random() < 0.15: 
                self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] * 0.9)
                pref_state_log_msg += f"Non-goal-driven preferred state {collapsed_outcome_handle} yielding mediocre results ({mod_valence:.2f}), slightly reduced goal_seeking_bias."
                if self.internal_state_parameters['goal_seeking_bias'] < 0.1:
                    self.internal_state_parameters['preferred_state_handle'] = None
                    pref_state_log_msg += " Preferred state cleared due to very low bias."
                action_taken_this_adaptation = True

        if action_taken_this_adaptation or (self.verbose >=2 and is_pref_state_goal_dictated and "No change to pref state here" not in pref_state_log_msg):
            if self.verbose >= 1: print(f"[{self.agent_id}] META.AdaptPrefState: {pref_state_log_msg}")
            self._log_lot_event("meta", "adapt_pref_state", {"message": pref_state_log_msg,
                                                           "new_pref_state_str": str(self.internal_state_parameters['preferred_state_handle']),
                                                           "mod_valence": mod_valence,
                                                           "is_goal_dictated": is_pref_state_goal_dictated,
                                                           "source_if_goal_dictated": goal_driven_pref_state_source if is_pref_state_goal_dictated else "N/A"})
        elif is_pref_state_goal_dictated and "No change to pref state here" in pref_state_log_msg and self.verbose >= 2 : 
             self._log_lot_event("meta", "adapt_pref_state_skipped_active_goal", {"current_pref_state": str(current_pref_state), "mod_valence": mod_valence, "reason_msg": pref_state_log_msg})


# =================== AFTER ===================
    def _meta_layer_perform_review(self):
        if self.verbose >= 1: print(f"[{self.agent_id}] --- META_LAYER.Review (Cycle {self.current_cycle_num}) ---")
        self._log_lot_event("meta", "review_start", {"cycle": self.current_cycle_num, "review_interval": self.metacognition_params['review_interval']})

        history_span_for_review = min(len(self.cycle_history), self.metacognition_params['review_interval'] * 3)
        if history_span_for_review < self.metacognition_params['review_interval'] * 0.6 :
            if self.verbose >= 1: print(f"    META.Review: Insufficient history ({history_span_for_review} cycles) for meaningful review.")
            self._log_lot_event("meta", "review_insufficient_history", {"history_len": history_span_for_review})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        recent_history_slice = list(self.cycle_history)[-history_span_for_review:]
        valid_cycles = [c for c in recent_history_slice if c.get('valence_mod_this_cycle') is not None and c.get('op_strategy')]
        if not valid_cycles:
            if self.verbose >= 1: print("    META.Review: No valid cycles with strategy info in recent history.")
            self._log_lot_event("meta", "review_no_valid_cycles", {"valid_cycles_count": len(valid_cycles)})
            self.metacognition_params['cycles_since_last_review'] = 0
            return

        self_model = self.metacognition_params['self_model_stats']
        use_self_model = self.metacognition_params.get('enable_self_model_adaptation', False)
        
        review_updates = collections.defaultdict(lambda: {'uses': 0, 'success': 0, 'valence': 0.0})
        for cycle in valid_cycles:
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
            lr = 0.1 
            self_model['total_reviews_for_model'] += 1
            for strat, data in review_updates.items():
                if data['uses'] > 0:
                    self_model['strategy_total_uses'][strat] += data['uses']
                    self_model['strategy_success_count'][strat] += data['success']
                    self_model['strategy_total_valence_accum'][strat] += data['valence']

                    recent_success_rate = data['success'] / data['uses']
                    self_model['strategy_success_rates'][strat] = (1 - lr) * self_model['strategy_success_rates'].get(strat, 0.0) + lr * recent_success_rate
                    
                    recent_avg_valence = data['valence'] / data['uses']
                    self_model['strategy_avg_valence'][strat] = (1 - lr) * self_model['strategy_avg_valence'].get(strat, 0.0) + lr * recent_avg_valence

            if self.verbose >= 2:
                print("    META.Review: Self-Model Updated.")
                for s in ['memory', 'problem_solve', 'goal_seek', 'curiosity']:
                    if self_model['strategy_total_uses'][s] > 0:
                         print(f"      - {s}: SuccessRate={self_model['strategy_success_rates'][s]:.2f}, AvgValence={self_model['strategy_avg_valence'][s]:.2f}, Uses(total)={self_model['strategy_total_uses'][s]}")
            self._log_lot_event("meta", "review_self_model_update", {"model_state_str": str(self_model['strategy_success_rates'])})


        avg_valence_overall = np.mean([c['valence_mod_this_cycle'] for c in valid_cycles])
        outcome_diversity = len(set(c['collapsed_to_handle'] for c in valid_cycles)) / len(valid_cycles) if valid_cycles else 0.0

        if use_self_model and self_model['total_reviews_for_model'] > 2: 
            if self.verbose >= 1: print("    META.Review: Applying adaptations based on SELF-MODEL.")
            
            if self_model['strategy_success_rates'].get('memory', 1.0) < 0.3 and self_model['strategy_avg_valence'].get('memory', 1.0) < 0.1 and self_model['strategy_total_uses']['memory'] > 5:
                if self.verbose >=2: print("      SELF-MODEL: Memory strategy is underperforming. Boosting curiosity.")
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.15)
                self.internal_state_parameters['computation_length_preference'] = max(1, self.internal_state_parameters['computation_length_preference'] - 1)
                self._log_lot_event("meta", "review_adapt_selfmodel_poor_memory", {})

            elif self_model['strategy_avg_valence'].get('curiosity', 1.0) < -0.2 and self_model['strategy_avg_valence'].get('problem_solve', 1.0) < -0.1 and self_model['strategy_total_uses']['curiosity'] > 5:
                if self.verbose >=2: print("      SELF-MODEL: Exploratory strategies are failing. Increasing weight of memory strategy.")
                sw = self.internal_state_parameters['strategy_weights']
                sw['memory'] = min(1.0, sw.get('memory', 0.1) * 1.3 + 0.1)
                sw['curiosity'] *= 0.7
                total = sum(v for k,v in sw.items() if isinstance(v, (int, float)) and k != 'default')
                if total > 0: [sw.update({k: v/total}) for k,v in sw.items() if isinstance(v, (int,float)) and k != 'default']
                self._log_lot_event("meta", "review_adapt_selfmodel_poor_exploration", {})

            td = self.orp_threshold_dynamics; prev_eor = self.E_OR_THRESHOLD
            adapt_rate_thresh = td.get('adapt_rate', DEFAULT_ORP_THRESHOLD_DYNAMICS['adapt_rate'])
            if self_model['strategy_success_rates'].get('memory', 0.0) > 0.6 and self_model['strategy_avg_valence'].get('problem_solve', 1.0) < 0.2:
                self.E_OR_THRESHOLD = max(td['min'], self.E_OR_THRESHOLD - adapt_rate_thresh * 0.8)
                if self.verbose >= 2: print(f"      SELF-MODEL: Memory success suggests lower E_OR_THRESH is efficient. Decreased to {self.E_OR_THRESHOLD:.3f}")
            elif self_model['strategy_success_rates'].get('problem_solve', 0.0) > 0.5 and self.E_OR_THRESHOLD < (td['max'] * 0.8):
                self.E_OR_THRESHOLD = min(td['max'], self.E_OR_THRESHOLD + adapt_rate_thresh)
                if self.verbose >= 2: print(f"      SELF-MODEL: Problem-solving success warrants higher E_OR_THRESH. Increased to {self.E_OR_THRESHOLD:.3f}")

        else: 
            if self.verbose >= 1: print("    META.Review: Applying adaptations based on GENERAL stats (Self-Model OFF or new).")
            if avg_valence_overall < self.metacognition_params['low_valence_threshold'] or outcome_diversity < self.metacognition_params['exploration_threshold_entropy']:
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + self.metacognition_params['curiosity_adaptation_rate'])
            if avg_valence_overall > self.metacognition_params['high_valence_threshold']:
                 self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + self.metacognition_params['goal_bias_adaptation_rate'])
            
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
                        is_persistent_gap = (self.current_cycle_num - data.get('first_cycle', 0)) > 15
                        if is_persistent_gap:
                            knowledge_gaps_found.append({'seq': seq_tuple, 'confidence': confidence})

            if knowledge_gaps_found:
                num_gaps = len(knowledge_gaps_found)
                knowledge_gaps_found.sort(key=lambda x: x['confidence'])
                worst_gap = knowledge_gaps_found[0]

                if self.verbose >= 1: print(f"      KNOWLEDGE GAPS DETECTED: {num_gaps} low-confidence memories. Worst gap: conf={worst_gap['confidence']:.2f}. Boosting Curiosity.")
                self._log_lot_event("meta", "review_knowledge_gap_found", {"count": num_gaps, "worst_gap_conf": worst_gap['confidence'], "worst_gap_seq_str": str(worst_gap['seq'])})
                
                curiosity_boost = self.metacognition_params.get('epistemic_curiosity_boost', 0.3)
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + curiosity_boost)
                if random.random() < 0.25:
                    self.internal_state_parameters['exploration_mode_countdown'] = max(
                        self.internal_state_parameters['exploration_mode_countdown'],
                        self.metacognition_params['exploration_mode_duration']
                    )
                    if self.verbose>=2: print("      Triggering exploration mode due to knowledge gaps.")

        self.metacognition_params['cycles_since_last_review'] = 0
        if self.verbose >= 1: print(f"[{self.agent_id}] --- Metacognitive Review Complete ---")
        self._log_lot_event("meta", "review_end", {"new_cur": self.internal_state_parameters['curiosity'], "new_gb":self.internal_state_parameters['goal_seeking_bias']})


    # ---  Feature 3: Synaptic Mutation Network (SMN) Methods (Enhanced Graph Version) ---
    def _initialize_smn_graph_structures(self):
        """Initializes SMN graph-related structures: param indices, influence matrix."""
        self.smn_param_indices = {name: i for i, name in enumerate(self.smn_controlled_params_definitions.keys())}
        self.smn_param_names_from_indices = {i: name for name, i in self.smn_param_indices.items()}

        num_smn_params = len(self.smn_controlled_params_definitions)
        self.smn_config['smn_influence_matrix_size'] = num_smn_params 

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
            self._log_lot_event("smn", "error_get_param", {"path_str":str(path_tuple), "error":str(e)})
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
                # Correctly use the path segment string, not the loop index
                current_dict_or_obj = current_dict_or_obj[path_tuple[key_part_idx]]
            current_dict_or_obj[path_tuple[-1]] = value
            return True
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            if self.verbose >= 1: print(f"    SMN_SET_PARAM_ERROR: Failed to set param at path {path_tuple} to {value}: {e}")
            self._log_lot_event("smn", "error_set_param", {"path_str":str(path_tuple), "value_set":value, "error":str(e)})
            return False


    def _smn_update_and_apply_mutations(self, valence_mod_this_cycle, valence_raw_this_cycle, prev_cycle_valence_mod, orp_at_collapse):
        if not self.smn_config.get('enabled', False) or not self.smn_param_indices: return 

        valence_gain = valence_mod_this_cycle - prev_cycle_valence_mod
        smn_pos_thresh = self.internal_state_parameters['smn_positive_valence_threshold']
        smn_neg_thresh = self.internal_state_parameters['smn_negative_valence_threshold']

        if self.verbose >= 2: print(f"  SMN Update & Mutate: ValenceMod={valence_mod_this_cycle:.2f}, PrevModVal={prev_cycle_valence_mod:.2f}, Gain={valence_gain:.2f}, ORP={orp_at_collapse:.3f}")
        self._log_lot_event("smn", "update_start", {"val_mod_curr":valence_mod_this_cycle, "val_mod_prev": prev_cycle_valence_mod, "val_gain":valence_gain, "orp_col":orp_at_collapse})

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
                        self._log_lot_event("smn", "update_mutation_applied", {"param_smn_key":param_smn_key, "path_str":str(param_path), "old_val":current_val, "new_val":new_val, "change":actual_change, "type":"primary"})
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
                        self._log_lot_event("smn_graph", "propagation_attempt", {
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
                        self._log_lot_event("smn", "update_mutation_applied", {"param_smn_key":target_param_smn_key, "old_val":current_target_val, "new_val":new_target_val, "change":actual_propagated_change, "type":"propagated"})
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
                            self._log_lot_event("smn_graph", "hebbian_update", {
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
            self._log_lot_event("firewall", "cooldown", {"remaining": self.firewall_cooldown_remaining})
            return

        self.firewall_cycles_since_last_check +=1
        if self.firewall_cycles_since_last_check < self.firewall_params['check_interval']:
            return

        self.firewall_cycles_since_last_check = 0
        intervention_made = False
        intervention_reason = "None"
        intervention_details = {}

        # Low valence check (This logic is universal and remains correct)
        low_val_streak_needed = self.firewall_params['low_valence_streak_needed']
        if not intervention_made and len(self.cycle_history) >= low_val_streak_needed:
            recent_valences = [c['valence_mod_this_cycle'] for c in list(self.cycle_history)[-low_val_streak_needed:] if 'valence_mod_this_cycle' in c]
            if len(recent_valences) == low_val_streak_needed and all(v < self.firewall_params['low_valence_threshold'] for v in recent_valences):
                intervention_reason = f"Persistent Low Valence (avg {np.mean(recent_valences):.2f} < {self.firewall_params['low_valence_threshold']} for {low_val_streak_needed} cycles)"
                self.internal_state_parameters['exploration_mode_countdown'] = max(
                    self.internal_state_parameters['exploration_mode_countdown'],
                    self.firewall_params['intervention_exploration_boost_duration']
                )
                self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.33)
                intervention_made = True; intervention_details = {'avg_valence': np.mean(recent_valences), 'streak': low_val_streak_needed}
        
        # --- START OF CORRECTED LOOP DETECTION LOGIC ---
        loop_window = self.firewall_params['loop_detection_window']
        if not intervention_made and len(self.cycle_history) >= loop_window:
            history_slice = list(self.cycle_history)[-loop_window:]
            behavior_patterns = []
            
            # This loop builds a list of patterns based on the type of cycle
            for c in history_slice:
                # The 'focus' key was added to the history dict in the new embodied run_cycle
                focus = c.get('focus')
                if focus and 'GOAL' in focus:
                    # A physical behavior pattern is (perception, action)
                    pattern = (c.get('perceived_state_handle'), c.get('action_taken'))
                    behavior_patterns.append(pattern)
                elif focus == 'ABSTRACT':
                    # An abstract behavior pattern is (outcome, strategy, ops)
                    ops_tuple = tuple(tuple(op) for op in c.get('ops_executed', []))
                    pattern = (c.get('collapsed_to_handle'), c.get('op_strategy'), ops_tuple)
                    behavior_patterns.append(pattern)
                # If 'focus' key is missing (e.g., old history data), we simply skip it for robustness
                else:
                    behavior_patterns.append(None) # Append a non-matchable item
            
            # The rest of the loop detection logic operates on the generated behavior_patterns
            if behavior_patterns:
                counts = collections.Counter(p for p in behavior_patterns if p is not None)
                for pattern_tuple, count in counts.items():
                    if count >= self.firewall_params['loop_detection_min_repeats']:
                        # The valence check logic is universal and works on the history_slice
                        loop_valences = []
                        # Correctly find the valences only for cycles that match the found pattern
                        for i, p_tuple in enumerate(behavior_patterns):
                             if p_tuple == pattern_tuple and 'valence_mod_this_cycle' in history_slice[i]:
                                loop_valences.append(history_slice[i]['valence_mod_this_cycle'])
                        
                        if loop_valences and np.mean(loop_valences) < self.firewall_params['low_valence_threshold'] * 0.75:
                            intervention_reason = f"Behavioral Loop Detected (pattern {str(pattern_tuple)[:80]}... repeated {count}x with low avg_val {np.mean(loop_valences):.2f})"
                            # The intervention itself (shaking up strategy weights) is fine
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

                            self.internal_state_parameters['preferred_state_handle'] = None
                            intervention_made = True
                            intervention_details = {'pattern':str(pattern_tuple), 'count':count, 'avg_loop_val':np.mean(loop_valences)}
                            break
        # --- END OF CORRECTED LOOP DETECTION LOGIC ---

        # Premature collapse check (Only applies to abstract cycles, so needs to be more careful)
        prem_collapse_streak = self.firewall_params['premature_collapse_streak_needed']
        if not intervention_made and len(self.cycle_history) >= prem_collapse_streak:
            # Filter for only the abstract computation cycles
            abstract_cycle_data = [c for c in list(self.cycle_history)[-prem_collapse_streak:] if c.get('focus') == 'ABSTRACT' and 'orp_at_collapse' in c]
            
            if len(abstract_cycle_data) >= prem_collapse_streak: # Ensure we have enough data points
                threshold_ratios = [c['orp_at_collapse'] / (c.get('E_OR_thresh_this_cycle', self.E_OR_THRESHOLD)+1e-6) for c in abstract_cycle_data if c.get('num_ops_executed',0) > 0]
                if threshold_ratios and all(ratio < self.firewall_params['premature_collapse_orp_max_ratio'] for ratio in threshold_ratios):
                     intervention_reason = f"Persistent Premature ORP Collapse (avg ratio {np.mean(threshold_ratios):.2f} < {self.firewall_params['premature_collapse_orp_max_ratio']} for {len(threshold_ratios)} abstract op-cycles)"
                     self.E_OR_THRESHOLD *= self.firewall_params['intervention_orp_threshold_increase_factor']
                     self.E_OR_THRESHOLD = min(self.E_OR_THRESHOLD, self.orp_threshold_dynamics['max'])
                     self.internal_state_parameters['computation_length_preference'] = min(8, max(self.internal_state_parameters['computation_length_preference'] + 1, 2))
                     intervention_made = True; intervention_details = {'avg_orp_ratio':np.mean(threshold_ratios), 'new_EOR':self.E_OR_THRESHOLD, 'new_comp_pref': self.internal_state_parameters['computation_length_preference']}

        if intervention_made:
            if self.verbose >= 1: print(f"[{self.agent_id}] COGNITIVE FIREWALL Activated: {intervention_reason}")
            self._log_lot_event("firewall", "intervention", {"reason": intervention_reason, "details_str":str(intervention_details)})
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
                self._log_lot_event("firewall", "intervention_attention_boost", {"old_attn": old_attn, "boost": attention_boost_after_firewall_wm_clear, "new_attn": self.internal_state_parameters['attention_level']})

            # --- CORRECTED TO USE LAST ARBITRATED GOAL ---
            active_goal = self.last_arbitrated_goal
            if active_goal and active_goal.status == "active":
                if self.verbose >= 1: print(f"[{self.agent_id}] FIREWALL: Active goal '{active_goal.current_goal}' status changed to 'pending' due to intervention.")
                active_goal.status = "pending" # Interrupt the goal, force re-evaluation
                active_goal.history.append({"cycle": self.current_cycle_num, "event": "firewall_interrupted_goal", "reason": intervention_reason[:50]})


    def _interactive_psych_probe(self):
        """
        An interactive debugging shell to inspect and manipulate the agent's state.
        This is a critical tool for AGI development.
        """
        # Ensure we don't probe during a silent run
        if self.verbose < 0: return

        print("\n\n" + "="*20 + " PSYCH-PROBE ENGAGED " + "="*20)
        print(f"Agent '{self.agent_id}' paused at the end of Cycle {self.current_cycle_num}.")
        
        while True:
            try:
                command_str = input(f"PROBE ({self.agent_id}) >> ").strip().lower()
                parts = command_str.split()
                if not parts: continue
                
                cmd = parts[0]
                
                if cmd in ['exit', 'quit', 'q', 'c', 'continue']:
                    print("="*21 + " RESUMING EXECUTION " + "="*21 + "\n")
                    break
                elif cmd in ['help', '?']:
                    print("--- Psych-Probe Commands ---")
                    print("  summary / s    : Print full internal state summary.")
                    print("  ltm [N]        : Show top N most confident LTM entries (default 5).")
                    print("  wm             : Show contents of working memory stack.")
                    print("  goals          : List all current goals in the GoalManager.")
                    print("  history [N]    : Show summary of last N cycles (default 3).")
                    print("  get [path]     : Get a parameter value (e.g., get internal_state_parameters.curiosity).")
                    print("  set [path] [v] : Set a parameter value (e.g., set internal_state_parameters.curiosity 0.99).")
                    print("  inject_goal    : (Interactive) Create and inject a new simple goal.")
                    print("  clear_wm       : Clear the working memory stack.")
                    print("  exit / q / c   : Exit probe and continue execution.")
                elif cmd in ['summary', 's']:
                    self.print_internal_state_summary()
                elif cmd == 'wm':
                    if self.working_memory.is_empty():
                        print("Working Memory is empty.")
                    else:
                        print("--- Working Memory (Top to Bottom) ---")
                        for i, item in enumerate(reversed(self.working_memory.stack)):
                            print(f"  [{i}] {item}")
                elif cmd == 'goals':
                    print("--- Current Goals in GoalManager ---")
                    if not self.goal_manager.goals:
                        print("  No goals are currently active.")
                    for i, goal in enumerate(self.goal_manager.goals):
                        print(f"  [{i}] {goal}")
                    print(f"Last Arbitrated Goal: {self.last_arbitrated_goal}")

                elif cmd == 'ltm':
                    num_entries = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                    if not self.long_term_memory:
                        print("Long-Term Memory is empty.")
                    else:
                        print(f"--- Top {num_entries} LTM Entries (by confidence) ---")
                        sorted_ltm = sorted(self.long_term_memory.items(), key=lambda item: item[1].get('confidence', 0), reverse=True)
                        for i, (seq, data) in enumerate(sorted_ltm[:num_entries]):
                            print(f"  [{i+1}] Seq: {seq}")
                            print(f"      Conf: {data.get('confidence',0):.3f}, Util: {data.get('utility',0):.3f}, AvgVal: {data.get('avg_valence',0):.3f}, Count: {data.get('count',0)}")
                elif cmd == 'history':
                     num_cycles = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 3
                     print(f"--- Last {num_cycles} Cycle Summaries ---")
                     for cycle_data in list(self.cycle_history)[-num_cycles:]:
                         print(f"  Cycle {cycle_data['cycle_num']}: Focus='{cycle_data.get('focus', 'N/A')}', Action={cycle_data.get('action_taken', 'None')}, Val={cycle_data.get('valence_mod_this_cycle', 0):.2f}, Surprise={cycle_data.get('surprise_this_cycle', 0):.2f}")
                elif cmd == 'get':
                    if len(parts) < 2: print("Usage: get path.to.parameter"); continue
                    path_parts = parts[1].split('.')
                    obj = self
                    try:
                        for p in path_parts: obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
                        print(f"{parts[1]} = {obj}")
                    except (AttributeError, KeyError):
                        print(f"ERROR: Could not find path '{parts[1]}'")
                elif cmd == 'set':
                    if len(parts) < 3: print("Usage: set path.to.parameter value"); continue
                    path_parts = parts[1].split('.')
                    value_str = parts[2]
                    try:
                        # Convert value to float or int if possible
                        try:
                            value = float(value_str)
                            if value.is_integer(): value = int(value)
                        except ValueError:
                            value = value_str # It's a string

                        obj = self
                        for p in path_parts[:-1]: obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
                        
                        if isinstance(obj, dict): obj[path_parts[-1]] = value
                        else: setattr(obj, path_parts[-1], value)
                        print(f"Set {parts[1]} to {value}")
                    except Exception as e:
                        print(f"ERROR setting value: {e}")
                elif cmd == 'clear_wm':
                    self.working_memory.clear()
                    print("Working memory cleared.")
                elif cmd == 'inject_goal':
                    print("--- Goal Injection ---")
                    g_desc = input("Goal description (e.g., 'Test Goal'): ")
                    g_type = input("Goal Type (MAIN_QUEST, OPPORTUNITY, SURVIVAL): ").upper()
                    if not g_desc or g_type not in ["MAIN_QUEST", "OPPORTUNITY", "SURVIVAL"]:
                        print("Invalid description or type. Aborting."); continue

                    g_target_concept = input("Target Concept Name (e.g., 'key', 'target', 'charging_pad'): ")

                    # Create a new goal.
                    # This simple injector creates a single-step goal.
                    goal = GoalState(
                        current_goal=g_desc,
                        goal_type=g_type,
                        base_priority=0.9, # High priority since it's manual
                        steps=[{"name": f"Manually achieve {g_target_concept}", "target_concept": g_target_concept}]
                    )
                    
                    if goal.goal_type == "MAIN_QUEST":
                        self.goal_manager.set_main_goal(goal)
                    else:
                        self.goal_manager.add_goal(goal)
                    print(f"New goal injected: {goal}")
                else:
                    print(f"Unknown command: '{cmd}'. Type 'help' for a list of commands.")
            except Exception as e:
                print(f"Probe ERROR: {e}")

    def sleep_train(self):
        """
        Triggers a "sleep" cycle where the agent retrains its perception and
        world models on a batch of interesting experiences from its replay buffer.
        """
        if not self.ll_params['enabled'] or len(self.experience_replay_buffer) < self.ll_params['training_batch_size']:
            return False

        if self.verbose >= 1:
            print(f"[{self.agent_id}] --- Entering Sleep/Training Cycle (Buffer size: {len(self.experience_replay_buffer)}) ---")
        self._log_lot_event("learning", "sleep_train_start", {"buffer_size": len(self.experience_replay_buffer)})
        
        batch = self.experience_replay_buffer.sample(self.ll_params['training_batch_size'])
        
        # --- 1. Fine-tune the VAE (Visual Cortex) ---
        vae_images = np.array([exp['state_img'] for exp in batch])
        self.visual_cortex.fit(
            vae_images,
            epochs=self.ll_params['training_epochs_per_cycle'],
            batch_size=self.ll_params['training_batch_size'],
            verbose=0
        )
        
        # --- 2. Fine-tune the World Model ---
        # We need to re-encode the images with the *newly updated* VAE
        states_imgs = np.array([exp['state_img'] for exp in batch])
        next_states_imgs = np.array([exp['next_state_img'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        
        latent_states = self.visual_cortex.observe_to_latent_vector(states_imgs).numpy()
        latent_next_states = self.visual_cortex.observe_to_latent_vector(next_states_imgs).numpy()
        actions_one_hot = tf.one_hot(actions, self.action_space.n)
        
        self.world_model.fit(
            [latent_states, actions_one_hot],
            latent_next_states,
            epochs=self.ll_params['training_epochs_per_cycle'],
            batch_size=self.ll_params['training_batch_size'],
            verbose=0
        )

        if self.verbose >= 1:
            print(f"[{self.agent_id}] --- Sleep/Training Cycle Complete ---")
        
        return True               


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


    def set_goal_state(self, goal_state_obj: Optional[GoalState]):
        """
        Sets the agent's main quest goal via the GoalManager.
        This will replace any previously existing main quest.
        """
        if goal_state_obj is None:
             # Clear all main quests if passed None
            self.goal_manager.goals = [g for g in self.goal_manager.goals if g.goal_type != 'MAIN_QUEST']
            self._log_lot_event("goal", "main_quest_cleared", {})
            return

        if not isinstance(goal_state_obj, GoalState):
            raise ValueError("goal_state_obj must be an instance of GoalState or None.")
        
        # Ensure it's marked as a main quest.
        goal_state_obj.goal_type = 'MAIN_QUEST'
        
        self.goal_manager.set_main_goal(goal_state_obj)
        if self.verbose >= 1: print(f"[{self.agent_id}] New main quest set: {goal_state_obj}")

    def is_main_goal_completed(self) -> bool:
        """
        Checks if the primary MAIN_QUEST goal is completed.
        Returns True if a main quest exists and its status is 'completed'.
        """
        # Find the main quest in the goal manager's list
        main_quest = next((g for g in self.goal_manager.goals if g.goal_type == 'MAIN_QUEST'), None)
        
        # If a main quest was found and its status is 'completed', return True.
        if main_quest and main_quest.status == 'completed':
            return True
        
        return False       


    def print_internal_state_summary(self, indent="  ", custom_logger=None):
        log_func = custom_logger if callable(custom_logger) else print

        log_func(f"{indent}--- Internal State Summary for Agent {self.agent_id} (Cycle {self.current_cycle_num}) ---")
        log_func(f"{indent}  State: Mood: {self.internal_state_parameters['mood']:.2f}, Attn: {self.internal_state_parameters['attention_level']:.2f}, Load: {self.internal_state_parameters['cognitive_load']:.2f}, Frust: {self.internal_state_parameters['frustration']:.2f}")
        pref_state_handle = self.internal_state_parameters.get('preferred_state_handle')
        log_func(f"{indent}  Cognition: Cur: {self.internal_state_parameters['curiosity']:.2f}, GoalBias: {self.internal_state_parameters['goal_seeking_bias']:.2f}, PrefState: {str(pref_state_handle)}, CompLenPref: {self.internal_state_parameters['computation_length_preference']}")
        log_func(f"{indent}  Exploration: Countdown: {self.internal_state_parameters['exploration_mode_countdown']}")
        log_func(f"{indent}  OrchOR: E_OR_THRESH: {self.E_OR_THRESHOLD:.3f} (AdaptRate: {self.orp_threshold_dynamics['adapt_rate']:.4f}), ORP_DECAY: {self.orp_decay_rate:.4f} (AdaptRate: {self.orp_decay_dynamics['adapt_rate']:.4f})")
        sw_str = ", ".join([f"{k[:4]}:{v:.2f}" for k,v in self.internal_state_parameters['strategy_weights'].items()])
        log_func(f"{indent}  StrategyWeights: {sw_str}")
        log_func(f"{indent}  MetaCog: ReviewIn: {self.metacognition_params['review_interval']-self.metacognition_params['cycles_since_last_review']}, AdaptRates(Cur/Goal): {self.metacognition_params['curiosity_adaptation_rate']:.3f}/{self.metacognition_params['goal_bias_adaptation_rate']:.3f}")
        
        if self.metacognition_params.get('enable_self_model_adaptation', False):
            self_model = self.metacognition_params.get('self_model_stats', {})
            if self_model and self_model.get('total_reviews_for_model', 0) > 0:
                log_func(f"{indent}  Self-Model ({self_model['total_reviews_for_model']} reviews):")
                perf_list = []
                for s in ['memory', 'problem_solve', 'goal_seek', 'curiosity']:
                    if self_model['strategy_total_uses'][s] > 0:
                        perf_list.append((s, self_model['strategy_avg_valence'][s], self_model['strategy_success_rates'][s]))
                if perf_list:
                    perf_list.sort(key=lambda x: x[1], reverse=True) 
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

        log_func(f"{indent}  Goals ({len(self.goal_manager.goals)} active):")
        for i, goal in enumerate(self.goal_manager.goals):
            log_func(f"{indent}    [{i}] {goal}")
        log_func(f"{indent}  Last Arbitrated Focus: {self.last_arbitrated_goal}")
        log_func(f"{indent}--- End of Summary ---")

    def get_public_state_summary(self):
        # Use the arbitrated goal for public state.
        active_goal = self.last_arbitrated_goal
        
        return {
            "agent_id": self.agent_id, "current_cycle_num": self.current_cycle_num,
            "mood": self.internal_state_parameters['mood'], "attention": self.internal_state_parameters['attention_level'],
            "frustration": self.internal_state_parameters['frustration'], "curiosity": self.internal_state_parameters['curiosity'],
            "collapsed_state": self.current_conceptual_state, 
            "preferred_state": self.internal_state_parameters.get('preferred_state_handle'),
            "E_OR_THRESHOLD": self.E_OR_THRESHOLD,
            "active_goal_name": str(active_goal.current_goal) if active_goal else None,
            "active_goal_progress": active_goal.progress if active_goal else None,
            "active_goal_current_step_name": active_goal.steps[active_goal.current_step_index].get("name", f"Step {active_goal.current_step_index+1}") if active_goal and active_goal.steps and 0 <= active_goal.current_step_index < len(active_goal.steps) else None,
            "working_memory_depth": len(self.working_memory) if self.working_memory else 0,
            "verbose": self.verbose,  
        }
    
    def _reasoning_tool_recall_physical_plan(self, active_goal: GoalState) -> Optional[List[int]]:
        """
        Recalls a physical plan from LTM, checking for relevance against the
        currently active, arbitrated goal.
        """
        if not self.long_term_memory or not self.last_perceived_state or not active_goal:
            return None
        
        target_latent_vector = self.known_goal_latent_vectors.get(active_goal.current_goal)
        if target_latent_vector is None:
            return None

        current_latent = self.last_perceived_state.latent_vector
        confidence_threshold = self.internal_state_parameters['ltm_physical_plan_confidence_threshold']
        max_dist = self.internal_state_parameters['ltm_physical_plan_max_distance']
        dist_penalty = self.internal_state_parameters['ltm_physical_plan_distance_penalty_factor']

        candidates = []
        for key, data in self.long_term_memory.items():
            if data.get('type') == 'physical_action_sequence':
                plan_confidence = data.get('confidence', 0.0)
                if plan_confidence > confidence_threshold:
                    distance_from_start = np.linalg.norm(current_latent - data['start_latent_vector'])
                    if distance_from_start < max_dist:
                        imagined_latent = tf.convert_to_tensor(current_latent, dtype=tf.float32)
                        recalled_plan = list(key[1])
                        for action in recalled_plan:
                            imagined_latent = self.world_model.predict_next_latent_state(imagined_latent, action)
                        
                        final_imagined_vector = imagined_latent.numpy().flatten()
                        distance_from_goal = np.linalg.norm(final_imagined_vector - target_latent_vector)
                        
                        relevance_bonus = np.clip(1.0 - (distance_from_goal / 10.0), 0, 1.0) * 0.5
                        effective_score = (plan_confidence * 0.7) + (relevance_bonus * 0.3) - (distance_from_start * dist_penalty)
                        
                        candidates.append({
                            'plan': recalled_plan, 
                            'score': effective_score,
                            'relevance': relevance_bonus
                        })

        if not candidates:
            return None

        best_plan = max(candidates, key=lambda x: x['score'])
        self._log_lot_event("reasoning_tool", "recall_physical_plan_with_relevance", {
            "plan_len": len(best_plan['plan']), 
            "score": best_plan['score'],
            "relevance_bonus": best_plan['relevance']
        })
        return best_plan['plan']
    
    def _reasoning_tool_physical_shooting_planner(self, active_goal: GoalState) -> Optional[List[int]]:
        """
        A reasoning tool that uses mental simulation to find a sequence of actions
        to reach the target concept of the active goal's current step.
        """
        if not active_goal or not active_goal.steps or not self.last_perceived_state:
            return None
        
        target_latent_vector = None
        target_concept_name = "None"
        
        if 0 <= active_goal.current_step_index < len(active_goal.steps):
            current_step = active_goal.steps[active_goal.current_step_index]
            target_concept_name = current_step.get("target_concept")

            if target_concept_name:
                target_latent_vector = self.known_concept_vectors.get(target_concept_name)
                self._log_lot_event("reasoning_tool", "shooting_planner_start", {"goal": active_goal.current_goal, "target_concept": target_concept_name})
            else:
                return None # Step has no specific concept target, planner cannot run.
        else:
            return None # Invalid step index.

        if target_latent_vector is None:
            self._log_lot_event("reasoning_tool", "shooting_planner_fail", {"reason": "target_concept_not_grounded", "concept_name": target_concept_name})
            return None

        current_latent = self.last_perceived_state.latent_vector
        num_candidates = 32
        plan_length = 8
        best_plan, best_distance = None, float('inf')

        for _ in range(num_candidates):
            candidate_plan = [self.action_space.sample() for _ in range(plan_length)]
            imagined_latent = tf.convert_to_tensor(current_latent, dtype=tf.float32)

            for action in candidate_plan:
                imagined_latent = self.world_model.predict_next_latent_state(imagined_latent, action)
            
            distance = np.linalg.norm(imagined_latent.numpy().flatten() - target_latent_vector)

            if distance < best_distance:
                best_distance = distance
                best_plan = candidate_plan
        
        self._log_lot_event("reasoning_tool", "shooting_planner_end", {"plan_len": len(best_plan or []), "best_dist": best_distance})
        return best_plan
    
    def _reasoning_decompose_task(self, active_goal: GoalState) -> Optional[List[Dict]]:
        """
        Autonomous task decomposition, contextualized by the active goal.
        """
        self._log_lot_event("planner", "task_decomposition_start", {"goal": active_goal.current_goal})
        if not active_goal or not self.last_perceived_state:
            return None

        # This planner is designed for the 'MAIN_QUEST' with a 'target' concept
        if 'target' not in self.known_concept_vectors:
            self._log_lot_event("planner", "decompose_fail", {"reason": "final_target_concept_unknown"})
            return None
        
        # --- Stage 1: Discover Causal Preconditions for known obstacles ---
        door_preconditions = {}
        if 'door' in self.known_concept_vectors:
            for (action, precon_tuple), data in self.long_term_memory.items():
                if data.get('type') != 'causal_link': continue
                for (was_positive, concept, term), count in data['outcomes'].items():
                    if concept == 'door' and was_positive:
                        for key, value in precon_tuple:
                            door_preconditions[key] = value
                            
        # --- Stage 2: Assemble Plan Based on Known Causal Links and Concepts ---
        final_step = {"name": "Achieve Final Goal", "target_concept": "target"}
        
        if door_preconditions.get('has_key') is True:
            self._log_lot_event("planner", "decompose_insight", {"insight": "'has_key' is precondition for 'door'"})
            
            if 'key' in self.known_concept_vectors:
                key_step = {"name": "Acquire Key", "target_concept": "key"}
                door_step = {"name": "Pass Door", "target_concept": "door"}
                generated_steps = [key_step, door_step, final_step]
                
                if self.verbose >= 1: print(f"[{self.agent_id}] AUTONOMOUS PLAN (Full): Generated steps for '{active_goal.current_goal}': {[s['name'] for s in generated_steps]}")
                self._log_lot_event("planner", "decompose_success", {"plan_type": "full", "steps": [s['name'] for s in generated_steps]})
                return generated_steps
            else:
                unknown_key_step = {"name": "Find Prerequisite for Door", "target_concept": None} 
                door_step = {"name": "Pass Door", "target_concept": "door"}
                generated_steps = [unknown_key_step, door_step, final_step]

                if self.verbose >= 1: print(f"[{self.agent_id}] AUTONOMOUS PLAN (Partial): Generated steps for '{active_goal.current_goal}': {[s['name'] for s in generated_steps]}")
                self._log_lot_event("planner", "decompose_success", {"plan_type": "partial_unknown_key", "steps": [s['name'] for s in generated_steps]})
                return generated_steps

        elif 'door' in self.known_concept_vectors:
            door_step = {"name": "Solve Door Obstacle", "target_concept": "door"}
            generated_steps = [door_step, final_step]
            if self.verbose >= 1: print(f"[{self.agent_id}] AUTONOMOUS PLAN (Partial): Generated steps for '{active_goal.current_goal}': {[s['name'] for s in generated_steps]}")
            self._log_lot_event("planner", "decompose_success", {"plan_type": "partial_solve_door", "steps": [s['name'] for s in generated_steps]})
            return generated_steps
            
        else:
            generated_steps = [final_step]
            if self.verbose >= 1: print(f"[{self.agent_id}] AUTONOMOUS PLAN (Direct): Generated steps for '{active_goal.current_goal}': {[s['name'] for s in generated_steps]}")
            self._log_lot_event("planner", "decompose_success", {"plan_type": "direct_to_target", "steps": [s['name'] for s in generated_steps]})
            return generated_steps
        

    def _executive_formulate_cognitive_strategy(self, active_goal: Optional[GoalState]) -> Tuple[Optional[str], Optional[List[Any]]]:
        """
        The unified reasoning engine, refactored for logical clarity. It determines the agent's
        next cognitive state: Thinking, Acting, or Exploring.
        """
        self._log_lot_event("planner", "strategy_formulation_start", {"active_goal": str(active_goal)})

        # --- Step 1: Handle Meta-Cognitive States (Thinking or Idling) ---

        if active_goal is None:
            self._log_lot_event("planner", "no_active_goal_idle", {})
            self.cycles_stuck_on_step = 0
            return 'PHYSICAL', [self.action_space.sample()]

        if active_goal.goal_type == 'PLAN_FORMULATION':
            main_quest = next((g for g in self.goal_manager.goals if g.goal_type == 'MAIN_QUEST'), None)
            if main_quest:
                new_steps = self._reasoning_decompose_task(main_quest)
                if new_steps:
                    main_quest.steps = new_steps
                    # Planning was successful, the goal is complete.
                    self.goal_manager.goals = [g for g in self.goal_manager.goals if g is not active_goal]
            self.cycles_stuck_on_step = 0
            return 'ABSTRACT', None  # The action for this cycle is to "think".

        # --- Step 2: Update Progress Tracker & Check for Frustration ---

        # If the active goal is the same as last cycle, increment the stuck counter. Otherwise, reset it.
        if self.last_arbitrated_goal and active_goal.current_goal == self.last_arbitrated_goal.current_goal:
            self.cycles_stuck_on_step += 1
        else:
            self.cycles_stuck_on_step = 0
        
        main_quest = next((g for g in self.goal_manager.goals if g.goal_type == 'MAIN_QUEST'), None)
        if self.cycles_stuck_on_step > 40 and main_quest and active_goal.current_goal == main_quest.current_goal:
            if self.verbose >= 1: print(f"[{self.agent_id}] High frustration! Re-evaluating plan for Main Quest.")
            self._log_lot_event("planner", "replan_trigger", {"reason": "stuck_on_step", "goal": main_quest.current_goal})
            main_quest.steps = []  # WIPE THE FAILED PLAN
            main_quest.current_step_index = 0
            self.cycles_stuck_on_step = 0
            self.internal_state_parameters['frustration'] = min(0.95, self.internal_state_parameters['frustration'] + 0.4)
            # Wiping steps will trigger PLAN_FORMULATION on the next cycle. For this cycle, we explore.
            return 'PHYSICAL', [self.action_space.sample()]

        # --- Step 3: Formulate and Execute a Physical Action Plan ---
        
        plan = None
        # For INVESTIGATE goals, the plan is simple and prescribed.
        if active_goal.goal_type == 'INVESTIGATE':
            action_to_take = active_goal.steps[0].get('action_to_take')
            if action_to_take is not None:
                self.goal_manager.goals = [g for g in self.goal_manager.goals if g is not active_goal] # It's a one-shot goal.
                plan = [action_to_take]
        
        # For all other physical goals, use the shooting planner.
        elif active_goal.goal_type in ["MAIN_QUEST", "SURVIVAL", "OPPORTUNITY"]:
            if active_goal.steps:
                plan = self._reasoning_tool_physical_shooting_planner(active_goal)

        if plan:
            self._log_lot_event("planner", "physical_plan_found", {"goal": active_goal.current_goal})
            return 'PHYSICAL', plan
        else:
            # If no plan could be formed for any reason, the default is to explore.
            self._log_lot_event("planner", "explore_fallback", {"reason": "no_plan_formed", "active_goal": active_goal.current_goal})
            return 'PHYSICAL', [self.action_space.sample()]


    def save_state(self):
        """Saves the agent's experiential intelligence to a .pkl file."""
        filepath = f"{self.agent_id}.pkl"
        print(f"[{self.agent_id}] Saving experiential state to {filepath}...")
        try:
            # We don't save the deep learning models here, as they have their own save mechanism.
            state_data = {
                'long_term_memory': self.long_term_memory,
                'experience_replay_buffer': self.experience_replay_buffer.buffer if self.experience_replay_buffer else None,
                'known_concept_vectors': self.known_concept_vectors,
                'known_goal_latent_vectors': self.known_goal_latent_vectors,
                'goals': self.goal_manager.goals,
                'internal_state_parameters': self.internal_state_parameters,
                'metacognition_params': self.metacognition_params,
                'E_OR_THRESHOLD': self.E_OR_THRESHOLD,
                'orp_decay_rate': self.orp_decay_rate,
                'cycle_history': self.cycle_history,
                'current_cycle_num': self.current_cycle_num,
                'smn_influence_matrix': self.smn_influence_matrix
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
            print(f"[{self.agent_id}] State successfully saved.")
        except Exception as e:
            print(f"[{self.agent_id}] ERROR saving state: {e}")

    def load_state(self):
        """Loads the agent's experiential intelligence from a .pkl file if it exists."""
        filepath = f"{self.agent_id}.pkl"
        if os.path.exists(filepath):
            print(f"[{self.agent_id}] Found existing state file. Loading from {filepath}...")
            try:
                with open(filepath, 'rb') as f:
                    state_data = pickle.load(f)
                
                # Carefully restore the state
                self.long_term_memory.update(state_data.get('long_term_memory', {}))
                if self.experience_replay_buffer and state_data.get('experience_replay_buffer'):
                    self.experience_replay_buffer.buffer = state_data['experience_replay_buffer']
                self.known_concept_vectors = state_data.get('known_concept_vectors', self.known_concept_vectors)
                self.known_goal_latent_vectors = state_data.get('known_goal_latent_vectors', {})
                self.goal_manager.goals = state_data.get('goals', [])

                # Load dictionaries by updating, not replacing. This ensures that new parameters
                # added to default configs are present even when loading an older state file.
                loaded_isp = state_data.get('internal_state_parameters')
                if loaded_isp:
                    self.internal_state_parameters.update(loaded_isp)

                loaded_mp = state_data.get('metacognition_params')
                if loaded_mp:
                    self.metacognition_params.update(loaded_mp)
                
                # Restore simple values and other structures
                self.E_OR_THRESHOLD = state_data.get('E_OR_THRESHOLD', self.E_OR_THRESHOLD)
                self.orp_decay_rate = state_data.get('orp_decay_rate', self.orp_decay_rate)
                self.cycle_history = state_data.get('cycle_history', self.cycle_history)
                self.current_cycle_num = state_data.get('current_cycle_num', 0)
                if 'smn_influence_matrix' in state_data and self.smn_influence_matrix.shape == state_data['smn_influence_matrix'].shape:
                    self.smn_influence_matrix = state_data['smn_influence_matrix']
                
                print(f"[{self.agent_id}] State successfully loaded. Resuming from cycle {self.current_cycle_num}.")

            except Exception as e:
                print(f"[{self.agent_id}] ERROR loading state file (it might be corrupted). Starting fresh. Error: {e}")
        else:
            print(f"[{self.agent_id}] No existing state file found. Starting with a fresh mind.")
    

# cognitive_engine.py

    def run_cycle(self, raw_image_observation: np.ndarray, last_reward: float, last_info: Dict, is_terminated: bool, current_full_obs: Dict, future_latents: Dict = None) -> Optional[int]:
        """
        The primary cognitive-motor cycle. Processes environmental feedback,
        arbitrates between competing goals (including self-generated ones from PEML),
        formulates a strategy, and decides on the next action or thought.
        """
        last_action = self.cycle_history[-1]['action_taken'] if self.cycle_history and 'action_taken' in self.cycle_history[-1] else None

        # --- CYCLE START ---
        self.current_cycle_num += 1
        start_time = time.time()
        self.current_cycle_lot_stream = []
        self._log_lot_event("system", "cycle_start", {"cycle": self.current_cycle_num, "reward_in": last_reward})
        
        # Pass future latent states to goal manager for curiosity-driven goal generation
        self._future_latents_this_cycle = future_latents

        # --- 1. PROCESS WORLD STATE & PERCEIVE ---
        self._process_environmental_feedback(last_reward, is_terminated, last_info)
        self._update_internal_world_model(current_full_obs)
        prediction_error = self._ingest_sensory_data(raw_image_observation)

        # --- 2. LEARNING & MEMORY ---
        if last_action is not None and hasattr(self, '_last_raw_image_obs') and self.experience_replay_buffer:
            experience = {'state_img': self._last_raw_image_obs, 'action': last_action, 'next_state_img': raw_image_observation}
            if self.experience_replay_buffer.add(experience, self.last_cycle_valence_mod, prediction_error):
                self._log_lot_event("learning", "replay_buffer_add", {"valence": self.last_cycle_valence_mod, "surprise": prediction_error})
        self._last_raw_image_obs = raw_image_observation

        # --- 3. GOAL ARBITRATION & COGNITIVE HOUSEKEEPING ---
        if self.last_arbitrated_goal and self.last_perceived_state:
             self._executive_update_goal_progress(self.last_arbitrated_goal, self.last_perceived_state)
        self._firewall_detect_and_correct_anomalies()
        
        # The Goal Manager will use `_future_latents_this_cycle` via `getattr`
        arbitrated_goal = self.goal_manager.update_and_arbitrate()
        self.last_arbitrated_goal = arbitrated_goal

        # --- 4. EFFICIENT STRATEGY & ACTION ---
        action_to_take = None
        
        # --- THE EFFICIENT IMAGINATION FIX ---
        # Only run the expensive planner IF there is no current action plan.
        # Once a plan is made, the agent should commit to it.
        if not self.current_action_plan:
            plan_type, new_plan = self._executive_formulate_cognitive_strategy(arbitrated_goal)
            if plan_type == 'PHYSICAL' and new_plan:
                self.current_action_plan = collections.deque(new_plan)
        # --- END OF FIX ---

        # If a plan exists (either just created or pre-existing), execute the next step.
        if self.current_action_plan:
            # Check for high-urgency interrupts before acting.
            if arbitrated_goal and arbitrated_goal.goal_type == 'SURVIVAL':
                 urgency_for_interrupt = self.goal_manager._calculate_urgency(arbitrated_goal)
                 # Interrupt if survival becomes critical OR if the current plan is for something else
                 is_different_goal = self.last_arbitrated_goal.goal_type != 'SURVIVAL' if self.last_arbitrated_goal else True
                 if urgency_for_interrupt > 2.5 and is_different_goal:
                    self._log_lot_event("planner", "interrupt", {"reason": "survival_urgent", "old_plan_len": len(self.current_action_plan)})
                    self.current_action_plan.clear()
                    # Immediately formulate a NEW plan for survival.
                    _, survival_plan = self._executive_formulate_cognitive_strategy(arbitrated_goal)
                    if survival_plan:
                        self.current_action_plan = collections.deque(survival_plan)

            # If there's still a plan after the interrupt check, execute the next action.
            if self.current_action_plan:
                 action_to_take = self.current_action_plan.popleft()

        # --- 5. IMAGINATION & CYCLE WRAP-UP ---
        if action_to_take is not None and self.last_perceived_state:
            imagined_next_state_tensor = self.world_model.predict_next_latent_state(
                current_latent=tf.convert_to_tensor(self.last_perceived_state.latent_vector, dtype=tf.float32),
                action=action_to_take)
            self.last_imagined_next_state = imagined_next_state_tensor.numpy().flatten()
        elif self.last_perceived_state:
            self.last_imagined_next_state = self.last_perceived_state.latent_vector

        # Log cycle data
        focus = f"GOAL:{arbitrated_goal.goal_type}" if arbitrated_goal else "IDLE"
        if self.last_perceived_state:
            cycle_data = {
                'cycle_num': self.current_cycle_num, 'focus': focus, 'action_taken': action_to_take,
                'arbitrated_goal': str(arbitrated_goal),
                'perceived_state_handle_id': self.last_perceived_state.id,
                'reward_received': last_reward, 'valence_mod_this_cycle': self.last_cycle_valence_mod,
                'mood_after_cycle': self.internal_state_parameters['mood'],
                'surprise_this_cycle': prediction_error,
            }
            self.cycle_history.append(cycle_data)
        
        self._log_lot_event("system", "cycle_end", {"duration_ms": (time.time() - start_time) * 1000, "action": str(action_to_take)})

        if self.verbose >= 1 and self.current_cycle_num % 10 == 0:
            goal_statuses = [str(g) for g in self.goal_manager.goals]
            print(f"[{self.agent_id} C{self.current_cycle_num}] Goals: {goal_statuses}")
            print(f"[{self.agent_id} C{self.current_cycle_num}] Active Focus: {arbitrated_goal}")

        return action_to_take


# ---------------------------------------------------------------------------
# CoAgentManager (Feature 5: Parallel Cognitive Threads)
# ---------------------------------------------------------------------------
class CoAgentManager:
    def __init__(self, num_agents, base_emulator_config_template, agent_config_variations_list=None, trainable_params_config=None, verbose=0):
        self.num_agents = num_agents
        self.base_config = base_emulator_config_template
        self.agent_variations = agent_config_variations_list if agent_config_variations_list else []
        self.verbose = verbose
        self.trainable_params_config = copy.deepcopy(trainable_params_config) if trainable_params_config is not None else copy.deepcopy(DEFAULT_TRAINABLE_PARAMS_CONFIG)

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
            # Start with a deep copy of the base configuration template
            agent_kwargs = copy.deepcopy(self.base_config)
            agent_kwargs['agent_id'] = agent_id

            if i < len(self.agent_variations):
                agent_custom_settings = copy.deepcopy(self.agent_variations[i])

                # Get the variation for internal_state_parameters_config, if it exists
                variation_internal_params = agent_custom_settings.pop('internal_state_parameters_config', {})
                
                # Ensure the base config has an internal_state_parameters_config key to merge into
                if 'internal_state_parameters_config' not in agent_kwargs:
                    agent_kwargs['internal_state_parameters_config'] = copy.deepcopy(DEFAULT_INTERNAL_PARAMS)
                
                # Merge the variation's internal params into the agent's config
                agent_kwargs['internal_state_parameters_config'].update(variation_internal_params)

                # Now update the rest of the agent_kwargs with the remaining variations
                for key, value in agent_custom_settings.items():
                    if key in agent_kwargs and isinstance(agent_kwargs.get(key), dict) and isinstance(value, dict):
                        agent_kwargs[key].update(value)
                    else:
                        agent_kwargs[key] = value

            agent_kwargs['shared_long_term_memory'] = self.shared_long_term_memory
            agent_kwargs['shared_attention_foci'] = self.shared_attention_foci
            
            final_agent_verbose = agent_kwargs.get('verbose', self.verbose - 1 if self.verbose > 0 else 0)
            agent_kwargs['verbose'] = final_agent_verbose

            try:
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

            for agent_idx, agent in enumerate(self.agents):
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
        
        consensus_pref_state_from_top_agents = None
        top_agent_pref_states = []
        for top_idx in range(min(2, len(avg_performances))): 
            top_agent_obj = avg_performances[top_idx]['agent_obj']
            if top_agent_obj.internal_state_parameters['preferred_state_handle'] is not None:
                top_agent_pref_states.append(top_agent_obj.internal_state_parameters['preferred_state_handle'])
        
        if top_agent_pref_states:
            counts = collections.Counter(top_agent_pref_states)
            most_common_pref_state, _ = counts.most_common(1)[0]
            consensus_pref_state_from_top_agents = most_common_pref_state
            if self.verbose >=2 : print(f"    CoAgentManager: Consensus preferred_state from top agents: {consensus_pref_state_from_top_agents}")


        for i in range(num_learners):
            learner_idx_from_bottom = i
            learner_data_idx_in_sorted_list = len(avg_performances) - 1 - learner_idx_from_bottom
            
            if learner_data_idx_in_sorted_list <= 0: break 
            learner_data = avg_performances[learner_data_idx_in_sorted_list]
            if teacher_data['agent_id'] == learner_data['agent_id']: continue

            performance_gap_threshold_abs = 0.15 
            
            learner_agent = learner_data['agent_obj']
            teacher_agent = teacher_data['agent_obj']

            should_intervene_learner = teacher_data['perf'] > learner_data['perf'] + performance_gap_threshold_abs and learner_data['perf'] < 0.15

            if should_intervene_learner:
                if self.verbose >= 1: print(f"    {learner_agent.agent_id} (perf {learner_data['perf']:.2f}) learning from {teacher_agent.agent_id} (perf {teacher_data['perf']:.2f})")

                params_to_align = list(self.trainable_params_config.keys()) 
                alignment_factor = random.uniform(0.1, 0.25)
                teacher_params_for_reference = {}
                for param_name_for_teacher in params_to_align:
                    config_teacher = self.trainable_params_config[param_name_for_teacher]
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
                    config = self.trainable_params_config[param_name] 
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
                    learner_agent._log_lot_event("coagent", "learn_from_peer_params", {"teacher_id": teacher_agent.agent_id, "learner_perf":learner_data['perf'], "teacher_perf":teacher_data['perf'], "num_params_aligned": len(learner_current_params_for_update)})
                
                if consensus_pref_state_from_top_agents and learner_agent.internal_state_parameters['preferred_state_handle'] != consensus_pref_state_from_top_agents:
                    old_pref_state_learner = learner_agent.internal_state_parameters['preferred_state_handle']
                    learner_agent.internal_state_parameters['preferred_state_handle'] = consensus_pref_state_from_top_agents
                    if self.verbose >= 1: print(f"      {learner_agent.agent_id} preferred_state aligned to consensus {consensus_pref_state_from_top_agents} (was {old_pref_state_learner}).")
                    learner_agent._log_lot_event("coagent", "learn_from_peer_pref_state", {"teacher_id": "consensus_top_agents", "old_pref_state": str(old_pref_state_learner), "new_pref_state": str(consensus_pref_state_from_top_agents)})
                    copied_count +=1

                if learner_agent.smn_config.get('enabled', False):
                    old_smn_scale = learner_agent.internal_state_parameters['smn_perturbation_scale_factor']
                    new_smn_scale = min(old_smn_scale * 1.20, 0.2) 
                    if new_smn_scale > old_smn_scale + 1e-4 :
                        learner_agent.internal_state_parameters['smn_perturbation_scale_factor'] = new_smn_scale
                        if self.verbose >= 1: print(f"      {learner_agent.agent_id} SMN perturbation_scale_factor increased to {new_smn_scale:.4f} (was {old_smn_scale:.4f}).")
                        learner_agent._log_lot_event("coagent", "learn_from_peer_smn_boost", {"old_smn_scale":old_smn_scale, "new_smn_scale":new_smn_scale})
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
             print(f"    Last shared focus by {last_focus['agent_id']}: state {last_focus['state']}, valence {last_focus['valence']:.2f}, cycle {last_focus['cycle']}")

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

        task_pref_state_handle = self.base_emulator_config.get('internal_state_parameters_config', {}).get('preferred_state_handle')
        if task_pref_state_handle:
            emulator.internal_state_parameters['preferred_state_handle'] = task_pref_state_handle

        if task_goal_state_obj:
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
            
            current_episode_goal_obj = copy.deepcopy(goal_obj_template_for_episode) if goal_obj_template_for_episode else None


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