# core_abstractions.py

"""
Contains the fundamental, world-agnostic data structures that form the 
foundation of the cognitive architecture. These classes define the concepts 
of states, memory, and goals, allowing the core engine to be independent
of any specific problem domain.
"""

import time
import collections
import copy
from typing import NamedTuple, List, Dict, Callable, Any
from dataclasses import dataclass, field
import numpy as np # <-- Needed for sanitizing in LogEntry


@dataclass
class LogEntry:
    """A structured entry for the agent's Language of Thought stream."""
    timestamp: float = field(default_factory=time.time)
    event_source: str = "SYSTEM"
    event_type: str = "GENERIC"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        # Format floats to be more readable
        formatted_details = {}
        for k, v in self.details.items():
            if isinstance(v, float):
                formatted_details[k] = f"{v:.3f}"
            else:
                formatted_details[k] = v

        detail_str = ", ".join(f"{k}={v}" for k, v in formatted_details.items())
        # Use strftime for consistent timestamp formatting and get milliseconds manually
        time_obj = time.localtime(self.timestamp)
        ms = f".{int((self.timestamp - int(self.timestamp)) * 1000):03d}"
        time_str = time.strftime('%H:%M:%S', time_obj) + ms

        return f"[{time_str}] [{self.event_source}:{self.event_type}] {detail_str}"


# ---------------------------------------------------------------------------
# State Abstraction Layer (The "Soul" That Can Inhabit Any "Body")
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateHandle:
    """
    An abstract, hashable handle for a state in the universe.
    The agent's cognitive logic now operates on these handles, which are
    grounded in the latent space of a self-supervised perception model.
    The `id` is a hashable representation of the latent vector for use in sets
    and dictionary keys.
    """
    id: str
    latent_vector: np.ndarray = field(hash=False, compare=False)
    properties: Dict = field(default_factory=dict, hash=False, compare=False)

    def __str__(self) -> str:
        # A more user-friendly representation for printing and logging.
        props_str = f", {len(self.properties)} props" if self.properties else ""
        # The ID is now a hash, so showing the first few characters is more useful.
        return f"State(id_hash={self.id[:8]}...{props_str})"

    def __repr__(self) -> str:
        # The formal representation, good for debugging.
        return f"StateHandle(id='{self.id}', latent_vector_shape={self.latent_vector.shape}, properties={self.properties})"


@dataclass
class SymbolicObject:
    """Represents a single perceived entity (e.g., an object in an ARC grid)."""
    id: str
    type: str = 'object'
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"SymbolicObject(id={self.id}, type={self.type}, attrs={list(self.attributes.keys())})"

class ActiveConceptNetGraph:
    """
    The Scientist's Whiteboard. A dynamic graph held in working memory,
    representing the agent's complete, structured understanding of the current
    problem state. It is built by the SymbolicPerceptionCore.
    """
    def __init__(self):
        self.nodes: Dict[str, SymbolicObject] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, node: SymbolicObject):
        """Adds a symbolic object to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, relation: str, **kwargs):
        """Adds a relationship edge between two nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.append({
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "attributes": kwargs
            })

    def __str__(self):
        return f"ConceptNet(Nodes: {len(self.nodes)}, Edges: {len(self.edges)})"

    @staticmethod
    def graph_diff(graph_before: 'ActiveConceptNetGraph', graph_after: 'ActiveConceptNetGraph') -> Dict[str, Any]:
        """
        The crucial "Diff" Operation. Compares two ConceptNet graphs to find
        the abstract changes, which becomes the input for hypothesis generation.
        """
        diff = {
            'nodes_added': [],
            'nodes_removed': [],
            'attributes_changed': [],
            'edges_added': [],
            'edges_removed': []
        }
        
        nodes_before_ids = set(graph_before.nodes.keys())
        nodes_after_ids = set(graph_after.nodes.keys())

        # Find added and removed nodes
        added_ids = nodes_after_ids - nodes_before_ids
        removed_ids = nodes_before_ids - nodes_after_ids
        
        for node_id in added_ids:
            diff['nodes_added'].append(graph_after.nodes[node_id].attributes)
        for node_id in removed_ids:
            diff['nodes_removed'].append(graph_before.nodes[node_id].attributes)
            
        # Find attribute changes on common nodes
        common_ids = nodes_before_ids.intersection(nodes_after_ids)
        for node_id in common_ids:
            obj_before = graph_before.nodes[node_id]
            obj_after = graph_after.nodes[node_id]
            
            all_keys = set(obj_before.attributes.keys()) | set(obj_after.attributes.keys())
            for key in all_keys:
                val_before = obj_before.attributes.get(key)
                val_after = obj_after.attributes.get(key)
                if str(val_before) != str(val_after): # Use string comparison for robustness
                    diff['attributes_changed'].append({
                        'id': node_id,
                        'attribute': key,
                        'from': val_before,
                        'to': val_after
                    })
        
        # Simple edge diff (can be made more sophisticated later)
        # For now, just detects if the number of edges of a certain type has changed.
        # A more detailed diff would compare specific edge connections.
        edges_before_summary = collections.Counter(e['relation'] for e in graph_before.edges)
        edges_after_summary = collections.Counter(e['relation'] for e in graph_after.edges)
        if edges_before_summary != edges_after_summary:
            diff['edges_changed_summary'] = {
                'from': dict(edges_before_summary),
                'to': dict(edges_after_summary)
                }

        return diff


# -----------------------------------------------------------------------------------------------
# Working Memory Components 
# -----------------------------------------------------------------------------------------------
class WorkingMemoryItem:
    """
    A single item to be stored in the WorkingMemoryStack, containing typed
    data and a description for context.
    """
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
    """
    A last-in, first-out (LIFO) stack representing the agent's short-term,
    active consciousness and scratchpad. It has a maximum depth to simulate
    cognitive limits.
    """
    def __init__(self, max_depth=20):
        self.stack = collections.deque(maxlen=max_depth)

    def push(self, item: WorkingMemoryItem) -> bool:
        """
        Pushes an item onto the stack. 
        Returns True if an old item was discarded to make space, False otherwise.
        """
        item_discarded_to_make_space = False
        if len(self.stack) == self.stack.maxlen and self.stack.maxlen > 0:
            item_discarded_to_make_space = True
        self.stack.append(item)
        return item_discarded_to_make_space

    def pop(self) -> WorkingMemoryItem | None:
        """Pops and returns the top item from the stack, or None if empty."""
        if not self.is_empty():
            item = self.stack.pop()
            return item
        return None

    def peek(self) -> WorkingMemoryItem | None:
        """Returns the top item from the stack without removing it, or None if empty."""
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self) -> bool:
        """Returns True if the working memory stack is empty."""
        return len(self.stack) == 0

    def clear(self):
        """Removes all items from the working memory stack."""
        self.stack.clear()

    def __len__(self):
        """Returns the current number of items in the stack."""
        return len(self.stack)

    def to_dict_summary(self):
        """Provides a dictionary summary of the working memory's current state."""
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
# GoalState Structure
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GoalState Structure
# ---------------------------------------------------------------------------
# core_abstractions.py

class GoalState:
    """
    Represents a structured goal, potentially with multiple steps, sub-goals,
    and complex completion criteria. This is the core data structure for
    the agent's goal-directed behavior. It also acts as a "cognitive primer"
    to influence the agent's reasoning process without explicit commands.
    """
    def __init__(self, current_goal: Any, 
                 steps: List[Dict],
                 goal_type: str = "MAIN_QUEST", # NEW: 'SURVIVAL', 'OPPORTUNITY'
                 base_priority: float = 0.5, # NEW: Default priority score
                 error_tolerance: float = 0.05,
                 initial_progress: float = 0.0,
                 # --- New "Cognitive Primer" properties ---
                 focus_concepts: List[StateHandle] = None,
                 reasoning_heuristic: str = None, # "ANALOGY", "LOGICAL_DEDUCTION", "CREATIVE_GENERATION"
                 evaluation_criteria: str = None  # "NOVELTY", "GOAL_COMPLETION"
                ):
        self.current_goal = current_goal
        self.steps = steps
        self.progress = initial_progress
        self.error_tolerance = error_tolerance
        self.current_step_index = 0
        self.status = "pending"  # pending, active, completed, failed
        self.history = []
        
        # New properties for arbitration and cognitive priming
        self.goal_type = goal_type
        self.base_priority = base_priority
        self.focus_concepts = focus_concepts if focus_concepts is not None else []
        self.reasoning_heuristic = reasoning_heuristic
        self.evaluation_criteria = evaluation_criteria

    def to_dict(self):
        """
        Serializes the GoalState object to a dictionary for logging and inspection.
        Handles nested sub-goals and non-serializable callables.
        """
        serializable_steps = []
        for step in self.steps:
            s_copy = step.copy()
            if callable(s_copy.get("completion_criteria")):
                s_copy["completion_criteria"] = "callable_function_not_serialized"
            if isinstance(s_copy.get("target_state"), StateHandle):
                s_copy["target_state"] = repr(s_copy["target_state"])
            if isinstance(s_copy.get("sub_goal"), GoalState):
                s_copy["sub_goal"] = s_copy["sub_goal"].to_dict()
            serializable_steps.append(s_copy)

        goal_desc = repr(self.current_goal) if isinstance(self.current_goal, StateHandle) else str(self.current_goal)

        return {
            "current_goal": goal_desc,
            "steps": serializable_steps,
            "progress": self.progress,
            "error_tolerance": self.error_tolerance,
            "current_step_index": self.current_step_index,
            "status": self.status,
            "history": self.history,
            "goal_type": self.goal_type,
            "base_priority": self.base_priority,
            "focus_concepts": [str(c) for c in self.focus_concepts],
            "reasoning_heuristic": self.reasoning_heuristic,
            "evaluation_criteria": self.evaluation_criteria,
        }

    def __str__(self) -> str:
        """Provides a readable string representation of the current goal status."""

        def get_step_display(goal, depth=0):
            if depth > 2: return "..."

            step_name_display = "None"
            if 0 <= goal.current_step_index < len(goal.steps):
                current_step = goal.steps[goal.current_step_index]
                step_name = current_step.get("name")
                if not step_name:
                    target = current_step.get("target_state")
                    step_name = str(target) if target else f"Step {goal.current_step_index + 1}"
                step_name_display = step_name

                sub_goal = current_step.get("sub_goal")
                if isinstance(sub_goal, GoalState) and sub_goal.status == "active":
                    sub_goal_desc = sub_goal.current_goal if not isinstance(sub_goal.current_goal, StateHandle) else str(sub_goal.current_goal)
                    sub_goal_display = get_step_display(sub_goal, depth + 1)
                    return f"{step_name_display} -> (SubGoal: '{sub_goal_desc}', Step: '{sub_goal_display}')"
            return step_name_display

        goal_desc_str = self.current_goal if not isinstance(self.current_goal, StateHandle) else str(self.current_goal)
        final_step_display = get_step_display(self)
        
        primer_str = []
        if self.reasoning_heuristic: primer_str.append(f"Heuristic:{self.reasoning_heuristic}")
        if self.evaluation_criteria: primer_str.append(f"Eval:{self.evaluation_criteria}")
        primer_info = f" Primer[{', '.join(primer_str)}]" if primer_str else ""

        return (f"Goal({self.goal_type}): '{goal_desc_str}' (Step: '{final_step_display}', "
                f"Prio: {self.base_priority:.2f}, Status: {self.status}){primer_info}")
    
    # ---------------------------------------------------------------------------
# >>> NEW: Pillar 2 - Language of Thought Abstractions
# ---------------------------------------------------------------------------

# A program is simply a list of function calls.
Program = List[Dict[str, Any]]

@dataclass
class Function:
    """Represents a function in the agent's LoT, either innate or learned."""
    name: str
    body: Callable # For innate functions, this is a direct Python callable.
    is_learned: bool = False
    docstring: str = ""
    # For learned functions, we store the original abstracted body
    learned_body: Program = field(default_factory=list)

class LanguageOfThought:
    """
    A library holding the agent's entire cognitive vocabulary. It contains both
    innate (hard-coded) functions and learned abstractions.
    """
    def __init__(self):
        self.innate_functions = self._get_innate_alphabet()
        self.learned_functions: Dict[str, Function] = {}

    def get_function(self, name: str) -> Function | None:
        """Retrieves a function by name from either the innate or learned set."""
        return self.learned_functions.get(name) or self.innate_functions.get(name)

    def _get_innate_alphabet(self) -> Dict[str, Function]:
        """Defines the hard-coded, primitive cognitive operations."""
        
        # Helper to create the standard function dictionary structure
        def make_func_call(func_name, *args):
             return {'function': func_name, 'args': list(args)}

        # Note: These functions do not EXECUTE the logic. They RETURN a
        # representation of the function call. The execution is handled by
        # an interpreter in Pillar 3. This keeps the language pure.
        
        # --- Data/Object Manipulation ---
        Move = Function('Move', lambda obj, dx, dy: make_func_call('Move', obj, dx, dy), docstring="Moves an object by a delta.")
        Copy = Function('Copy', lambda obj: make_func_call('Copy', obj), docstring="Copies an object.")
        Recolor = Function('Recolor', lambda obj, color: make_func_call('Recolor', obj, color), docstring="Changes the color of an object.")
        Delete = Function('Delete', lambda obj: make_func_call('Delete', obj), docstring="Deletes an object.")

        # --- Grid-level Manipulation ---
        FillGrid = Function('FillGrid', lambda color: make_func_call('FillGrid', color), docstring="Fills the entire grid with a single color.")
        SetGridProperty = Function('SetGridProperty', lambda prop, val: make_func_call('SetGridProperty', prop, val), docstring="Sets a global property of the grid, like background color.")

        # --- Data Access & Querying ---
        Filter = Function('Filter', lambda collection, predicate: make_func_call('Filter', collection, predicate), docstring="Filters a collection based on a condition.")
        GetProperty = Function('GetProperty', lambda entity, key: make_func_call('GetProperty', entity, key), docstring="Retrieves a property from an entity.")
        Count = Function('Count', lambda collection: make_func_call('Count', collection), docstring="Counts items in a collection.")
        
        # --- Control Flow ---
        Map = Function('Map', lambda collection, operation: make_func_call('Map', collection, operation), docstring="Applies an operation to each item in a collection.")

        innate_map = {f.name: f for f in [
            Move, Copy, Recolor, Delete,
            FillGrid, SetGridProperty,
            Filter, GetProperty, Count,
            Map
        ]}
        return innate_map

    def add_learned_function(self, new_function: Function):
        """Adds a new function created by the LambdaLearner to the LoT."""
        if new_function.name not in self.innate_functions:
            self.learned_functions[new_function.name] = new_function