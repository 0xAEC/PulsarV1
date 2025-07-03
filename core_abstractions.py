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

# ---------------------------------------------------------------------------
# State Abstraction Layer (The "Soul" That Can Inhabit Any "Body")
# ---------------------------------------------------------------------------
class StateHandle(NamedTuple):
    """
    An abstract, hashable handle for a state in the universe.
    The agent's cognitive logic should operate on these handles, not on
    their underlying string representations, allowing the cognitive architecture
    to be independent of the specific problem space (e.g., bitstrings, colors, etc.).
    """
    id: str

    def __str__(self) -> str:
        # A more user-friendly representation for printing and logging.
        return f"State({self.id})"

    def __repr__(self) -> str:
        # The formal representation, good for debugging.
        return f"StateHandle(id='{self.id}')"


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
class GoalState:
    """
    Represents a structured goal, potentially with multiple steps, sub-goals,
    and complex completion criteria. This is the core data structure for
    the agent's goal-directed behavior.
    """
    def __init__(self, current_goal: Any, steps: List[Dict], error_tolerance: float = 0.05, initial_progress: float = 0.0):
        # The 'current_goal' is a high-level description or a target StateHandle itself.
        # Step 'target_state' values should be StateHandle objects.
        self.current_goal = current_goal
        self.steps = steps  # List of step dictionaries
        self.progress = initial_progress
        self.error_tolerance = error_tolerance
        self.current_step_index = 0
        self.status = "pending"  # pending, active, completed, failed
        self.history = []

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
                s_copy["target_state"] = repr(s_copy["target_state"])  # Serialize StateHandle to string
            # If a step itself is a GoalState, recursively serialize
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
        }

    def __str__(self) -> str:
        """Provides a readable string representation of the current goal status."""
        
        def get_step_display(goal, depth=0):
            """Internal helper to recursively display the current step, including sub-goals."""
            if depth > 2:  # Limit recursion to prevent absurdly long strings
                return "..."

            step_name_display = "None"
            if 0 <= goal.current_step_index < len(goal.steps):
                current_step = goal.steps[goal.current_step_index]
                # Default display uses the step 'name', or target StateHandle if name is not present.
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
        return f"Goal: '{goal_desc_str}' (Step: '{final_step_display}', Progress: {self.progress * 100:.1f}%, Status: {self.status})"
