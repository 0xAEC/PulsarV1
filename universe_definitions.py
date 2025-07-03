# universe_definitions.py

"""
Defines a specific "reality" for an agent to inhabit. This file contains the
state space, valence landscape, and any universe-specific functions. This
configuration is a self-contained bundle that can be plugged into the 
cognitive engine.
"""

from typing import List, Dict

from core_abstractions import StateHandle

# --- 2-Qubit Computational Basis Universe Definition ---

# 1. StateHandle Constants: Define each possible state as a unique, hashable object.
STATE_00 = StateHandle('00')
STATE_01 = StateHandle('01')
STATE_10 = StateHandle('10')
STATE_11 = StateHandle('11')

# 2. State Collections: Group the states for easy iteration and lookup.
TWO_QUBIT_STATE_HANDLES: List[StateHandle] = [STATE_00, STATE_01, STATE_10, STATE_11]
TWO_QUBIT_STATE_HANDLE_BY_ID: Dict[str, StateHandle] = {handle.id: handle for handle in TWO_QUBIT_STATE_HANDLES}

# 3. Valence Map: Define the intrinsic "feeling" or value associated with each state.
TWO_QUBIT_VALENCE_MAP: Dict[StateHandle, float] = {
    STATE_00: 0.0,
    STATE_01: 0.5,
    STATE_10: -0.5,
    STATE_11: 1.0,
}

# 4. NOTE: The 'transition function' mentioned in the directive is implemented within
#    the SimplifiedOrchOREmulator class as `_apply_logical_op_to_superposition`,
#    as it describes the "physics" of the engine's operations rather than being a
#    property of the state-space definition itself. No separate function is defined here.

# 5. Universe Configuration Bundle: Aggregate all components into a single dictionary
#    that the cognitive engine will use to initialize an agent's reality.
TWO_QUBIT_UNIVERSE_CONFIG = {
    "name": "2-Qubit Computational Basis",
    "states": TWO_QUBIT_STATE_HANDLES,
    "state_to_comp_basis": {
        state: state.id for state in TWO_QUBIT_STATE_HANDLES
    },
    "comp_basis_to_state": {
        state.id: state for state in TWO_QUBIT_STATE_HANDLES
    },
    "valence_map": TWO_QUBIT_VALENCE_MAP,
    "start_state": STATE_00,
}

# An example of an alternative universe the agent could inhabit without code changes.
# This is left here as a reference but is not part of the primary 2-qubit definition.
COLOR_UNIVERSE_REFERENCE_EXAMPLE = {
    "name": "Simple Color World",
    "states": [StateHandle('RED'), StateHandle('GREEN'), StateHandle('BLUE'), StateHandle('YELLOW')],
    "state_to_comp_basis": {
        StateHandle('RED'):    '00', StateHandle('GREEN'):  '01',
        StateHandle('BLUE'):   '10', StateHandle('YELLOW'): '11'
    },
    "comp_basis_to_state": {
        '00': StateHandle('RED'),    '01': StateHandle('GREEN'),
        '10': StateHandle('BLUE'),   '11': StateHandle('YELLOW')
    },
    "valence_map": {
        StateHandle('RED'):   -0.8, StateHandle('GREEN'):  1.0,
        StateHandle('BLUE'):  -0.2, StateHandle('YELLOW'): 0.3
    },
    "start_state": StateHandle('RED')
}
