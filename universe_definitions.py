# universe_definitions.py
from typing import List, Dict
import numpy as np
from core_abstractions import StateHandle
from configurations import DEFAULT_VAE_CONFIG

PLACEHOLDER_LATENT_VECTOR = np.zeros(DEFAULT_VAE_CONFIG['LATENT_DIM'])

STATE_00 = StateHandle('00', latent_vector=PLACEHOLDER_LATENT_VECTOR)
STATE_01 = StateHandle('01', latent_vector=PLACEHOLDER_LATENT_VECTOR)
STATE_10 = StateHandle('10', latent_vector=PLACEHOLDER_LATENT_VECTOR)
STATE_11 = StateHandle('11', latent_vector=PLACEHOLDER_LATENT_VECTOR)

TWO_QUBIT_STATE_HANDLES: List[StateHandle] = [STATE_00, STATE_01, STATE_10, STATE_11]
TWO_QUBIT_STATE_HANDLE_BY_ID: Dict[str, StateHandle] = {handle.id: handle for handle in TWO_QUBIT_STATE_HANDLES}
TWO_QUBIT_VALENCE_MAP: Dict[StateHandle, float] = {STATE_00: 0.0, STATE_01: 0.5, STATE_10: -0.5, STATE_11: 1.0}
TWO_QUBIT_UNIVERSE_CONFIG = {"name": "2-Qubit Computational Basis", "states": TWO_QUBIT_STATE_HANDLES, "state_to_comp_basis": {state: state.id for state in TWO_QUBIT_STATE_HANDLES}, "comp_basis_to_state": {state.id: state for state in TWO_QUBIT_STATE_HANDLES}, "valence_map": TWO_QUBIT_VALENCE_MAP, "start_state": STATE_00}