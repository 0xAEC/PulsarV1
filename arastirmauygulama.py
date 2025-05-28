import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.exceptions import QiskitError
import copy
import time # For potential timing/performance checks if needed
import random # For more explicit random choices

# ---------------------------------------------------------------------------
# Class Definition: ToricCode (Generalized Toric code and Quantum error correction)
# ---------------------------------------------------------------------------
class ToricCode:
    """
    Implements a Toric Code on an Lx x Ly lattice.
    Handles qubit definition, stabilizer generation, logical operator definition,
    and basic quantum operations (preparation, measurement, error injection).
    """
    def __init__(self, Lx=2, Ly=2):
        if Lx < 2 or Ly < 2:
            raise ValueError("Lx and Ly must be at least 2 for a valid toric code.")
        self.Lx = Lx
        self.Ly = Ly
        
        # Data qubits: 2 per vertex in an Lx * Ly grid of vertices.
        # num_h_qubits = Ly * Lx (horizontal edges)
        # num_v_qubits = Ly * Lx (vertical edges)
        self.num_data_qubits = 2 * self.Lx * self.Ly
        
        # Qubit mapping:
        # h_qubits_map[(r, c)]: horizontal qubit to the "east" of vertex (r,c)
        # v_qubits_map[(r, c)]: vertical qubit "south" of vertex (r,c)
        # Vertex indices: r from 0 to Ly-1, c from 0 to Lx-1
        self.h_qubits_map = { 
            (r, c): r * self.Lx + c 
            for r in range(self.Ly) for c in range(self.Lx) 
        }
        self.v_qubits_map = { 
            (r, c): (self.Lx * self.Ly) + (r * self.Lx + c) 
            for r in range(self.Ly) for c in range(self.Lx) 
        }
        
        self.star_operators_indices = [] 
        self.plaquette_operators_indices = []
        self._define_stabilizers()
        
        self.logical_Z_indices = [None, None] 
        self.logical_X_indices = [None, None] 
        self._define_logical_operators()
        
        self.data_qreg = QuantumRegister(self.num_data_qubits, 'data')
        self.anc_qreg = QuantumRegister(1, 'anc') 
        
        # Number of stabilizers: Lx*Ly stars + Lx*Ly plaquettes
        self.num_stabilizers = len(self.star_operators_indices) + len(self.plaquette_operators_indices)
        self.stab_creg = ClassicalRegister(self.num_stabilizers, 'stab_syndrome')
        self.log_creg = ClassicalRegister(2, 'log_readout') # For 2 logical qubits
        
        self.simulator = AerSimulator(method='statevector')

    def get_qubit_indices(self, type_char, r, c):
        """
        Get the flat index for a qubit given its type ('h' or 'v') and lattice coordinates (r,c).
        Handles periodic boundary conditions for coordinates.
        """
        actual_r, actual_c = r % self.Ly, c % self.Lx
        if type_char == 'h': return self.h_qubits_map[(actual_r, actual_c)]
        if type_char == 'v': return self.v_qubits_map[(actual_r, actual_c)]
        raise ValueError("type_char must be 'h' or 'v'")

    def _define_stabilizers(self):
        """
        Defines star (X-type) and plaquette (Z-type) stabilizer operators.
        Star operators are centered on vertices.
        Plaquette operators are centered on faces.
        """
        self.star_operators_indices = []
        self.plaquette_operators_indices = []

        # Star operators (X stabilizers) - one for each vertex (vert_r, vert_c)
        # Involves 4 qubits incident to the vertex:
        # h-qubit to its east, h-qubit to its west
        # v-qubit below it, v-qubit above it
        for vert_r in range(self.Ly):
            for vert_c in range(self.Lx):
                star_op = sorted(list(set([
                    self.get_qubit_indices('h', vert_r, vert_c),      # h-qubit east of (vert_r, vert_c)
                    self.get_qubit_indices('h', vert_r, vert_c - 1),  # h-qubit west of (vert_r, vert_c)
                    self.get_qubit_indices('v', vert_r, vert_c),      # v-qubit south of (vert_r, vert_c)
                    self.get_qubit_indices('v', vert_r - 1, vert_c)   # v-qubit north of (vert_r, vert_c)
                ])))
                self.star_operators_indices.append(star_op)

        # Plaquette operators (Z stabilizers) - one for each plaquette
        # Plaquette identified by its top-left vertex (plaq_r, plaq_c)
        # Involves 4 qubits forming the boundary of the plaquette:
        # h-qubit on top, v-qubit on right
        # h-qubit on bottom, v-qubit on left
        for plaq_r in range(self.Ly):
            for plaq_c in range(self.Lx):
                plaquette_op = sorted(list(set([
                    self.get_qubit_indices('h', plaq_r, plaq_c),        # Top h-qubit
                    self.get_qubit_indices('v', plaq_r, plaq_c + 1),    # Right v-qubit
                    self.get_qubit_indices('h', plaq_r + 1, plaq_c),    # Bottom h-qubit
                    self.get_qubit_indices('v', plaq_r, plaq_c)         # Left v-qubit
                ])))
                self.plaquette_operators_indices.append(plaquette_op)

    def _define_logical_operators(self):
        """
        Defines two pairs of logical Z and logical X operators.
        These correspond to non-contractible loops on the torus.
        - Z_L0 (or Z_L1 by some conventions): String of Zs on horizontal qubits along a row (wraps around y-axis).
        - X_L0 (or X_L1): String of Xs on vertical qubits along a column (wraps around x-axis). (Dual to Z_L0)
        - Z_L1 (or Z_L2): String of Zs on horizontal qubits along a column (wraps around x-axis).
        - X_L1 (or X_L2): String of Xs on vertical qubits along a row (wraps around y-axis). (Dual to Z_L1)
        The choice here follows common conventions.
        logical_Z_indices[0] and logical_X_indices[0] form one logical qubit.
        logical_Z_indices[1] and logical_X_indices[1] form the second.
        """
        # Logical Qubit 0 (often called Z_1, X_1 or Z_y, X_x)
        # Z_L0: Path of Z operators on horizontal links winding vertically (e.g., a column of h-links)
        # This interpretation can vary. The key is consistent anti-commutation.
        # Original prompt version was:
        # Z_L0 (h-qubits in row 0): loop in y-direction of Z ops
        self.logical_Z_indices[0] = sorted([self.get_qubit_indices('h', 0, c) for c in range(self.Lx)])
        # X_L0 (v-qubits in col 0): loop in x-direction of X ops
        self.logical_X_indices[0] = sorted([self.get_qubit_indices('v', r, 0) for r in range(self.Ly)])

        # Logical Qubit 1 (often called Z_2, X_2 or Z_x, X_y)
        # Z_L1 (h-qubits in col 0): loop in x-direction of Z ops
        self.logical_Z_indices[1] = sorted([self.get_qubit_indices('h', r, 0) for r in range(self.Ly)])
        # X_L1 (v-qubits in row 0): loop in y-direction of X ops
        self.logical_X_indices[1] = sorted([self.get_qubit_indices('v', 0, c) for c in range(self.Lx)])
        
        # Sanity check: print definitions if needed
        # print(f"ToricCode {self.Lx}x{self.Ly}: Z0:{self.logical_Z_indices[0]}, X0:{self.logical_X_indices[0]}")
        # print(f"ToricCode {self.Lx}x{self.Ly}: Z1:{self.logical_Z_indices[1]}, X1:{self.logical_X_indices[1]}")

    def prepare_ground_state_00L_specific_2x2(self, qc):
        """
        Prepares the logical |00>_L state specifically for a 2x2 Toric Code.
        This sequence is from the original user-provided code.
        WARNING: This is hardcoded for 2x2 (8 data qubits).
        """
        if self.Lx != 2 or self.Ly != 2:
            raise NotImplementedError("prepare_ground_state_00L_specific_2x2 is only for 2x2 codes.")
        # This sequence from original code prepares a +1 eigenstate of all stabilizers.
        qc.h(self.data_qreg[0]); qc.cx(self.data_qreg[0], self.data_qreg[1]); qc.cx(self.data_qreg[0], self.data_qreg[4]); qc.cx(self.data_qreg[0], self.data_qreg[6]); qc.barrier()
        qc.h(self.data_qreg[5]); qc.cx(self.data_qreg[5], self.data_qreg[0]); qc.cx(self.data_qreg[5], self.data_qreg[1]); qc.cx(self.data_qreg[5], self.data_qreg[7]); qc.barrier()
        qc.h(self.data_qreg[2]); qc.cx(self.data_qreg[2], self.data_qreg[3]); qc.cx(self.data_qreg[2], self.data_qreg[4]); qc.cx(self.data_qreg[2], self.data_qreg[6]); qc.barrier()

    def prepare_general_ground_state(self, qc):
        """
        Prepares a general ground state, e.g., |0...0> for Z-basis logical measurements.
        For simulation, |0...0> is a +1 eigenstate of all Z-plaquettes.
        A full ground state (+1 for all stabilizers) prep is more complex.
        This is sufficient for many simulation purposes where an initial logical state is set.
        """
        # Initialize all data qubits to |0>. This state has +1 eigenvalue for all Z-type (plaquette) stabilizers.
        # To also satisfy X-type (star) stabilizers, further operations (e.g. H on all then CNOTs) are needed.
        # For the abstract emulator, exact physical ground state is less critical than having *a* valid code state.
        # For now, this simple initialization is a placeholder for a more rigorous general state prep.
        for i in range(self.num_data_qubits):
            qc.reset(self.data_qreg[i]) # Ensures |0> state.
        qc.barrier()

    def get_initialization_circuit(self):
        """
        Returns a quantum circuit initialized in a logical ground state.
        Uses specific 2x2 prep if applicable, otherwise general prep.
        """
        qc = QuantumCircuit(self.data_qreg, self.anc_qreg, self.stab_creg, self.log_creg)
        if self.Lx == 2 and self.Ly == 2:
            self.prepare_ground_state_00L_specific_2x2(qc)
        else:
            # For general Lx, Ly, preparing the true ground state (all stabilizers +1) is non-trivial.
            # The |0...0> state satisfies all Z stabilizers. H |0...0> satisfies all X stabilizers.
            # Using a simplified state preparation that ensures a valid logical state can be encoded.
            # For the purposes of this emulator, where we manipulate logical states abstractly and
            # only use physical layer for "grounding checks", exact ground state is secondary.
            self.prepare_general_ground_state(qc)
            # print(f"Warning: Using simplified general ground state for {self.Lx}x{self.Ly} Toric Code. "
            #       "This may not be a +1 eigenstate of all X-stabilizers.")
        return qc
        
    def _measure_one_stabilizer(self, qc, stabilizer_q_indices, stabilizer_type, cbit_idx_to_store):
        qc.reset(self.anc_qreg[0])
        if stabilizer_type.upper() == 'X': # Star operators are X-type
            for q_idx in stabilizer_q_indices: qc.h(self.data_qreg[q_idx])
            for q_idx in stabilizer_q_indices: qc.cx(self.data_qreg[q_idx], self.anc_qreg[0])
            for q_idx in stabilizer_q_indices: qc.h(self.data_qreg[q_idx]) # unHadamard
        elif stabilizer_type.upper() == 'Z': # Plaquette operators are Z-type
            for q_idx in stabilizer_q_indices: qc.cx(self.data_qreg[q_idx], self.anc_qreg[0])
        else: raise ValueError("stabilizer_type must be 'X' or 'Z'.")
        qc.measure(self.anc_qreg[0], self.stab_creg[cbit_idx_to_store])
        qc.barrier(self.data_qreg, self.anc_qreg)

    def measure_all_stabilizers(self, qc):
        cbit_idx = 0
        for indices in self.star_operators_indices: 
            self._measure_one_stabilizer(qc, indices, 'X', cbit_idx)
            cbit_idx += 1
        for indices in self.plaquette_operators_indices: 
            self._measure_one_stabilizer(qc, indices, 'Z', cbit_idx)
            cbit_idx += 1

    def _measure_one_logical_Z(self, qc, logical_Z_q_indices, log_cbit_idx):
        qc.reset(self.anc_qreg[0]) 
        for q_idx in logical_Z_q_indices: qc.cx(self.data_qreg[q_idx], self.anc_qreg[0])
        qc.measure(self.anc_qreg[0], self.log_creg[log_cbit_idx])
        qc.barrier(self.data_qreg, self.anc_qreg) 

    def measure_all_logical_Zs(self, qc):
        # Measures Z_L0 (logical_Z_indices[0]) into log_creg[0]
        # Measures Z_L1 (logical_Z_indices[1]) into log_creg[1]
        self._measure_one_logical_Z(qc, self.logical_Z_indices[0], 0) 
        self._measure_one_logical_Z(qc, self.logical_Z_indices[1], 1) 

    def apply_logical_op_on_qc(self, qc, op_char, logical_qubit_idx):
        op_map = {'X': (self.logical_X_indices, qc.x), 'Z': (self.logical_Z_indices, qc.z)}
        if op_char.upper() not in op_map or logical_qubit_idx not in [0,1]:
            raise ValueError(f"Invalid logical op {op_char} or index {logical_qubit_idx}")
        indices_list, gate_func = op_map[op_char.upper()]
        qubits_to_act_on = indices_list[logical_qubit_idx]
        for q_idx in qubits_to_act_on: gate_func(self.data_qreg[q_idx])
        qc.barrier(self.data_qreg)
        
    def apply_physical_error(self, qc, errors_to_inject):
        if errors_to_inject:
            for gate_char, qubit_idx in errors_to_inject:
                if not (0 <= qubit_idx < self.num_data_qubits): 
                    raise ValueError(f"Invalid qubit index {qubit_idx} for {self.num_data_qubits} data qubits.")
                op_map = {'X': qc.x, 'Y': qc.y, 'Z': qc.z}
                if gate_char.upper() not in op_map: raise ValueError(f"Invalid error gate {gate_char}")
                op_map[gate_char.upper()](self.data_qreg[qubit_idx])
            qc.barrier(self.data_qreg)

    def _perform_simulation_and_parse(self, qc, measure_logicals=True, shots=1, verbose=0):
        transpiled_qc = transpile(qc, self.simulator)
        result = self.simulator.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts(qc) if hasattr(result, 'get_counts') else {}
        
        full_bitstring_from_counts = max(counts, key=counts.get) if counts else ""
        if verbose >= 3 and counts: print(f"    Raw counts: {counts}, Selected BS: {full_bitstring_from_counts}")
        
        syndrome_bs = "E" * self.num_stabilizers
        log_Z_bs = "E" * self.log_creg.size if measure_logicals else None

        if not full_bitstring_from_counts: 
            if verbose >= 1: print("    Warning: No counts from simulation.")
            return syndrome_bs, log_Z_bs

        parts = full_bitstring_from_counts.split(' ')
        raw_syndrome_part = ""
        raw_log_part = ""

        # Expected Qiskit bitstring order for 'log_creg stab_creg': "value_for_log_creg value_for_stab_creg"
        # And log_creg[1] log_creg[0] etc.
        # If qc.cregs = [stab_creg, log_creg], then typical qiskit result string is "log_val stab_val"
        # log_creg bits: c1 c0 (so index 1, index 0)
        # stab_creg bits: s_N-1 ... s_0
        # Parsing based on circuit definition: stab_creg, then log_creg
        # qc = QuantumCircuit(..., self.stab_creg, self.log_creg)
        # Means log_creg classical bits have higher indices overall. So they appear first in Qiskit string.
        # E.g., 'log_c1log_c0 stab_s(N-1)...stab_s0'
        if measure_logicals:
            if len(parts) == 2: 
                raw_log_part = parts[0]    # Corresponds to self.log_creg
                raw_syndrome_part = parts[1] # Corresponds to self.stab_creg
            elif len(parts) == 1 and len(full_bitstring_from_counts) == self.log_creg.size + self.num_stabilizers: 
                raw_log_part = full_bitstring_from_counts[:self.log_creg.size]
                raw_syndrome_part = full_bitstring_from_counts[self.log_creg.size:]
            else:
                if verbose >= 1: print(f"    Warning: Could not parse bitstring '{full_bitstring_from_counts}' with measure_logicals=True.")

            if len(raw_log_part) == self.log_creg.size: log_Z_bs = raw_log_part # log_Z_bs should be "ZL1_val ZL0_val"
            if len(raw_syndrome_part) == self.num_stabilizers: syndrome_bs = raw_syndrome_part[::-1] # Reverse for s0s1... order
            
        else: # measure_logicals is False, only syndrome expected
            raw_syndrome_part_candidate = full_bitstring_from_counts.replace(" ", "")
            if len(raw_syndrome_part_candidate) == self.num_stabilizers:
                syndrome_bs = raw_syndrome_part_candidate[::-1]
                log_Z_bs = None
            elif len(raw_syndrome_part_candidate) == self.num_stabilizers + self.log_creg.size :
                # If log_creg bits are still there but unwanted, try to extract syndrome based on typical order
                syndrome_bs = raw_syndrome_part_candidate[self.log_creg.size:][::-1] # Assume log bits first
                log_Z_bs = None 
                if verbose >= 2: print(f"    Info: Full bitstring '{full_bitstring_from_counts}' contained log bits; extracting syndrome only.")
            else:
                if verbose >= 1: print(f"    Warning: Could not parse syndrome-only bitstring '{full_bitstring_from_counts}'.")
        
        if "E" in syndrome_bs or (measure_logicals and log_Z_bs and "E" in log_Z_bs):
             if verbose >= 1: print(f"    Warning: Error in parsing, 'E' found. Syndrome: {syndrome_bs}, Log: {log_Z_bs}")

        return syndrome_bs, log_Z_bs

    def __str__(self):
        return (f"ToricCode{self.Lx}x{self.Ly} (DataQ: {self.num_data_qubits}, Stab: {self.num_stabilizers}, LogQ: 2)\n"
                f"  Z_L0 (indices): {self.logical_Z_indices[0]}, X_L0 (indices): {self.logical_X_indices[0]}\n"
                f"  Z_L1 (indices): {self.logical_Z_indices[1]}, X_L1 (indices): {self.logical_X_indices[1]}")


# ---------------------------------------------------------------------------
# Orch OR Emulator Defaults
# ---------------------------------------------------------------------------
DEFAULT_INTERNAL_PARAMS = {
    'curiosity': 0.5, 'goal_seeking_bias': 0.3, 
    'preferred_L0_val': None, 'preferred_L1_val': None,
    'computation_length_preference': 3, # Average number of ops per sequence
    'attention_level': 1.0, # 0.0 (low) to 1.0 (high)
    'cognitive_load': 0.0, # 0.0 (low) to 1.0 (high)
    'mood': 0.0, # -1.0 (neg) to 1.0 (pos)
}
DEFAULT_METACOGNITION_PARAMS = {
    'review_interval': 10, # cycles
    'cycles_since_last_review': 0,
    'curiosity_adaptation_rate': 0.05,
    'goal_bias_adaptation_rate': 0.05,
    'low_valence_threshold': -0.2, # For triggering metacognitive adjustments
    'high_valence_threshold': 0.7,
    'exploration_threshold': 0.1, # Entropy or state diversity based
    'enable_threshold_adaptation': False, # Whether E_OR_THRESHOLD can change
    'enable_decay_adaptation': False, # Whether orp_decay_rate can change
}
DEFAULT_ORP_THRESHOLD_DYNAMICS = {
    'min': 0.5, 'max': 2.5, 'adapt_rate': 0.02, # How much E_OR_THRESHOLD can change per review
}
DEFAULT_ORP_DECAY_DYNAMICS = {
    'min': 0.0, 'max': 0.15, 'adapt_rate': 0.005, # How much orp_decay_rate can change
}

# ---------------------------------------------------------------------------
# Class Definition: SimplifiedOrchOREmulator
# ---------------------------------------------------------------------------
class SimplifiedOrchOREmulator:
    def __init__(self, Lx=2, Ly=2, cycle_history_max_len=50,
                 initial_E_OR_THRESHOLD=1.0, initial_orp_decay_rate=0.0,
                 initial_internal_states=None, metacognition_config=None,
                 orp_threshold_dynamics_config=None, orp_decay_dynamics_config=None,
                 verbose=0):
        
        self.toric_code = ToricCode(Lx, Ly)
        self.logical_superposition = {"00": 1.0 + 0j} 
        self.collapsed_logical_state_str = "00" # ZL1ZL0 format
        self.objective_reduction_potential = 0.0
        self.E_OR_THRESHOLD = initial_E_OR_THRESHOLD
        self.orp_decay_rate = initial_orp_decay_rate
        
        self.operation_costs = { 
            'X': 0.1, 'Z': 0.1, 'H': 0.4, 'CNOT': 0.5, 'CZ': 0.5, 'ERROR': 0.05 # Cost for simulated error
        }
        self.outcome_valence = {"00": 0.0, "01": 0.5, "10": -0.5, "11": 1.0} 
        self.last_cycle_valence = 0.0
        self.current_orp_before_reset = 0.0

        self.internal_state_parameters = copy.deepcopy(DEFAULT_INTERNAL_PARAMS)
        if initial_internal_states: self.internal_state_parameters.update(initial_internal_states)

        self.metacognition_params = copy.deepcopy(DEFAULT_METACOGNITION_PARAMS)
        if metacognition_config: self.metacognition_params.update(metacognition_config)
        
        self.orp_threshold_dynamics = copy.deepcopy(DEFAULT_ORP_THRESHOLD_DYNAMICS)
        if orp_threshold_dynamics_config: self.orp_threshold_dynamics.update(orp_threshold_dynamics_config)

        self.orp_decay_dynamics = copy.deepcopy(DEFAULT_ORP_DECAY_DYNAMICS)
        if orp_decay_dynamics_config: self.orp_decay_dynamics.update(orp_decay_dynamics_config)

        self.long_term_memory = {} # Stores (op_sequence_tuple): {'count': N, 'total_valence': V_sum, 'avg_valence': V_avg}
        self.long_term_memory_capacity = 50 # Max number of distinct sequences to remember
        self.successful_sequence_threshold = 0.6 # Valence needed for a sequence to be "successful" for memory

        self.cycle_history = [] 
        self.cycle_history_max_len = cycle_history_max_len
        self.current_cycle_num = 0
        self.verbose = verbose

        if self.verbose >= 1: print(f"Simplified Orch OR Emulator Initialized (Toric Code: {Lx}x{Ly}, Max Hist: {cycle_history_max_len}).")

    def _apply_logical_op_to_superposition(self, op_char, logical_arg, current_superposition, current_orp):
        new_superposition = {state: 0.0j for state in ["00", "01", "10", "11"]}
        new_orp = current_orp 
        sqrt2_inv = 1/np.sqrt(2)
        op_char_upper = op_char.upper()
        new_orp += self.operation_costs.get(op_char_upper, 0.05) 

        for basis_state_str, amp in current_superposition.items(): 
            if amp == 0j: continue
            lq0_val, lq1_val = int(basis_state_str[1]), int(basis_state_str[0]) # ZL1ZL0: L0 is state[1], L1 is state[0]

            if op_char_upper == 'X':
                idx_to_flip = logical_arg 
                if idx_to_flip == 0: new_basis_state_str = f"{lq1_val}{1-lq0_val}"
                else: new_basis_state_str = f"{1-lq1_val}{lq0_val}"
                new_superposition[new_basis_state_str] += amp
            elif op_char_upper == 'Z':
                idx_to_phase = logical_arg 
                phase = -1 if ((idx_to_phase == 0 and lq0_val == 1) or \
                               (idx_to_phase == 1 and lq1_val == 1)) else 1
                new_superposition[basis_state_str] += amp * phase
            elif op_char_upper == 'H':
                idx_to_h = logical_arg
                if idx_to_h == 0: # H on L0
                    s0_str, s1_str = f"{lq1_val}{0}", f"{lq1_val}{1}"
                    new_superposition[s0_str] += amp * sqrt2_inv * (+1 if lq0_val == 0 else +1)
                    new_superposition[s1_str] += amp * sqrt2_inv * (+1 if lq0_val == 0 else -1)
                else: # H on L1
                    s0_str, s1_str = f"{0}{lq0_val}", f"{1}{lq0_val}"
                    new_superposition[s0_str] += amp * sqrt2_inv * (+1 if lq1_val == 0 else +1)
                    new_superposition[s1_str] += amp * sqrt2_inv * (+1 if lq1_val == 0 else -1)
            elif op_char_upper == 'CNOT':
                ctrl_idx, target_idx = logical_arg
                control_active = (lq0_val == 1 if ctrl_idx == 0 else lq1_val == 1)
                if control_active:
                    new_lq0_val_temp = 1 - lq0_val if target_idx == 0 else lq0_val
                    new_lq1_val_temp = 1 - lq1_val if target_idx == 1 else lq1_val
                    new_basis_state_str = f"{new_lq1_val_temp}{new_lq0_val_temp}"
                    new_superposition[new_basis_state_str] += amp
                else:
                    new_superposition[basis_state_str] += amp
            elif op_char_upper == 'CZ': 
                idx1, idx2 = logical_arg 
                q_val_idx1 = (lq0_val if idx1 == 0 else lq1_val)
                q_val_idx2 = (lq0_val if idx2 == 0 else lq1_val)
                phase = -1 if (q_val_idx1 == 1 and q_val_idx2 == 1) else 1
                new_superposition[basis_state_str] += amp * phase
            else: # Unknown op, pass through state, but still add cost
                new_superposition[basis_state_str] += amp

        norm_sq = sum(np.abs(a)**2 for a in new_superposition.values())
        if norm_sq > 1e-9:
            norm = np.sqrt(norm_sq)
            for state_key in new_superposition: new_superposition[state_key] /= norm
        
        return new_superposition, new_orp

    def _calculate_superposition_entropy(self, superposition_dict=None):
        target_superposition = superposition_dict if superposition_dict is not None else self.logical_superposition
        probabilities = np.array([np.abs(amp)**2 for amp in target_superposition.values()])
        probabilities = probabilities[probabilities > 1e-9] 
        if not probabilities.any(): return 0.0
        current_sum_probs = np.sum(probabilities)
        if not np.isclose(current_sum_probs, 1.0) and current_sum_probs > 1e-9: 
            probabilities /= current_sum_probs
        return -np.sum(probabilities * np.log2(probabilities))

    def run_cognitive_cycle_step1_input_prep(self, classical_input_str="00"): # "ZL1ZL0"
        if self.verbose >= 2: print(f"\n  --- Ciclo Cognitivo Paso 1: Procesamiento de Entrada y Preparación de Estado Abstracto ---")
        if self.verbose >= 2: print(f"  Objetivo estado base lógico inicial (lectura ZL1ZL0) = |{classical_input_str}>_L")
        
        self.logical_superposition = {"00": 0.0j, "01": 0.0j, "10": 0.0j, "11": 0.0j}
        self.logical_superposition[classical_input_str] = 1.0 +0j
        self.objective_reduction_potential = 0.0 # Reset ORP for the new cycle's evolution

        if self.verbose >= 3: print(f"    Superposición lógica abstracta preparada: {self.logical_superposition_str()}")
        if self.verbose >= 3: print(f"    ORP después de preparación de entrada: {self.objective_reduction_potential:.3f}")
        
        qc_physical_prep = self.toric_code.get_initialization_circuit()
        
        target_Z_L0 = int(classical_input_str[1])
        target_Z_L1 = int(classical_input_str[0])

        if target_Z_L0 == 1: self.toric_code.apply_logical_op_on_qc(qc_physical_prep, 'X', 0)
        if target_Z_L1 == 1: self.toric_code.apply_logical_op_on_qc(qc_physical_prep, 'X', 1)
        
        self.toric_code.measure_all_stabilizers(qc_physical_prep)
        self.toric_code.measure_all_logical_Zs(qc_physical_prep)
        syndrome, logical_readout = self.toric_code._perform_simulation_and_parse(qc_physical_prep, verbose=self.verbose)
        
        if self.verbose >= 2: print(f"    Comprobación de Base Física (para estado base |{classical_input_str}>_L): Stab: '{syndrome}', Log: '{logical_readout}'")
        
        preparation_successful = (syndrome == "0" * self.toric_code.num_stabilizers and logical_readout == classical_input_str)
        if not preparation_successful and self.verbose >= 1:
            print(f"    FALLO: Base física para |{classical_input_str}>_L inconsistente. Esperado Log: {classical_input_str}, Obtenido: {logical_readout}, Síndrome: {syndrome}")
        return preparation_successful

    def quantum_computation_phase(self, computation_sequence_ops):
        if self.verbose >= 2: print(f"\n  --- Ciclo Cognitivo Paso 2: Computación Cuántica Abstracta ---")

        orp_before_decay_this_phase = self.objective_reduction_potential
        self.objective_reduction_potential = max(0, self.objective_reduction_potential - self.orp_decay_rate)
        if self.verbose >=3 and self.orp_decay_rate > 0 and orp_before_decay_this_phase != self.objective_reduction_potential:
            print(f"    Decaimiento de ORP aplicado: {orp_before_decay_this_phase:.3f} -> {self.objective_reduction_potential:.3f} (tasa de decaimiento: {self.orp_decay_rate})")
        if self.verbose >= 3: print(f"    Iniciando computación con: {self.logical_superposition_str()}, ORP tras decaimiento: {self.objective_reduction_potential:.3f}")
        
        or_triggered_early = False
        if not computation_sequence_ops: 
            if self.verbose >= 3: print(f"    Sin operaciones este ciclo. ORP permanece: {self.objective_reduction_potential:.3f}")
        else:
            if self.verbose >= 3: print(f"    Aplicando secuencia de computación: {computation_sequence_ops}")
            current_superposition_in_phase = copy.deepcopy(self.logical_superposition) # Work on a copy during phase
            current_orp_in_phase = self.objective_reduction_potential

            for i, (op_char, logical_arg) in enumerate(computation_sequence_ops):
                current_superposition_in_phase, current_orp_in_phase = \
                    self._apply_logical_op_to_superposition(op_char, logical_arg, current_superposition_in_phase, current_orp_in_phase)
                
                if self.verbose >= 3:
                    print(f"      Después de op {i+1} ('{op_char}', {logical_arg}): "
                          f"{', '.join([f'{amp:.2f}|{s}>' for s, amp in current_superposition_in_phase.items() if abs(amp) > 1e-9])}, "
                          f"ORP: {current_orp_in_phase:.3f}")

                if current_orp_in_phase >= self.E_OR_THRESHOLD:
                    if self.verbose >= 2: print(f"      >>> UMBRAL OR ALCANZADO ({current_orp_in_phase:.3f} >= {self.E_OR_THRESHOLD}) después de {i+1} ops. <<<")
                    or_triggered_early = True
                    break 
            
            self.logical_superposition = current_superposition_in_phase # Commit changes
            self.objective_reduction_potential = current_orp_in_phase
        
        if self.verbose >= 2: print(f"    Superposición final antes de evento OR: {self.logical_superposition_str()}, ORP: {self.objective_reduction_potential:.3f}")
        return True, or_triggered_early # comp_phase_successful currently always true

    def trigger_OR_event(self):
        if self.verbose >= 2: print(f"\n  --- Ciclo Cognitivo Paso 3: Reducción Objetiva (Colapso Probabilístico) ---")
        if self.verbose >= 3: print(f"    Superposición Pre-OR: {self.logical_superposition_str()}, Potencial: {self.objective_reduction_potential:.3f}")
        
        basis_states = list(self.logical_superposition.keys())
        amplitudes = np.array([self.logical_superposition[s] for s in basis_states], dtype=complex)
        probabilities = np.abs(amplitudes)**2
        sum_probs = np.sum(probabilities)

        if not np.isclose(sum_probs, 1.0) and sum_probs > 1e-9: 
            probabilities /= sum_probs # Normalize if slightly off
        elif sum_probs < 1e-9:
            if self.verbose >= 1: print("    Error: Superposición de norma cero. Por defecto a '00'.")
            self.collapsed_logical_state_str = "00"
            self.logical_superposition = {"00":1.0+0j, "01":0.0j, "10":0.0j, "11":0.0j}
            self.current_orp_before_reset = self.objective_reduction_potential 
            self.objective_reduction_potential = 0.0 
            return self.collapsed_logical_state_str

        self.collapsed_logical_state_str = np.random.choice(basis_states, p=probabilities)
        if self.verbose >= 2: print(f"    Evento OR: Colapsado probabilísticamente a |{self.collapsed_logical_state_str}>_L")
        
        self.current_orp_before_reset = self.objective_reduction_potential # Store value just before reset
        self.objective_reduction_potential = 0.0 # ORP is 'expended' or reset
        
        # Update superposition to the collapsed state
        for state in self.logical_superposition: 
            self.logical_superposition[state] = 1.0 + 0j if state == self.collapsed_logical_state_str else 0.0j
        return self.collapsed_logical_state_str

    def _update_cognitive_load_and_attention(self, orp_at_collapse, num_ops):
        # Cognitive Load: Increases with effort (ORP, num_ops), decays slowly
        load_increase = (orp_at_collapse / self.E_OR_THRESHOLD) * 0.1 + num_ops * 0.02
        self.internal_state_parameters['cognitive_load'] = np.clip(
            self.internal_state_parameters['cognitive_load'] * 0.9 + load_increase, 0.0, 1.0) # Decay old load, add new

        # Attention: Decreases with high load or strong negative mood, recovers towards baseline or with positive mood
        attention_decay_factor = 0.95 - self.internal_state_parameters['cognitive_load'] * 0.1
        attention_recovery = (1.0 - self.internal_state_parameters['attention_level']) * 0.05 # Tendency to recover to 1.0
        self.internal_state_parameters['attention_level'] = np.clip(
            self.internal_state_parameters['attention_level'] * attention_decay_factor + attention_recovery + \
            self.internal_state_parameters['mood'] * 0.01, # Mood slightly influences attention
            0.1, 1.0)

    def _update_mood(self, current_valence):
        # Mood: Slowly shifts towards current valence
        self.internal_state_parameters['mood'] = np.clip(
            self.internal_state_parameters['mood'] * 0.9 + current_valence * 0.2, # Stronger influence from valence, slower self-decay
            -1.0, 1.0)
            
    def _update_long_term_memory(self, op_sequence, valence):
        if not op_sequence: return # Don't store empty sequences
        
        seq_tuple = tuple(op_sequence) # Make it hashable
        if valence >= self.successful_sequence_threshold: # Only store "successful" sequences
            if seq_tuple in self.long_term_memory:
                self.long_term_memory[seq_tuple]['count'] += 1
                self.long_term_memory[seq_tuple]['total_valence'] += valence
                self.long_term_memory[seq_tuple]['avg_valence'] = self.long_term_memory[seq_tuple]['total_valence'] / self.long_term_memory[seq_tuple]['count']
            else:
                # Prune memory if capacity is reached - remove least used or lowest valence
                if len(self.long_term_memory) >= self.long_term_memory_capacity:
                    # Simple pruning: remove sequence with lowest count, then lowest avg_valence
                    prune_candidate = min(self.long_term_memory.items(), key=lambda x: (x[1]['count'], x[1]['avg_valence']))
                    del self.long_term_memory[prune_candidate[0]]
                
                self.long_term_memory[seq_tuple] = {'count': 1, 'total_valence': valence, 'avg_valence': valence}

    def classical_consequence(self, logical_outcome_str, orp_at_collapse, num_superposition_terms, entropy, num_ops, op_sequence):
        if self.verbose >= 2: print(f"\n  --- Ciclo Cognitivo Paso 4: Consecuencia Clásica y Próxima Entrada ---")
        if self.verbose >= 3: print(f"    Resultado: |{logical_outcome_str}>_L, ORP al Colapso: {orp_at_collapse:.3f}, Términos Sup: {num_superposition_terms}, Entropía: {entropy:.3f}, Num Ops: {num_ops}")
        
        if logical_outcome_str is None: 
            if self.verbose >= 1: print("    Sin resultado lógico válido. Estado interno mayormente sin cambios. Proponiendo entrada por defecto '00'.")
            self.last_cycle_valence = 0.0
            return "00" # Default next input

        current_valence = self.outcome_valence.get(logical_outcome_str, 0.0)
        self.last_cycle_valence = current_valence
        if self.verbose >= 2: print(f"    Valencia del Resultado: {current_valence:.2f}")

        # Update primary internal states (curiosity, goal bias)
        self.internal_state_parameters['curiosity'] = np.clip(
            self.internal_state_parameters['curiosity'] - current_valence * 0.1 + entropy * 0.05 + (1.0 - self.internal_state_parameters['attention_level']) * 0.02, 
            0.01, 0.99)
        self.internal_state_parameters['goal_seeking_bias'] = np.clip(
            self.internal_state_parameters['goal_seeking_bias'] + current_valence * 0.2 - entropy * 0.02, 
            0.01, 0.99)

        # Update newer internal states (mood, attention, load)
        self._update_mood(current_valence)
        self._update_cognitive_load_and_attention(orp_at_collapse, num_ops)
        self._update_long_term_memory(op_sequence, current_valence) # Update LTM

        # Update preferred state based on strong valence
        if current_valence > self.metacognition_params.get('high_valence_threshold', 0.7): 
            self.internal_state_parameters['preferred_L0_val'] = int(logical_outcome_str[1])
            self.internal_state_parameters['preferred_L1_val'] = int(logical_outcome_str[0])
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + 0.3)
            if self.verbose >= 2: print(f"      Fuerte valencia positiva! Estableciendo estado preferido a |{logical_outcome_str}>_L y aumentando goal_seeking_bias.")
        elif current_valence < self.metacognition_params.get('low_valence_threshold', -0.2) * 2: # Stronger negative reaction for clearing preference
             self.internal_state_parameters['preferred_L0_val'] = None 
             self.internal_state_parameters['preferred_L1_val'] = None
             self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - 0.3) 
             self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + 0.2)
             if self.verbose >= 2: print(f"      Fuerte valencia negativa! Limpiando estado preferido y aumentando curiosidad.")
        
        if self.verbose >= 3: self.print_internal_state_summary()
        
        next_input_state_str = self._select_next_input_str(logical_outcome_str)
        if self.verbose >= 2: print(f"    Acción Clásica: Sistema procesando estado asociado a '{logical_outcome_str}'.")
        if self.verbose >= 2: print(f"    Proponiendo próximo estado base de entrada: |{next_input_state_str}>_L.")
        return next_input_state_str

    def _select_next_input_str(self, current_outcome_str):
        """Determines the next input string, potentially influenced by internal state."""
        # Simple cyclic for now, with a small mood influence.
        # Can be expanded for more complex "motivation" or "planning".
        base_map = {"00": "01", "01": "10", "10": "11", "11": "00"}
        
        # If mood is very positive, might try to repeat input that led to good state (if this one was good)
        # If mood is very negative, might try something drastically different
        # This is a placeholder for more sophisticated goal-driven input selection.
        if self.internal_state_parameters['mood'] > 0.7 and current_outcome_str == self.collapsed_logical_state_str : # if positive mood AND outcome was as expected by collapse
            # Potentially stick or explore nearby if current input led to good outcome state
            # For now, keep simple
            pass
        elif self.internal_state_parameters['mood'] < -0.7:
            # High negative mood might trigger more random exploration for next input
            # return random.choice(["00", "01", "10", "11"]) # too random maybe
            pass
            
        return base_map.get(current_outcome_str, "00")

    def _replay_from_memory_strategy(self, simulated_orp):
        """Attempts to select an op_sequence from long-term memory."""
        if not self.long_term_memory: return None, simulated_orp

        # Select a sequence from memory, prioritizing high average valence and usage count
        # More sophisticated selection could be UCB1-like for exploration/exploitation of sequences
        # For now, weighted random choice based on avg_valence
        candidate_sequences = []
        weights = []
        for seq, data in self.long_term_memory.items():
            if data['avg_valence'] > 0 : # Only consider positively valenced sequences for replay
                candidate_sequences.append(list(seq)) # op_sequence is a list of tuples
                weights.append(data['avg_valence']**2 * data['count']) # Bias towards high valence and frequent
        
        if not candidate_sequences: return None, simulated_orp
        
        chosen_sequence = random.choices(candidate_sequences, weights=weights, k=1)[0]
        
        # Check if this sequence would immediately hit OR threshold (simplified check)
        projected_orp = simulated_orp
        for op_char, _ in chosen_sequence:
            projected_orp += self.operation_costs.get(op_char.upper(), 0.05)
        
        if projected_orp >= self.E_OR_THRESHOLD and len(chosen_sequence)>1:
             # If too long, maybe try a shorter prefix if available and sensible or just abandon
            if self.verbose >= 3: print(f"      Estrategia de Memoria: Secuencia elegida {chosen_sequence} es demasiado larga para ORP actual. No usada.")
            return None, simulated_orp # Don't use if it blows past threshold (or trim it - more complex)

        if self.verbose >= 3: print(f"      Estrategia de Memoria: Reintentando secuencia {chosen_sequence} (Valencia Promedio: {self.long_term_memory[tuple(chosen_sequence)]['avg_valence']:.2f})")
        
        temp_orp_after_sequence = simulated_orp
        for op_char, _ in chosen_sequence:
             temp_orp_after_sequence += self.operation_costs.get(op_char.upper(), 0.05)
        return chosen_sequence, temp_orp_after_sequence


    def _generate_computation_sequence(self):
        ops_sequence = []
        num_ops_target_base = self.internal_state_parameters['computation_length_preference']
        
        # Adjust num_ops_target based on cognitive load (high load -> shorter sequences)
        load_factor = 1.0 - self.internal_state_parameters['cognitive_load'] * 0.5 # Max 50% reduction
        num_ops_target = max(1, int(np.random.normal(loc=num_ops_target_base * load_factor, scale=1.0)))
        
        simulated_orp = self.objective_reduction_potential # Current ORP after input_prep and decay
        
        if self.verbose >= 3: 
            print(f"    Generando estrategia de computación (objetivo ~{num_ops_target} ops):")
            print(f"      (Comenzando desde ORP: {simulated_orp:.3f}, E_OR_Thr: {self.E_OR_THRESHOLD:.3f})")
            self.print_internal_state_summary(indent="      ")

        # Strategy decision: Memory Replay vs. Goal-Seeking vs. Curious Exploration
        strategy_choice = random.random()
        chosen_strategy_name = "N/A"

        # Memory Replay Strategy (less likely if highly curious)
        if self.long_term_memory and strategy_choice < (1.0 - self.internal_state_parameters['curiosity']) * 0.4 : # Higher chance if not curious
            replay_ops, orp_after_replay = self._replay_from_memory_strategy(simulated_orp)
            if replay_ops:
                ops_sequence = replay_ops
                simulated_orp = orp_after_replay
                chosen_strategy_name = "MemoryReplay"
                if self.verbose >= 3: print(f"      Estrategia: {chosen_strategy_name} seleccionada. Secuencia: {ops_sequence}")


        # Goal Seeking or Curious Exploration if no memory replay or replay failed/not chosen
        if not ops_sequence: 
            for op_count in range(num_ops_target):
                chosen_op, logical_arg = None, None
                op_added_by_goal = False

                # Attention factor: low attention might lead to "mistakes" or simpler ops
                effective_attention = self.internal_state_parameters['attention_level']
                if random.random() > effective_attention: # Chance of "lapse" proportional to (1 - attention)
                    if self.verbose >=3: print(f"      Atención baja detectada! Considerando op más simple o error.")
                    # For a lapse, could pick a random simple op or skip op decision.
                    # Here, let's slightly increase chance of simple op or no-op by falling through to curiosity
                    # with a higher chance of simple ops.
                    pass # Fall through to general choice, with attention factor possibly affecting it too.

                # Goal Seeking Logic (more likely if high bias, less likely if low attention)
                if random.random() < self.internal_state_parameters['goal_seeking_bias'] * effective_attention:
                    chosen_strategy_name = "GoalSeeking"
                    # L0 preference (state[1])
                    if self.internal_state_parameters['preferred_L0_val'] is not None:
                        # Simplified: check if current basis state (e.g. "00") matches pref. Requires superposition knowledge.
                        # For now, use LAST collapsed state as proxy (as in original)
                        current_l0_proxy = int(self.collapsed_logical_state_str[1]) 
                        if current_l0_proxy != self.internal_state_parameters['preferred_L0_val']:
                            chosen_op, logical_arg = 'X', 0; op_added_by_goal = True
                    # L1 preference (state[0])
                    if not op_added_by_goal and self.internal_state_parameters['preferred_L1_val'] is not None:
                         current_l1_proxy = int(self.collapsed_logical_state_str[0])
                         if current_l1_proxy != self.internal_state_parameters['preferred_L1_val']:
                            chosen_op, logical_arg = 'X', 1; op_added_by_goal = True
                    if op_added_by_goal and self.verbose >= 3: print(f"      Estrategia ({chosen_strategy_name} Intento {op_count+1}): Considerando ('{chosen_op}', {logical_arg}).")

                # General Op Choice (Curiosity/Randomness)
                if not chosen_op: 
                    chosen_strategy_name = "CuriosityDriven"
                    op_type_choice = random.random()
                    curiosity_factor = self.internal_state_parameters['curiosity'] * effective_attention # Attention modulates curiosity
                    
                    if op_type_choice < curiosity_factor * 0.5: chosen_op = 'H' # More H if curious
                    elif op_type_choice < 0.2 + curiosity_factor * 0.7 : chosen_op = random.choice(['CNOT','CZ']) # Entangling if very curious
                    else: chosen_op = random.choice(['X','Z']) # Default to X/Z

                    if chosen_op in ['H','X','Z']: logical_arg = random.randint(0,2-1) # L0 or L1
                    elif chosen_op in ['CNOT','CZ']: logical_arg = tuple(random.sample([0,1], 2))
                    if self.verbose >= 3: print(f"      Estrategia ({chosen_strategy_name} Intento {op_count+1}): Considerando ('{chosen_op}', {logical_arg}).")

                if chosen_op:
                    op_cost = self.operation_costs.get(chosen_op.upper(), 0.05)
                    # Attention mechanism: low attention might cause "wrong" argument for chosen op
                    if random.random() > effective_attention and logical_arg is not None and chosen_op not in ['CNOT', 'CZ']: # Simpler ops
                        original_arg = logical_arg
                        logical_arg = 1 - logical_arg # Flip the logical qubit target
                        if self.verbose >=3: print(f"        Error de atención! Arg cambiado de {original_arg} a {logical_arg} para {chosen_op}.")
                        op_cost += self.operation_costs['ERROR'] # Small ORP cost for the "mistake"

                    if simulated_orp + op_cost >= self.E_OR_THRESHOLD and ops_sequence: 
                        if self.verbose >= 3: print(f"      Estrategia anticipa umbral OR ({simulated_orp + op_cost:.3f}) con la próxima op. Secuencia limitada a {len(ops_sequence)} ops.")
                        break 
                    
                    ops_sequence.append((chosen_op, logical_arg))
                    simulated_orp += op_cost 
                elif self.verbose >= 3: print(f"      Estrategia: No se eligió op en este paso (paso {op_count+1}).")
        
        if not ops_sequence and self.verbose >= 3: print("    Estrategia: No se seleccionaron operaciones para esta secuencia.")
        return ops_sequence, chosen_strategy_name


    def _adapt_thresholds_and_decay(self, avg_orp_at_collapse, avg_valence):
        """Adapts E_OR_THRESHOLD and orp_decay_rate based on performance."""
        # Adapt E_OR_THRESHOLD
        if self.metacognition_params.get('enable_threshold_adaptation', False):
            # If avg ORP is consistently too low (many small OR events), consider increasing threshold to allow longer evolution
            # If avg ORP is consistently very high (long evolutions but maybe too costly), consider decreasing
            if avg_orp_at_collapse < self.E_OR_THRESHOLD * 0.7: # Often collapsing much below current threshold
                self.E_OR_THRESHOLD = min(self.orp_threshold_dynamics['max'], self.E_OR_THRESHOLD + self.orp_threshold_dynamics['adapt_rate'])
                if self.verbose >= 2: print(f"      Metacognición: Aumentando E_OR_THRESHOLD a {self.E_OR_THRESHOLD:.3f} (ORP promedio bajo en colapso).")
            elif avg_orp_at_collapse > self.E_OR_THRESHOLD * 1.3: # Often collapsing far above (implies it could sustain more ops easily, maybe reduce for efficiency?)
                 self.E_OR_THRESHOLD = max(self.orp_threshold_dynamics['min'], self.E_OR_THRESHOLD - self.orp_threshold_dynamics['adapt_rate'])
                 if self.verbose >= 2: print(f"      Metacognición: Disminuyendo E_OR_THRESHOLD a {self.E_OR_THRESHOLD:.3f} (ORP promedio alto en colapso).")
        
        # Adapt orp_decay_rate
        if self.metacognition_params.get('enable_decay_adaptation', False):
            # If valence is consistently low, maybe reduce decay to allow more ORP build-up (if threshold is hard to reach)
            # If valence is high, system is doing well, maybe can afford more decay (or keep it low)
            if avg_valence < self.metacognition_params['low_valence_threshold'] * 0.5 : # Consistently very low valence
                self.orp_decay_rate = max(self.orp_decay_dynamics['min'], self.orp_decay_rate - self.orp_decay_dynamics['adapt_rate'])
                if self.verbose >= 2: print(f"      Metacognición: Disminuyendo orp_decay_rate a {self.orp_decay_rate:.4f} (valencia promedio baja).")
            elif avg_valence > self.metacognition_params['high_valence_threshold'] * 0.8: # Consistently high valence
                 self.orp_decay_rate = min(self.orp_decay_dynamics['max'], self.orp_decay_rate + self.orp_decay_dynamics['adapt_rate'] * 0.5) # Slower increase if good
                 if self.verbose >= 2: print(f"      Metacognición: Ajustando ligeramente orp_decay_rate a {self.orp_decay_rate:.4f} (valencia promedio alta).")


    def perform_metacognitive_review(self):
        """Reviews cycle history and adjusts high-level parameters like curiosity, goal bias, and potentially ORP thresholds/decay."""
        if self.verbose >= 1: print(f"\n   gerçekleştiren Metacognitive İnceleme (Döngü {self.current_cycle_num}) ---")
        
        history_span = min(len(self.cycle_history), self.metacognition_params['review_interval'] * 2) # Look at recent history
        recent_history = self.cycle_history[-history_span:]
        if not recent_history: 
            if self.verbose >= 1: print("    Yetersiz geçmiş incelenecek.")
            return

        avg_valence = np.mean([c['valence_this_cycle'] for c in recent_history if c['valence_this_cycle'] is not None]) 
        avg_orp_at_collapse = np.mean([c['orp_at_collapse'] for c in recent_history if c['orp_at_collapse'] is not None])
        num_unique_outcomes = len(set(c['collapsed_to'] for c in recent_history if c['collapsed_to'] != "N/A"))
        total_outcomes = len([c for c in recent_history if c['collapsed_to'] != "N/A"])
        exploration_diversity = num_unique_outcomes / total_outcomes if total_outcomes > 0 else 0

        if self.verbose >= 2:
            print(f"    Son {history_span} döngü analizi: Ort. Valens={avg_valence:.2f}, Ort. ORP={avg_orp_at_collapse:.3f}, Çeşitlilik={exploration_diversity:.2f}")

        # Adjust Curiosity
        cur_adapt_rate = self.metacognition_params['curiosity_adaptation_rate']
        if avg_valence < self.metacognition_params['low_valence_threshold'] or exploration_diversity < self.metacognition_params['exploration_threshold']:
            self.internal_state_parameters['curiosity'] = min(0.99, self.internal_state_parameters['curiosity'] + cur_adapt_rate)
            if self.verbose >= 2: print(f"      Metacognición: Merak artırıldı {self.internal_state_parameters['curiosity']:.2f} (düşük valens/çeşitlilik).")
        elif avg_valence > self.metacognition_params['high_valence_threshold']:
            self.internal_state_parameters['curiosity'] = max(0.01, self.internal_state_parameters['curiosity'] - cur_adapt_rate * 0.5) # Less reduction if good
            if self.verbose >= 2: print(f"      Metacognición: Merak azaltıldı {self.internal_state_parameters['curiosity']:.2f} (yüksek valens).")

        # Adjust Goal Seeking Bias
        goal_adapt_rate = self.metacognition_params['goal_bias_adaptation_rate']
        if avg_valence > self.metacognition_params['high_valence_threshold'] and self.internal_state_parameters['preferred_L0_val'] is not None:
            self.internal_state_parameters['goal_seeking_bias'] = min(0.99, self.internal_state_parameters['goal_seeking_bias'] + goal_adapt_rate)
            if self.verbose >= 2: print(f"      Metacognición: Hedef arama eğilimi artırıldı {self.internal_state_parameters['goal_seeking_bias']:.2f} (yüksek valens & tercih edilen durum var).")
        elif avg_valence < self.metacognition_params['low_valence_threshold']:
             self.internal_state_parameters['goal_seeking_bias'] = max(0.01, self.internal_state_parameters['goal_seeking_bias'] - goal_adapt_rate)
             if self.verbose >= 2: print(f"      Metacognición: Hedef arama eğilimi azaltıldı {self.internal_state_parameters['goal_seeking_bias']:.2f} (düşük valens).")
        
        # Adapt ORP Threshold and Decay Rate (if enabled)
        self._adapt_thresholds_and_decay(avg_orp_at_collapse, avg_valence)
        
        self.metacognition_params['cycles_since_last_review'] = 0 # Reset counter
        if self.verbose >= 1: print(f"  --- Metacognitive İnceleme Tamamlandı ---")


    def run_full_cognitive_cycle(self, classical_input_str, computation_sequence_ops=None, test_error_info=None): 
        self.current_cycle_num += 1
        if self.verbose >= 1: print(f"\n===== Döngü {self.current_cycle_num} Giriş Tabanı: |{classical_input_str}>_L =====")
        
        start_time = time.time()
        
        prep_success = self.run_cognitive_cycle_step1_input_prep(classical_input_str)
        
        # Initial ORP (after prep and decay, which is 0 for prep but explicit here)
        current_orp_after_prep = self.objective_reduction_potential
        
        # Decay ORP that might have carried over IF we implement carry-over for some reason. (Currently reset in step 1)
        # self.objective_reduction_potential = max(0, current_orp_after_prep - self.orp_decay_rate)
        # This decay logic is also present in quantum_computation_phase, applied *before* new ops.

        chosen_op_strategy = "Provided"
        if not prep_success:
            if self.verbose >= 1: print("Döngü iptal edildi: Hazırlık hatası (fiziksel temel).")
            self.cycle_history.append({
                "cycle_num": self.current_cycle_num, "input_state": classical_input_str, 
                "ops_used": "PREP_FAIL", "op_strategy": "N/A",
                "collapsed_to": "N/A", "orp_at_collapse": self.objective_reduction_potential,
                "num_terms": 0, "entropy": 0.0, "valence_this_cycle": 0.0,
                "mood": self.internal_state_parameters['mood'], 
                "attention": self.internal_state_parameters['attention_level'],
                "cog_load": self.internal_state_parameters['cognitive_load'],
                "curiosity_after_cycle": self.internal_state_parameters['curiosity'],
                "goal_bias_after_cycle": self.internal_state_parameters['goal_seeking_bias'],
                "E_OR_thresh": self.E_OR_THRESHOLD, "orp_decay": self.orp_decay_rate
            })
            if len(self.cycle_history) > self.cycle_history_max_len: self.cycle_history.pop(0)
            # Even on prep fail, mood etc should be slightly affected.
            self._update_mood(-0.1) # Small negative valence for failure
            self._update_cognitive_load_and_attention(0, 0) # Minimal load
            return self.classical_consequence(None, self.objective_reduction_potential, 0, 0.0, 0, []) # Returns "00"

        actual_computation_sequence = computation_sequence_ops
        if actual_computation_sequence is None:
            actual_computation_sequence, chosen_op_strategy = self._generate_computation_sequence()
        
        if self.verbose >= 1 and actual_computation_sequence: print(f"  Kullanılan Hesaplama Sırası ({chosen_op_strategy}): {actual_computation_sequence}")
        elif self.verbose >=1: print(f"  Hesaplama sırası oluşturulmadı veya sağlanmadı ({chosen_op_strategy}).")

        _, or_triggered_early = self.quantum_computation_phase(actual_computation_sequence)
        
        orp_before_collapse = self.objective_reduction_potential # This is the ORP *after* computation, right before collapse
        num_superposition_terms = len([amp for amp in self.logical_superposition.values() if abs(amp) > 1e-9])
        entropy_before_collapse = self._calculate_superposition_entropy()

        collapsed_outcome = self.trigger_OR_event() 
        # self.current_orp_before_reset correctly captures the orp_before_collapse value (set in trigger_OR_event)
        
        next_input = self.classical_consequence(
            collapsed_outcome, self.current_orp_before_reset, 
            num_superposition_terms, entropy_before_collapse, 
            len(actual_computation_sequence if actual_computation_sequence else []),
            actual_computation_sequence
        )
        
        self.cycle_history.append({
            "cycle_num": self.current_cycle_num, "input_state": classical_input_str, 
            "ops_used": actual_computation_sequence, "op_strategy": chosen_op_strategy,
            "collapsed_to": collapsed_outcome, "orp_at_collapse": self.current_orp_before_reset,
            "num_terms": num_superposition_terms, "entropy": entropy_before_collapse,
            "valence_this_cycle": self.last_cycle_valence, 
            "mood": self.internal_state_parameters['mood'],
            "attention": self.internal_state_parameters['attention_level'],
            "cog_load": self.internal_state_parameters['cognitive_load'],
            "curiosity_after_cycle": self.internal_state_parameters['curiosity'],
            "goal_bias_after_cycle": self.internal_state_parameters['goal_seeking_bias'],
            "E_OR_thresh": self.E_OR_THRESHOLD, "orp_decay": self.orp_decay_rate
        })
        if len(self.cycle_history) > self.cycle_history_max_len: self.cycle_history.pop(0)

        # Metacognitive Review
        self.metacognition_params['cycles_since_last_review'] += 1
        if self.metacognition_params['cycles_since_last_review'] >= self.metacognition_params['review_interval']:
            self.perform_metacognitive_review()

        if test_error_info: # This is more for QEC demo, less part of cognitive core
            # Error demo would ideally be more integrated, here it's an add-on display
            if self.verbose >=2: print("\n  --- Hata Düzeltme Gösterimi (Soyutlanmış Adım) ---")
            # Original `_demonstrate_error_correction_on_collapsed_state` was a pass.
            # For now, it's illustrative that this layer exists.
            # Pass, as the original didn't detail its implementation and it's complex.
            pass
        
        cycle_duration = time.time() - start_time
        if self.verbose >= 1: print(f"===== Döngü Sonu (Süre: {cycle_duration:.3f}s, Sonraki Giriş: |{next_input}>_L) =====")
        return next_input

    def logical_superposition_str(self):
        active_terms = [f"{amp.real:.2f}{amp.imag:+.2f}j |{state}>" if amp.imag != 0 else f"{amp.real:.2f} |{state}>"
                        for state, amp in self.logical_superposition.items() if abs(amp) > 1e-9]
        if not active_terms: return "Sıfır Süperpozisyon"
        return ", ".join(active_terms)
    
    def print_internal_state_summary(self, indent="  "):
        print(f"{indent}İç Durum Özeti:")
        for key, val in self.internal_state_parameters.items():
            if isinstance(val, float): print(f"{indent}  {key}: {val:.2f}")
            else: print(f"{indent}  {key}: {val}")
        print(f"{indent}  E_OR_THRESHOLD: {self.E_OR_THRESHOLD:.3f}, orp_decay_rate: {self.orp_decay_rate:.4f}")


    def run_chained_cognitive_cycles(self, initial_input_str, num_cycles, 
                                      computation_sequence_ops_template=None, 
                                      test_error_info_template=None):
        if self.verbose >= 0: # Always print this header if any verbosity
            print(f"\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f"%%%%% ZİNCİRLEME BİLİŞSEL DÖNGÜLER BAŞLATILIYOR (Toplam: {num_cycles}) %%%%%")
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        
        current_input_str = initial_input_str
        for i in range(num_cycles):
            if self.verbose >= 1:
                pref_L0 = self.internal_state_parameters['preferred_L0_val']
                pref_L1 = self.internal_state_parameters['preferred_L1_val']
                pref_str = f"({pref_L1},{pref_L0})" if pref_L1 is not None else "Yok"

                print(f"\n>>>> Zincirleme Döngü {i+1}/{num_cycles} (Giriş |{current_input_str}>_L; "
                      f"C:{self.internal_state_parameters['curiosity']:.2f} "
                      f"GB:{self.internal_state_parameters['goal_seeking_bias']:.2f} "
                      f"M:{self.internal_state_parameters['mood']:.2f} "
                      f"A:{self.internal_state_parameters['attention_level']:.2f} "
                      f"CL:{self.internal_state_parameters['cognitive_load']:.2f} "
                      f"TercihL1,L0:{pref_str}) <<<<")
            
            current_comp_ops = None
            if computation_sequence_ops_template:
                 current_comp_ops = computation_sequence_ops_template[i % len(computation_sequence_ops_template)] if isinstance(computation_sequence_ops_template, list) else computation_sequence_ops_template

            current_test_error = None
            if test_error_info_template:
                current_test_error = test_error_info_template[i % len(test_error_info_template)] if isinstance(test_error_info_template, list) else test_error_info_template
            
            next_input_str = self.run_full_cognitive_cycle(current_input_str, current_comp_ops, test_error_info=current_test_error)
            if next_input_str is None: 
                if self.verbose >= 0: print(f"HATA: Sonraki giriş dizesi None. Zincirleme döngüler {i+1}. döngüde iptal ediliyor"); 
                break
            current_input_str = next_input_str
        
        if self.verbose >= 0:
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f"%%%%% ZİNCİRLEME BİLİŞSEL DÖNGÜLER TAMAMLANDI ({self.current_cycle_num} döngü çalıştırıldı) %%%%%")
            self.print_internal_state_summary(indent="  Final ")
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42) # For reproducible random choices
    random.seed(42)

    # --- DEMO 0: Toric Code functionality test for different sizes ---
    print("\n\n--- DEMO 0: Toric Code İşlevsellik Testi ---")
    try:
        tc_3x3 = ToricCode(Lx=3, Ly=3)
        print(str(tc_3x3))
        # Create a dummy circuit to test error injection on larger code
        qc_test_3x3 = tc_3x3.get_initialization_circuit()
        tc_3x3.apply_physical_error(qc_test_3x3, [('X', 0), ('Z', tc_3x3.num_data_qubits -1)])
        print("3x3 Toric Code oluşturuldu ve hata eklendi (soyut olarak).")
    except Exception as e:
        print(f"Toric Code 3x3 testi sırasında hata: {e}")
    
    tc_2x2_default = ToricCode(Lx=2, Ly=2) # For other demos
    print(str(tc_2x2_default))
    print("Standart 2x2 Toric Code örneği tamamlandı.\n")


    # DEMO 1 and 2: Using default OrchOR parameters, run quietly for brevity in main output
    print("\n\n--- DEMO 1: CZ Kapısı ve İç Durum (Sessiz Çalışma) ---")
    # Use verbose=0 for minimal output for these demos. Increase for details.
    emulator_demo1 = SimplifiedOrchOREmulator(Lx=2, Ly=2, verbose=0) 
    emulator_demo1.E_OR_THRESHOLD = 0.8 
    emulator_demo1.internal_state_parameters['curiosity'] = 0.2 
    emulator_demo1.internal_state_parameters['goal_seeking_bias'] = 0.0 
    bell_ops = [('H',0), ('CNOT',(0,1))] 
    _ = emulator_demo1.run_full_cognitive_cycle(classical_input_str="00", computation_sequence_ops=bell_ops)
    
    emulator_demo1.E_OR_THRESHOLD = 1.2 
    emulator_demo1.internal_state_parameters['curiosity'] = 0.1 
    cz_test_ops = [('H',0), ('H',1), ('CZ',(0,1))]
    _ = emulator_demo1.run_full_cognitive_cycle(classical_input_str="00", computation_sequence_ops=cz_test_ops)
    print("Demo 1 tamamlandı.")

    print("\n\n--- DEMO 2: Dahili olarak üretilen operasyonlarla zincirleme döngüler (Sessiz Çalışma) ---")
    emulator_demo2 = SimplifiedOrchOREmulator(Lx=2, Ly=2, cycle_history_max_len=10, verbose=0)
    emulator_demo2.E_OR_THRESHOLD = 0.6 
    emulator_demo2.outcome_valence = {"00": 0.1, "01": 0.8, "10": -0.9, "11": 0.5} 
    emulator_demo2.internal_state_parameters['computation_length_preference'] = 3 
    emulator_demo2.internal_state_parameters['curiosity'] = 0.6 
    emulator_demo2.internal_state_parameters['goal_seeking_bias'] = 0.4
    emulator_demo2.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=5)
    print("Demo 2 tamamlandı. Son durum ve geçmiş (kısaltılmış):")
    emulator_demo2.print_internal_state_summary(indent="  ")
    print("  Örnek Döngü Geçmişi (en son {}):".format(min(3, len(emulator_demo2.cycle_history))))
    for record in emulator_demo2.cycle_history[-3:]: # Print last 3
        print(f"    Döngü {record['cycle_num']}: Giriş={record['input_state']}, Ops={record['ops_used'][:20]}..., Strateji={record['op_strategy']}, "
              f"Çöktü=|{record['collapsed_to']}>, ORP={record['orp_at_collapse']:.2f}, V={record['valence_this_cycle']:.2f}")


    print("\n\n--- DEMO 3: E_OR_THRESHOLD ve ORP_DECAY_RATE için Sistematik Parametre Taraması ---")
    E_OR_THRESHOLD_values = [0.7, 1.5] # Reduced for brevity
    orp_decay_rate_values = [0.0, 0.05]  # Reduced for brevity
    num_cycles_for_sweep = 10  # Reduced for brevity
    initial_input_for_sweep = "00"
    all_sweep_results = []

    for e_or_thresh in E_OR_THRESHOLD_values:
        for decay_rate in orp_decay_rate_values:
            print(f"\n===== TARAMA ÇALIŞTIRILIYOR: E_OR_THRESHOLD = {e_or_thresh}, ORP_DECAY_RATE = {decay_rate} =====")
            # verbose=0 for sweep runs to keep output clean
            emulator_sweep = SimplifiedOrchOREmulator(Lx=2, Ly=2, cycle_history_max_len=num_cycles_for_sweep, 
                                                      initial_E_OR_THRESHOLD=e_or_thresh, 
                                                      initial_orp_decay_rate=decay_rate, verbose=0)
            
            # Consistent initial internal parameters for fair comparison within sweep
            custom_initial_states = copy.deepcopy(DEFAULT_INTERNAL_PARAMS)
            custom_initial_states.update({'computation_length_preference': 3, 'curiosity': 0.5, 'goal_seeking_bias': 0.3})
            emulator_sweep.internal_state_parameters = custom_initial_states
            emulator_sweep.outcome_valence = {"00": 0.1, "01": 0.6, "10": -0.6, "11": 0.3}

            emulator_sweep.run_chained_cognitive_cycles(initial_input_str=initial_input_for_sweep, num_cycles=num_cycles_for_sweep)

            run_summary = {
                "E_OR_THRESHOLD": e_or_thresh, "orp_decay_rate": decay_rate,
                "final_curiosity": emulator_sweep.internal_state_parameters['curiosity'],
                "final_goal_bias": emulator_sweep.internal_state_parameters['goal_seeking_bias'],
                "final_mood": emulator_sweep.internal_state_parameters['mood'],
                "final_attention": emulator_sweep.internal_state_parameters['attention_level'],
                "final_cog_load": emulator_sweep.internal_state_parameters['cognitive_load'],
                "avg_orp_at_collapse": np.mean([c['orp_at_collapse'] for c in emulator_sweep.cycle_history if c['orp_at_collapse'] is not None and c['collapsed_to'] != "N/A"] or [0]),
                "num_prep_fails": sum(1 for c in emulator_sweep.cycle_history if c['collapsed_to'] == "N/A"),
                "history_sample": emulator_sweep.cycle_history[-3:] # last 3 cycles as sample
            }
            all_sweep_results.append(run_summary)
    
    print("\n\n--- PARAMETRE TARAMASI ÖZETİ ---")
    for i, res in enumerate(all_sweep_results):
        print(f"\nÇalışma {i+1}: Parametreler: E_OR_THRESHOLD={res['E_OR_THRESHOLD']}, ORP_Decay_Rate={res['orp_decay_rate']}")
        print(f"  Son Durum: Merak={res['final_curiosity']:.2f}, HedefEğilim={res['final_goal_bias']:.2f}, Ruh Hali={res['final_mood']:.2f}, Dikkat={res['final_attention']:.2f}, BilişselYük={res['final_cog_load']:.2f}")
        print(f"  Ort. Çöküş ORP: {res['avg_orp_at_collapse']:.3f}, Hazırlık Hataları: {res['num_prep_fails']}")


    print("\n\n--- DEMO 4: Metacognition and Dynamic Parameters Showcase ---")
    # Configure emulator for metacognition and dynamic ORP parameters
    meta_config_demo4 = copy.deepcopy(DEFAULT_METACOGNITION_PARAMS)
    meta_config_demo4['review_interval'] = 5 # Review more frequently for demo
    meta_config_demo4['enable_threshold_adaptation'] = True
    meta_config_demo4['enable_decay_adaptation'] = True

    # Verbose level 1 to see metacognitive review summaries
    emulator_demo4 = SimplifiedOrchOREmulator(
        Lx=2, Ly=2, cycle_history_max_len=30,
        initial_E_OR_THRESHOLD=1.0, initial_orp_decay_rate=0.02,
        metacognition_config=meta_config_demo4,
        verbose=1 # Set to 1 or 2 to see details from this demo
    )
    # Setup specific valences that might drive adaptation
    emulator_demo4.outcome_valence = {"00": -0.5, "01": 0.8, "10": -0.8, "11": 0.2} # Some strong negatives
    print(f"Başlangıç E_OR_THRESHOLD: {emulator_demo4.E_OR_THRESHOLD:.3f}, Başlangıç orp_decay_rate: {emulator_demo4.orp_decay_rate:.4f}")
    
    emulator_demo4.run_chained_cognitive_cycles(initial_input_str="00", num_cycles=25)

    print("\nDemo 4 Sonrası Nihai Durum:")
    emulator_demo4.print_internal_state_summary(indent="  ")
    print(f"  Son E_OR_THRESHOLD: {emulator_demo4.E_OR_THRESHOLD:.3f}")
    print(f"  Son orp_decay_rate: {emulator_demo4.orp_decay_rate:.4f}")
    print("\nDemo 4 Uzun Vadeli Hafıza Örneği (varsa):")
    ltm_sample = list(emulator_demo4.long_term_memory.items())[:3]
    if ltm_sample:
        for seq_tuple, data in ltm_sample:
            print(f"  Sekans: {seq_tuple}, Sayı: {data['count']}, Ort. Valens: {data['avg_valence']:.2f}")
    else:
        print("  Uzun vadeli hafıza boş veya yeterince başarılı sekans yok.")

    print("\n\nTüm Demolar Tamamlandı.")
