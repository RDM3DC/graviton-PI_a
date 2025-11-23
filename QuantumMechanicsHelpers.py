import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm

# --- Quantum Mechanics Helpers ---

def partial_trace_to_pair(psi, n, keep_indices):
    """
    Calculating reduced density matrix for a pair of qubits.
    This is the computationally heavy part (2^N scaling).
    """
    # Reshape state vector to tensor
    psi_tensor = psi.reshape([2] * n)
    
    # Move kept indices to front
    all_indices = list(range(n))
    trace_indices = [i for i in all_indices if i not in keep_indices]
    permuted_indices = keep_indices + trace_indices
    
    psi_permuted = np.transpose(psi_tensor, permuted_indices)
    
    # Reshape to (4, 2^(N-2))
    dim_keep = 2 ** len(keep_indices)
    dim_trace = 2 ** len(trace_indices)
    psi_mat = psi_permuted.reshape(dim_keep, dim_trace)
    
    # Rho = Psi * Psi_dagger
    rho = np.dot(psi_mat, psi_mat.conj().T)
    return rho

def get_von_neumann_entropy(rho):
    """S = -Tr(rho log rho)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter out zeros/negative due to precision to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def rxx_gate(phi):
    """
    Exp(-i * phi/2 * XX). 
    Rotation by angle phi around XX axis.
    """
    c = np.cos(phi / 2)
    s = -1j * np.sin(phi / 2)
    # Matrix form of XX interaction
    XX = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]])
    return c * np.eye(4) + s * XX

def apply_gate(psi, gate, n, q1, q2):
    """Apply 4x4 gate to qubits q1, q2 of state psi"""
    psi_tensor = psi.reshape([2] * n)
    # Swap q1 to pos 0, q2 to pos 1
    perm = list(range(n))
    perm[q1], perm[0] = perm[0], perm[q1]
    # Note: if q2 was 0, it's now at q1. Handle carefully.
    curr_q2 = 0 if q2 == perm[q1] else q2 # where q2 is currently
    # This logic is getting messy, let's use matrix multiplication on full Hilbert space
    # Ideally, for speed, we stick to small N and build full 2^N matrix
    return psi # Placeholder logic, see Main Loop below for full matrix method

# --- Main Simulation ---

# System Params
N = 5  # Number of qubits (Keep small! 2^5 = 32 dim)
steps = 100
dt = 0.1
eta = 0.5  # Learning rate for geometry
S_target = 0.4 # The holographic bound

# State Initialization |00000>
dim = 2**N
psi = np.zeros(dim, dtype=complex)
psi[0] = 1.0 

# Geometry Initialization (Linear Chain)
# pi_a represents coupling strength. Start with uniform weak coupling.
edges = [(i, i+1) for i in range(N-1)]
pi_a = {e: 0.2 for e in edges} 

# History for plotting
history_S = {e: [] for e in edges}
history_pi = {e: [] for e in edges}

print(f"Simulating {N} qubits for {steps} steps...")

for t in range(steps):
    
    # 1. EVOLVE QUANTUM STATE (Apply Graviton Gates)
    # We build the full Hamiltonian for this step to handle N-body evolution
    H_total = np.zeros((dim, dim), dtype=complex)
    
    for e in edges:
        u, v = e
        # Construct X_u X_v operator on full Hilbert space
        op_list = [np.eye(2)] * N
        op_list[u] = np.array([[0, 1], [1, 0]]) # X
        op_list[v] = np.array([[0, 1], [1, 0]]) # X
        
        # Tensor product to get full operator
        term = op_list[0]
        for k in range(1, N):
            term = np.kron(term, op_list[k])
            
        # Add to Hamiltonian weighted by curvature pi_a
        H_total += pi_a[e] * term

    # Unitary Step U = exp(-i * H * dt)
    # Use diagonalization for exponentiation
    evals, evecs = np.linalg.eigh(H_total)
    U_total = evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T
    psi = U_total @ psi

    # 2. MEASURE ENTANGLEMENT & UPDATE GEOMETRY
    for e in edges:
        u, v = e
        
        # Calculate Entropy across the bond
        # For a chain, trace out everything except u, v
        rho_uv = partial_trace_to_pair(psi, N, [u, v])
        S_val = get_von_neumann_entropy(rho_uv)
        
        # 3. ARP BACKREACTION
        # Error signal
        error = S_val - S_target
        
        # Update Pi (Geometry)
        # Rule: If Entropy is high, reduce coupling (expand space).
        #       If Entropy is low, increase coupling (contract space).
        pi_a[e] -= eta * error * dt
        
        # Clamp pi to avoid negative coupling (optional, purely for stability)
        pi_a[e] = max(0.0, pi_a[e])

        # Store history
        history_S[e].append(S_val)
        history_pi[e].append(pi_a[e])

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Entanglement Entropy
for e in edges:
    ax1.plot(history_S[e], label=f"Bond {e}")
ax1.axhline(S_target, color='k', linestyle='--', label="Target S")
ax1.set_ylabel("Entanglement Entropy (S)")
ax1.set_title("Matter: Entanglement Evolution")
ax1.legend(loc='upper right')

# Plot Geometry (Pi_a)
for e in edges:
    ax2.plot(history_pi[e], label=f"Bond {e}")
ax2.set_ylabel("Coupling Strength (π_a)")
ax2.set_xlabel("Time Step")
ax2.set_title("Geometry: Adaptive Coupling Evolution")

plt.tight_layout()
plt.show()
