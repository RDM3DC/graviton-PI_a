# Graviton-PI_a: Quantum Entanglement-Geometry Backreaction Simulator

A Python-based quantum simulation framework exploring the feedback loop between quantum entanglement and emergent spacetime geometry. This project implements "graviton gates" that couple quantum matter to adaptive geometric parameters, demonstrating how information flow can shape spacetime connectivity.

## Overview

This repository contains implementations of quantum circuit simulations where:
- **Quantum matter** evolves through unitary gates on a chain of qubits
- **Geometric parameters** (πₐ) modulate gate coupling strengths
- **Entanglement entropy** drives geometric backreaction, creating adaptive spacetime

The simulation explores a toy model of quantum gravity where spacetime geometry emerges from and responds to quantum entanglement patterns.

## Scientific Background

Traditional quantum gravity approaches treat spacetime and matter separately. This project explores a different paradigm:

1. **Quantum State Evolution**: Qubits arranged in a chain evolve under local Hamiltonian dynamics
2. **Entanglement Measurement**: Von Neumann entropy quantifies quantum correlations across spatial cuts
3. **Geometric Backreaction**: Geometry parameters (πₐ) adapt based on entanglement error signals
4. **Emergent Spacetime**: The coupled quantum-geometric system exhibits phase transitions and memory effects

This implements ideas from:
- Holographic entanglement entropy
- AdS/CFT correspondence
- Adaptive Resonance Plasticity (ARP)
- Quantum information geometry

For detailed mathematical background, see [THEORY_NOTES.md](THEORY_NOTES.md).

## Features

### Two Evolution Modes

1. **Gate Mode**: Sequential application of two-qubit RXX gates with adaptive parameters
2. **Hamiltonian Mode**: Global Hamiltonian evolution with real-time geometric updates

### Simulation Capabilities

- Von Neumann entropy calculation for entanglement quantification
- Adaptive coupling strength updates based on entanglement feedback
- Phase diagram exploration across parameter space (η, S_target)
- Time-series visualization of entropy and geometry evolution
- Support for chains of 4-10 qubits (limited by 2^N scaling)

### Analysis Tools

- **Phase sweeps**: Map steady-state behavior across learning rates (η) and target entropies
- **Trajectory plotting**: Visualize co-evolution of matter and geometry
- **Heatmap generation**: Identify phase boundaries and instabilities

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

### Setup

```bash
# Clone the repository
git clone https://github.com/RDM3DC/graviton-PI_a.git
cd graviton-PI_a

# Install dependencies
pip install numpy matplotlib scipy
```

## Usage

### Quick Start

Run a basic simulation with default parameters:

```bash
python entanglement_backreaction.py
```

This runs 200 steps with a 6-qubit chain in gate mode and prints final entropies and coupling strengths.

### Gate Mode Example

```bash
python entanglement_backreaction.py \
    --mode gate \
    --n-qubits 6 \
    --steps 200 \
    --eta 0.3 \
    --entropy-target 0.4 \
    --lambda 0.8
```

**Parameters:**
- `--n-qubits`: Chain length (4-8 recommended)
- `--steps`: Number of evolution steps
- `--eta`: Learning rate for geometric adaptation
- `--entropy-target`: Target von Neumann entropy
- `--lambda`: Gate-angle coupling strength

### Hamiltonian Mode Example

```bash
python entanglement_backreaction.py \
    --mode hamiltonian \
    --n-qubits 5 \
    --steps 150 \
    --eta 0.5 \
    --entropy-target 0.4 \
    --initial-pi 0.2 \
    --plot-history
```

The `--plot-history` flag generates plots showing entropy and πₐ evolution over time.

### Phase Diagram Sweep

Explore parameter space to identify phase boundaries:

```bash
python entanglement_backreaction.py \
    --phase-sweep \
    --sweep-etas 0.1 0.5 1.0 2.0 \
    --sweep-targets 0.1 0.3 0.5 0.7 \
    --sweep-steps 150 \
    --plot-phase-diagram \
    --plot-sweep-trajectories
```

This generates:
- Heatmaps of steady-state coupling strength and variability
- Sample trajectories showing different dynamical regimes

### Simple Helper Script

The `QuantumMechanicsHelpers.py` provides a standalone example that runs automatically:

```bash
python QuantumMechanicsHelpers.py
```

This generates `Figure_143.png` (or displays interactively) showing entanglement and geometry evolution.

## Code Structure

### Main Components

**`entanglement_backreaction.py`**
- `SimulationParams`: Configuration for gate-mode simulations
- `HamiltonianParams`: Configuration for Hamiltonian-mode simulations
- `SweepParams`: Configuration for parameter sweeps
- `QuantumGravityUniverse`: Class wrapping Hamiltonian evolution
- `run_simulation()`: Execute gate-mode simulation
- `run_hamiltonian_simulation()`: Execute Hamiltonian-mode simulation
- `run_parameter_sweep()`: Perform phase diagram exploration

**`QuantumMechanicsHelpers.py`**
- Simplified standalone implementation
- Demonstrates basic entanglement-geometry feedback
- Generates visualization automatically

### Key Functions

```python
rxx(phi)                          # RXX gate matrix
apply_two_qubit(U, psi, n, q1, q2)  # Apply 2-qubit gate to state
reduced_density_matrix(psi, n, keep)  # Partial trace operation
von_neumann_entropy(rho)          # Calculate S = -Tr(ρ log ρ)
```

## Example Output

After running a simulation, you'll see output like:

```
Final edge entropies and curvatures:
(0, 1): S=0.389, pi_a=3.142
(1, 2): S=0.412, pi_a=3.089
(2, 3): S=0.401, pi_a=3.115
(3, 4): S=0.395, pi_a=3.128
(4, 5): S=0.388, pi_a=3.143
```

This shows:
- Each edge's entanglement entropy (S) converging toward the target
- Adaptive coupling strengths (πₐ) settling into a quasi-uniform pattern

### Sample Visualization

The included `Figure_143.png` demonstrates typical output showing:
- **Top panel**: Entanglement entropy evolution across different bonds
- **Bottom panel**: Adaptive coupling strength (πₐ) evolution
- Target entropy marked as horizontal dashed line

## Research Questions Addressed

1. **Does πₐ converge?** Yes, under appropriate learning rates (η)
2. **Phase transitions?** Yes, high η leads to oscillatory instabilities
3. **Memory storage?** Geometric configuration stores history of information flow
4. **Entanglement-geometry correlation?** Strong correlation emerges in steady state

## Performance Notes

- **Memory**: O(2^N) where N is number of qubits
- **Time complexity**: O(2^N × steps) for gate mode, O(2^(2N)) for Hamiltonian diagonalization
- **Practical limit**: N ≤ 10 qubits on typical hardware
- **Recommended**: N = 4-6 for interactive exploration

## Mathematical Details

### Gate Form

```
U_v(θ_v, π_a(v)) = exp(-i θ_v H_v(π_a(v)))
```

where H_v is a local Hamiltonian (RXX interaction).

### Backreaction Rule

```
π_a(v) ← π_a(v) + η·dt·(S_v - S_target)
```

Coupling increases when entanglement is below target, decreases when above.

### Von Neumann Entropy

```
S_v = -Tr(ρ_v log ρ_v)
```

where ρ_v is the reduced density matrix for the local region.

## Future Directions

Potential extensions (see THEORY_NOTES.md for details):

1. **Tier 2**: Promote πₐ to quantum ancilla (controlled gates)
2. **Tier 3**: Dynamic connectivity via ARP-style rewiring
3. **Tier 4**: Multi-dimensional lattices beyond 1D chains
4. **Tier 5**: Incorporate curvature tensors and field theory

## Contributing

Contributions welcome! Areas of interest:

- Optimization for larger qubit counts
- Alternative entanglement measures
- Connection to tensor network methods
- Comparison with exact quantum gravity results
- GPU acceleration

## Citation

If you use this code in research, please cite:

```
@software{graviton_pi_a,
  title = {Graviton-PI_a: Quantum Entanglement-Geometry Backreaction Simulator},
  author = {RDM3DC},
  year = {2024},
  url = {https://github.com/RDM3DC/graviton-PI_a}
}
```

## License

This project is available under the MIT License. See LICENSE file for details (if applicable).

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Discussions welcome in the Issues tab

## Acknowledgments

This work draws inspiration from:
- Holographic principle and AdS/CFT
- Quantum information approaches to gravity
- Adaptive network theory
- Tensor network quantum simulations

---

*"Spacetime is not fundamental—it emerges from quantum entanglement."*
