# graviton-PI_a

A research codebase for ARP-controlled quantum search experiments and supporting classical hardware interfaces.

## Overview

This repository implements an Adaptive Reward-Penalty (ARP) controller for Grover-like quantum search experiments and documents an integration with a proposed Adaptive Impedance Network (AIN) — a classical hardware layer that provides active decoherence suppression informed by the ARP controller.

New: A detailed predictive hardware interface specification for the AIN has been added:
- docs/AIN_Predictive_Hardware_Interface.md — Predictive Hardware Interface Specification — Adaptive Impedance Network (AIN)

See the documentation for:
- ARP dynamics and steady-state angle-gain scaling with problem size N.
- AIN control law linking steady-state angle gain to hardware conductance and noise-cancellation capability.
- Predictive scaling relations for required noise-cancellation vs. N and expected stable fidelity.

## Key concepts (short)

- ARP controller tunes an effective rotation gain G_angle to stabilize the quantum search under decoherence.
- Empirical large-N scaling: G_angle^steady ≈ C / N (C ≈ 0.4).
- The AIN maps G_angle^steady → G_AIN and provides a noise-canceling effect Γ_noise^cancel ∝ G_AIN ∝ 1/N.
- Practical consequence: For fixed analog design ratios, the per-iteration active noise budget required from hardware decreases as 1/N.

## Interface parameters (quick reference)

- N — search space size (N = 2^n for n qubits)
- C — geometry / tuning constant (calibrated)
- G_angle^steady — residual rotation gain from ARP controller
- κ_link — mapping factor from angle gain to conductance
- G_AIN — tunable conductance implemented in hardware
- k_AIN — hardware efficiency factor
- Γ_noise^cancel — maximum decoherence rate the AIN can cancel (target: ∝ 1/N)

## Where to start

1. Read the AIN predictive hardware spec: docs/AIN_Predictive_Hardware_Interface.md
2. Review ARP controller code and simulation examples (see src/ or examples/ as applicable).
3. For hardware teams: use the interface parameters in the spec to size tunable impedances and set calibration procedures.

## Diagram

A unified architecture diagram and a TikZ skeleton are included in the AIN spec document for authors who want to embed the figure in papers.

## Contributing

If you want the README to include additional developer notes (build, run examples, simulation commands, hardware calibration steps), tell me what to include and I will add them.