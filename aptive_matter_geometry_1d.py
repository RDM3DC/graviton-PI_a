import numpy as np
import matplotlib.pyplot as plt

class AdaptiveMatterGeometry1D:
    """
    1D tight-binding particle on a chain with adaptive geometry.

    Matter:
        - Single quantum particle on N sites.
        - State psi[t] \in C^N, normalized, evolves via H(t).

    Geometry:
        - Edge field pi_edges[i] on bond (i, i+1).
        - Enters as an effective hopping J_eff = J * pi_edges[i].

    Backreaction (matter -> geometry):
        - Compute local bond energy E_bond(i) = <psi|h_i|psi>
        - Update pi_edges by ARP-style rule:
              dπ_i/dt = -η_E (E_bond(i) - E_target) - μ_π (π_i - π_0)
    """

    def __init__(
        self,
        n_sites=6,
        dt=0.05,
        J=1.0,
        eta_E=0.5,
        mu_pi=0.1,
        pi0=1.0,
        E_target=0.0,
        mass_profile=None,
        init_site=0,
    ):
        self.n = n_sites
        self.dt = dt
        self.J = J
        self.eta_E = eta_E
        self.mu_pi = mu_pi
        self.pi0 = pi0
        self.E_target = E_target

        # Quantum state: single particle on chain, start localized at init_site
        self.psi = np.zeros(n_sites, dtype=complex)
        self.psi[init_site] = 1.0

        # Geometry: one π per bond (i, i+1)
        self.pi_edges = np.ones(n_sites - 1, dtype=float) * pi0

        # Matter: on-site "mass" / potential
        if mass_profile is None:
            self.m = np.zeros(n_sites, dtype=float)
        else:
            mass_profile = np.array(mass_profile, dtype=float)
            if mass_profile.shape != (n_sites,):
                raise ValueError("mass_profile must have length n_sites")
            self.m = mass_profile

        # History
        self.history_pi = []       # list of arrays, shape (n_sites-1,)
        self.history_density = []  # list of arrays, shape (n_sites,)

    # ----- Hamiltonian pieces -----

    def hamiltonian(self):
        """
        Build full N x N Hamiltonian at current geometry:
          H = sum_i [ -J*pi_i (|i><i+1| + |i+1><i|) ] + sum_i m_i |i><i|
        """
        H = np.zeros((self.n, self.n), dtype=complex)

        # Hopping terms
        for i in range(self.n - 1):
            J_eff = self.J * self.pi_edges[i]
            H[i, i + 1] = -J_eff
            H[i + 1, i] = -J_eff

        # On-site mass / potential
        for i in range(self.n):
            H[i, i] += self.m[i]

        return H

    # ----- Single time step -----

    def step(self):
        # 1) Unitary matter evolution with current H
        H = self.hamiltonian()
        evals, evecs = np.linalg.eigh(H)  # H is Hermitian
        U = evecs @ np.diag(np.exp(-1j * evals * self.dt)) @ evecs.conj().T
        self.psi = U @ self.psi

        # 2) Compute local bond energies & update geometry
        new_pi = self.pi_edges.copy()

        for i in range(self.n - 1):
            # local 2x2 bond Hamiltonian for sites (i, i+1)
            J_eff = self.J * self.pi_edges[i]
            h_local = np.array([[0, -J_eff],
                                [-J_eff, 0]], dtype=complex)
            psi_local = np.array([self.psi[i], self.psi[i + 1]])
            E_bond = np.real(np.vdot(psi_local, h_local @ psi_local))

            # ARP-like update: dπ/dt = -η_E (E_bond - E_target) - μ_π (π - π0)
            d_pi = -self.eta_E * (E_bond - self.E_target) - self.mu_pi * (self.pi_edges[i] - self.pi0)
            new_pi[i] += self.dt * d_pi

        self.pi_edges = new_pi

        # 3) Record history
        self.history_pi.append(self.pi_edges.copy())
        self.history_density.append(np.abs(self.psi) ** 2)

    def run(self, steps=200):
        for _ in range(steps):
            self.step()

    # ----- Convenience: convert history to arrays -----

    def get_history_arrays(self):
        """
        Returns:
            density: array shape (steps, n_sites)
            pi_hist: array shape (steps, n_sites-1)
        """
        density = np.array(self.history_density)          # (T, N)
        pi_hist = np.array(self.history_pi)               # (T, N-1)
        return density, pi_hist


# ----- Demo / quick experiment -----

def demo():
    """
    Simple demo:
      - 10-site chain
      - Mass "well" at the right side
      - Run matter + geometry backreaction
      - Plot:
          1) |psi(x,t)|^2 as spacetime density map
          2) pi_i(t) for each bond
    """
    n_sites = 10
    dt = 0.05

    # Put a shallow mass well near the right
    mass_profile = np.zeros(n_sites)
    mass_profile[6:9] = -0.5  # slightly attractive region

    sim = AdaptiveMatterGeometry1D(
        n_sites=n_sites,
        dt=dt,
        J=1.0,
        eta_E=0.8,
        mu_pi=0.3,
        pi0=1.0,
        E_target=-0.3,           # bonds try to settle to this local energy
        mass_profile=mass_profile,
        init_site=0,             # start particle on the left end
    )

    steps = 300
    sim.run(steps=steps)
    density, pi_hist = sim.get_history_arrays()

    # ---- Plot matter density |psi(x,t)|^2 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    im = ax0.imshow(density.T, origin='lower', aspect='auto',
                    extent=[0, steps * dt, 0, n_sites - 1])
    ax0.set_xlabel("time")
    ax0.set_ylabel("site index")
    ax0.set_title("Matter density |ψ(x,t)|²")
    fig.colorbar(im, ax=ax0, label="probability")

    # ---- Plot geometry evolution π_i(t) ----
    ax1 = axes[1]
    t = np.arange(steps) * dt
    for i in range(n_sites - 1):
        ax1.plot(t, pi_hist[:, i], label=f"π_{i}-{i+1}")
    ax1.set_xlabel("time")
    ax1.set_ylabel("π (edge strength)")
    ax1.set_title("Geometry backreaction π_i(t)")
    ax1.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()