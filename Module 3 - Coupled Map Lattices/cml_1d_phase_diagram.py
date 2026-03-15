"""
Module 3: Coupled Map Lattices
================================
1D CML — Spatiotemporal Chaos and Phase Diagram (Kaneko 1984)

The 1D diffusively-coupled logistic CML on a ring of N sites:

    x_i(t+1) = (1 - ε) f(x_i(t))  +  (ε/2) [f(x_{i-1}(t)) + f(x_{i+1}(t))]

    f(x) = r x (1 - x)   (logistic map)

Kaneko identified five dynamical regimes in (r, ε) parameter space:
  I.   Frozen random pattern
  II.  Pattern selection / traveling waves
  III. Spatiotemporal intermittency
  IV.  Fully-developed spatiotemporal chaos

The maximum Lyapunov exponent λ_max is computed via tangent-vector
renormalization to quantify the onset of spatiotemporal chaos.

References
----------
- Kaneko, K. (1984). Period-doubling of kink-antikink patterns, quasi-periodicity,
  antiintegrability and spatial intermittency. Progress of Theoretical Physics,
  72(3), 480-496.
- Kaneko, K. (1989). Spatiotemporal chaos in one- and two-dimensional coupled
  map lattices. Physica D, 37(1-3), 60-82.

Author: Soumadip R. Bhowmick, MS22074
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12


# ---------------------------------------------------------------------------
# CML dynamics
# ---------------------------------------------------------------------------

def logistic(x, r):
    """Logistic map f(x) = r x (1 - x)."""
    return r * x * (1.0 - x)


def cml_step(x, r, eps):
    """One synchronous update of the 1D diffusive CML with periodic BCs."""
    fx = logistic(x, r)
    return (1.0 - eps) * fx + 0.5 * eps * (np.roll(fx, -1) + np.roll(fx, 1))


def run_cml(r, eps, N=256, T_trans=500, T_run=300, seed=None):
    """
    Evolve the CML for T_run steps after T_trans transient steps.
    Returns the (T_run × N) trajectory array.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.4, 0.6, N)
    for _ in range(T_trans):
        x = cml_step(x, r, eps)
    traj = np.empty((T_run, N))
    for t in range(T_run):
        x = cml_step(x, r, eps)
        traj[t] = x
    return traj


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def spatial_disorder(traj):
    """
    Time-averaged spatial standard deviation (order parameter ρ).
    ρ → 0: coherent / uniform state; ρ > 0: spatially disordered.
    """
    return float(np.mean(np.std(traj, axis=1)))


def max_lyapunov_exponent(r, eps, N=128, T_trans=500, T_run=1000, seed=0):
    """
    Maximum Lyapunov exponent via tangent-vector renormalization.

    Tangent map (linearisation of the CML around the trajectory):
        δx_i' = (1-ε) f'(x_i) δx_i  +  (ε/2)[f'(x_{i-1}) δx_{i-1}
                                               + f'(x_{i+1}) δx_{i+1}]
    f'(x) = r (1 - 2x)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.4, 0.6, N)
    for _ in range(T_trans):
        x = cml_step(x, r, eps)

    delta = rng.standard_normal(N)
    delta /= np.linalg.norm(delta)

    log_sum = 0.0
    for _ in range(T_run):
        fp = r * (1.0 - 2.0 * x)          # f'(x_i) at current state
        Jd = fp * delta
        delta = (1.0 - eps) * Jd + 0.5 * eps * (np.roll(Jd, -1) + np.roll(Jd, 1))
        norm = np.linalg.norm(delta)
        log_sum += np.log(norm)
        delta /= norm
        x = cml_step(x, r, eps)

    return log_sum / T_run


# ===========================================================================
# Figure 1 — Space-time diagrams for five Kaneko dynamical regimes
# ===========================================================================

REGIMES = [
    ("Frozen Random Pattern\n$r=3.50,\\ \\varepsilon=0.40$",       3.50, 0.40),
    ("Pattern Selection\n$r=3.60,\\ \\varepsilon=0.25$",           3.60, 0.25),
    ("Traveling Waves\n$r=3.70,\\ \\varepsilon=0.15$",             3.70, 0.15),
    ("Spatiotemporal\nIntermittency\n$r=3.80,\\ \\varepsilon=0.10$", 3.80, 0.10),
    ("Fully-Developed\nSpatiotemporal Chaos\n$r=3.95,\\ \\varepsilon=0.04$", 3.95, 0.04),
]

N_st, T_st = 200, 300

fig, axes = plt.subplots(1, 5, figsize=(16, 5.5), constrained_layout=True)
fig.suptitle("1D CML — Five Dynamical Regimes (Space–Time Diagrams)",
             fontsize=13, fontweight='bold')

for ax, (label, r, eps) in zip(axes, REGIMES):
    traj = run_cml(r, eps, N=N_st, T_trans=500, T_run=T_st, seed=42)
    im = ax.imshow(traj, aspect='auto', origin='upper', cmap='inferno',
                   vmin=0, vmax=1, extent=[0, N_st, T_st, 0])
    ax.set_title(label, fontsize=8.5, pad=4)
    ax.set_xlabel("Site $i$", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Time step $t$", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label=r"$x_i$")

plt.savefig("simulations/cml_spacetime_regimes.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_spacetime_regimes.png")


# ===========================================================================
# Figure 2 — Phase diagram: spatial disorder ρ(r, ε) and λ_max(r, ε)
# ===========================================================================

Nr, Neps = 35, 30
r_vals   = np.linspace(3.40, 4.00, Nr)
eps_vals = np.linspace(0.00, 0.50, Neps)

print("Computing phase diagram (may take ~2 min) …")
rho_map = np.zeros((Neps, Nr))
lam_map = np.zeros((Neps, Nr))

for i, eps in enumerate(eps_vals):
    for j, r in enumerate(r_vals):
        traj = run_cml(r, eps, N=128, T_trans=300, T_run=150, seed=0)
        rho_map[i, j] = spatial_disorder(traj)
        lam_map[i, j] = max_lyapunov_exponent(r, eps, N=128,
                                               T_trans=300, T_run=400, seed=0)
    print(f"  ε = {eps:.3f}  done")

R, E = np.meshgrid(r_vals, eps_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("1D CML Phase Diagram", fontsize=13, fontweight='bold')

im0 = axes[0].pcolormesh(R, E, rho_map, cmap='plasma', shading='auto')
axes[0].set_xlabel(r"Logistic parameter $r$", fontsize=12)
axes[0].set_ylabel(r"Coupling strength $\varepsilon$", fontsize=12)
axes[0].set_title(r"Spatial disorder $\rho$", fontsize=12)
plt.colorbar(im0, ax=axes[0],
             label=r"$\rho = \langle\,\sigma_{\mathrm{space}}\,\rangle_t$")

im1 = axes[1].pcolormesh(R, E, lam_map, cmap='RdBu_r',
                          vmin=-0.6, vmax=0.6, shading='auto')
axes[1].set_xlabel(r"Logistic parameter $r$", fontsize=12)
axes[1].set_ylabel(r"Coupling strength $\varepsilon$", fontsize=12)
axes[1].set_title(r"Max Lyapunov exponent $\lambda_{\max}$", fontsize=12)
plt.colorbar(im1, ax=axes[1], label=r"$\lambda_{\max}$")

cs = axes[1].contour(R, E, lam_map, levels=[0.0], colors='k', linewidths=1.5)
axes[1].clabel(cs, fmt=r"$\lambda_{\max}=0$", fontsize=9)

for ax in axes:
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)

plt.tight_layout()
plt.savefig("simulations/cml_phase_diagram.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_phase_diagram.png")


# ===========================================================================
# Figure 3 — Lyapunov exponent vs r for different coupling strengths
# ===========================================================================

r_scan   = np.linspace(2.8, 4.0, 80)
eps_list = [0.0, 0.10, 0.20, 0.40]

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(eps_list)))

for eps, col in zip(eps_list, colors):
    lams = [max_lyapunov_exponent(r, eps, N=128, T_trans=400, T_run=600, seed=1)
            for r in r_scan]
    ax.plot(r_scan, lams, color=col, lw=1.5, label=fr"$\varepsilon={eps}$")

ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r"Logistic parameter $r$", fontsize=12)
ax.set_ylabel(r"Max Lyapunov exponent $\lambda_{\max}$", fontsize=12)
ax.set_title("CML: Lyapunov Exponent vs $r$ for Different Coupling Strengths",
             fontsize=12)
ax.legend(fontsize=11)
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True)

plt.tight_layout()
plt.savefig("simulations/cml_lyapunov_vs_r.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_lyapunov_vs_r.png")
