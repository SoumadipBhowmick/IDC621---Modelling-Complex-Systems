"""
Module 3: Coupled Map Lattices
================================
CML Synchronization — Drive-Response Coupling and Critical Transition

Two coupled 1D CML rings operate in a unidirectional (drive-response)
configuration. The drive CML (x) evolves freely; the response CML (y)
follows its own internal dynamics but is also pulled toward x with
coupling strength σ:

    x_i(t+1) = (1−ε) f(x_i) + (ε/2) [f(x_{i−1}) + f(x_{i+1})]

    y_i(t+1) = (1−ε) f(y_i) + (ε/2) [f(y_{i−1}) + f(y_{i+1})]
               + σ [x_i(t) − y_i(t)]

    f(x) = r x (1 − x)   (logistic map)

The synchronization error

    E(t) = (1/N) Σ_i [x_i(t) − y_i(t)]²

decays to zero (complete synchronization) once σ exceeds a critical
threshold σ_c.  This threshold is related to the maximum transverse
Lyapunov exponent of the synchronized manifold (Pecora–Carroll 1990).

References
----------
- Pecora, L. M., & Carroll, T. L. (1990). Synchronization in chaotic systems.
  Phys. Rev. Lett., 64(8), 821.
- Kocarev, L., & Parlitz, U. (1996). Generalized synchronization, predictability,
  and equivalence of unidirectionally coupled dynamical systems.
  Phys. Rev. Lett., 76(11), 1816.
- Boccaletti, S., et al. (2002). The synchronization of chaotic systems.
  Phys. Rep., 366(1-2), 1-101.

Author: Soumadip R. Bhowmick, MS22074
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12


# ---------------------------------------------------------------------------
# CML dynamics (identical to cml_1d_phase_diagram.py)
# ---------------------------------------------------------------------------

def logistic(x, r):
    return r * x * (1.0 - x)


def cml_step(x, r, eps):
    """One synchronous update of the 1D diffusive CML (periodic BCs)."""
    fx = logistic(x, r)
    return (1.0 - eps) * fx + 0.5 * eps * (np.roll(fx, -1) + np.roll(fx, 1))


def coupled_cml_step(x, y, r, eps_int, sigma):
    """
    One step of the drive (x) and response (y) systems.
    sigma: unidirectional pull from x to y.
    """
    x_new = cml_step(x, r, eps_int)
    y_internal = cml_step(y, r, eps_int)
    y_new = y_internal + sigma * (x - y)
    return x_new, np.clip(y_new, 0.0, 1.0)


def sync_error(x, y):
    """Mean-squared synchronization error."""
    return float(np.mean((x - y) ** 2))


# ===========================================================================
# Figure 1 — Error vs time for three coupling regimes
#            + space-time diagram of x_i - y_i before and after coupling
# ===========================================================================

N_sync = 200
T_trans = 200
T_run   = 600
r0      = 3.9           # deep in the spatiotemporally chaotic regime
eps0    = 0.10

SIGMA_CASES = [
    (0.00, r"$\sigma=0.00$ (no coupling)",  '#e74c3c'),
    (0.05, r"$\sigma=0.05$ (partial sync)", '#f39c12'),
    (0.20, r"$\sigma=0.20$ (synchronized)", '#27ae60'),
]

np.random.seed(7)
fig, (ax_err, ax_st) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("CML Synchronization via Unidirectional Drive–Response Coupling",
             fontsize=13, fontweight='bold')

for sigma, label, col in SIGMA_CASES:
    x = np.random.rand(N_sync) * 0.4 + 0.3
    y = np.clip(x + 0.3 * np.random.randn(N_sync), 0.0, 1.0)
    for _ in range(T_trans):
        x, y = coupled_cml_step(x, y, r0, eps0, sigma)
    errors = np.empty(T_run)
    for t in range(T_run):
        x, y = coupled_cml_step(x, y, r0, eps0, sigma)
        errors[t] = sync_error(x, y)
    ax_err.semilogy(errors, color=col, lw=1.4, label=label)

ax_err.set_xlabel("Time step $t$", fontsize=12)
ax_err.set_ylabel(r"$E(t) = \langle(x_i - y_i)^2\rangle$", fontsize=11)
ax_err.set_title("Synchronization Error vs Time", fontsize=12)
ax_err.legend(fontsize=10)
ax_err.minorticks_on()
ax_err.tick_params(which='both', direction='in', top=True, right=True)

# --- Build space-time diagram of x_i - y_i: 100 steps free, then 100 coupled ---
sigma_demo = 0.20
x = np.random.rand(N_sync) * 0.4 + 0.3
y = np.clip(x + 0.3 * np.random.randn(N_sync), 0.0, 1.0)

# Phase 1: free (no coupling) — let systems diverge
diff_before = np.empty((100, N_sync))
for t in range(100):
    x, y = coupled_cml_step(x, y, r0, eps0, 0.0)
    diff_before[t] = x - y

# Phase 2: turn on coupling — watch synchronization
diff_after = np.empty((100, N_sync))
for t in range(100):
    x, y = coupled_cml_step(x, y, r0, eps0, sigma_demo)
    diff_after[t] = x - y

diff_combined = np.vstack([diff_before, diff_after])
im_st = ax_st.imshow(diff_combined, aspect='auto', origin='upper',
                      cmap='seismic', vmin=-0.5, vmax=0.5,
                      extent=[0, N_sync, 200, 0])
ax_st.axhline(100, color='k', lw=1.5, ls='--',
              label=fr"coupling $\sigma={sigma_demo}$ switched on")
ax_st.set_xlabel("Site $i$", fontsize=12)
ax_st.set_ylabel("Time step $t$", fontsize=12)
ax_st.set_title(r"$x_i(t) - y_i(t)$: desynchronized → synchronized", fontsize=12)
ax_st.legend(fontsize=10, loc='upper right')
plt.colorbar(im_st, ax=ax_st, label=r"$x_i - y_i$")

plt.tight_layout()
plt.savefig("simulations/cml_synchronization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_synchronization.png")


# ===========================================================================
# Figure 2 — Critical coupling: time-averaged ⟨E⟩ vs σ
# ===========================================================================

print("Scanning σ for the synchronization transition …")

sigma_scan  = np.linspace(0.0, 0.40, 50)
T_trans2    = 600
T_avg       = 300
E_final     = []
LOG_EPSILON = 1e-15   # guard against log(0) when E_final is exactly zero

np.random.seed(42)
for sigma in sigma_scan:
    x = np.random.rand(N_sync) * 0.4 + 0.3
    y = np.clip(x + 0.4 * np.random.randn(N_sync), 0.0, 1.0)
    for _ in range(T_trans2):
        x, y = coupled_cml_step(x, y, r0, eps0, sigma)
    running = 0.0
    for _ in range(T_avg):
        x, y = coupled_cml_step(x, y, r0, eps0, sigma)
        running += sync_error(x, y)
    E_final.append(running / T_avg)

E_arr = np.array(E_final)

# Estimate σ_c as the steepest descent in log(E)
grad = np.gradient(np.log10(E_arr + LOG_EPSILON), sigma_scan)
sigma_c = float(sigma_scan[np.argmin(grad)])

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(sigma_scan, E_arr, 'o-', color='#2c7bb6', ms=4, lw=1.5)
ax.axvline(sigma_c, color='r', ls='--', lw=1.5,
           label=fr"$\sigma_c \approx {sigma_c:.2f}$")
ax.set_xlabel(r"Unidirectional coupling strength $\sigma$", fontsize=12)
ax.set_ylabel(r"Time-averaged sync. error $\langle E \rangle$", fontsize=12)
ax.set_title(r"CML Synchronization Transition ($r=3.9,\;\varepsilon=0.10$)",
             fontsize=12)
ax.legend(fontsize=11)
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True)

plt.tight_layout()
plt.savefig("simulations/cml_sync_transition.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_sync_transition.png")
print(f"Estimated critical coupling: σ_c ≈ {sigma_c:.3f}")
