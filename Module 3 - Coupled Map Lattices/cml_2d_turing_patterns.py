"""
Module 3: Coupled Map Lattices
================================
2D CML — Turing Pattern Formation via Gray-Scott Reaction-Diffusion

The Gray-Scott model (Pearson 1993) on a 2D lattice:

    ∂u/∂t = D_u ∇²u  −  u v²  +  F (1 − u)
    ∂v/∂t = D_v ∇²v  +  u v²  −  (F + k) v

Discretised with forward Euler on a periodic lattice (CML framework):

    u_{i,j}(t+1) = u_{i,j}(t) + Δt [D_u Δu − u v² + F(1−u)]
    v_{i,j}(t+1) = v_{i,j}(t) + Δt [D_v Δv + u v² − (F+k) v]

The coupling between sites enters through the discrete 2D Laplacian Δ.
Different (F, k) pairs select distinct Turing pattern morphologies that
closely parallel those seen in animal markings (Turing 1952).

Pattern types explored:
  - Spots  : isolated activator spots (leopard-like)
  - Stripes: periodic stripe patterns (zebra-like)
  - Mazes  : labyrinthine / coral patterns
  - Waves  : target patterns and spirals

References
----------
- Turing, A. M. (1952). The chemical basis of morphogenesis.
  Phil. Trans. Roy. Soc. London B, 237, 37-72.
- Pearson, J. E. (1993). Complex patterns in a simple system.
  Science, 261(5118), 189-192.
- Gray, P., & Scott, S. K. (1983). Autocatalytic reactions in the isothermal
  continuous stirred tank reactor. Chemical Engineering Science, 38(1), 29-43.

Author: Soumadip R. Bhowmick, MS22074
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12


# ---------------------------------------------------------------------------
# 2D discrete Laplacian — 5-point stencil with periodic BCs
# ---------------------------------------------------------------------------

def laplacian2d(Z):
    """Discrete 2D Laplacian: Δf_{i,j} = f_{i+1,j}+f_{i-1,j}+f_{i,j+1}+f_{i,j-1} - 4 f_{i,j}."""
    return (np.roll(Z,  1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z,  1, 1) + np.roll(Z, -1, 1) - 4.0 * Z)


# ---------------------------------------------------------------------------
# Gray-Scott CML time step
# ---------------------------------------------------------------------------

def gray_scott_step(u, v, Du, Dv, F, k, dt):
    """One forward-Euler step of the Gray-Scott system."""
    uvv = u * v * v
    u_new = u + dt * (Du * laplacian2d(u) - uvv + F * (1.0 - u))
    v_new = v + dt * (Dv * laplacian2d(v) + uvv - (F + k) * v)
    return np.clip(u_new, 0.0, 1.0), np.clip(v_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Initialisation: homogeneous state with small seeded perturbations
# ---------------------------------------------------------------------------

def init_gray_scott(N, n_seeds=25, seed=0):
    """
    Start near the homogeneous steady state (u≈1, v≈0) and perturb
    a set of random square patches to nucleate pattern growth.
    """
    rng = np.random.default_rng(seed)
    u = np.ones((N, N))
    v = np.zeros((N, N))
    for _ in range(n_seeds):
        cx, cy = rng.integers(10, N - 10, size=2)
        r = rng.integers(4, 9)
        sl = (slice(cx - r, cx + r), slice(cy - r, cy + r))
        u[sl] = 0.50 + 0.02 * rng.standard_normal((2 * r, 2 * r))
        v[sl] = 0.25 + 0.02 * rng.standard_normal((2 * r, 2 * r))
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def simulate_gray_scott(N, F, k, Du=0.16, Dv=0.08, dt=1.0,
                         T_run=10_000, n_seeds=25, seed=0):
    """Run the 2D Gray-Scott CML for T_run steps and return (u, v)."""
    u, v = init_gray_scott(N, n_seeds=n_seeds, seed=seed)
    for _ in range(T_run):
        u, v = gray_scott_step(u, v, Du, Dv, F, k, dt)
    return u, v


# ===========================================================================
# Figure 1 — Four canonical Turing pattern morphologies
# ===========================================================================

PATTERNS = [
    ("Spots\n$F=0.022,\\ k=0.051$",   0.022, 0.051),
    ("Stripes\n$F=0.035,\\ k=0.065$", 0.035, 0.065),
    ("Mazes\n$F=0.029,\\ k=0.057$",   0.029, 0.057),
    ("Waves\n$F=0.014,\\ k=0.054$",   0.014, 0.054),
]

N_gs = 128
T_gs = 12_000

fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
fig.suptitle(
    "2D CML: Gray–Scott Reaction–Diffusion — Turing Pattern Morphologies\n"
    r"($D_u = 0.16,\; D_v = 0.08$)",
    fontsize=13, fontweight='bold'
)

for col, (label, F, k) in enumerate(PATTERNS):
    print(f"Simulating {label.split(chr(10))[0]} (F={F}, k={k}) …")
    u, v = simulate_gray_scott(N_gs, F, k, T_run=T_gs, seed=col)

    axes[0, col].imshow(u, cmap='RdBu', origin='lower', vmin=0, vmax=1)
    axes[0, col].set_title(label, fontsize=10)
    axes[0, col].set_xticks([])
    axes[0, col].set_yticks([])
    if col == 0:
        axes[0, col].set_ylabel("Activator $u$", fontsize=10)

    im = axes[1, col].imshow(v, cmap='RdBu', origin='lower', vmin=0, vmax=0.4)
    axes[1, col].set_xticks([])
    axes[1, col].set_yticks([])
    if col == 0:
        axes[1, col].set_ylabel("Inhibitor $v$", fontsize=10)

    plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

plt.savefig("simulations/cml_turing_patterns.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_turing_patterns.png")


# ===========================================================================
# Figure 2 — F-k morphology map: mean inhibitor concentration ⟨v⟩
# ===========================================================================

print("Computing F–k morphology map (coarse scan) …")

NF, Nk = 18, 18
F_vals = np.linspace(0.010, 0.060, NF)
k_vals = np.linspace(0.045, 0.075, Nk)

v_mean = np.zeros((Nk, NF))
N_scan = 64
T_scan = 6_000

for i, k in enumerate(k_vals):
    for j, F in enumerate(F_vals):
        _, v = simulate_gray_scott(N_scan, F, k, T_run=T_scan, seed=0, n_seeds=15)
        v_mean[i, j] = float(np.mean(v))
    print(f"  k = {k:.3f}  done")

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.pcolormesh(F_vals, k_vals, v_mean, cmap='magma', shading='auto')
ax.set_xlabel(r"Feed rate $F$", fontsize=12)
ax.set_ylabel(r"Kill rate $k$", fontsize=12)
ax.set_title(r"Gray–Scott Morphology Map: $\langle v \rangle$ at $t=6000$", fontsize=12)
plt.colorbar(im, ax=ax, label=r"$\langle v \rangle$")

ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True)

# Annotate known morphology regions
for Fa, ka, name in [(0.022, 0.051, "spots"),
                      (0.035, 0.065, "stripes"),
                      (0.029, 0.057, "mazes")]:
    ax.annotate(name, (Fa, ka), fontsize=9, color='cyan', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='k', alpha=0.5))

plt.tight_layout()
plt.savefig("simulations/cml_gs_morphology_map.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: simulations/cml_gs_morphology_map.png")
