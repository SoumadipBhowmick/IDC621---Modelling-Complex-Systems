# Module 3: Coupled Map Lattices

This is the submission for Module 3 of the course **IDC621 — Modelling Complex Systems**.

## Topic: Spatiotemporal Chaos, Turing Pattern Formation & Synchronization

Coupled Map Lattices (CMLs) are spatially extended dynamical systems in which
each lattice site evolves according to a discrete-time map, coupled to its
neighbours. Introduced by Kaneko (1984), CMLs are a minimal yet powerful
framework for studying:

- **Spatiotemporal chaos** and bifurcation cascades
- **Pattern formation** and Turing instabilities
- **Collective synchronization** in extended nonlinear systems

---

## Files

| File | Description |
|------|-------------|
| `cml_1d_phase_diagram.py` | 1D Kaneko CML: five dynamical regimes, phase diagram ρ(r,ε) and λ_max(r,ε), Lyapunov exponent vs r |
| `cml_2d_turing_patterns.py` | 2D Gray–Scott reaction–diffusion CML: spots, stripes, mazes, waves; F–k morphology map |
| `cml_synchronization.py` | Drive–response CML: synchronization error vs time, critical coupling σ_c |
| `simulations/` | Output figures from each script |

Language: Python (NumPy, Matplotlib)  
Author: Soumadip R. Bhowmick, MS22074

---

## Physics Concepts

### 1 — Spatiotemporal Chaos and Phase Diagram (`cml_1d_phase_diagram.py`)

The **1D diffusively-coupled logistic CML** (Kaneko 1984):

$$x_i(t+1) = (1-\varepsilon)\,f(x_i(t)) \;+\; \frac{\varepsilon}{2}\bigl[f(x_{i-1}(t)) + f(x_{i+1}(t))\bigr]$$

$$f(x) = r\,x\,(1-x) \qquad \text{(logistic map)}$$

**Five dynamical regimes** are identified in the $(r, \varepsilon)$ parameter space:

| Regime | $(r, \varepsilon)$ example | Description |
|--------|---------------------------|-------------|
| Frozen random pattern | (3.50, 0.40) | Spatial disorder frozen in time |
| Pattern selection | (3.60, 0.25) | Periodic spatial structures |
| Traveling waves | (3.70, 0.15) | Moving kink-antikink fronts |
| Spatiotemporal intermittency | (3.80, 0.10) | Laminar–turbulent coexistence |
| Fully-developed spatiotemporal chaos | (3.95, 0.04) | Positive Lyapunov exponent everywhere |

The **maximum Lyapunov exponent** $\lambda_{\max}$ is computed via tangent-vector renormalization. The zero contour $\lambda_{\max}=0$ marks the onset of chaos in the phase diagram.

**Outputs:**  
`simulations/cml_spacetime_regimes.png` — space-time diagrams for all five regimes  
`simulations/cml_phase_diagram.png` — $\rho(r,\varepsilon)$ and $\lambda_{\max}(r,\varepsilon)$ maps  
`simulations/cml_lyapunov_vs_r.png` — Lyapunov exponent vs $r$ at several $\varepsilon$

---

### 2 — Turing Pattern Formation (`cml_2d_turing_patterns.py`)

The **Gray–Scott reaction–diffusion system** on a 2D periodic lattice:

$$\frac{\partial u}{\partial t} = D_u\,\nabla^2 u \;-\; uv^2 \;+\; F(1-u)$$

$$\frac{\partial v}{\partial t} = D_v\,\nabla^2 v \;+\; uv^2 \;-\; (F+k)\,v$$

This is a direct implementation of Turing's (1952) reaction–diffusion mechanism for morphogenesis. The inter-site coupling enters through the discrete 2D Laplacian, making this a two-component CML.

Four canonical **Turing pattern morphologies** are produced:

| Pattern | $(F, k)$ | Physical analogue |
|---------|-----------|-------------------|
| Spots   | (0.022, 0.051) | Leopard / cheetah markings |
| Stripes | (0.035, 0.065) | Zebra / tiger stripes |
| Mazes   | (0.029, 0.057) | Coral / brain coral |
| Waves   | (0.014, 0.054) | Spiral / target waves |

**Outputs:**  
`simulations/cml_turing_patterns.png` — four pattern morphologies (activator & inhibitor)  
`simulations/cml_gs_morphology_map.png` — F–k parameter space morphology map

---

### 3 — CML Synchronization (`cml_synchronization.py`)

Two 1D CML rings in a **drive–response** configuration:

$$y_i(t+1) = (1-\varepsilon)\,f(y_i) + \frac{\varepsilon}{2}\bigl[f(y_{i-1})+f(y_{i+1})\bigr] + \sigma\,\bigl[x_i(t)-y_i(t)\bigr]$$

The **synchronization error** $E(t) = \frac{1}{N}\sum_i [x_i-y_i]^2$ decays to zero when $\sigma$ exceeds a **critical coupling** $\sigma_c$, analogous to the Pecora–Carroll (1990) criterion for chaotic synchronization. The space-time diagram of $x_i - y_i$ visually demonstrates the transition from incoherence to synchrony.

**Outputs:**  
`simulations/cml_synchronization.png` — error vs time + space-time diagram  
`simulations/cml_sync_transition.png` — $\langle E \rangle$ vs $\sigma$ with $\sigma_c$ annotated

---

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

## Running

```bash
# Figure 1: 1D phase diagram, space-time diagrams, Lyapunov exponents
python cml_1d_phase_diagram.py

# Figure 2: 2D Turing patterns and morphology map
python cml_2d_turing_patterns.py

# Figure 3: Synchronization phenomena
python cml_synchronization.py
```

> **Note:** `cml_1d_phase_diagram.py` and `cml_2d_turing_patterns.py` include
> parameter sweeps that may take a few minutes to complete.

---

## References

1. Kaneko, K. (1984). Period-doubling of kink-antikink patterns,
   quasi-periodicity, antiintegrability and spatial intermittency.
   *Progress of Theoretical Physics*, 72(3), 480–496.

2. Kaneko, K. (1989). Spatiotemporal chaos in one- and two-dimensional
   coupled map lattices. *Physica D*, 37(1–3), 60–82.

3. Turing, A. M. (1952). The chemical basis of morphogenesis.
   *Phil. Trans. Roy. Soc. London B*, 237, 37–72.

4. Pearson, J. E. (1993). Complex patterns in a simple system.
   *Science*, 261(5118), 189–192.

5. Pecora, L. M., & Carroll, T. L. (1990). Synchronization in chaotic systems.
   *Phys. Rev. Lett.*, 64(8), 821.

6. Boccaletti, S., et al. (2002). The synchronization of chaotic systems.
   *Phys. Rep.*, 366(1–2), 1–101.
