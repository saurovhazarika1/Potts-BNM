# MD Probability-Based Communication Analysis

This module analyzes communication pathways by partitioning molecular dynamics (MD) trajectories according to the **equilibrium probability** of each MD window computed from the Potts Hamiltonian. Rather than clustering structures using geometric coordinates (e.g., RMSD or PCA), the method classifies conformational regions using the statistical-mechanical energy learned by the Potts model.

The central objective is to investigate how communication pathways differ between thermodynamically favorable and unfavorable regions of the equilibrium ensemble.

---

# Required Preprocessing

Before running this workflow, the molecular dynamics (MD) trajectory must first be converted into discrete residue states and a corresponding Bayesian Network (BN). These preprocessing steps are **not** performed by this repository.

## 1. Per-residue Energy or Contact-State Calculation

For every MD frame, each residue must be assigned a discrete state. The Potts+BNM framework is independent of how these states are generated, and users may employ any suitable approach for discretizing the MD trajectory.

Possible approaches include, but are not limited to,

- Per-residue interaction energies
- Binary residue-residue contact states
- Distance-based contact maps
- Other residue-level descriptors derived from MD trajectories

The only requirement is that every residue is represented by a discrete state for each MD frame. These discrete residue states constitute the input to the Potts model.

## 2. Bayesian Network Construction

A Bayesian Network (BN) must then be inferred from the discretized MD data to identify the direct probabilistic dependencies between residues.

We recommend using **BaNDyT** for Bayesian Network construction:

**BaNDyT GitHub Repository**

https://github.com/bandyt-group/bandyt

The inferred Bayesian Network defines the network topology used by the Potts+BNM framework. Pairwise Potts couplings are learned only for residue pairs connected by Bayesian Network edges.

Once these preprocessing steps have been completed, the resulting discrete residue states and Bayesian Network can be used directly with the workflow described below.

---

# Workflow

```text
MD Trajectory
      │
      ▼
Per-residue Energy /
Contact-State Analysis
      │
      ▼
Discrete Residue States
      │
      ▼
Bayesian Network
(BaNDyT)
      │
      ▼
Potts Hamiltonian (h, J)
      │
      ▼
Frame Energy
      │
      ▼
Window Averaging
      │
      ▼
Boltzmann Probability
      │
      ▼
Probability Binning
      │
      ├──────────────┐
      ▼              ▼
   High MD        Low MD
      │              │
      └──────┬───────┘
             ▼
Communication Path Analysis
```

---

# 1. Potts Energy of Each MD Frame

For every MD frame, the Potts model assigns an interaction energy

```math
E_{\mathrm{MD}}(t)
=
-\sum_i h_i\!\left(x_i(t)\right)
-\sum_{i<j} J_{ij}\!\left(x_i(t),x_j(t)\right)
```

where

- $x_i(t)$ is the discrete state of residue $i$ at frame $t$,
- $h_i$ denotes the single-site field,
- $J_{ij}$ denotes the pairwise coupling between residues.

This energy represents the interaction energy of the complete protein for that frame.

# 2. Window Averaging

The MD trajectory is divided into consecutive windows of fixed size.

Example

```text
Window 1 : frames   0–99
Window 2 : frames 100–199
Window 3 : frames 200–299
...
```

The average energy of each window is

```math
E_{\mathrm{MD}}(w)
=
\frac{1}{N_w}
\sum_{t\in w}
E_{\mathrm{MD}}(t),
```

where $N_w$ is the number of frames in the window.

Window averaging reduces statistical noise and provides a natural timescale for communication analysis.

---

# 3. MD Window Probability

Each window energy is converted into an equilibrium probability using the Boltzmann distribution

```math
P_{\mathrm{MD}}(w)
=
\frac{
e^{-\beta E_{\mathrm{MD}}(w)}
}{
\sum_{w'}
e^{-\beta E_{\mathrm{MD}}(w')}
}.
```

Throughout this work,

```python
FRAME_BETA = 1.0
```

Therefore,

- lower-energy windows receive higher probability,
- higher-energy windows receive lower probability.

---

# 4. Defining High- and Low-Probability MD Regions

The MD windows are ranked according to

```math
P_{\mathrm{MD}}(w).
```

The ranked windows are divided into equal-population quantiles.

For example,

```python
N_PROB_BINS = 5
```

produces

```text
Bin 1  → Lowest probability
Bin 2
Bin 3
Bin 4
Bin 5  → Highest probability
```

The analysis defines

- **highMD** = highest-probability windows
- **lowMD** = lowest-probability windows

Equivalently,

```math
\text{highMD}
=
\text{low Potts energy},
```

and

```math
\text{lowMD}
=
\text{high Potts energy}.
```

---

# Citation

If you use this code in your work, please cite the associated publication (to be added).
