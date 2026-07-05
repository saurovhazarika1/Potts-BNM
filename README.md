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

$$
E_{\mathrm{MD}}(t)
=
-
\sum_i h_i(x_i(t))
-
\sum_{i<j}
J_{ij}\left(x_i(t),x_j(t)\right),
$$

where

- $x_i(t)$ is the discrete state of residue $i$ at frame $t$,
- $h_i$ denotes the single-site field,
- $J_{ij}$ denotes the pairwise coupling between residues.

This energy represents the interaction energy of the complete protein for that frame.

---

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

$$
E_{\mathrm{MD}}(w)
=
\frac{1}{N_w}
\sum_{t\in w}
E_{\mathrm{MD}}(t),
$$

where $N_w$ is the number of frames in the window.

Window averaging reduces statistical noise and provides a natural timescale for communication analysis.

---

# 3. MD Window Probability

Each window energy is converted into an equilibrium probability using the Boltzmann distribution

$$
P_{\mathrm{MD}}(w)
=
\frac{
e^{-\beta E_{\mathrm{MD}}(w)}
}{
\sum_{w'}
e^{-\beta E_{\mathrm{MD}}(w')}
}.
$$

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

$$
P_{\mathrm{MD}}(w).
$$

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

$$
\text{highMD}
=
\text{low Potts energy},
$$

and

$$
\text{lowMD}
=
\text{high Potts energy}.
$$

---

# 5. Communication Path Construction

For every MD window,

1. the communication network is constructed,
2. communication pathways between the source and target residues are identified,
3. pathway energies and pathway probabilities are calculated.

The pathway energy is

$$
E_{\mathrm{path}}
=
-
\sum_{i\in\mathrm{path}}
h_i
-
\sum_{(i,j)\in\mathrm{path}}
J_{ij}.
$$

Depending on the selected energy mode, only the $J$ term or both $h$ and $J$ contributions may be included.

---

# Occupancy Analysis

Each communication pathway is represented by a binary activity trajectory

$$
A_p(w)
=
\begin{cases}
1,
&
\text{path is active}
\\
0,
&
\text{otherwise}.
\end{cases}
$$

Using this binary trajectory, the code computes

- fraction of paths ever active,
- mean run length,
- maximum run length,
- occupancy fraction,
- survival probability,

for both highMD and lowMD regions.

---

# Persistence Analysis

Persistence quantifies how long communication pathways remain active.

For lag time $\tau$,

$$
S(\tau)
=
P
\left(
A(t+\tau)=1
\mid
A(t)=1
\right).
$$

This is the probability that a pathway remains active after $\tau$ windows, given that it is active now.

The analysis computes

- global persistence,
- highMD persistence,
- lowMD persistence,
- ratios between highMD and lowMD persistence.

---

# Communication-Energy Spectrum Analysis

For each pathway, the communication energy distribution is analyzed separately within highMD and lowMD regions.

The following quantities are computed.

## Mean Path Energy

$$
\langle E_{\mathrm{path}}\rangle.
$$

---

## Energy Entropy

$$
S
=
-
\sum_i
p_i
\ln p_i,
$$

where $p_i$ denotes the probability of occupying energy bin $i$.

---

## Participation Ratio

$$
PR
=
\frac{1}
{\sum_i p_i^2}.
$$

This estimates the effective number of communication-energy states contributing to the ensemble.

---

## Effective Number of Energy Bins

The effective number of populated energy bins is calculated from the communication-energy distribution.

---

## Tail Polarization

Define

$$
f_{\mathrm{low}}
=
P(E<E_{10}),
$$

and

$$
f_{\mathrm{high}}
=
P(E>E_{90}),
$$

where $E_{10}$ and $E_{90}$ are the 10th and 90th percentiles of the pathway-energy distribution.

Tail polarization is

$$
TP
=
f_{\mathrm{low}}
+
f_{\mathrm{high}}.
$$

---

## Additional Metrics

The code also computes

- Tail asymmetry
- Tail ratio
- Tail log-ratio
- Jensen–Shannon divergence
- Wasserstein distance
- Cohen's d
- Mean energy difference
- Free-energy difference

between highMD and lowMD communication ensembles.

---

# Outputs

The analysis automatically generates

- CSV files containing all calculated metrics
- Robustness summaries
- Pearson and Spearman correlations
- Scatter plots versus experimental efficacy
- Heatmaps
- Robustness bar plots
- Publication-quality figures

---

# Interpretation

This workflow distinguishes two different quantities.

## MD Equilibrium Probability

The Potts Hamiltonian defines the equilibrium probability of each MD window

$$
P_{\mathrm{MD}}
=
\frac{
e^{-\beta E_{\mathrm{MD}}}
}
{Z},
$$

which identifies thermodynamically favorable and unfavorable conformational regions.

---

## Communication Properties

Within these regions, the communication network is analyzed through

- pathway energy,
- occupancy,
- persistence,
- communication diversity,
- energy spectrum,
- pathway participation.

Thus, the Potts model is **not** used to estimate the global conformational free-energy landscape. Instead, it provides an interaction-space Hamiltonian that enables quantitative analysis of communication pathways within different regions of the equilibrium ensemble.

---

# Citation

If you use this code in your work, please cite the associated publication (to be added).
