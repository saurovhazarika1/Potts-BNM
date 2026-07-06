# MD Probability-Based Communication Analysis

This repository implements a Potts Hamiltonian-based framework for analyzing allosteric communication pathways from molecular dynamics (MD) simulations.

Rather than partitioning conformations using geometric similarity (e.g., RMSD or PCA), this method uses the equilibrium probability of MD windows computed from a learned Potts Hamiltonian to identify thermodynamically favorable and unfavorable regions of the conformational ensemble. Communication pathways are then reconstructed and analyzed separately within these regions.

The method is built on the Potts+BNM framework, in which the interaction network is constrained by a Bayesian Network learned from MD trajectories.

---

## Workflow

```
MD Trajectory
      │
      ▼
Discrete Residue States
      │
      ▼
Bayesian Network (BaNDyT)
      │
      ▼
Potts+BNM Parameter Learning
      │
      ▼
Frame Potts Energy
      │
      ▼
Window Averaging
      │
      ▼
Boltzmann Probability
      │
      ▼
Probability-Based MD Regions
      │
      ▼
Communication Network Construction
      │
      ▼
Shortest Communication Paths
      │
      ▼
Path Energies & Path Probabilities
      │
      ▼
Communication Analysis
```

---

## Required Preprocessing

This repository assumes that the Potts+BNM model has already been trained.

Required inputs are:

- MD trajectory converted into discrete residue states
- Bayesian Network inferred from the discrete trajectory
- Learned Potts parameters (`h.npy` and `J.npy`)

Bayesian Networks can be generated using BaNDyT.

---

## Communication Network Construction

The learned Potts couplings define a weighted residue interaction network.

For each interacting residue pair:

$$S_{ij} = \|J_{ij}\|_F$$

where $S_{ij}$ is the Frobenius norm of the Potts coupling tensor.

Communication edge costs are then defined as:

$$c_{ij} = S_{\max} - S_{ij} + \varepsilon$$

so that stronger Potts couplings correspond to lower communication cost.

---

## Communication Path Identification

Communication pathways between the specified source and target residues are generated using Dijkstra's shortest-path algorithm (with optional shortest-simple-path enumeration).

For each path, the communication energy is computed from the Potts Hamiltonian. Two energy definitions are supported:

- **HJ** — single-site fields + pairwise couplings
- **J_ONLY** — pairwise couplings only

Both length-normalized versions (`HJ_MEAN` and `J_ONLY_MEAN`) are also available.

---

## MD Probability Landscape

For every MD frame, the full Potts Hamiltonian is evaluated:

$$E_{\mathrm{MD}}(t) = -\sum_i h_i(x_i) - \sum_{i<j} J_{ij}(x_i,x_j)$$

Frame energies are averaged over user-defined windows:

$$E_{\mathrm{MD}}(w) = \frac{1}{N_w} \sum_{t \in w} E_{\mathrm{MD}}(t)$$

and converted into equilibrium probabilities:

$$P_{\mathrm{MD}}(w) = \frac{e^{-\beta E_{\mathrm{MD}}(w)}}{\sum_{w'} e^{-\beta E_{\mathrm{MD}}(w')}}$$

The MD windows are ranked according to their equilibrium probability and partitioned into **highMD** and **lowMD** regions.

---

## Path Probability

Within each MD window, every communication pathway is assigned a Boltzmann probability:

$$P_{\mathrm{path}}(p) = \frac{e^{-\beta E_{\mathrm{path}}(p)}}{\sum_q e^{-\beta E_{\mathrm{path}}(q)}}$$

Only the most probable communication pathways are retained for downstream analyses.

---

## Analyses Performed

The repository computes a comprehensive set of communication metrics, including:

- Communication-path occupancy
- Path persistence
- Communication-energy spectra
- Path probability distributions
- Path diversity
- Positive/negative pathway correlations
- Communication frustration
- Robustness analyses across parameter settings
- Correlations with experimental functional measurements

---

## Outputs

The workflow automatically generates:

- Communication pathways
- Path probability tables
- Communication metrics
- CSV summaries
- Publication-quality figures
- Robustness analyses
- Correlation analyses against experimental efficacy
