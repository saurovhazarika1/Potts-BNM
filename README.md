# MD Probability-Based Communication Analysis

This repository implements a Potts Hamiltonian-based framework for analyzing allosteric communication pathways from molecular dynamics (MD) simulations.

Rather than partitioning conformations using geometric similarity (e.g., RMSD or PCA), this method uses the **equilibrium probability** of MD windows computed from a learned Potts Hamiltonian to identify thermodynamically favorable and unfavorable regions of the conformational ensemble. Communication pathways are then reconstructed and analyzed separately within these regions.

The method is built on the Potts+BNM framework, in which the interaction network is constrained by a Bayesian Network learned from MD trajectories.

The central objective is to investigate how communication pathways differ between thermodynamically favorable and unfavorable regions of the equilibrium ensemble.

---

## Repository Structure

```
.
├──fit_potts_bnm.py # Potts+BNM parameter learning
├── communication_analysis.py          # Communication pathway analysis
├── energy_spectrum_analysis.py  # communication-energy spectrum changes between highMD and lowMD regions
└── README.md
```

---

## Workflow

```text
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
Probability Binning
      │
      ├──────────────┐
      ▼              ▼
   High MD        Low MD
      │              │
      └──────┬───────┘
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

## Model Training: Potts+BNM Parameter Learning

Before any communication analysis can be performed, the Potts Hamiltonian must be fit to the MD trajectory. This is done in `potts_fit_md_bn_elastic_graph.py`.

The Potts Hamiltonian is learned from MD trajectories and Bayesian Network samples by matching one- and two-body marginal distributions.

The target marginals are:

$$f_i^{\mathrm{target}} = (1-\alpha) f_i^{\mathrm{MD}} + \alpha f_i^{\mathrm{BN}}$$

and

$$f_{ij}^{\mathrm{target}} = (1-\alpha) f_{ij}^{\mathrm{MD}} + \alpha f_{ij}^{\mathrm{BN}}$$

where $\alpha$ controls the contribution of Bayesian Network samples relative to the raw MD statistics.

The parameters ($h$ and $J$) are optimized using Monte Carlo sampling and moment matching until convergence.

### Bayesian-Network-Constrained Couplings

Unlike a conventional Potts model, pairwise couplings are learned only between residues connected by Bayesian Network edges.

This dramatically reduces the number of free parameters while preserving the probabilistic dependency structure learned from MD, and is one of the central novelties of the Potts+BNM approach.

### Graph-Regularized Potts Model

To prevent overfitting, the pairwise couplings are regularized using graph-guided elastic penalties.

Edges inferred by the Bayesian Network receive weaker regularization than non-network residue pairs.

The optimization includes:

- L2 block regularization
- Group-L1 regularization
- Zero-sum gauge fixing

throughout training.

### Monte Carlo Sampling

Model expectations (the marginals used for moment matching) are estimated using Metropolis Monte Carlo sampling rather than computed analytically.

Each optimization iteration consists of:

1. Burn-in
2. Monte Carlo sweeps
3. Estimation of one-body marginals
4. Estimation of pairwise marginals
5. Parameter update

Adaptive Monte Carlo settings are used to keep sampling efficient as the parameters evolve during training.

### Training Outputs

The fitting procedure automatically saves:

- learned $h$ fields (`h.npy`)
- learned $J$ couplings (`J.npy`)
- Monte Carlo samples
- one-body marginals
- two-body marginals
- convergence history (`training_history.csv`)
- run metadata

---

## Required Preprocessing

The communication analysis in this repository (`communication_analysis.py`) assumes that the Potts+BNM model has already been trained as described above.

Required inputs are:

- MD trajectory converted into discrete residue states
- Bayesian Network inferred from the discrete trajectory
- Learned Potts parameters (`h.npy` and `J.npy`)

Bayesian Networks can be generated using BaNDyT.

---

## 1. Potts Energy of Each MD Frame

For every MD frame, the Potts model assigns a total statistical energy:

$E_{\mathrm{MD}}(t) = -\sum_i h_i(x_i(t)) - \sum_{i<j} J_{ij}(x_i(t), x_j(t))$

where:

- $x_i(t)$ is the discrete state of residue $i$ at frame $t$
- $h_i$ denotes the single-site (local) field for residue $i$
- $J_{ij}$ denotes the pairwise coupling between residues $i$ and $j$

This energy represents the total local-field and pairwise-interaction contributions to the Potts energy of the complete protein for that frame.

---

## 2. Window Averaging

The MD trajectory is divided into consecutive windows of fixed size.

Example:

```text
Window 1 : frames   0–99
Window 2 : frames 100–199
Window 3 : frames 200–299
...
```

The average energy of each window is:

$$E_{\mathrm{MD}}(w) = \frac{1}{N_w} \sum_{t \in w} E_{\mathrm{MD}}(t)$$

where $N_w$ is the number of frames in the window.

Window averaging reduces statistical noise and provides a natural timescale for communication analysis.

---

## 3. MD Window Probability

Each window energy is converted into an equilibrium probability using the Boltzmann distribution:

$$P_{\mathrm{MD}}(w) = \frac{e^{-\beta E_{\mathrm{MD}}(w)}}{\sum_{w'} e^{-\beta E_{\mathrm{MD}}(w')}}$$

Throughout this work:

```python
FRAME_BETA = 1.0
```

Therefore:

- lower-energy windows receive higher probability
- higher-energy windows receive lower probability

---

## 4. Defining High- and Low-Probability MD Regions

The MD windows are ranked according to $P_{\mathrm{MD}}(w)$ and divided into equal-population quantiles.

For example:

```python
N_PROB_BINS = 5
```

produces:

```text
Bin 1  → Lowest probability
Bin 2
Bin 3
Bin 4
Bin 5  → Highest probability
```

The analysis defines:

- **highMD** = highest-probability windows
- **lowMD** = lowest-probability windows

Equivalently:

$$\text{highMD} = \text{low Potts energy}$$

$$\text{lowMD} = \text{high Potts energy}$$

---

## 5. Communication Network Construction

The learned Potts couplings define a weighted residue interaction network.

For each interacting residue pair:

$$S_{ij} = \|J_{ij}\|_F$$

where $S_{ij}$ is the Frobenius norm of the Potts coupling tensor.

Communication edge costs are then defined as:

$$c_{ij} = S_{\max} - S_{ij} + \varepsilon$$

so that stronger Potts couplings correspond to lower communication cost.

---

## 6. Communication Path Construction

For every MD window:

1. the communication network is constructed
2. communication pathways between the source and target residues are identified using Dijkstra's shortest-path algorithm (with optional shortest-simple-path enumeration)
3. pathway energies and pathway probabilities are calculated

The pathway energy is:

$$E_{\mathrm{path}} = -\sum_{i \in \mathrm{path}} h_i - \sum_{(i,j) \in \mathrm{path}} J_{ij}$$

Two energy definitions are supported:

- **HJ** — single-site fields + pairwise couplings
- **J_ONLY** — pairwise couplings only

Both length-normalized versions (`HJ_MEAN` and `J_ONLY_MEAN`) are also available. Depending on the selected energy mode, only the $J$ term or both $h$ and $J$ contributions may be included.

---

## 7. Path Probability

Once every pathway's communication energy is known, each pathway is assigned a Boltzmann probability within its MD window:

$$P_{\mathrm{path}}(p) = \frac{e^{-\beta E_{\mathrm{path}}(p)}}{\sum_{p'} e^{-\beta E_{\mathrm{path}}(p')}}$$

This probability is used to compare communication pathways against one another: pathways with lower communication energy are assigned higher probability, and only the most probable pathways are retained for downstream analyses.

---

## Occupancy Analysis

Each communication pathway is represented by a binary activity trajectory:

$$A_p(w) = \begin{cases} 1, & \text{path is active} \\ 0, & \text{otherwise} \end{cases}$$

Using this binary trajectory, the code computes:

- fraction of paths ever active
- mean run length
- maximum run length
- occupancy fraction
- survival probability

for both highMD and lowMD regions.

---

## Persistence Analysis

Persistence quantifies how long communication pathways remain active.

For lag time $\tau$:

$$S(\tau) = P\left(A(t+\tau) = 1 \mid A(t) = 1\right)$$

This is the probability that a pathway remains active after $\tau$ windows, given that it is active now.

The analysis computes:

- global persistence
- highMD persistence
- lowMD persistence
- ratios between highMD and lowMD persistence

---

## Communication-Energy Spectrum Analysis

For each pathway, the communication energy distribution is analyzed separately within highMD and lowMD regions. The following quantities are computed.

**Mean Path Energy**

$$\langle E_{\mathrm{path}} \rangle$$

**Energy Entropy**

$$S = -\sum_i p_i \ln p_i$$

where $p_i$ denotes the probability of occupying energy bin $i$.

**Participation Ratio**

$$PR = \frac{1}{\sum_i p_i^2}$$

This estimates the effective number of communication-energy states contributing to the ensemble.

**Effective Number of Energy Bins**

The effective number of populated energy bins is calculated from the communication-energy distribution.

**Tail Polarization**

Define:

$$f_{\mathrm{low}} = P(E < E_{10})$$

$$f_{\mathrm{high}} = P(E > E_{90})$$

where $E_{10}$ and $E_{90}$ are the 10th and 90th percentiles of the pathway-energy distribution.

Tail polarization is:

$$TP = f_{\mathrm{low}} + f_{\mathrm{high}}$$

**Additional Metrics**

The code also computes:

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
- CSV summaries containing all calculated metrics
- Robustness summaries
- Pearson and Spearman correlations
- Scatter plots versus experimental efficacy
- Heatmaps
- Robustness bar plots
- Publication-quality figures

---

## Interpretation

This workflow distinguishes two different quantities.

**MD Equilibrium Probability**

The Potts Hamiltonian defines the equilibrium probability of each MD window:

$$P_{\mathrm{MD}} = \frac{e^{-\beta E_{\mathrm{MD}}}}{Z}$$

which identifies thermodynamically favorable and unfavorable conformational regions.

**Communication Properties**

Within these regions, the communication network is analyzed through:

- pathway energy
- occupancy
- persistence
- communication diversity
- energy spectrum
- pathway participation

Thus, the Potts model is **not** used to estimate the global conformational free-energy landscape. Instead, it provides an interaction-space Hamiltonian that enables quantitative analysis of communication pathways within different regions of the equilibrium ensemble.

---


