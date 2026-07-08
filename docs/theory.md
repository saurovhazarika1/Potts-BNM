# Theory and Derivation, Part I

## Bayesian-Regularized Maximum Entropy Potts Model

> **Note on math rendering:** This document uses GitHub's native math support (`$...$` for inline math and `$$...$$` on their own lines for display math). It renders correctly in the GitHub web UI, GitHub-flavored Markdown previews, and most modern static-site generators. If you view this file on a platform without KaTeX/MathJax support, the raw LaTeX will show instead of rendered equations.

## Table of Contents

1. [Discrete Representation of Biomolecular Dynamics](#1-discrete-representation-of-biomolecular-dynamics)
2. [Maximum Entropy Principle](#2-maximum-entropy-principle)
3. [Potts Hamiltonian](#3-potts-hamiltonian)
4. [Bayesian Network Model](#4-bayesian-network-model)
5. [Maximum Likelihood Potts Learning](#5-maximum-likelihood-potts-learning)
6. [KL-Regularized Potts Learning](#6-kl-regularized-potts-learning)
7. [Mixed Target Moment Interpretation](#7-mixed-target-moment-interpretation)
8. [Topological Regularization](#8-topological-regularization)
9. [Full Gradient for Couplings](#9-full-gradient-for-couplings)
10. [Full Gradient for Fields](#10-full-gradient-for-fields)
11. [Hard versus Soft Network Constraints](#11-hard-versus-soft-network-constraints)
12. [Interpretation of the Full Objective](#12-interpretation-of-the-full-objective)
13. [Computational Estimation of the Three Moment Terms](#13-computational-estimation-of-the-three-moment-terms)
14. [Learning Updates](#14-learning-updates)
15. [Special Cases](#15-special-cases)
16. [Summary of Part I](#16-summary-of-part-i)

---

This document develops the first part of the theory behind the Potts+BNM framework. The central idea is to learn a thermodynamically interpretable Potts Hamiltonian from molecular dynamics (MD), while using a Bayesian Network Model (BNM) in two complementary ways:

1. as a **distributional prior** through a KL-divergence term, and
2. as a **topological prior** that penalizes couplings unsupported by the BNM graph.

The full objective is

$$
\mathcal{J}(\theta) = \mathbb{E}_{MD}\big[\log P_\theta(\sigma)\big] - \lambda D_{KL}\big(P_0 \Vert P_\theta\big) - \gamma \sum_{(i,j)\notin E_{BNM}} \Vert J_{ij}\Vert_F^2
$$

where

$$
\theta = \lbrace h_i(a), J_{ij}(a,b) \rbrace
$$

are the Potts parameters.

The three terms have distinct roles:

| Term | Role |
|---|---|
| $\mathbb{E}_{MD}[\log P_\theta(\sigma)]$ | fit the MD ensemble |
| $-\lambda D_{KL}(P_0 \Vert P_\theta)$ | stay close to the BNM distribution |
| $-\gamma \sum_{(i,j)\notin E_{BNM}} \Vert J_{ij}\Vert_F^2$ | respect the BNM graph topology |

---

## 1. Discrete Representation of Biomolecular Dynamics

We represent a biomolecular trajectory as a sequence of discrete configurations

$$
\sigma^{(1)}, \sigma^{(2)}, \ldots, \sigma^{(M)}.
$$

Each configuration is

$$
\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_N),
$$

where $N$ is the number of modeled sites, residues, contacts, or collective variables.

Each variable takes one of $q$ discrete states:

$$
\sigma_i \in \lbrace1, 2, \ldots, q\rbrace.
$$

Examples include:

- binary contact state: $q = 2$
- rotameric state: $q = 3$ or larger
- discretized distance bins
- conformational microstates
- contact-cluster states

For each site $i$, define the one-hot indicator

$$
x_i^a(\sigma) = \mathbf{1}[\sigma_i = a]
$$

that is, $x_i^a(\sigma) = 1$ if $\sigma_i = a$, and $x_i^a(\sigma) = 0$ if $\sigma_i \neq a$.

For a pair of sites $i, j$, define

$$
x_{ij}^{ab}(\sigma) = x_i^a(\sigma) x_j^b(\sigma) = \mathbf{1}[\sigma_i = a, \sigma_j = b].
$$

The empirical one-site and two-site moments from MD are

$$
\hat{p}_i^{MD}(a) = \frac{1}{M} \sum_{m=1}^{M} x_i^a(\sigma^{(m)}),
$$

and

$$
\hat{p}_{ij}^{MD}(a,b) = \frac{1}{M} \sum_{m=1}^{M} x_i^a(\sigma^{(m)}) x_j^b(\sigma^{(m)}).
$$

Equivalently,

$$
\hat{p}_i^{MD}(a) = \big\langle x_i^a \big\rangle_{MD}, \qquad \hat{p}_{ij}^{MD}(a,b) = \big\langle x_i^a x_j^b \big\rangle_{MD}.
$$

These are the sufficient statistics the Potts model will try to reproduce.

---

## 2. Maximum Entropy Principle

We seek a probability distribution $P(\sigma)$ that reproduces selected statistics while making no unnecessary assumptions about higher-order structure.

The Shannon entropy is

$$
S[P] = -\sum_{\sigma} P(\sigma) \log P(\sigma).
$$

The maximum entropy problem is

$$
\max_{P} S[P],
$$

subject to normalization,

$$
\sum_{\sigma} P(\sigma) = 1,
$$

and moment constraints,

$$
\sum_\sigma P(\sigma) x_i^a(\sigma) = p_i(a), \qquad \sum_\sigma P(\sigma) x_i^a(\sigma) x_j^b(\sigma) = p_{ij}(a,b).
$$

We form the Lagrangian

$$
\mathcal{L}[P] = -\sum_\sigma P(\sigma)\log P(\sigma) - \alpha\Big[\sum_\sigma P(\sigma) - 1\Big] - \sum_{i,a} \eta_i(a)\Big[\sum_\sigma P(\sigma) x_i^a(\sigma) - p_i(a)\Big] - \sum_{i \lt j}\sum_{a,b} \eta_{ij}(a,b)\Big[\sum_\sigma P(\sigma) x_i^a(\sigma) x_j^b(\sigma) - p_{ij}(a,b)\Big].
$$

Take the functional derivative with respect to $P(\sigma)$:

$$
\frac{\partial \mathcal{L}}{\partial P(\sigma)} = -\log P(\sigma) - 1 - \alpha - \sum_{i,a} \eta_i(a) x_i^a(\sigma) - \sum_{i \lt j}\sum_{a,b} \eta_{ij}(a,b) x_i^a(\sigma) x_j^b(\sigma).
$$

At the optimum,

$$
\frac{\partial \mathcal{L}}{\partial P(\sigma)} = 0.
$$

Therefore,

$$
\log P(\sigma) = -1 - \alpha - \sum_{i,a} \eta_i(a) x_i^a(\sigma) - \sum_{i \lt j}\sum_{a,b} \eta_{ij}(a,b) x_i^a(\sigma) x_j^b(\sigma).
$$

Exponentiating,

$$
P(\sigma) = \exp[-1-\alpha] \exp\left[-\sum_{i,a} \eta_i(a) x_i^a(\sigma) - \sum_{i \lt j}\sum_{a,b} \eta_{ij}(a,b) x_i^a(\sigma) x_j^b(\sigma)\right].
$$

Define the partition function $Z = \exp[1+\alpha]$, so

$$
P(\sigma) = \frac{1}{Z} \exp\left[-\sum_{i,a} \eta_i(a) x_i^a(\sigma) - \sum_{i \lt j}\sum_{a,b} \eta_{ij}(a,b) x_i^a(\sigma) x_j^b(\sigma)\right].
$$

This is an exponential-family distribution.

By defining $h_i(a) = -\eta_i(a)$ and $J_{ij}(a,b) = -\eta_{ij}(a,b)$, we obtain

$$
P_\theta(\sigma) = \frac{1}{Z_\theta} \exp\left[\sum_i h_i(\sigma_i) + \sum_{i \lt j} J_{ij}(\sigma_i,\sigma_j)\right].
$$

Equivalently, defining the Hamiltonian

$$
H_\theta(\sigma) = -\sum_i h_i(\sigma_i) - \sum_{i \lt j} J_{ij}(\sigma_i,\sigma_j),
$$

we get the Boltzmann form

$$
P_\theta(\sigma) = \frac{1}{Z_\theta} \exp[-H_\theta(\sigma)],
$$

or, with explicit inverse temperature $\beta$,

$$
P_\theta(\sigma) = \frac{1}{Z_\theta} \exp[-\beta H_\theta(\sigma)].
$$

In the rest of this document we absorb $\beta$ into the parameters and use

$$
P_\theta(\sigma) = \frac{1}{Z_\theta} e^{-H_\theta(\sigma)}.
$$

---

## 3. Potts Hamiltonian

The pairwise maximum entropy Hamiltonian is

$$
H_\theta(\sigma) = -\sum_i h_i(\sigma_i) - \sum_{i \lt j} J_{ij}(\sigma_i,\sigma_j)
$$

where $h_i(a)$ is the local field for state $a$ at site $i$, and $J_{ij}(a,b)$ is the coupling between state $a$ at site $i$ and state $b$ at site $j$.

The partition function is

$$
Z_\theta = \sum_{\sigma} e^{-H_\theta(\sigma)}.
$$

The model expectation of an observable $A(\sigma)$ is

$$
\langle A \rangle_{P_\theta} = \sum_\sigma A(\sigma) P_\theta(\sigma).
$$

The model moments are

$$
p_i^\theta(a) = \langle x_i^a \rangle_{P_\theta}, \qquad p_{ij}^\theta(a,b) = \langle x_i^a x_j^b \rangle_{P_\theta}.
$$

The goal of inverse Potts learning is to choose $\theta$ so that

$$
p_i^\theta(a) \approx \hat{p}_i^{MD}(a), \qquad p_{ij}^\theta(a,b) \approx \hat{p}_{ij}^{MD}(a,b).
$$

---

## 4. Bayesian Network Model

A Bayesian Network Model defines a directed graph over the same variables,

$$
G_{BNM} = (V, E_{BNM}), \qquad V = \lbrace1, \ldots, N\rbrace.
$$

The parents of node $i$ are denoted $Pa(i)$.

The BNM factorizes the joint distribution as

$$
P_0(\sigma) = \prod_{i=1}^{N} P_0\big(\sigma_i \mid \sigma_{Pa(i)}\big)
$$

where each local conditional probability table is estimated from discretized MD data.

The BNM therefore defines a probability distribution $P_0(\sigma)$ over molecular configurations.

However, $P_0$ is not written as a thermodynamic Hamiltonian. It is a graphical probabilistic model. Its role here is to provide prior information about:

1. which configurations are probable,
2. which variables are conditionally dependent,
3. which interactions are structurally supported.

Thus the BNM contributes two priors: the distribution $P_0(\sigma)$ (**distributional prior**) and the edge set $E_{BNM}$ (**topological prior**).

---

## 5. Maximum Likelihood Potts Learning

Given MD samples $\lbrace\sigma^{(m)}\rbrace_{m=1}^{M}$, the standard maximum likelihood objective is

$$
\mathcal{L}(\theta) = \frac{1}{M} \sum_{m=1}^{M} \log P_\theta(\sigma^{(m)}).
$$

Since $\log P_\theta(\sigma) = -H_\theta(\sigma) - \log Z_\theta$, we have

$$
\mathcal{L}(\theta) = -\big\langle H_\theta(\sigma) \big\rangle_{MD} - \log Z_\theta.
$$

Now substitute the Hamiltonian:

$$
-H_\theta(\sigma) = \sum_i h_i(\sigma_i) + \sum_{i \lt j} J_{ij}(\sigma_i,\sigma_j).
$$

Therefore,

$$
\mathcal{L}(\theta) = \sum_{i,a} h_i(a) \big\langle x_i^a \big\rangle_{MD} + \sum_{i \lt j}\sum_{a,b} J_{ij}(a,b) \big\langle x_i^a x_j^b \big\rangle_{MD} - \log Z_\theta.
$$

Now differentiate with respect to a field $h_i(a)$:

$$
\frac{\partial \mathcal{L}}{\partial h_i(a)} = \big\langle x_i^a \big\rangle_{MD} - \frac{\partial \log Z_\theta}{\partial h_i(a)}.
$$

We compute

$$
\frac{\partial Z_\theta}{\partial h_i(a)} = \frac{\partial}{\partial h_i(a)} \sum_\sigma \exp[-H_\theta(\sigma)].
$$

Since $\dfrac{\partial[-H_\theta(\sigma)]}{\partial h_i(a)} = x_i^a(\sigma)$, we get

$$
\frac{\partial Z_\theta}{\partial h_i(a)} = \sum_\sigma x_i^a(\sigma) e^{-H_\theta(\sigma)}.
$$

Therefore,

$$
\frac{\partial \log Z_\theta}{\partial h_i(a)} = \frac{1}{Z_\theta} \sum_\sigma x_i^a(\sigma) e^{-H_\theta(\sigma)} = \big\langle x_i^a \big\rangle_{P_\theta}.
$$

Thus,

$$
\frac{\partial \mathcal{L}}{\partial h_i(a)} = \big\langle x_i^a \big\rangle_{MD} - \big\langle x_i^a \big\rangle_{P_\theta}.
$$

Similarly,

$$
\frac{\partial \mathcal{L}}{\partial J_{ij}(a,b)} = \big\langle x_i^a x_j^b \big\rangle_{MD} - \big\langle x_i^a x_j^b \big\rangle_{P_\theta}.
$$

This is the classical Boltzmann-machine learning rule:

$$
\nabla_\theta \mathcal{L} = \langle f \rangle_{MD} - \langle f \rangle_{P_\theta}.
$$

The model is trained by reducing the difference between empirical MD moments and model-generated moments.

---

## 6. KL-Regularized Potts Learning

The BNM distribution is now incorporated by adding $-\lambda D_{KL}(P_0 \Vert P_\theta)$ to the objective.

The KL divergence is

$$
D_{KL}(P_0 \Vert P_\theta) = \sum_\sigma P_0(\sigma) \log \frac{P_0(\sigma)}{P_\theta(\sigma)}.
$$

Equivalently,

$$
D_{KL}(P_0 \Vert P_\theta) = \sum_\sigma P_0(\sigma) \log P_0(\sigma) - \sum_\sigma P_0(\sigma) \log P_\theta(\sigma).
$$

In expectation notation,

$$
D_{KL}(P_0 \Vert P_\theta) = \mathbb{E}_{P_0}[\log P_0] - \mathbb{E}_{P_0}[\log P_\theta].
$$

The first term does not depend on $\theta$, so

$$
\nabla_\theta D_{KL}(P_0 \Vert P_\theta) = -\nabla_\theta \mathbb{E}_{P_0}[\log P_\theta].
$$

Now $\log P_\theta(\sigma) = -H_\theta(\sigma) - \log Z_\theta$. Therefore,

$$
\mathbb{E}_{P_0}[\log P_\theta] = -\mathbb{E}_{P_0}[H_\theta] - \log Z_\theta.
$$

Using the same derivative identities as above,

$$
\nabla_\theta \mathbb{E}_{P_0}[\log P_\theta] = \langle f \rangle_{P_0} - \langle f \rangle_{P_\theta}.
$$

Therefore,

$$
\nabla_\theta \big[-\lambda D_{KL}(P_0 \Vert P_\theta)\big] = \lambda \big( \langle f \rangle_{P_0} - \langle f \rangle_{P_\theta} \big).
$$

Adding the MD likelihood gradient,

$$
\nabla_\theta \mathcal{J} = \big( \langle f \rangle_{MD} - \langle f \rangle_{P_\theta} \big) + \lambda \big( \langle f \rangle_{P_0} - \langle f \rangle_{P_\theta} \big).
$$

Thus,

$$
\nabla_\theta \mathcal{J} = \langle f \rangle_{MD} + \lambda \langle f \rangle_{P_0} - (1+\lambda) \langle f \rangle_{P_\theta}.
$$

This is the key result.

---

## 7. Mixed Target Moment Interpretation

The gradient can be rewritten as

$$
\nabla_\theta \mathcal{J} = (1+\lambda) \left[ \frac{\langle f \rangle_{MD} + \lambda \langle f \rangle_{P_0}}{1+\lambda} - \langle f \rangle_{P_\theta} \right].
$$

Define the mixed target moment

$$
\langle f \rangle_{target} = \frac{\langle f \rangle_{MD} + \lambda \langle f \rangle_{P_0}}{1+\lambda}.
$$

Then

$$
\nabla_\theta \mathcal{J} = (1+\lambda) \big[ \langle f \rangle_{target} - \langle f \rangle_{P_\theta} \big].
$$

This shows that the KL-regularized problem remains a moment-matching problem.

The only change is that the target moments are no longer pure MD moments. They are a convex combination of MD moments and BNM moments.

For $\lambda = 0$, $\langle f \rangle_{target} = \langle f \rangle_{MD}$.

For $\lambda \to \infty$, $\langle f \rangle_{target} \to \langle f \rangle_{P_0}$.

Thus $\lambda$ interpolates between MD-driven learning and BNM-driven learning.

---

## 8. Topological Regularization

The KL term constrains the learned distribution. It does not directly constrain which Potts couplings are allowed.

Therefore we add a separate graph-topology penalty.

Let $E_{BNM}$ be the undirected version of the BNM edge set. That is, if either $i \to j$ or $j \to i$ appears in the BNM, we treat the pair $(i,j)$ as supported.

For unsupported edges, $(i,j) \notin E_{BNM}$, we penalize the coupling matrix $J_{ij}$. The penalty is

$$
R_{graph}(\theta) = \gamma \sum_{(i,j)\notin E_{BNM}} \Vert J_{ij}\Vert_F^2.
$$

The Frobenius norm is

$$
\Vert J_{ij}\Vert_F^2 = \sum_{a=1}^{q} \sum_{b=1}^{q} J_{ij}(a,b)^2.
$$

The derivative is

$$
\frac{\partial R_{graph}}{\partial J_{ij}(a,b)} = 2\gamma J_{ij}(a,b), \qquad (i,j)\notin E_{BNM}.
$$

For supported BNM edges, $(i,j) \in E_{BNM}$, there is no graph penalty:

$$
\frac{\partial R_{graph}}{\partial J_{ij}(a,b)} = 0.
$$

Because the objective subtracts the penalty, $\mathcal{J} = \cdots - R_{graph}$, the gradient contribution is $-2\gamma J_{ij}(a,b)$ for non-BNM edges.

---

## 9. Full Gradient for Couplings

For each coupling element $J_{ij}(a,b)$, define

$$
f_{ij}^{ab}(\sigma) = x_i^a(\sigma) x_j^b(\sigma).
$$

The full gradient is

$$
\frac{\partial \mathcal{J}}{\partial J_{ij}(a,b)} = \langle f_{ij}^{ab} \rangle_{MD} + \lambda \langle f_{ij}^{ab} \rangle_{P_0} - (1+\lambda) \langle f_{ij}^{ab} \rangle_{P_\theta} - 2\gamma J_{ij}(a,b) \mathbf{1}[(i,j)\notin E_{BNM}].
$$

Equivalently, at the matrix level,

$$
\nabla_{J_{ij}} \mathcal{J} = \langle f_{ij} \rangle_{MD} + \lambda \langle f_{ij} \rangle_{P_0} - (1+\lambda) \langle f_{ij} \rangle_{P_\theta} - 2\gamma J_{ij} \mathbf{1}[(i,j)\notin E_{BNM}].
$$

For edges **inside** the BNM, $(i,j) \in E_{BNM}$, the update becomes

$$
\nabla_{J_{ij}} \mathcal{J} = \langle f_{ij} \rangle_{MD} + \lambda \langle f_{ij} \rangle_{P_0} - (1+\lambda) \langle f_{ij} \rangle_{P_\theta}.
$$

For edges **outside** the BNM, $(i,j) \notin E_{BNM}$, the update is

$$
\nabla_{J_{ij}} \mathcal{J} = \langle f_{ij} \rangle_{MD} + \lambda \langle f_{ij} \rangle_{P_0} - (1+\lambda) \langle f_{ij} \rangle_{P_\theta} - 2\gamma J_{ij}.
$$

Thus unsupported couplings are allowed but penalized.

---

## 10. Full Gradient for Fields

If the graph penalty only applies to pairwise couplings, the field gradient is

$$
\frac{\partial \mathcal{J}}{\partial h_i(a)} = \langle x_i^a \rangle_{MD} + \lambda \langle x_i^a \rangle_{P_0} - (1+\lambda) \langle x_i^a \rangle_{P_\theta},
$$

or

$$
\nabla_{h_i} \mathcal{J} = \langle f_i \rangle_{MD} + \lambda \langle f_i \rangle_{P_0} - (1+\lambda) \langle f_i \rangle_{P_\theta},
$$

where $f_i^a(\sigma) = x_i^a(\sigma)$.

---

## 11. Hard versus Soft Network Constraints

The soft penalty is

$$
\gamma \sum_{(i,j)\notin E_{BNM}} \Vert J_{ij}\Vert_F^2.
$$

This allows unsupported couplings if strongly supported by MD.

A hard network constraint instead imposes

$$
J_{ij} = 0, \qquad (i,j) \notin E_{BNM}.
$$

The hard constraint gives a sparse Potts model with couplings only on BNM-supported edges.

The soft constraint is usually preferable because it allows the MD data to override the BNM when necessary.

A more general weighted penalty is

$$
R_{graph} = \gamma \sum_{i \lt j} w_{ij} \Vert J_{ij}\Vert_F^2, \qquad w_{ij} = 1 - A_{ij},
$$

where $A_{ij}$ is an edge confidence or normalized BNM adjacency score. Then

$$
\frac{\partial R_{graph}}{\partial J_{ij}} = 2\gamma w_{ij} J_{ij}.
$$

The full coupling gradient becomes

$$
\nabla_{J_{ij}} \mathcal{J} = \langle f_{ij} \rangle_{MD} + \lambda \langle f_{ij} \rangle_{P_0} - (1+\lambda) \langle f_{ij} \rangle_{P_\theta} - 2\gamma w_{ij} J_{ij}.
$$

This formulation permits graded confidence in BNM edges.

---

## 12. Interpretation of the Full Objective

The full objective contains three distinct pressures:

$$
\mathcal{J}(\theta) = \mathbb{E}_{MD}[\log P_\theta] - \lambda D_{KL}(P_0 \Vert P_\theta) - \gamma \sum_{(i,j)\notin E_{BNM}} \Vert J_{ij}\Vert_F^2.
$$

| Term | Asks that... |
|---|---|
| MD likelihood | $P_\theta \approx P_{MD}$ |
| KL term | $P_\theta \approx P_0$ |
| Topology penalty | $J_{ij} \approx 0$ when $(i,j) \notin E_{BNM}$ |

Thus the learned Potts model is not merely fit to data. It is regularized by both the BNM distribution and the BNM graph.

---

## 13. Computational Estimation of the Three Moment Terms

The learning rule requires three expectations: $\langle f \rangle_{MD}$, $\langle f \rangle_{P_0}$, and $\langle f \rangle_{P_\theta}$.

The MD moments are estimated directly from trajectory frames:

$$
\langle f \rangle_{MD} \approx \frac{1}{M} \sum_{m=1}^{M} f(\sigma^{(m)}).
$$

The BNM moments are estimated by sampling configurations from the Bayesian Network, $\sigma^{(s)} \sim P_0$, then computing

$$
\langle f \rangle_{P_0} \approx \frac{1}{S} \sum_{s=1}^{S} f(\sigma^{(s)}).
$$

The Potts moments are estimated by Monte Carlo sampling from

$$
P_\theta(\sigma) = \frac{1}{Z_\theta} e^{-H_\theta(\sigma)}.
$$

Because exact summation over all states scales as $q^N$, Monte Carlo is required for realistic systems.

---

## 14. Learning Updates

A generic gradient ascent update is

$$
h_i(a)^{t+1} = h_i(a)^t + \eta_h \frac{\partial \mathcal{J}}{\partial h_i(a)}.
$$

For couplings,

$$
J_{ij}(a,b)^{t+1} = J_{ij}(a,b)^t + \eta_J \frac{\partial \mathcal{J}}{\partial J_{ij}(a,b)}.
$$

Substituting the gradient,

$$
J_{ij}(a,b)^{t+1} = J_{ij}(a,b)^t + \eta_J \Big[ \langle f_{ij}^{ab} \rangle_{MD} + \lambda \langle f_{ij}^{ab} \rangle_{P_0} - (1+\lambda) \langle f_{ij}^{ab} \rangle_{P_\theta} - 2\gamma J_{ij}(a,b) \mathbf{1}[(i,j)\notin E_{BNM}] \Big].
$$

This is the practical update rule.

---

## 15. Special Cases

### 15.1 Standard Maximum-Likelihood Potts Model

Set $\lambda = 0$, $\gamma = 0$. Then

$$
\nabla \mathcal{J} = \langle f \rangle_{MD} - \langle f \rangle_{P_\theta}.
$$

This recovers standard inverse Potts learning.

### 15.2 KL-Regularized Potts Model

Set $\gamma = 0$. Then

$$
\nabla \mathcal{J} = \langle f \rangle_{MD} + \lambda \langle f \rangle_{P_0} - (1+\lambda) \langle f \rangle_{P_\theta}.
$$

This fits the mixed MD-BNM moment target.

### 15.3 Topology-Regularized Potts Model

Set $\lambda = 0$. Then

$$
\nabla_{J_{ij}} \mathcal{J} = \langle f_{ij} \rangle_{MD} - \langle f_{ij} \rangle_{P_\theta} - 2\gamma J_{ij} \mathbf{1}[(i,j)\notin E_{BNM}].
$$

This learns from MD but suppresses unsupported edges.

### 15.4 Full Potts+BNM Model

For $\lambda > 0$, $\gamma > 0$, the model combines:

1. MD likelihood,
2. BNM distributional prior,
3. BNM graph-topology prior.

---

## 16. Summary of Part I

The first part of the theory establishes the following:

1. The Potts Hamiltonian arises naturally from maximum entropy constraints on one- and two-site statistics.
2. Standard inverse Potts learning is equivalent to moment matching.
3. A Bayesian Network defines a prior distribution $P_0$ and a prior graph $E_{BNM}$.
4. KL regularization modifies the target moments to a convex combination of MD and BNM moments.
5. Graph regularization penalizes Potts couplings unsupported by the BNM topology.
6. The final learning rule preserves the structure of Boltzmann-machine learning but replaces pure MD targets with Bayesian-regularized targets.

This provides the theoretical foundation for the full Potts+BNM framework.
