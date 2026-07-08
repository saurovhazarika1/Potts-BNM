# Theory and Derivation, Part I

## Bayesian-Regularized Maximum Entropy Potts Model

This document develops the first part of the theory behind the Potts+BNM framework. The central idea is to learn a thermodynamically interpretable Potts Hamiltonian from molecular dynamics (MD), while using a Bayesian Network Model (BNM) in two complementary ways:

1. as a **distributional prior** through a KL-divergence term, and  
2. as a **topological prior** that penalizes couplings unsupported by the BNM graph.

The full objective is

$$
\boxed{
\mathcal J(\theta)
=
\mathbb E_{MD}
\left[
\log P_\theta(\sigma)
\right]
-
\lambda
D_{KL}
\left(
P_0 \| P_\theta
\right)
-
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2
}
$$

where

$$
\theta=\{h_i(a),J_{ij}(a,b)\}
$$

are the Potts parameters.

The three terms have distinct roles:

$$
\underbrace{
\mathbb E_{MD}[\log P_\theta(\sigma)]
}_{\text{fit MD ensemble}}
-
\underbrace{
\lambda D_{KL}(P_0\|P_\theta)
}_{\text{stay close to BNM distribution}}
-
\underbrace{
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2
}_{\text{respect BNM graph topology}}
$$

---

## 1. Discrete Representation of Biomolecular Dynamics

We represent a biomolecular trajectory as a sequence of discrete configurations

$$
\sigma^{(1)},\sigma^{(2)},\ldots,\sigma^{(M)}.
$$

Each configuration is

$$
\sigma =
(\sigma_1,\sigma_2,\ldots,\sigma_N),
$$

where \(N\) is the number of modeled sites, residues, contacts, or collective variables.

Each variable takes one of \(q\) discrete states:

$$
\sigma_i\in\{1,2,\ldots,q\}.
$$

Examples include:

- binary contact state: \(q=2\),
- rotameric state: \(q=3\) or larger,
- discretized distance bins,
- conformational microstates,
- contact-cluster states.

For each site \(i\), define the one-hot indicator

$$
x_i^a(\sigma)
=
\mathbf 1[\sigma_i=a],
$$

where

$$
x_i^a(\sigma)=
\begin{cases}
1, & \sigma_i=a,\\
0, & \sigma_i\neq a.
\end{cases}
$$

For a pair of sites \(i,j\), define

$$
x_{ij}^{ab}(\sigma)
=
x_i^a(\sigma)x_j^b(\sigma)
=
\mathbf 1[\sigma_i=a,\sigma_j=b].
$$

The empirical one-site and two-site moments from MD are

$$
\hat p_i^{MD}(a)
=
\frac1M
\sum_{m=1}^M
x_i^a(\sigma^{(m)}),
$$

and

$$
\hat p_{ij}^{MD}(a,b)
=
\frac1M
\sum_{m=1}^M
x_i^a(\sigma^{(m)})
x_j^b(\sigma^{(m)}).
$$

Equivalently,

$$
\hat p_i^{MD}(a)
=
\left\langle
x_i^a
\right\rangle_{MD},
$$

$$
\hat p_{ij}^{MD}(a,b)
=
\left\langle
x_i^a x_j^b
\right\rangle_{MD}.
$$

These are the sufficient statistics the Potts model will try to reproduce.

---

## 2. Maximum Entropy Principle

We seek a probability distribution \(P(\sigma)\) that reproduces selected statistics while making no unnecessary assumptions about higher-order structure.

The Shannon entropy is

$$
S[P]
=
-\sum_{\sigma}
P(\sigma)\log P(\sigma).
$$

The maximum entropy problem is

$$
\max_{P}
S[P],
$$

subject to normalization,

$$
\sum_{\sigma}P(\sigma)=1,
$$

and moment constraints,

$$
\sum_\sigma
P(\sigma)
x_i^a(\sigma)
=
p_i(a),
$$

$$
\sum_\sigma
P(\sigma)
x_i^a(\sigma)x_j^b(\sigma)
=
p_{ij}(a,b).
$$

We form the Lagrangian

$$
\begin{aligned}
\mathcal L[P]
=&
-\sum_\sigma P(\sigma)\log P(\sigma)
-\alpha
\left[
\sum_\sigma P(\sigma)-1
\right]\\
&-\sum_{i,a}
\eta_i(a)
\left[
\sum_\sigma P(\sigma)x_i^a(\sigma)-p_i(a)
\right]\\
&-\sum_{i<j}\sum_{a,b}
\eta_{ij}(a,b)
\left[
\sum_\sigma P(\sigma)x_i^a(\sigma)x_j^b(\sigma)
-
p_{ij}(a,b)
\right].
\end{aligned}
$$

Take the functional derivative with respect to \(P(\sigma)\):

$$
\frac{\partial \mathcal L}{\partial P(\sigma)}
=
-\log P(\sigma)-1
-\alpha
-\sum_{i,a}
\eta_i(a)x_i^a(\sigma)
-\sum_{i<j}\sum_{a,b}
\eta_{ij}(a,b)x_i^a(\sigma)x_j^b(\sigma).
$$

At the optimum,

$$
\frac{\partial \mathcal L}{\partial P(\sigma)}=0.
$$

Therefore,

$$
\log P(\sigma)
=
-1-\alpha
-\sum_{i,a}
\eta_i(a)x_i^a(\sigma)
-\sum_{i<j}\sum_{a,b}
\eta_{ij}(a,b)x_i^a(\sigma)x_j^b(\sigma).
$$

Exponentiating,

$$
P(\sigma)
=
\exp[-1-\alpha]
\exp
\left[
-\sum_{i,a}
\eta_i(a)x_i^a(\sigma)
-\sum_{i<j}\sum_{a,b}
\eta_{ij}(a,b)x_i^a(\sigma)x_j^b(\sigma)
\right].
$$

Define the partition function

$$
Z
=
\exp[1+\alpha],
$$

so

$$
\boxed{
P(\sigma)
=
\frac1Z
\exp
\left[
-\sum_{i,a}
\eta_i(a)x_i^a(\sigma)
-\sum_{i<j}\sum_{a,b}
\eta_{ij}(a,b)x_i^a(\sigma)x_j^b(\sigma)
\right]
}
$$

This is an exponential-family distribution.

By defining

$$
h_i(a)=-\eta_i(a),
$$

$$
J_{ij}(a,b)=-\eta_{ij}(a,b),
$$

we obtain

$$
P_\theta(\sigma)
=
\frac1{Z_\theta}
\exp
\left[
\sum_i h_i(\sigma_i)
+
\sum_{i<j}
J_{ij}(\sigma_i,\sigma_j)
\right].
$$

Equivalently, defining the Hamiltonian

$$
H_\theta(\sigma)
=
-
\sum_i h_i(\sigma_i)
-
\sum_{i<j}
J_{ij}(\sigma_i,\sigma_j),
$$

we get the Boltzmann form

$$
\boxed{
P_\theta(\sigma)
=
\frac1{Z_\theta}
\exp[-H_\theta(\sigma)]
}
$$

or, with explicit inverse temperature \(\beta\),

$$
P_\theta(\sigma)
=
\frac1{Z_\theta}
\exp[-\beta H_\theta(\sigma)].
$$

In the rest of this document we absorb \(\beta\) into the parameters and use

$$
P_\theta(\sigma)
=
\frac1{Z_\theta}
e^{-H_\theta(\sigma)}.
$$

---

## 3. Potts Hamiltonian

The pairwise maximum entropy Hamiltonian is

$$
\boxed{
H_\theta(\sigma)
=
-
\sum_i h_i(\sigma_i)
-
\sum_{i<j}
J_{ij}(\sigma_i,\sigma_j)
}
$$

where \(h_i(a)\) is the local field for state \(a\) at site \(i\), and \(J_{ij}(a,b)\) is the coupling between state \(a\) at site \(i\) and state \(b\) at site \(j\).

The partition function is

$$
Z_\theta
=
\sum_{\sigma}
e^{-H_\theta(\sigma)}.
$$

The model expectation of an observable \(A(\sigma)\) is

$$
\langle A\rangle_{P_\theta}
=
\sum_\sigma
A(\sigma)P_\theta(\sigma).
$$

The model moments are

$$
p_i^\theta(a)
=
\langle x_i^a\rangle_{P_\theta},
$$

and

$$
p_{ij}^\theta(a,b)
=
\langle x_i^a x_j^b\rangle_{P_\theta}.
$$

The goal of inverse Potts learning is to choose \(\theta\) so that

$$
p_i^\theta(a)\approx \hat p_i^{MD}(a),
$$

$$
p_{ij}^\theta(a,b)\approx \hat p_{ij}^{MD}(a,b).
$$

---

## 4. Bayesian Network Model

A Bayesian Network Model defines a directed graph over the same variables,

$$
G_{BNM}=(V,E_{BNM}),
$$

where

$$
V=\{1,\ldots,N\}.
$$

The parents of node \(i\) are denoted

$$
Pa(i).
$$

The BNM factorizes the joint distribution as

$$
\boxed{
P_0(\sigma)
=
\prod_{i=1}^N
P_0
\left(
\sigma_i
\mid
\sigma_{Pa(i)}
\right)
}
$$

where each local conditional probability table is estimated from discretized MD data.

The BNM therefore defines a probability distribution \(P_0(\sigma)\) over molecular configurations.

However, \(P_0\) is not written as a thermodynamic Hamiltonian. It is a graphical probabilistic model. Its role here is to provide prior information about:

1. which configurations are probable,
2. which variables are conditionally dependent,
3. which interactions are structurally supported.

Thus the BNM contributes two priors:

$$
P_0(\sigma)
\quad \text{distributional prior}
$$

and

$$
E_{BNM}
\quad \text{topological prior}.
$$

---

## 5. Maximum Likelihood Potts Learning

Given MD samples \(\{\sigma^{(m)}\}_{m=1}^M\), the standard maximum likelihood objective is

$$
\mathcal L(\theta)
=
\frac1M
\sum_{m=1}^M
\log P_\theta(\sigma^{(m)}).
$$

Since

$$
\log P_\theta(\sigma)
=
-H_\theta(\sigma)
-
\log Z_\theta,
$$

we have

$$
\mathcal L(\theta)
=
-\left\langle
H_\theta(\sigma)
\right\rangle_{MD}
-
\log Z_\theta.
$$

Now substitute the Hamiltonian:

$$
-H_\theta(\sigma)
=
\sum_i h_i(\sigma_i)
+
\sum_{i<j}
J_{ij}(\sigma_i,\sigma_j).
$$

Therefore,

$$
\mathcal L(\theta)
=
\sum_{i,a}
h_i(a)
\left\langle
x_i^a
\right\rangle_{MD}
+
\sum_{i<j}\sum_{a,b}
J_{ij}(a,b)
\left\langle
x_i^a x_j^b
\right\rangle_{MD}
-
\log Z_\theta.
$$

Now differentiate with respect to a field \(h_i(a)\):

$$
\frac{\partial \mathcal L}{\partial h_i(a)}
=
\left\langle
x_i^a
\right\rangle_{MD}
-
\frac{\partial \log Z_\theta}{\partial h_i(a)}.
$$

We compute

$$
\frac{\partial Z_\theta}{\partial h_i(a)}
=
\frac{\partial}{\partial h_i(a)}
\sum_\sigma
\exp[-H_\theta(\sigma)].
$$

Since

$$
\frac{\partial[-H_\theta(\sigma)]}{\partial h_i(a)}
=
x_i^a(\sigma),
$$

we get

$$
\frac{\partial Z_\theta}{\partial h_i(a)}
=
\sum_\sigma
x_i^a(\sigma)
e^{-H_\theta(\sigma)}.
$$

Therefore,

$$
\frac{\partial \log Z_\theta}{\partial h_i(a)}
=
\frac1{Z_\theta}
\sum_\sigma
x_i^a(\sigma)
e^{-H_\theta(\sigma)}
=
\left\langle
x_i^a
\right\rangle_{P_\theta}.
$$

Thus,

$$
\boxed{
\frac{\partial \mathcal L}{\partial h_i(a)}
=
\left\langle
x_i^a
\right\rangle_{MD}
-
\left\langle
x_i^a
\right\rangle_{P_\theta}
}
$$

Similarly,

$$
\boxed{
\frac{\partial \mathcal L}{\partial J_{ij}(a,b)}
=
\left\langle
x_i^a x_j^b
\right\rangle_{MD}
-
\left\langle
x_i^a x_j^b
\right\rangle_{P_\theta}
}
$$

This is the classical Boltzmann-machine learning rule:

$$
\boxed{
\nabla_\theta \mathcal L
=
\langle f\rangle_{MD}
-
\langle f\rangle_{P_\theta}
}
$$

The model is trained by reducing the difference between empirical MD moments and model-generated moments.

---

## 6. KL-Regularized Potts Learning

The BNM distribution is now incorporated by adding

$$
-\lambda D_{KL}(P_0\|P_\theta)
$$

to the objective.

The KL divergence is

$$
D_{KL}(P_0\|P_\theta)
=
\sum_\sigma
P_0(\sigma)
\log
\frac{P_0(\sigma)}{P_\theta(\sigma)}.
$$

Equivalently,

$$
D_{KL}(P_0\|P_\theta)
=
\sum_\sigma
P_0(\sigma)\log P_0(\sigma)
-
\sum_\sigma
P_0(\sigma)\log P_\theta(\sigma).
$$

In expectation notation,

$$
D_{KL}(P_0\|P_\theta)
=
\mathbb E_{P_0}[\log P_0]
-
\mathbb E_{P_0}[\log P_\theta].
$$

The first term does not depend on \(\theta\), so

$$
\nabla_\theta D_{KL}(P_0\|P_\theta)
=
-
\nabla_\theta
\mathbb E_{P_0}[\log P_\theta].
$$

Now

$$
\log P_\theta(\sigma)
=
-H_\theta(\sigma)-\log Z_\theta.
$$

Therefore,

$$
\mathbb E_{P_0}[\log P_\theta]
=
-\mathbb E_{P_0}[H_\theta]
-
\log Z_\theta.
$$

Using the same derivative identities as above,

$$
\nabla_\theta
\mathbb E_{P_0}[\log P_\theta]
=
\langle f\rangle_{P_0}
-
\langle f\rangle_{P_\theta}.
$$

Therefore,

$$
\nabla_\theta
\left[
-\lambda D_{KL}(P_0\|P_\theta)
\right]
=
\lambda
\left(
\langle f\rangle_{P_0}
-
\langle f\rangle_{P_\theta}
\right).
$$

Adding the MD likelihood gradient,

$$
\nabla_\theta \mathcal J
=
\left(
\langle f\rangle_{MD}
-
\langle f\rangle_{P_\theta}
\right)
+
\lambda
\left(
\langle f\rangle_{P_0}
-
\langle f\rangle_{P_\theta}
\right).
$$

Thus,

$$
\boxed{
\nabla_\theta \mathcal J
=
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
-
(1+\lambda)
\langle f\rangle_{P_\theta}
}
$$

This is the key result.

---

## 7. Mixed Target Moment Interpretation

The gradient can be rewritten as

$$
\nabla_\theta\mathcal J
=
(1+\lambda)
\left[
\frac{
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
}
{1+\lambda}
-
\langle f\rangle_{P_\theta}
\right].
$$

Define the mixed target moment

$$
\boxed{
\langle f\rangle_{target}
=
\frac{
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
}
{1+\lambda}
}
$$

Then

$$
\boxed{
\nabla_\theta\mathcal J
=
(1+\lambda)
\left[
\langle f\rangle_{target}
-
\langle f\rangle_{P_\theta}
\right]
}
$$

This shows that the KL-regularized problem remains a moment-matching problem.

The only change is that the target moments are no longer pure MD moments. They are a convex combination of MD moments and BNM moments.

For \(\lambda=0\),

$$
\langle f\rangle_{target}
=
\langle f\rangle_{MD}.
$$

For \(\lambda\to\infty\),

$$
\langle f\rangle_{target}
\to
\langle f\rangle_{P_0}.
$$

Thus \(\lambda\) interpolates between MD-driven learning and BNM-driven learning.

---

## 8. Topological Regularization

The KL term constrains the learned distribution. It does not directly constrain which Potts couplings are allowed.

Therefore we add a separate graph-topology penalty.

Let \(E_{BNM}\) be the undirected version of the BNM edge set. That is, if either

$$
i\to j
$$

or

$$
j\to i
$$

appears in the BNM, we treat the pair \((i,j)\) as supported.

For unsupported edges,

$$
(i,j)\notin E_{BNM},
$$

we penalize the coupling matrix \(J_{ij}\).

The penalty is

$$
R_{graph}(\theta)
=
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2.
$$

The Frobenius norm is

$$
\|J_{ij}\|_F^2
=
\sum_{a=1}^q
\sum_{b=1}^q
J_{ij}(a,b)^2.
$$

The derivative is

$$
\frac{\partial R_{graph}}{\partial J_{ij}(a,b)}
=
2\gamma J_{ij}(a,b),
$$

for

$$
(i,j)\notin E_{BNM}.
$$

For supported BNM edges,

$$
(i,j)\in E_{BNM},
$$

there is no graph penalty:

$$
\frac{\partial R_{graph}}{\partial J_{ij}(a,b)}
=
0.
$$

Because the objective subtracts the penalty,

$$
\mathcal J
=
\cdots
-
R_{graph},
$$

the gradient contribution is

$$
-2\gamma J_{ij}(a,b)
$$

for non-BNM edges.

---

## 9. Full Gradient for Couplings

For each coupling element \(J_{ij}(a,b)\), define

$$
f_{ij}^{ab}(\sigma)
=
x_i^a(\sigma)x_j^b(\sigma).
$$

The full gradient is

$$
\boxed{
\frac{\partial\mathcal J}{\partial J_{ij}(a,b)}
=
\left\langle
f_{ij}^{ab}
\right\rangle_{MD}
+
\lambda
\left\langle
f_{ij}^{ab}
\right\rangle_{P_0}
-
(1+\lambda)
\left\langle
f_{ij}^{ab}
\right\rangle_{P_\theta}
-
2\gamma
J_{ij}(a,b)
\mathbf 1[(i,j)\notin E_{BNM}]
}
$$

Equivalently, at the matrix level,

$$
\boxed{
\nabla_{J_{ij}}\mathcal J
=
\langle f_{ij}\rangle_{MD}
+
\lambda
\langle f_{ij}\rangle_{P_0}
-
(1+\lambda)
\langle f_{ij}\rangle_{P_\theta}
-
2\gamma
J_{ij}
\mathbf 1[(i,j)\notin E_{BNM}]
}
$$

For edges inside the BNM,

$$
(i,j)\in E_{BNM},
$$

the update becomes

$$
\boxed{
\nabla_{J_{ij}}\mathcal J
=
\langle f_{ij}\rangle_{MD}
+
\lambda
\langle f_{ij}\rangle_{P_0}
-
(1+\lambda)
\langle f_{ij}\rangle_{P_\theta}
}
$$

For edges outside the BNM,

$$
(i,j)\notin E_{BNM},
$$

the update is

$$
\boxed{
\nabla_{J_{ij}}\mathcal J
=
\langle f_{ij}\rangle_{MD}
+
\lambda
\langle f_{ij}\rangle_{P_0}
-
(1+\lambda)
\langle f_{ij}\rangle_{P_\theta}
-
2\gamma J_{ij}
}
$$

Thus unsupported couplings are allowed but penalized.

---

## 10. Full Gradient for Fields

If the graph penalty only applies to pairwise couplings, the field gradient is

$$
\boxed{
\frac{\partial\mathcal J}{\partial h_i(a)}
=
\left\langle x_i^a\right\rangle_{MD}
+
\lambda
\left\langle x_i^a\right\rangle_{P_0}
-
(1+\lambda)
\left\langle x_i^a\right\rangle_{P_\theta}
}
$$

or

$$
\boxed{
\nabla_{h_i}\mathcal J
=
\langle f_i\rangle_{MD}
+
\lambda
\langle f_i\rangle_{P_0}
-
(1+\lambda)
\langle f_i\rangle_{P_\theta}
}
$$

where

$$
f_i^a(\sigma)=x_i^a(\sigma).
$$

---

## 11. Hard versus Soft Network Constraints

The soft penalty is

$$
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2.
$$

This allows unsupported couplings if strongly supported by MD.

A hard network constraint instead imposes

$$
J_{ij}=0
\qquad
(i,j)\notin E_{BNM}.
$$

The hard constraint gives a sparse Potts model with couplings only on BNM-supported edges.

The soft constraint is usually preferable because it allows the MD data to override the BNM when necessary.

A more general weighted penalty is

$$
R_{graph}
=
\gamma
\sum_{i<j}
w_{ij}
\|J_{ij}\|_F^2,
$$

where

$$
w_{ij}=1-A_{ij},
$$

and \(A_{ij}\) is an edge confidence or normalized BNM adjacency score.

Then

$$
\frac{\partial R_{graph}}{\partial J_{ij}}
=
2\gamma w_{ij}J_{ij}.
$$

The full coupling gradient becomes

$$
\boxed{
\nabla_{J_{ij}}\mathcal J
=
\langle f_{ij}\rangle_{MD}
+
\lambda
\langle f_{ij}\rangle_{P_0}
-
(1+\lambda)
\langle f_{ij}\rangle_{P_\theta}
-
2\gamma w_{ij}J_{ij}
}
$$

This formulation permits graded confidence in BNM edges.

---

## 12. Interpretation of the Full Objective

The full objective contains three distinct pressures:

$$
\mathcal J(\theta)
=
\underbrace{
\mathbb E_{MD}[\log P_\theta]
}_{\text{fit MD}}
-
\underbrace{
\lambda D_{KL}(P_0\|P_\theta)
}_{\text{distribution prior}}
-
\underbrace{
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2
}_{\text{topology prior}}.
$$

The MD likelihood asks:

$$
P_\theta \approx P_{MD}.
$$

The KL term asks:

$$
P_\theta \approx P_0.
$$

The topology penalty asks:

$$
J_{ij}\approx 0
\quad
\text{when}
\quad
(i,j)\notin E_{BNM}.
$$

Thus the learned Potts model is not merely fit to data. It is regularized by both the BNM distribution and the BNM graph.

---

## 13. Computational Estimation of the Three Moment Terms

The learning rule requires three expectations:

$$
\langle f\rangle_{MD},
$$

$$
\langle f\rangle_{P_0},
$$

$$
\langle f\rangle_{P_\theta}.
$$

The MD moments are estimated directly from trajectory frames:

$$
\langle f\rangle_{MD}
\approx
\frac1M
\sum_{m=1}^M
f(\sigma^{(m)}).
$$

The BNM moments are estimated by sampling configurations from the Bayesian Network:

$$
\sigma^{(s)}\sim P_0,
$$

then computing

$$
\langle f\rangle_{P_0}
\approx
\frac1S
\sum_{s=1}^S
f(\sigma^{(s)}).
$$

The Potts moments are estimated by Monte Carlo sampling from

$$
P_\theta(\sigma)
=
\frac1{Z_\theta}
e^{-H_\theta(\sigma)}.
$$

Because exact summation over all states scales as

$$
q^N,
$$

Monte Carlo is required for realistic systems.

---

## 14. Learning Updates

A generic gradient ascent update is

$$
h_i(a)^{t+1}
=
h_i(a)^t
+
\eta_h
\frac{\partial\mathcal J}{\partial h_i(a)}.
$$

For couplings,

$$
J_{ij}(a,b)^{t+1}
=
J_{ij}(a,b)^t
+
\eta_J
\frac{\partial\mathcal J}{\partial J_{ij}(a,b)}.
$$

Substituting the gradient,

$$
\begin{aligned}
J_{ij}(a,b)^{t+1}
=&
J_{ij}(a,b)^t
+
\eta_J
\Big[
\langle f_{ij}^{ab}\rangle_{MD}
+
\lambda
\langle f_{ij}^{ab}\rangle_{P_0}\\
&-
(1+\lambda)
\langle f_{ij}^{ab}\rangle_{P_\theta}
-
2\gamma J_{ij}(a,b)
\mathbf 1[(i,j)\notin E_{BNM}]
\Big].
\end{aligned}
$$

This is the practical update rule.

---

## 15. Special Cases

### 15.1 Standard Maximum-Likelihood Potts Model

Set

$$
\lambda=0,\qquad \gamma=0.
$$

Then

$$
\nabla\mathcal J
=
\langle f\rangle_{MD}
-
\langle f\rangle_{P_\theta}.
$$

This recovers standard inverse Potts learning.

### 15.2 KL-Regularized Potts Model

Set

$$
\gamma=0.
$$

Then

$$
\nabla\mathcal J
=
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
-
(1+\lambda)
\langle f\rangle_{P_\theta}.
$$

This fits the mixed MD-BNM moment target.

### 15.3 Topology-Regularized Potts Model

Set

$$
\lambda=0.
$$

Then

$$
\nabla_{J_{ij}}\mathcal J
=
\langle f_{ij}\rangle_{MD}
-
\langle f_{ij}\rangle_{P_\theta}
-
2\gamma J_{ij}
\mathbf 1[(i,j)\notin E_{BNM}].
$$

This learns from MD but suppresses unsupported edges.

### 15.4 Full Potts+BNM Model

For

$$
\lambda>0,\qquad \gamma>0,
$$

the model combines:

1. MD likelihood,
2. BNM distributional prior,
3. BNM graph-topology prior.

---

## 16. Summary of Part I

The first part of the theory establishes the following:

1. The Potts Hamiltonian arises naturally from maximum entropy constraints on one- and two-site statistics.
2. Standard inverse Potts learning is equivalent to moment matching.
3. A Bayesian Network defines a prior distribution \(P_0\) and a prior graph \(E_{BNM}\).
4. KL regularization modifies the target moments to a convex combination of MD and BNM moments.
5. Graph regularization penalizes Potts couplings unsupported by the BNM topology.
6. The final learning rule preserves the structure of Boltzmann-machine learning but replaces pure MD targets with Bayesian-regularized targets.

This provides the theoretical foundation for the full Potts+BNM framework.
