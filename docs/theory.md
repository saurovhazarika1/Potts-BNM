
# Theory

# Bayesian-Regularized Maximum Entropy Potts Model

> **This document is a theoretical reference for the Potts+BNM framework.**
> It derives the learning objective from first principles and explains how the Bayesian Network prior and graph regularization modify classical inverse Potts learning.

---

# 1. Maximum Entropy Principle

Suppose a biomolecular system can occupy discrete conformational states

\[
\sigma\in\Omega.
\]

Among all probability distributions \(P(\sigma)\), we seek the least-biased distribution consistent with experimentally or computationally measured observables.

The entropy is

\[
S[P]=-\sum_{\sigma}P(\sigma)\log P(\sigma).
\]

Subject to

\[
\sum_\sigma P(\sigma)=1
\]

and

\[
\sum_\sigma P(\sigma)f_k(\sigma)=c_k,\qquad k=1,\ldots,K.
\]

Construct the Lagrangian

\[
\mathcal L=
-S
+\alpha\left(\sum_\sigma P(\sigma)-1\right)
+\sum_k\lambda_k
\left(
\sum_\sigma P(\sigma)f_k(\sigma)-c_k
\right).
\]

Setting

\[
\frac{\partial\mathcal L}{\partial P(\sigma)}=0
\]

gives

\[
-\log P(\sigma)-1+\alpha+\sum_k\lambda_kf_k(\sigma)=0.
\]

Hence

\[
P(\sigma)
=
\exp
\left(
\alpha-1-\sum_k\lambda_kf_k(\sigma)
\right).
\]

Defining

\[
Z=\exp(1-\alpha),
\]

we obtain

\[
\boxed{
P(\sigma)
=
\frac1Z
\exp
\left(
-\sum_k\lambda_kf_k(\sigma)
\right)
}
\]

which is the maximum-entropy distribution.

---

# 2. Potts Hamiltonian

For a q-state Potts model

\[
\sigma_i\in\{1,\ldots,q\},
\]

the sufficient statistics are

\[
f=
\left\{
\delta_{\sigma_i,a},
\delta_{\sigma_i,a}\delta_{\sigma_j,b}
\right\}.
\]

The Hamiltonian therefore becomes

\[
H(\sigma)
=
-\sum_i h_i(\sigma_i)
-
\sum_{i<j}
J_{ij}(\sigma_i,\sigma_j).
\]

The equilibrium distribution is

\[
P_\theta(\sigma)
=
\frac{1}{Z_\theta}
e^{-\beta H_\theta(\sigma)}.
\]

The partition function

\[
Z_\theta
=
\sum_\sigma
e^{-\beta H_\theta(\sigma)}
\]

normalizes the probability.

The free energy is

\[
F=-\beta^{-1}\log Z.
\]

---

# 3. Gauge Freedom

The Potts parameters are non-unique.

Adding constants to rows or columns of

\[
J_{ij}(a,b)
\]

can be compensated by changes in

\[
h_i(a)
\]

without changing

\[
P(\sigma).
\]

A unique representation is obtained by imposing the zero-sum gauge

\[
\sum_a h_i(a)=0,
\]

\[
\sum_aJ_{ij}(a,b)=0,
\]

\[
\sum_bJ_{ij}(a,b)=0.
\]

---

# 4. Bayesian Network Prior

A Bayesian Network factorizes

\[
P_0(\sigma)
=
\prod_i
P(\sigma_i|Pa(i)).
\]

The BN is learned from discretized MD trajectories.

Its conditional probability tables (CPTs) define a probabilistic prior over configurations.

Unlike the Potts model it does not define an equilibrium Hamiltonian.

---

# 5. Maximum Likelihood Learning

Given MD samples

\[
\{\sigma^{(m)}\}_{m=1}^M,
\]

the log likelihood is

\[
\mathcal L(\theta)
=
\sum_m
\log P_\theta(\sigma^{(m)}).
\]

Substituting the Boltzmann distribution,

\[
\mathcal L
=
-\beta
\sum_mH(\sigma^{(m)})
-
M\log Z.
\]

Differentiate

\[
\frac{\partial\mathcal L}{\partial\theta}
=
-\beta
\sum_m
\frac{\partial H}{\partial\theta}
-
M
\frac{\partial\log Z}{\partial\theta}.
\]

Since

\[
\frac{\partial\log Z}{\partial\theta}
=
-\beta
\left<
\frac{\partial H}{\partial\theta}
\right>_{P_\theta},
\]

we obtain

\[
\boxed{
\nabla_\theta\mathcal L
=
M
\left(
\langle f\rangle_{MD}
-
\langle f\rangle_{P_\theta}
\right)
}
\]

which is the classical Boltzmann-machine learning rule.

---

# 6. Bayesian-Regularized Objective

We propose

\[
\boxed{
\mathcal J(\theta)
=
\mathbb E_{MD}[\log P_\theta]
-
\lambda
D_{KL}(P_0\|P_\theta)
-
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2
}
\]

Expand

\[
D_{KL}(P_0\|P_\theta)
=
\mathbb E_{P_0}
[\log P_0-\log P_\theta].
\]

Since

\[
P_0
\]

is independent of

\[
\theta,
\]

\[
\nabla
D_{KL}
=
-
\mathbb E_{P_0}[f]
+
\mathbb E_{P_\theta}[f].
\]

Combining terms,

\[
\boxed{
\nabla\mathcal J
=
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
-
(1+\lambda)
\langle f\rangle_{P_\theta}
}
\]

---

# 7. Mixed Target Moments

Setting

\[
\nabla\mathcal J=0
\]

yields

\[
\boxed{
\langle f\rangle_{P_\theta}
=
\frac{
\langle f\rangle_{MD}
+
\lambda
\langle f\rangle_{P_0}
}
{1+\lambda}
}
\]

Thus only the target moments change.

The optimizer remains identical to classical Boltzmann machine learning.

---

# 8. Graph Regularization

The BN graph defines

\[
E_{BNM}.
\]

Edges outside the graph are penalized

\[
R=
\gamma
\sum_{(i,j)\notin E_{BNM}}
\|J_{ij}\|_F^2.
\]

Gradient

\[
\frac{\partial R}{\partial J_{ij}}
=
2\gamma J_{ij}.
\]

Hence

\[
\nabla_{J_{ij}}
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
\mathbf1[(i,j)\notin E_{BNM}].
\]

---

# 9. Computational Algorithm

1. Discretize MD.
2. Learn BN.
3. Estimate CPTs.
4. Sample BN to estimate \(P_0\).
5. Compute BN moments.
6. Compute MD moments.
7. Form mixed target moments.
8. Monte Carlo sample Potts model.
9. Update \(h,J\).
10. Repeat until convergence.

The existing inverse-Ising implementation is reused:

- `msa.cpp` : compute moments
- `model.cpp` : Hamiltonian
- `mc.cpp` : Monte Carlo
- `run_ising.cpp` : optimization

Only the target moments are replaced.

---

# 10. Limiting Cases

\[
\lambda=\gamma=0
\]

reduces to standard inverse Potts learning.

\[
\gamma=0
\]

gives KL regularization only.

\[
\lambda=0
\]

gives graph regularization only.

The complete model combines all three sources of information:

- MD equilibrium statistics
- Bayesian Network distribution
- Bayesian Network topology

into a single thermodynamically consistent Hamiltonian.
