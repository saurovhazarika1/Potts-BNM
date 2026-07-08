# Theory

## Motivation

The Potts+BNM framework combines Bayesian Network Modeling (BNM) with a
pairwise Potts Hamiltonian to construct an interaction-space statistical
mechanics model of biomolecular communication.

## Maximum Entropy

Given empirical marginals, maximize

$$
S=-\sum_\sigma P(\sigma)\ln P(\sigma)
$$

subject to normalization and empirical one- and two-body marginals.

The solution is

$$
P(\sigma)=\frac{1}{Z}\exp[-\beta H(\sigma)].
$$

## Potts Hamiltonian

$$
H(\sigma)=
-\sum_i h_i(\sigma_i)
-\sum_{i<j}J_{ij}(\sigma_i,\sigma_j).
$$

The Bayesian network defines the interaction graph on which Potts
couplings are learned.

## MD Frame Probability

$$
P_{frame}=\frac{e^{-\beta H(\sigma)}}{Z}.
$$

Window probabilities are obtained from window-averaged frame energies.
Communication pathways are then analyzed within high-probability and
low-probability regions of the equilibrium ensemble.
