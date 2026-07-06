#!/usr/bin/env python3
"""
fit_potts_bnm.py

Fit a graph-regularized Potts+BNM model using discretized molecular dynamics
(MD) trajectories, Bayesian Network-derived samples, and a Bayesian Network
edge list.

Inputs
------
--md
    CSV file containing discretized MD residue/contact states.

--bn
    CSV file containing Bayesian Network-generated samples.

--edges
    CSV file containing Bayesian Network edges with columns i,j.

Outputs
-------
out_dir/
    final_h.npy
    final_J.npy
    final_params.txt
    training_history.csv
    run_metadata.json
    model_stats_*.npy
    samples_iter_*.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


class PottsModel:
    def __init__(self, N: int, q: int, reg_lambda: float = 0.0, seed: int = 12345):
        self.N = N
        self.q = q
        self.reg_lambda = reg_lambda
        self.rng = np.random.default_rng(seed)

        # h[i, a]
        self.h = np.zeros((N, q), dtype=float)

        # J[i, j, a, b], only i < j is used
        self.J = np.zeros((N, N, q, q), dtype=float)

    def energy(self, seq: np.ndarray) -> float:
        e = 0.0

        for i in range(self.N):
            e += self.h[i, seq[i]]

        for i in range(self.N - 1):
            si = seq[i]
            for j in range(i + 1, self.N):
                sj = seq[j]
                e += self.J[i, j, si, sj]

        return e

    def delta_energy(self, seq: np.ndarray, site: int, new_state: int) -> float:
        old_state = seq[site]
        if new_state == old_state:
            return 0.0

        dE = self.h[site, new_state] - self.h[site, old_state]

        for j in range(self.N):
            if j == site:
                continue
            sj = seq[j]
            if site < j:
                dE += self.J[site, j, new_state, sj] - self.J[site, j, old_state, sj]
            else:
                dE += self.J[j, site, sj, new_state] - self.J[j, site, sj, old_state]

        return dE

    def random_sequence(self) -> np.ndarray:
        return self.rng.integers(0, self.q, size=self.N, dtype=int)

    def metropolis_sweep(self, seq: np.ndarray, T: float = 1.0) -> None:
        for _ in range(self.N):
            site = int(self.rng.integers(0, self.N))
            old_state = seq[site]

            new_state = old_state
            while new_state == old_state:
                new_state = int(self.rng.integers(0, self.q))

            dE = self.delta_energy(seq, site, new_state)
            if dE <= 0.0 or self.rng.random() < np.exp(-dE / T):
                seq[site] = new_state


def load_discrete_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    data = df.to_numpy(dtype=int)

    if data.ndim != 2:
        raise ValueError(f"{csv_file} is not a 2D table.")
    if np.min(data) < 0:
        raise ValueError(f"{csv_file} contains negative states.")

    columns = list(df.columns)
    N = data.shape[1]
    q = int(np.max(data)) + 1
    return df, data, columns, N, q


def align_bn_columns(md_df: pd.DataFrame, bn_df: pd.DataFrame) -> pd.DataFrame:
    if bn_df.shape[1] != md_df.shape[1]:
        raise ValueError(
            f"Column mismatch: MD has {md_df.shape[1]} columns, BN has {bn_df.shape[1]} columns."
        )

    if list(bn_df.columns) == list(md_df.columns):
        return bn_df

    bn_df = bn_df.copy()
    bn_df.columns = md_df.columns
    return bn_df


def compute_moments(data: np.ndarray, q: int):
    nseq, N = data.shape
    f1 = np.zeros((N, q), dtype=float)
    f2 = np.zeros((N, N, q, q), dtype=float)

    for seq in data:
        for i in range(N):
            f1[i, seq[i]] += 1.0

        for i in range(N - 1):
            si = seq[i]
            for j in range(i + 1, N):
                sj = seq[j]
                f2[i, j, si, sj] += 1.0

    f1 /= nseq
    f2 /= nseq
    return f1, f2


def build_edge_weight_matrix(N: int, edge_csv: str, w_in: float = 0.1, w_out: float = 1.0):
    W = np.full((N, N), w_out, dtype=float)
    np.fill_diagonal(W, 0.0)

    edges = pd.read_csv(edge_csv)
    if not {"i", "j"}.issubset(edges.columns):
        raise ValueError(f"{edge_csv} must contain columns i,j")

    for row in edges.itertuples(index=False):
        i = int(row.i)
        j = int(row.j)
        if i == j:
            continue
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError(f"Edge ({i},{j}) out of bounds for N={N}")
        a, b = min(i, j), max(i, j)
        W[a, b] = w_in
        W[b, a] = w_in

    return W


def initialize_fields_from_target(model: PottsModel, f1_target: np.ndarray, pseudocount: float = 1e-4):
    model.h = -np.log(f1_target + pseudocount)
    model.J.fill(0.0)
    apply_zero_sum_gauge(model)


def estimate_model_moments(
    model: PottsModel,
    mc_steps: int,
    burnin_sweeps: int = 500,
    sample_every: int = 1,
    T: float = 1.0,
    init_seq: np.ndarray | None = None,
    num_save_seqs: int = 100,
):
    if init_seq is None:
        seq = model.random_sequence()
    else:
        seq = init_seq.copy()

    for _ in range(burnin_sweeps):
        model.metropolis_sweep(seq, T=T)

    N, q = model.N, model.q
    f1_model = np.zeros((N, q), dtype=float)
    f2_model = np.zeros((N, N, q, q), dtype=float)

    saved = []
    energies = []
    nsamples = 0
    save_stride = max(1, mc_steps // max(1, num_save_seqs))

    for step in range(mc_steps):
        model.metropolis_sweep(seq, T=T)

        if step % sample_every == 0:
            nsamples += 1
            energies.append(model.energy(seq))

            if step % save_stride == 0 and len(saved) < num_save_seqs:
                saved.append(seq.copy())

            for i in range(N):
                f1_model[i, seq[i]] += 1.0

            for i in range(N - 1):
                si = seq[i]
                for j in range(i + 1, N):
                    sj = seq[j]
                    f2_model[i, j, si, sj] += 1.0

    f1_model /= nsamples
    f2_model /= nsamples
    avg_energy = float(np.mean(energies))

    return f1_model, f2_model, np.array(saved, dtype=int), avg_energy


def apply_zero_sum_gauge(model: PottsModel):
    model.h -= model.h.mean(axis=1, keepdims=True)

    N = model.N
    for i in range(N - 1):
        for j in range(i + 1, N):
            block = model.J[i, j]
            row_mean = block.mean(axis=1, keepdims=True)
            col_mean = block.mean(axis=0, keepdims=True)
            grand_mean = block.mean()
            model.J[i, j] = block - row_mean - col_mean + grand_mean


def rms_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def block_frobenius_norms(J: np.ndarray):
    N = J.shape[0]
    B = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        for j in range(i + 1, N):
            B[i, j] = np.linalg.norm(J[i, j], ord="fro")
            B[j, i] = B[i, j]
    return B


def weighted_block_l2_penalty(J: np.ndarray, W: np.ndarray, alpha_graph_l2: float):
    N = J.shape[0]
    val = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            val += W[i, j] * np.sum(J[i, j] ** 2)
    return alpha_graph_l2 * val


def weighted_block_l1_penalty(J: np.ndarray, W: np.ndarray, alpha_graph_l1: float):
    N = J.shape[0]
    val = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            val += W[i, j] * np.linalg.norm(J[i, j], ord="fro")
    return alpha_graph_l1 * val


def save_parameters(model: PottsModel, out_prefix: str):
    np.save(f"{out_prefix}_h.npy", model.h)
    np.save(f"{out_prefix}_J.npy", model.J)

    with open(f"{out_prefix}_params.txt", "w") as f:
        N, q = model.h.shape

        for i in range(N):
            for a in range(q):
                f.write(f"h {i} {a} {model.h[i, a]:.12g}\n")

        for i in range(N - 1):
            for j in range(i + 1, N):
                for a in range(q):
                    for b in range(q):
                        f.write(f"J {i} {j} {a} {b} {model.J[i, j, a, b]:.12g}\n")


def save_samples(samples: np.ndarray, columns: list[str], out_csv: str):
    pd.DataFrame(samples, columns=columns).to_csv(out_csv, index=False)


def save_moments(f1: np.ndarray, f2: np.ndarray, prefix: str):
    np.save(f"{prefix}_1p.npy", f1)
    np.save(f"{prefix}_2p.npy", f2)


def fit_potts_md_bn_elastic_graph(
    md_csv: str,
    bn_csv: str,
    edge_csv: str,
    out_dir: str = "potts_md_bn_elastic_graph_output",
    bn_weight: float = 0.2,
    reg_lambda: float = 1e-4,
    alpha_graph_l2: float = 1e-3,
    alpha_graph_l1: float = 5e-3,
    w_in: float = 0.1,
    w_out: float = 1.0,
    seed: int = 12345,
    max_iter: int = 150,
    mc_steps: int = 20000,
    burnin_sweeps: int = 500,
    sample_every: int = 1,
    eps0_h: float = 0.10,
    eps0_J: float = 0.01,
    eps_inc: float = 1.05,
    eps_dec: float = 0.50,
    eps_min_h: float = 1e-4,
    eps_max_h: float = 0.50,
    eps_min_J: float = 1e-5,
    eps_max_J_N: float = 1.0,
    gamma_mom: float = 0.5,
    T: float = 1.0,
    pseudocount: float = 1e-4,
    conv_type: str = "rmsd",
    cutoff_rmsd: float = 1e-3,
    cutoff_freq: float = 1e-3,
    adaptive_stepsize_on: bool = True,
    adaptive_sampling_on: bool = True,
    save_every: int = 5,
    l1_eps: float = 1e-8,
):
    if not (0.0 <= bn_weight <= 1.0):
        raise ValueError("bn_weight must be between 0 and 1.")

    os.makedirs(out_dir, exist_ok=True)

    md_df, md_data, md_columns, N, q_md = load_discrete_csv(md_csv)

    bn_df = pd.read_csv(bn_csv)
    bn_df = align_bn_columns(md_df, bn_df)
    bn_data = bn_df.to_numpy(dtype=int)
    q_bn = int(np.max(bn_data)) + 1

    if md_data.shape[1] != bn_data.shape[1]:
        raise ValueError("MD and BN data must have same number of columns.")

    q = max(q_md, q_bn)

    W = build_edge_weight_matrix(N=N, edge_csv=edge_csv, w_in=w_in, w_out=w_out)

    f1_md, f2_md = compute_moments(md_data, q)
    f1_bn, f2_bn = compute_moments(bn_data, q)

    alpha = bn_weight
    f1_target = (1.0 - alpha) * f1_md + alpha * f1_bn
    f2_target = (1.0 - alpha) * f2_md + alpha * f2_bn

    model = PottsModel(N=N, q=q, reg_lambda=reg_lambda, seed=seed)
    initialize_fields_from_target(model, f1_target, pseudocount=pseudocount)

    alpha_h = np.full((N, q), eps0_h, dtype=float)
    alpha_J = np.full((N, N, q, q), eps0_J, dtype=float)

    dh_prev = np.ones((N, q), dtype=float)
    dJ_prev = np.ones((N, N, q, q), dtype=float)

    change_h = np.zeros((N, q), dtype=float)
    change_J = np.zeros((N, N, q, q), dtype=float)

    init_seq = md_data[0].copy()
    history = []
    tri_i, tri_j = np.triu_indices(N, k=1)

    converged = False
    niter = 0

    while not converged:
        f1_model, f2_model, samples, avg_energy = estimate_model_moments(
            model=model,
            mc_steps=mc_steps,
            burnin_sweeps=burnin_sweeps,
            sample_every=sample_every,
            T=T,
            init_seq=init_seq,
            num_save_seqs=100,
        )

        dh = f1_model - f1_target
        dJ = f2_model - f2_target

        # Standard global L2 weight decay
        dh -= 2.0 * model.reg_lambda * model.h
        dJ -= 2.0 * model.reg_lambda * model.J

        # Graph-guided elastic block regularization on J
        for i in range(N - 1):
            for j in range(i + 1, N):
                Jij = model.J[i, j]
                w = W[i, j]

                # block L2
                dJ[i, j] -= 2.0 * alpha_graph_l2 * w * Jij

                # block L1 / group lasso
                norm_ij = np.linalg.norm(Jij, ord="fro")
                if norm_ij > 0.0:
                    dJ[i, j] -= alpha_graph_l1 * w * (Jij / (norm_ij + l1_eps))

        change_h = gamma_mom * change_h + alpha_h * dh
        change_J = gamma_mom * change_J + alpha_J * dJ

        diff1 = rms_diff(f1_model, f1_target)
        diff2 = rms_diff(f2_model[tri_i, tri_j, :, :], f2_target[tri_i, tri_j, :, :])
        diff1_md = rms_diff(f1_model, f1_md)
        diff2_md = rms_diff(f2_model[tri_i, tri_j, :, :], f2_md[tri_i, tri_j, :, :])
        diff1_bn = rms_diff(f1_model, f1_bn)
        diff2_bn = rms_diff(f2_model[tri_i, tri_j, :, :], f2_bn[tri_i, tri_j, :, :])

        max1 = float(np.max(np.abs(dh)))
        max2 = float(np.max(np.abs(dJ[tri_i, tri_j, :, :])))
        total_rmsd = diff1 + diff2

        graph_pen_l2 = weighted_block_l2_penalty(model.J, W, alpha_graph_l2)
        graph_pen_l1 = weighted_block_l1_penalty(model.J, W, alpha_graph_l1)

        block_norms = block_frobenius_norms(model.J)
        in_vals = [block_norms[i, j] for i in range(N - 1) for j in range(i + 1, N) if W[i, j] == w_in]
        out_vals = [block_norms[i, j] for i in range(N - 1) for j in range(i + 1, N) if W[i, j] == w_out]

        avg_block_in = float(np.mean(in_vals)) if len(in_vals) > 0 else 0.0
        avg_block_out = float(np.mean(out_vals)) if len(out_vals) > 0 else 0.0

        print(
            f"iter {niter:4d} | mc_steps={mc_steps:7d} | "
            f"rmsd_target_1p={diff1:.6e} | rmsd_target_2p={diff2:.6e} | total={total_rmsd:.6e} | "
            f"rmsd_md_1p={diff1_md:.6e} | rmsd_md_2p={diff2_md:.6e} | "
            f"rmsd_bn_1p={diff1_bn:.6e} | rmsd_bn_2p={diff2_bn:.6e} | "
            f"graph_pen_l2={graph_pen_l2:.6e} | graph_pen_l1={graph_pen_l1:.6e} | "
            f"avg||J||_in={avg_block_in:.6e} | avg||J||_out={avg_block_out:.6e} | "
            f"avgE={avg_energy:.6f}"
        )

        history.append(
            {
                "iter": niter,
                "mc_steps": mc_steps,
                "rmsd_target_1p": diff1,
                "rmsd_target_2p": diff2,
                "total_rmsd_target": total_rmsd,
                "rmsd_md_1p": diff1_md,
                "rmsd_md_2p": diff2_md,
                "rmsd_bn_1p": diff1_bn,
                "rmsd_bn_2p": diff2_bn,
                "max_abs_dh": max1,
                "max_abs_dJ": max2,
                "graph_penalty_l2": graph_pen_l2,
                "graph_penalty_l1": graph_pen_l1,
                "avg_block_norm_in": avg_block_in,
                "avg_block_norm_out": avg_block_out,
                "avg_energy": avg_energy,
            }
        )

        if niter % save_every == 0:
            save_parameters(model, os.path.join(out_dir, f"iter_{niter:04d}"))
            save_samples(samples, md_columns, os.path.join(out_dir, f"samples_iter_{niter:04d}.csv"))
            save_moments(f1_model, f2_model, os.path.join(out_dir, f"model_stats_iter_{niter:04d}"))

        if (
            niter > max_iter
            or (conv_type == "max" and max1 < cutoff_freq and max2 < cutoff_freq)
            or (conv_type == "rmsd" and total_rmsd < cutoff_rmsd)
        ):
            converged = True
            print("Converged.")
            break

        if adaptive_sampling_on and total_rmsd < 5e-3:
            mc_steps *= 2
            print(f"Increasing mc_steps to {mc_steps}")

        model.h += change_h
        model.J += change_J
        apply_zero_sum_gauge(model)

        if adaptive_stepsize_on:
            prod_h = dh_prev * dh
            alpha_h = np.where((prod_h > 0) & (alpha_h < eps_max_h), eps_inc * alpha_h, alpha_h)
            alpha_h = np.where((prod_h < 0) & (alpha_h > eps_min_h), eps_dec * alpha_h, alpha_h)
            alpha_h = np.clip(alpha_h, eps_min_h, eps_max_h)

            eps_max_J = eps_max_J_N / N
            prod_J = dJ_prev * dJ
            alpha_J = np.where((prod_J > 0) & (alpha_J < eps_max_J), eps_inc * alpha_J, alpha_J)
            alpha_J = np.where((prod_J < 0) & (alpha_J > eps_min_J), eps_dec * alpha_J, alpha_J)
            alpha_J = np.clip(alpha_J, eps_min_J, eps_max_J)

        dh_prev = dh.copy()
        dJ_prev = dJ.copy()

        init_seq = samples[-1].copy() if len(samples) > 0 else model.random_sequence()
        niter += 1

    save_parameters(model, os.path.join(out_dir, "final"))
    save_moments(f1_md, f2_md, os.path.join(out_dir, "md_stats"))
    save_moments(f1_bn, f2_bn, os.path.join(out_dir, "bn_stats"))
    save_moments(f1_target, f2_target, os.path.join(out_dir, "target_stats"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    meta = {
        "md_csv": md_csv,
        "bn_csv": bn_csv,
        "edge_csv": edge_csv,
        "N": N,
        "q": q,
        "bn_weight": bn_weight,
        "reg_lambda": reg_lambda,
        "alpha_graph_l2": alpha_graph_l2,
        "alpha_graph_l1": alpha_graph_l1,
        "w_in": w_in,
        "w_out": w_out,
        "seed": seed,
        "max_iter": max_iter,
        "mc_steps_init": history[0]["mc_steps"] if history else mc_steps,
        "burnin_sweeps": burnin_sweeps,
        "sample_every": sample_every,
        "eps0_h": eps0_h,
        "eps0_J": eps0_J,
        "eps_inc": eps_inc,
        "eps_dec": eps_dec,
        "eps_min_h": eps_min_h,
        "eps_max_h": eps_max_h,
        "eps_min_J": eps_min_J,
        "eps_max_J_N": eps_max_J_N,
        "gamma_mom": gamma_mom,
        "T": T,
        "pseudocount": pseudocount,
        "conv_type": conv_type,
        "cutoff_rmsd": cutoff_rmsd,
        "cutoff_freq": cutoff_freq,
        "adaptive_stepsize_on": adaptive_stepsize_on,
        "adaptive_sampling_on": adaptive_sampling_on,
        "l1_eps": l1_eps,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Finished. Outputs saved in: {out_dir}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fit a graph-regularized Potts+BNM model."
    )

    parser.add_argument(
        "--md",
        required=True,
        help="CSV file containing discretized MD residue/contact states."
    )

    parser.add_argument(
        "--bn",
        required=True,
        help="CSV file containing Bayesian Network-generated samples."
    )

    parser.add_argument(
        "--edges",
        required=True,
        help="CSV file containing Bayesian Network edges with columns i,j."
    )

    parser.add_argument(
        "--out",
        default="potts_md_bn_elastic_graph_output",
        help="Output directory. Default: potts_md_bn_elastic_graph_output"
    )

    args = parser.parse_args()

    fit_potts_md_bn_elastic_graph(
        md_csv=args.md,
        bn_csv=args.bn,
        edge_csv=args.edges,
        out_dir=args.out,
        bn_weight=0.20,
        reg_lambda=1e-4,
        alpha_graph_l2=1e-3,
        alpha_graph_l1=5e-3,
        w_in=0.10,
        w_out=1.00,
        seed=12345,
        max_iter=150,
        mc_steps=20000,
        burnin_sweeps=500,
        sample_every=1,
        eps0_h=0.10,
        eps0_J=0.01,
        eps_inc=1.05,
        eps_dec=0.50,
        eps_min_h=1e-4,
        eps_max_h=0.50,
        eps_min_J=1e-5,
        eps_max_J_N=1.0,
        gamma_mom=0.5,
        T=1.0,
        pseudocount=1e-4,
        conv_type="rmsd",
        cutoff_rmsd=1e-3,
        cutoff_freq=1e-3,
        adaptive_stepsize_on=True,
        adaptive_sampling_on=True,
        save_every=5,
        l1_eps=1e-8,
    )
