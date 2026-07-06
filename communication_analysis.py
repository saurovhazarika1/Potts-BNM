#!/usr/bin/env python3
"""
communication_analysis.py

Purpose
-------
Probability-conditioned ligand->G-protein communication analysis using paths
regenerated directly on percentile-thresholded learned-J graphs.

This script DOES NOT use temporal_path_probabilities.csv to define paths.
Instead, for each ligand and each EDGE_PERCENTILE:

  1. Load contact-state CSV, h, and J.
  2. Compute pairwise coupling strengths S_ij = ||J_ij||_F.
  3. Build a learned-J graph after applying EDGE_PERCENTILE cutoff.
       Example: EDGE_PERCENTILE=98 keeps strongest 2% of nonzero J edges.
  4. Generate source->target paths on this graph using Dijkstra/Yen-style
     shortest-simple paths with edge cost:

          cost_ij = Jmax - ||J_ij||_F + eps

     so stronger J edges are preferred with non-negative Dijkstra costs.
  5. For each generated path, compute window path energies using both modes:

          HJ          : E_path(t) = -sum_i h_i[x_i(t)] - sum_edges J_ij[x_i(t),x_j(t)]
          J_ONLY      : E_path(t) =                 - sum_edges J_ij[x_i(t),x_j(t)]
          HJ_MEAN     : HJ energy divided by (number of nodes + number of edges)
          J_ONLY_MEAN : J_ONLY energy divided by number of edges

  6. Convert path energies to Boltzmann-like probabilities within each window.
  7. Separately compute the full Potts/BNM energy of every MD frame/window using
     all h_i[x_i(t)] and J_ij[x_i(t),x_j(t)] terms.
  8. Define low/high MD-probability regions from the full MD-window Potts energy,
     NOT from path-derived communication probability.
  9. For each probability-mass path subset (PMASS_80/90/95), compute
     communication/frustration contrasts between low/high MD-probability regions.
 10. Correlate every metric with FRET/efficacy, requiring all 5 ligands.
 11. Produce robustness summaries across EDGE_PERCENTILE, ENERGY_MODE, and probability-mass cutoffs.
 12. Additionally compute positive/negative correlation balance metrics:

          positive_fraction      : fraction of path pairs with C_ij > 0
          negative_fraction      : fraction of path pairs with C_ij < 0
          pos_neg_ratio          : positive_fraction / negative_fraction
          log_pos_neg_ratio      : log(positive_fraction / negative_fraction)
          strong_pos_neg_ratio   : fraction(C_ij > +0.3) / fraction(C_ij < -0.3)
          log_strong_pos_neg_ratio : log(strong_pos_neg_ratio)

     and their low/high/high_minus_low/high_over_low contrasts.

Main outputs
------------
OUTDIR/
  path_generation_diagnostics.csv
  generated_paths_all_settings.csv
  window_metrics_all_settings.csv
  binned_metrics_all_settings.csv
  contrast_metrics_all_settings.csv
  all5_correlations_all_settings.csv
  incomplete_metric_diagnostics.csv
  metric_robustness_summary.csv
  metric_rank_stability_summary.csv
  figures/*.png

How to run
----------
python communication_analysis.py \
    --base MASTER_temporal_MD_contact_path_analysis_balancedMI_K500 \
    --ligands 0,1,2,3,4 \
    --fret 0:0.018,1:0.080,2:0.060,3:0.032,4:0.030 \
    --sources 286 \
    --targets 287

Recommended temporary setting for your presentation
---------------------------------------------------
Use EDGE_PERCENTILES = [98.0] initially. Later add [97.5, 97.0, 95.0, 90.0]
for a broader robustness scan.
"""

import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE = "MASTER_temporal_MD_contact_path_analysis_balancedMI_K500"

CONTACT_PATTERN = "ligand_{lig}/bandyt/rec_lig_Gpep_lig_{lig}_ds20.csv"
H_PATTERN = "ligand_{lig}/bandyt/potts_md_bn_elastic_graph_output/iter_0150_h.npy"
J_PATTERN = "ligand_{lig}/bandyt/potts_md_bn_elastic_graph_output/iter_0150_J.npy"
DOT_PATTERN = "ligand_{lig}/bandyt/rec_lig_Gpep_ds20rendering.dot"  # diagnostic only

OUTDIR = os.path.join(BASE, "REGENERATED_PATHS_MDframe_potts_probability_pos_neg_balance_HJ_JONLY")
FIGDIR = os.path.join(OUTDIR, "figures")

LIGANDS = [0, 1, 2, 3, 4]
FRET = {0: 0.018, 1: 0.080, 2: 0.060, 3: 0.032, 4: 0.030}
EXPECTED_EFFICACY_ORDER_HIGH_TO_LOW = [1, 2, 3, 4, 0]

# Ligand -> G-peptide convention from your previous analysis.
SOURCE_NODES = [286]
TARGET_NODES = [287]

# Edge percentile cutoffs. 98 keeps strongest 2% of nonzero J edges.
# For tomorrow, you can keep only [98.0]. For robustness, add more values.
EDGE_PERCENTILES = [98.0]
# EDGE_PERCENTILES = [98.0, 97.5, 97.0, 96.0, 95.0, 94.0, 93.0, 90.0]

ENERGY_MODES = ["HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"]
# Probability-mass path selection replaces arbitrary TOP_N values.
# Each PMASS label selects the minimum number of probability-ranked paths
# needed to capture that fraction of total path probability for each ligand.
# Instead of arbitrary fixed TOP_N, select the minimal number of paths needed
# to capture a target cumulative probability mass for each ligand/energy mode.
PROB_MASS_CUTOFFS = [0.80, 0.90, 0.95]
TOP_N_LIST = [f"PMASS_{int(100*x)}" for x in PROB_MASS_CUTOFFS]

# Path generation settings
MAX_PATHS = 1000
PATHS_PER_SOURCE = 1000
PATHS_PER_TARGET = 1000
TARGETS_PER_SOURCE = 25
MIN_PATH_LEN = 2
MAX_PATH_LEN = 14

# Window and probability-bin settings
WINDOW_SIZE = 100
N_PROB_BINS = 5
MIN_SELECTED_PATHS = 3
MIN_WINDOWS_PER_BIN_FOR_COMM = 3

# Energy-mode-specific beta values from the energy-spread diagnostics.
# These roughly scale beta to the typical spread of path energies.
BETA_BY_ENERGY_MODE = {
    "HJ": 2.0,
    "HJ_MEAN": 50.0,
    "J_ONLY": 25.0,
    "J_ONLY_MEAN": 200.0,
}

# Separate beta for the MD-frame/window probability landscape.
# This is independent of path-probability beta. High/low bins are now defined
# from full Potts MD-frame energy, not from path-derived communication scores.
FRAME_BETA = 1.0
DEFAULT_BETA = 1.0
POS_THRESH = 0.3
NEG_THRESH = -0.3
EPS = 1e-12

# Correlation policy
REQUIRE_ALL_LIGANDS_FOR_CORR = True
MIN_LIGANDS_FOR_CORR = len(LIGANDS)

FOCUS_METRICS = [
    "mean_path_corr_high_over_low",
    "mean_path_corr_high_minus_low",
    "strong_positive_fraction_gt_0p3_high_over_low",
    "strong_positive_fraction_gt_0p3_high_minus_low",
    "strong_negative_fraction_lt_minus_0p3_high_over_low",
    "strong_negative_fraction_lt_minus_0p3_high_minus_low",
    "strong_pos_neg_ratio_high_over_low",
    "strong_pos_neg_ratio_high_minus_low",
    "log_strong_pos_neg_ratio_high_over_low",
    "log_strong_pos_neg_ratio_high_minus_low",
    "conflict_fraction_high_over_low",
    "conflict_fraction_high_minus_low",
    "positive_fraction_high_over_low",
    "positive_fraction_high_minus_low",
    "negative_fraction_high_over_low",
    "negative_fraction_high_minus_low",
    "pos_neg_ratio_high_over_low",
    "pos_neg_ratio_high_minus_low",
    "log_pos_neg_ratio_high_over_low",
    "log_pos_neg_ratio_high_minus_low",
    "spectral_effective_modes_high_over_low",
    "spectral_effective_modes_high_minus_low",
    "spectral_entropy_high_over_low",
    "spectral_entropy_high_minus_low",
    "top_eigen_fraction_high_over_low",
    "top_eigen_fraction_high_minus_low",
    "participation_ratio_high_over_low",
    "participation_ratio_high_minus_low",
    "mean_Jmin_frustration_sum_high_over_low",
    "mean_Jmin_frustration_sum_high_minus_low",
    "mean_Jmin_frustration_mean_high_over_low",
    "mean_Jmin_frustration_mean_high_minus_low",
]


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def beta_for_mode(energy_mode):
    return float(BETA_BY_ENERGY_MODE.get(str(energy_mode).upper(), DEFAULT_BETA))


def stable_prob(E, beta=DEFAULT_BETA):
    E = np.asarray(E, dtype=float)
    if E.size == 0 or np.all(~np.isfinite(E)):
        return np.full_like(E, np.nan, dtype=float)
    logw = -float(beta) * E
    logw -= np.nanmax(logw)
    w = np.exp(logw)
    return w / (np.nansum(w) + EPS)


def parse_path(p):
    try:
        x = ast.literal_eval(str(p))
        if isinstance(x, list):
            return [str(v) for v in x]
    except Exception:
        pass
    p = str(p).strip()
    if "->" in p:
        return [x.strip() for x in p.split("->")]
    if "," in p:
        return [x.strip() for x in p.split(",")]
    return p.split()


def load_contact_data(lig):
    path = CONTACT_PATTERN.format(lig=lig)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    exclude = {"time", "Time", "frame", "Frame", "Time(ps)"}
    cols = [str(c) for c in df.columns if c not in exclude]
    X = df[cols].values.astype(int)
    label_to_idx = {str(c): i for i, c in enumerate(cols)}
    idx_to_label = {i: str(c) for i, c in enumerate(cols)}
    return X, cols, label_to_idx, idx_to_label


def load_potts(lig):
    h_file = H_PATTERN.format(lig=lig)
    j_file = J_PATTERN.format(lig=lig)
    if not os.path.exists(h_file):
        raise FileNotFoundError(h_file)
    if not os.path.exists(j_file):
        raise FileNotFoundError(j_file)
    h = np.load(h_file, allow_pickle=True)
    J = np.load(j_file, allow_pickle=True)
    return h, J


def read_dot_edges(lig):
    """Diagnostic only. The .dot file is not used for pruning."""
    dot_file = DOT_PATTERN.format(lig=lig)
    if not os.path.exists(dot_file):
        return set()
    edge_re = re.compile(r'"?([^"\s;]+)"?\s*->\s*"?([^"\s;]+)"?')
    edges = set()
    with open(dot_file, "r") as f:
        for line in f:
            m = edge_re.search(line)
            if not m:
                continue
            a, b = str(m.group(1)), str(m.group(2))
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return edges


def resolve_node_label(node, label_to_idx, idx_to_label=None):
    s = str(node)
    candidates = [s]
    if s.startswith("R") and s[1:].isdigit():
        candidates.append(s[1:])
    if s.isdigit():
        candidates.append("R" + s)
    try:
        i = int(float(s))
        candidates.extend([str(i), "R" + str(i)])
    except Exception:
        pass

    for c in candidates:
        if c in label_to_idx:
            return c

    # Fall back to positional index. This matches your older 286/287 convention.
    try:
        i = int(float(s))
        if idx_to_label is not None and i in idx_to_label:
            return idx_to_label[i]
    except Exception:
        pass

    raise KeyError(f"Could not resolve node {node}; tried {candidates[:6]} and positional index")


def node_to_index(node, label_to_idx, idx_to_label=None):
    return label_to_idx[resolve_node_label(node, label_to_idx, idx_to_label)]


def make_windows(nframes, window_size=WINDOW_SIZE):
    rows = []
    w = 0
    for start in range(0, nframes, window_size):
        stop = min(start + window_size, nframes)
        if stop > start:
            rows.append({"window": w, "start_frame": start, "stop_frame": stop})
            w += 1
    return pd.DataFrame(rows)


def window_mean(vals, windows):
    vals = np.asarray(vals, dtype=float)
    out = []
    n = len(vals)
    for _, row in windows.iterrows():
        start = max(0, int(row["start_frame"]))
        stop = min(n, int(row["stop_frame"]))
        out.append(np.nan if stop <= start else float(np.nanmean(vals[start:stop])))
    return np.asarray(out, dtype=float)



def probability_mass_cutoff_from_label(top_n_label):
    """Return probability-mass cutoff for labels like PMASS_80, PMASS_90, PMASS_95."""
    if isinstance(top_n_label, str) and top_n_label.upper().startswith("PMASS_"):
        num = float(top_n_label.split("_", 1)[1])
        return num / 100.0
    try:
        # Backward compatible fixed TOP_N behavior if an integer is supplied.
        return None
    except Exception:
        return None


def select_paths_by_probability_mass(path_info, top_n_label):
    """Select paths either by fixed integer TOP_N or by cumulative probability mass.

    If top_n_label is PMASS_80/90/95, select the smallest set of probability-ranked
    paths whose cumulative mean_path_probability reaches that threshold.
    Returns selected ranks, effective number of selected paths, and the cutoff used.
    """
    ordered = path_info.sort_values("probability_order").copy()
    cutoff = probability_mass_cutoff_from_label(top_n_label)

    if cutoff is None:
        n = int(top_n_label)
        selected = ordered.head(n)["rank"].tolist()
        return selected, len(selected), np.nan

    probs = ordered["mean_path_probability"].astype(float).values
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    n = min(idx + 1, len(ordered))
    selected = ordered.head(n)["rank"].tolist()
    return selected, n, float(cutoff)

# =============================================================================
# GRAPH BUILDING AND PATH GENERATION
# =============================================================================

def all_J_edges_sorted(J, idx_to_label):
    """Return [(strength, label_i, label_j, idx_i, idx_j), ...] sorted descending."""
    n = J.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(np.linalg.norm(J[i, j]))
            if np.isfinite(s) and s > 0:
                edges.append((s, idx_to_label[i], idx_to_label[j], i, j))
    edges.sort(key=lambda x: x[0], reverse=True)
    return edges


def build_graph_from_percentile(sorted_edges, percentile, idx_to_label):
    if len(sorted_edges) == 0:
        raise RuntimeError("No nonzero J edges available")
    strengths = np.asarray([e[0] for e in sorted_edges], dtype=float)
    cutoff = float(np.percentile(strengths, percentile))
    kept = [e for e in sorted_edges if e[0] >= cutoff]

    G = nx.Graph()
    for _, lab in idx_to_label.items():
        G.add_node(lab)

    # Pathway-analysis Dijkstra logic:
    # strong J -> small non-negative cost.
    # cost_ij = Jmax - ||Jij||_F + EPS
    max_strength = max(float(e[0]) for e in kept)

    for s, li, lj, _, _ in kept:
        cost = (max_strength - float(s)) + EPS
        G.add_edge(li, lj, strength=float(s), weight=cost)

    return G, cutoff, len(kept)


def connectivity_stats(G, sources, targets):
    sources = [s for s in sources if s in G]
    targets = [t for t in targets if t in G]
    if len(sources) == 0 or len(targets) == 0:
        return {
            "n_sources": len(sources),
            "n_targets": len(targets),
            "source_reach_frac": 0.0,
            "target_reach_frac": 0.0,
            "pair_reach_frac": 0.0,
            "n_connected_pairs": 0,
            "largest_component_size": 0,
            "n_components": 0,
        }

    comps = list(nx.connected_components(G))
    node_to_comp = {}
    for cid, comp in enumerate(comps):
        for node in comp:
            node_to_comp[node] = cid

    connected_pairs = 0
    connected_sources = set()
    connected_targets = set()
    for s in sources:
        cs = node_to_comp.get(s)
        for t in targets:
            if cs is not None and cs == node_to_comp.get(t):
                connected_pairs += 1
                connected_sources.add(s)
                connected_targets.add(t)

    total_pairs = len(sources) * len(targets)
    return {
        "n_sources": len(sources),
        "n_targets": len(targets),
        "source_reach_frac": len(connected_sources) / max(len(sources), 1),
        "target_reach_frac": len(connected_targets) / max(len(targets), 1),
        "pair_reach_frac": connected_pairs / max(total_pairs, 1),
        "n_connected_pairs": int(connected_pairs),
        "largest_component_size": max((len(c) for c in comps), default=0),
        "n_components": len(comps),
    }


def generate_paths_fast(G, sources, targets):
    """Generate alternative shortest-simple paths on the percentile-thresholded graph."""
    rows = []
    seen = set()

    for s in sources:
        if s not in G:
            continue
        try:
            lengths, _ = nx.single_source_dijkstra(G, source=s, weight="weight")
        except Exception:
            continue

        reachable_targets = [t for t in targets if t in lengths and t != s]
        reachable_targets.sort(key=lambda t: lengths[t])
        reachable_targets = reachable_targets[:TARGETS_PER_SOURCE]

        accepted_for_source = 0
        for t in reachable_targets:
            try:
                gen = nx.shortest_simple_paths(G, s, t, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            except Exception:
                continue

            accepted_for_target = 0
            for path in gen:
                if len(path) < MIN_PATH_LEN or len(path) > MAX_PATH_LEN:
                    continue
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)

                w = 0.0
                for a, b in zip(path[:-1], path[1:]):
                    w += float(G[a][b].get("weight", 1.0))

                rows.append({
                    "rank": len(rows) + 1,
                    "source": s,
                    "target": t,
                    "path": str(path),
                    "path_length_nodes": len(path),
                    "path_length_edges": len(path) - 1,
                    "dijkstra_weight": w,
                })

                accepted_for_source += 1
                accepted_for_target += 1
                if accepted_for_target >= PATHS_PER_TARGET:
                    break
                if accepted_for_source >= PATHS_PER_SOURCE:
                    break
                if len(rows) >= MAX_PATHS:
                    return pd.DataFrame(rows)

            if accepted_for_source >= PATHS_PER_SOURCE:
                break

    return pd.DataFrame(rows)


# =============================================================================
# ENERGY, FRUSTRATION, COMMUNICATION METRICS
# =============================================================================

def full_potts_frame_energy(X, h, J):
    """Compute full Potts/BNM energy for every MD frame.

    E_frame(t) = -sum_i h_i[x_i(t)] - sum_{i<j, nonzero J} J_ij[x_i(t), x_j(t)]

    This is independent of any source->target path. It is used only to define
    the MD conformational probability landscape and the high/low MD regions.
    """
    X = np.asarray(X, dtype=int)
    nframes, n_nodes = X.shape
    E = np.zeros(nframes, dtype=float)

    # Node-field terms: h_i[state_i(t)]
    for i in range(n_nodes):
        E -= h[i, X[:, i]]

    # Pair-coupling terms: J_ij[state_i(t), state_j(t)]
    # Only include nonzero coupling tensors to avoid unnecessary work.
    for i in range(n_nodes):
        Xi = X[:, i]
        for j in range(i + 1, n_nodes):
            Jij = J[i, j]
            if not np.any(np.isfinite(Jij)) or float(np.linalg.norm(Jij)) <= 0.0:
                continue
            E -= Jij[Xi, X[:, j]]

    return E


def build_caches(X, h, J):
    node_cache = {}
    edge_energy_cache = {}
    edge_J_cache = {}

    def node_energy(i):
        if i not in node_cache:
            node_cache[i] = -h[i, X[:, i]]
        return node_cache[i]

    def edge_energy(i, j):
        a, b = sorted((i, j))
        key = (a, b)
        if key not in edge_energy_cache:
            edge_energy_cache[key] = -J[a, b, X[:, a], X[:, b]]
        return edge_energy_cache[key]

    def edge_J_obs(i, j):
        a, b = sorted((i, j))
        key = (a, b)
        if key not in edge_J_cache:
            edge_J_cache[key] = J[a, b, X[:, a], X[:, b]]
        return edge_J_cache[key]

    return node_energy, edge_energy, edge_J_obs


def path_energy(path_idx, node_energy, edge_energy, energy_mode):
    """Frame-wise path energy in four probability modes.

    HJ:
        Total path energy = node h terms + edge J terms.
    J_ONLY:
        Total edge-only path energy.
    HJ_MEAN:
        Length-normalized HJ energy divided by (number of nodes + number of edges).
    J_ONLY_MEAN:
        Length-normalized J-only energy divided by number of edges.

    The path itself is still generated only from the J graph.
    These modes affect only path probability/ranking and downstream metrics.
    """
    mode = energy_mode.upper()
    E = np.zeros_like(node_energy(path_idx[0]), dtype=float)

    include_h = mode in {"HJ", "HJ_MEAN"}
    include_j = mode in {"HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"}

    if mode not in {"HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"}:
        raise ValueError("energy_mode must be HJ, J_ONLY, HJ_MEAN, or J_ONLY_MEAN")

    if include_h:
        for i in path_idx:
            E += node_energy(i)

    if include_j:
        for i, j in zip(path_idx[:-1], path_idx[1:]):
            E += edge_energy(i, j)

    if mode == "HJ_MEAN":
        denom = len(path_idx) + max(len(path_idx) - 1, 1)
        E = E / float(denom)
    elif mode == "J_ONLY_MEAN":
        denom = max(len(path_idx) - 1, 1)
        E = E / float(denom)

    return E

def Jmin_frustration(path_idx, X, J, edge_J_obs):
    edge_frust = []
    for i, j in zip(path_idx[:-1], path_idx[1:]):
        a, b = sorted((i, j))
        Jij_obs = edge_J_obs(a, b)
        Jij_min = float(np.nanmin(J[a, b, :, :]))
        edge_frust.append(Jij_obs - Jij_min)

    if not edge_frust:
        return {
            "Jmin_frustration_mean": np.full(X.shape[0], np.nan),
            "Jmin_frustration_sum": np.full(X.shape[0], np.nan),
        }

    edge_frust = np.vstack(edge_frust).T
    return {
        "Jmin_frustration_mean": np.nanmean(edge_frust, axis=1),
        "Jmin_frustration_sum": np.nansum(edge_frust, axis=1),
    }


def communication_metrics(P_matrix):
    """Compute path-path correlation/spectral metrics from window/bin x paths probability matrix."""
    P_matrix = np.asarray(P_matrix, dtype=float)
    if P_matrix.ndim != 2 or P_matrix.shape[0] < MIN_WINDOWS_PER_BIN_FOR_COMM or P_matrix.shape[1] < MIN_SELECTED_PATHS:
        return {
            "n_variable_paths": 0 if P_matrix.ndim != 2 else int(P_matrix.shape[1]),
            "mean_path_corr": np.nan,
            "strong_positive_fraction_gt_0p3": np.nan,
            "strong_negative_fraction_lt_minus_0p3": np.nan,
            "strong_pos_neg_ratio": np.nan,
            "log_strong_pos_neg_ratio": np.nan,
            "conflict_fraction": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
            "pos_neg_ratio": np.nan,
            "log_pos_neg_ratio": np.nan,
            "spectral_effective_modes": np.nan,
            "spectral_entropy": np.nan,
            "top_eigen_fraction": np.nan,
            "participation_ratio": np.nan,
        }

    std = np.nanstd(P_matrix, axis=0)
    keep = std > EPS
    X = P_matrix[:, keep]

    if X.shape[1] < MIN_SELECTED_PATHS or X.shape[0] < MIN_WINDOWS_PER_BIN_FOR_COMM:
        return {
            "n_variable_paths": int(X.shape[1]),
            "mean_path_corr": np.nan,
            "strong_positive_fraction_gt_0p3": np.nan,
            "strong_negative_fraction_lt_minus_0p3": np.nan,
            "strong_pos_neg_ratio": np.nan,
            "log_strong_pos_neg_ratio": np.nan,
            "conflict_fraction": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
            "pos_neg_ratio": np.nan,
            "log_pos_neg_ratio": np.nan,
            "spectral_effective_modes": np.nan,
            "spectral_entropy": np.nan,
            "top_eigen_fraction": np.nan,
            "participation_ratio": np.nan,
        }

    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    vals = C[np.triu_indices(C.shape[0], k=1)]

    eig = np.linalg.eigvalsh(C)
    eig = np.sort(np.clip(eig, 0, None))[::-1]
    eig_sum = np.sum(eig) + EPS
    p = eig / eig_sum
    H = -np.sum(p[p > 0] * np.log(p[p > 0] + EPS))
    PR = (np.sum(eig) ** 2) / (np.sum(eig ** 2) + EPS)

    # Strong sign-balance metrics using thresholded path-pair correlations.
    # These answer whether the strongly coupled part of the communication ensemble
    # is more co-fluctuation dominated (Cij > +0.3) or competition/substitution
    # dominated (Cij < -0.3).
    strong_positive_fraction = float(np.mean(vals > POS_THRESH))
    strong_negative_fraction = float(np.mean(vals < NEG_THRESH))
    strong_pos_neg_ratio = float((strong_positive_fraction + EPS) / (strong_negative_fraction + EPS))
    log_strong_pos_neg_ratio = float(np.log((strong_positive_fraction + EPS) / (strong_negative_fraction + EPS)))

    # Sign-balance metrics using ALL positive/negative path-pair correlations,
    # not only strong correlations. These answer whether a communication ensemble
    # is more co-fluctuation dominated (Cij > 0) or competition/substitution
    # dominated (Cij < 0).
    positive_fraction = float(np.mean(vals > 0))
    negative_fraction = float(np.mean(vals < 0))
    pos_neg_ratio = float((positive_fraction + EPS) / (negative_fraction + EPS))
    log_pos_neg_ratio = float(np.log((positive_fraction + EPS) / (negative_fraction + EPS)))

    return {
        "n_variable_paths": int(X.shape[1]),
        "mean_path_corr": float(np.nanmean(vals)),
        "strong_positive_fraction_gt_0p3": strong_positive_fraction,
        "strong_negative_fraction_lt_minus_0p3": strong_negative_fraction,
        "strong_pos_neg_ratio": strong_pos_neg_ratio,
        "log_strong_pos_neg_ratio": log_strong_pos_neg_ratio,
        "conflict_fraction": negative_fraction,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "pos_neg_ratio": pos_neg_ratio,
        "log_pos_neg_ratio": log_pos_neg_ratio,
        "spectral_effective_modes": float(np.exp(H)),
        "spectral_entropy": float(H),
        "top_eigen_fraction": float(p[0]),
        "participation_ratio": float(PR),
    }


# =============================================================================
# PER-LIGAND PATH + WINDOW ANALYSIS
# =============================================================================

def compute_ligand_setting(lig, edge_percentile, energy_mode):
    print(f"\nProcessing ligand {lig}, EDGE_PERCENTILE={edge_percentile}, ENERGY_MODE={energy_mode}")

    X, cols, label_to_idx, idx_to_label = load_contact_data(lig)
    h, J = load_potts(lig)

    if h.shape[0] != X.shape[1]:
        raise ValueError(f"Ligand {lig}: h dim {h.shape[0]} != contact columns {X.shape[1]}")
    if J.shape[0] != X.shape[1] or J.shape[1] != X.shape[1]:
        raise ValueError(f"Ligand {lig}: J dims {J.shape[:2]} != contact columns {X.shape[1]}")

    sources = [resolve_node_label(x, label_to_idx, idx_to_label) for x in SOURCE_NODES]
    targets = [resolve_node_label(x, label_to_idx, idx_to_label) for x in TARGET_NODES]

    dot_edges = read_dot_edges(lig)
    sorted_edges = all_J_edges_sorted(J, idx_to_label)
    G, cutoff, retained_edges = build_graph_from_percentile(sorted_edges, edge_percentile, idx_to_label)
    stats = connectivity_stats(G, sources, targets)

    paths_df = generate_paths_fast(G, sources, targets)
    if len(paths_df):
        paths_df = paths_df.sort_values("dijkstra_weight").head(MAX_PATHS).reset_index(drop=True)
        paths_df["rank"] = np.arange(1, len(paths_df) + 1)

    diag = {
        "ligand": lig,
        "EDGE_PERCENTILE": edge_percentile,
        "ENERGY_MODE": energy_mode,
        "BETA": beta_for_mode(energy_mode),
        "FRAME_BETA": FRAME_BETA,
        "n_contact_nodes": X.shape[1],
        "n_frames": X.shape[0],
        "n_dot_BNM_edges": len(dot_edges),
        "total_nonzero_J_edges": len(sorted_edges),
        "retained_J_edges": retained_edges,
        "J_strength_cutoff": cutoff,
        "n_generated_paths": len(paths_df),
        **stats,
    }

    print(
        f"Ligand {lig}: retained_J_edges={retained_edges}, cutoff={cutoff:.6g}, "
        f"pair_reach_frac={stats['pair_reach_frac']:.3f}, generated_paths={len(paths_df)}"
    )

    if len(paths_df) == 0:
        return None, diag, pd.DataFrame()

    node_energy, edge_energy, edge_J_obs = build_caches(X, h, J)
    windows = make_windows(X.shape[0], WINDOW_SIZE)

    # Full MD-frame Potts energy/probability landscape.
    # This is separate from path energy. High/low bins downstream are based on
    # this MD-window probability score.
    Eframe_full_potts = full_potts_frame_energy(X, h, J)
    Ewin_full_potts = window_mean(Eframe_full_potts, windows)
    md_window_probability_score = stable_prob(Ewin_full_potts, beta=FRAME_BETA)

    path_rows = []
    E_by_rank = {}
    Fmean_by_rank = {}
    Fsum_by_rank = {}

    for _, prow in paths_df.iterrows():
        rank = int(prow["rank"])
        path_nodes = parse_path(prow["path"])
        path_idx = [node_to_index(x, label_to_idx, idx_to_label) for x in path_nodes]

        Eframe = path_energy(path_idx, node_energy, edge_energy, energy_mode)
        frust = Jmin_frustration(path_idx, X, J, edge_J_obs)

        Ewin = window_mean(Eframe, windows)
        Fmean_win = window_mean(frust["Jmin_frustration_mean"], windows)
        Fsum_win = window_mean(frust["Jmin_frustration_sum"], windows)

        E_by_rank[rank] = Ewin
        Fmean_by_rank[rank] = Fmean_win
        Fsum_by_rank[rank] = Fsum_win

        path_rows.append({
            "ligand": lig,
            "EDGE_PERCENTILE": edge_percentile,
            "ENERGY_MODE": energy_mode,
            "BETA": beta_for_mode(energy_mode),
            "rank": rank,
            "source": prow["source"],
            "target": prow["target"],
            "path": prow["path"],
            "path_length_nodes": int(prow["path_length_nodes"]),
            "path_length_edges": int(prow["path_length_edges"]),
            "dijkstra_weight": float(prow["dijkstra_weight"]),
            "mean_energy": float(np.nanmean(Ewin)),
            "mean_Jmin_frustration_mean": float(np.nanmean(Fmean_win)),
            "mean_Jmin_frustration_sum": float(np.nanmean(Fsum_win)),
        })

    path_info = pd.DataFrame(path_rows)
    if path_info.empty:
        return None, diag, pd.DataFrame()

    beta = beta_for_mode(energy_mode)
    pmean = stable_prob(path_info["mean_energy"].values, beta=beta)
    path_info["mean_path_probability"] = pmean
    path_info["probability_order"] = (
        path_info["mean_path_probability"].rank(ascending=False, method="first").astype(int)
    )
    path_info = path_info.sort_values("probability_order").reset_index(drop=True)

    setting_payload = {
        "windows": windows,
        "path_info": path_info,
        "md_window_energy": Ewin_full_potts,
        "md_window_probability_score": md_window_probability_score,
        "E_by_rank": E_by_rank,
        "Fmean_by_rank": Fmean_by_rank,
        "Fsum_by_rank": Fsum_by_rank,
    }
    return setting_payload, diag, path_info


def analyze_topn_probability_conditioned(payload, lig, edge_percentile, energy_mode, top_n):
    path_info = payload["path_info"]
    windows = payload["windows"]
    md_window_energy = payload["md_window_energy"]
    md_window_probability_score = payload["md_window_probability_score"]
    E_by_rank = payload["E_by_rank"]
    Fmean_by_rank = payload["Fmean_by_rank"]
    Fsum_by_rank = payload["Fsum_by_rank"]

    selected, top_n_effective, prob_mass_cutoff = select_paths_by_probability_mass(path_info, top_n)
    selected = [r for r in selected if r in E_by_rank]

    if len(selected) < MIN_SELECTED_PATHS:
        return pd.DataFrame()

    Ewin = np.vstack([E_by_rank[r] for r in selected]).T
    Fmean = np.vstack([Fmean_by_rank[r] for r in selected]).T
    Fsum = np.vstack([Fsum_by_rank[r] for r in selected]).T

    # Window-level probability distribution over selected paths.
    beta = beta_for_mode(energy_mode)
    Pwin = np.vstack([stable_prob(Ewin[t, :], beta=beta) for t in range(Ewin.shape[0])])

    # Communication/path-derived weighted quantities. These are NOT used to
    # define high/low bins anymore.
    path_window_energy = np.nansum(Pwin * Ewin, axis=1)
    path_window_probability_score = stable_prob(path_window_energy, beta=beta)
    window_Fmean = np.nansum(Pwin * Fmean, axis=1)
    window_Fsum = np.nansum(Pwin * Fsum, axis=1)

    rows = []
    for wi, win in windows.iterrows():
        row = {
            "ligand": lig,
            "FRET": FRET[lig],
            "EDGE_PERCENTILE": edge_percentile,
            "ENERGY_MODE": energy_mode,
            "BETA": beta,
            "TOP_N": str(top_n),
            "PROB_MASS_CUTOFF": prob_mass_cutoff,
            "TOP_N_EFFECTIVE": int(top_n_effective),
            "window": int(win["window"]),
            "start_frame": int(win["start_frame"]),
            "stop_frame": int(win["stop_frame"]),
            "n_selected_paths": len(selected),
            # MD landscape quantities used for high/low binning
            "md_window_energy": float(md_window_energy[wi]),
            "md_window_probability_score": float(md_window_probability_score[wi]),
            # Backward-compatible aliases: these now refer to MD-frame/window landscape
            "window_energy": float(md_window_energy[wi]),
            "window_probability_score": float(md_window_probability_score[wi]),
            # Path-derived quantities kept separately for diagnostics
            "path_window_energy": float(path_window_energy[wi]),
            "path_window_probability_score": float(path_window_probability_score[wi]),
            "window_Jmin_frustration_mean": float(window_Fmean[wi]),
            "window_Jmin_frustration_sum": float(window_Fsum[wi]),
        }
        for k, rank in enumerate(selected):
            row[f"P_rank_{rank}"] = float(Pwin[wi, k])
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    try:
        out["probability_bin"] = pd.qcut(
            out["window_probability_score"],
            q=N_PROB_BINS,
            labels=np.arange(1, N_PROB_BINS + 1),
            duplicates="drop",
        ).astype(int)
    except Exception:
        return pd.DataFrame()

    return out


# =============================================================================
# BIN, CONTRAST, CORRELATION, ROBUSTNESS
# =============================================================================

def bin_summary(window_df):
    if window_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "ligand", "TOP_N", "PROB_MASS_CUTOFF", "probability_bin"]

    for key, sub in window_df.groupby(group_cols):
        edge_percentile, energy_mode, beta, lig, top_n, prob_mass_cutoff, prob_bin = key
        p_cols = [c for c in sub.columns if c.startswith("P_rank_")]
        P = sub[p_cols].values.astype(float) if p_cols else np.empty((0, 0))
        comm = communication_metrics(P)

        rows.append({
            "EDGE_PERCENTILE": edge_percentile,
            "ENERGY_MODE": energy_mode,
            "BETA": float(beta),
            "ligand": int(lig),
            "FRET": FRET[int(lig)],
            "TOP_N": str(top_n),
            "PROB_MASS_CUTOFF": float(prob_mass_cutoff) if pd.notna(prob_mass_cutoff) else np.nan,
            "probability_bin": int(prob_bin),
            "n_windows": int(len(sub)),
            "n_selected_paths": int(sub["n_selected_paths"].iloc[0]),
            "mean_probability_score": float(np.nanmean(sub["window_probability_score"])),
            "mean_energy": float(np.nanmean(sub["window_energy"])),
            "mean_md_probability_score": float(np.nanmean(sub["md_window_probability_score"])),
            "mean_md_energy": float(np.nanmean(sub["md_window_energy"])),
            "mean_path_probability_score": float(np.nanmean(sub["path_window_probability_score"])),
            "mean_path_energy": float(np.nanmean(sub["path_window_energy"])),
            "mean_Jmin_frustration_mean": float(np.nanmean(sub["window_Jmin_frustration_mean"])),
            "mean_Jmin_frustration_sum": float(np.nanmean(sub["window_Jmin_frustration_sum"])),
            **comm,
        })

    return pd.DataFrame(rows)


def contrast_summary(binned):
    if binned.empty:
        return pd.DataFrame()

    metric_cols = [
        "mean_probability_score",
        "mean_energy",
        "mean_md_probability_score",
        "mean_md_energy",
        "mean_path_probability_score",
        "mean_path_energy",
        "mean_Jmin_frustration_mean",
        "mean_Jmin_frustration_sum",
        "mean_path_corr",
        "strong_positive_fraction_gt_0p3",
        "strong_negative_fraction_lt_minus_0p3",
        "strong_pos_neg_ratio",
        "log_strong_pos_neg_ratio",
        "conflict_fraction",
        "positive_fraction",
        "negative_fraction",
        "pos_neg_ratio",
        "log_pos_neg_ratio",
        "spectral_effective_modes",
        "spectral_entropy",
        "top_eigen_fraction",
        "participation_ratio",
    ]

    rows = []
    group_cols = ["EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "ligand", "TOP_N", "PROB_MASS_CUTOFF"]

    for key, sub in binned.groupby(group_cols):
        edge_percentile, energy_mode, beta, lig, top_n, prob_mass_cutoff = key
        low = sub[sub["probability_bin"] == sub["probability_bin"].min()]
        high = sub[sub["probability_bin"] == sub["probability_bin"].max()]
        if len(low) == 0 or len(high) == 0:
            continue

        row = {
            "EDGE_PERCENTILE": edge_percentile,
            "ENERGY_MODE": energy_mode,
            "BETA": float(beta),
            "ligand": int(lig),
            "FRET": FRET[int(lig)],
            "TOP_N": str(top_n),
            "PROB_MASS_CUTOFF": float(prob_mass_cutoff) if pd.notna(prob_mass_cutoff) else np.nan,
            "TOP_N_EFFECTIVE_MEAN": float(sub["n_selected_paths"].mean()),
            "TOP_N_EFFECTIVE_MIN": int(sub["n_selected_paths"].min()),
            "TOP_N_EFFECTIVE_MAX": int(sub["n_selected_paths"].max()),
            "n_selected_paths": int(sub["n_selected_paths"].max()),
        }

        for m in metric_cols:
            lo = float(low[m].iloc[0]) if m in low else np.nan
            hi = float(high[m].iloc[0]) if m in high else np.nan
            row[f"{m}_low"] = lo
            row[f"{m}_high"] = hi
            row[f"{m}_high_minus_low"] = hi - lo
            row[f"{m}_high_over_low"] = hi / (lo + EPS)

        rows.append(row)

    return pd.DataFrame(rows)


def rank_order_string(sub, metric, descending=True):
    return ">".join(str(int(v)) for v in sub.sort_values(metric, ascending=not descending)["ligand"].values)


def correlate_all5(contrast):
    if contrast.empty:
        return pd.DataFrame(), pd.DataFrame()

    id_cols = {"EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "ligand", "FRET", "TOP_N", "PROB_MASS_CUTOFF", "TOP_N_EFFECTIVE_MEAN", "TOP_N_EFFECTIVE_MIN", "TOP_N_EFFECTIVE_MAX", "n_selected_paths"}
    metric_cols = [c for c in contrast.columns if c not in id_cols]

    rows = []
    diagnostics = []
    group_cols = ["EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "TOP_N", "PROB_MASS_CUTOFF"]

    for key, sub0 in contrast.groupby(group_cols):
        edge_percentile, energy_mode, beta, top_n, prob_mass_cutoff = key

        for metric in metric_cols:
            sub = sub0[["ligand", "FRET", metric]].copy()
            sub = sub[sub["ligand"].isin(LIGANDS)].copy()
            sub["is_finite"] = np.isfinite(sub[metric].astype(float))
            finite = sub[sub["is_finite"]].copy()
            ligs_present = sorted(finite["ligand"].astype(int).tolist())
            missing = [l for l in LIGANDS if l not in ligs_present]

            if REQUIRE_ALL_LIGANDS_FOR_CORR and len(finite) != len(LIGANDS):
                diagnostics.append({
                    "EDGE_PERCENTILE": edge_percentile,
                    "ENERGY_MODE": energy_mode,
                    "BETA": float(beta),
                    "TOP_N": str(top_n),
                    "PROB_MASS_CUTOFF": float(prob_mass_cutoff) if pd.notna(prob_mass_cutoff) else np.nan,
                    "metric": metric,
                    "n_ligands_used": len(finite),
                    "missing_or_nan_ligands": ",".join(str(x) for x in missing),
                })
                continue

            if len(finite) < MIN_LIGANDS_FOR_CORR:
                continue

            x = finite[metric].astype(float).values
            y = finite["FRET"].astype(float).values
            if np.nanstd(x) < EPS:
                diagnostics.append({
                    "EDGE_PERCENTILE": edge_percentile,
                    "ENERGY_MODE": energy_mode,
                    "BETA": float(beta),
                    "TOP_N": str(top_n),
                    "PROB_MASS_CUTOFF": float(prob_mass_cutoff) if pd.notna(prob_mass_cutoff) else np.nan,
                    "metric": metric,
                    "n_ligands_used": len(finite),
                    "missing_or_nan_ligands": "constant_metric",
                })
                continue

            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            order_desc = rank_order_string(finite, metric, descending=True)
            order_asc = rank_order_string(finite, metric, descending=False)

            rows.append({
                "EDGE_PERCENTILE": edge_percentile,
                "ENERGY_MODE": energy_mode,
                "BETA": float(beta),
                "TOP_N": str(top_n),
                "PROB_MASS_CUTOFF": float(prob_mass_cutoff) if pd.notna(prob_mass_cutoff) else np.nan,
                "metric": metric,
                "n_ligands_used": len(finite),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
                "abs_pearson_r": abs(float(pr.statistic)),
                "abs_spearman_rho": abs(float(sr.statistic)),
                "rank_order_high_to_low": order_desc,
                "rank_order_low_to_high": order_asc,
                "matches_expected_order_high_to_low": order_desc == ">".join(map(str, EXPECTED_EFFICACY_ORDER_HIGH_TO_LOW)),
                "matches_expected_order_low_to_high": order_asc == ">".join(map(str, EXPECTED_EFFICACY_ORDER_HIGH_TO_LOW)),
                "values_by_ligand": "; ".join(
                    f"{int(r.ligand)}:{getattr(r, metric):.6g}" for r in finite.itertuples(index=False)
                ),
            })

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def robustness_summary(corr):
    if corr.empty:
        return pd.DataFrame()

    expected_settings = len(EDGE_PERCENTILES) * len(TOP_N_LIST)
    rows = []

    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        pearson = sub["pearson_r"].astype(float).values
        spearman = sub["spearman_rho"].astype(float).values
        signs = np.sign(pearson[np.isfinite(pearson)])
        nonzero = signs[signs != 0]

        if len(nonzero) == 0:
            sign_consistency = np.nan
            dominant_sign = 0
        else:
            pos = np.sum(nonzero > 0)
            neg = np.sum(nonzero < 0)
            sign_consistency = max(pos, neg) / len(nonzero)
            dominant_sign = 1 if pos >= neg else -1

        rows.append({
            "ENERGY_MODE": energy_mode,
            "metric": metric,
            "n_settings": int(len(sub)),
            "expected_settings": int(expected_settings),
            "setting_coverage_fraction": float(len(sub) / max(expected_settings, 1)),
            "dominant_pearson_sign": int(dominant_sign),
            "pearson_sign_consistency": float(sign_consistency),
            "mean_pearson_r": float(np.nanmean(pearson)),
            "median_pearson_r": float(np.nanmedian(pearson)),
            "min_pearson_r": float(np.nanmin(pearson)),
            "max_pearson_r": float(np.nanmax(pearson)),
            "mean_abs_pearson_r": float(np.nanmean(np.abs(pearson))),
            "median_abs_pearson_r": float(np.nanmedian(np.abs(pearson))),
            "min_abs_pearson_r": float(np.nanmin(np.abs(pearson))),
            "std_pearson_r": float(np.nanstd(pearson)),
            "mean_spearman_rho": float(np.nanmean(spearman)),
            "median_spearman_rho": float(np.nanmedian(spearman)),
            "min_abs_spearman_rho": float(np.nanmin(np.abs(spearman))),
            "frac_abs_pearson_ge_0p7": float(np.mean(np.abs(pearson) >= 0.7)),
            "frac_abs_spearman_ge_0p8": float(np.mean(np.abs(spearman) >= 0.8)),
            "n_exact_expected_rank_high_to_low": int(sub["matches_expected_order_high_to_low"].sum()),
            "n_exact_expected_rank_low_to_high": int(sub["matches_expected_order_low_to_high"].sum()),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["robustness_score"] = (
            out["setting_coverage_fraction"].fillna(0)
            * out["pearson_sign_consistency"].fillna(0)
            * out["mean_abs_pearson_r"].fillna(0)
            * out["frac_abs_spearman_ge_0p8"].fillna(0)
        )
        out = out.sort_values("robustness_score", ascending=False).reset_index(drop=True)
    return out


def rank_stability_summary(corr):
    if corr.empty:
        return pd.DataFrame()
    rows = []
    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        vc = sub["rank_order_high_to_low"].value_counts()
        rows.append({
            "ENERGY_MODE": energy_mode,
            "metric": metric,
            "n_settings": int(len(sub)),
            "most_common_rank_order_high_to_low": vc.index[0],
            "most_common_rank_order_count": int(vc.iloc[0]),
            "rank_order_stability_fraction": float(vc.iloc[0] / max(len(sub), 1)),
            "unique_rank_orders": int(len(vc)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["rank_order_stability_fraction", "most_common_rank_order_count"], ascending=False)
    return out


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_heatmap(corr, metric, energy_mode):
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    top_ns = sorted(sub["TOP_N"].unique())
    edge_ps = sorted(sub["EDGE_PERCENTILE"].unique())
    mat = np.full((len(edge_ps), len(top_ns)), np.nan)

    for i, ep in enumerate(edge_ps):
        for j, tn in enumerate(top_ns):
            d = sub[(sub["EDGE_PERCENTILE"] == ep) & (sub["TOP_N"] == tn)]
            if len(d):
                mat[i, j] = d.iloc[0]["pearson_r"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_yticks(np.arange(len(edge_ps)))
    ax.set_yticklabels([str(x) for x in edge_ps])
    ax.set_xticks(np.arange(len(top_ns)))
    ax.set_xticklabels([str(x) for x in top_ns])
    ax.set_ylabel("J-edge percentile cutoff")
    ax.set_xlabel("TOP_N paths")
    ax.set_title(f"Pearson r with FRET\n{energy_mode}: {metric}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    plt.tight_layout()
    safe = metric.replace("/", "_")[:120]
    out = os.path.join(FIGDIR, f"heatmap_{energy_mode}_{safe}.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_path_generation(path_diag):
    if path_diag.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for lig in LIGANDS:
        d = path_diag[(path_diag["ligand"] == lig) & (path_diag["ENERGY_MODE"] == ENERGY_MODES[0])]
        if len(d):
            plt.plot(d["EDGE_PERCENTILE"], d["n_generated_paths"], marker="o", label=f"ligand {lig}")
    plt.gca().invert_xaxis()
    plt.xlabel("J-edge percentile cutoff")
    plt.ylabel("Generated paths")
    plt.title("Generated source→target paths after learned-J graph cutoff")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "generated_paths_vs_edge_percentile.png")
    plt.savefig(out, dpi=300)
    plt.close()




def safe_filename(text, max_len=160):
    """Make a filesystem-safe filename component."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def plot_all_robustness_scores(robust):
    """Horizontal bar plot of robustness score for every ENERGY_MODE/metric pair."""
    if robust.empty:
        return
    d = robust.sort_values("robustness_score", ascending=True).copy()
    labels = [f"{r.ENERGY_MODE} | {r.metric}" for r in d.itertuples(index=False)]
    h = max(6, 0.28 * len(d))
    plt.figure(figsize=(12, h))
    plt.barh(np.arange(len(d)), d["robustness_score"].values)
    plt.yticks(np.arange(len(d)), labels, fontsize=7)
    plt.xlabel("Robustness score")
    plt.title("Robustness score for all metrics")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "ALL_metrics_robustness_scores.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_metric_robustness_components(robust):
    """For each metric/mode, plot the ingredients of the robustness score."""
    if robust.empty:
        return
    comp_dir = os.path.join(FIGDIR, "robustness_components")
    os.makedirs(comp_dir, exist_ok=True)
    cols = [
        "setting_coverage_fraction",
        "pearson_sign_consistency",
        "mean_abs_pearson_r",
        "frac_abs_spearman_ge_0p8",
        "robustness_score",
    ]
    for _, r in robust.iterrows():
        vals = [float(r[c]) if pd.notna(r[c]) else np.nan for c in cols]
        labels = [
            "coverage",
            "sign consistency",
            "mean |Pearson r|",
            "frac |Spearman|≥0.8",
            "robustness",
        ]
        plt.figure(figsize=(7.5, 4))
        plt.bar(labels, vals)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title(f"Robustness components\n{r['ENERGY_MODE']}: {r['metric']}")
        plt.tight_layout()
        out = os.path.join(comp_dir, f"components_{safe_filename(r['ENERGY_MODE'])}_{safe_filename(r['metric'])}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_corr_heatmaps_for_all_metrics(corr):
    """Make Pearson-r heatmaps for every metric and ENERGY_MODE."""
    if corr.empty:
        return
    heat_dir = os.path.join(FIGDIR, "heatmaps_all_metrics")
    os.makedirs(heat_dir, exist_ok=True)

    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        top_ns = sorted(sub["TOP_N"].unique())
        edge_ps = sorted(sub["EDGE_PERCENTILE"].unique())
        mat = np.full((len(edge_ps), len(top_ns)), np.nan)
        for i, ep in enumerate(edge_ps):
            for j, tn in enumerate(top_ns):
                d = sub[(sub["EDGE_PERCENTILE"] == ep) & (sub["TOP_N"] == tn)]
                if len(d):
                    mat[i, j] = float(d.iloc[0]["pearson_r"])

        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(top_ns)), max(3.5, 0.55 * len(edge_ps))))
        im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_yticks(np.arange(len(edge_ps)))
        ax.set_yticklabels([str(x) for x in edge_ps])
        ax.set_xticks(np.arange(len(top_ns)))
        ax.set_xticklabels([str(x) for x in top_ns])
        ax.set_ylabel("J-edge percentile")
        ax.set_xlabel("TOP_N paths")
        ax.set_title(f"Pearson r with FRET\n{energy_mode}: {metric}", fontsize=9)
        for i in range(len(edge_ps)):
            for j in range(len(top_ns)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pearson r")
        plt.tight_layout()
        out = os.path.join(heat_dir, f"heatmap_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def _parse_values_by_ligand(values_str):
    out = {}
    for item in str(values_str).split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        try:
            out[int(float(k.strip()))] = float(v.strip())
        except Exception:
            continue
    return out


def plot_best_setting_scatter_for_all_metrics(corr):
    """For every metric/mode, plot the setting with the largest |Pearson r| against FRET."""
    if corr.empty:
        return
    scat_dir = os.path.join(FIGDIR, "best_setting_scatter_all_metrics")
    os.makedirs(scat_dir, exist_ok=True)

    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        sub = sub.copy()
        sub["abs_r"] = sub["pearson_r"].abs()
        row = sub.sort_values("abs_r", ascending=False).iloc[0]
        vals = _parse_values_by_ligand(row["values_by_ligand"])
        ligs = [lig for lig in LIGANDS if lig in vals]
        if len(ligs) < 2:
            continue
        x = np.asarray([vals[lig] for lig in ligs], dtype=float)
        y = np.asarray([FRET[lig] for lig in ligs], dtype=float)

        plt.figure(figsize=(5.5, 4.5))
        plt.scatter(x, y, s=55)
        for lig, xv, yv in zip(ligs, x, y):
            plt.text(xv, yv, f" L{lig}", fontsize=9, va="center")
        if len(x) >= 2 and np.nanstd(x) > EPS:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            plt.plot(xx, m * xx + b, linewidth=1)
        plt.xlabel(metric)
        plt.ylabel("FRET efficacy")
        plt.title(
            f"Best setting: {energy_mode}\n{metric}\n"
            f"EDGE={row['EDGE_PERCENTILE']}, TOP_N={row['TOP_N']}, r={row['pearson_r']:.3f}, ρ={row['spearman_rho']:.3f}",
            fontsize=9,
        )
        plt.tight_layout()
        out = os.path.join(scat_dir, f"scatter_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_best_setting_ligand_bars_for_all_metrics(corr):
    """For every metric/mode, bar plot ligand values at the best-|r| setting."""
    if corr.empty:
        return
    bar_dir = os.path.join(FIGDIR, "best_setting_ligand_bars_all_metrics")
    os.makedirs(bar_dir, exist_ok=True)

    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        sub = sub.copy()
        sub["abs_r"] = sub["pearson_r"].abs()
        row = sub.sort_values("abs_r", ascending=False).iloc[0]
        vals = _parse_values_by_ligand(row["values_by_ligand"])
        ligs = [lig for lig in LIGANDS if lig in vals]
        if len(ligs) == 0:
            continue
        y = [vals[lig] for lig in ligs]
        labels = [f"L{lig}" for lig in ligs]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, y)
        plt.xlabel("Ligand")
        plt.ylabel(metric)
        plt.title(
            f"Ligand values at best setting\n{energy_mode}: {metric}\n"
            f"EDGE={row['EDGE_PERCENTILE']}, TOP_N={row['TOP_N']}",
            fontsize=9,
        )
        plt.tight_layout()
        out = os.path.join(bar_dir, f"bars_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_topn_trends_for_all_metrics(corr):
    """For every metric/mode and edge percentile, plot Pearson r versus TOP_N."""
    if corr.empty:
        return
    trend_dir = os.path.join(FIGDIR, "topn_trends_all_metrics")
    os.makedirs(trend_dir, exist_ok=True)

    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        plt.figure(figsize=(6.5, 4.5))
        any_line = False
        for ep, d in sub.groupby("EDGE_PERCENTILE"):
            d = d.sort_values("TOP_N")
            if len(d):
                plt.plot(d["TOP_N"], d["pearson_r"], marker="o", label=f"EDGE {ep}")
                any_line = True
        if not any_line:
            plt.close()
            continue
        plt.axhline(0, linewidth=0.8)
        plt.ylim(-1.05, 1.05)
        plt.xlabel("TOP_N paths")
        plt.ylabel("Pearson r with FRET")
        plt.title(f"TOP_N sensitivity\n{energy_mode}: {metric}", fontsize=9)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        out = os.path.join(trend_dir, f"topn_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_distribution_summary(contrast):
    """For every metric column, make ligand-value box/point summaries across settings."""
    if contrast.empty:
        return
    dist_dir = os.path.join(FIGDIR, "metric_distributions_by_ligand")
    os.makedirs(dist_dir, exist_ok=True)
    id_cols = {"EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "ligand", "FRET", "TOP_N", "PROB_MASS_CUTOFF", "TOP_N_EFFECTIVE_MEAN", "TOP_N_EFFECTIVE_MIN", "TOP_N_EFFECTIVE_MAX", "n_selected_paths"}
    metric_cols = [c for c in contrast.columns if c not in id_cols]

    for metric in metric_cols:
        vals = contrast[["ligand", "ENERGY_MODE", metric]].copy()
        vals = vals[np.isfinite(vals[metric].astype(float))]
        if vals.empty:
            continue
        for energy_mode, d in vals.groupby("ENERGY_MODE"):
            plt.figure(figsize=(6.5, 4.2))
            data = [d[d["ligand"] == lig][metric].astype(float).values for lig in LIGANDS]
            positions = np.arange(1, len(LIGANDS) + 1)
            plt.boxplot(data, positions=positions, labels=[f"L{lig}" for lig in LIGANDS], showfliers=False)
            for pos, arr in zip(positions, data):
                if len(arr):
                    x = np.full(len(arr), pos, dtype=float) + np.random.normal(0, 0.035, size=len(arr))
                    plt.scatter(x, arr, s=16, alpha=0.7)
            plt.xlabel("Ligand")
            plt.ylabel(metric)
            plt.title(f"Distribution across settings\n{energy_mode}: {metric}", fontsize=9)
            plt.tight_layout()
            out = os.path.join(dist_dir, f"dist_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
            plt.savefig(out, dpi=300)
            plt.close()


def metric_base_dir(metric):
    """Create/return a separate folder for one metric."""
    d = os.path.join(FIGDIR, "by_metric", safe_filename(metric))
    os.makedirs(d, exist_ok=True)
    return d


def metric_mode_dir(metric, energy_mode):
    """Create/return a separate folder for one metric and one energy mode."""
    d = os.path.join(metric_base_dir(metric), safe_filename(energy_mode))
    os.makedirs(d, exist_ok=True)
    return d


def plot_metric_heatmap_to_metric_folder(corr, metric, energy_mode):
    """Pearson-r heatmap saved inside figures/by_metric/<metric>/<ENERGY_MODE>/."""
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    top_ns = sorted(sub["TOP_N"].unique())
    edge_ps = sorted(sub["EDGE_PERCENTILE"].unique())
    mat = np.full((len(edge_ps), len(top_ns)), np.nan)

    for i, ep in enumerate(edge_ps):
        for j, tn in enumerate(top_ns):
            d = sub[(sub["EDGE_PERCENTILE"] == ep) & (sub["TOP_N"] == tn)]
            if len(d):
                mat[i, j] = float(d.iloc[0]["pearson_r"])

    outdir = metric_mode_dir(metric, energy_mode)
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(top_ns)), max(3.8, 0.6 * len(edge_ps))))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_yticks(np.arange(len(edge_ps)))
    ax.set_yticklabels([str(x) for x in edge_ps])
    ax.set_xticks(np.arange(len(top_ns)))
    ax.set_xticklabels([str(x) for x in top_ns])
    ax.set_ylabel("J-edge percentile")
    ax.set_xlabel("TOP_N paths")
    ax.set_title(f"Pearson r with FRET\n{energy_mode}: {metric}", fontsize=9)

    for i in range(len(edge_ps)):
        for j in range(len(top_ns)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    plt.tight_layout()
    out = os.path.join(outdir, "heatmap_pearson_by_edge_and_topN.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_metric_topn_trend_to_metric_folder(corr, metric, energy_mode):
    """Pearson-r versus TOP_N saved inside metric folder."""
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    outdir = metric_mode_dir(metric, energy_mode)
    plt.figure(figsize=(6.5, 4.5))
    any_line = False

    for ep, d in sub.groupby("EDGE_PERCENTILE"):
        d = d.sort_values("TOP_N")
        if len(d):
            plt.plot(d["TOP_N"], d["pearson_r"], marker="o", label=f"EDGE {ep}")
            any_line = True

    if not any_line:
        plt.close()
        return

    plt.axhline(0, linewidth=0.8)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("TOP_N paths")
    plt.ylabel("Pearson r with FRET")
    plt.title(f"TOP_N sensitivity\n{energy_mode}: {metric}", fontsize=9)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = os.path.join(outdir, "topN_sensitivity_pearson.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_metric_all_setting_scatters(corr, metric, energy_mode):
    """One scatter plot for every EDGE_PERCENTILE/TOP_N setting for this metric/mode."""
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    outdir = os.path.join(metric_mode_dir(metric, energy_mode), "all_setting_scatters")
    os.makedirs(outdir, exist_ok=True)

    for _, row in sub.iterrows():
        vals = _parse_values_by_ligand(row["values_by_ligand"])
        ligs = [lig for lig in LIGANDS if lig in vals]
        if len(ligs) < 2:
            continue

        x = np.asarray([vals[lig] for lig in ligs], dtype=float)
        y = np.asarray([FRET[lig] for lig in ligs], dtype=float)

        plt.figure(figsize=(5.5, 4.5))
        plt.scatter(x, y, s=60)
        for lig, xv, yv in zip(ligs, x, y):
            plt.text(xv, yv, f" L{lig}", fontsize=9, va="center")

        if len(x) >= 2 and np.nanstd(x) > EPS:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            plt.plot(xx, m * xx + b, linewidth=1)

        plt.xlabel(metric)
        plt.ylabel("FRET efficacy")
        plt.title(
            f"{energy_mode}: {metric}\n"
            f"EDGE={row['EDGE_PERCENTILE']}, TOP_N={row['TOP_N']}, "
            f"r={row['pearson_r']:.3f}, rho={row['spearman_rho']:.3f}",
            fontsize=9,
        )
        plt.tight_layout()

        out = os.path.join(
            outdir,
            f"scatter_EDGE_{safe_filename(row['EDGE_PERCENTILE'])}_TOPN_{safe_filename(row['TOP_N'])}.png"
        )
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_all_setting_bars(corr, metric, energy_mode):
    """One ligand-value bar plot for every EDGE_PERCENTILE/TOP_N setting."""
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    outdir = os.path.join(metric_mode_dir(metric, energy_mode), "all_setting_ligand_bars")
    os.makedirs(outdir, exist_ok=True)

    for _, row in sub.iterrows():
        vals = _parse_values_by_ligand(row["values_by_ligand"])
        ligs = [lig for lig in LIGANDS if lig in vals]
        if len(ligs) == 0:
            continue

        labels = [f"L{lig}" for lig in ligs]
        y = [vals[lig] for lig in ligs]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, y)
        plt.xlabel("Ligand")
        plt.ylabel(metric)
        plt.title(
            f"{energy_mode}: {metric}\n"
            f"EDGE={row['EDGE_PERCENTILE']}, TOP_N={row['TOP_N']}",
            fontsize=9,
        )
        plt.tight_layout()

        out = os.path.join(
            outdir,
            f"bars_EDGE_{safe_filename(row['EDGE_PERCENTILE'])}_TOPN_{safe_filename(row['TOP_N'])}.png"
        )
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_topn_panel_scatters(corr, metric, energy_mode):
    """For each edge percentile, make a multi-panel figure containing all TOP_N scatter plots."""
    sub = corr[(corr["metric"] == metric) & (corr["ENERGY_MODE"] == energy_mode)].copy()
    if sub.empty:
        return

    outdir = metric_mode_dir(metric, energy_mode)

    for ep, d_edge in sub.groupby("EDGE_PERCENTILE"):
        d_edge = d_edge.sort_values("TOP_N")
        n = len(d_edge)
        if n == 0:
            continue

        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), squeeze=False)

        for ax in axes.ravel():
            ax.axis("off")

        for ax, (_, row) in zip(axes.ravel(), d_edge.iterrows()):
            ax.axis("on")
            vals = _parse_values_by_ligand(row["values_by_ligand"])
            ligs = [lig for lig in LIGANDS if lig in vals]
            if len(ligs) < 2:
                ax.set_title(f"TOP_N={row['TOP_N']}\ninsufficient data")
                continue

            x = np.asarray([vals[lig] for lig in ligs], dtype=float)
            y = np.asarray([FRET[lig] for lig in ligs], dtype=float)

            ax.scatter(x, y, s=55)
            for lig, xv, yv in zip(ligs, x, y):
                ax.text(xv, yv, f"L{lig}", fontsize=8, va="center")

            if len(x) >= 2 and np.nanstd(x) > EPS:
                m, b = np.polyfit(x, y, 1)
                xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xx, m * xx + b, linewidth=1)

            ax.set_xlabel(metric, fontsize=8)
            ax.set_ylabel("FRET efficacy", fontsize=8)
            ax.set_title(
                f"TOP_N={row['TOP_N']}\n"
                f"r={row['pearson_r']:.2f}, rho={row['spearman_rho']:.2f}",
                fontsize=9,
            )

        fig.suptitle(f"{energy_mode}: {metric}\nEDGE_PERCENTILE={ep} — all TOP_N settings", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = os.path.join(outdir, f"panel_all_TOPN_EDGE_{safe_filename(ep)}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_robustness_to_metric_folder(robust, metric):
    """For one metric, plot robustness scores by energy mode and components in its own folder."""
    sub = robust[robust["metric"] == metric].copy()
    if sub.empty:
        return

    outdir = metric_base_dir(metric)

    d = sub.sort_values("robustness_score", ascending=True)
    plt.figure(figsize=(7, max(3.5, 0.45 * len(d))))
    plt.barh(d["ENERGY_MODE"], d["robustness_score"])
    plt.xlabel("Robustness score")
    plt.ylabel("Energy mode")
    plt.title(f"Robustness by energy mode\n{metric}", fontsize=10)
    plt.tight_layout()
    out = os.path.join(outdir, "robustness_by_energy_mode.png")
    plt.savefig(out, dpi=300)
    plt.close()

    comp_cols = [
        "setting_coverage_fraction",
        "pearson_sign_consistency",
        "mean_abs_pearson_r",
        "frac_abs_spearman_ge_0p8",
        "robustness_score",
    ]

    for _, row in sub.iterrows():
        labels = ["coverage", "sign consistency", "mean |r|", "frac |rho|≥0.8", "robustness"]
        vals = [float(row[c]) if pd.notna(row[c]) else np.nan for c in comp_cols]

        plt.figure(figsize=(7.5, 4))
        plt.bar(labels, vals)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title(f"Robustness components\n{row['ENERGY_MODE']}: {metric}", fontsize=9)
        plt.tight_layout()
        out = os.path.join(outdir, f"robustness_components_{safe_filename(row['ENERGY_MODE'])}.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_distributions_to_metric_folder(contrast, metric):
    """For one metric, make ligand distributions across settings and energy modes."""
    if contrast.empty or metric not in contrast.columns:
        return

    outdir = metric_base_dir(metric)
    vals = contrast[["ligand", "ENERGY_MODE", "TOP_N", "EDGE_PERCENTILE", metric]].copy()
    vals = vals[np.isfinite(vals[metric].astype(float))]
    if vals.empty:
        return

    for energy_mode, d in vals.groupby("ENERGY_MODE"):
        mode_dir = metric_mode_dir(metric, energy_mode)
        plt.figure(figsize=(6.5, 4.2))
        data = [d[d["ligand"] == lig][metric].astype(float).values for lig in LIGANDS]
        positions = np.arange(1, len(LIGANDS) + 1)
        plt.boxplot(data, positions=positions, labels=[f"L{lig}" for lig in LIGANDS], showfliers=False)

        rng = np.random.default_rng(12345)
        for pos, arr in zip(positions, data):
            if len(arr):
                x = np.full(len(arr), pos, dtype=float) + rng.normal(0, 0.035, size=len(arr))
                plt.scatter(x, arr, s=18, alpha=0.7)

        plt.xlabel("Ligand")
        plt.ylabel(metric)
        plt.title(f"Distribution across all settings\n{energy_mode}: {metric}", fontsize=9)
        plt.tight_layout()
        out = os.path.join(mode_dir, "distribution_across_all_settings_by_ligand.png")
        plt.savefig(out, dpi=300)
        plt.close()


def plot_all_figures(corr, robust, contrast, path_diag):
    """Generate all figures organized as figures/by_metric/<metric>/<energy_mode>/.

    For every metric and every energy mode, this writes:
      - heatmap_pearson_by_edge_and_topN.png
      - topN_sensitivity_pearson.png
      - panel_all_TOPN_EDGE_<edge>.png
      - all_setting_scatters/scatter_EDGE_<edge>_TOPN_<topn>.png
      - all_setting_ligand_bars/bars_EDGE_<edge>_TOPN_<topn>.png
      - distribution_across_all_settings_by_ligand.png
      - robustness components and robustness by energy mode
    """
    os.makedirs(FIGDIR, exist_ok=True)
    plot_path_generation(path_diag)

    # Keep the global overview figures too.
    plot_all_robustness_scores(robust)
    plot_metric_robustness_components(robust)
    plot_corr_heatmaps_for_all_metrics(corr)
    plot_best_setting_scatter_for_all_metrics(corr)
    plot_best_setting_ligand_bars_for_all_metrics(corr)
    plot_topn_trends_for_all_metrics(corr)
    plot_metric_distribution_summary(contrast)

    if corr.empty:
        return

    all_metrics = sorted(corr["metric"].dropna().unique().tolist())

    for metric in all_metrics:
        plot_metric_robustness_to_metric_folder(robust, metric)
        plot_metric_distributions_to_metric_folder(contrast, metric)

        for energy_mode in sorted(corr.loc[corr["metric"] == metric, "ENERGY_MODE"].dropna().unique()):
            plot_metric_heatmap_to_metric_folder(corr, metric, energy_mode)
            plot_metric_topn_trend_to_metric_folder(corr, metric, energy_mode)
            plot_metric_topn_panel_scatters(corr, metric, energy_mode)
            plot_metric_all_setting_scatters(corr, metric, energy_mode)
            plot_metric_all_setting_bars(corr, metric, energy_mode)




def probability_mass_selection_summary(generated_paths):
    """Summarize how many paths are needed to reach each probability-mass cutoff."""
    if generated_paths.empty or "mean_path_probability" not in generated_paths.columns:
        return pd.DataFrame()

    rows = []
    group_cols = ["EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "ligand"]
    for key, sub in generated_paths.groupby(group_cols):
        edge_percentile, energy_mode, beta, lig = key
        d = sub.sort_values("mean_path_probability", ascending=False).copy()
        probs = d["mean_path_probability"].astype(float).values
        probs = probs / (np.nansum(probs) + EPS)
        cum = np.cumsum(probs)

        row = {
            "EDGE_PERCENTILE": edge_percentile,
            "ENERGY_MODE": energy_mode,
            "BETA": float(beta),
            "ligand": int(lig),
            "n_paths_total": int(len(d)),
        }
        for cutoff in PROB_MASS_CUTOFFS:
            idx = int(np.searchsorted(cum, cutoff, side="left"))
            row[f"N_for_{int(100*cutoff)}pct"] = int(min(idx + 1, len(d)))
        for n in [25, 50, 100, 250, 500]:
            row[f"top{n}_cumulative_probability"] = float(cum[min(n, len(cum)) - 1]) if len(cum) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def _parse_int_list(text):
    """Parse comma-separated integers, e.g. '0,1,2'."""
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_float_list(text):
    """Parse comma-separated floats, e.g. '98,97.5,95'."""
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_str_list(text):
    """Parse comma-separated strings, e.g. 'HJ,J_ONLY'."""
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_fret_map(text):
    """Parse ligand:FRET pairs, e.g. '0:0.018,1:0.080'."""
    out = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid FRET entry: {item}. Expected ligand:value.")
        lig, val = item.split(":", 1)
        out[int(lig.strip())] = float(val.strip())
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run probability-conditioned Potts+BNM communication analysis using "
            "learned h/J parameters and Dijkstra-generated communication paths."
        )
    )

    parser.add_argument("--base", default=BASE, help="Base directory containing ligand folders.")
    parser.add_argument("--contact-pattern", default=CONTACT_PATTERN,
                        help="Pattern for discretized contact/state CSV files. Use {lig} for ligand index.")
    parser.add_argument("--h-pattern", default=H_PATTERN,
                        help="Pattern for learned h parameter .npy files. Use {lig} for ligand index.")
    parser.add_argument("--j-pattern", default=J_PATTERN,
                        help="Pattern for learned J parameter .npy files. Use {lig} for ligand index.")
    parser.add_argument("--dot-pattern", default=DOT_PATTERN,
                        help="Pattern for optional BaNDyT .dot files. Use {lig} for ligand index.")
    parser.add_argument("--outdir", default=None,
                        help="Output directory. Default: BASE/REGENERATED_PATHS_MDframe_potts_probability_pos_neg_balance_HJ_JONLY")

    parser.add_argument("--ligands", default=",".join(map(str, LIGANDS)),
                        help="Comma-separated ligand indices, e.g. 0,1,2,3,4.")
    parser.add_argument("--fret", default=",".join(f"{k}:{v}" for k, v in FRET.items()),
                        help="Comma-separated ligand:FRET pairs, e.g. 0:0.018,1:0.080.")
    parser.add_argument("--sources", default=",".join(map(str, SOURCE_NODES)),
                        help="Comma-separated source node labels/indices.")
    parser.add_argument("--targets", default=",".join(map(str, TARGET_NODES)),
                        help="Comma-separated target node labels/indices.")

    parser.add_argument("--edge-percentiles", default=",".join(map(str, EDGE_PERCENTILES)),
                        help="Comma-separated J-strength percentile cutoffs, e.g. 98,97.5,95.")
    parser.add_argument("--energy-modes", default=",".join(ENERGY_MODES),
                        help="Comma-separated energy modes: HJ,J_ONLY,HJ_MEAN,J_ONLY_MEAN.")
    parser.add_argument("--prob-mass-cutoffs", default=",".join(map(str, PROB_MASS_CUTOFFS)),
                        help="Comma-separated cumulative path-probability cutoffs, e.g. 0.80,0.90,0.95.")

    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--n-prob-bins", type=int, default=N_PROB_BINS)
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS)
    parser.add_argument("--paths-per-source", type=int, default=PATHS_PER_SOURCE)
    parser.add_argument("--paths-per-target", type=int, default=PATHS_PER_TARGET)
    parser.add_argument("--targets-per-source", type=int, default=TARGETS_PER_SOURCE)
    parser.add_argument("--min-path-len", type=int, default=MIN_PATH_LEN)
    parser.add_argument("--max-path-len", type=int, default=MAX_PATH_LEN)

    parser.add_argument("--frame-beta", type=float, default=FRAME_BETA)
    parser.add_argument("--pos-thresh", type=float, default=POS_THRESH)
    parser.add_argument("--neg-thresh", type=float, default=NEG_THRESH)

    return parser.parse_args(argv)


def configure_from_args(args):
    """Update module-level settings from command-line arguments."""
    global BASE, CONTACT_PATTERN, H_PATTERN, J_PATTERN, DOT_PATTERN, OUTDIR, FIGDIR
    global LIGANDS, FRET, SOURCE_NODES, TARGET_NODES
    global EDGE_PERCENTILES, ENERGY_MODES, PROB_MASS_CUTOFFS, TOP_N_LIST
    global WINDOW_SIZE, N_PROB_BINS, MAX_PATHS, PATHS_PER_SOURCE, PATHS_PER_TARGET
    global TARGETS_PER_SOURCE, MIN_PATH_LEN, MAX_PATH_LEN
    global FRAME_BETA, POS_THRESH, NEG_THRESH, MIN_LIGANDS_FOR_CORR

    BASE = args.base
    CONTACT_PATTERN = args.contact_pattern
    H_PATTERN = args.h_pattern
    J_PATTERN = args.j_pattern
    DOT_PATTERN = args.dot_pattern

    OUTDIR = args.outdir or os.path.join(
        BASE,
        "REGENERATED_PATHS_MDframe_potts_probability_pos_neg_balance_HJ_JONLY",
    )
    FIGDIR = os.path.join(OUTDIR, "figures")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)

    LIGANDS = _parse_int_list(args.ligands)
    FRET = _parse_fret_map(args.fret)
    SOURCE_NODES = _parse_str_list(args.sources)
    TARGET_NODES = _parse_str_list(args.targets)

    EDGE_PERCENTILES = _parse_float_list(args.edge_percentiles)
    ENERGY_MODES = _parse_str_list(args.energy_modes)
    PROB_MASS_CUTOFFS = _parse_float_list(args.prob_mass_cutoffs)
    TOP_N_LIST = [f"PMASS_{int(100 * x)}" for x in PROB_MASS_CUTOFFS]

    WINDOW_SIZE = args.window_size
    N_PROB_BINS = args.n_prob_bins
    MAX_PATHS = args.max_paths
    PATHS_PER_SOURCE = args.paths_per_source
    PATHS_PER_TARGET = args.paths_per_target
    TARGETS_PER_SOURCE = args.targets_per_source
    MIN_PATH_LEN = args.min_path_len
    MAX_PATH_LEN = args.max_path_len

    FRAME_BETA = args.frame_beta
    POS_THRESH = args.pos_thresh
    NEG_THRESH = args.neg_thresh

    MIN_LIGANDS_FOR_CORR = len(LIGANDS)


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    configure_from_args(args)

    all_path_info = []
    all_window_rows = []
    all_diag = []

    for edge_percentile in EDGE_PERCENTILES:
        for energy_mode in ENERGY_MODES:
            print(f"\n=== EDGE_PERCENTILE={edge_percentile}, ENERGY_MODE={energy_mode} ===")
            for lig in LIGANDS:
                payload, diag, path_info = compute_ligand_setting(lig, edge_percentile, energy_mode)
                all_diag.append(diag)

                if path_info is not None and len(path_info):
                    all_path_info.append(path_info)

                if payload is None:
                    continue

                for top_n in TOP_N_LIST:
                    out = analyze_topn_probability_conditioned(
                        payload=payload,
                        lig=lig,
                        edge_percentile=edge_percentile,
                        energy_mode=energy_mode,
                        top_n=top_n,
                    )
                    if len(out):
                        all_window_rows.append(out)

    path_diag = pd.DataFrame(all_diag)
    path_diag.to_csv(os.path.join(OUTDIR, "path_generation_diagnostics.csv"), index=False)
    plot_path_generation(path_diag)

    if all_path_info:
        generated_paths = pd.concat(all_path_info, ignore_index=True)
    else:
        generated_paths = pd.DataFrame()
    generated_paths.to_csv(os.path.join(OUTDIR, "generated_paths_all_settings.csv"), index=False)

    path_mass_summary = probability_mass_selection_summary(generated_paths)
    path_mass_summary.to_csv(os.path.join(OUTDIR, "path_probability_mass_selection_summary.csv"), index=False)

    if not all_window_rows:
        raise RuntimeError("No window analyses were produced. Relax EDGE_PERCENTILES or check graph connectivity/path generation.")

    window_df = pd.concat(all_window_rows, ignore_index=True)
    window_df.to_csv(os.path.join(OUTDIR, "window_metrics_all_settings.csv"), index=False)

    binned = bin_summary(window_df)
    binned.to_csv(os.path.join(OUTDIR, "binned_metrics_all_settings.csv"), index=False)

    contrast = contrast_summary(binned)
    contrast.to_csv(os.path.join(OUTDIR, "contrast_metrics_all_settings.csv"), index=False)

    corr, diagnostics = correlate_all5(contrast)
    corr.to_csv(os.path.join(OUTDIR, "all5_correlations_all_settings.csv"), index=False)
    diagnostics.to_csv(os.path.join(OUTDIR, "incomplete_metric_diagnostics.csv"), index=False)

    robust = robustness_summary(corr)
    robust.to_csv(os.path.join(OUTDIR, "metric_robustness_summary.csv"), index=False)

    rank_stability = rank_stability_summary(corr)
    rank_stability.to_csv(os.path.join(OUTDIR, "metric_rank_stability_summary.csv"), index=False)

    # Generate all figures for every metric, not only FOCUS_METRICS.
    plot_all_figures(corr, robust, contrast, path_diag)

    print("\nSaved outputs in:")
    print(OUTDIR)

    print("\nKey tables:")
    for fname in [
        "path_generation_diagnostics.csv",
        "generated_paths_all_settings.csv",
        "path_probability_mass_selection_summary.csv",
        "window_metrics_all_settings.csv",
        "binned_metrics_all_settings.csv",
        "contrast_metrics_all_settings.csv",
        "all5_correlations_all_settings.csv",
        "incomplete_metric_diagnostics.csv",
        "metric_robustness_summary.csv",
        "metric_rank_stability_summary.csv",
    ]:
        print(os.path.join(OUTDIR, fname))

    if len(corr):
        print("\nTop individual all-5 correlations, but DO NOT cherry-pick these alone:")
        print(
            corr.sort_values("abs_pearson_r", ascending=False)[
                [
                    "EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "TOP_N", "PROB_MASS_CUTOFF", "metric",
                    "n_ligands_used", "pearson_r", "spearman_rho",
                    "rank_order_high_to_low", "values_by_ligand",
                ]
            ].head(30).to_string(index=False)
        )

    if len(robust):
        print("\nMost robust metrics across edge cutoffs and TOP_N:")
        print(
            robust[
                [
                    "ENERGY_MODE", "metric", "n_settings", "expected_settings",
                    "setting_coverage_fraction", "robustness_score",
                    "pearson_sign_consistency", "mean_abs_pearson_r",
                    "min_abs_pearson_r", "frac_abs_spearman_ge_0p8",
                    "median_pearson_r", "std_pearson_r",
                ]
            ].head(30).to_string(index=False)
        )

    if len(diagnostics):
        print(f"\nWARNING: incomplete/constant metric groups recorded: {len(diagnostics)}")
        print("See incomplete_metric_diagnostics.csv")
    else:
        print("\nAll reported correlation groups used all 5 ligands and finite values.")


if __name__ == "__main__":
    main()
