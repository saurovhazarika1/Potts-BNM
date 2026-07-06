#!/usr/bin/env python3
"""
energy_spectrum_analysis.py

Purpose
-------
Analyze the communication-path energy spectrum separately in:

    HIGH-probability MD regions = LOW full-Potts-energy MD windows
    LOW-probability MD regions  = HIGH full-Potts-energy MD windows

This script is designed to test whether efficacy is associated with
reorganization of the communication-energy spectrum, rather than only with
mean path-energy shifts.

Key idea
--------
1. Regenerate ligand -> G-peptide paths from a learned-J graph using:

       S_ij = ||J_ij||_F
       cost_ij = Jmax - S_ij + eps

   so Dijkstra/Yen-style shortest paths prefer strong learned couplings.

2. For every MD frame, compute the FULL Potts/BNM frame energy:

       E_MD(t) = -sum_i h_i[x_i(t)]
                 -sum_{i<j} J_ij[x_i(t), x_j(t)]

   This is independent of path energies.

3. Convert MD-window energies into MD-window probabilities:

       P_MD(window) proportional to exp(-FRAME_BETA * E_MD_window)

   or, equivalently, use this probability ranking to define bins.

4. Define:

       high region = highest MD-probability bin = lowest MD-energy windows
       low region  = lowest MD-probability bin  = highest MD-energy windows

5. Within those two MD regions, collect path energies and compute spectral
   descriptors of the path-energy distribution:

       mean_E_high / mean_E_low
       std_E_high / std_E_low / std_E_high_over_low
       delta_mean_E
       abs_delta_mean_E
       energy percentiles: E05, E10, E25, E50, E75, E90, E95
       IQR, tail width, central width
       energy entropy from histogram P(E)
       effective number of energy bins = exp(entropy)
       energy participation ratio from histogram probabilities
       bimodality coefficient
       skewness, kurtosis
       low_energy_tail_fraction
       high_energy_tail_fraction
       tail_asymmetry = low_energy_tail_fraction - high_energy_tail_fraction
       tail_ratio = low_energy_tail_fraction / high_energy_tail_fraction
       tail_log_ratio = log(low_energy_tail_fraction / high_energy_tail_fraction)
       tail_polarization = low_energy_tail_fraction + high_energy_tail_fraction

6. Correlate all metrics with ligand efficacy/FRET across all five ligands.
7. Produce robustness summaries across ENERGY_MODE and PMASS cutoffs.

Outputs
-------
OUTDIR/
  path_generation_diagnostics.csv
  generated_paths_all_settings.csv
  window_energy_region_table.csv
  energy_spectrum_metrics_all_settings.csv
  all5_correlations_energy_spectrum.csv
  metric_robustness_energy_spectrum.csv
  metric_rank_stability_energy_spectrum.csv
  figures/

Important naming
----------------
In all output metrics:

    *_high = measured in HIGH MD-probability regions = LOW MD-energy windows
    *_low  = measured in LOW MD-probability regions  = HIGH MD-energy windows

So do NOT interpret *_low as low energy. It means low probability.
"""

import os
import re
import ast
import math
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, skew, kurtosis

try:
    from scipy.stats import wasserstein_distance
except Exception:
    wasserstein_distance = None

try:
    from scipy.spatial.distance import jensenshannon
except Exception:
    jensenshannon = None


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE = "MASTER_temporal_MD_contact_path_analysis_balancedMI_K500"

CONTACT_PATTERN = "ligand_{lig}/bandyt/rec_lig_Gpep_lig_{lig}_ds20.csv"
H_PATTERN = "ligand_{lig}/bandyt/potts_md_bn_elastic_graph_output/iter_0150_h.npy"
J_PATTERN = "ligand_{lig}/bandyt/potts_md_bn_elastic_graph_output/iter_0150_J.npy"
DOT_PATTERN = "ligand_{lig}/bandyt/rec_lig_Gpep_ds20rendering.dot"  # diagnostic only

OUTDIR = os.path.join(BASE, "MD_REGION_COMMUNICATION_ENERGY_TAIL_ASYMMETRY")
FIGDIR = os.path.join(OUTDIR, "figures")

LIGANDS = [0, 1, 2, 3, 4]
FRET = {0: 0.018, 1: 0.080, 2: 0.060, 3: 0.032, 4: 0.030}
EXPECTED_EFFICACY_ORDER_HIGH_TO_LOW = [1, 2, 3, 4, 0]

SOURCE_NODES = [286]
TARGET_NODES = [287]

EDGE_PERCENTILES = [98.0]
# For broader robustness later:
# EDGE_PERCENTILES = [98.0, 97.5, 97.0, 96.0, 95.0, 94.0, 93.0, 90.0]

ENERGY_MODES = ["HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"]
PROB_MASS_CUTOFFS = [0.80, 0.90, 0.95]
TOP_N_LIST = [f"PMASS_{int(100*x)}" for x in PROB_MASS_CUTOFFS]

MAX_PATHS = 1000
PATHS_PER_SOURCE = 1000
PATHS_PER_TARGET = 1000
TARGETS_PER_SOURCE = 25
MIN_PATH_LEN = 2
MAX_PATH_LEN = 14

WINDOW_SIZE = 100
N_PROB_BINS = 5
MIN_SELECTED_PATHS = 3
MIN_REGION_SAMPLES = 10

# These are effective scaling parameters for path probabilities/ranking.
# They are not physical temperatures.
BETA_BY_ENERGY_MODE = {
    "HJ": 2.0,
    "HJ_MEAN": 50.0,
    "J_ONLY": 25.0,
    "J_ONLY_MEAN": 200.0,
}
DEFAULT_BETA = 1.0

# Separate beta for MD-frame/window probability landscape.
# If you want to avoid beta sensitivity, you can set USE_ENERGY_RANK_FOR_MD_BINS=True.
FRAME_BETA = 1.0
USE_ENERGY_RANK_FOR_MD_BINS = False

# Histogram settings for energy-spectrum metrics.
N_ENERGY_BINS = 30
TAIL_QUANTILE = 0.10
EPS = 1e-12

REQUIRE_ALL_LIGANDS_FOR_CORR = True
MIN_LIGANDS_FOR_CORR = len(LIGANDS)


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


def safe_filename(text, max_len=160):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


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
    if isinstance(top_n_label, str) and top_n_label.upper().startswith("PMASS_"):
        return float(top_n_label.split("_", 1)[1]) / 100.0
    return None


def select_paths_by_probability_mass(path_info, top_n_label):
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
# GRAPH AND PATH GENERATION
# =============================================================================

def all_J_edges_sorted(J, idx_to_label):
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
# ENERGY FUNCTIONS
# =============================================================================

def full_potts_frame_energy(X, h, J):
    """Full MD-frame Potts energy over all nodes and all nonzero J tensors."""
    X = np.asarray(X, dtype=int)
    nframes, n_nodes = X.shape
    E = np.zeros(nframes, dtype=float)

    for i in range(n_nodes):
        E -= h[i, X[:, i]]

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

    return node_energy, edge_energy


def path_energy(path_idx, node_energy, edge_energy, energy_mode):
    mode = energy_mode.upper()
    if mode not in {"HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"}:
        raise ValueError("energy_mode must be HJ, J_ONLY, HJ_MEAN, or J_ONLY_MEAN")

    E = np.zeros_like(node_energy(path_idx[0]), dtype=float)

    if mode in {"HJ", "HJ_MEAN"}:
        for i in path_idx:
            E += node_energy(i)

    if mode in {"HJ", "J_ONLY", "HJ_MEAN", "J_ONLY_MEAN"}:
        for i, j in zip(path_idx[:-1], path_idx[1:]):
            E += edge_energy(i, j)

    if mode == "HJ_MEAN":
        denom = len(path_idx) + max(len(path_idx) - 1, 1)
        E = E / float(denom)
    elif mode == "J_ONLY_MEAN":
        denom = max(len(path_idx) - 1, 1)
        E = E / float(denom)

    return E


# =============================================================================
# ENERGY-SPECTRUM METRICS
# =============================================================================

def histogram_probabilities(x, bins=N_ENERGY_BINS, value_range=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(x, bins=bins, range=value_range, density=False)
    p = counts.astype(float) / (np.sum(counts) + EPS)
    return p, edges


def shannon_entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p + EPS)))


def participation_ratio_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    return float(1.0 / (np.sum(p ** 2) + EPS))


def bimodality_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 or np.nanstd(x) < EPS:
        return np.nan
    g = float(skew(x, bias=False))
    # scipy kurtosis with fisher=False gives Pearson kurtosis, normal=3.
    k = float(kurtosis(x, fisher=False, bias=False))
    if not np.isfinite(k) or abs(k) < EPS:
        return np.nan
    # Adjusted bimodality coefficient sometimes uses finite-size correction.
    return float((g * g + 1.0) / (k + EPS))


def free_energy_from_samples(E, beta):
    E = np.asarray(E, dtype=float)
    E = E[np.isfinite(E)]
    if len(E) == 0:
        return np.nan
    logw = -float(beta) * E
    m = np.max(logw)
    logZ = m + np.log(np.sum(np.exp(logw - m)) + EPS)
    return float(-logZ / (float(beta) + EPS))


def energy_spectrum_metrics(E_high, E_low, beta=DEFAULT_BETA, prefix=""):
    """Compute distribution-level energy metrics for high/low MD-probability regions.

    high = high MD probability = low MD energy
    low  = low MD probability = high MD energy
    """
    E_high = np.asarray(E_high, dtype=float)
    E_low = np.asarray(E_low, dtype=float)
    E_high = E_high[np.isfinite(E_high)]
    E_low = E_low[np.isfinite(E_low)]

    out = {
        "n_energy_samples_high": int(len(E_high)),
        "n_energy_samples_low": int(len(E_low)),
    }

    if len(E_high) < MIN_REGION_SAMPLES or len(E_low) < MIN_REGION_SAMPLES:
        # Fill all expected fields with NaN.
        names = [
            "mean_E", "std_E", "E05", "E10", "E25", "E50", "E75", "E90", "E95",
            "IQR_E", "tail_width_E", "central_width_E", "energy_entropy", "energy_effective_bins",
            "energy_participation_ratio", "energy_skewness", "energy_kurtosis", "bimodality_coefficient",
            "low_energy_tail_fraction", "high_energy_tail_fraction",
            "tail_asymmetry", "tail_ratio", "tail_log_ratio", "tail_polarization", "F"
        ]
        for region in ["high", "low"]:
            for name in names:
                out[f"{name}_{region}"] = np.nan
        contrast_names = [
            "delta_mean_E", "abs_delta_mean_E", "std_E_high_over_low", "std_E_high_minus_low",
            "energy_entropy_high_minus_low", "energy_entropy_high_over_low",
            "energy_effective_bins_high_minus_low", "energy_effective_bins_high_over_low",
            "energy_participation_ratio_high_minus_low", "energy_participation_ratio_high_over_low",
            "bimodality_coefficient_high_minus_low", "bimodality_coefficient_high_over_low",
            "tail_asymmetry_high_minus_low", "tail_asymmetry_high_over_low",
            "tail_ratio_high_minus_low", "tail_ratio_high_over_low",
            "tail_log_ratio_high_minus_low", "tail_log_ratio_high_over_low",
            "tail_polarization_high_minus_low", "tail_polarization_high_over_low",
            "delta_F", "abs_delta_F", "cohens_d", "JS_divergence", "wasserstein_distance"
        ]
        for name in contrast_names:
            out[name] = np.nan
        return out

    # Shared histogram range so high/low distributions are comparable.
    allE = np.concatenate([E_high, E_low])
    lo, hi = np.nanmin(allE), np.nanmax(allE)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < EPS:
        value_range = None
    else:
        pad = 0.02 * (hi - lo)
        value_range = (lo - pad, hi + pad)

    p_high, _ = histogram_probabilities(E_high, bins=N_ENERGY_BINS, value_range=value_range)
    p_low, _ = histogram_probabilities(E_low, bins=N_ENERGY_BINS, value_range=value_range)

    def region_metrics(x, region, p):
        q05, q10, q25, q50, q75, q90, q95 = np.nanpercentile(x, [5, 10, 25, 50, 75, 90, 95])
        H = shannon_entropy_from_probs(p)
        out[f"mean_E_{region}"] = float(np.nanmean(x))
        out[f"std_E_{region}"] = float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan
        out[f"E05_{region}"] = float(q05)
        out[f"E10_{region}"] = float(q10)
        out[f"E25_{region}"] = float(q25)
        out[f"E50_{region}"] = float(q50)
        out[f"E75_{region}"] = float(q75)
        out[f"E90_{region}"] = float(q90)
        out[f"E95_{region}"] = float(q95)
        out[f"IQR_E_{region}"] = float(q75 - q25)
        out[f"tail_width_E_{region}"] = float(q95 - q05)
        out[f"central_width_E_{region}"] = float(q90 - q10)
        out[f"energy_entropy_{region}"] = H
        out[f"energy_effective_bins_{region}"] = float(np.exp(H)) if np.isfinite(H) else np.nan
        out[f"energy_participation_ratio_{region}"] = participation_ratio_from_probs(p)
        out[f"energy_skewness_{region}"] = float(skew(x, bias=False)) if len(x) >= 3 else np.nan
        out[f"energy_kurtosis_{region}"] = float(kurtosis(x, fisher=False, bias=False)) if len(x) >= 4 else np.nan
        out[f"bimodality_coefficient_{region}"] = bimodality_coefficient(x)
        # Tail fractions are defined relative to the combined high+low path-energy spectrum
        # for this ligand/setting. Low-energy tail = exceptionally favorable path energies;
        # high-energy tail = exceptionally unfavorable path energies.
        low_tail = float(np.mean(x <= np.nanpercentile(allE, 100 * TAIL_QUANTILE)))
        high_tail = float(np.mean(x >= np.nanpercentile(allE, 100 * (1.0 - TAIL_QUANTILE))))
        out[f"low_energy_tail_fraction_{region}"] = low_tail
        out[f"high_energy_tail_fraction_{region}"] = high_tail

        # Tail-balance metrics within this MD-probability region.
        # tail_asymmetry > 0 means favorable-energy tail dominates unfavorable-energy tail.
        # tail_polarization measures total weight in both tails, i.e., energy-spectrum broadening.
        out[f"tail_asymmetry_{region}"] = float(low_tail - high_tail)
        out[f"tail_ratio_{region}"] = float((low_tail + EPS) / (high_tail + EPS))
        out[f"tail_log_ratio_{region}"] = float(np.log((low_tail + EPS) / (high_tail + EPS)))
        out[f"tail_polarization_{region}"] = float(low_tail + high_tail)

        out[f"F_{region}"] = free_energy_from_samples(x, beta=beta)

    region_metrics(E_high, "high", p_high)
    region_metrics(E_low, "low", p_low)

    # Contrasts. Positive delta_mean_E = high-probability region has higher path energy.
    out["delta_mean_E"] = out["mean_E_high"] - out["mean_E_low"]
    out["abs_delta_mean_E"] = abs(out["delta_mean_E"])
    out["std_E_high_over_low"] = out["std_E_high"] / (out["std_E_low"] + EPS)
    out["std_E_high_minus_low"] = out["std_E_high"] - out["std_E_low"]

    out["energy_entropy_high_minus_low"] = out["energy_entropy_high"] - out["energy_entropy_low"]
    out["energy_entropy_high_over_low"] = out["energy_entropy_high"] / (out["energy_entropy_low"] + EPS)
    out["energy_effective_bins_high_minus_low"] = out["energy_effective_bins_high"] - out["energy_effective_bins_low"]
    out["energy_effective_bins_high_over_low"] = out["energy_effective_bins_high"] / (out["energy_effective_bins_low"] + EPS)
    out["energy_participation_ratio_high_minus_low"] = out["energy_participation_ratio_high"] - out["energy_participation_ratio_low"]
    out["energy_participation_ratio_high_over_low"] = out["energy_participation_ratio_high"] / (out["energy_participation_ratio_low"] + EPS)
    out["bimodality_coefficient_high_minus_low"] = out["bimodality_coefficient_high"] - out["bimodality_coefficient_low"]
    out["bimodality_coefficient_high_over_low"] = out["bimodality_coefficient_high"] / (out["bimodality_coefficient_low"] + EPS)

    # HIGH vs LOW MD-probability contrasts for tail-balance metrics.
    # high = high MD probability = low MD energy; low = low MD probability = high MD energy.
    out["tail_asymmetry_high_minus_low"] = out["tail_asymmetry_high"] - out["tail_asymmetry_low"]
    out["tail_asymmetry_high_over_low"] = out["tail_asymmetry_high"] / (out["tail_asymmetry_low"] + EPS)
    out["tail_ratio_high_minus_low"] = out["tail_ratio_high"] - out["tail_ratio_low"]
    out["tail_ratio_high_over_low"] = out["tail_ratio_high"] / (out["tail_ratio_low"] + EPS)
    out["tail_log_ratio_high_minus_low"] = out["tail_log_ratio_high"] - out["tail_log_ratio_low"]
    out["tail_log_ratio_high_over_low"] = out["tail_log_ratio_high"] / (out["tail_log_ratio_low"] + EPS)
    out["tail_polarization_high_minus_low"] = out["tail_polarization_high"] - out["tail_polarization_low"]
    out["tail_polarization_high_over_low"] = out["tail_polarization_high"] / (out["tail_polarization_low"] + EPS)

    out["delta_F"] = out["F_high"] - out["F_low"]
    out["abs_delta_F"] = abs(out["delta_F"])

    n1, n2 = len(E_high), len(E_low)
    s1, s2 = np.nanstd(E_high, ddof=1), np.nanstd(E_low, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1))
    out["cohens_d"] = float((np.nanmean(E_high) - np.nanmean(E_low)) / (pooled + EPS))

    if jensenshannon is not None and len(p_high) == len(p_low) and len(p_high) > 0:
        js_dist = float(jensenshannon(p_high + EPS, p_low + EPS, base=np.e))
        out["JS_divergence"] = float(js_dist ** 2)
    else:
        out["JS_divergence"] = np.nan

    if wasserstein_distance is not None:
        out["wasserstein_distance"] = float(wasserstein_distance(E_high, E_low))
    else:
        out["wasserstein_distance"] = np.nan

    return out


# =============================================================================
# PER-LIGAND ANALYSIS
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
        "USE_ENERGY_RANK_FOR_MD_BINS": USE_ENERGY_RANK_FOR_MD_BINS,
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

    node_energy, edge_energy = build_caches(X, h, J)
    windows = make_windows(X.shape[0], WINDOW_SIZE)

    # Independent MD-frame/window Potts landscape.
    Eframe_full_potts = full_potts_frame_energy(X, h, J)
    Ewin_full_potts = window_mean(Eframe_full_potts, windows)
    md_window_probability_score = stable_prob(Ewin_full_potts, beta=FRAME_BETA)

    if USE_ENERGY_RANK_FOR_MD_BINS:
        # Lower energy = higher probability. Convert to a monotonic score where larger = more probable.
        md_binning_score = -Ewin_full_potts
    else:
        md_binning_score = md_window_probability_score

    path_rows = []
    E_by_rank = {}

    for _, prow in paths_df.iterrows():
        rank = int(prow["rank"])
        path_nodes = parse_path(prow["path"])
        path_idx = [node_to_index(x, label_to_idx, idx_to_label) for x in path_nodes]

        Eframe = path_energy(path_idx, node_energy, edge_energy, energy_mode)
        Ewin = window_mean(Eframe, windows)
        E_by_rank[rank] = Ewin

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
        })

    path_info = pd.DataFrame(path_rows)
    if path_info.empty:
        return None, diag, pd.DataFrame()

    beta = beta_for_mode(energy_mode)
    pmean = stable_prob(path_info["mean_energy"].values, beta=beta)
    path_info["mean_path_probability"] = pmean
    path_info["probability_order"] = path_info["mean_path_probability"].rank(ascending=False, method="first").astype(int)
    path_info = path_info.sort_values("probability_order").reset_index(drop=True)

    payload = {
        "windows": windows,
        "path_info": path_info,
        "md_window_energy": Ewin_full_potts,
        "md_window_probability_score": md_window_probability_score,
        "md_binning_score": md_binning_score,
        "E_by_rank": E_by_rank,
    }
    return payload, diag, path_info


def analyze_topn_energy_spectrum(payload, lig, edge_percentile, energy_mode, top_n):
    path_info = payload["path_info"]
    windows = payload["windows"]
    md_window_energy = payload["md_window_energy"]
    md_window_probability_score = payload["md_window_probability_score"]
    md_binning_score = payload["md_binning_score"]
    E_by_rank = payload["E_by_rank"]

    selected, top_n_effective, prob_mass_cutoff = select_paths_by_probability_mass(path_info, top_n)
    selected = [r for r in selected if r in E_by_rank]

    if len(selected) < MIN_SELECTED_PATHS:
        return pd.DataFrame(), pd.DataFrame()

    Ewin = np.vstack([E_by_rank[r] for r in selected]).T  # windows x paths

    # Define high/low MD-probability bins from independent full MD Potts landscape.
    window_table = windows.copy()
    window_table["ligand"] = lig
    window_table["FRET"] = FRET[lig]
    window_table["EDGE_PERCENTILE"] = edge_percentile
    window_table["ENERGY_MODE"] = energy_mode
    window_table["BETA"] = beta_for_mode(energy_mode)
    window_table["TOP_N"] = str(top_n)
    window_table["PROB_MASS_CUTOFF"] = prob_mass_cutoff
    window_table["TOP_N_EFFECTIVE"] = int(top_n_effective)
    window_table["n_selected_paths"] = len(selected)
    window_table["md_window_energy"] = md_window_energy
    window_table["md_window_probability_score"] = md_window_probability_score
    window_table["md_binning_score"] = md_binning_score

    try:
        window_table["probability_bin"] = pd.qcut(
            window_table["md_binning_score"],
            q=N_PROB_BINS,
            labels=np.arange(1, N_PROB_BINS + 1),
            duplicates="drop",
        ).astype(int)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    low_bin = int(window_table["probability_bin"].min())
    high_bin = int(window_table["probability_bin"].max())

    # IMPORTANT:
    # low_bin means low probability / high MD energy.
    # high_bin means high probability / low MD energy.
    low_window_idx = window_table.index[window_table["probability_bin"] == low_bin].to_numpy()
    high_window_idx = window_table.index[window_table["probability_bin"] == high_bin].to_numpy()

    E_high = Ewin[high_window_idx, :].ravel()
    E_low = Ewin[low_window_idx, :].ravel()

    metrics = energy_spectrum_metrics(E_high, E_low, beta=beta_for_mode(energy_mode))
    row = {
        "ligand": lig,
        "FRET": FRET[lig],
        "EDGE_PERCENTILE": edge_percentile,
        "ENERGY_MODE": energy_mode,
        "BETA": beta_for_mode(energy_mode),
        "FRAME_BETA": FRAME_BETA,
        "TOP_N": str(top_n),
        "PROB_MASS_CUTOFF": prob_mass_cutoff,
        "TOP_N_EFFECTIVE": int(top_n_effective),
        "n_selected_paths": len(selected),
        "n_windows_total": int(len(window_table)),
        "n_windows_low_probability": int(len(low_window_idx)),
        "n_windows_high_probability": int(len(high_window_idx)),
        "mean_md_energy_high": float(np.nanmean(md_window_energy[high_window_idx])),
        "mean_md_energy_low": float(np.nanmean(md_window_energy[low_window_idx])),
        "mean_md_probability_high": float(np.nanmean(md_window_probability_score[high_window_idx])),
        "mean_md_probability_low": float(np.nanmean(md_window_probability_score[low_window_idx])),
        **metrics,
    }

    return pd.DataFrame([row]), window_table


# =============================================================================
# CORRELATION AND ROBUSTNESS
# =============================================================================

def rank_order_string(sub, metric, descending=True):
    return ">".join(str(int(v)) for v in sub.sort_values(metric, ascending=not descending)["ligand"].values)


def correlate_all5(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    id_cols = {
        "ligand", "FRET", "EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "FRAME_BETA", "TOP_N",
        "PROB_MASS_CUTOFF", "TOP_N_EFFECTIVE", "n_selected_paths", "n_windows_total",
        "n_windows_low_probability", "n_windows_high_probability",
    }
    metric_cols = [c for c in metrics_df.columns if c not in id_cols]

    rows = []
    diagnostics = []
    group_cols = ["EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "TOP_N", "PROB_MASS_CUTOFF"]

    for key, sub0 in metrics_df.groupby(group_cols):
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
            "mean_abs_pearson_r": float(np.nanmean(np.abs(pearson))),
            "min_abs_pearson_r": float(np.nanmin(np.abs(pearson))),
            "std_pearson_r": float(np.nanstd(pearson)),
            "mean_spearman_rho": float(np.nanmean(spearman)),
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


def plot_top_robustness(robust):
    if robust.empty:
        return
    d = robust.head(40).sort_values("robustness_score", ascending=True).copy()
    labels = [f"{r.ENERGY_MODE} | {r.metric}" for r in d.itertuples(index=False)]
    plt.figure(figsize=(12, max(7, 0.30 * len(d))))
    plt.barh(np.arange(len(d)), d["robustness_score"].values)
    plt.yticks(np.arange(len(d)), labels, fontsize=7)
    plt.xlabel("Robustness score")
    plt.title("Top energy-spectrum robustness metrics")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "top_energy_spectrum_robustness_scores.png")
    plt.savefig(out, dpi=300)
    plt.close()


def plot_best_metric_bars(corr, n=30):
    if corr.empty:
        return
    outdir = os.path.join(FIGDIR, "best_metric_bars")
    os.makedirs(outdir, exist_ok=True)
    d = corr.copy()
    d["abs_r"] = d["pearson_r"].abs()
    d = d.sort_values("abs_r", ascending=False).head(n)
    for _, row in d.iterrows():
        vals = _parse_values_by_ligand(row["values_by_ligand"])
        ligs = [lig for lig in LIGANDS if lig in vals]
        if len(ligs) == 0:
            continue
        plt.figure(figsize=(6.0, 4.0))
        plt.bar([f"L{lig}" for lig in ligs], [vals[lig] for lig in ligs])
        plt.xlabel("Ligand")
        plt.ylabel(row["metric"])
        plt.title(
            f"{row['ENERGY_MODE']} | {row['metric']}\n"
            f"TOP_N={row['TOP_N']}, r={row['pearson_r']:.3f}, rho={row['spearman_rho']:.3f}",
            fontsize=9,
        )
        plt.tight_layout()
        out = os.path.join(
            outdir,
            f"bars_{safe_filename(row['ENERGY_MODE'])}_{safe_filename(row['metric'])}_{safe_filename(row['TOP_N'])}.png",
        )
        plt.savefig(out, dpi=300)
        plt.close()


def plot_metric_heatmaps(corr):
    if corr.empty:
        return
    outdir = os.path.join(FIGDIR, "heatmaps_by_metric")
    os.makedirs(outdir, exist_ok=True)
    for (energy_mode, metric), sub in corr.groupby(["ENERGY_MODE", "metric"]):
        top_ns = sorted(sub["TOP_N"].unique())
        edge_ps = sorted(sub["EDGE_PERCENTILE"].unique())
        mat = np.full((len(edge_ps), len(top_ns)), np.nan)
        for i, ep in enumerate(edge_ps):
            for j, tn in enumerate(top_ns):
                d = sub[(sub["EDGE_PERCENTILE"] == ep) & (sub["TOP_N"] == tn)]
                if len(d):
                    mat[i, j] = float(d.iloc[0]["pearson_r"])
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(top_ns)), max(3.5, 0.55 * len(edge_ps))))
        im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_yticks(np.arange(len(edge_ps)))
        ax.set_yticklabels([str(x) for x in edge_ps])
        ax.set_xticks(np.arange(len(top_ns)))
        ax.set_xticklabels([str(x) for x in top_ns])
        ax.set_ylabel("J-edge percentile")
        ax.set_xlabel("Path subset")
        ax.set_title(f"Pearson r with efficacy\n{energy_mode}: {metric}", fontsize=9)
        for i in range(len(edge_ps)):
            for j in range(len(top_ns)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pearson r")
        plt.tight_layout()
        out = os.path.join(outdir, f"heatmap_{safe_filename(energy_mode)}_{safe_filename(metric)}.png")
        plt.savefig(out, dpi=300)
        plt.close()




def plot_scatter_metric_vs_FRET_from_values(row, outdir, tag):
    """Plot one metric-vs-FRET scatter from one row of correlation table."""
    os.makedirs(outdir, exist_ok=True)

    vals = _parse_values_by_ligand(row.get("values_by_ligand", ""))
    rows = []
    for lig in LIGANDS:
        if lig in vals and np.isfinite(vals[lig]) and lig in FRET:
            rows.append((lig, float(FRET[lig]), float(vals[lig])))

    if len(rows) < 3:
        return None

    ligs = [r[0] for r in rows]
    x = np.asarray([r[1] for r in rows], dtype=float)
    y = np.asarray([r[2] for r in rows], dtype=float)

    if np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        return None

    pearson_r = float(row.get("pearson_r", np.nan))
    pearson_p = float(row.get("pearson_p", np.nan)) if "pearson_p" in row else np.nan
    spearman_rho = float(row.get("spearman_rho", np.nan)) if "spearman_rho" in row else np.nan
    energy_mode = str(row.get("ENERGY_MODE", "NA"))
    top_n = str(row.get("TOP_N", "NA"))
    metric = str(row.get("metric", "metric"))
    edge_percentile = row.get("EDGE_PERCENTILE", "NA")
    prob_mass_cutoff = row.get("PROB_MASS_CUTOFF", np.nan)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    ax.scatter(x, y, s=70, zorder=3)

    for lig, xi, yi in rows:
        ax.annotate(
            f"L{lig}",
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    # Least-squares regression line for visual guidance.
    try:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yy = m * xx + b
        ax.plot(xx, yy, linestyle="--", linewidth=1.5, zorder=2)
    except Exception:
        pass

    ax.set_xlabel("Experimental FRET / efficacy")
    ax.set_ylabel(metric)
    ax.set_title(
        f"{metric}\n{energy_mode}, {top_n}, edge={edge_percentile}",
        fontsize=9,
    )

    text = f"Pearson r = {pearson_r:.3f}"
    if np.isfinite(pearson_p):
        text += f"\np = {pearson_p:.3g}"
    if np.isfinite(spearman_rho):
        text += f"\nSpearman ρ = {spearman_rho:.3f}"
    if pd.notna(prob_mass_cutoff):
        text += f"\nPMASS = {float(prob_mass_cutoff):.2f}"

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fname = (
        f"{tag}_scatter_"
        f"{safe_filename(energy_mode)}_"
        f"{safe_filename(top_n)}_"
        f"{safe_filename(metric)}.png"
    )
    outpath = os.path.join(outdir, fname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return outpath


def plot_FRET_scatter_plots(corr, robust, max_all_plots=None, top_n_individual=80, top_n_robust=80):
    """Create metric value vs experimental FRET scatter plots.

    Outputs
    -------
    FIGDIR/scatter_vs_FRET/
      all_correlations/
      top_individual_correlations/
      top_robust_metrics/
      scatter_plot_index.csv

    Notes
    -----
    - all_correlations contains every finite row in all5_correlations_energy_spectrum.csv
      unless max_all_plots is set.
    - top_individual_correlations contains the strongest individual metric/settings
      by absolute Pearson r.
    - top_robust_metrics contains the best robust metrics. For each robust metric,
      all matching setting-specific scatter plots are generated.
    """
    if corr is None or corr.empty:
        return

    scatter_root = os.path.join(FIGDIR, "scatter_vs_FRET")
    all_dir = os.path.join(scatter_root, "all_correlations")
    top_ind_dir = os.path.join(scatter_root, "top_individual_correlations")
    top_rob_dir = os.path.join(scatter_root, "top_robust_metrics")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(top_ind_dir, exist_ok=True)
    os.makedirs(top_rob_dir, exist_ok=True)

    index_rows = []

    # 1. Every individual correlation row.
    all_corr = corr.copy()
    all_corr = all_corr[np.isfinite(all_corr["pearson_r"].astype(float))]
    all_corr = all_corr.sort_values("abs_pearson_r", ascending=False)
    if max_all_plots is not None:
        all_corr = all_corr.head(int(max_all_plots))

    for i, (_, row) in enumerate(all_corr.iterrows(), start=1):
        out = plot_scatter_metric_vs_FRET_from_values(row, all_dir, tag=f"all_{i:04d}")
        if out:
            index_rows.append({
                "category": "all_correlations",
                "rank": i,
                "plot_file": out,
                "ENERGY_MODE": row.get("ENERGY_MODE"),
                "TOP_N": row.get("TOP_N"),
                "metric": row.get("metric"),
                "pearson_r": row.get("pearson_r"),
                "pearson_p": row.get("pearson_p", np.nan),
                "spearman_rho": row.get("spearman_rho", np.nan),
                "abs_pearson_r": row.get("abs_pearson_r", np.nan),
            })

    # 2. Top individual correlations.
    top_ind = corr.copy()
    top_ind = top_ind[np.isfinite(top_ind["pearson_r"].astype(float))]
    top_ind = top_ind.sort_values("abs_pearson_r", ascending=False).head(int(top_n_individual))

    for i, (_, row) in enumerate(top_ind.iterrows(), start=1):
        out = plot_scatter_metric_vs_FRET_from_values(row, top_ind_dir, tag=f"top_individual_{i:03d}")
        if out:
            index_rows.append({
                "category": "top_individual_correlations",
                "rank": i,
                "plot_file": out,
                "ENERGY_MODE": row.get("ENERGY_MODE"),
                "TOP_N": row.get("TOP_N"),
                "metric": row.get("metric"),
                "pearson_r": row.get("pearson_r"),
                "pearson_p": row.get("pearson_p", np.nan),
                "spearman_rho": row.get("spearman_rho", np.nan),
                "abs_pearson_r": row.get("abs_pearson_r", np.nan),
            })

    # 3. Top robust metrics: generate all setting-specific rows for each robust metric.
    if robust is not None and not robust.empty:
        robust_top = robust.head(int(top_n_robust)).copy()
        robust_pairs = set(zip(robust_top["ENERGY_MODE"].astype(str), robust_top["metric"].astype(str)))
        d = corr.copy()
        d["_pair"] = list(zip(d["ENERGY_MODE"].astype(str), d["metric"].astype(str)))
        d = d[d["_pair"].isin(robust_pairs)].copy()
        d = d.merge(
            robust_top[["ENERGY_MODE", "metric", "robustness_score"]],
            on=["ENERGY_MODE", "metric"],
            how="left",
        )
        d = d.sort_values(["robustness_score", "abs_pearson_r"], ascending=False)

        for i, (_, row) in enumerate(d.iterrows(), start=1):
            out = plot_scatter_metric_vs_FRET_from_values(row, top_rob_dir, tag=f"top_robust_{i:03d}")
            if out:
                index_rows.append({
                    "category": "top_robust_metrics",
                    "rank": i,
                    "plot_file": out,
                    "ENERGY_MODE": row.get("ENERGY_MODE"),
                    "TOP_N": row.get("TOP_N"),
                    "metric": row.get("metric"),
                    "pearson_r": row.get("pearson_r"),
                    "pearson_p": row.get("pearson_p", np.nan),
                    "spearman_rho": row.get("spearman_rho", np.nan),
                    "abs_pearson_r": row.get("abs_pearson_r", np.nan),
                    "robustness_score": row.get("robustness_score", np.nan),
                })

    if index_rows:
        pd.DataFrame(index_rows).to_csv(os.path.join(scatter_root, "scatter_plot_index.csv"), index=False)



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
            "Analyze communication-path energy spectra in high- and low-probability "
            "MD regions using a learned Potts+BNM model."
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
                        help="Output directory. Default: BASE/MD_REGION_COMMUNICATION_ENERGY_TAIL_ASYMMETRY")

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
    parser.add_argument("--use-energy-rank-for-md-bins", action="store_true",
                        help="Use -E_MD_window instead of Boltzmann probability for MD binning.")
    parser.add_argument("--n-energy-bins", type=int, default=N_ENERGY_BINS)
    parser.add_argument("--tail-quantile", type=float, default=TAIL_QUANTILE)

    return parser.parse_args(argv)


def configure_from_args(args):
    """Update module-level settings from command-line arguments."""
    global BASE, CONTACT_PATTERN, H_PATTERN, J_PATTERN, DOT_PATTERN, OUTDIR, FIGDIR
    global LIGANDS, FRET, SOURCE_NODES, TARGET_NODES
    global EDGE_PERCENTILES, ENERGY_MODES, PROB_MASS_CUTOFFS, TOP_N_LIST
    global WINDOW_SIZE, N_PROB_BINS, MAX_PATHS, PATHS_PER_SOURCE, PATHS_PER_TARGET
    global TARGETS_PER_SOURCE, MIN_PATH_LEN, MAX_PATH_LEN
    global FRAME_BETA, USE_ENERGY_RANK_FOR_MD_BINS, N_ENERGY_BINS, TAIL_QUANTILE
    global MIN_LIGANDS_FOR_CORR

    BASE = args.base
    CONTACT_PATTERN = args.contact_pattern
    H_PATTERN = args.h_pattern
    J_PATTERN = args.j_pattern
    DOT_PATTERN = args.dot_pattern

    OUTDIR = args.outdir or os.path.join(BASE, "MD_REGION_COMMUNICATION_ENERGY_TAIL_ASYMMETRY")
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
    USE_ENERGY_RANK_FOR_MD_BINS = args.use_energy_rank_for_md_bins
    N_ENERGY_BINS = args.n_energy_bins
    TAIL_QUANTILE = args.tail_quantile

    MIN_LIGANDS_FOR_CORR = len(LIGANDS)


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    configure_from_args(args)

    all_diag = []
    all_paths = []
    all_windows = []
    all_metrics = []

    for edge_percentile in EDGE_PERCENTILES:
        for energy_mode in ENERGY_MODES:
            for lig in LIGANDS:
                try:
                    payload, diag, path_info = compute_ligand_setting(lig, edge_percentile, energy_mode)
                    all_diag.append(diag)
                    if path_info is not None and len(path_info):
                        all_paths.append(path_info)
                    if payload is None:
                        continue
                    for top_n in TOP_N_LIST:
                        metrics_row, window_table = analyze_topn_energy_spectrum(
                            payload, lig, edge_percentile, energy_mode, top_n
                        )
                        if len(metrics_row):
                            all_metrics.append(metrics_row)
                        if len(window_table):
                            all_windows.append(window_table)
                except Exception as e:
                    print(f"ERROR ligand={lig}, edge={edge_percentile}, mode={energy_mode}: {e}")
                    all_diag.append({
                        "ligand": lig,
                        "EDGE_PERCENTILE": edge_percentile,
                        "ENERGY_MODE": energy_mode,
                        "error": str(e),
                    })

    diag_df = pd.DataFrame(all_diag)
    paths_df = pd.concat(all_paths, ignore_index=True) if all_paths else pd.DataFrame()
    windows_df = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    diag_df.to_csv(os.path.join(OUTDIR, "path_generation_diagnostics.csv"), index=False)
    paths_df.to_csv(os.path.join(OUTDIR, "generated_paths_all_settings.csv"), index=False)
    windows_df.to_csv(os.path.join(OUTDIR, "window_energy_region_table.csv"), index=False)
    metrics_df.to_csv(os.path.join(OUTDIR, "energy_spectrum_metrics_all_settings.csv"), index=False)

    corr_df, diagnostics_df = correlate_all5(metrics_df)
    robust_df = robustness_summary(corr_df)
    rank_df = rank_stability_summary(corr_df)

    corr_df.to_csv(os.path.join(OUTDIR, "all5_correlations_energy_spectrum.csv"), index=False)
    diagnostics_df.to_csv(os.path.join(OUTDIR, "incomplete_metric_diagnostics_energy_spectrum.csv"), index=False)
    robust_df.to_csv(os.path.join(OUTDIR, "metric_robustness_energy_spectrum.csv"), index=False)
    rank_df.to_csv(os.path.join(OUTDIR, "metric_rank_stability_energy_spectrum.csv"), index=False)

    plot_top_robustness(robust_df)
    plot_best_metric_bars(corr_df, n=40)
    plot_metric_heatmaps(corr_df)
    plot_FRET_scatter_plots(corr_df, robust_df, max_all_plots=None, top_n_individual=100, top_n_robust=100)

    print("\nDone.")
    print("Output directory:", OUTDIR)
    if len(corr_df):
        print("\nTop individual all-5 correlations:")
        cols = [
            "EDGE_PERCENTILE", "ENERGY_MODE", "BETA", "TOP_N", "PROB_MASS_CUTOFF",
            "metric", "n_ligands_used", "pearson_r", "spearman_rho",
            "rank_order_high_to_low", "values_by_ligand",
        ]
        print(corr_df.sort_values("abs_pearson_r", ascending=False).head(30)[cols].to_string(index=False))
    if len(robust_df):
        print("\nMost robust energy-spectrum metrics:")
        cols = [
            "ENERGY_MODE", "metric", "n_settings", "expected_settings", "setting_coverage_fraction",
            "robustness_score", "pearson_sign_consistency", "mean_abs_pearson_r",
            "min_abs_pearson_r", "frac_abs_spearman_ge_0p8", "median_pearson_r", "std_pearson_r",
        ]
        print(robust_df.head(40)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
