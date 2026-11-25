"""Fast pre-flight checks for the shared IEA37-CS4-inspired 50T case."""

from __future__ import annotations

import argparse

import networkx as nx
import numpy as np

import large_case_config as case


def minimum_spacing(layout: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(layout[i + 1 :] - layout[i], axis=1).min())
        for i in range(len(layout) - 1)
    )


def candidate_graph_is_connected(layout: np.ndarray) -> bool:
    positions = np.vstack([layout, case.SUBSTATION_POSITIONS])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(positions)))
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) <= case.MAX_CABLE_LENGTH_M:
                graph.add_edge(i, j)
    return nx.is_connected(graph)


def verify_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    layout = case.sample_large_case_layout(rng=rng)
    spacing = case.MIN_SPACING_FACTOR * case.ROTOR_DIAMETER_M
    result = {
        "seed": seed,
        "turbines": len(layout),
        "site_violations": case.count_site_violations(layout),
        "violation_score": case.layout_violation_score(layout, spacing),
        "minimum_spacing_m": minimum_spacing(layout),
        "candidate_graph_connected": candidate_graph_is_connected(layout),
    }
    assert result["turbines"] == case.NUM_TURBINES
    assert result["site_violations"] == 0
    assert result["violation_score"] <= 1e-6
    assert result["minimum_spacing_m"] + 1e-6 >= spacing
    assert result["candidate_graph_connected"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 42, 98, 133])
    args = parser.parse_args()

    scenarios = case.build_wind_scenarios()
    assert case.SITE_GEOMETRY.is_valid
    assert len(case.SAFE_SITE_COMPONENTS) == 5
    assert len(scenarios) == 12
    assert abs(sum(row["weight"] for row in scenarios) - 1.0) <= 1e-12

    print(f"case: {case.CASE_NAME}")
    print(f"safe lease area: {case.SAFE_SITE_GEOMETRY.area / 1e6:.3f} km^2")
    print(f"wind sectors: {len(scenarios)}; frequency sum: 1.000000")
    for result in map(verify_seed, args.seeds):
        print(
            "seed={seed}: turbines={turbines}, site_violations={site_violations}, "
            "min_spacing={minimum_spacing_m:.3f} m, connected={candidate_graph_connected}"
            .format(**result)
        )
    print("verification: PASS")


if __name__ == "__main__":
    main()
