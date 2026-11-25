"""Shared 50-turbine semi-real offshore case used by all V7-V10 variants.

The lease-area coordinates and wind resource originate from IEA Wind Task 37
WFLO Case Study 4, which was adapted from Borssele III and IV.  The official
case has 81 turbines.  This project uses the same site/wind-resource basis but
sets the plant size to 50 IEA 10 MW reference turbines so that the case remains
within the requested 40-60 turbine range.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union


CASE_ID = "iea37_cs4_borssele_inspired_50t"
CASE_NAME = "IEA37 Case Study 4 inspired 50-turbine semi-real offshore case"
CASE_SOURCE_URL = (
    "https://github.com/byuflowlab/iea37-wflo-casestudies/tree/master/cs3-4"
)
CASE_PAPER_URL = "https://doi.org/10.5194/wes-8-865-2023"

NUM_TURBINES = 50
NUM_SUBSTATIONS = 1
GRID_SIZE = (10_500.0, 12_050.0)
SUBSTATION_POSITIONS = np.asarray([[4_330.0, 6_930.0]], dtype=float)
RATED_POWER_MW = 10.0
ROTOR_DIAMETER_M = 198.0
HUB_HEIGHT_M = 119.0
MIN_SPACING_FACTOR = 3.5
BOUNDARY_MARGIN_M = 100.0
MAX_CABLE_LENGTH_M = 8_500.0
COLLECTION_VOLTAGE_KV = 66.0
MAX_TOPOLOGY_STEPS = 120
MAX_ALLOWED_CROSSINGS = 10

# The complex five-region lease-area definition is copied from the official
# IEA37 Case Study 4 boundary YAML (ETRS89 / UTM zone 31U, local coordinates).
SITE_REGIONS = {
    "IIIa": [
        (10363.8, 6490.3), (9449.7, 1602.2), (9387.0, 1056.6),
        (9365.1, 625.5), (9360.8, 360.2), (9361.5, 126.9),
        (9361.3, 137.1), (7997.6, 1457.9), (6098.3, 3297.5),
        (8450.3, 6455.3), (8505.4, 6422.3), (9133.0, 6127.4),
        (9332.8, 6072.6), (9544.2, 6087.1), (9739.0, 6171.2),
        (9894.9, 6316.9), (10071.8, 6552.5), (10106.9, 6611.1),
    ],
    "IIIb": [
        (5588.4, 3791.3), (4670.7, 4680.2), (7274.9, 7940.8),
        (7369.9, 7896.2), (7455.1, 7784.3), (7606.5, 7713.0),
        (7638.9, 7708.4), (8297.1, 7398.9),
    ],
    "IVa": [
        (3267.1, 10100.6), (4164.1, 9586.6), (5749.8, 9068.6),
        (6054.7, 8925.3), (1468.5, 7781.7), (107.4, 9100.0),
    ],
    "IVb": [
        (6764.9, 8399.7), (4176.8, 5158.6), (2047.8, 7220.7),
    ],
    "IVc": [
        (8953.7, 11901.5), (7048.3, 9531.5), (6127.7, 9962.7),
        (4578.1, 10464.9), (4524.1, 10498.7),
    ],
}

# No additional artificial circular exclusion zones are imposed.  The five
# disconnected/concave lease regions already provide a substantially harder
# geometric feasibility problem than the former rectangle-plus-two-circles case.
FORBIDDEN_ZONES: list[dict] = []

# The official 360-direction Case Study 4 resource was aggregated into 12
# 30-degree meteorological sectors.  Frequencies sum to one.  Each speed is the
# conditional mean obtained from the official 20-bin speed distribution within
# that sector.  The solver uses propagation direction, hence the +180-degree
# conversion performed by build_wind_scenarios().
WIND_ROSE_MET_DIRECTIONS_DEG = np.arange(0.0, 360.0, 30.0)
WIND_ROSE_FREQUENCIES = np.asarray([
    0.05595004, 0.04292559, 0.04510266, 0.06600197,
    0.08347316, 0.07052928, 0.08663829, 0.11265941,
    0.11858778, 0.11267793, 0.11204917, 0.09340471,
], dtype=float)
WIND_ROSE_MEAN_SPEEDS_MS = np.asarray([
    8.1020, 7.5531, 7.6931, 8.7615,
    9.6954, 9.9254, 9.9913, 10.0120,
    9.8495, 9.7828, 9.7328, 9.1169,
], dtype=float)
WIND_ROSE_FREQUENCIES /= WIND_ROSE_FREQUENCIES.sum()


def _build_site_geometry():
    polygons = []
    for coordinates in SITE_REGIONS.values():
        polygon = Polygon(coordinates)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons)


SITE_GEOMETRY = _build_site_geometry()
SAFE_SITE_GEOMETRY = SITE_GEOMETRY.buffer(-BOUNDARY_MARGIN_M)
if SAFE_SITE_GEOMETRY.is_empty:
    raise RuntimeError("The configured boundary margin removes the entire site.")


def _polygon_components(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [part for part in geometry.geoms if isinstance(part, Polygon)]


SAFE_SITE_COMPONENTS = _polygon_components(SAFE_SITE_GEOMETRY)
SAFE_SITE_AREAS = np.asarray([part.area for part in SAFE_SITE_COMPONENTS], dtype=float)
SAFE_SITE_WEIGHTS = SAFE_SITE_AREAS / SAFE_SITE_AREAS.sum()


def _rng_uniform(rng, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _rng_choice(rng, size: int, probabilities: np.ndarray) -> int:
    return int(rng.choice(size, p=probabilities))


def sample_site_point(rng=np.random) -> np.ndarray:
    """Draw one point from the buffered multi-polygon, weighted by region area."""
    for _ in range(20_000):
        component = SAFE_SITE_COMPONENTS[
            _rng_choice(rng, len(SAFE_SITE_COMPONENTS), SAFE_SITE_WEIGHTS)
        ]
        min_x, min_y, max_x, max_y = component.bounds
        point = Point(
            _rng_uniform(rng, min_x, max_x),
            _rng_uniform(rng, min_y, max_y),
        )
        if component.covers(point):
            return np.asarray(point.coords[0], dtype=float)
    raise RuntimeError("Unable to sample a point from the IEA37 lease regions.")


def project_points_to_site(layout: Sequence[Sequence[float]]) -> np.ndarray:
    """Project every out-of-bound point onto the nearest buffered lease region."""
    points = np.asarray(layout, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("layout must have shape (n_turbines, 2)")

    for index, coordinates in enumerate(points):
        point = Point(float(coordinates[0]), float(coordinates[1]))
        if SAFE_SITE_GEOMETRY.covers(point):
            continue
        component = min(SAFE_SITE_COMPONENTS, key=point.distance)
        nearest = nearest_points(point, component)[1]
        target = component.representative_point()
        vector = np.asarray(target.coords[0]) - np.asarray(nearest.coords[0])
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            projected = np.asarray(nearest.coords[0]) + vector / norm * 1e-3
        else:
            projected = np.asarray(nearest.coords[0])
        points[index] = projected
    return points


def _outside_forbidden_zones(point: np.ndarray, forbidden_zones: Iterable[dict],
                             obstacle_margin: float) -> bool:
    for zone in forbidden_zones:
        center = np.asarray(zone["center"], dtype=float)
        limit = float(zone["radius"]) + float(obstacle_margin)
        if np.linalg.norm(point - center) < limit:
            return False
    return True


def sample_large_case_layout(
    n_turbines: int = NUM_TURBINES,
    min_spacing: float | None = None,
    forbidden_zones: Iterable[dict] = FORBIDDEN_ZONES,
    obstacle_margin: float = 0.0,
    rng=np.random,
    max_restarts: int = 40,
) -> np.ndarray:
    """Generate a strict-feasible layout by area-weighted Poisson sampling."""
    spacing = float(
        min_spacing
        if min_spacing is not None
        else MIN_SPACING_FACTOR * ROTOR_DIAMETER_M
    )
    best: list[np.ndarray] = []
    for _ in range(max_restarts):
        selected: list[np.ndarray] = []
        failed_draws = 0
        while len(selected) < n_turbines and failed_draws < 80_000:
            candidate = sample_site_point(rng)
            if not _outside_forbidden_zones(candidate, forbidden_zones, obstacle_margin):
                failed_draws += 1
                continue
            if selected:
                distances = np.linalg.norm(np.asarray(selected) - candidate, axis=1)
                if float(np.min(distances)) < spacing:
                    failed_draws += 1
                    continue
            selected.append(candidate)
            failed_draws = 0
        if len(selected) > len(best):
            best = selected
        if len(selected) == n_turbines:
            return np.asarray(selected, dtype=float)
    raise RuntimeError(
        f"Could place only {len(best)}/{n_turbines} turbines with "
        f"minimum spacing {spacing:.1f} m."
    )


def layout_violation_score(
    layout: Sequence[Sequence[float]],
    min_spacing: float,
    forbidden_zones: Iterable[dict] = FORBIDDEN_ZONES,
    obstacle_margin: float = 0.0,
) -> float:
    """Return zero only for layouts satisfying site, spacing and zone limits."""
    points = np.asarray(layout, dtype=float)
    if points.shape != (NUM_TURBINES, 2) or not np.all(np.isfinite(points)):
        return 1e12

    score = 0.0
    for coordinates in points:
        point = Point(float(coordinates[0]), float(coordinates[1]))
        if not SAFE_SITE_GEOMETRY.covers(point):
            score += max(1.0, point.distance(SAFE_SITE_GEOMETRY))
        for zone in forbidden_zones:
            center = np.asarray(zone["center"], dtype=float)
            limit = float(zone["radius"]) + float(obstacle_margin)
            score += max(0.0, limit - float(np.linalg.norm(coordinates - center)))

    for i in range(len(points)):
        distances = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
        if len(distances):
            score += float(np.maximum(float(min_spacing) - distances, 0.0).sum())
    return score


def repair_large_case_layout(
    layout: Sequence[Sequence[float]],
    n_turbines: int = NUM_TURBINES,
    min_spacing: float | None = None,
    forbidden_zones: Iterable[dict] = FORBIDDEN_ZONES,
    obstacle_margin: float = 0.0,
    rng=np.random,
    iterations: int = 120,
) -> np.ndarray:
    """Repair a genetic candidate and fall back to strict resampling if needed."""
    spacing = float(
        min_spacing
        if min_spacing is not None
        else MIN_SPACING_FACTOR * ROTOR_DIAMETER_M
    )
    points = np.asarray(layout, dtype=float).copy()
    if points.shape != (n_turbines, 2) or not np.all(np.isfinite(points)):
        return sample_large_case_layout(
            n_turbines, spacing, forbidden_zones, obstacle_margin, rng
        )
    points = project_points_to_site(points)

    for _ in range(max(1, int(iterations))):
        moved = False
        order = np.arange(n_turbines)
        rng.shuffle(order)
        for left_index in range(n_turbines):
            i = int(order[left_index])
            for right_index in range(left_index + 1, n_turbines):
                j = int(order[right_index])
                vector = points[i] - points[j]
                distance = float(np.linalg.norm(vector))
                if distance >= spacing:
                    continue
                if distance < 1e-9:
                    angle = _rng_uniform(rng, 0.0, 2.0 * math.pi)
                    vector = np.asarray([math.cos(angle), math.sin(angle)])
                    distance = 1.0
                displacement = vector / distance * (spacing - distance) * 0.55
                points[i] += displacement
                points[j] -= displacement
                moved = True
        points = project_points_to_site(points)
        if layout_violation_score(
            points, spacing, forbidden_zones, obstacle_margin
        ) <= 1e-6:
            return points
        if not moved:
            break

    # Reinsert conflicting points while preserving as much of the candidate as
    # possible.  A fully fresh layout is used only when local reinsertion jams.
    accepted: list[np.ndarray] = []
    for candidate in points:
        if not _outside_forbidden_zones(candidate, forbidden_zones, obstacle_margin):
            continue
        if accepted and np.min(np.linalg.norm(np.asarray(accepted) - candidate, axis=1)) < spacing:
            continue
        accepted.append(candidate.copy())

    failed_draws = 0
    while len(accepted) < n_turbines and failed_draws < 120_000:
        candidate = sample_site_point(rng)
        if not _outside_forbidden_zones(candidate, forbidden_zones, obstacle_margin):
            failed_draws += 1
            continue
        if accepted and np.min(np.linalg.norm(np.asarray(accepted) - candidate, axis=1)) < spacing:
            failed_draws += 1
            continue
        accepted.append(candidate)
        failed_draws = 0
    if len(accepted) == n_turbines:
        return np.asarray(accepted, dtype=float)
    return sample_large_case_layout(
        n_turbines, spacing, forbidden_zones, obstacle_margin, rng
    )


def count_site_violations(layout: Sequence[Sequence[float]]) -> int:
    points = np.asarray(layout, dtype=float)
    return sum(
        not SAFE_SITE_GEOMETRY.covers(Point(float(x), float(y)))
        for x, y in points
    )


def is_large_case_layout_feasible(
    layout: Sequence[Sequence[float]],
    min_spacing: float,
    forbidden_zones: Iterable[dict] = FORBIDDEN_ZONES,
    obstacle_margin: float = 0.0,
) -> bool:
    return layout_violation_score(
        layout, min_spacing, forbidden_zones, obstacle_margin
    ) <= 1e-6


def build_wind_scenarios() -> list[dict]:
    scenarios = []
    for met_direction, speed, frequency in zip(
        WIND_ROSE_MET_DIRECTIONS_DEG,
        WIND_ROSE_MEAN_SPEEDS_MS,
        WIND_ROSE_FREQUENCIES,
    ):
        scenarios.append({
            "wind_speed": float(speed),
            "wind_direction": float((met_direction + 180.0) % 360.0),
            "meteorological_direction": float(met_direction),
            "weight": float(frequency),
        })
    return scenarios


def dominant_wind_scenario() -> dict:
    return dict(max(build_wind_scenarios(), key=lambda row: row["weight"]))


def case_metadata() -> dict:
    return {
        "case_id": CASE_ID,
        "case_name": CASE_NAME,
        "case_type": "semi-real 50-turbine adaptation",
        "source_url": CASE_SOURCE_URL,
        "paper_url": CASE_PAPER_URL,
        "source_reference_system": "ETRS89 / UTM zone 31U, local coordinates",
        "official_case_turbines": 81,
        "adapted_case_turbines": NUM_TURBINES,
        "turbine": {
            "model_basis": "IEA 10 MW reference turbine",
            "rated_power_mw": RATED_POWER_MW,
            "rotor_diameter_m": ROTOR_DIAMETER_M,
            "hub_height_m": HUB_HEIGHT_M,
            "minimum_spacing_rotor_diameters": MIN_SPACING_FACTOR,
        },
        "grid_size_m": list(GRID_SIZE),
        "boundary_margin_m": BOUNDARY_MARGIN_M,
        "substation_positions_m": SUBSTATION_POSITIONS.tolist(),
        "site_regions": {
            name: [list(point) for point in coordinates]
            for name, coordinates in SITE_REGIONS.items()
        },
        "wind_rose": {
            "aggregation": "official 360 directions aggregated into 12 sectors",
            "meteorological_directions_deg": WIND_ROSE_MET_DIRECTIONS_DEG.tolist(),
            "frequencies": WIND_ROSE_FREQUENCIES.tolist(),
            "conditional_mean_speeds_ms": WIND_ROSE_MEAN_SPEEDS_MS.tolist(),
        },
    }


def write_case_definition(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case_metadata(), indent=2), encoding="utf-8")

