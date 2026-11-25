"""Verify the archived design exports without importing training modules."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]


def audit_design(design, turbines, reliability):
    nodes = design['nodes']
    pos = {n['id']: (float(n['x']), float(n['y'])) for n in nodes}
    turbine_ids = [n['id'] for n in nodes if n['type'] == 'turbine']
    stations = set(pos) - set(turbine_ids)
    graph = nx.Graph()
    graph.add_nodes_from(pos)
    edges = [(e['u'], e['v']) for e in design['edges']]
    graph.add_edges_from(edges)
    assert len(turbine_ids) == turbines and len(stations) == 1
    lengths = [math.dist(pos[u], pos[v]) for u, v in edges]
    spacing = min(math.dist(pos[u], pos[v]) for u, v in itertools.combinations(turbine_ids, 2))
    crossings = sum(LineString([pos[u], pos[v]]).intersects(LineString([pos[a], pos[b]]))
                    for (u, v), (a, b) in itertools.combinations(edges, 2) if not {u, v} & {a, b})
    connected = nx.is_connected(graph)
    if turbines == 25:
        outside = sum(not (199.97 <= pos[i][0] <= 6300.03 and 199.97 <= pos[i][1] <= 4800.03) for i in turbine_ids)
        circles = [((1500, 1000), 200), ((3000, 2000), 150)]
        clearance = sum(math.dist(pos[i], c) < radius + 50 - .03 for i in turbine_ids for c, radius in circles)
        obstacle = sum(LineString([pos[u], pos[v]]).distance(Point(c)) < radius - .03 for u, v in edges for c, radius in circles)
        minimum, max_length = 656, 4500
    else:
        case = design['case_definition']
        site = unary_union([Polygon(points).buffer(0) for points in case['site_regions'].values()])
        safe = site.buffer(-case['boundary_margin_m']).buffer(.03)
        outside = sum(not safe.covers(Point(pos[i])) for i in turbine_ids)
        clearance = obstacle = 0
        minimum, max_length = 693, 8500
    bridges = len(list(nx.bridges(graph)))
    strict = bool(connected and spacing >= minimum-.03 and max(lengths) <= max_length+.03 and
                  crossings == 0 and outside == 0 and clearance == 0 and obstacle == 0 and (not reliability or bridges == 0))
    return {'strict': strict, 'connected': connected, 'cable_km': sum(lengths)/1000,
            'spacing_m': spacing, 'crossings': crossings, 'boundary_violations': outside,
            'exclusion_clearance_violations': clearance, 'bridges': bridges}


def main():
    rows = list(csv.DictReader((ROOT / 'data/run_metrics.csv').open(encoding='utf-8')))
    designs = json.loads((ROOT / 'data/final_designs.json').read_text(encoding='utf-8'))
    assert len(rows) == len(designs) == 192
    assert len({r['run_id'] for r in rows}) == 192
    result = []
    for r in rows:
        j = designs[r['run_id']]
        check = audit_design(j, int(r['turbines']), int(r['version']) == 10)
        assert check['strict'] == (r['strict_pass'] == 'True'), (r['run_id'], check)
        assert abs(check['cable_km'] - float(r['cable_km'])) < 1e-7, r['run_id']
        assert abs(j['cost']['total']/1e6 - float(r['score_million_units'])) < 1e-8, r['run_id']
        assert abs(j['cost']['net_power'] - float(r['net_power_mw'])) < 1e-5, r['run_id']
        assert int(r['episodes']) == 2500
        assert all(abs(n.get('yaw_deg', 0)) <= 25.00001 for n in j['nodes'])
        result.append({'run_id': r['run_id'], **check})
    for r in json.loads((ROOT / 'data/source_code_manifest.json').read_text(encoding='utf-8')):
        assert hashlib.sha256((ROOT / r['file']).read_bytes()).hexdigest() == r['sha256'], r['file']
    summary = {'runs': len(result), 'strict_pass': sum(r['strict'] for r in result),
               'strict_25t': sum(r['strict'] for r in result if r['run_id'].startswith('25T')),
               'strict_50t': sum(r['strict'] for r in result if r['run_id'].startswith('50T')),
               'source_modules_verified': 35}
    print(json.dumps(summary, indent=2))
    assert summary['strict_pass'] == 184


if __name__ == '__main__':
    main()
