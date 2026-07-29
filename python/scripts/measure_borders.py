# python/scripts/measure_borders.py
"""Acceptance gate: pooled interior-country-border macro tortuosity and country
size balance for the bottom-up partitioner. Prints numbers; exits non-zero if
below the spec thresholds (macro tortuosity >= 1.6, country size CV <= 0.45)."""
import sys
import numpy as np
from shapely.ops import linemerge
from mimesis_earth.spec import WorldSpec
from mimesis_earth.generate import generate


def lines_of(sh):
    if sh.geom_type == "LineString":
        return [sh]
    m = linemerge(sh)
    return list(getattr(m, "geoms", [m]))


def macro_tortuosity(world):
    units = world.units_at(0)
    lm = {u.id: u.landmass for u in units}
    t, w = [], []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            if lm[a.id] != lm[b.id] or not a.geometry.intersects(b.geometry):
                continue
            sh = a.geometry.boundary.intersection(b.geometry.boundary)
            if sh.is_empty or sh.length == 0:
                continue
            for ln in lines_of(sh):
                if ln.geom_type != "LineString" or len(ln.coords) < 2:
                    continue
                p0, p1 = ln.coords[0], ln.coords[-1]
                span = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
                if span < 3:
                    continue
                t.append(ln.simplify(1.2).length / span)
                w.append(span)
    if not w:
        return None
    w = np.array(w)
    return float((np.array(t) * w).sum() / w.sum())


def country_cv(world):
    a = np.array([u.area_km2 for u in world.units_at(0)])
    return float(a.std() / a.mean())


torts, cvs = [], []
for seed in range(6):
    w = generate(WorldSpec(n_landmasses=3, levels=[6, 5, 6], resolution=20000,
                           land_fraction=0.35, seed=seed))
    mt = macro_tortuosity(w)
    if mt is not None:
        torts.append(mt)
    cvs.append(country_cv(w))

mt = float(np.mean(torts))
cv = float(np.mean(cvs))
print(f"pooled macro tortuosity (interior country borders): {mt:.3f}  (>= 1.60)")
print(f"country area CV: {cv:.3f}  (<= 0.45)")
ok = mt >= 1.60 and cv <= 0.45
print("ACCEPTANCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
