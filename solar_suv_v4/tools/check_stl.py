#!/usr/bin/env python3
"""Print-readiness check for STL files (pip install trimesh numpy).
Usage: python3 check_stl.py stl/*.stl
Reports: watertight, every edge shared by exactly two faces, shells, volume, size,
bed-contact area, and area facing down more than 45 degrees (bridges/supports)."""
import sys, math
from collections import Counter
import numpy as np, trimesh

for path in sys.argv[1:]:
    m = trimesh.load(path)
    edges = Counter(len(g) for g in trimesh.grouping.group_rows(m.edges_sorted))
    n, a, c = m.face_normals, m.area_faces, m.triangles_center
    zmin = m.bounds[0, 2]
    bed = (c[:, 2] < zmin + 1e-3) & (n[:, 2] < -0.99)
    over = (n[:, 2] < -math.cos(math.radians(45)) - 1e-3) & ~bed
    size = m.bounds[1] - m.bounds[0]
    print(f"{path}\n  faces {len(m.faces):,}  watertight {m.is_watertight}  "
          f"edges shared by 2 faces only {set(edges) == {2}}  shells {len(m.split(only_watertight=False))}\n"
          f"  volume {m.volume/1000:.1f} cm3  size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm\n"
          f"  bed contact {a[bed].sum():.0f} mm2  downward > 45 deg {a[over].sum():.0f} mm2")
