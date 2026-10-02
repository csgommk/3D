#!/usr/bin/env python3
"""Kit figures from the colour assembly preview (stl/kit/kit_assembly_preview.glb).

    python3 tools/render_kit_figs.py stl/kit/kit_assembly_preview.glb figs/

Writes kit_cutaway.png, kit_exploded.png, kit_chassis_top.png, kit_chassis_iso.png,
kit_cabin.png and kit_front_bay.png.  Coordinates are model mm (x rearward, y right,
z up from the print bed), as written by solar_suv_v4_kit.assembly_preview.
"""
import os, sys
import numpy as np
import trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_webgl import Scene

SHELLS = ("front", "rear", "door_fl", "door_fr", "door_rl", "door_rr", "hood", "canopy")
BG = (0.93, 0.94, 0.95)


def load(path):
    sc = trimesh.load(path, force="scene")
    out = {}
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        col = np.asarray(m.visual.face_colors[0][:3], float)/255
        out[node] = (np.asarray(m.vertices), np.asarray(m.faces), col)
    return out


def scene(P, pick, offsets=None, clip=None, ghost=(), w=1800, h=1050):
    s = Scene(w, h, bg=BG)
    for name, (v, f, col) in P.items():
        if not pick(name):
            continue
        d = np.asarray((offsets or {}).get(_group(name), (0, 0, 0)), float)
        c = clip(name) if clip else None
        if name in ghost:
            s.add(v + d, f, col, alpha=0.18, clip=c, spec=0.2)
        else:
            s.add(v + d, f, col, clip=c, spec=0.45 if name in SHELLS else 0.3)
    return s


def _group(name):
    if name.startswith("chassis_"):
        return "chassis"
    if name.startswith("cabin_") or name == "steering_wheel":
        return "cabin"
    return name


def main(glb, out):
    os.makedirs(out, exist_ok=True)
    P = load(glb)
    lo = np.min([v.min(0) for v, f, c in P.values()], 0); hi = np.max([v.max(0) for v, f, c in P.values()], 0)
    ctr = (lo + hi)/2
    L = hi[0] - lo[0]
    allp = lambda n: True
    # 1. cutaway: driver half (y < 0) of every shell removed, canopy lifted and ghosted
    cut = lambda n: [0, -1, 0, 0] if n in SHELLS else None
    s = scene(P, lambda n: n != "cargo_deck", offsets={"canopy": (0, 0, 22)}, clip=cut, ghost=("canopy",))
    s.camera(ctr + np.array([-0.62*L, -0.85*L, 0.62*L]), ctr + np.array([-4, 0, -6]), fov=26)
    s.render(os.path.join(out, "kit_cutaway.png"))
    # 2. exploded view
    off = dict(chassis=(0, 0, 0), cabin=(0, 0, 24), cargo_deck=(0, 0, 40), front=(-34, 0, 38), rear=(34, 0, 38),
               door_fl=(0, -42, 30), door_rl=(0, -42, 30), door_fr=(0, 42, 30), door_rr=(0, 42, 30),
               hood=(-34, 0, 62), canopy=(0, 0, 72))
    s = scene(P, allp, offsets=off)
    s.camera(ctr + np.array([-1.0*L, -1.1*L, 0.9*L]), ctr + np.array([0, 0, 26]), fov=29)
    s.render(os.path.join(out, "kit_exploded.png"))
    # 3. chassis plan view and 3/4 view (skateboard with the cabin floor removed)
    chas = lambda n: n.startswith("chassis_")
    s = scene(P, chas, w=1800, h=900)
    s.camera((ctr[0], ctr[1], 400), (ctr[0], ctr[1], 0), up=(0, 1, 0), ortho=0.27*L)    # nose to the left
    s.render(os.path.join(out, "kit_chassis_top.png"))
    s = scene(P, chas)
    s.camera(ctr + np.array([-0.55*L, -0.7*L, 0.55*L]), ctr + np.array([0, 0, -12]), fov=26)
    s.render(os.path.join(out, "kit_chassis_iso.png"))
    # 4. cabin: canopy and left doors off, looking in from the rear left
    s = scene(P, lambda n: n not in ("canopy", "door_fl", "door_rl", "cargo_deck"))
    s.camera(ctr + np.array([0.42*L, -0.55*L, 0.62*L]), ctr + np.array([-18, 0, 6]), fov=28)
    s.render(os.path.join(out, "kit_cabin.png"))
    # 5. front bay: hood off, close up from the front left
    s = scene(P, lambda n: n != "hood", clip=lambda n: [0, -1, 0, 0] if n == "front" else None)
    fb = np.array([lo[0] + 0.17*L, 0, lo[2] + 0.45*(hi[2] - lo[2])])
    s.camera(fb + np.array([-0.36*L, -0.34*L, 0.34*L]), fb, fov=30)
    s.render(os.path.join(out, "kit_front_bay.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stl/kit/kit_assembly_preview.glb", sys.argv[2] if len(sys.argv) > 2 else "figs")
