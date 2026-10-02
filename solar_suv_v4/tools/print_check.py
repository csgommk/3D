#!/usr/bin/env python3
"""Layer-by-layer support check for FDM (pip install numpy scipy scikit-image manifold3d trimesh).

Slices a part like a slicer (default 0.2 mm layers, 0.1 mm pixels) and finds every pixel of a
layer that is farther than one layer height (45 deg) from the material of the layer below.
Each unsupported region is classified:
  slope       within 0.3 mm of support (normal stair-stepping, prints fine)
  bridge      a flat span anchored on two opposite sides, span <= --bridge mm
  long bridge anchored on two sides, longer than --bridge
  cantilever  anchored on one side only (needs support if longer than --cant mm)
  island      no support at all below (needs support)

A region counts as bridged when its anchors form groups on opposite sides of it (or surround it).

Usage: python3 print_check.py stl/kit/*.stl [--layer 0.2] [--px 0.1] [--bridge 12] [--cant 0.8] [--json out.json]
       python3 print_check.py --selftest
"""
import argparse, json, math, sys
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
import trimesh
from manifold3d import Manifold, Mesh


def load(path):
    m = trimesh.load(path, force="mesh")
    man = Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    if man.status().name != "NoError":
        mesh = Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32))
        mesh.merge(); man = Manifold(mesh)
    if man.status().name != "NoError" or man.is_empty():
        raise ValueError(f"{path}: not a closed 2-manifold mesh ({man.status().name}); run check_stl.py")
    return man


def raster_polys(polys, lo, px, shape):
    """Winding-rule raster of slice contours (PIL scanline fill per contour, bbox-local)."""
    acc = np.zeros(shape, np.int16)
    for p in polys:
        p = np.asarray(p, float)
        if len(p) < 3:
            continue
        a = 0.5*np.sum(p[:, 0]*np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1)*p[:, 1])
        r = (p[:, 0] - lo[0])/px - 0.5; c = (p[:, 1] - lo[1])/px - 0.5
        r0, r1 = max(0, int(math.floor(r.min()))), min(shape[0], int(math.ceil(r.max())) + 1)
        c0, c1 = max(0, int(math.floor(c.min()))), min(shape[1], int(math.ceil(c.max())) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        img = Image.new("1", (c1 - c0, r1 - r0), 0)
        ImageDraw.Draw(img).polygon(list(zip(c - c0, r - r0)), fill=1)
        acc[r0:r1, c0:c1] += np.asarray(img, np.int16)*(1 if a > 0 else -1)
    return acc > 0

def raster(man, z, lo, px, shape):
    return raster_polys(man.slice(z).to_polygons(), lo, px, shape)


def anchored_on_two_sides(amask, c):
    """A region is bridged when its anchor pixels form groups on opposite sides of it (two group
    centroids more than 135 degrees apart around the region centroid c), or when the anchors
    surround it (no angular gap above 90 degrees).  One anchored edge, even a long or notched
    one, is a cantilever."""
    lab, n = ndimage.label(amask, structure=np.ones((3, 3), bool))
    if n == 0:
        return False
    pts = np.argwhere(amask).astype(float) - c
    ang = np.sort(np.arctan2(pts[:, 1], pts[:, 0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2*math.pi])
    if gaps.max() < math.pi/2:
        return True
    cen = np.array(ndimage.center_of_mass(amask, lab, range(1, n + 1))) - c
    cen = cen[np.linalg.norm(cen, axis=1) > 0]
    if len(cen) < 2:
        return False
    u = cen/np.linalg.norm(cen, axis=1, keepdims=True)
    return float((u @ u.T).min()) < -math.cos(math.radians(45))


def check(man, layer=0.2, px=0.1, bridge=12.0, cant=0.8, slope_tol=0.3):
    b = man.bounding_box()
    lo = np.array(b[:3]) - 2*px
    shape = (int(math.ceil((b[3] - b[0])/px)) + 5, int(math.ceil((b[4] - b[1])/px)) + 5)
    nz = int(math.ceil((b[5] - b[2])/layer))
    allow = layer/px + 0.5                                       # 45 deg in pixels, + half-pixel rasterisation
    prev = None
    out = dict(slope_mm2=0.0, bridges=[], long_bridges=[], cantilevers=[], islands=[])
    for k in range(nz):
        z = b[2] + (k + 0.5)*layer
        cur = raster(man, z, lo, px, shape)
        if prev is None:
            prev = cur
            continue
        if not cur.any():
            prev = cur
            continue
        if not prev.any():
            D = np.full(shape, np.inf)
        else:
            D = ndimage.distance_transform_edt(~prev)
        U = cur & (D > allow)
        if U.any():
            # cheap pass: regions that never reach beyond the slope tolerance are plain slopes
            big = U & (D > allow + slope_tol/px)
            if not big.any():
                out["slope_mm2"] += float(U.sum())*px*px
                prev = cur
                continue
            lab, n = ndimage.label(U)
            sup = cur & ~U                                            # supported material in this layer
            keep = np.unique(lab[big])
            out["slope_mm2"] += float((U & ~np.isin(lab, keep)).sum())*px*px
            objs = ndimage.find_objects(lab)
            for i in keep:
                sl = objs[i - 1]
                sl = tuple(slice(max(0, q.start - 3), q.stop + 3) for q in sl)
                comp = lab[sl] == i
                Dl = D[sl]
                reach = float(Dl[comp].max() - allow)*px              # how far it reaches beyond the 45-deg limit
                area = float(comp.sum())*px*px
                ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
                anchors = np.argwhere(ring & sup[sl])
                xs, ys = np.nonzero(comp)
                ox, oy = sl[0].start, sl[1].start
                rec = dict(z=round(z, 2), area_mm2=round(area, 2), reach_mm=round(reach, 2),
                           x=[round(lo[0] + (ox + xs.min())*px, 1), round(lo[0] + (ox + xs.max())*px, 1)],
                           y=[round(lo[1] + (oy + ys.min())*px, 1), round(lo[1] + (oy + ys.max())*px, 1)])
                if len(anchors) == 0 or not np.isfinite(Dl[comp]).all():
                    out["islands"].append(rec)
                    continue
                c = np.c_[xs, ys].mean(0)
                two_sided = anchored_on_two_sides(ring & sup[sl], c)
                span = 2*reach + 2*allow*px
                rec["span_mm"] = round(span, 2)
                if two_sided:
                    (out["bridges"] if span <= bridge else out["long_bridges"]).append(rec)
                elif reach > cant:
                    out["cantilevers"].append(rec)
                else:
                    out["slope_mm2"] += area
        prev = cur
    out["slope_mm2"] = round(out["slope_mm2"], 1)
    out["summary"] = dict(bridges=len(out["bridges"]), bridge_area_mm2=round(sum(r["area_mm2"] for r in out["bridges"]), 1),
                          max_bridge_mm=max([r["span_mm"] for r in out["bridges"]], default=0.0),
                          long_bridges=len(out["long_bridges"]), cantilevers=len(out["cantilevers"]),
                          max_cantilever_mm=max([r["reach_mm"] for r in out["cantilevers"]], default=0.0),
                          islands=len(out["islands"]), max_island_mm2=max([r["area_mm2"] for r in out["islands"]], default=0.0))
    return out


def selftest():
    """Known shapes: 10 mm bridge, hairline bridge, roof over a hole, cantilever, notched-edge
    cantilever, island."""
    box = lambda x0, x1, y0, y1, z0, z1: Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    m = box(0, 3, 0, 10, 0, 10) + box(13, 16, 0, 10, 0, 10) + box(0, 16, 0, 10, 10, 11)
    m = m + box(25, 28, 0, 10, 0, 10) + box(25, 31, 0, 10, 10, 11)
    m = m + box(36, 38, 4, 6, 6, 7)
    m = m + box(45, 47, 0, 2, 0, 8) + box(57, 59, 0, 2, 0, 8) + box(45, 59, 0.95, 1.05, 7.9, 8.0)
    m = m + box(65, 68, 0, 4, 0, 10) + box(65, 68, 6, 10, 0, 10) + box(65, 71, 0, 10, 10, 11)
    m = m + (box(80, 92, 0, 12, 0, 10) - box(83, 89, 3, 9, -1, 10)) + box(80, 92, 0, 12, 10, 11)
    r = check(m)
    got = sorted((round(q["x"][0]), k) for k in ("bridges", "cantilevers", "islands") for q in r[k])
    want = [(3, "bridges"), (28, "cantilevers"), (36, "islands"), (47, "bridges"), (65, "cantilevers"), (83, "bridges")]
    print("selftest", "ok" if got == want else f"FAILED: {got}")
    return got == want


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--px", type=float, default=0.1)
    ap.add_argument("--bridge", type=float, default=12.0)
    ap.add_argument("--cant", type=float, default=0.8)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for f in a.files:
        try:
            r = check(load(f), a.layer, a.px, a.bridge, a.cant)
        except ValueError as e:
            print(f"{f}\n  ERROR {e}")
            continue
        res[f] = r
        s = r["summary"]
        print(f"{f}\n  bridges {s['bridges']} (max {s['max_bridge_mm']} mm, {s['bridge_area_mm2']} mm2)  long bridges {s['long_bridges']}"
              f"  cantilevers {s['cantilevers']} (max {s['max_cantilever_mm']} mm)  islands {s['islands']}  slopes {r['slope_mm2']} mm2")
        for kind in ("long_bridges", "cantilevers", "islands"):
            for rec in sorted(r[kind], key=lambda q: -q["area_mm2"])[:6]:
                print(f"    {kind[:-1]}: z {rec['z']}  x {rec['x']}  y {rec['y']}  area {rec['area_mm2']} mm2  reach {rec['reach_mm']} mm")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
