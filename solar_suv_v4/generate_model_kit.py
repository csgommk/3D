#!/usr/bin/env python3
"""Solar SUV v4 - automated 1:20 model-kit pipeline.

    python3 generate_model_kit.py                     # full resolution, about 15 minutes
    python3 generate_model_kit.py --res 12            # quick draft
    python3 generate_model_kit.py --out DIR --no-3mf

Builds 20 modular, separately printable parts of the cutaway kit from the parametric
generators (solar_suv_v4_1to20.py exterior, solar_suv_v4_kit.py internals):

    stls/01_chassis/            chassis_baseplate_m2_bosses, front/rear_subframe_suspension
    stls/02_powertrain/         battery_tray_lower_housing, battery_module_cells_combined,
                                front/rear_e_axle_and_inverter, hv_wiring_harness_conduits
    stls/03_interior/           interior_floor_tub, front_bucket_seats_pair, rear_bench_seat,
                                dashboard_steering_console, door_cards_set
    stls/04_body_shell/         removable_roof_solar_canopy, hood_solar_bonnet,
                                main_outer_body_shell, tailgate_rear_hatch
    stls/05_wheels_and_hardware/ wheels_rims_x4, tires_treaded_x4, clear_lens_headlights_taillights

Every male/female joint (pins, pegs, magnet pockets, screw bosses, plug sockets, lens windows,
tyre on rim, axle bores) has 0.25 mm clearance per side.  Where two parts touch, a 0.25 mm
Minkowski offset of one part is cut from the other (open-top slots in the part installed first,
bottom-open notches in the part that goes on later).  Each file is exported as a binary STL in
model millimetres (1 unit = 1 mm at 1:20) with its bed face at z = 0, plus one 3MF project per
folder.  Everything is then validated: watertight, 2-manifold edges, pieces, scale, bed contact,
layer-by-layer supports, measured joint clearances and female features, interference and
assembly paths.  The script writes ASSEMBLY_AND_PRINT_GUIDE.md and kit_manifest.json next to the
stls/ folder and exits with status 1 if any check fails.
"""
import argparse, importlib, json, math, os, subprocess, sys, time, zipfile

# ---------------------------------------------------------------------------
# 0. dependencies
# ---------------------------------------------------------------------------
REQUIRED = {"numpy": "numpy", "scipy": "scipy", "skimage": "scikit-image", "manifold3d": "manifold3d",
            "shapely": "shapely", "fast_simplification": "fast-simplification", "trimesh": "trimesh",
            "PIL": "pillow"}
OPTIONAL = {"cadquery": "cadquery", "solid2": "solidpython2"}     # not needed by the build; reported only


def ensure_deps(install=True):
    """Import every required package, pip-installing the missing ones."""
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing and install:
        print("installing:", " ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    still = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if still:
        sys.exit("missing packages: " + " ".join(still))
    return {mod: importlib.util.find_spec(mod) is not None for mod in OPTIONAL}


HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    OPTIONAL_FOUND = ensure_deps()
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np
from manifold3d import Manifold, Mesh
from shapely.geometry import Polygon, Point, LineString, box as sbox
from shapely.ops import unary_union
import solar_suv_v4_1to20 as v4
import solar_suv_v4_kit as K

M = K.M                                   # model mm -> full-size mm
SCALE = K.SCALE

# ---------------------------------------------------------------------------
# 1. tolerances (model mm).  CLR is the clearance on every mating face of a male/female joint.
# ---------------------------------------------------------------------------
CLR = 0.25
TOL = dict(clearance_per_side=CLR, magnet=(3.0, 2.0), magnet_pocket=(3.0 + 2*CLR, 2.0 + CLR),
           pin=1.5, pin_socket=1.5 + 2*CLR, peg=2.0, peg_socket=2.0 + 2*CLR, axle=3.0, axle_bore=3.0 + 2*CLR,
           m2_pilot=1.6, m2_clear=2.0 + 2*CLR, m2_head=3.5 + 2*CLR, m3_pilot=2.5, m3_clear=3.0 + 2*CLR,
           m3_head=5.7 + 2*CLR, m2_head_h=1.3, panel_gap=2*CLR)


def apply_tolerances():
    """Push the kit-wide clearances into the generator modules before anything is built."""
    K.GAP = CLR                                   # pin / peg sockets: d + 2*GAP
    K.SPLIT = CLR                                 # half-gap at every split face (hood, roof, tailgate, shell)
    K.S_GAP = M(K.SPLIT)
    K.MAG_D, K.MAG_H = TOL["magnet_pocket"]
    K.M2.update(clear=TOL["m2_clear"], head=TOL["m2_head"], head_h=TOL["m2_head_h"])
    K.M3.update(clear=TOL["m3_clear"], head=TOL["m3_head"], head_h=1.8)
    v4.AXLE_CLEAR = CLR
    # routing changes for a split chassis (see the engineering notes in the guide)
    K.COOL_F = (-280.0, -220.0)                   # pack coolant ports clear the front subframe side member
    K.CARGO_POSTS = ((4100.0, 600.0), (4500.0, 560.0))   # first post clear of the rear lower-arm plate
    K.LED_PASS = [(1300.0, 420.0), (1300.0, -420.0)]   # front-bay wire exits (rear lamps: through the rear wheel houses)
    K.CLIP_M2 = [(360.0, 770.0), (4480.0, 760.0)]  # rear body screws outboard of the rear struts
    K.CLIP_M3 = [(1450.0, 860.0), (3330.0, 860.0)]  # rocker-end body screws (M2): counterbore inside the rocker (>= 0.4 mm wall to
                                                    # its sloped outer face and to its inner face), clear of the harness connector pockets
                                                    # (x <= 1395 at |y| 880) and the door cards
    K.HOOD_MAG = [(1150.0, 640.0)]                 # rear hood magnets on the body-shell ledge (front pair: HOOD_POST)
    K.TUB_SCREWS = [(K.XMEMBERS[2] + 18, 760.0)]   # cabin-floor screws: pilot inside the battery side frame (>= 0.7 mm wall);
                                                    # the head counterbore is cut up through the sill trim (open notch)


C = M(CLR)                                        # 5 mm full size = 0.25 mm on the model
HOOD_POST = (395.0, 170.0)                        # front hood magnets: on posts of the radiator support
POST_TOP = 690.0                                  # cargo posts: top (the deck rests on them through its bosses)
POST_R = 55.0                                     # cargo post cap / deck boss radius (5.5 mm on the model)
ZT = 234.0                                        # undertray top: every chassis module stands here
BALL = None


def ball():
    global BALL
    if BALL is None:
        BALL = Manifold.sphere(C, 16)
    return BALL


def bbox(m):
    return np.array(m.bounding_box(), float)


def boxm(lo, hi):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return Manifold.cube((hi - lo).tolist()).translate(lo.tolist())


def overlaps(a, b, pad=0.0):
    ba, bb = bbox(a), bbox(b)
    return not (np.any(ba[3:] + pad < bb[:3]) or np.any(bb[3:] + pad < ba[:3]))


def carve(B, A, mode="ball", r=None, min_vol=1.0):
    """B minus a clearance offset of A, computed only near B.  mode:
      'slot'  - A is lowered in after B: 0.25 mm sideways and everything of B above A goes (open-top
                slots and recesses; nothing is dug under A's feet);
      'drape' - B is lowered over A after it: 0.25 mm sideways and everything of B below A goes
                (notches open at the bottom; B may rest on top of A);
      'drape_c' - as 'drape', with 0.25 mm above A as well;
      'ball'  - 0.25 mm all round.
    Returns (carved B, overlap volume removed in full-size mm3)."""
    r = r or C
    if B is None or A is None or B.is_empty() or A.is_empty() or not overlaps(A, B, r + 1):
        return B, 0.0
    bb = bbox(B)
    region = boxm(bb[:3] - r - 2, bb[3:] + r + 2)
    Aloc = A ^ region
    if Aloc.is_empty():
        return B, 0.0
    if mode == "slot":
        tool = Manifold.cylinder(2000.0, r, r, 16)
    elif mode == "drape":                          # B may rest on top of A
        tool = Manifold.cylinder(2000.5, r, r, 16).translate([0, 0, -2000.0])
    elif mode == "drape_c":                        # B is lowered over A and keeps 0.25 mm above it too
        tool = Manifold.cylinder(2000.0 + r, r, r, 16).translate([0, 0, -2000.0])
    else:
        tool = ball()
    D = Aloc.minkowski_sum(tool)
    removed = (B ^ D).volume()
    if removed < min_vol:
        return B, 0.0
    return B - D, removed


def largest(m):
    comps = m.decompose()
    if len(comps) <= 1:
        return m, 0.0
    comps.sort(key=lambda c: -c.volume())
    return comps[0], sum(c.volume() for c in comps[1:])


def keep_big(m, frac=0.002, min_vol=8000.0):
    """Drop boolean crumbs but keep every intended shell (sets such as seats, cards, module pairs)."""
    comps = m.decompose()
    if len(comps) <= 1:
        return m, 0.0
    vmax = max(c.volume() for c in comps)
    keep = [c for c in comps if c.volume() >= max(min_vol, frac*vmax)]
    drop = sum(c.volume() for c in comps if c.volume() < max(min_vol, frac*vmax))
    return K.union(keep), drop


# ---------------------------------------------------------------------------
# 2. registries: joints (measured after export), fasteners and magnets (bill of materials)
# ---------------------------------------------------------------------------
JOINTS = []        # dict(kind, male, female, p (full mm, installed), axis, male_d, depth, back, fdepth, blind (model mm))
SCREWS = []        # dict(size, length, where, parts)
MAGNETS = []       # dict(where, parts, count)


def joint(kind, male, female, p, axis, male_d, depth, back=None, fdepth=None, blind=False):
    """Register a male/female joint.  p: a point on the joint axis inside the female feature, axis
    pointing into it; back: distance (model mm) from p back to the female's mouth; fdepth: depth of the
    female feature from its mouth (model mm); blind: the female must have a floor (magnet pockets)."""
    JOINTS.append(dict(kind=kind, male=male, female=female, p=[float(v) for v in p],
                       axis=[float(v) for v in axis], male_d=male_d, depth=depth,
                       back=back, fdepth=fdepth, blind=blind))


def pin_solid(p, axis, d, length, tip=0.3, tear=True):
    """Printed locating pin (model sizes): cylinder with a chamfered tip, starting 5 mm inside the base."""
    p = np.asarray(p, float); a = np.asarray(axis, float); a /= np.linalg.norm(a)
    if abs(a[2]) > 0.9:
        body = K.cyl(p - a*5, p + a*M(length - tip), M(d/2), seg=24)
        cone = K.cyl(p + a*(M(length - tip) - 0.01), p + a*M(length), M(d/2), M(d/2 - tip), seg=24)
        return K.union([body, cone])
    return K.tube([p - a*5, p + a*M(length)], M(d/2), tear=tear, n=20)       # horizontal: teardrop keel


def socket(p, axis, d, depth, flat=False):
    """Female hole for a pin/peg/plug of diameter d (model mm): d + 2*CLR, depth + CLR.  flat: plain
    cylinder (a bridged roof) for holes that open on a bed face, instead of the 45-deg cone roof."""
    if flat:
        p = np.asarray(p, float); a = np.asarray(axis, float)
        return K.cyl(p - a, p + a*M(depth + CLR), M(d + 2*CLR)/2, seg=32)
    return K.teardrop_hole(p, axis, M(d + 2*CLR), M(depth + CLR) + 1)


# ---------------------------------------------------------------------------
# 3. chassis and powertrain groups (full-size mm, installed positions)
# ---------------------------------------------------------------------------
def sections_of(builder):
    """Run a kit builder with section bookkeeping on; returns {section: Manifold}."""
    K.SECTIONS = {}
    P = {}
    builder(P)
    out = {name: K.union([K.union(ms) for ms in tags.values()]) for name, tags in K.SECTIONS.items()}
    K.SECTIONS = None
    return out


def harness_split():
    """The kit harness re-cut for separate modules.  Runs lying on the undertray form the
    harness part; every vertical riser and the elevated run at its top travel with the module
    it feeds (the e-axle / thermal module), so no run spans two parts without support.
    Returns (floor_runs, front_pigtails, rear_pigtails, hvac_risers, connectors)."""
    zt, (r_hv, r_sol, r_cool, r_ref) = ZT, (13.0, 10.0, 14.0, 9.0)
    zc = lambda r: zt + r
    floor, fpig, rpig, hvac, conn = [], [], [], [], []
    X0, X1 = K.PACK_X0, K.PACK_X1
    FDU, RDU = K.FDU, K.RDU
    xr_f = K.X_FA + FDU["diff_r"] + r_hv                      # front riser line, just behind the diff
    zi = FDU["inv"][4] + 45
    xin = FDU["motor"][0] + FDU["motor"][2] + 15 + 24
    zp = 750.0
    for y in K.HV_PORTS_F:                                    # battery -> front inverter
        floor.append(K.tube([(X0 - 26, y, 290), (X0 - 40, y, zc(r_hv)), (xr_f, y, zc(r_hv))], r_hv))
        fpig.append(K.tube([(xr_f, y, zt), (xr_f, y, zi), (xin - 34, y, zi)], r_hv))
    for y in K.HV_PORTS_C[:2]:                                # battery <-> charging stack
        floor.append(K.tube([(X0 - 26, y, 290), (X0 - 40, y, zc(r_hv)), (xr_f, y, zc(r_hv))], r_hv))
        fpig.append(K.tube([(xr_f, y, zt), (xr_f, y, zp), (800, y, zp)], r_hv))
    xr_r = K.X_RA - RDU["diff_r"] - r_hv
    zr = RDU["inv"][4] + 45
    xrin = RDU["motor"][0] - RDU["motor"][2] - 15 - 24
    for y in K.HV_PORTS_R:                                    # battery -> rear inverter
        floor.append(K.tube([(X1 + 26, y, 290), (X1 + 40, y, zc(r_hv)), (xr_r, y, zc(r_hv))], r_hv))
        rpig.append(K.tube([(xr_r, y, zt), (xr_r, y, zr), (xrin + 34, y, zr)], r_hv))
    xk, yk = X0 - 56, K.INLET_Y                               # rocker-front connectors, cross-car lane
    conn.append(K.cbox(xk - 26, xk + 26, yk - 30, yk + 30, zt, K.Z_SKIRT, ct=6))
    floor.append(K.tube([(xk, yk - 30, zc(r_hv)), (xk, K.HV_PORTS_C[2] + 30, zc(r_hv)), (X0 - 26, K.HV_PORTS_C[2], zc(r_hv))], r_hv))
    ya = K.SOLAR_Y + 26                                       # AC branch: on top of the DC run, then to the OBC
    floor.append(K.tube([(xk, yk - 30, zc(r_hv) + 2*r_hv), (xk, ya, zc(r_hv) + 2*r_hv), (xk - 40, ya, zc(10)), (xr_f + 2, ya, zc(10))], 10))
    fpig.append(K.tube([(xr_f + 2, ya, zt), (xr_f + 2, ya, zp), (800, ya, zp)], 10))
    for s in (1, -1):
        y0 = s*yk if s < 0 else yk - 70
        conn.append(K.cbox(xk - 26, xk + 26, y0 - 22, y0 + 22, zt, K.Z_SKIRT, ct=6))
    ys = K.SOLAR_Y - 8
    floor.append(K.tube([(xk - 30, -yk + 22, zc(r_sol)), (xk - 30, ys, zc(r_sol)), (xr_f + 2, ys, zc(r_sol))], r_sol))
    floor.append(K.tube([(xk + 30, yk - 92, zc(r_sol)), (xk + 30, ys + 22, zc(r_sol)), (xk - 30, ys + 22, zc(r_sol))], r_sol))
    fpig.append(K.tube([(xr_f + 2, ys, zt), (xr_f + 2, ys, 830), (788, ys, 830)], r_sol))
    xd = 1000.0                                               # chiller <-> pack cold plates
    for y in K.COOL_F:
        fpig.append(K.tube([(800, y, 760), (xd, y, 760), (xd, y, zt)], r_cool))
        floor.append(K.tube([(xd, y, zc(r_cool)), (X0 - 30, y, zc(r_cool))], r_cool))
    xdr = xr_r - 2*r_hv - 6                                   # pack centre duct -> rear e-axle
    xri = RDU["motor"][0] - RDU["motor"][2] - 15 + 12
    for y in K.COOL_R:
        floor.append(K.tube([(X1 + 30, y, zc(r_cool)), (xdr, y, zc(r_cool))], r_cool))
        rpig.append(K.tube([(xdr, y, zt), (xdr, y, zr - 20), (xri, y, zr - 20)], r_cool))
        rpig.append(K.box(3785, 3815, y, y + 24, 395.0, zr - 20) if y < 0 else K.box(3785, 3815, y - 12, y + 12, 500.0, zr - 20))
    xq = 1026.0                                               # refrigerant to the cabin HVAC, riser behind the diff
    for y in K.REFRIG_Y:
        fpig.append(K.tube([(800, y, 770), (xq, y, 770), (xq, y, zt)], r_ref))
        floor.append(K.tube([(xq, y, zc(r_ref)), (K.HVAC_X - 40, y, zc(r_ref))], r_ref))
        hvac.append(K.tube([(K.HVAC_X - 40, y, zc(r_ref)), (K.HVAC_X - 40, y, 500)], r_ref))
    # carrier straps on the undertray tie each group of runs into one printable piece
    floor.append(K.box(1270, 1296, -296, 268, zt, zt + 10))
    floor.append(K.box(X1 + 50, X1 + 76, -166, 166, zt, zt + 10))
    floor.append(K.box(xk - 44, xk - 18, ys, ya + 10, zt, zt + 10))       # ties the AC / DC lane to the solar lane
    floor.append(K.box(xk + 18, xk + 44, ys + 12, K.HV_PORTS_C[2] + 30, zt, zt + 10))
    return floor, fpig, rpig, hvac, conn


def thermal_module():
    """Front-end module, heat pump, charging stack and carriers, with free-standing carrier posts
    beside the subframe (the kit's posts on the rails / subframe belong to other parts here),
    the radiator hoses and refrigerant lines between them, and two radiator support brackets."""
    S = sections_of(K.thermal_parts)
    parts = [S["frontend"], S["carrier_plates"], S["heatpump"], S["stack"]]
    for s in (1, -1):
        parts.append(K.box(610, 690, min(s*388, s*432), max(s*388, s*432), ZT, 700))
        y0, y1 = (220.0, 285.0) if s > 0 else (263.0, 287.0)        # between the inverter and the subframe
        parts.append(K.box(590, 790, min(s*y0, s*y1), max(s*y0, s*y1), ZT, 686))
        parts.append(Manifold.hull_points(np.array([(x, s*y, z) for x in (585.0, 795.0)
                                                    for y, z in ((y1 - 4, 500.0), (y1 - 4, 686.0), (432.0, 686.0))])))
    r_cool, r_ref = 14.0, 9.0
    parts.append(K.tube([(340, -150, 760 + r_cool), (600, -150, 760 + r_cool)], r_cool))
    parts.append(K.tube([(340, -60, 760 + r_cool), (630, -60, 760 + r_cool)], r_cool))
    parts.append(K.tube([(600, -230, 720), (575, -230, 720), (575, -230, 640)], r_cool))
    parts.append(K.tube([(640, -380, 862), (360, -380, 862), (360, -380, 760 + r_ref), (290, -380, 760 + r_ref)], r_ref))   # level, then down onto the shroud
    parts.append(K.tube([(300, -330, 760 + r_ref), (722, -310, 760 + r_ref)], r_ref))
    parts.append(K.tube([(700, 60, 770), (700, -240, 770), (690, -290, 790)], 10.0))   # PDU -> e-compressor
    for y0, y1, xe in ((-230.0, -200.0, 600.0), (245.0, 275.0, 592.0)):                # radiator support brackets
        parts.append(K.box(330, xe, y0, y1, 745, 760))
    parts.append(K.box(612, 668, -425, -292, 699, 748))                                 # e-compressor saddle on the plate
    z_post = K.hood_bottom() - K.S_GAP
    for s in (1, -1):                                   # hood magnet posts on the shroud (radiator support), 53-deg flare
        x, y = HOOD_POST
        flare = Manifold.hull_points(np.array([(xx, s*y + dy, 758.0) for xx in (342.0, 396.0) for dy in (-28, 28)]
                                              + K.section_pts(np.array([x, s*y, 830.0]), (0, 0, 1), POST_R, False, 40)))
        post = K.cyl((x, s*y, 829.99), (x, s*y, z_post), POST_R, seg=40)
        pocket = K.teardrop_hole((x, s*y, z_post + 1), (0, 0, -1), M(K.MAG_D), M(K.MAG_H) + 1)
        parts.append(K.union([flare, post]) - pocket)
    return K.union(parts)


X_RAMP = 4559.0          # the rear bumper ramp (body shell underside) starts here: aft of it the bumper
                          # beam, crush cans and upper struts are moulded into the body shell


def rear_bumper_structure(crash):
    """Bumper beam, crush cans (carried forward to the ramp) and upper struts, for the body shell."""
    aft = crash["rear_crash"] ^ boxm((X_RAMP, -1500, 0), (6000, 1500, 3000))
    cans = K.symy(K.cbox(X_RAMP, 4662, 300, 380, 560, 682, cs=14))
    return K.union([aft, cans])


def cargo_posts():
    """Cargo-deck posts: 3 mm square columns with a 50-deg flare to a 5.5 mm round cap, so the 3.5 mm
    magnet pocket keeps a 1 mm wall."""
    out = []
    for x, y in K.CARGO_POSTS:
        for s in (1, -1):
            col = K.box(x - 30, x + 30, s*y - 30, s*y + 30, ZT, POST_TOP - 60)
            flare = Manifold.hull_points(np.array([(x + dx, s*y + dy, POST_TOP - 90) for dx in (-30, 30) for dy in (-30, 30)]
                                                  + K.section_pts(np.array([x, s*y, POST_TOP - 60]), (0, 0, 1), POST_R, False, 40)))
            cap = K.cyl((x, s*y, POST_TOP - 60.01), (x, s*y, POST_TOP), POST_R, seg=40)
            out.append(K.union([col, flare, cap]))
    return K.union(out)


def cargo_deck_kit():
    """Cargo deck (as the kit's) with 2.5 mm bosses under the four magnet sites: the 1.2 mm deck alone is
    too thin for a blind 2.25 mm pocket.  Printed upside down (top face on the bed)."""
    x0, x1, hw = 3650.0, 4560.0, 640.0
    deck = K.cbox(x0, x1, -hw, hw, K.Z_CARGO[0], K.Z_CARGO[1], ct=6)
    bosses = [K.cyl((x, s*y, POST_TOP), (x, s*y, K.Z_CARGO[0] + 1), POST_R, seg=40) for x, y in K.CARGO_POSTS for s in (1, -1)]
    grooves = [K.box(x0 + 40, x1 - 40, y - M(0.3), y + M(0.3), K.Z_CARGO[1] - M(0.6), K.Z_CARGO[1] + 1) for y in np.arange(-560, 561, 80)]
    rings = [K.cyl((x, s*560, K.Z_CARGO[1] - M(0.6)), (x, s*560, K.Z_CARGO[1] + 1), 28, seg=24) for x in (3720, 4480) for s in (1, -1)]
    handle = K.box(4400, 4520, -60, 60, K.Z_CARGO[1] - M(0.8), K.Z_CARGO[1] + 1)
    towers = [K.cyl((3966, s*552, POST_TOP - 5), (3966, s*552, K.Z_CARGO[1] + 5), 90, seg=40) for s in (1, -1)]
    mags = [K.cyl((x, s*y, POST_TOP - 1), (x, s*y, POST_TOP + M(K.MAG_H)), M(K.MAG_D)/2, seg=32) for x, y in K.CARGO_POSTS for s in (1, -1)]
    return K.diff(K.union([deck] + bosses), grooves + rings + [handle] + mags + towers)


def chassis_groups(B, R):
    """All chassis / powertrain groups before the clearance pass (installed, full size)."""
    G = {}
    crash = sections_of(K.crash_parts)
    susp = sections_of(K.suspension_parts)
    pack = sections_of(K.pack_parts)
    # ---- baseplate: undertray + sills (solid rockers) + crash structure + locating pins
    hollow = B["inset_shell"] ^ boxm((-500, -1500, ZT), (6000, 1500, K.Z_SKIRT + 10))
    hollow = hollow - K.symy(K.box(1340, 3390, K.PACK_HW - 10, 1300, 0, K.Z_SKIRT + 20))
    skirt = (B["body"] ^ R["skirt"]) - hollow
    pins = []
    for xs in K.DOOR_PINS:
        for x in xs:
            yp = float(K.skin_y(x, [K.Z_SKIRT])[0]) - M(K.D_DOOR)/2
            for s in (1, -1):
                pins.append(pin_solid((x, s*yp, K.Z_SKIRT - K.S_GAP), (0, 0, 1), TOL["pin"], 2.0))
                joint("pin", "chassis_baseplate_m2_bosses", "main_outer_body_shell", (x, s*yp, K.Z_SKIRT + K.S_GAP + M(1.0)),
                      (0, 0, 1), TOL["pin"], 2.0, back=1.0, fdepth=2.25)
    cols = [K.symy(K.cbox(x - 60, x + 60, y - 60, y + 60, K.GC, K.Z_SKIRT - K.S_GAP)) ^ B["base"] for x, y in K.CLIP_M2]
    rear_posts = boxm((4300, -1500, 0), (6000, 1500, 3000))            # cargo posts on the rear struts
    posts = cargo_posts()
    G["baseplate"] = K.union([skirt, posts - rear_posts] + pins + cols)
    # ---- front / rear subframe with suspension, steering, sway bars and the crash structure
    #      (bumper beam, crush cans, rails and their towers), so nothing of the baseplate hangs over them
    rbrk = K.box(470, 500, -345, -315, K.SUB_Z[1] - 2, 760)             # refrigerant line rest
    G["front_sub"] = K.union([crash["crash"], crash["front_subframe"], susp["front_susp"], rbrk])
    mounts = K.symy(K.union([K.box(3480, 3600, 300, 480, ZT, 320),     # subframe mount blocks (screws outboard of the rails),
                             K.box(3480, 3600, 375, 480, ZT, 360)]))     # raised (fused to the rail) so the screw clamps >= 1.5 mm
    rails = crash["rear_crash"] ^ boxm((K.PACK_X1 + C, -1500, 0), (X_RAMP - 2*C, 1500, 3000))   # tray .. bumper ramp
    feet = K.symy(K.box(K.PACK_X1 + C, 3490, 300, 380, ZT, 330))         # rails carried down to the undertray
    # the lower struts end at the bumper ramp: a cross-tie at their ends joins them, the cargo posts and
    # (by two webs) the rail kick-ups into one piece
    tie = K.union([K.box(X_RAMP - 2*C - 34, X_RAMP - 2*C - 1, -690, 690, 380, 430),        # 1 mm short of the cut plane:
                   K.symy(K.box(X_RAMP - 2*C - 34, X_RAMP - 2*C - 1, 312, 368, 425, 505))])   # no coplanar faces
    G["rear_sub"] = K.union([crash["rear_subframe"], susp["rear_susp"], mounts, rails, feet, tie, posts ^ rear_posts])
    G["_rear_bumper"] = rear_bumper_structure(crash)
    # ---- battery: tray housing and the 16 modules (8 drop-in pairs) + BMS chain
    G["tray"] = pack["tray"]
    G["modules"] = K.union([pack["modules"], pack["bms_chain"]])
    # ---- e-axles with their pigtails; harness
    floor, fpig, rpig, hvac, conn = harness_split()
    P = {}; K.drive_unit(P, True); du_f = K.union([K.union(v) for v in P.values()])
    P = {}; K.drive_unit(P, False); du_r = K.union([K.union(v) for v in P.values()])
    # carrier plates tie each row of pigtail risers (0.9-1.4 mm pipes, up to 25 mm tall on the model) into a
    # rigid comb, and a web joins the two refrigerant risers of the harness
    fcar = K.box(1010, 1024, -285, 258, ZT, 700)
    rcar = K.box(3680, 3700, -158, 158, ZT, 690)
    web = K.box(K.HVAC_X - 46, K.HVAC_X - 34, min(K.REFRIG_Y), max(K.REFRIG_Y), ZT, 495)
    G["eaxle_front"] = K.union([du_f, thermal_module(), fcar] + fpig)
    G["eaxle_rear"] = K.union([du_r, rcar] + rpig)
    G["harness"] = K.union(floor + hvac + conn + [web])
    # every module stands on the undertray plane: keels of horizontal teardrops that dipped into the
    # undertray in the one-piece chassis are cut flat, so each part has a true flat bed face
    above = boxm((-500, -1500, ZT), (6000, 1500, 3000))
    for g in ("front_sub", "rear_sub", "tray", "modules", "eaxle_front", "eaxle_rear", "harness"):
        G[g] = G[g] ^ above
    return G


# ---------------------------------------------------------------------------
# 4. module mounts: M2 boss pads on the baseplate (pilot holes) and pockets in the modules.
#    'screw' mounts take an M2 button-head screw from above through the module; 'locator'
#    mounts only register the module (the axles capture the e-axles, the plugs the harness).
# ---------------------------------------------------------------------------
PAD_D, PAD_H, PILOT_DEPTH = 3.0, 1.5, 2.4          # model mm
STD_M2 = [3, 4, 5, 6, 8, 10, 12]
MOUNTS = [  # (group, x, y, kind, top of the module above the mount (full mm))
    ("front_sub", 1205.0, 365.0, "locator", 0.0), ("front_sub", 1205.0, -365.0, "locator", 0.0),
    ("rear_sub", 3540.0, 425.0, "screw", 360.0), ("rear_sub", 3540.0, -425.0, "screw", 360.0),
    ("tray", 1550.0, 0.0, "screw", 372.0), ("tray", 1550.0, 120.0, "screw", 372.0), ("tray", 3200.0, 0.0, "locator", 0.0),
    ("eaxle_front", 890.0, -90.0, "locator", 0.0), ("eaxle_front", 900.0, 157.0, "locator", 0.0),
    ("eaxle_rear", 3840.0, -90.0, "locator", 0.0), ("eaxle_rear", 3830.0, 160.0, "locator", 0.0),
]
FILE_OF = {}        # group key -> file stem (filled in by the part table)


def axle_bores():
    """Clearance bores for the 3 mm axles with a teardrop roof (point up, as the parts print upright)."""
    r = M(v4.AXLE_D/2 + v4.AXLE_CLEAR)
    out = []
    for xa in (K.X_FA, K.X_RA):
        pts = (K.section_pts(np.array([xa, -720.0, K.HUB_Z]), (0, 1, 0), r, True, 24, up=True)
               + K.section_pts(np.array([xa, 720.0, K.HUB_Z]), (0, 1, 0), r, True, 24, up=True))
        out.append(Manifold.hull_points(np.array(pts)))
    return K.union(out)


BODY_SCREW_ZBOT = {}      # (x, y) -> underside of the baseplate under the screw head (full mm)
DROPPED = {}              # file stem -> detached material removed by largest() (model mm3)


def bed_trim(m, z_bed, dz=0.2):
    """Trim the lowest dz (full mm, 0.01 mm on the model) off a part that rests on the plane z_bed.  A knife
    edge meeting the bed leaves a zero-thickness film exactly on that plane; a horizontal cut removes it
    without making new walls along the skin."""
    return m - K.box(-500, 6000, -1500, 1500, z_bed - 1, z_bed + dz)


def strip_bed_fins(m, z_bed, w=M(0.2), h=M(1.25), dz=M(0.03), min_area=0.02*SCALE**2):
    """Remove fins thinner than w that stand on the print bed z_bed (a knife edge or a sliver between two
    cuts): the section dz above the bed is opened by w, and every piece of more than min_area that the
    opening drops is cut away up to h.  Narrow contact strips under sloped faces are not fins and stay."""
    poly = K.slice_poly(m, z_bed + dz)
    thin = poly.difference(poly.buffer(-w/2).buffer(w/2))
    parts = [g for g in getattr(thin, "geoms", [thin]) if g.area > min_area]
    if not parts:
        return m
    return m - K.prism(unary_union(parts).buffer(M(0.025)), "xy", z_bed - 1, z_bed + h)


def underside_under(B, x, y, r):
    """Highest point of the body underside within radius r of (x, y): where a counterbore entered from
    below must start so that its whole rim is in material."""
    col = B["body"] ^ K.cyl((x, y, K.GC - 5), (x, y, K.Z_SKIRT), r, seg=24)
    lo, hi = K.GC, K.Z_SKIRT - 40.0
    full = math.pi*r*r*0.97
    for _ in range(14):
        mid = 0.5*(lo + hi)
        if K.slice_poly(col, mid).area >= full:
            hi = mid
        else:
            lo = mid
    return hi


def screw_hole_from_below(x, y, z_top, spec, z_bot):
    """Clearance hole + head counterbore entered from below; the counterbore starts at z_bot."""
    hh, rh = M(spec["head_h"]), M(spec["head"]/2)
    return K.union([K.cyl((x, y, K.GC - 1), (x, y, z_top + 1), M(spec["clear"]/2), seg=24),
                    K.cyl((x, y, K.GC - 1), (x, y, z_bot + hh), rh, seg=32)])     # flat seat: a 4 mm bridge


def mount_geometry():
    """Returns (pads for the baseplate, pilot holes for the baseplate, {group: [cuts]})."""
    pads, pilots, cuts = [], [], {}
    top = ZT + M(PAD_H)
    for g, x, y, kind, h_top in MOUNTS:
        pads.append(K.union([K.cyl((x, y, ZT - 1), (x, y, top - 6), M(PAD_D/2), seg=32),
                             K.cyl((x, y, top - 6.01), (x, y, top), M(PAD_D/2), M(PAD_D/2) - 6, seg=32)]))
        pocket_top = top + C
        c = [K.cyl((x, y, ZT - 1), (x, y, pocket_top), M(PAD_D/2) + C, seg=32),
             K.cyl((x, y, pocket_top - 0.01), (x, y, pocket_top + M(PAD_D/2) + C), M(PAD_D/2) + C, 0.0, seg=32)]
        joint("boss", "chassis_baseplate_m2_bosses", g, (x, y, ZT + M(PAD_H)/2), (0, 0, 1), PAD_D, PAD_H,
              back=PAD_H/2, fdepth=PAD_H + CLR)
        if kind == "screw":
            pilots.append(K.cyl((x, y, ZT - M(PILOT_DEPTH - PAD_H)), (x, y, top + 1), M(TOL["m2_pilot"]/2), seg=24))
            a = (h_top - pocket_top)/SCALE                       # module material above the pocket (model mm)
            # longest standard screw whose head still sits 0.25 mm below the surface; tip 0.25 mm short of the pilot bottom
            L = max(l for l in STD_M2 if a - (l - PILOT_DEPTH + CLR) >= TOL["m2_head_h"] + CLR)
            if L - PILOT_DEPTH + CLR < 1.5:
                raise ValueError(f"mount {g} at ({x:.0f}, {y:.0f}): only {L - PILOT_DEPTH + CLR:.2f} mm under the screw head")
            cb = a - L + PILOT_DEPTH - CLR                       # counterbore depth
            c.append(K.cyl((x, y, pocket_top - 1), (x, y, h_top + 400), M(TOL["m2_clear"]/2), seg=24))
            c.append(K.cyl((x, y, h_top - M(cb)), (x, y, h_top + 400), M(TOL["m2_head"]/2), seg=32))
            SCREWS.append(dict(size="M2", length=L, head="button head", group=g,
                               where=f"{FILE_OF[g]} to baseplate boss at x {x/SCALE:.1f}, y {y/SCALE:+.1f} mm (from above)",
                               clamp_mm=round(a - cb, 2), engagement_mm=round(L - (a - cb) - CLR, 2)))
            joint("screw", None, g, (x, y, pocket_top + M(0.5)), (0, 0, 1), 2.0, 1.0, back=0.5,
                  fdepth=(h_top - M(cb) - pocket_top)/SCALE - 0.1)
            joint("cbore", None, g, (x, y, h_top - M(0.5)), (0, 0, -1), TOL["m2_head"] - 2*CLR, 1.0, back=0.5,
                  fdepth=cb - 0.05)
        cuts.setdefault(g, []).extend(c)
    return pads, pilots, cuts


# ---------------------------------------------------------------------------
# 5. interior (cabin) parts
# ---------------------------------------------------------------------------
SEAT_PEGS = dict(seats=[(2120.0, -490.0), (2450.0, -290.0), (2120.0, 290.0), (2450.0, 490.0)],
                 bench=[(3100.0, -370.0), (3100.0, 370.0), (3250.0, 0.0)],
                 dashboard=[(1450.0, 0.0), (2450.0, 0.0)])


def interior_groups(envelope):
    K.SECTIONS = {}
    P = K.interior_parts(envelope=envelope)
    S, K.SECTIONS = K.SECTIONS, None
    sec = {name: K.union([K.union(ms) for ms in tags.values()]) for name, tags in S.items()}
    holes = P["holes"]                                   # 2 tub-peg holes, 4 tub-screw holes, steering socket
    zf = K.Z_FLOOR
    above = boxm((-500, -1500, zf), (6000, 1500, 2600))
    G = {}
    pegs, sockets = [], {}
    for g, pts in SEAT_PEGS.items():
        for x, y in pts:
            pegs.append(pin_solid((x, y, zf), (0, 0, 1), TOL["peg"], 1.5))
            sockets.setdefault(g, []).append(socket((x, y, zf), (0, 0, 1), TOL["peg"], 1.5))
            joint("peg", "interior_floor_tub", g, (x, y, zf + M(0.75)), (0, 0, 1), TOL["peg"], 1.5, back=0.75, fdepth=1.75)
    zb = K.Z_FLOOR + M(2.0)                               # top of the floor screw bosses (interior_parts)
    tub_holes = list(holes[:2])
    for x, y in K.TUB_SCREWS:
        for s in (1, -1):
            tub_holes += [K.cyl((x, s*y, K.Z_TUB - 1), (x, s*y, zb + 1), M(TOL["m2_clear"]/2), seg=24),
                          K.cyl((x, s*y, zb - M(TOL["m2_head_h"]) - C), (x, s*y, zb + M(2.5)), M(TOL["m2_head"]/2), seg=32)]
    G["floor_tub"] = K.diff(K.union([sec["floor"]] + pegs), tub_holes)
    G["seats"] = K.diff(sec["seats_front"] ^ above, sockets["seats"])
    G["bench"] = K.diff(sec["bench"] ^ above, sockets["bench"])
    a = np.array([math.cos(math.radians(K.SW_TILT)), 0, math.sin(math.radians(K.SW_TILT))])
    hub = np.array([K.SW_C[0], -K.SW_C[1], K.SW_C[2]]) - a*45
    sock = Manifold.hull_points(np.array(K.section_pts(hub + a*2, a, M(K.SW_PEG/2 + CLR), False, 24)
                                         + K.section_pts(hub - a*M(4.4), a, M(K.SW_PEG/2 + CLR), False, 24)))
    collar = Manifold.hull_points(np.array(K.section_pts(hub - a*0.5, a, M(K.SW_PEG/2 + CLR + 0.8), False, 32)
                                           + K.section_pts(hub - a*M(4.8), a, M(K.SW_PEG/2 + CLR + 0.8), False, 32)))
    G["dashboard"] = K.diff(K.union([sec["dash"] ^ above, collar]), holes[6:] + [sock] + sockets["dashboard"])
    joint("peg", "dashboard_steering_console", "dashboard_steering_console", tuple(hub - a*M(2.0)), tuple(-a), K.SW_PEG, 4.0,
          back=2.0, fdepth=4.3)
    JOINTS[-1].update(male="dashboard_steering_console", male_shell="steering_wheel")
    for x, y in K.TUB_PEGS:
        joint("peg", "battery_tray_lower_housing", "interior_floor_tub", (x, y, K.Z_TUB + M(0.6)), (0, 0, 1), 2.0, 1.2,
              back=0.6, fdepth=(K.Z_FLOOR - K.Z_TUB)/SCALE - 0.05)
    return G


# ---------------------------------------------------------------------------
# 6. body: regions, features, door cards, lenses
# ---------------------------------------------------------------------------
K.Z_TAIL = 880.0                                  # liftgate bottom edge (liftgate_region)


def body_regions(z_hb):
    s = K.S_GAP
    big = lambda z0, z1: boxm((-500, -1500, z0), (6000, 1500, z1))
    fe, re = K.door_edges()
    xzp = lambda poly: K.prism(poly, "xz", -1500, 1500)
    band = big(K.Z_SKIRT + s, K.Z_CAN - s)
    loop = K.hood_loop()
    hood_out = K.prism(loop.buffer(s), "xy", z_hb, 2200)

    def lift(d):
        """Liftgate region grown by d (d < 0) or shrunk (d > 0), its bottom edge at z 880 included."""
        L = K.liftgate_region(d)
        return L ^ big(K.Z_TAIL + d, 2200) if d > 0 else K.union([L, L.translate([0, 0, d])])
    L_out = lift(-s)
    ov = 8.0                                                  # zones overlap so the shell unions into one part
    R = dict(skirt=big(-100, K.Z_SKIRT - s),
             front=(band ^ xzp(K._edge_poly(fe, True, -ov))) - hood_out,
             door=band ^ xzp(K._edge_poly(fe, False, -ov)) ^ xzp(K._edge_poly(re, True, -ov)),
             rear=(band ^ xzp(K._edge_poly(re, False, -ov))) - L_out,
             tail=lift(s),
             canopy=big(K.Z_CAN + s, 2200) - L_out,
             hood=K.prism(loop.buffer(-s), "xy", z_hb, 2200),
             fill_front=(band ^ xzp(K._edge_poly(fe, True, 6.0))) - hood_out,
             fill_rear=(band ^ xzp(K._edge_poly(re, False, 6.0))) - lift(-s - 6.0),
             fill_tail=lift(s + 6.0))
    return R, fe, re


def window_slabs(xface, ys, z0, z1, d_out, d_in, inward):
    """Union of hull slabs following a curved face: from d_out outside the skin to d_in inside."""
    out = []
    for a, b in zip(ys[:-1], ys[1:]):
        pts = []
        for y in (a, b):
            xf = xface(y)
            for z in (z0, z1):
                pts += [(xf - inward*d_out, y, z), (xf + inward*d_in, y, z)]
        out.append(Manifold.hull_points(np.array(pts)))
    return K.union(out)


CANOPY_PINS = []


def trim_path(path, d):
    """Polyline shortened by d at both ends."""
    p = [np.asarray(q, float) for q in path]
    for i, j in ((0, 1), (-1, -2)):
        u = p[j] - p[i]
        p[i] = p[i] + u/np.linalg.norm(u)*d
    return [tuple(q) for q in p]


def body_features(B, R, fe, re, z_hb, log=print):
    """Bosses (E) and cuts (C) for the main shell, tailgate, canopy and hood, plus the lenses."""
    s = K.S_GAP
    E = {k: [] for k in ("main", "tail", "pins")}
    Cc = {k: [] for k in ("main", "tail", "canopy", "hood")}
    zt, zb = K.Z_CAN - s, K.Z_SKIRT + s
    up, down = (0, 0, 1), (0, 0, -1)
    # -- body shell <-> baseplate: screws from below into pilot bosses
    for (x, y), spec, size, hb, hin in ((K.CLIP_M2[0], K.M2, "M2", 58.0, 58.0), (K.CLIP_M3[0], K.M2, "M2", 42.0, 28.0),
                                        (K.CLIP_M2[1], K.M2, "M2", 58.0, 58.0), (K.CLIP_M3[1], K.M2, "M2", 42.0, 28.0)):
        for sg in (1, -1):
            zbot = BODY_SCREW_ZBOT[(x, sg*y)]
            # boss block reaches out to the skin, so it never depends on the support fill (rocker-end bosses stop
            # 28 mm inboard of the screw, clear of the cabin-floor sill trims)
            E["main"].append(K.cbox(x - hb, x + hb, min(sg*(y - hin), sg*1200), max(sg*(y - hin), sg*1200), zb, zb + M(9)))
            Cc["main"].append(K.pilot_hole_up(x, sg*y, zb, 8.0, spec))
            head_h = spec["head_h"]
            clamp = (zb - zbot)/SCALE - head_h
            L = max(l for l in STD_M2 if l - clamp <= 7.7)           # tip stays 0.3 mm short of the 8 mm pilot
            SCREWS.append(dict(size=size, length=L, head="button head", group="main",
                               where=f"body shell to baseplate at x {x/SCALE:.1f}, y {sg*y/SCALE:+.1f} mm (from below)",
                               clamp_mm=round(clamp, 2), engagement_mm=round(L - clamp, 2)))
            joint("screw", None, "chassis_baseplate_m2_bosses", (x, sg*y, zbot + M(head_h + 2.0)), up,
                  2.0, 2.0, back=head_h + 2.0, fdepth=(K.Z_SKIRT - K.S_GAP - zbot)/SCALE - 0.1)
            joint("cbore", None, "chassis_baseplate_m2_bosses", (x, sg*y, zbot + M(0.5)), up, TOL["m2_head"] - 2*CLR, 1.0,
                  back=0.5, fdepth=head_h - 0.05)
    # -- sockets for the baseplate pins under the door sills
    for xs in K.DOOR_PINS:
        for x in xs:
            yp = float(K.skin_y(x, [K.Z_SKIRT])[0]) - M(K.D_DOOR)/2
            for sg in (1, -1):
                Cc["main"].append(socket((x, sg*yp, zb), up, TOL["pin"], 2.0))
    # -- canopy pins: the canopy overlaps the door tops on a strip only ~1.5 mm wide (shoulder tumblehome),
    #    so the pins stand on the A-pillar bases and on bosses at the C-pillar bases
    can_b = B["body"] ^ R["canopy"]
    can_sl = None
    for dz in (1.0, M(1.2), M(2.3)):                       # flat-roof sockets and pockets: 2.25 mm deep
        sl_ = K.slice_poly(can_b, K.Z_CAN + s + dz)
        can_sl = sl_ if can_sl is None else can_sl.intersection(sl_)
    for x, front in ((1590.0, True), (3700.0, False)):
        if front:                                          # pin on the door top (inner face skin - D_DOOR), socket in the canopy
            sk = float(K.skin_y(x, [K.Z_CAN - 10])[0])
            bp_ = K.best_point(can_sl, (x - 10.0, x + 10.0), (sk - M(K.D_DOOR) + 20.0, sk - 20.0), M(1.6), step=2.0)
            yp = bp_[1] if bp_ else sk - M(K.D_DOOR)/2
            x = bp_[0] if bp_ else x
            log(f"    A-pillar roof pins at x {x:.0f}, |y| {yp:.0f}" + ("" if bp_ else " (fallback)"))
        else:
            yp = float(K.skin_y(x, [K.Z_CAN + s + M(3.25) + 10])[0]) - M(TOL["pin_socket"]/2) - 40.0
        for sg in (1, -1):
            if not front:
                E["main"].append(K.cbox(x - 60, x + 60, min(sg*(yp - 60), sg*1200), max(sg*(yp - 60), sg*1200), zt - M(5), zt))
            E["pins"].append(pin_solid((x, sg*yp, zt), up, TOL["pin"], 2.0))
            Cc["canopy"].append(socket((x, sg*yp, K.Z_CAN + s), up, TOL["pin"], 2.0, flat=True))   # canopy prints on this face
            joint("pin", "main_outer_body_shell", "removable_roof_solar_canopy", (x, sg*yp, K.Z_CAN + s + M(0.75)), up, TOL["pin"], 2.0,
                  back=0.75, fdepth=2.25)
    # -- canopy magnets over the C-pillars
    # 0.6 mm wall round the pocket in the canopy; the body-shell boss starts 0.5 mm inboard of the pocket,
    # outboard of the rear wheel-house wall (the support fill cannot reach under it there)
    bp = K.best_point(can_sl, (3800.0, 3960.0), (735.0, 880.0), M(2.35), step=5.0)
    if bp is None:
        raise RuntimeError("no canopy magnet site")
    for sg in (1, -1):
        x, y = bp[0], sg*bp[1]
        E["main"].append(K.cbox(x - 60, x + 60, min(y - sg*45, sg*1200), max(y - sg*45, sg*1200), zt - M(5), zt))
        Cc["main"].append(K.teardrop_hole((x, y, zt + 1), down, M(K.MAG_D), M(K.MAG_H) + 1))
        Cc["canopy"].append(K.cyl((x, y, K.Z_CAN + s - 1), (x, y, K.Z_CAN + s + M(K.MAG_H)), M(K.MAG_D)/2, seg=32))
        joint("magnet", None, "main_outer_body_shell", (x, y, zt - M(1.0)), down, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
        joint("magnet", None, "removable_roof_solar_canopy", (x, y, K.Z_CAN + s + M(1.0)), up, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    MAGNETS.append(dict(where="roof canopy to body shell, over the C-pillars", count=4))
    CANOPY_PINS[:] = [1590.0, 3700.0]
    log(f"    canopy magnets at x {bp[0]:.0f}, |y| {bp[1]:.0f}")
    # -- hood ledge, magnets, PV pigtail
    loop = K.hood_loop()
    E["main"].append(K.prism(loop.buffer(-s).difference(loop.buffer(-s - M(2.2))), "xy", z_hb - M(1.6), z_hb - s) - B["wells"])
    for x, y in K.HOOD_MAG:
        for sg in (1, -1):
            ye = float(v4.hood_halfwidth(x))
            E["main"].append(K.cbox(x - 60, x + 60, min(sg*(y - 60), sg*ye), max(sg*(y - 60), sg*ye), z_hb - M(4.5), z_hb - s))
            Cc["main"].append(K.teardrop_hole((x, sg*y, z_hb - s + 1), down, M(K.MAG_D), M(K.MAG_H) + 1))
            Cc["hood"].append(K.cyl((x, sg*y, z_hb - 1), (x, sg*y, z_hb + M(K.MAG_H)), M(K.MAG_D)/2, seg=32))
            joint("magnet", None, "main_outer_body_shell", (x, sg*y, z_hb - s - M(1.0)), down, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
            joint("magnet", None, "hood_solar_bonnet", (x, sg*y, z_hb + M(1.0)), up, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    MAGNETS.append(dict(where="hood to body shell ledge", count=4))
    for sg in (1, -1):                       # front pair: the hood is only ~2.8 mm thick here, so flat-roof pockets
        x, y = HOOD_POST
        Cc["hood"].append(K.cyl((x, sg*y, z_hb - 1), (x, sg*y, z_hb + M(K.MAG_H)), M(K.MAG_D)/2, seg=32))
        joint("magnet", None, "hood_solar_bonnet", (x, sg*y, z_hb + M(1.0)), up, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
        joint("magnet", None, "front_e_axle_and_inverter", (x, sg*y, z_hb - s - M(1.0)), down, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    MAGNETS.append(dict(where="hood to the radiator-support posts", count=4))
    Cc["hood"].append(K.teardrop_hole((725.0, 230.0, z_hb - 1), up, M(K.WIRE_D), M(3.0)))
    rc = M(K.LED_W)/2
    lenses = []
    # -- headlights: LED channel behind the lamp band, wire exit, blade slits, clear-lens window
    zc = 905.0
    ys = np.linspace(-840, 840, 43)
    xf = np.array([K.face_x(y, zc) for y in ys])
    xfi = lambda y: float(np.interp(y, ys, xf))
    path = [(x + M(K.T_SHELL) + rc, y, zc) for x, y in zip(xf, ys)]
    E["main"].append(K.tube(path, rc + M(K.T_SHELL), tear=True))
    Cc["main"].append(K.tunnel(trim_path(path, M(K.T_SHELL)), rc))      # closed ends: no coplanar end caps
    Cc["main"].append(K.tunnel([(xf[21] + M(K.T_SHELL) + rc, 0.0, zc), (xf[21] + 300, 0.0, zc)], M(1.5)))
    for a, b in zip(ys[1:-2], ys[2:-1]):
        if abs(a) <= 800 and abs(b) <= 800:
            Cc["main"].append(K.gabled_slit(xfi, a, b, 941.0, 947.0, +1))
    zw = (880.0, 932.0)
    yl = np.linspace(-780, 780, 40)
    lens_h = B["body"] ^ window_slabs(xfi, yl, zw[0], zw[1], 20.0, M(K.T_SHELL), +1)
    groove = window_slabs(xfi, yl, zc - M(0.4), zc + M(0.4), -(M(K.T_SHELL) - M(0.5)), M(K.T_SHELL) + 20.0, +1)
    lenses.append(lens_h - groove)
    yo = np.r_[-780 - C, yl[1:-1], 780 + C]
    Cc["main"].append(window_slabs(xfi, yo, zw[0] - C, zw[1] + C, 20.0, M(K.T_SHELL) + rc, +1))
    joint("lens", "clear_lens_headlights_taillights", "main_outer_body_shell", (xfi(0.0) + M(K.T_SHELL)/2, 0.0, sum(zw)/2), up, None, None)
    joint("lens", "clear_lens_headlights_taillights", "main_outer_body_shell", (xfi(0.0) + M(K.T_SHELL)/2, 0.0, sum(zw)/2), (1, 0, 0), None, None)
    # -- tail bar (tailgate): channel, wire exit, clear-lens window
    zc = 1192.0
    ys = np.linspace(-640, 640, 33)
    xr = np.array([K.face_x(y, zc, front=False) for y in ys])
    xri = lambda y: float(np.interp(y, ys, xr))
    path = [(x - M(K.T_SHELL) - rc, y, zc) for x, y in zip(xr, ys)]
    E["tail"].append(K.tube(path, rc + M(K.T_SHELL), tear=True))
    Cc["tail"].append(K.tunnel(trim_path(path, M(K.T_SHELL)), rc))
    Cc["tail"].append(K.tunnel([(xr[16] - M(K.T_SHELL) - rc, 0.0, zc), (xr[16] - 300, 0.0, zc)], M(1.5)))
    zw = (1170.0, 1214.0)
    yl = np.linspace(-600, 600, 31)
    # the tail bar sits on a sloping face: the lens is the skin layer itself (constant 1.6 mm, full height)
    lens_t = (B["body"] ^ window_slabs(xri, yl, zw[0], zw[1], 20.0, M(K.T_SHELL) + 60.0, -1)) - B["inset_shell"]
    inner = B["inset_shell"] ^ window_slabs(xri, yl, zc - M(0.4), zc + M(0.4), -(M(K.T_SHELL) - 30.0), M(K.T_SHELL) + 60.0, -1)
    groove = inner.translate([M(0.5), 0, 0]) - B["inset_shell"]    # 0.5 mm off the inner skin, outward (+x at the tail)
    lenses.append(largest(lens_t - groove)[0])
    yo = np.r_[-600 - C, yl[1:-1], 600 + C]
    Cc["tail"].append(window_slabs(xri, yo, zw[0] - C, zw[1] + C, 20.0, M(K.T_SHELL) + rc + 20.0, -1))
    joint("lens", "clear_lens_headlights_taillights", "tailgate_rear_hatch", (xri(0.0) - M(K.T_SHELL)/2, 0.0, sum(zw)/2), up, None, None)
    joint("lens", "clear_lens_headlights_taillights", "tailgate_rear_hatch", (xri(0.0) - M(K.T_SHELL)/2, 0.0, sum(zw)/2), (1, 0, 0), None, None)
    # -- solar strings: roof -> A-pillars (canopy) -> body shell -> rocker connectors; tailgate string
    rw = M(K.WIRE_D)/2
    for sg in (1, -1):
        pts = []
        for z in np.linspace(1640, K.Z_CAN + 40, 12):
            xa = v4.x_apillar(z); ya = float(v4.y_glass(xa, z))
            pts.append((xa + 35, sg*(ya - 60), z - 40))
        xe, ye = 1398.0, 745.0
        roof = [(2400.0, 0.0, 1655.0), (2350.0, sg*420.0, 1640.0), (2330.0, sg*620.0, 1615.0)]
        Cc["canopy"].append(K.tunnel(roof + pts + [(xe, sg*ye, K.Z_CAN + 30), (xe, sg*ye, K.Z_CAN - 5)], rw))
        yk = K.INLET_Y if sg < 0 else K.INLET_Y - 70
        Cc["main"].append(K.tunnel([(xe, sg*ye, K.Z_CAN + 5), (xe, sg*ye, 1000.0), (K.PACK_X0 - 56, sg*yk, 600.0),
                                    (K.PACK_X0 - 56, sg*yk, K.Z_SKIRT - 5)], rw))
    xh0 = 4236.0 - 2*s
    cen = [(x, 0.0, float(v4.ROOF(x)) - 70) for x in np.linspace(xh0 + 10, 2400, 14)]
    Cc["canopy"].append(K.tunnel(cen, rw))
    Cc["tail"].append(K.tunnel([(x, 0.0, float(v4.ROOF(x)) - 70) for x in np.linspace(xh0 - 10, xh0 + 220, 4)], rw))
    # -- charge inlet housing and harness tunnel to the rocker connector
    yi = float(K.skin_y(1258.0, [958.0])[0])
    E["main"].append(K.cbox(1205, 1312, yi - M(K.T_SHELL) - 70, yi - 2, 915, 1000))
    Cc["main"].append(K.tunnel([(1258.0, yi - M(K.T_SHELL) - 40, 930.0), (1300.0, yi - 70, 700.0), (K.PACK_X0 - 56, K.INLET_Y, 400.0),
                                (K.PACK_X0 - 56, K.INLET_Y, K.Z_SKIRT - 5)], rw))
    # -- tailgate magnets: bottom edge onto the bumper ledge, top edge against the canopy at the hinge
    zl = K.Z_TAIL
    t_in = M(K.T_SHELL)
    for y in (420.0, -420.0):
        fz = [K.face_x(y, z, front=False) for z in np.linspace(zl - M(5) - 10, zl + M(5) + 10, 9)]
        x = min(fz) - t_in - 45.0                      # pocket 10 mm inside the inner skin face
        xo = max(fz) + 10.0                            # boss reaches through the skin (clipped to the body)
        E["tail"].append(K.box(x - 60, xo, y - 60, y + 60, zl + s, zl + s + M(5)))
        Cc["tail"].append(K.cyl((x, y, zl + s - 1), (x, y, zl + s + M(K.MAG_H)), M(K.MAG_D)/2, seg=32))
        E["main"].append(K.box(x - 60, xo, y - 60, y + 60, zl - s - M(5), zl - s))
        Cc["main"].append(K.teardrop_hole((x, y, zl - s + 1), down, M(K.MAG_D), M(K.MAG_H) + 1))
        joint("magnet", None, "tailgate_rear_hatch", (x, y, zl + s + M(1.0)), up, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
        joint("magnet", None, "main_outer_body_shell", (x, y, zl - s - M(1.0)), down, 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    MAGNETS.append(dict(where="tailgate bottom edge to body shell", count=4))
    for yy in (-300.0, 300.0):
        xh = 4236.0 - 30*(yy/600)**2
        zr = float(v4.ROOF(xh)) - 60
        zm = zr - 35
        E["tail"].append(K.box(xh + s, xh + s + M(4), yy - 60, yy + 60, zm - 70, zr + 20) ^ B["base"])
        Cc["tail"].append(K.teardrop_hole((xh + s - 1, yy, zm), (1, 0, 0), M(K.MAG_D), M(K.MAG_H) + 1))
        Cc["canopy"].append(K.teardrop_hole((xh - s + 1, yy, zm), (-1, 0, 0), M(K.MAG_D), M(K.MAG_H) + 1))
        joint("magnet", None, "tailgate_rear_hatch", (xh + s + M(1.0), yy, zm), (1, 0, 0), 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
        joint("magnet", None, "removable_roof_solar_canopy", (xh - s - M(1.0), yy, zm), (-1, 0, 0), 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    MAGNETS.append(dict(where="tailgate hinge edge to roof canopy", count=4))
    return E, Cc, lenses


DOOR_GRID = (1400.0, 3550.0, K.Z_SKIRT - 20, K.Z_CAN + 20)    # door panels and cards share this sampling


def door_slope_cos(x_rng, z0, z1):
    """cos of the steepest angle between the door skin normal and the y axis over the card areas."""
    worst = 1.0
    for x0, x1 in x_rng.values():
        xs = np.linspace(x0, x1, 25); zs = np.linspace(z0, z1, 25)
        Y = np.array([K.skin_y(x, zs) for x in xs])
        gx = np.gradient(Y, xs, axis=0); gz = np.gradient(Y, zs, axis=1)
        c = 1/np.sqrt(1 + gx**2 + gz**2)
        worst = min(worst, float(np.nanmin(c)))
    return max(worst, 0.7)


def door_cards():
    """Four trim cards that clip onto the inner face of the body shell's door panels with two
    pegs each; armrest, map pocket, speaker grille, pull handle recess and trim inlay."""
    cards, slab_holes = [], []
    x_rng = {True: (1500.0, 2460.0), False: (2585.0, 3420.0)}     # front card starts behind the rocker-screw boss
    z0, z1 = 480.0, K.Z_CAN - K.S_GAP - 20
    fe, re = K.door_edges()
    # sampled exactly like the door-panel inner face (DOOR_GRID), offset along y by the clearance
    # divided by the cosine of the steepest skin slope, so the normal gap is >= 0.25 mm everywhere
    gap = C/door_slope_cos(x_rng, z0, z1)
    outer = K.inner_y_solid(*DOOR_GRID, M(K.D_DOOR) + gap)
    inner = K.inner_y_solid(*DOOR_GRID, M(K.D_DOOR) + gap + 20)
    for front in (True, False):
        x0, x1 = x_rng[front]
        for s in (1, -1):
            side = boxm((x0, 0 if s > 0 else -1500, z0), (x1, 1500 if s > 0 else 0, z1))
            reg = side if front else side ^ K.prism(K._edge_poly(re, True, 25.0), "xz", -1500, 1500)
            panel = (outer - inner) ^ reg
            ext, cut = K.door_card(s, front, s)
            ext = K.union(ext) ^ outer ^ boxm((x0, -1500, z0), (x1, 1500, z1))
            shift = [cut[0]] + [c.translate([0, -s*(gap + 20), 0]) for c in cut[1:]]   # face recesses move to the card face
            card = K.diff(K.union([panel, ext]), shift)
            for xp in (x0 + 0.25*(x1 - x0), x0 + 0.75*(x1 - x0)):
                zp = 700.0
                yin = float(K.skin_y(xp, [zp])[0]) - M(K.D_DOOR)
                card = K.union([card, pin_solid((xp, s*(yin - gap), zp), (0, s, 0), TOL["pin"], 1.25, tear=False)])
                slab_holes.append(K.teardrop_hole((xp, s*(yin - 10), zp), (0, s, 0), M(TOL["pin_socket"]), 10 + M(1.0 + CLR) + 4))
                joint("pin", "door_cards_set", "main_outer_body_shell", (xp, s*(yin + M(0.5)), zp), (0, s, 0), TOL["pin"], 1.0,
                      back=0.5, fdepth=1.25)
            cards.append(largest(card)[0])
    return K.union(cards), slab_holes


# ---------------------------------------------------------------------------
# 7. part table
# ---------------------------------------------------------------------------
PARTS = [  # (folder, file stem, group key, expected shells, print orientation note)
    ("01_chassis", "chassis_baseplate_m2_bosses", "baseplate", 1, "flat underside on the bed"),
    ("01_chassis", "front_subframe_suspension", "front_sub", 1, "subframe bars on the bed"),
    ("01_chassis", "rear_subframe_suspension", "rear_sub", 1, "subframe bars on the bed"),
    ("02_powertrain", "battery_tray_lower_housing", "tray", 1, "tray floor on the bed"),
    ("02_powertrain", "battery_module_cells_combined", "modules", 8, "8 module pairs, cell bases on the bed"),
    ("02_powertrain", "front_e_axle_and_inverter", "eaxle_front", 1, "standing on its feet as installed"),
    ("02_powertrain", "rear_e_axle_and_inverter", "eaxle_rear", 1, "standing on its feet as installed"),
    ("02_powertrain", "hv_wiring_harness_conduits", "harness", 2, "front and rear harness, carrier straps on the bed"),
    ("03_interior", "interior_floor_tub", "floor_tub", 2, "cabin floor flat on the bed; cargo deck upside down beside it"),
    ("03_interior", "front_bucket_seats_pair", "seats", 2, "seat bases on the bed"),
    ("03_interior", "rear_bench_seat", "bench", 1, "seat base on the bed"),
    ("03_interior", "dashboard_steering_console", "dashboard", 2, "dashboard upright; steering wheel face down beside it"),
    ("03_interior", "door_cards_set", "door_cards", 4, "upright on their bottom edges"),
    ("04_body_shell", "removable_roof_solar_canopy", "canopy", 1, "flat belt-line underside on the bed"),
    ("04_body_shell", "hood_solar_bonnet", "hood", 1, "flat underside on the bed"),
    ("04_body_shell", "main_outer_body_shell", "main", 1, "upright on its sill line"),
    ("04_body_shell", "tailgate_rear_hatch", "tail", 1, "upright on its bottom edge"),
    ("05_wheels_and_hardware", "wheels_rims_x4", "rims", 4, "inner face down"),
    ("05_wheels_and_hardware", "tires_treaded_x4", "tyres", 4, "inner sidewall down"),
    ("05_wheels_and_hardware", "clear_lens_headlights_taillights", "lenses", 2, "upright on their bottom edges"),
]
for _f, stem, key, _n, _o in PARTS:
    FILE_OF[key] = stem
STEM_OF = {stem: key for _f, stem, key, _n, _o in PARTS}


def fix_joint_names():
    for j in JOINTS:
        for k in ("male", "female"):
            if j.get(k) in FILE_OF:
                j[k] = FILE_OF[j[k]]


# ---------------------------------------------------------------------------
# 8. build everything (installed positions, full size)
# ---------------------------------------------------------------------------
def wheel_placements():
    """Transforms (3x4, model mm -> full size) placing a local wheel (axis z, inner face z = 0)
    on each hub, as in the assembled car."""
    out = []
    for xa in (K.X_FA, K.X_RA):
        for sg in (1, -1):
            y0 = sg*(v4.TYRE_OUT - v4.TYRE_W)
            if sg > 0:
                out.append(np.array([[20, 0, 0, xa], [0, 0, 20, y0], [0, -20, 0, K.HUB_Z]], float))
            else:
                out.append(np.array([[20, 0, 0, xa], [0, 0, -20, y0], [0, 20, 0, K.HUB_Z]], float))
    return out


def build(res=6.0, mirrors="mirror", log=print):
    t0 = time.time()
    for reg in (JOINTS, SCREWS, MAGNETS):
        reg.clear()
    DROPPED.clear()
    BODY_SCREW_ZBOT.clear()
    apply_tolerances()
    B = K.build_body_kit(res, mirrors, log=log)
    z_hb = K.hood_bottom()
    R, fe, re = body_regions(z_hb)
    log(f"  regions ({time.time() - t0:.0f} s)")
    G = chassis_groups(B, R)
    pads, pilots, mcuts = mount_geometry()
    log(f"  chassis groups ({time.time() - t0:.0f} s)")
    env = K.inner_y_solid(1250, 3700, 370, 1130, M(K.D_DOOR) + 30)
    G.update(interior_groups(env))
    wl = K.steering_wheel()
    G["steering_wheel"] = K.place_steering_wheel(wl)
    G["cargo_deck"] = cargo_deck_kit()
    cards, slab_holes = door_cards()
    G["door_cards"] = cards
    tray_pegs = [pin_solid((x, y, K.Z_PACK_TOP), (0, 0, 1), 2.0, 2.0) for x, y in K.TUB_PEGS]
    G["tray"] = K.union([G["tray"]] + tray_pegs)
    log(f"  interior ({time.time() - t0:.0f} s)")
    # ---- clearance pass: the part installed first wins, the later one is offset by 0.25 mm
    rules = [  # (winner, loser, mode): 'slot' when the loser is installed first, 'drape' when it goes on later
        ("eaxle_front", "harness", "slot"), ("eaxle_rear", "harness", "slot"),
        ("harness", "baseplate", "slot"), ("harness", "front_sub", "drape_c"), ("harness", "rear_sub", "drape_c"),
        ("harness", "tray", "drape_c"), ("harness", "dashboard", "drape_c"),
        ("eaxle_front", "front_sub", "slot"), ("eaxle_rear", "rear_sub", "slot"),
        ("eaxle_front", "baseplate", "slot"), ("eaxle_rear", "baseplate", "slot"),
        ("front_sub", "baseplate", "slot"), ("rear_sub", "baseplate", "slot"),
        ("tray", "baseplate", "slot"), ("tray", "modules", "drape_c"),     # modules drop 0.25 mm onto the cold plate
        ("tray", "dashboard", "drape_c"), ("tray", "floor_tub", "drape"),   # the cabin floor rests on the battery frame
        ("dashboard", "floor_tub", "slot"), ("dashboard", "seats", "drape_c"),
    ]
    carved = {}
    for win, lose, mode in rules:
        G[lose], v = carve(G[lose], G[win], mode)
        if v > 0:
            carved[f"{win} > {lose}"] = round(v/SCALE**3, 3)
    log(f"  clearance pass: {len(carved)} contacts offset ({time.time() - t0:.0f} s)")
    G["baseplate"] = K.union([G["baseplate"], K.union(pads)])          # boss pads after the pass
    # ---- holes that pass through several chassis parts
    ab = axle_bores()
    base_cuts = [ab] + pilots
    for x, y in list(K.CLIP_M2) + list(K.CLIP_M3):                   # body screws: counterbores from the local underside
        for s_ in (1, -1):
            BODY_SCREW_ZBOT[(x, s_*y)] = underside_under(B, x, s_*y, M(K.M2["head"]/2) + 16)
    for x, y in list(K.CLIP_M2) + list(K.CLIP_M3):
        base_cuts += [screw_hole_from_below(x, s*y, K.Z_SKIRT, K.M2, BODY_SCREW_ZBOT[(x, s*y)]) for s in (1, -1)]
    post_cuts = {"baseplate": [], "rear_sub": []}
    for x, y in K.CARGO_POSTS:
        g = "rear_sub" if x > 4300 else "baseplate"
        post_cuts[g] += [K.teardrop_hole((x, s*y, POST_TOP + 1), (0, 0, -1), M(K.MAG_D), M(K.MAG_H) + 1) for s in (1, -1)]
        for s in (1, -1):
            joint("magnet", None, FILE_OF[g], (x, s*y, POST_TOP - M(1.0)), (0, 0, -1), 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
            joint("magnet", None, "cargo_deck", (x, s*y, POST_TOP + M(1.0)), (0, 0, 1), 3.0, 2.0, back=1.0, fdepth=2.25, blind=True)
    base_cuts += post_cuts["baseplate"]
    G["rear_sub"] = K.diff(G["rear_sub"], post_cuts["rear_sub"])
    MAGNETS.append(dict(where="cargo deck on its four posts", count=8))
    for x, y in K.LED_PASS:
        base_cuts.append(K.cyl((x, y, K.GC - 1), (x, y, 300), M(2.0), seg=24))
    G["baseplate"] = K.diff(G["baseplate"], base_cuts)
    for g in ("front_sub", "rear_sub", "eaxle_front", "eaxle_rear"):
        G[g] = K.diff(G[g], [ab])
    for g, cs in mcuts.items():
        G[g] = K.diff(G[g], cs)
    # cabin floor screws: clearance + head pocket in the floor (interior_parts), pilots in the battery side frames
    tub_pilots = []
    zb = K.Z_FLOOR + M(2.0)
    for x, y in K.TUB_SCREWS:
        for s in (1, -1):
            tub_pilots.append(K.teardrop_hole((x, s*y, K.Z_PACK_TOP + 1), (0, 0, -1), M(TOL["m2_pilot"]), M(7.0)))
            joint("screw", None, "interior_floor_tub", (x, s*y, K.Z_TUB + M(0.5)), (0, 0, 1), 2.0, 1.0,
                  back=0.5, fdepth=(zb - K.Z_TUB)/SCALE - 0.1)
            clamp = (zb - M(K.M2["head_h"]) - C - K.Z_TUB)/SCALE
            SCREWS.append(dict(size="M2", length=8, head="button head (ISO 7380), self-tapping into the printed pilot",
                               group="floor_tub", where=f"cabin floor to battery side frame at x {x/SCALE:.1f}, y {s*y/SCALE:+.1f} mm (from above)",
                               clamp_mm=round(clamp, 2), engagement_mm=round(8 - clamp, 2)))
    G["tray"] = K.diff(G["tray"], tub_pilots)
    for xa in (K.X_FA, K.X_RA):
        for yy, part in ((660.0, "front_sub" if xa == K.X_FA else "rear_sub"), (0.0, "eaxle_front" if xa == K.X_FA else "eaxle_rear")):
            joint("axle", None, FILE_OF[part], (xa, yy, K.HUB_Z), (0, 1, 0), TOL["axle"], 2.0, back=0.5, fdepth=1.0)
    log(f"  chassis holes ({time.time() - t0:.0f} s)")
    # ---- body: features, main shell (front zone + door panels + rear zone), tailgate, canopy, hood
    E, Cc, lenses = body_features(B, R, fe, re, z_hb, log)
    keep = K.union([G[k] for k in ("baseplate", "front_sub", "rear_sub", "tray", "modules", "eaxle_front", "eaxle_rear",
                                   "harness", "floor_tub", "seats", "bench", "dashboard", "steering_wheel", "cargo_deck",
                                   "door_cards")])
    keep_out = K.union([B["wells"], keep])
    def filled(reg, fill_reg, extra, name):
        part = K.union([K.shell_part(B, reg)] + [m ^ B["base"] ^ reg for m in extra])
        bb = bbox(reg); bbb = bbox(B["base"])
        lo = np.maximum(bb[:3], bbb[:3] - 10); hi = np.minimum(bb[3:], bbb[3:] + 10); lo[2] = bb[2]
        out, _ = K.support_filled(part, (lo, hi), B["inset_fill"], fill_reg, keep_out, log, name, shadow=True)
        return out
    front = filled(R["front"], R["fill_front"], E["main"], "front zone")
    rear = filled(R["rear"], R["fill_rear"], E["main"] + [G.pop("_rear_bumper")], "rear zone")
    door_in = K.inner_y_solid(*DOOR_GRID, M(K.D_DOOR))
    doors = (B["body"] ^ R["door"]) - door_in
    doors = K.union([doors] + [m ^ B["base"] ^ R["door"] for m in E["main"]])
    main = K.diff(K.union([front, doors, rear] + E["pins"]), Cc["main"] + slab_holes)
    G["main"], dr = largest(main)
    DROPPED["main_outer_body_shell"] = round(dr/SCALE**3, 3)
    G["main"], v = carve(G["main"], G["front_sub"], "drape_c")         # exact 0.25 mm over the subframe tops
    if v > 0:
        carved["front_sub > main"] = round(v/SCALE**3, 3)
    for g in ("floor_tub", "dashboard", "seats", "bench"):            # e.g. the wheel-house liners vs the floor corners
        G[g], v = carve(G[g], G["main"], "slot")
        if v > 0:
            carved[f"main > {g}"] = round(v/SCALE**3, 3)
    tail = filled(R["tail"], R["fill_tail"], E["tail"], "tailgate")
    G["tail"], dr = largest(K.diff(tail, Cc["tail"]))
    DROPPED["tailgate_rear_hatch"] = round(dr/SCALE**3, 3)
    R_can = dict(canopy=R["canopy"])
    can, _cav = K.canopy_kit(B, R_can, K.union([G["dashboard"], G["seats"], G["bench"], G["steering_wheel"], G["floor_tub"]]), log,
                             shadow=True)
    G["canopy"], dr = largest(bed_trim(K.diff(can, Cc["canopy"]), K.Z_CAN + K.S_GAP))
    DROPPED["removable_roof_solar_canopy"] = round(dr/SCALE**3, 3)
    G["hood"], dr = largest(K.diff(B["body"] ^ R["hood"], Cc["hood"]))
    DROPPED["hood_solar_bonnet"] = round(dr/SCALE**3, 3)
    G["lenses"] = K.union(lenses)
    log(f"  body ({time.time() - t0:.0f} s)")
    # ---- wheels: rim (rotor, caliper, nuts) and tyre as separate parts
    (rim, tyre), _ = v4.build_wheel(split_clearance=CLR)
    W = dict(rim=rim, tyre=tyre)
    G["rims"] = K.union([rim.transform(T) for T in wheel_placements()])
    G["tyres"] = K.union([tyre.transform(T) for T in wheel_placements()])
    dropped = {}
    for g in list(G):
        G[g], crumbs = keep_big(G[g])
        if crumbs > 0:
            dropped[g] = round(crumbs/SCALE**3, 3)
    log(f"  boolean crumbs dropped (model mm3): {dropped or 'none'}; detached pieces dropped: {DROPPED}")
    fins = {}
    for g in list(G):                          # every piece on its own print bed: the lowest plane as installed,
        if g in ("rims", "tyres", "steering_wheel"):   # the top face for the cargo deck (printed upside down)
            continue
        flip = (lambda m: m.mirror([0, 0, 1])) if g == "cargo_deck" else (lambda m: m)
        comps = G[g].decompose()
        out = [flip(strip_bed_fins(flip(c), flip(c).bounding_box()[2])) for c in comps]
        v = sum(c.volume() for c in comps) - sum(c.volume() for c in out)
        if v > 0:
            G[g] = K.union(out)
            fins[FILE_OF.get(g, g)] = round(v/SCALE**3, 4)
    log(f"  bed fins thinner than 0.2 mm removed (model mm3): {fins or 'none'}")
    dropped.update({k: v for k, v in DROPPED.items() if v > 0})
    fix_joint_names()
    log(f"kit built in {time.time() - t0:.0f} s")
    return G, W, wl, dict(z_hood=z_hb, carved=carved, dropped=dropped, fins=fins, time=time.time() - t0, res=res, mirrors=mirrors,
                          clr=CLR, script=script_hash())


def save_cache(path, G, W, wl, info):
    import pickle
    pack = lambda m: mesh_arrays(m)
    data = dict(G={k: pack(m) for k, m in G.items()}, W={k: pack(m) for k, m in W.items()}, wl=pack(wl), info=info,
                JOINTS=JOINTS, SCREWS=SCREWS, MAGNETS=MAGNETS)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_cache(path):
    import pickle
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    un = lambda vf: Manifold(Mesh(vert_properties=np.asarray(vf[0], np.float32), tri_verts=np.asarray(vf[1], np.uint32)))
    JOINTS[:] = d["JOINTS"]; SCREWS[:] = d["SCREWS"]; MAGNETS[:] = d["MAGNETS"]
    return {k: un(v) for k, v in d["G"].items()}, {k: un(v) for k, v in d["W"].items()}, un(d["wl"]), d["info"]


# ---------------------------------------------------------------------------
# 9. print layouts (model mm, bed at z = 0)
# ---------------------------------------------------------------------------
def to_model(m):
    return m.scale([1/SCALE]*3)


def drop_shells(m):
    """Every separate piece of a set drops onto the bed on its own."""
    comps = m.decompose()
    return K.union([c.translate([0, 0, -c.bounding_box()[2]]) for c in comps])


DECK_SHIFT = 6.0          # model mm: the cargo deck sits 0.5 mm behind the cabin floor when installed


def print_layout(key, G, W, wl):
    if key in ("rims", "tyres"):
        part = W["rim" if key == "rims" else "tyre"]
        d = 2*v4.WR + 4.0
        return K.union([part.translate([i*d, j*d, 0]) for i in (0, 1) for j in (0, 1)])
    m = to_model(G[key])
    if key == "dashboard":                                  # steering wheel face down in front of the dashboard
        bb = bbox(m)
        w = to_model(wl)
        wb = bbox(w)
        w = w.translate([bb[0] - 4.0 - wb[3], -0.5*(wb[1] + wb[4]), -wb[2]])
        m = K.union([m.translate([0, 0, -bb[2]]), w])
    if key == "floor_tub":                                  # cargo deck behind the floor, 6 mm clear of it on the bed
        deck = to_model(G["cargo_deck"])
        zc = 0.5*(bbox(deck)[2] + bbox(deck)[5])
        deck = deck.translate([0, 0, -zc]).rotate([180, 0, 0]).translate([DECK_SHIFT, 0, zc])   # top face down
        m = K.union([m, deck])
    m = drop_shells(m)
    return m.translate([0, 0, -m.bounding_box()[2]])


# ---------------------------------------------------------------------------
# 10. export: STL (binary, model mm) and 3MF projects
# ---------------------------------------------------------------------------
def mesh_arrays(m):
    mm = m.to_mesh()
    return np.asarray(mm.vert_properties)[:, :3].astype(np.float64), np.asarray(mm.tri_verts).astype(np.int64)


def clean_for_stl(m, expected):
    """Unique vertex positions (STL readers merge by position), crumbs removed, repairs that keep
    every intended shell.  Returns (verts, faces, report)."""
    import trimesh
    v, f = mesh_arrays(m)
    v, nd = K.separate_pinches(v, f)
    f, dropped = K.drop_slivers(v, f)
    tm = trimesh.Trimesh(np.asarray(v, np.float32).astype(np.float64), f, process=True)
    rep = dict(pinches_separated=int(nd), slivers_dropped=len(dropped), repaired_faces=0)
    if not tm.is_watertight:
        n0 = len(tm.faces)
        F = tm.faces
        tm.update_faces((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]))
        tm.update_faces(tm.unique_faces()); tm.remove_unreferenced_vertices()
        if not tm.is_watertight:
            tm, _ = K.split_t_junctions(tm)
        if not tm.is_watertight:
            trimesh.repair.fill_holes(tm)
        rep["repaired_faces"] = n0 - len(tm.faces)
    comps = tm.split(only_watertight=False)
    rep["dropped_pieces"], rep["dropped_mm3"] = 0, 0.0
    if len(comps) > expected:
        comps = sorted(comps, key=lambda c: -abs(c.volume))
        rep["dropped_pieces"] = len(comps) - expected
        rep["dropped_mm3"] = round(float(sum(abs(c.volume) for c in comps[expected:])), 3)
        tm = trimesh.util.concatenate(comps[:expected])
    v = np.asarray(tm.vertices).copy()
    v[v[:, 2] < 2e-4, 2] = 0.0                     # the bed plane exactly at z = 0 (pinch nudges leave ~1e-4)
    return v, np.asarray(tm.faces, np.int64), rep


def write_stl(v, f, path, name="solar_suv_1to20_model_kit"):
    tri = v[f].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def plate_offsets(objects, gap=5.0, width=250.0):
    """Build-item translations: the parts of one folder shelf-packed into rows up to 250 mm wide (x),
    rows stacked along y, every part on z = 0."""
    out, x, y, row = [], 0.0, 0.0, 0.0
    for _name, v, _f in objects:
        lo, hi = v.min(0), v.max(0)
        w, d = hi[0] - lo[0], hi[1] - lo[1]
        if x > 0 and x + w > width:
            x, y, row = 0.0, y + row + gap, 0.0
        out.append((x - lo[0], y - lo[1], -lo[2]))
        x += w + gap
        row = max(row, d)
    return out


def write_3mf(path, objects):
    """Minimal 3MF (core spec): one object per part, millimetre units, laid side by side on one plate."""
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            '</Relationships>')
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>']
    for i, (name, v, f) in enumerate(objects, start=1):
        parts.append(f'<object id="{i}" name="{name}" type="model"><mesh><vertices>')
        v32 = np.asarray(v, np.float32)                     # the STL precision; 9 significant digits round-trip float32 exactly
        parts.append("".join(f'<vertex x="{a:.9g}" y="{b:.9g}" z="{c:.9g}"/>' for a, b, c in v32.tolist()))
        parts.append("</vertices><triangles>")
        parts.append("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f))
        parts.append("</triangles></mesh></object>")
    parts.append("</resources><build>")
    parts.append("".join(f'<item objectid="{i}" transform="1 0 0 0 1 0 0 0 1 {dx:.4f} {dy:.4f} {dz:.4f}"/>'
                         for i, (dx, dy, dz) in enumerate(plate_offsets(objects), start=1)))
    parts.append("</build></model>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "".join(parts))


# ---------------------------------------------------------------------------
# 11. validation
# ---------------------------------------------------------------------------
def check_3mf(path, expected):
    """Reload a 3MF: object count, millimetre unit, every object watertight."""
    import trimesh
    with zipfile.ZipFile(path) as z:
        xml = z.read("3D/3dmodel.model").decode()
    sc = trimesh.load(path, force="scene")
    objs = list(sc.geometry.values())
    return dict(objects=len(objs), expected=expected, unit_mm='unit="millimeter"' in xml,
                all_watertight=all(g.is_watertight for g in objs))


def mesh_report(path, expected_shells):
    import trimesh
    from collections import Counter
    tm = trimesh.load(path)
    edges = Counter(len(g) for g in trimesh.grouping.group_rows(tm.edges_sorted))
    n, a, c = tm.face_normals, tm.area_faces, tm.triangles_center
    zmin = tm.bounds[0, 2]
    bed = (c[:, 2] < zmin + 1e-3) & (n[:, 2] < -0.99)
    film = (c[:, 2] < zmin + 1e-3) & (n[:, 2] > 0.99)                 # up-facing faces on the bed plane: zero-thickness film
    try:                                                                # fins thinner than 0.2 mm standing on the bed
        man = Manifold(Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
        sec = K.slice_poly(man, zmin + 0.03)
        thin = sec.difference(sec.buffer(-0.1).buffer(0.1))
        fins = sorted((g.area for g in getattr(thin, "geoms", [thin]) if g.area > 0.02), reverse=True)
    except Exception:                                                   # noqa: BLE001 - not a manifold: reported above
        fins = []
    shells = tm.split(only_watertight=False)
    shell_bed = [round(float(s.area_faces[(s.triangles_center[:, 2] < zmin + 1e-3) & (s.face_normals[:, 2] < -0.99)].sum()), 1)
                 for s in shells]
    return dict(triangles=len(tm.faces), watertight=bool(tm.is_watertight), edges_2_manifold=set(edges) == {2},
                winding_consistent=bool(tm.is_winding_consistent), shells=len(shells), shells_expected=expected_shells,
                volume_cm3=round(tm.volume/1000, 2), size_mm=np.round(tm.bounds[1] - tm.bounds[0], 4).tolist(),
                bed_z=round(float(zmin), 4), bed_contact_mm2=round(float(a[bed].sum()), 1),
                bed_film_mm2=round(float(a[film].sum()), 3), bed_fins_mm2=[round(float(f_), 3) for f_ in fins],
                bytes=os.path.getsize(path), min_shell_bed_contact_mm2=min(shell_bed) if shell_bed else 0.0, mb=round(os.path.getsize(path)/1e6, 2))


def rot_to_z(d):
    """Rotation (3x4) taking unit vector d to +z."""
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    v = np.cross(d, z); s_ = np.linalg.norm(v); c_ = float(d @ z)
    if s_ < 1e-9:
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx*((1 - c_)/s_**2)
    return np.c_[R, np.zeros(3)]


def section_geom(man, z):
    """Horizontal section as a shapely geometry.  Contours are combined even-odd, so islands that
    sit inside a hole (a box inside a frame) are kept."""
    g = Polygon()
    for p in man.slice(z).to_polygons():
        p = np.asarray(p, float)
        if len(p) >= 3:
            g = g.symmetric_difference(Polygon(p).buffer(0))
    return g


def measure_joint(j, Gm, wheel_local):
    """Clearance between the male and female of a joint, measured on the model-scale parts:
    both are cut by the plane through the joint point normal to its axis, and the smallest
    gap between the male section and the female material around it is reported."""
    p = np.asarray(j["p"])/SCALE
    T = rot_to_z(j["axis"])
    def local(m):
        return m.transform(np.c_[np.eye(3), -p]).transform(T)
    fem = Gm[j["female"]]
    if j["kind"] == "lens":
        male = Gm[j["male"]]
    elif j.get("male_shell") == "steering_wheel":
        male = Gm["steering_wheel"]
    elif j["male"] is None:
        male = None
    else:
        male = Gm[j["male"]]
    win = Point(0, 0).buffer(4.0 if j["kind"] != "lens" else 60.0)
    F = section_geom(local(fem), 0.0).intersection(win)
    if male is None:
        Mx = Point(0, 0).buffer(j["male_d"]/2, resolution=32)
    else:
        Mx = section_geom(local(male), 0.0).intersection(win)
        if j["kind"] != "lens":
            parts = [g for g in getattr(Mx, "geoms", [Mx]) if g.distance(Point(0, 0)) < 0.5]
            Mx = unary_union(parts) if parts else Mx
    if F.is_empty:
        raise ValueError(f"no {j['female']} material in the joint section")
    if Mx.is_empty:
        raise ValueError(f"no {j['male']} material in the joint section")
    return round(float(Mx.distance(F)), 3)


def plane_section(man, o, axis):
    """Section of man (model mm) by the plane through o normal to axis, in plane coordinates
    centred on o (a z plane is sliced directly; other planes after a rotation)."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    if abs(a[2]) > 0.999:
        g = section_geom(man, float(o[2]))
        from shapely import affinity
        g = affinity.translate(g, -o[0], -o[1])
        return g if a[2] > 0 else affinity.scale(g, 1.0, -1.0, origin=(0, 0))
    return section_geom(man.transform(np.c_[np.eye(3), -np.asarray(o)]).transform(rot_to_z(a)), 0.0)


def female_check(j, Gm):
    """The female feature of a joint, checked at three depths from its mouth: the hole around the
    joint axis must be a closed ring (also with a 0.4 mm wall: one extrusion width), at least
    male_d + 2 x 0.25 mm across, and a blind pocket must have a floor.  None if not applicable."""
    if j.get("fdepth") is None or j["male_d"] is None:
        return None
    fem = Gm[j["female"]]
    a = np.asarray(j["axis"], float); a /= np.linalg.norm(a)
    mouth = np.asarray(j["p"])/SCALE - a*j["back"]
    win = Point(0, 0).buffer(4.0, resolution=64)
    r_need = j["male_d"]/2 + CLR - 0.03                      # 0.03: facets of a 32-gon
    out = dict(closed=True, wall_04=True, wide=True, not_oversize=True, floor=None, open_at=None)
    a_max = 1.3*math.pi*(r_need + 0.03)**2                   # a teardrop roof adds ~20 %; counterbored screw holes are exempt
    for t in np.arange(0.2, j["fdepth"] - 0.05, 0.1):
        F = plane_section(fem, mouth + a*t, a).intersection(win)
        hole = win.difference(F)
        comp = [g for g in getattr(hole, "geoms", [hole]) if g.distance(Point(0, 0)) < 1e-9]
        closed = bool(comp) and comp[0].distance(win.exterior) > 1e-6
        if not closed:
            if out["closed"]:
                out["closed"], out["open_at"] = False, round(t, 2)
            continue
        H = Polygon(comp[0].exterior)
        if not H.buffer(1e-6).contains(Point(0, 0).buffer(r_need, resolution=32)):
            out["wide"] = False
        if j["kind"] != "screw" and H.area > a_max:
            out["not_oversize"] = False
        if t >= 0.3:                                       # one extrusion width of material all round the hole
            band = H.buffer(0.4, resolution=16).difference(H)
            if band.difference(F).area > 0.005:
                out["wall_04"] = False
    if j.get("blind"):                                     # the whole pocket disc must be closed by material just past the bottom
        disc = Point(0, 0).buffer(j["male_d"]/2 + CLR - 0.03, resolution=32)
        out["floor"] = any(plane_section(fem, mouth + a*t, a).buffer(1e-6).contains(disc)
                           for t in np.arange(j["fdepth"] + 0.05, j["fdepth"] + 0.55, 0.05))
    return out


def raster_layers(man, lo, px, shape, zs):
    import print_check as pc
    return [pc.raster(man, z, lo, px, shape) for z in zs]


def path_blockers(P, Q, d_remove=(0, 0, 1), px=0.15, dz=0.25):
    """Can P be lifted out along d_remove without touching the already-installed Q?
    Returns the blocked area (mm2, model) and where.  Q blocks if some of its material sits in
    a column above material of P (in the removal direction)."""
    import print_check as pc
    T = rot_to_z(d_remove)
    Pr, Qr = P.transform(T), Q.transform(T)
    bp, bq = bbox(Pr), bbox(Qr)
    lo = np.maximum(bp[:3], bq[:3]); hi = np.minimum(bp[3:], bq[3:])
    if np.any(hi[:2] <= lo[:2]):
        return 0.0, None
    z0, z1 = bp[2], max(bp[5], bq[5])
    shape = (int(math.ceil((hi[0] - lo[0])/px)) + 3, int(math.ceil((hi[1] - lo[1])/px)) + 3)
    lo2 = lo[:2] - px
    seen = np.zeros(shape, bool)
    blocked = np.zeros(shape, bool)
    where = None
    for z in np.arange(z0 + dz/2, z1, dz):
        qk = pc.raster(Qr, z, lo2, px, shape) if bq[2] <= z <= bq[5] else None
        if qk is not None:
            b = qk & seen
            if b.any() and where is None:
                ij = np.argwhere(b)[0]
                where = [round(float(lo2[0] + ij[0]*px), 2), round(float(lo2[1] + ij[1]*px), 2), round(float(z), 2)]
            blocked |= b
        if bp[2] <= z <= bp[5]:
            seen |= pc.raster(Pr, z, lo2, px, shape)
    opened = ndimage_open(blocked)
    area = float(opened.sum())*px*px
    if area == 0:
        return 0.0, None
    ij = np.argwhere(opened)[0]
    return round(area, 2), [round(float(lo2[0] + ij[0]*px), 2), round(float(lo2[1] + ij[1]*px), 2), where[2] if where else None]


def ndimage_open(mask):
    from scipy import ndimage
    return ndimage.binary_opening(mask, structure=np.ones((2, 2), bool))   # ignore single-pixel rasterisation specks


CARD_SHIFT = 10.0         # model mm: a door card is lowered into the shell this far inboard, then pressed outward

ASSEMBLY = [  # (step, group, removal direction, installed after)
    (1, "baseplate", None), (2, "harness", (0, 0, 1)), (3, "front_sub", (0, 0, 1)), (3, "rear_sub", (0, 0, 1)),
    (4, "eaxle_front", (0, 0, 1)), (4, "eaxle_rear", (0, 0, 1)), (5, "tray", (0, 0, 1)), (5, "modules", (0, 0, 1)),
    (6, "floor_tub", (0, 0, 1)), (7, "cargo_deck", (0, 0, 1)), (8, "dashboard", (0, 0, 1)), (8, "seats", (0, 0, 1)),
    (8, "bench", (0, 0, 1)), (11, "main", (0, 0, 1)), (13, "hood", (0, 0, 1)), (13, "canopy", (0, 0, 1)),
    (13, "tail", (1, 0, 0)),
]


def validate(G, W, wl, out_files, log=print):
    """Interference, lowering paths, joint clearances (model mm)."""
    Gm = {k: to_model(v) for k, v in G.items()}
    res = {}
    # ---- interference between installed parts
    names = [k for k in Gm if k not in ("steering_wheel",)] + ["steering_wheel"]
    inter = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = Gm[names[i]], Gm[names[j]]
            if not overlaps(a, b):
                continue
            v = (a ^ b).volume()
            if v > 0.005:
                inter[f"{names[i]} x {names[j]}"] = round(v, 3)
    res["interference_mm3"] = inter
    log(f"  interference: {len(inter)} pairs above 0.005 mm3")
    shell_gaps = {}
    for k, v in Gm.items():                       # the hood rests on its ledge and the door cards mount into the shell
        if k not in ("main", "hood", "door_cards") and overlaps(Gm["main"], v, 1.0):
            shell_gaps[k] = round(float(Gm["main"].min_gap(v, 1.0)), 3)
    res["shell_gaps_mm"] = shell_gaps
    log(f"  body shell clearance: smallest {min(shell_gaps.values()):.3f} mm ({min(shell_gaps, key=shell_gaps.get)})")
    # ---- lowering paths in assembly order
    paths = {}
    installed = [Gm["baseplate"]]
    for step, g, d in ASSEMBLY[1:]:
        if g == "main":                      # the door cards are on the shell; the wheels are already on the axles
            installed += [Gm["rims"], Gm["tyres"]]
        Q = K.union(installed)
        P = K.union([Gm["main"], Gm["door_cards"]]) if g == "main" else Gm[g]
        area, where = path_blockers(P, Q, d)
        paths[g] = dict(step=step, blocked_mm2=area, at=where)
        installed.append(Gm[g])
    card_area, card_at = 0.0, None
    for s in (1, -1):
        half = boxm((-1, 0 if s > 0 else -200, -1), (400, 200 if s > 0 else 0, 200))
        cards = Gm["door_cards"] ^ half
        for P_, Q_, d_ in ((cards, Gm["main"] ^ half, (0, -s, 0)),                       # off the pins, inward
                           (cards.translate([0, -s*CARD_SHIFT, 0]), Gm["main"], (0, 0, 1))):   # then up out of the shell
            a_, w_ = path_blockers(P_, Q_, d_)
            if a_ > card_area:
                card_area, card_at = a_, w_
    paths["door_cards"] = dict(step=10, blocked_mm2=card_area, at=card_at,
                               direction=f"along y onto the pins, from {CARD_SHIFT:g} mm inboard after lowering into the shell")
    lens = Gm["lenses"].decompose()
    lens.sort(key=lambda m: m.bounding_box()[0])
    a1, w1 = path_blockers(lens[0], Gm["main"], (-1, 0, 0))
    a2, w2 = path_blockers(lens[-1], Gm["tail"], (1, 0, 0))
    paths["lenses"] = dict(step=12, blocked_mm2=max(a1, a2), at=w1 or w2, direction="into the lamp windows from outside")
    res["assembly_paths"] = paths
    log(f"  lowering paths checked for {len(paths)} parts")
    # ---- joint clearances
    Gm["steering_wheel"] = to_model(G["steering_wheel"])
    meas = []
    by_file = {FILE_OF[k]: v for k, v in Gm.items() if k in FILE_OF}
    by_file["steering_wheel"] = Gm["steering_wheel"]
    by_file["cargo_deck"] = Gm["cargo_deck"]
    for j in JOINTS:
        try:
            gap, err = measure_joint(j, by_file, wl), None
        except Exception as e:                                # noqa: BLE001 - recorded in the manifest
            gap, err = None, f"{type(e).__name__}: {e}"
        try:
            fc = female_check(j, by_file)
        except Exception as e:                                # noqa: BLE001
            fc = dict(error=f"{type(e).__name__}: {e}")
        meas.append(dict(j, measured_gap_mm=gap, female=j["female"], female_check=fc, **({"error": err} if err else {})))
    # tyre on rim and wheel bore, measured on the local wheel parts
    rim, tyre = W["rim"], W["tyre"]
    zmid = v4.WW/2
    meas.append(dict(kind="tyre_on_rim", male="wheels_rims_x4", female="tires_treaded_x4", p=[0, 0, zmid], axis=[0, 0, 1],
                     male_d=None, depth=None, measured_gap_mm=round(float(section_geom(rim, zmid).distance(section_geom(tyre, zmid))), 3)))
    bore = section_geom(rim, 1.0)
    meas.append(dict(kind="axle_in_hub", male=None, female="wheels_rims_x4", p=[0, 0, 1.0], axis=[0, 0, 1], male_d=3.0, depth=None,
                     measured_gap_mm=round(float(Point(0, 0).buffer(1.5, resolution=32).distance(bore)), 3)))
    res["joints"] = meas
    res["lens_heights_mm"] = sorted(round(float(c.bounding_box()[5] - c.bounding_box()[2]), 3) for c in Gm["lenses"].decompose())
    gaps = [m["measured_gap_mm"] for m in meas if m["measured_gap_mm"] is not None]
    log(f"  joints measured: {len(gaps)} of {len(meas)}, gap {min(gaps):.3f}-{max(gaps):.3f} mm")
    bad = [m for m in meas if m.get("female_check") and not female_ok(m["female_check"])]
    for m in bad:
        log(f"  FEMALE FEATURE PROBLEM: {m['kind']} in {m['female']} at {[round(v) for v in m['p']]}: {m['female_check']}")
    res["female_features_checked"] = sum(1 for m in meas if m.get("female_check"))
    res["female_features_failed"] = len(bad)
    return res


# ---------------------------------------------------------------------------
# 12. print settings, bill of materials, assembly guide
# ---------------------------------------------------------------------------
SETTINGS = {   # file stem: (material, layer mm, walls, infill %, note)
    "chassis_baseplate_m2_bosses": ("PLA or PETG", 0.20, 3, 20, "bridging on; brim 3 mm"),
    "front_subframe_suspension": ("PLA or PETG", 0.20, 3, 25, "slow outer walls for the springs"),
    "rear_subframe_suspension": ("PLA or PETG", 0.20, 3, 25, ""),
    "battery_tray_lower_housing": ("PLA or PETG", 0.20, 3, 20, ""),
    "battery_module_cells_combined": ("PLA or PETG", 0.20, 2, 100, "8 separate pieces on one plate"),
    "front_e_axle_and_inverter": ("PLA or PETG", 0.20, 3, 25, "minimum layer time 10 s for the thin pipe risers"),
    "rear_e_axle_and_inverter": ("PLA or PETG", 0.20, 3, 25, ""),
    "hv_wiring_harness_conduits": ("PETG (orange)", 0.20, 2, 100, "2 pieces; print slowly"),
    "interior_floor_tub": ("PLA", 0.12, 3, 15, ""),
    "front_bucket_seats_pair": ("PLA", 0.12, 3, 15, ""),
    "rear_bench_seat": ("PLA", 0.12, 3, 15, ""),
    "dashboard_steering_console": ("PLA", 0.12, 3, 15, ""),
    "door_cards_set": ("PLA", 0.12, 2, 100, "4 cards"),
    "removable_roof_solar_canopy": ("PLA", 0.12, 3, 15, "a near-solid greenhouse block: use gyroid infill"),
    "hood_solar_bonnet": ("PLA", 0.12, 3, 15, ""),
    "main_outer_body_shell": ("PLA", 0.12, 3, 15, "240 mm long: needs a 250 mm bed"),
    "tailgate_rear_hatch": ("PLA", 0.12, 3, 15, ""),
    "wheels_rims_x4": ("PLA or PETG", 0.12, 3, 30, ""),
    "tires_treaded_x4": ("TPU 95A (or PLA)", 0.12, 3, 20, "TPU at 20 mm/s with short or no retraction"),
    "clear_lens_headlights_taillights": ("clear PETG or resin", 0.12, 4, 100, "or clear resin"),
}
LENS_HEIGHTS = [(932.0 - 880.0)/20, (1214.0 - 1170.0)/20]   # headlight and tail-bar windows, model mm (lens = window band)
LEDS = [("headlight channel behind the headlight lens strip (6 per side)", 12, "0603 SMD LED, white, 1.6 x 0.8 x 0.6 mm, pre-wired with 0.1 mm enamelled wire (or 1.8 mm round LEDs)"),
        ("tail-bar channel behind the tail lens", 6, "0603 SMD LED, red, 1.6 x 0.8 x 0.6 mm, pre-wired"),
        ("driver cluster light box", 1, "0603 SMD LED, white, 1.6 x 0.8 x 0.6 mm, pre-wired"),
        ("centre touchscreen light box", 1, "0603 SMD LED, white, 1.6 x 0.8 x 0.6 mm, pre-wired")]


def female_ok(fc):
    return ("error" not in fc and fc["closed"] and fc["wall_04"] and fc["wide"] and fc.get("not_oversize", True)
            and fc["floor"] is not False)


def support_verdict(sup):
    if sup is None or "error" in sup:
        return "check failed"
    bad = []
    if sup["long_bridges"]:
        bad.append(f"{sup['long_bridges']} long bridges")
    if sup["max_cantilever_mm"] > 2.4:
        bad.append(f"cantilever {sup['max_cantilever_mm']} mm")
    if sup.get("max_island_mm2", 0) > 1.0:
        bad.append(f"island {sup['max_island_mm2']} mm2")
    return "none" if not bad else "tree supports (" + ", ".join(bad) + ")"


def bom():
    screws = {}
    for s_ in SCREWS:
        key = (s_["size"], s_["length"])
        screws.setdefault(key, []).append(s_["where"])
    mags = sum(m["count"] for m in MAGNETS)
    return screws, mags


def assembly_steps(screws):
    def fix(group):
        c = {}
        for s_ in SCREWS:
            if s_["group"] == group:
                c[(s_["size"], s_["length"])] = c.get((s_["size"], s_["length"]), 0) + 1
        if len(c) == 1:
            return " and ".join(f"{n} x {sz} x {L} mm" for (sz, L), n in sorted(c.items()))
        parts = []
        for (sz, L), n in sorted(c.items()):
            at = "; ".join(s_["where"].split(" at ")[1].split(" mm")[0] for s_ in SCREWS if s_["group"] == group and s_["length"] == L)
            parts.append(f"{n} x {sz} x {L} mm (at {at} mm)")
        return " and ".join(parts)
    return [
        ("Baseplate", "Lay `chassis_baseplate_m2_bosses` on the bench. Glue a 3x2 mm magnet into the top of each of its two cargo posts (mind the polarity: all four post magnets the same pole up)."),
        ("Harness", "Drop `hv_wiring_harness_conduits` (2 pieces) onto the baseplate: the rocker connectors sit in their sill pockets and the runs lie flat on the undertray. The two refrigerant risers of the front piece stand just ahead of the cabin floor line."),
        ("Subframes", "Lower `front_subframe_suspension` (cradle, front suspension and steering, bumper beam, crush cans and rails) and `rear_subframe_suspension` (cradle, rear suspension, rear rails with their kick-ups, lower struts, two cargo posts) onto the baseplate. The boss pads register in the pockets under the cradles; the harness runs pass through the crossmember notches. "
                      f"Fix the rear subframe with {fix('rear_sub')} button-head screws from above. Glue a magnet into each rear cargo post."),
        ("E-axles", "Lower `front_e_axle_and_inverter` (with the cooling module, heat pump and charging stack) and `rear_e_axle_and_inverter` between the knuckles: two boss pads register each unit, and the pipe and cable risers land on the ends of the floor harness. Slide each 3 mm x 88.25 mm axle through its bores: front knuckle, subframe, drive unit, subframe, knuckle; rear knuckle, drive unit, knuckle. Glue a magnet into the top of each of the two hood posts on the radiator shroud."),
        ("Battery", "Lower `battery_tray_lower_housing` between the sills and fix it to the baseplate bosses with "
                    f"{fix('tray')} button-head screws from above. Drop the 8 module pairs of `battery_module_cells_combined` into the bays between the cross-members."),
        ("Cabin floor", f"Set the cabin floor of `interior_floor_tub` on the battery: the two pegs on the battery front wall locate it. Fix it with {fix('floor_tub')} button-head screws into the battery side frames in the rear footwells (the heads sit in notches at the foot of the sill trims)."),
        ("Cargo deck", "Glue four magnets into the bosses under the cargo deck (second piece of `interior_floor_tub`, printed upside down), with the pole that attracts the post magnets facing down, and set it on its four posts over the rear e-axle."),
        ("Seats and dashboard", "Push `front_bucket_seats_pair`, `rear_bench_seat` and `dashboard_steering_console` onto their floor pegs (a drop of glue if wanted). Plug the steering wheel's 2 mm peg into the column socket."),
        ("Wheels", "Slide each tyre of `tires_treaded_x4` onto a rim of `wheels_rims_x4` from the inner face until it seats on the outer lip (glue if printed in PLA). Push the wheels onto the axle ends and fix with a drop of CA."),
        ("Door cards", "Press the four cards of `door_cards_set` onto the inside of the body shell's door panels (two pins each, from inside the shell)."),
        ("Body shell", "Thread the headlight and dashboard LED wires out through the 4 mm undertray exits; the tail-bar wires leave through the rear wheel houses. Lower `main_outer_body_shell` (the rear bumper beam, crush cans and upper struts are moulded inside its rear bumper) over the cabin onto the sill pins and screw it from below with "
                       f"{fix('main')} button-head screws (bumper corners and rocker ends)."),
        ("Lenses", "Fit the two clear lenses of `clear_lens_headlights_taillights` into the headlight window (body shell) and the tail-bar window (tailgate) with a drop of clear UV glue."),
        ("Hood, roof, tailgate", "Glue the magnets into their pockets (each pair facing with opposite poles). Set `hood_solar_bonnet` on its ledge: 2 magnet pairs at the rear corners of the ledge and 2 on the radiator-shroud posts. Then lower `removable_roof_solar_canopy` onto its four pins (A-pillar and C-pillar bases); its magnets meet the C-pillars. Finally slide `tailgate_rear_hatch` forward into the liftgate opening from behind until its hinge magnets meet the roof and its bottom magnets sit on the bumper ledge."),
    ]


FIT_NAMES = {  # (kind, male, female) -> (joint, male feature, female feature)
    ("pin", "chassis_baseplate_m2_bosses", "main_outer_body_shell"): ("Sill pins", "Ø1.5 pin on the baseplate rockers", "Ø2.0 socket in the body-shell sill"),
    ("pin", "main_outer_body_shell", "removable_roof_solar_canopy"): ("Roof pins", "Ø1.5 pin on the A/C-pillar bases", "Ø2.0 socket in the roof canopy"),
    ("pin", "door_cards_set", "main_outer_body_shell"): ("Door-card pins", "Ø1.5 pin on each card", "Ø2.0 socket in the door panel"),
    ("peg", "interior_floor_tub", None): ("Floor pegs", "Ø2.0 peg on the cabin floor", "Ø2.5 socket in seats / bench / dashboard"),
    ("peg", "battery_tray_lower_housing", "interior_floor_tub"): ("Battery pegs", "Ø2.0 peg on the battery front wall", "Ø2.5 hole in the cabin floor"),
    ("peg", "dashboard_steering_console", "dashboard_steering_console"): ("Steering wheel", "Ø2.0 peg on the wheel hub", "Ø2.5 socket in the column"),
    ("boss", None, None): ("Locating bosses", "Ø3.0 x 1.5 mm boss pad on the baseplate", "Ø3.5 pocket under the subframes, battery tray and e-axles"),
    ("magnet", None, None): ("Magnet pockets", "3 x 2 mm disc magnet", "Ø3.5 x 2.25 mm pocket"),
    ("screw", None, None): ("Screw clearance holes", "M2 screw shank", "Ø2.5 clearance hole"),
    ("cbore", None, None): ("Screw-head counterbores", "M2 button head Ø3.5", "Ø4.0 flat-bottomed counterbore"),
    ("axle", None, None): ("Axle bores", "Ø3.0 steel or carbon rod", "Ø3.5 bore"),
    ("axle_in_hub", None, None): ("Wheel hub", "Ø3.0 axle", "Ø3.5 hub bore"),
    ("tyre_on_rim", None, None): ("Tyre on rim", "rim seat", "tyre bead (0.25 mm larger)"),
    ("lens", None, None): ("Clear lenses", "lens strip (LED groove on the back)", "lamp window (0.25 mm larger all round)"),
}


def fit_name(j):
    for key in ((j["kind"], j.get("male"), j.get("female")), (j["kind"], j.get("male"), None), (j["kind"], None, None)):
        if key in FIT_NAMES:
            return FIT_NAMES[key]
    return (j["kind"], "-", "-")


def write_guide(out, rows, val, screws, mags, args, info, plates_3mf, problems, elapsed):
    L = []
    w = L.append
    w("# Solar SUV v4: 1:20 modular model kit, assembly and print guide\n")
    checked = not val.get("skipped")
    w(f"This kit was generated by `generate_model_kit.py --res {info.get('res', args.res):g}` (geometry {info['time']/60:.0f} min, "
      f"{elapsed/60:.0f} min with export and checks). "
      f"It has 20 STL files in 5 folders" + ("" if args.no_3mf else ", plus one 3MF per folder") + ". "
      + ("Every file was reloaded from disk and checked (section 8). " if checked else
         "**The support, joint, interference and assembly checks were skipped (`--skip-checks`).** ")
      + "The solids and booleans use the manifold3d kernel, and the meshes are checked with trimesh. CadQuery and SolidPython "
        "are not needed.\n")
    if problems:
        w("**Problems found by the checks:**\n")
        for p_ in problems:
            w(f"- {p_}")
        w("")
    w("## 1. Files\n")
    w("```")
    w("solar_suv_1to20_model_kit/")
    folders = []
    for r in rows:
        if r["folder"] not in folders:
            folders.append(r["folder"])
    w("├── stls/")
    for i, fo in enumerate(folders):
        last = i == len(folders) - 1
        w(f"│   {'└──' if last else '├──'} {fo}/")
        items = [r for r in rows if r["folder"] == fo]
        names = [f"{r['stem']}.stl ({r['mesh']['shells']} piece{'s' if r['mesh']['shells'] > 1 else ''}, {r['mesh']['bytes']/1e6:.1f} MB)" for r in items]   # 1 MB = 10^6 bytes
        names += [f"{fo}_parts.3mf (all {len(items)} parts of the folder)"] if not args.no_3mf else []
        for k, nme in enumerate(names):
            w(f"│   {'    ' if last else '│   '}{'└──' if k == len(names) - 1 else '├──'} {nme}")
    w("├── ASSEMBLY_AND_PRINT_GUIDE.md")
    w("└── kit_manifest.json          (all checks and measured clearances, machine-readable)")
    w("```\n")
    w("## 2. Units, scale and orientation\n")
    w("- **1 unit = 1 mm on the model** (1:20). The car is 4.80 m long, so the body shell is 240 mm long.")
    w("- Every STL lies with its print face **flat at z = 0**: the chassis baseplate on its undertray, the cabin floor tub on the "
      "floor plate, the roof canopy on its belt-line underside, the hood on its underside, the body shell and tailgate upright on "
      "their bottom edges.")
    w("- Each part keeps its car x/y position divided by 20. The exceptions are the wheel and tyre plates (laid out 2 x 2), the "
      "steering wheel (laid in front of the dashboard) and the cargo deck (6 mm behind the cabin floor, upside down). Import the "
      "single-piece files together and raise each by `installed_z_offset_mm` from `kit_manifest.json` to see the assembled car.")
    if not args.no_3mf:
        w("- Each folder's 3MF holds the same meshes in millimetres, one object per file, laid out in rows up to 250 mm wide. It is "
          "a project file, not a ready plate: some parts need different materials (TPU tyres, clear lenses) and the folders do not all "
          "fit one bed. Split the parts onto plates in your slicer.")
    w("")
    w("## 3. Fits and tolerances\n")
    w("Every male/female joint has **0.25 mm clearance per side**: the female feature is 0.5 mm larger in diameter than the male "
      "one, and 0.25 mm deeper. Where two parts touch, the clearance pass keeps them 0.25 mm apart. A part that is installed "
      "first gets open-top slots and recesses, and a part that goes on later gets notches that are open at the bottom, so every "
      "part can be lowered straight into place. There are no snap-fits: parts are located by pins, pegs and boss pads, and "
      "held by magnets and screws.\n")
    if checked:
        w("Two checks were run on the final part geometry. These are the same meshes that are exported, before they are moved onto "
          "the bed.")
        w("- **Gap**: each joint was cut normal to its axis, and the smallest gap between male and female was measured.")
        w("- **Female feature**: each hole or pocket was cut every 0.1 mm from 0.2 mm below its mouth to its bottom. At every depth "
          "it must be a closed ring with at least 0.4 mm (one extrusion) of material all round. It must be at least 0.5 mm wider "
          "than its male part and not oversized. Blind pockets must also be closed by a full floor.\n")
    if checked:
        w("| Joint | Male | Female | Measurements | Measured gap (mm) | Female feature check |")
        w("|---|---|---|---|---|---|")
    groups = {}
    for j in val["joints"]:
        name, male, fem = fit_name(j)
        g = groups.setdefault(name, dict(male=male, fem=fem, gaps=[], missing=0, fc_ok=0, fc_n=0))
        if j["measured_gap_mm"] is None:
            g["missing"] += 1
        else:
            g["gaps"].append(j["measured_gap_mm"])
        if j.get("female_check"):
            g["fc_n"] += 1
            g["fc_ok"] += int(female_ok(j["female_check"]))
    for name, g in groups.items():
        rng = f"{min(g['gaps']):.3f} - {max(g['gaps']):.3f}" if g["gaps"] else "-"
        miss = f" ({g['missing']} not measurable)" if g["missing"] else ""
        fc = f"{g['fc_ok']}/{g['fc_n']} pass" if g["fc_n"] else "n/a (fit along the whole part)"
        w(f"| {name} | {g['male']} | {g['fem']} | {len(g['gaps']) + g['missing']} | {rng}{miss} | {fc} |")
    w("")
    gaps = [j["measured_gap_mm"] for j in val["joints"] if j["measured_gap_mm"] is not None]
    if not checked:
        w("Not measured in this run (`--skip-checks`).\n")
    elif gaps and min(gaps) >= CLR - 0.02:
        w(f"The smallest gap is {min(gaps):.3f} mm: gaps a little under 0.25 mm come from the faceting of round holes. The M2 screw "
          "pilots (Ø1.6 mm) are deliberately undersized for self-tapping.\n")
    else:
        w(f"**PROBLEM: the smallest measured gap is {min(gaps) if gaps else 'n/a'} mm.**\n")
    w("## 4. Recommended print settings\n")
    w("FDM, 0.4 mm nozzle. Use **0.12 mm layers for the interior and body parts** and **0.20 mm for the chassis and powertrain**. "
      "Turn bridge detection on. The *Supports* column comes from a layer-by-layer check (`tools/print_check.py`). A file is marked "
      "as needing supports if it has a bridge over 12 mm, an overhang that reaches more than 2.4 mm past the 45° line, or an island over "
      "1 mm². These limits are looser than print_check's strict default (0.8 mm), which flags any visible sag. Section 8 lists the "
      "largest overhang and island in every file, so you can see where some sag may show. Use a 5 mm brim on tall parts that "
      "stand on a narrow edge: the body shell, tailgate, door cards and lenses.\n")
    w("| File | Pieces | Size x, y, z (mm) | Material | Layer | Walls | Infill | Supports | Orientation / notes |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        mat, lay, walls, inf, note = SETTINGS[r["stem"]]
        m = r["mesh"]
        w(f"| `{r['stem']}` | {m['shells']} | {' x '.join(f'{v:.1f}' for v in m['size_mm'])} | {mat} | {lay:.2f} mm | {walls} | "
          f"{inf} % | {support_verdict(r.get('support')) if checked else 'not checked'} | {r['orient']}{'; ' + note if note else ''} |")
    w("")
    w("## 5. Hardware\n")
    w("Screw positions are in the coordinates of the part's STL (model mm, car x rearward, y to the right): put the listed length "
      "in each hole.\n")
    w("| Item | Qty | Where |")
    w("|---|---|---|")
    for (size, length), where in sorted(screws.items()):
        w(f"| {size} x {length} mm button-head screw (ISO 7380 style, head Ø3.5 x 1.3 mm), self-tapping into a printed Ø1.6 pilot | {len(where)} | "
          + "; ".join(where) + " |")
    w(f"| 3 x 2 mm N52 neodymium disc magnet | {mags} | " + "; ".join(f"{m['where']} ({m['count']})" for m in MAGNETS) + " |")
    w("| 3 mm steel or carbon rod, 88.25 mm long | 2 | front axle: knuckle, subframe, drive unit, subframe, knuckle; rear axle: knuckle, drive unit, knuckle |")
    for where, q, kind in LEDS:
        w(f"| {kind} | {q} | {where} |")
    w("| 0.1 mm enamelled copper wire | 2 m | LED feeds through the wire channels |")
    w(f"| Resistor, 1 per LED, on a 5 V supply | {sum(q for _w, q, _k in LEDS)} | white LEDs 220-270 ohm, red LEDs 330-390 ohm (about 8 mA): R = (5 V - Vf) / I |")
    w("| CA glue, clear UV-curing glue | - | magnets, tyres (if PLA), lenses |")
    w("")
    w("## 6. LED lighting (optional)\n")
    w("- **Lens strips**: each clear lens has a 0.8 x 0.5 mm groove along its back face. The groove sits in front of the LED channel, "
      "so the light reaches the whole strip; lay the LED wire or a strip of 0603 LEDs in it.")
    w("- **Headlights**: a 4 mm channel runs behind the headlight lens strip in the body shell, with a 3 mm wire exit back along the centre line.")
    w("- **Tail bar**: a 4 mm channel runs behind the tail lens strip in the tailgate, with a 3 mm wire exit forward.")
    w("- **Cluster and centre display**: these are light boxes behind the windows of the dashboard pod.")
    w("- **Wire exits**: two Ø4 mm holes through the undertray just ahead of the battery (baseplate x = 65 mm, y = ±21 mm) take "
      "the headlight and dashboard wires out under the model; the front subframe and harness cover part of each hole, so feed the "
      "wire through its open inboard side. Lead the tail-bar wires down the inside of the rear quarters and out through the rear "
      "wheel houses, which are open on the inboard side.")
    w("- **Wire size**: 0603 SMD LEDs (1.6 x 0.8 x 0.6 mm) with 0.1 mm enamelled wire pass every channel. Round 1.8 mm LEDs fit the 4 mm channels.\n")
    w("## 7. Assembly sequence\n")
    for i, (title, text) in enumerate(assembly_steps(screws), start=1):
        w(f"{i}. **{title}.** {text}")
    w("")
    w("To show the inside, take the kit apart in reverse order:")
    w("- Roof canopy and tailgate off: the cabin.")
    w("- Hood off: the front bay.")
    w("- Body shell off (screws from below): the whole interior and chassis.")
    w("- Cabin floor out (2 screws): the battery, motors and harness.\n")
    w("## 8. Verification\n")
    meshes_ok = all(r["mesh"]["watertight"] and r["mesh"]["edges_2_manifold"] and r["mesh"]["shells"] == r["mesh"]["shells_expected"]
                    and abs(r["mesh"]["bed_z"]) <= 1e-6 and r["mesh"]["bed_contact_mm2"] > 0 for r in rows)
    scale_ok = all(r["scale"].startswith("ok") for r in rows)
    w("Each STL was reloaded from disk and checked: watertight, every edge shared by exactly 2 faces, the expected number of "
      "pieces, the lowest point at z = 0 with a flat bed face, and the size against the full-size geometry divided by 20. "
      + ("Every file passes." if meshes_ok and scale_ok else "**Some files fail: see the table and the problem list at the top.**") + "\n")
    w("| File | Triangles | Watertight | 2-manifold | Pieces (expected) | Bed z | Bed contact | Scale check | Max bridge | Max cantilever | Islands (largest) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m, sp = r["mesh"], r.get("support") or {}
        w(f"| `{r['stem']}` | {m['triangles']:,} | {'yes' if m['watertight'] else 'NO'} | {'yes' if m['edges_2_manifold'] else 'NO'} | "
          f"{m['shells']} ({m['shells_expected']}) | {m['bed_z']:.3f} | {m['bed_contact_mm2']:,.0f} mm² | {r['scale']} | "
          f"{sp.get('max_bridge_mm', '-')} mm | {sp.get('max_cantilever_mm', '-')} mm | {sp.get('islands', '-')} ({sp.get('max_island_mm2', '-')} mm²) |")
    w("")
    if checked:
        isl = max((r.get("support") or {}).get("max_island_mm2", 0.0) for r in rows)
        w("The bridge, overhang and island figures come from print_check run at each file's recommended layer height. Islands are "
      "tiny layer regions that start in the air, such as the tip of a downward-pointing feature. "
          + (f"The largest is {isl} mm². That is small enough to print without supports, but expect a little sag at those spots."
             if isl <= 1.0 else f"**The largest is {isl} mm²: add supports there.**") + "\n")
    if plates_3mf:
        ok = all(p["unit_mm"] and p["all_watertight"] and p["objects"] == p["expected"] for p in plates_3mf.values())
        w(f"**3MF files**: {len(plates_3mf)} reloaded. " + ("Each has every part as a separate watertight object, in millimetres."
                                                             if ok else "PROBLEM: " + json.dumps(plates_3mf)) + "\n")
    if not checked:
        w("**Interference, assembly paths and joints**: not checked in this run (`--skip-checks`).\n")
    else:
        inter = val["interference_mm3"]
        w("**Interference**: every pair of installed parts was intersected. "
          + ("No pair overlaps by more than 0.005 mm³." if not inter else "**Overlaps: " + ", ".join(f"{k} {v} mm³" for k, v in inter.items()) + ".**") + "\n")
        sgp = val.get("shell_gaps_mm", {})
        if sgp:
            w(f"**Body shell clearance**: the shell keeps at least {min(sgp.values()):.2f} mm from every part it is lowered over "
              "(the hood rests on its ledge and the door cards mount into it, so they are not counted).\n")
        blocked = {k: v for k, v in val["assembly_paths"].items() if v["blocked_mm2"] > 0}
        w("**Assembly paths**: in assembly order, each part was moved along its insertion direction against everything installed "
          "before it. The insertion is straight down, except the door cards (outward onto their pins), the lenses (horizontally "
          "into their windows from outside) and the tailgate (forward). "
          "The body shell was checked with the door cards fitted and the wheels on. "
          + ("Nothing blocks any part." if not blocked else "**Blocked: " + ", ".join(f"{k} {v['blocked_mm2']} mm² at {v['at']}" for k, v in blocked.items()) + "**") + "\n")
        w(f"**Problems**: " + ("none." if not problems else f"{len(problems)}, listed at the top of this guide.") + "\n")
        w("**Known limitation**: where the printable support fill meets the skin of the body shell and tailgate, a few dozen "
          "triangle pairs cross each other by up to about 0.15 mm (a few mm² in all). The meshes are still watertight and "
          "2-manifold, and slicers merge these spots.\n")
    with open(os.path.join(out, "ASSEMBLY_AND_PRINT_GUIDE.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def scale_check(key, size, G, W):
    """Exported size against the full-size geometry divided by 20 (x, y, z for single parts;
    x, y for sets whose pieces drop to the bed separately; the 2 x 2 wheel plates by layout)."""
    size = np.asarray(size)
    if key in ("rims", "tyres"):
        dp = (lambda b: b[3:] - b[:3])(bbox(W["rim" if key == "rims" else "tyre"]))
        step = 2*v4.WR + 4.0
        ref = np.array([dp[0] + step, dp[1] + step, dp[2]])
        err = float(np.max(np.abs(size - ref)))
        what = f"tyre O.D. {2*v4.WR:.2f} mm = {2*v4.TYRE_R:.0f} mm full size" if key == "tyres" else f"rim O.D. {dp[0]:.2f} mm"
        return (f"ok, {what}" if err < 0.05 else f"OFF by {err:.2f} mm")
    if key == "dashboard":
        g = G["dashboard"]
    elif key == "floor_tub":
        g = K.union([G["floor_tub"], G["cargo_deck"]])
    else:
        g = G[key]
    ref = (bbox(g)[3:] - bbox(g)[:3])/SCALE
    if key == "floor_tub":
        ref[0] += DECK_SHIFT
    n = 3 if key not in ("lenses", "floor_tub", "dashboard") else 2
    if key == "dashboard":                                    # steering wheel laid in front: compare y and z
        err = float(np.max(np.abs(size[1:] - ref[1:])))
        return f"ok (1:20, ±{err:.3f} mm)" if err < 0.05 else f"OFF by {err:.2f} mm"
    err = float(np.max(np.abs(size[:n] - ref[:n])))
    return f"ok (1:20, ±{err:.3f} mm)" if err < 0.05 else f"OFF by {err:.2f} mm"


# ---------------------------------------------------------------------------
# 13. main
# ---------------------------------------------------------------------------
def script_hash():
    import hashlib
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


def check_summary(rows, val, plates_3mf, args, info):
    """Every failed check, as text (empty when the kit is good)."""
    out = []
    if info.get("script") != script_hash() or info.get("clr") != CLR:
        out.append(f"the parts were built by another version of this script ({info.get('script')}) - rebuild without --cache")
    for k, v in info.get("dropped", {}).items():
        if v > 1.0:
            out.append(f"{k}: {v} mm3 of detached material was dropped during the build")
    for r in rows:
        m, e = r["mesh"], r["export"]
        if not (m["watertight"] and m["edges_2_manifold"]):
            out.append(f"{r['stem']}: not watertight / 2-manifold")
        if m["shells"] != m["shells_expected"]:
            out.append(f"{r['stem']}: {m['shells']} pieces, expected {m['shells_expected']}")
        if abs(m["bed_z"]) > 1e-6 or m["bed_contact_mm2"] <= 0 or m["min_shell_bed_contact_mm2"] < 5.0:
            out.append(f"{r['stem']}: no flat bed face at z = 0 for every piece (smallest {m['min_shell_bed_contact_mm2']} mm2)")
        if not m["winding_consistent"]:
            out.append(f"{r['stem']}: inconsistent winding")
        if m.get("bed_film_mm2", 0.0) > 0.01:
            out.append(f"{r['stem']}: {m['bed_film_mm2']} mm2 of zero-thickness film on the bed plane")
        if m.get("bed_fins_mm2"):
            out.append(f"{r['stem']}: bed features thinner than 0.2 mm ({m['bed_fins_mm2']} mm2)")
        if e["dropped_mm3"] >= 0.5:
            out.append(f"{r['stem']}: {e['dropped_pieces']} extra pieces ({e['dropped_mm3']} mm3) dropped on export")
        if not r["scale"].startswith("ok"):
            out.append(f"{r['stem']}: scale check {r['scale']}")
        if not args.skip_checks and support_verdict(r.get("support")) != "none":
            out.append(f"{r['stem']}: supports needed ({support_verdict(r.get('support'))})")
    for f, p_ in (plates_3mf or {}).items():
        if not (p_["unit_mm"] and p_["all_watertight"] and p_["objects"] == p_["expected"]):
            out.append(f"{f}_parts.3mf: {p_}")
    if val.get("skipped"):
        return out
    for k, v in val["interference_mm3"].items():
        out.append(f"interference {k}: {v} mm3")
    for k, v in val.get("shell_gaps_mm", {}).items():
        if v < 0.2:
            out.append(f"body shell only {v} mm from {k} (0.25 mm clearance expected)")
    want = sorted(LENS_HEIGHTS)
    got = val.get("lens_heights_mm", [])
    if len(got) != len(want) or any(abs(g_ - w_) > 0.05 for g_, w_ in zip(got, want)):
        out.append(f"lens strip heights {got} mm, expected {want} mm (window minus 2 x 0.25 mm)")
    for k, v in val["assembly_paths"].items():
        if v["blocked_mm2"] > 0:
            out.append(f"assembly path of {k} blocked: {v['blocked_mm2']} mm2 at {v['at']}")
    for j in val["joints"]:
        g = j["measured_gap_mm"]
        where = f"{j['kind']} {j.get('male')} / {j['female']} at {[round(c) for c in j['p']]}"
        if g is None:
            out.append(f"joint not measurable: {where}: {j.get('error')}")
        elif g < CLR - 0.02:
            out.append(f"joint gap {g} mm < {CLR}: {where}")
        elif g > CLR + 0.1:
            out.append(f"joint gap {g} mm > {CLR + 0.1}: {where}")
        if j.get("female_check") and not female_ok(j["female_check"]):
            out.append(f"female feature: {where}: {j['female_check']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Solar SUV v4 1:20 modular model kit pipeline")
    ap.add_argument("--out", default=os.path.join(HERE, "solar_suv_1to20_model_kit"))
    ap.add_argument("--res", type=float, default=6.0, help="body field grid, full-size mm (6 = 0.3 mm on the model)")
    ap.add_argument("--mirrors", default="mirror", choices=["mirror", "camera", "none"])
    ap.add_argument("--no-3mf", action="store_true")
    ap.add_argument("--skip-checks", action="store_true", help="export only (no support / path / joint checks)")
    ap.add_argument("--cache", help="pickle of the built parts: loaded if it exists, else written after the build")
    args = ap.parse_args(argv)
    log = lambda *a: print(*a, flush=True)
    t0 = time.time()
    if args.cache and os.path.exists(args.cache):
        apply_tolerances()
        G, W, wl, info = load_cache(args.cache)
        same = (info.get("res"), info.get("mirrors"), info.get("clr"), info.get("script")) == (args.res, args.mirrors, CLR, script_hash())
        log(f"parts loaded from {args.cache}" + ("" if same else
            f" - WARNING: built with res {info.get('res')}, mirrors {info.get('mirrors')}, an older script or other tolerances; "
            "the guide and manifest report the cached settings"))
    else:
        G, W, wl, info = build(args.res, args.mirrors, log)
        if args.cache:
            save_cache(args.cache, G, W, wl, info)
    import print_check as pc
    stl_dir = os.path.join(args.out, "stls")
    for root, _dirs, files in os.walk(stl_dir):                  # no stale files from an earlier run
        for fn in files:
            if fn.endswith((".stl", ".3mf")):
                os.remove(os.path.join(root, fn))
    rows = []
    plates = {}
    for folder, stem, key, n_exp, orient in PARTS:
        d = os.path.join(args.out, "stls", folder)
        os.makedirs(d, exist_ok=True)
        m = print_layout(key, G, W, wl)
        v, f, rep = clean_for_stl(m, n_exp)
        path = os.path.join(d, stem + ".stl")
        write_stl(v, f, path)
        mr = mesh_report(path, n_exp)
        scale = scale_check(key, mr["size_mm"], G, W)
        row = dict(folder=folder, stem=stem, key=key, orient=orient, mesh=mr, export=rep, scale=scale, path=os.path.relpath(path, args.out))
        if key not in ("rims", "tyres", "lenses", "dashboard", "floor_tub"):
            row["installed_z_offset_mm"] = round(float(bbox(G[key])[2])/SCALE - v4.GC/SCALE, 6)   # raise by this to reassemble (wheels on the ground)
        if not args.skip_checks:
            try:
                row["support"] = pc.check(pc.load(path), layer=SETTINGS[stem][1])["summary"]
            except ValueError as e:
                row["support"] = dict(error=str(e))
        rows.append(row)
        plates.setdefault(folder, []).append((stem, v, f))
        log(f"  {folder}/{stem}.stl: {mr['triangles']:,} tri, watertight {mr['watertight']}, pieces {mr['shells']}/{n_exp}, "
            f"{mr['size_mm']} mm" + (f", supports: {support_verdict(row.get('support'))}" if not args.skip_checks else ""))
    plates_3mf = {}
    if not args.no_3mf:
        for folder, objs in plates.items():
            path = os.path.join(args.out, "stls", folder, f"{folder}_parts.3mf")
            write_3mf(path, objs)
            plates_3mf[folder] = check_3mf(path, len(objs))
            log(f"  {folder}_parts.3mf: {plates_3mf[folder]}")
    log(f"  export and print checks done ({(time.time() - t0)/60:.1f} min)")
    val = validate(G, W, wl, rows, log) if not args.skip_checks else dict(skipped=True, joints=[], interference_mm3={}, assembly_paths={})
    log(f"  validation done ({(time.time() - t0)/60:.1f} min)")
    screws, mags = bom()
    problems = check_summary(rows, val, plates_3mf, args, info)
    write_guide(args.out, rows, val, screws, mags, args, info, plates_3mf, problems, time.time() - t0)
    manifest = dict(generator="generate_model_kit.py", script_sha1=script_hash(), geometry_script_sha1=info.get("script"),
                    dropped_mm3=info.get("dropped", {}), bed_fins_removed_mm3=info.get("fins", {}), res=info.get("res", args.res),
                    mirrors=info.get("mirrors", args.mirrors), units="model mm (1:20)", clearance_per_side_mm=CLR, problems=problems,
                    tolerances=TOL, parts=rows, plates_3mf=plates_3mf, clearance_pass=info["carved"], validation=val,
                    screws=SCREWS, magnets=MAGNETS, leds=LEDS, optional_toolchains=OPTIONAL_FOUND if __name__ == "__main__" else {},
                    build_minutes=round((time.time() - t0)/60, 1))
    json.dump(manifest, open(os.path.join(args.out, "kit_manifest.json"), "w"), indent=1, default=float)
    for p_ in problems:
        log(f"  PROBLEM: {p_}")
    log(f"done in {(time.time() - t0)/60:.1f} min: {len(rows)} files, "
        + ("every check passed" if not problems else f"{len(problems)} problems") + (" (checks skipped)" if args.skip_checks else ""))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
