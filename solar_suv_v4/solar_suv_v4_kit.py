#!/usr/bin/env python3
# =====================================================================
#  SOLAR SUV v4 - INTERNAL ARCHITECTURE & 1:20 MODULAR KIT
#
#  Turns the v4 exterior (solar_suv_v4_1to20.py) into a multi-part model kit:
#    chassis      skateboard: undertray, rockers, 112s2p battery pack (16 modules,
#                 224 prismatic cells), BJB + BMS, crash rails and subframes,
#                 dual e-axles with inverters, MacPherson front / multi-link rear,
#                 steering rack, sway bars, radiator, heat-pump module, MPPT/OBC,
#                 orange HV harness and coolant loops
#    interior     removable cabin floor: 5 seats, IP, steering column + stalks,
#                 cluster + infotainment light boxes, console, pedals, HVAC module,
#                 B-pillar belt retractors
#    shells       front clip, rear clip (with liftgate), 4 doors with door cards,
#                 hood, greenhouse canopy (roof), cargo deck
#    small parts  steering wheel, M2/M3 screw list, 3x2 magnets, wheels, axles
#
#  Every printed part is a watertight single shell with a flat bed face; undersides
#  are built at 46-50 degrees (or bridges of about 12 mm) so FDM needs no supports.
#  Geometry is in full-size millimetres (x rearward from the front bumper, y to the
#  car's right, z up from the road), exported at 1:20 with the floor (z = 210) at 0.
#  Print tolerances (clearances, magnet and screw pockets) are in model millimetres.
# =====================================================================
import math, json, time
import numpy as np
from manifold3d import Manifold, Mesh, CrossSection, OpType, FillRule
from shapely.geometry import Polygon, LineString, Point, box as sbox
from shapely.ops import unary_union
import solar_suv_v4_1to20 as v4

if not hasattr(Manifold, "intersect"):              # manifold3d spells intersection as `^`
    Manifold.intersect = lambda self, other: self ^ other

SCALE, GC = v4.SCALE, v4.GC
fast_simplification = v4.fast_simplification
M = v4.M                                   # model mm -> full-size mm
X_FA, X_RA, HUB_Z = v4.X_FA, v4.X_RA, v4.HUB_Z
T48 = math.tan(math.radians(48))           # design angle for undersides (2 deg margin on 46)

# ---------------------------------------------------------------------
#  KIT SPLIT AND PRINT TOLERANCES
# ---------------------------------------------------------------------
Z_SKIRT = 352.0          # chassis / body split = door bottom line (full size)
Z_CAN = 1105.0           # body / greenhouse canopy split (just under the 1118 beltline)
T_SHELL = 1.6            # shell wall (model mm)
T_FLOOR = 1.2            # undertray plate (model mm)
D_DOOR = 4.5             # door thickness from the skin to the door card (model mm)
GAP = 0.15               # sliding clearance (model mm)
SPLIT = 0.10             # half-gap at a split face (model mm)
MAG_D, MAG_H = 3.2, 2.1  # pocket for a 3 x 2 mm neodymium magnet (model mm)
M2 = dict(pilot=1.6, clear=2.3, head=4.0, head_h=1.6)     # M2 x 10 self-tapping
M3 = dict(pilot=2.5, clear=3.3, head=5.8, head_h=2.0)     # M3 x 10 self-tapping
LED_W, LED_H = 4.0, 3.5  # LED channel width / height (model mm)
WIRE_D = 2.0             # wire channel diameter (model mm)

# ---------------------------------------------------------------------
#  PACKAGE HARD POINTS (full size).  LHD: driver on the left (-y).
# ---------------------------------------------------------------------
Z_UNDER = GC                         # underbody / print bed of the chassis
Z_PACK_TOP = 380.0                   # pack frame top = cabin-floor underside
Z_FLOOR = 404.0                      # carpet top (cabin floor 24 mm on the pack)
HP_F = (2390.0, 390.0, 739.0)        # front H-point (x, |y|, z): H30 = 335, L53 = 830
AHP = (1560.0, 390.0, Z_FLOOR)       # accelerator heel point
HP_R = (3260.0, 370.0, 754.0)        # rear outboard H-point, couple distance 870
BACK_F, BACK_R = 25.0, 27.0          # torso angles from vertical
EYE = (2620.0, 390.0, 1399.0)        # driver eye point (eyellipse centroid)
SW_C = (1872.0, 390.0, 1104.0)       # steering-wheel centre, rim 370 mm, 23 deg from vertical
SW_R, SW_TILT = 185.0, 23.0

# battery pack: 112s2p, 224 prismatic NMC cells 173 x 45 x 115 mm, 120 Ah, 3.67 V
CELL = dict(l=173.0, t=45.0, h=115.0, ah=120.0, v=3.67, kg=2.0)
PACK_X0, PACK_X1, PACK_HW = 1420.0, 3340.0, 805.0
FRAME_W = 70.0                       # extruded side frame
MOD_CELLS = 14                       # 7s2p module, 654 x 185 x 128 mm
MOD_Y0 = 70.0                        # inboard end of the modules (centre tunnel |y| < 60)
MOD_LEN = MOD_CELLS*CELL["t"] + 2*12.0
MOD_W = CELL["l"] + 12.0
Z_COLD = (234.0, 246.0)              # cold plate
Z_CELL = (246.0, 361.0)              # module base to cell tops
Z_MODTOP = 374.0                     # busbars and CMU boards
BJB = (1462.0, 1640.0, -280.0, 280.0)        # battery junction box (x0, x1, y0, y1)
BMU = (1462.0, 1640.0, 320.0, 700.0)         # BMS master unit

def module_rows():
    """x start of the 8 module rows: front zone (BJB/BMS), 3 cross-members between pairs."""
    xs, x = [], PACK_X0 + 32.0 + 200.0
    for i in range(8):
        xs.append(x)
        x += MOD_W + (5.0 if i % 2 == 0 else 44.0)
    return xs
ROWS = module_rows()
XMEMBERS = [ROWS[i] + MOD_W + 4.0 for i in (1, 3, 5)]      # x0 of each 36 mm cross-member

# front and rear structure
RAIL_Y = (440.0, 520.0)
FRAIL_Z, RRAIL_Z = (430.0, 580.0), (380.0, 530.0)
SUB_Z = (234.0, 320.0)               # subframes sit on the undertray
FDU = dict(motor=(670.0, 450.0, 115.0, -250.0, 100.0),      # (x, z, r, y0, y1)
           gear=(100.0, 215.0), diff_r=118.0, inv=(560.0, 870.0, -260.0, 215.0, 590.0, 685.0))
RDU = dict(motor=(4065.0, 455.0, 125.0, -260.0, 100.0),
           gear=(100.0, 220.0), diff_r=122.0, inv=(3895.0, 4255.0, -265.0, 220.0, 600.0, 700.0))
KNUCKLE_Y = (628.0, 688.0)           # uprights sit just inboard of the solid wheel
BJ_Y = 655.0                         # lower ball joint
Z_CARGO = (740.0, 764.0)             # removable cargo deck

def mm(v):                            # full size -> model mm
    return v/SCALE

# ---------------------------------------------------------------------
#  PRIMITIVES (full-size mm).  Everything that would print as an overhang is built
#  either with a 48-deg chamfered underside, a teardrop section, or rests on a support.
# ---------------------------------------------------------------------
def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def cbox(x0, x1, y0, y1, z0, z1, cb=0.0, ct=0.0, cs=0.0, cbx=None, cby=None):
    """Box with a printable chamfered underside (horizontal inset cb at 48 deg), a 45-deg top
    chamfer ct and vertical-edge chamfers cs.  cbx / cby restrict the bottom chamfer to the
    x or y faces (tuples (lo, hi) of insets)."""
    bx = cbx if cbx is not None else (cb, cb)
    by = cby if cby is not None else (cb, cb)
    hb = max(max(bx), max(by))*T48
    pts = []
    def ring(z, ix0, ix1, iy0, iy1):
        c = cs
        for (x, y) in ((x0 + ix0, y0 + iy0 + c), (x0 + ix0 + c, y0 + iy0), (x1 - ix1 - c, y0 + iy0), (x1 - ix1, y0 + iy0 + c),
                       (x1 - ix1, y1 - iy1 - c), (x1 - ix1 - c, y1 - iy1), (x0 + ix0 + c, y1 - iy1), (x0 + ix0, y1 - iy1 - c)):
            pts.append((x, y, z))
    ring(z0, bx[0], bx[1], by[0], by[1])
    ring(z0 + hb, 0, 0, 0, 0)
    ring(z1 - ct, 0, 0, 0, 0)
    ring(z1, ct, ct, ct, ct)
    return Manifold.hull_points(np.array(pts))

def _basis(d):
    d = np.asarray(d, float); w = d/np.linalg.norm(d)
    a = np.array([0, 0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, w); u /= np.linalg.norm(u); v = np.cross(w, u)
    return u, v, w

def cyl(p0, p1, r, r1=None, seg=40):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    u, v, w = _basis(p1 - p0)
    T = np.c_[u, v, w, p0]
    return Manifold.cylinder(L, r, r if r1 is None else r1, seg).transform(T)

def section_pts(c, d, r, tear=True, n=20, up=False):
    """Points of a tube section centred at c, normal to direction d.  Shallow runs get a
    teardrop whose tip points down at 48 deg so the underside prints without support
    (up=True points the tip up: the roof of a tunnel cut into a part)."""
    d = np.asarray(d, float); d = d/np.linalg.norm(d)
    vert = np.array([0, 0, 1.0 if up else -1.0])
    down = vert - d*(vert @ d)
    if not tear or np.linalg.norm(down) < 0.35 or abs(d[2]) > math.sin(math.radians(50)):
        u, v, _ = _basis(d)
        a = np.linspace(0, 2*math.pi, n, endpoint=False)
        return [c + r*(math.cos(t)*u + math.sin(t)*v) for t in a]
    down /= np.linalg.norm(down); side = np.cross(d, down)
    b = math.radians(48)
    a = np.linspace(b - math.pi/2, 1.5*math.pi - b, n)           # arc over the top
    pts = [c + r*(math.cos(t)*side - math.sin(t)*down) for t in a]
    pts.append(c + down*r/math.cos(b))
    return pts

def tube(path, r, tear=True, n=20, up=False):
    """Swept tube along a polyline: convex hulls of consecutive sections, joints filled."""
    P = [np.asarray(p, float) for p in path]
    parts = []
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        if np.linalg.norm(d) < 1e-6:
            continue
        u = d/np.linalg.norm(d)
        a = P[i] - (u*0.3*r if i > 0 else 0)                  # overlap the joints: no shared faces
        b = P[i + 1] + (u*0.3*r if i + 1 < len(P) - 1 else 0)
        parts.append(Manifold.hull_points(np.array(section_pts(a, d, r, tear, n, up) + section_pts(b, d, r, tear, n, up))))
        if 0 < i:
            d0 = P[i] - P[i - 1]
            parts.append(Manifold.hull_points(np.array(section_pts(P[i], d0, r, tear, n, up) + section_pts(P[i], d, r, tear, n, up))))
    return Manifold.batch_boolean(parts, OpType.Add)

def tunnel(path, r, n=20):
    """Cutter for an internal channel (wires, LEDs): teardrop roof pointing up."""
    return tube(path, r, True, n, up=True)

def hcyl_y(x, z, r, y0, y1, seg=48, tear=True):
    """Cylinder along y (motors, rack, bars): teardrop keel underneath for printing."""
    pts = section_pts(np.array([x, y0, z]), (0, 1, 0), r, tear, seg) + section_pts(np.array([x, y1, z]), (0, 1, 0), r, tear, seg)
    return Manifold.hull_points(np.array(pts))

def prism(geom, plane, w0, w1):
    return v4.prism(geom, plane, w0, w1)

def cs_poly(pts):
    return CrossSection([np.asarray(pts, float)], FillRule.Positive)

def extrude_xz(pts, y0, y1):
    """Extrude a polygon given in (x, z) along y."""
    return prism(Polygon(pts).buffer(0), "xz", y0, y1)

def extrude_yz(pts, x0, x1):
    return prism(Polygon(pts).buffer(0), "yz", x0, x1)

def extrude_xy(pts, z0, z1):
    return prism(Polygon(pts).buffer(0), "xy", z0, z1)

def symy(m):
    """Mirror copy about the centre plane, unioned."""
    return Manifold.batch_boolean([m, m.mirror([0, 1, 0])], OpType.Add)

def union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return Manifold.batch_boolean(ms, OpType.Add) if ms else Manifold()

def diff(a, bs):
    bs = [b for b in bs if b is not None and not b.is_empty()]
    return Manifold.batch_boolean([a] + bs, OpType.Subtract) if bs else a

def teardrop_hole(p, axis, d, depth, inward=True):
    """Horizontal blind hole (magnets, pins) with a 45-deg roof so it prints without
    support.  p is the hole mouth centre, axis the unit direction into the material."""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    r = d/2
    if abs(axis[2]) > 0.9:                        # vertical: plain cylinder (+ cone roof if opening down)
        h = cyl(p, np.asarray(p) + axis*depth, r, seg=32)
        if axis[2] > 0:
            tip = np.asarray(p) + axis*depth
            h = union([h, cyl(tip - axis*0.01, tip + axis*r, r, 0.0, seg=32)])
        return h
    up = np.array([0, 0, 1.0]); side = np.cross(axis, up)
    pts = []
    for t in np.linspace(0, 2*math.pi, 28, endpoint=False):
        q = r*(math.cos(t)*side + math.sin(t)*up)
        pts += [np.asarray(p) + q - axis*0.02, np.asarray(p) + q + axis*depth]
    for s in (-0.02, depth):
        pts.append(np.asarray(p) + up*r*math.sqrt(2) + axis*s)
    return Manifold.hull_points(np.array(pts))

# colour tags used by the renders and the report
COL = dict(chassis=(0.32, 0.34, 0.37), alu=(0.70, 0.72, 0.75), cell=(0.20, 0.36, 0.62), cellgap=(0.12, 0.14, 0.18),
           hv=(1.00, 0.45, 0.05), lv=(0.10, 0.10, 0.10), coolant=(0.10, 0.65, 0.85), coolant_hot=(0.90, 0.25, 0.20),
           refrigerant=(0.55, 0.85, 0.30), motor=(0.55, 0.58, 0.62), inverter=(0.82, 0.83, 0.85), susp=(0.25, 0.25, 0.27),
           spring=(0.85, 0.15, 0.15), brake=(0.60, 0.60, 0.62), electronics=(0.20, 0.45, 0.30), solar=(0.95, 0.80, 0.10),
           radiator=(0.45, 0.47, 0.50), seat=(0.18, 0.18, 0.20), trim=(0.55, 0.50, 0.45), screen=(0.05, 0.08, 0.12),
           body=(0.92, 0.92, 0.94), glass=(0.20, 0.25, 0.32), tyre=(0.08, 0.08, 0.08), rim=(0.72, 0.73, 0.76))

def revolve_axis(prof, p0, d, seg=48):
    """Revolve an (r, h) profile about the axis through p0 along d (h measured along d)."""
    cs = CrossSection([np.asarray(prof, float)], FillRule.Positive)
    m = Manifold.revolve(cs, seg)
    u, v, w = _basis(d)
    return m.transform(np.c_[u, v, w, np.asarray(p0, float)])

def spring_sleeve(p0, p1, ro, ri, coils, wire):
    """Coil spring as a printable corrugated sleeve: every downward flank is >= 48 deg."""
    L = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
    pitch = L/coils
    tf = math.tan(math.radians(62))
    d = min(wire*0.9, (pitch*0.5)/tf)
    prof = [(ri, 0.0)]
    for k in range(coils):
        z = k*pitch
        prof += [(ro - d, z), (ro, z + d*tf), (ro, z + pitch*0.5), (ro - d, z + pitch*0.5 + d)]
    prof += [(ro - d, L), (ri, L)]
    return revolve_axis(prof, p0, np.asarray(p1, float) - np.asarray(p0, float))

def spring_seat(p, r_in, r_out):
    """Lower spring seat: a 50-deg cone from the damper body up to the spring's outer radius."""
    h = (r_out - r_in)*math.tan(math.radians(50))
    return revolve_axis([(0, -h - 2), (r_in, -h - 2), (r_out, 0), (r_out, 4), (0, 4)], p, (0, 0, 1), 40)

def bellows(p0, p1, r0, r1, folds=5, depth=8.0):
    """Rubber boot (rack, CV) as a printable teardrop frustum with fold rings on top."""
    return Manifold.hull_points(np.array(section_pts(np.asarray(p0, float), np.asarray(p1, float) - np.asarray(p0, float), r0)
                                         + section_pts(np.asarray(p1, float), np.asarray(p1, float) - np.asarray(p0, float), r1)))

# =====================================================================
#  CHASSIS (full-size mm).  chassis_parts() -> {tag: [manifold, ...]}
# =====================================================================
def _add(P, tag, *ms):
    for m in ms:
        if m is not None and not m.is_empty():
            P.setdefault(tag, []).append(m)

def pack_parts(P):
    zt = Z_PACK_TOP
    yi = PACK_HW - FRAME_W
    # tray: side frames (extrusions with a grooved outer face), end walls, cold plate
    side = box(PACK_X0, PACK_X1, yi, PACK_HW, 234, zt)
    for z in (275, 320):
        side = side - box(PACK_X0 + 40, PACK_X1 - 40, PACK_HW - 6, PACK_HW + 1, z, z + 10)
    _add(P, "alu", symy(side), box(PACK_X0, PACK_X0 + 32, -PACK_HW, PACK_HW, 234, zt),
         box(PACK_X1 - 32, PACK_X1, -PACK_HW, PACK_HW, 234, zt))
    _add(P, "alu", box(PACK_X0 + 32, PACK_X1 - 32, -yi, yi, 234, Z_COLD[1]))
    # cross-members (seat-rail mounts) with a notch for the centre duct
    for x0 in XMEMBERS:
        _add(P, "alu", symy(box(x0, x0 + 36, 60, yi, Z_COLD[1], zt)))
    # centre duct: coolant supply / return inside, HV busbars on top
    _add(P, "alu", box(BJB[1], ROWS[-1] + MOD_W, -60, 60, Z_COLD[1], 330))
    _add(P, "hv", symy(box(BJB[1] - 10, ROWS[-1] + MOD_W, 14, 40, 330, 338)))
    # modules
    yc = [MOD_Y0 + 12 + CELL["t"]*(k + 0.5) for k in range(MOD_CELLS)]
    for r, x0 in enumerate(ROWS):
        x1 = x0 + MOD_W
        y0, y1 = MOD_Y0, MOD_Y0 + MOD_LEN
        blk = box(x0, x1, y0 + 12, y1 - 12, Z_CELL[0], Z_CELL[1])
        gaps = [box(x0 - 1, x1 + 1, MOD_Y0 + 12 + CELL["t"]*k - M(0.3), MOD_Y0 + 12 + CELL["t"]*k + M(0.3),
                    Z_CELL[1] - M(1.1), Z_CELL[1] + 1) for k in range(1, MOD_CELLS)]
        cells = diff(blk, gaps)
        ends = union([box(x0, x1, y0, y0 + 12, Z_CELL[0], Z_CELL[1] + 8), box(x0, x1, y1 - 12, y1, Z_CELL[0], Z_CELL[1] + 8)])
        straps = union([box(x0 - 3, x0 + 2, y0, y1, 280, 300), box(x1 - 2, x1 + 3, y0, y1, 280, 300)])
        term = []
        for k, y in enumerate(yc):
            for xt in (x0 + 38, x1 - 38):
                term.append(cyl((xt, y, Z_CELL[1] - 1), (xt, y, Z_CELL[1] + 6), 12, seg=16))
        bars = []
        for g in range(MOD_CELLS//2 - 1):                       # 7s2p: busbar joins group g to g+1
            ya, yb = yc[2*g] - 18, yc[2*g + 3] + 18
            xt = x0 + 38 if g % 2 == 0 else x1 - 38
            bars.append(box(xt - 18, xt + 18, ya, yb, Z_CELL[1] + 6, Z_MODTOP))
        for g, xt in ((0, x1 - 38), (MOD_CELLS//2 - 1, x0 + 38)):
            bars.append(box(xt - 18, xt + 18, yc[2*g] - 18, yc[2*g + 1] + 18, Z_CELL[1] + 6, Z_MODTOP))
        cmu = box(x0 + 66, x1 - 66, (y0 + y1)/2 - 110, (y0 + y1)/2 + 110, Z_CELL[1], Z_CELL[1] + 4)
        link = box(x0 - 6 if r % 2 else x0 + 20, x1 + 6 if r % 2 == 0 else x1 - 20, y0 + 2, y0 + 30, Z_CELL[1] + 8, Z_MODTOP)
        for s in (1, -1):
            f = (lambda m: m) if s > 0 else (lambda m: m.mirror([0, 1, 0]))
            _add(P, "cell", f(cells))
            _add(P, "alu", f(ends), f(straps), f(union(bars)), f(union(term)))
            _add(P, "electronics", f(cmu))
            _add(P, "hv", f(link))
    # BMS daisy chain over the CMU boards, into the BMU
    for s in (1, -1):
        ym = s*(MOD_Y0 + MOD_LEN/2)
        _add(P, "lv", tube([(BMU[0] + 60, s*480, 330), (BMU[1] + 30, s*480, Z_CELL[1] + 11), (ROWS[-1] + MOD_W - 40, s*480, Z_CELL[1] + 11)], 7, n=12))
    # battery junction box (contactors, pyro fuse, current sensor) and BMS master unit
    bjb = cbox(BJB[0], BJB[1], BJB[2], BJB[3], Z_COLD[1], 372, ct=10)
    ribs = union([box(BJB[0] + 20, BJB[1] - 20, y - 6, y + 6, 372, 378) for y in (-180, -60, 60, 180)])
    _add(P, "inverter", bjb, ribs)
    _add(P, "hv", symy(box(BJB[1] - 4, BJB[1] + 14, 14, 40, 300, 338)))
    bmu = cbox(BMU[0], BMU[1], BMU[2], BMU[3], Z_COLD[1], 335, ct=8)
    fins = union([box(BMU[0] + 15, BMU[1] - 15, y, y + 8, 335, 345) for y in np.arange(BMU[2] + 30, BMU[3] - 20, 40)])
    _add(P, "electronics", bmu, fins)
    # external HV / coolant connectors on the front and rear walls (grounded on the undertray)
    for y in HV_PORTS_F + HV_PORTS_C:
        _add(P, "hv", cbox(PACK_X0 - 26, PACK_X0 + 2, y - 22, y + 22, 234, 330, ct=6))
    for y in HV_PORTS_R:
        _add(P, "hv", cbox(PACK_X1 - 2, PACK_X1 + 26, y - 22, y + 22, 234, 330, ct=6))
    for y, tag in ((COOL_F[0], "coolant"), (COOL_F[1], "coolant_hot")):
        _add(P, tag, cbox(PACK_X0 - 30, PACK_X0 + 2, y - 20, y + 20, 234, 300, ct=6))
    for y, tag in ((COOL_R[0], "coolant"), (COOL_R[1], "coolant_hot")):
        _add(P, tag, cbox(PACK_X1 - 2, PACK_X1 + 30, y - 20, y + 20, 234, 300, ct=6))

HV_PORTS_F = (-20.0, 40.0)           # front drive unit DC+/DC-
HV_PORTS_C = (120.0, 180.0, 240.0)   # charging stack (OBC/MPPT/DCDC) +/-, DC fast charge
HV_PORTS_R = (-30.0, 30.0)           # rear drive unit
COOL_F = (-290.0, -230.0)            # pack coolant in / out (front wall)
COOL_R = (-150.0, 150.0)             # rear drive-unit loop through the centre duct

def crash_parts(P):
    """Bumper beams, crush cans, rails (front kick-down onto the pack, rear kick-up), subframes."""
    # ---- front
    ys = np.linspace(-780, 780, 41)
    xb = lambda y: 150.0 + 0.9*v4.bow_front(y)
    beam = extrude_xy([(xb(y), y) for y in ys] + [(xb(y) + 62, y) for y in ys[::-1]], 440, 560)
    beam = beam - extrude_xy([(xb(y) - 1, y) for y in ys[2:-2]] + [(xb(y) + 14, y) for y in ys[2:-2][::-1]], 470, 530)
    _add(P, "alu", beam)
    _add(P, "chassis", box(xb(0) + 4, xb(0) + 58, -30, 30, 200, 441))   # tow-eye bracket / lower load path
    for y0, y1 in ((190.0, 250.0), (660.0, 780.0)):           # posts follow the beam's plan bow out to its ends
        yy = np.linspace(y0, y1, 7)
        _add(P, "chassis", symy(extrude_xy([(xb(y) + 4, y) for y in yy] + [(xb(y) + 58, y) for y in yy[::-1]], 200, 441)))
    _add(P, "alu", symy(cbox(212, 332, RAIL_Y[0], RAIL_Y[1], 440, 560, cs=14)))
    _add(P, "chassis", symy(box(232, 342, RAIL_Y[0], RAIL_Y[1], 200, 441)))     # lower load path under cans + rail tip
    rail = union([box(330, 1152, RAIL_Y[0], RAIL_Y[1], FRAIL_Z[0], FRAIL_Z[1]),
                  extrude_xz([(1150, FRAIL_Z[0]), (1150, FRAIL_Z[1]), (1302, Z_PACK_TOP), (1420, Z_PACK_TOP), (1420, 234), (1302, 234)],
                             RAIL_Y[0], RAIL_Y[1])])
    _add(P, "chassis", symy(rail))
    for x0, x1 in ((470.0, 530.0), (700.0, 760.0), (830.0, 950.0), (1060.0, 1120.0), (1210.0, 1270.0)):
        _add(P, "chassis", symy(box(x0, x1, 340, RAIL_Y[1], 234, FRAIL_Z[0] + 2)))   # body-mount towers (<= 9 mm spans); 830-950 carries the axle bore
    # front subframe (cradle) on the undertray, rack crossmember with harness notches
    sub = union([symy(box(450, 1250, 300, 380, SUB_Z[0], SUB_Z[1])), box(450, 520, -380, 380, SUB_Z[0], SUB_Z[1]),
                 box(1040, 1250, -420, 420, SUB_Z[0], SUB_Z[1])])
    notch = [box(1030, 1262, y - 20, y + 20, 230, 268) for y in HV_PORTS_F + HV_PORTS_C[:2] + COOL_F + REFRIG_Y + (SOLAR_Y,)]
    _add(P, "chassis", diff(sub, notch))
    # ---- rear
    _add(P, "chassis", symy(union([box(PACK_X1 - 4, 4382, 300, 380, 320, Z_PACK_TOP),
                                   extrude_xz([(4380, 320), (4380, Z_PACK_TOP), (4582, 682), (4582, 560)], 300, 380)])))
    rs = union([symy(box(3480, 4390, 300, 380, SUB_Z[0], SUB_Z[1])), box(3480, 3560, -380, 380, SUB_Z[0], SUB_Z[1]),
                box(4220, 4300, -380, 380, SUB_Z[0], SUB_Z[1])])
    notch = [box(3470, 3570, y - 20, y + 20, 230, 268) for y in HV_PORTS_R + COOL_R]
    _add(P, "chassis", diff(rs, notch))
    _add(P, "alu", symy(cbox(4580, 4662, 300, 380, 560, 682, cs=14)))
    xr = lambda y: 4662.0 - (v4.bow_rear(y) - v4.bow_rear(0))
    yr = np.linspace(-700, 700, 37)
    rbeam = extrude_xy([(xr(y), y) for y in yr] + [(xr(y) + 60, y) for y in yr[::-1]], 560, 680)
    _add(P, "alu", rbeam)
    for y in (0.0, 220.0, 340.0, 560.0, 660.0):              # struts: the skirt ends ahead of the rear chamfer
        st = tube([(4480, y, 330), (xr(y) + 25, y, 566)], 28, tear=False)
        if y != 560.0:                                       # +-560 start on the cargo-deck posts
            st = union([st, box(4450, 4510, y - 30, y + 30, 234, 345)])
        _add(P, "chassis", symy(st) if y else st)
    # cargo-deck posts
    for x, y in CARGO_POSTS:
        _add(P, "chassis", symy(cbox(x - 30, x + 30, y - 30, y + 30, 234, Z_CARGO[0], cs=8)))

TOP_MOUNT = [(0, 0), (36, 0), (68, 42), (68, 56), (0, 56)]      # strut top mount, 53-deg underside
CARGO_POSTS = ((3700.0, 600.0), (4500.0, 560.0))
REFRIG_Y = (-130.0, -80.0)           # refrigerant suction / liquid lines to the cabin HVAC module
SOLAR_Y = 230.0                       # solar + charging bundle riser

def drive_unit(P, front=True):
    """Parallel-axis e-axle: motor + 2-stage reduction + open differential, inverter on top,
    3-phase busbar loop to the stator terminal box, axle bore along the wheel axis."""
    d = FDU if front else RDU
    xm, zm, rm, ya, yb = d["motor"]
    xa = X_FA if front else X_RA
    sgn = 1 if front else -1                       # motor ahead of (front) or behind (rear) the axle
    gy0, gy1 = d["gear"]
    motor = hcyl_y(xm, zm, rm, ya, yb)
    bolt = revolve_axis([(rm - 34, -1), (rm - 22, -1), (rm - 22, 8), (rm - 34, 8)], (xm, ya, zm), (0, 1, 0))
    motor = motor - bolt                                              # end-bell joint, engraved
    ribs = union([hcyl_y(xm, zm, rm + 6, y, y + 12) for y in np.linspace(ya + 40, yb - 30, 6)])
    cradle = box(xm - 40, xm + 40, ya, yb, 234, zm - rm + 40)          # mount rail under the keel
    _add(P, "motor", motor, ribs, cradle)
    xi = 0.5*(xm + xa)                             # idler shaft between motor and diff
    zi = 0.5*(zm + HUB_Z) + 75
    rd = d["diff_r"]
    lo, hi = min(xm - rm, xa - rd), max(xm + rm, xa + rd)
    gear_xz = unary_union([Point(xm, zm).buffer(rm, 32), Point(xi, zi).buffer(75, 32), Point(xa, HUB_Z).buffer(rd, 32),
                           sbox(lo + 12, 234, hi - 12, HUB_Z)]).convex_hull
    gear = prism(gear_xz, "xz", gy0, gy1)
    bell = union([hcyl_y(xa, HUB_Z, rd - 25, -80, gy0 + 1), box(xa - rd + 30, xa + rd - 30, -150, gy0, 234, HUB_Z)])
    cv = union([hcyl_y(xa, HUB_Z, 52, -150, -79), hcyl_y(xa, HUB_Z, 52, gy1 - 1, gy1 + 70),
                box(xa - 45, xa + 45, gy1 - 1, gy1 + 70, 234, HUB_Z)])
    _add(P, "motor", gear, bell, cv)
    # inverter on top of the motor (hull onto the motor's upper half so the underside prints)
    ix0, ix1, iy0, iy1, iz0, iz1 = d["inv"]
    ix0, ix1 = (xm - rm + 20, xm + rm + 15) if front else (xm - rm - 15, xm + rm - 20)
    iy0 = ya
    inv = cbox(ix0, ix1, iy0, iy1, iz0, iz1, ct=12)
    up = [(xm + rm*math.cos(t), y, zm + rm*math.sin(t)) for t in np.linspace(0, math.pi, 13) for y in (iy0, yb, iy1)]
    up += [(x, y, iz0 + 1) for x in (ix0, ix1) for y in (iy0, iy1)]
    _add(P, "inverter", inv, Manifold.hull_points(np.array(up)))
    _add(P, "inverter", union([box(ix0 + 20, ix1 - 20, y, y + 10, iz1, iz1 + 8) for y in np.arange(iy0 + 30, iy1 - 20, 45)]))
    # stator terminal box on the motor shoulder, 3-phase loop from the inverter end face
    xt = xm - sgn*(rm - 30)
    tbox = cbox(min(xt, xt - sgn*60), max(xt, xt - sgn*60), ya + 30, ya + 130, zm + 30, zm + 118, ct=6,
                cbx=(48, 0) if front else (0, 48), cby=(0, 0))
    _add(P, "inverter", tbox)
    xf = ix0 if front else ix1                     # inverter end face over the terminal box
    xo = xf - sgn*32
    for k, y in enumerate((ya + 50, ya + 80, ya + 110)):
        _add(P, "hv", tube([(xf + sgn*5, y, iz0 + 45), (xo, y, iz0 + 45), (xo, y, zm + 90), (xt - sgn*30, y, zm + 90)], 11, n=14))
    # DC input connector on the inverter face toward the axle
    xc = ix1 if front else ix0
    for y in (HV_PORTS_F if front else HV_PORTS_R):
        _add(P, "hv", cbox(min(xc - sgn*30, xc + sgn*24), max(xc - sgn*30, xc + sgn*24), y - 20, y + 20, iz0 - 10, iz0 + 70, ct=5,
                           cbx=(0, 54) if front else (54, 0), cby=(0, 0)))
    # coolant stubs on the inverter
    for y, tag in ((iy0 + 40, "coolant"), (iy0 + 90, "coolant_hot")):
        _add(P, tag, cyl((xm, y, iz1 - 5), (xm, y, iz1 + 30), 14, seg=20))
    return xa

def axle_bores():
    """Clearance bores for the 3 mm axles (knuckles + drive units), teardrop roofs."""
    r = M(v4.AXLE_D/2 + v4.AXLE_CLEAR)
    out = []
    for xa in (X_FA, X_RA):
        pts = section_pts(np.array([xa, -720.0, HUB_Z]), (0, 1, 0), r, True, 24) + section_pts(np.array([xa, 720.0, HUB_Z]), (0, 1, 0), r, True, 24)
        out.append(Manifold.hull_points(np.array(pts)))
    return union(out)

def knuckle(P, xa, front=True):
    """Upright just inboard of the solid wheel: lower block on the undertray (ball joint), web,
    hub boss with outboard CV, steering / toe arm plate, strut or upper-link clamp."""
    y0, y1 = KNUCKLE_Y
    xbj = xa - 20
    parts = [box(xbj - 50, xa + 50, 600, y1, 234, 330), cbox(xa - 42, xa + 42, y0, y1, 300, 610, ct=10),
             hcyl_y(xa, HUB_Z, 70, y0 - 28, y1), cbox(xa - 52, xa + 52, 600, y0 + 2, 234, HUB_Z + 10),
             Manifold.sphere(32, 24).translate([xbj, BJ_Y, 262])]
    xarm = xa + 140 if front else xa + 120
    parts.append(box(xa + 30, xarm + 10, 638, 672, 234, 375))                  # steering (front) / toe (rear) arm
    parts.append(cyl((xa + 5, 652, 560), (xa + 5, 652, 650), 33, seg=28))      # strut / upper-arm clamp
    _add(P, "susp", symy(union(parts)))
    return xarm

def suspension_parts(P):
    # ---------------- FRONT: MacPherson strut, A-arm, rack-and-pinion EPS, sway bar
    xarm = knuckle(P, X_FA, True)
    xbj = X_FA - 20
    piv = [(800.0, 390.0), (1150.0, 390.0)]
    arm = unary_union([LineString([p, (xbj, BJ_Y)]).buffer(26) for p in piv])
    _add(P, "susp", symy(prism(arm, "xy", 234, 284)))
    for x, y in piv:
        _add(P, "susp", symy(tube([(x - 40, y, 268), (x + 40, y, 268)], 30, n=18)))
    # strut: damper, printable corrugated spring, top mount
    p0, p1 = np.array([X_FA + 5, 652.0, 600.0]), np.array([X_FA + 35, 590.0, 905.0])     # KPI 11.5 deg, castor 5.6 deg
    _add(P, "susp", symy(cyl(p0, p1, 28, seg=28)))
    s0 = p0 + (p1 - p0)*0.22; s1 = p0 + (p1 - p0)*0.85
    _add(P, "spring", symy(spring_sleeve(s0, s1, 52, 26, 6, 13)))
    _add(P, "susp", symy(spring_seat(s0, 28, 54)))
    _add(P, "susp", symy(revolve_axis(TOP_MOUNT, p1 - (0, 0, 30), (0, 0, 1))))
    # rack-and-pinion with rack-parallel EPS motor, boots, tie rods
    _add(P, "susp", tube([(1080, -400, 352), (1080, 400, 352)], 28, n=24))
    _add(P, "susp", union([box(1050, 1110, y - 22, y + 22, 318, 392) for y in (-250.0, 250.0)]))
    _add(P, "susp", symy(bellows((1080, 330, 352), (1080, 405, 352), 34, 22)))
    _add(P, "motor", tube([(1142, 110, 366), (1142, 330, 366)], 44, n=24), box(1078, 1190, 300, 345, 318, 405))
    _add(P, "susp", cyl((1080, -300, 352), (1150, -310, 440), 34, seg=24), tube([(1150, -310, 440), (1295, -340, 610)], 14, n=14),
         cbox(1270, 1320, -365, -315, 585, 640, ct=8))
    _add(P, "susp", symy(tube([(1080, 405, 354), (xarm, 655, 354)], 12, n=14)), symy(Manifold.sphere(18, 16).translate([xarm, 655, 354])))
    # sway bar on the rack crossmember, ends on the lower arms
    _add(P, "susp", tube([(1048, -495, 300), (1205, -380, 334), (1205, 380, 334), (1048, 495, 300)], 13, n=14))
    # ---------------- REAR: 5-link (wishbone + camber + toe), coilover, sway bar
    xarm = knuckle(P, X_RA, False)
    xbj = X_RA - 20
    plate = unary_union([Point(3620, 390).buffer(35), Point(4000, 390).buffer(35), Point(xbj, BJ_Y).buffer(32),
                         Point(3960, 560).buffer(72)]).convex_hull
    _add(P, "susp", symy(prism(plate, "xy", 234, 284)))
    for x in (3620.0, 4000.0):
        _add(P, "susp", symy(tube([(x - 40, 390, 268), (x + 40, 390, 268)], 30, n=18)))
    _add(P, "chassis", symy(box(3730, 3790, 300, 432, 234, 655)))                       # camber-link tower
    _add(P, "susp", symy(tube([(3760, 432, 626), (X_RA + 5, 640, 626)], 17, n=14)))
    _add(P, "chassis", symy(box(3975, 4025, 300, 440, 234, 380)))                       # toe-link bracket
    _add(P, "susp", symy(tube([(4000, 440, 362), (xarm, 655, 362)], 12, n=14)))
    q0, q1 = np.array([3960.0, 560.0, 284.0]), np.array([3966.0, 552.0, 780.0])
    _add(P, "susp", symy(cyl(q0, q1, 26, seg=28)))
    _add(P, "spring", symy(spring_sleeve(q0 + (q1 - q0)*0.2, q0 + (q1 - q0)*0.82, 54, 26, 6, 13)))
    _add(P, "susp", symy(spring_seat(q0 + (q1 - q0)*0.2, 26, 56)))
    _add(P, "susp", symy(revolve_axis(TOP_MOUNT, q1 - (0, 0, 30), (0, 0, 1))))
    _add(P, "susp", tube([(3992, -545, 300), (4050, -340, 300), (4262, -340, 300), (4262, 340, 300), (4050, 340, 300), (3992, 545, 300)], 12, n=14))

def thermal_parts(P):
    """Front-end cooling module, heat-pump thermal module (left carrier) and the
    charging stack (right carrier) above the front inverter."""
    # radiator + condenser down to the undertray (lower tank = air dam), fan shroud with 2 fans
    _add(P, "radiator", box(300, 340, -400, 400, 234, 760), box(278, 300, -380, 380, 234, 740))
    shroud = box(340, 398, -420, 420, 234, 760)
    for y in (-205.0, 205.0):
        c = np.array([398.0, y, 560.0])
        shroud = shroud - revolve_axis([(150, -1), (158, -1), (158, 12), (150, 12)], c - (12, 0, 0), (1, 0, 0), 64)
        shroud = shroud - cyl(c - (10, 0, 0), c + (1, 0, 0), 20, seg=24)                # fan hub
        for k in range(7):                                          # fan blades, engraved
            a = math.radians(k*360/7)
            u = np.array([0, math.cos(a), math.sin(a)]); v = np.array([0, -math.sin(a), math.cos(a)])
            pts = [c + 30*u + 8*v, c + 140*u + 30*v, c + 140*u - 10*v, c + 30*u - 8*v]
            blade = Manifold.hull_points(np.array([p - (s, 0, 0) for p in pts for s in (-1, 8)]))
            shroud = shroud - blade
    _add(P, "radiator", shroud)
    # carriers on the inverter, outer edges on posts standing on the rails
    for s in (1, -1):
        _add(P, "chassis", box(575, 800, min(s*20, s*440), max(s*20, s*440), 685, 700),
             box(640, 760, min(s*436, s*480), max(s*436, s*480), FRAIL_Z[1] - 2, 700),
             box(640, 760, min(s*300, s*380), max(s*300, s*380), SUB_Z[1] - 2, 690))       # posts on the subframe
    # LEFT: heat-pump thermal module
    _add(P, "motor", tube([(640, -430, 795), (640, -290, 795)], 62, n=28), cbox(600, 690, -290, -250, 700, 850, ct=8))  # e-compressor
    _add(P, "electronics", cbox(610, 680, -400, -320, 852, 870, ct=6))                       # compressor inverter lid
    _add(P, "alu", cyl((755, -370, 699), (755, -370, 900), 40, seg=32), cyl((755, -370, 899), (755, -370, 918), 26, seg=24))  # accumulator
    _add(P, "alu", cbox(712, 800, -320, -205, 700, 805, ct=8))                             # chiller
    _add(P, "alu", cbox(712, 800, -195, -95, 700, 790, ct=8))                              # water-cooled condenser
    _add(P, "inverter", cbox(588, 700, -240, -110, 700, 885, ct=10))                       # coolant reservoir
    _add(P, "motor", cbox(600, 700, -100, -25, 700, 760, ct=6),                            # 8-way valve hub
         cyl((650, -62, 759), (650, -62, 805), 34, seg=28), cbox(625, 675, -85, -40, 804, 835, ct=4))
    for y in (-90.0, -40.0):                                                               # coolant pumps
        _add(P, "motor", cyl((752, y, 789), (752, y, 860), 24, seg=24))
    _add(P, "alu", cbox(712, 800, -85, -25, 700, 790, ct=6))
    # RIGHT: charging stack - power conversion unit (11 kW OBC + DC-DC + PDU), MPPT, 12 V LFP
    pcu = cbox(650, 800, 40, 420, 700, 800, ct=10)
    _add(P, "inverter", pcu, union([box(660, 790, y, y + 8, 800, 806) for y in np.arange(60, 410, 40)]))
    mppt = cbox(662, 788, 80, 380, 806, 850, ct=6)
    fins = union([box(670, 780, y, y + 9, 850, 880) for y in np.arange(92, 372, 22)])
    _add(P, "electronics", mppt, fins)
    _add(P, "lv", cbox(582, 642, 120, 400, 700, 832, ct=8), cyl((612, 160, 831), (612, 160, 850), 12), cyl((612, 360, 831), (612, 360, 850), 12))

def harness_parts(P):
    """Physical cable and pipe runs (teardrop sections, supported along their length)."""
    zt = 234.0                                       # undertray top
    r_hv, r_sol, r_cool, r_ref = 13.0, 10.0, 14.0, 9.0
    zc = lambda r: zt + r                            # centre height of a run lying on the undertray
    # 1. battery -> front inverter (DC+ / DC-): forward on the undertray, under the rack, up the
    #    diff housing, over to the inverter connector
    xr_f = X_FA + FDU["diff_r"] + r_hv
    zi = FDU["inv"][4] + 45
    xin = FDU["motor"][0] + FDU["motor"][2] + 15 + 24
    for y in HV_PORTS_F:
        _add(P, "hv", tube([(PACK_X0 - 26, y, 290), (PACK_X0 - 40, y, zc(r_hv)), (xr_f, y, zc(r_hv)), (xr_f, y, zi), (xin - 34, y, zi)], r_hv))
    # 2. battery <-> charging stack (MPPT / OBC / DC-DC), riser against the gearbox
    zp = 750.0
    for y in HV_PORTS_C[:2]:
        _add(P, "hv", tube([(PACK_X0 - 26, y, 290), (PACK_X0 - 40, y, zc(r_hv)), (xr_f, y, zc(r_hv)), (xr_f, y, zp), (800, y, zp)], r_hv))
    # 3. battery -> rear inverter
    xr_r = X_RA - RDU["diff_r"] - r_hv
    zr = RDU["inv"][4] + 45
    xrin = RDU["motor"][0] - RDU["motor"][2] - 15 - 24
    for y in HV_PORTS_R:
        _add(P, "hv", tube([(PACK_X1 + 26, y, 290), (PACK_X1 + 40, y, zc(r_hv)), (xr_r, y, zc(r_hv)), (xr_r, y, zr), (xrin + 34, y, zr)], r_hv))
    # 4. charge inlet (right front fender, on the front clip) -> rocker-front connector ->
    #    DC fast charge to the BJB port, AC branch to the OBC
    xk = PACK_X0 - 56                                # cross-car harness lane in front of the pack
    yk = INLET_Y
    _add(P, "hv", cbox(xk - 26, xk + 26, yk - 30, yk + 30, 234, Z_SKIRT, ct=6))           # harness connector
    _add(P, "hv", tube([(xk, yk - 30, zc(r_hv)), (xk, HV_PORTS_C[2] + 30, zc(r_hv)), (PACK_X0 - 26, HV_PORTS_C[2], zc(r_hv))], r_hv))
    _add(P, "hv", tube([(xk, yk - 30, zc(r_hv) + 2*r_hv), (xk, SOLAR_Y + 26, zc(r_hv) + 2*r_hv), (xk - 40, SOLAR_Y + 26, zc(r_hv)),
                        (xr_f + 2, SOLAR_Y + 26, zc(r_hv)), (xr_f + 2, SOLAR_Y + 26, zp), (800, SOLAR_Y + 26, zp)], 10))
    # 5. solar strings: roof + tailgate down both A-pillars (canopy / clip wire channels) to the
    #    rocker-front connectors, across to the MPPT; hood string drops in from the cowl
    for s in (1, -1):
        y0 = s*INLET_Y if s < 0 else INLET_Y - 70
        _add(P, "solar", cbox(xk - 26, xk + 26, y0 - 22, y0 + 22, 234, Z_SKIRT, ct=6))
    ys = SOLAR_Y - 8
    _add(P, "solar", tube([(xk - 30, -INLET_Y + 22, zc(r_sol)), (xk - 30, ys, zc(r_sol)), (xr_f + 2, ys, zc(r_sol)), (xr_f + 2, ys, 830),
                           (788, ys, 830)], r_sol))
    _add(P, "solar", tube([(xk + 30, INLET_Y - 92, zc(r_sol)), (xk + 30, ys + 22, zc(r_sol)), (xk - 30, ys + 22, zc(r_sol))], r_sol))
    _add(P, "solar", cyl((725, 230, 879), (725, 230, 930), 10), cbox(705, 745, 210, 250, 925, 940, ct=4))   # hood pigtail
    # 6. 3-phase loops are built with the drive units; compressor feed from the PDU
    _add(P, "hv", tube([(700, 60, 770), (700, -240, 770), (690, -290, 790)], 10))
    # 7. coolant: radiator <-> valve hub, hub -> front inverter, chiller <-> pack, pack duct -> rear unit
    _add(P, "coolant_hot", tube([(340, -150, 760 + r_cool), (600, -150, 760 + r_cool)], r_cool))
    _add(P, "coolant", tube([(340, -60, 760 + r_cool), (630, -60, 760 + r_cool)], r_cool))
    _add(P, "coolant", tube([(600, -230, 720), (575, -230, 720), (575, -230, 640)], r_cool))
    xd = 1000.0
    for y, tag in ((COOL_F[0], "coolant"), (COOL_F[1], "coolant_hot")):
        _add(P, tag, tube([(800, y, 760), (xd, y, 760), (xd, y, zc(r_cool)), (PACK_X0 - 30, y, zc(r_cool))], r_cool))
    xdr = xr_r - 2*r_hv - 6
    xri = RDU["motor"][0] - RDU["motor"][2] - 15 + 12            # into the inverter's front face
    for y, tag in ((COOL_R[0], "coolant"), (COOL_R[1], "coolant_hot")):
        _add(P, tag, tube([(PACK_X1 + 30, y, zc(r_cool)), (xdr, y, zc(r_cool)), (xdr, y, zr - 20), (xri, y, zr - 20)], r_cool))
        if y < 0:                                                       # pipe brackets: diff bell (left), gearbox (right)
            _add(P, "chassis", box(3785, 3815, y, y + 24, 395.0, zr - 20))
        else:
            _add(P, "chassis", box(3785, 3815, y - 12, y + 12, 500.0, zr - 20))
    # 8. refrigerant: compressor -> front condenser; condenser -> chiller; lines to the cabin HVAC
    _add(P, "refrigerant", tube([(640, -380, 862), (398, -380, 760 + r_ref), (300, -380, 740 + r_ref)], r_ref))
    _add(P, "refrigerant", tube([(300, -330, 760 + r_ref), (722, -310, 760 + r_ref)], r_ref),
         box(470, 500, -345, -315, SUB_Z[1] - 2, 760 + r_ref))                         # line bracket on the subframe
    for y in REFRIG_Y:
        _add(P, "refrigerant", tube([(800, y, 770), (xd - 30, y, 770), (xd - 30, y, zc(r_ref)), (HVAC_X - 40, y, zc(r_ref)),
                                     (HVAC_X - 40, y, 500)], r_ref))
        _add(P, "refrigerant", cbox(HVAC_X - 62, HVAC_X - 18, y - 20, y + 20, 490, 530, ct=4))

INLET_Y = 880.0                       # rocker-front harness connectors (|y|)
HVAC_X = 1340.0                       # front face of the cabin HVAC module (on the interior part)

# =====================================================================
#  FIELD PROBES (outer skin of the v4 body)
# =====================================================================
def skin_y(x, zs, y_hi=1150.0):
    """Half-width of the outer skin at station x for heights zs (first sign change along +y)."""
    zs = np.atleast_1d(np.asarray(zs, float))
    ys = np.linspace(0.0, y_hi, 1151)
    f = v4.field(np.array([float(x)])[:, None, None], ys[None, :, None], zs[None, None, :])[0]   # (ny, nz)
    out = []
    for j in range(len(zs)):
        i = np.where(f[:, j] > 0)[0]
        if len(i) == 0 or i[0] == 0:
            out.append(np.nan); continue
        k = i[0]; f0, f1 = f[k - 1, j], f[k, j]
        out.append(ys[k - 1] + (ys[k] - ys[k - 1])*(-f0)/(f1 - f0))
    return np.array(out)

# =====================================================================
#  INTERIOR (removable cabin floor), printed upright on its flat underside (z = 380,
#  the pack top).  interior_parts() -> {tag: [...]}.  Footwells under the IP are tents
#  (50-deg roofs), seat cushions have 48-deg front undersides, headrests sit on posts.
# =====================================================================
Z_TUB = Z_PACK_TOP
FLOOR_X0, FLOOR_X1 = 1340.0, 3640.0
TUB_PEGS = [(PACK_X0 + 16, -300.0), (PACK_X0 + 16, 300.0)]   # pegs on the pack front wall
CONSOLE_HW = 125.0
TAN50 = math.tan(math.radians(50))

def slab_xz(pts, y0, y1):
    return extrude_xz(pts, y0, y1)

def seat(P, yc, front=True):
    """Bucket seat (front) or rear outboard place: base on the floor, cushion with a 48-deg
    front underside, reclined backrest with wings, headrest on two posts, buckle stalk,
    0.3 mm paint pockets for the perforated inserts."""
    zf = Z_FLOOR
    if front:
        w, xb0, xb1, zc_f, zc_r, xc_f, xc_r, back, hgt, z_hr, hw = 245.0, 2000.0, 2470.0, 690.0, 640.0, 1925.0, 2460.0, BACK_F, 600.0, 1490.0, 140.0
    else:
        w, xb0, xb1, zc_f, zc_r, xc_f, xc_r, back, hgt, z_hr, hw = 230.0, 2985.0, 3420.0, 700.0, 660.0, 2930.0, 3425.0, BACK_R, 560.0, 1488.0, 125.0
    tr = math.tan(math.radians(back))
    nx, nz = math.cos(math.radians(back)), -math.sin(math.radians(back))
    y0, y1 = yc - w, yc + w
    if front:
        for yy in (yc - 195, yc + 195):
            _add(P, "alu", box(xb0 - 60, xb1 + 60, yy - 15, yy + 15, zf - 4, zf + 22))
    top = lambda x: zc_f + (zc_r - zc_f)*(x - xc_f)/(xc_r - xc_f)
    bx0_, bz0_ = xc_r, zc_r - 15
    base = box(xb0, bx0_ + 115*nx + 20, y0, y1, zf - 4, 590)                  # base runs under the backrest
    cushion = slab_xz([(xb0, 585), (xb0 - (zc_f - 30 - 585)/T48, zc_f - 30), (xc_f, zc_f), (xc_r, zc_r), (xc_r, 585)], y0, y1)
    bol = union([slab_xz([(xc_f + 60, zc_f - 40), (xc_f + 60, top(xc_f + 60) + 28), (xc_r - 40, top(xc_r - 40) + 45), (xc_r - 40, zc_r - 40)],
                         yy, yy + 55) for yy in (y0, y1 - 55)])
    bx0, bz0 = xc_r, zc_r - 15
    bx1, bz1 = bx0 + hgt*tr, bz0 + hgt
    th = 115.0
    backrest = slab_xz([(bx0, bz0), (bx1, bz1), (bx1 + th*nx, bz1 + th*nz), (bx0 + th*nx + 20, 585), (bx0, 585)], y0, y1)
    wings = union([slab_xz([(bx0 + 10, bz0 + 40), (bx1 - 30*tr, bz1 - 60), (bx1 - 30*tr + th*nx, bz1 - 60 + th*nz),
                            (bx0 + 10 + th*nx + 20, bz0 + 40 + th*nz)], yy, yy + 50) for yy in (y0 - 20, y1 - 30)])
    pockets = union([slab_xz([(xc_f + 95, top(xc_f + 95) - M(0.3)), (xc_f + 95, top(xc_f + 95) + 30), (xc_r - 70, top(xc_r - 70) + 30),
                              (xc_r - 70, top(xc_r - 70) - M(0.3))], y0 + 70, y1 - 70),
                     slab_xz([(bx0 + 80*tr - M(0.3), bz0 + 80), (bx1 - 90*tr - M(0.3), bz1 - 90), (bx1 - 90*tr - 40, bz1 - 90),
                              (bx0 + 80*tr - 40, bz0 + 80)], y0 + 70, y1 - 70)])
    _add(P, "seat", diff(union([base, cushion, bol, backrest, wings]), [pockets]))
    hx = bx1 + 25*tr + 25
    hz0 = bz1 + 50
    for yy in (yc - 70, yc + 70):
        _add(P, "alu", cyl((hx, yy, bz1 - 60), (hx + 50*tr, yy, hz0 + 20), 10, seg=16))
    sk = (z_hr - hz0)*0.18
    _add(P, "seat", slab_xz([(hx - 50 + 50*tr, hz0), (hx + 50 + 50*tr, hz0), (hx + 70 + sk, z_hr - 25), (hx + 45 + sk, z_hr),
                             (hx - 55 + sk, z_hr), (hx - 68 + 50*tr, hz0 + 18/T48*1.0 + 18)], yc - hw, yc + hw))
    yin = yc - math.copysign(w + 12, yc)
    _add(P, "lv", cbox(xc_r - 75, xc_r - 35, yin - 12, yin + 12, zf - 4, zc_r + 70, ct=8))
    return bx1 + th*nx, bz1 + th*nz

def ws_clearance(off=75.0):
    """Region above the windscreen inner surface (minus off), as convex slabs across y."""
    parts = []
    ys = np.linspace(-900, 900, 23)
    for a, b in zip(ys[:-1], ys[1:]):
        pts = []
        for y in (a, b):
            for x in (1050.0, 2000.0):
                pts += [(x, y, v4.Z_WS0 + (x - float(v4.x_ws_base(y)))*v4.TAN_WS - off), (x, y, 1600.0)]
        parts.append(Manifold.hull_points(np.array(pts)))
    return union(parts)

def light_box(x0, x1, y0, y1, zb, zt, lean, exit_to_x):
    """Display pod with an internal 3-3.5 mm light box behind a see-through window and a
    LED wire channel down and forward through the IP.  Returns (pod, cavity, window)."""
    t = 20.0
    pod = slab_xz([(x0, zb), (x1, zb), (x1 + (zt - zb)*lean, zt), (x0 + (zt - zb)*lean, zt)], y0, y1)
    cz0, cz1 = zb + t, zt - t
    rid = (x1 - x0 - 2*t)/2*TAN50
    cav = slab_xz([(x0 + t, cz0), (x1 - t, cz0), (x1 - t + (cz1 - rid - zb)*lean, cz1 - rid),
                   ((x0 + x1)/2 + (cz1 - zb)*lean, cz1), (x0 + t + (cz1 - rid - zb)*lean, cz1 - rid)], y0 + t, y1 - t)
    win = slab_xz([(x1 - t - 2 + (cz0 + 10 - zb)*lean, cz0 + 10), (x1 + 30 + (cz0 + 10 - zb)*lean, cz0 + 10),
                   (x1 + 30 + (cz1 - rid - zb)*lean, cz1 - rid), (x1 - t - 2 + (cz1 - rid - zb)*lean, cz1 - rid)], y0 + t + 10, y1 - t - 10)
    ym = (y0 + y1)/2; xm = (x0 + x1)/2
    chan = tunnel([(xm, ym, cz0 + 30), (xm, ym, 950), (exit_to_x - 30, ym, 950)], M(LED_W)/2)
    return pod, union([cav, chan]), win

def interior_parts(envelope=None, wheel=None):
    """envelope: cabin volume inside the door cards (trims the IP ends); wheel: the placed
    steering wheel, unioned only for clearance studies (it prints separately)."""
    P = {}
    zf = Z_FLOOR
    # ---- floor plate on the pack, sill trims, floor mats
    fl = Polygon([(FLOOR_X0, -560), (FLOOR_X0 + 120, -850), (3255, -850), (3255, -660), (FLOOR_X1, -660),
                  (FLOOR_X1, 660), (3255, 660), (3255, 850), (FLOOR_X0 + 120, 850), (FLOOR_X0, 560)])
    holes = [teardrop_hole((x, y, Z_TUB - 1), (0, 0, 1), M(2.0 + 2*GAP), M(2.5) + 1) for x, y in TUB_PEGS]
    zb = zf + M(2.0)                                     # screw boss: the 1.6 mm head pocket needs more than the 1.2 mm floor
    for x, y in TUB_SCREWS:
        _add(P, "trim", symy(cyl((x, y, zf - 1), (x, y, zb), M(3.0), seg=32)))
        for s in (1, -1):
            holes += [cyl((x, s*y, Z_TUB - 1), (x, s*y, zb + 80), M(M2["clear"]/2), seg=24),
                      cyl((x, s*y, zb - M(M2["head_h"])), (x, s*y, zb + 80), M(M2["head"]/2), seg=24)]
    _add(P, "trim", prism(fl, "xy", Z_TUB, zf))
    _add(P, "trim", symy(cbox(FLOOR_X0 + 110, 3250, 790, 846, zf - 4, 468, ct=14)))
    for x, y, lx, ly in ((1690, -390, 400, 400), (1690, 390, 400, 400), (2790, -370, 220, 380), (2790, 370, 220, 380)):
        _add(P, "seat", box(x - lx/2, x + lx/2, y - ly/2, y + ly/2, zf - 4, zf + 8))
    # ---- seats: 2 front buckets, rear 60/40 bench (outboard places + centre)
    for yc in (-HP_F[1], HP_F[1]):
        seat(P, yc, True)
    for yc in (-HP_R[1], HP_R[1]):
        seat(P, yc, False)
    tr = math.tan(math.radians(BACK_R))
    _add(P, "seat", box(2985, 3547, -150, 150, zf - 4, 590), slab_xz([(2985, 585), (2930, 690), (3425, 650), (3425, 585)], -150, 150),
         slab_xz([(3425, 645), (3425 + 520*tr, 1165), (3425 + 520*tr + 102, 1165 - 52), (3545, 585), (3425, 585)], -150, 150))
    _add(P, "seat", slab_xz([(3425 + 545*tr + 15, 1235), (3425 + 545*tr + 110, 1235), (3425 + 610*tr + 115, 1420), (3425 + 610*tr + 5, 1420)],
                            -110, 110))
    for yy in (-60.0, 60.0):
        _add(P, "alu", cyl((3425 + 500*tr + 50, yy, 1130), (3425 + 545*tr + 60, yy, 1250), 10, seg=16))
    _add(P, "lv", cbox(3375, 3415, -12, 12, zf - 4, 730, ct=8))
    # ---- B-pillar lower trims with belt retractor, webbing and D-ring
    zs = np.linspace(zf, Z_CAN - 12, 12)
    ysk = skin_y(2520.0, zs)
    for s in (1, -1):
        poly = [(s*785, zf - 4)] + [(s*(yy - M(D_DOOR) - 22), z) for yy, z in zip(ysk, zs)] + [(s*785, Z_CAN - 12)]
        _add(P, "trim", prism(Polygon(poly).buffer(0), "yz", 2470, 2575))
        yr = s*785
        _add(P, "lv", cbox(2490, 2555, min(yr, yr - s*45), max(yr, yr - s*45), 470, 600, ct=10, cbx=(0, 0),
                           cby=(45, 0) if s > 0 else (0, 45)),
             box(2510, 2535, min(yr, yr - s*14), max(yr, yr - s*14), 590, Z_CAN - 60),
             cbox(2495, 2550, min(yr, yr - s*28), max(yr, yr - s*28), Z_CAN - 95, Z_CAN - 50, ct=6))
    # ---- instrument panel: tent-roofed footwells, centre stack = HVAC case
    ip = slab_xz([(FLOOR_X0, zf - 4), (FLOOR_X0, 1010), (1385, 1082), (1600, 1094), (1700, 1080), (1762, 1040), (1768, 900),
                  (1720, 845), (1720, zf - 4)], -820, 820)
    caves = []
    for s in (1, -1):
        ya, yb, zs_ = CAVE_Y[0], CAVE_Y[1], 600.0
        half = (yb - ya)/2
        tent = [(s*ya, zf - 10), (s*ya, zs_), (s*(ya + half), zs_ + half*TAN50), (s*yb, zs_), (s*yb, zf - 10)]
        caves.append(prism(Polygon(tent).buffer(0), "yz", FLOOR_X0 + 70, 1800))
    corner = Polygon([(1300, 500), (1300, 900), (1445, 900), (1445, 690)])  # clear the cowl gussets at the A-pillar bases
    cuts = [ws_clearance(), prism(corner, "xy", 300, 1300), prism(corner, "xy", 300, 1300).mirror([0, 1, 0])]
    for k in range(9):                                                       # defroster grille
        y = -460 + k*110
        cuts.append(box(1430, 1510, y, y + 80, 1062, 1130))
    for yc in (-700.0, -95.0, 95.0, 700.0):                                  # face vents: recess + slat grooves
        cuts.append(box(1765 - M(0.3), 1830, yc - 65, yc + 65, 968, 1030))
        for k in range(4):
            cuts.append(box(1765 - M(0.8), 1830, yc - 58, yc + 58, 976 + k*14, 976 + k*14 + M(0.6)))
    glove = sbox(240, 905, 690, 1010).buffer(25).buffer(-25)                  # glovebox shut line (y, z)
    cuts.append(prism(glove.buffer(M(0.35)).difference(glove.buffer(-M(0.35))), "yz", 1768 - M(1.0), 1830))
    cuts.append(box(1765 - M(0.3), 1830, -800, 800, 1015, 1030))              # trim inlay band (paint pocket)
    ip = diff(ip, caves + cuts)
    _add(P, "trim", ip.intersect(envelope) if envelope is not None else ip)
    hv = cbox(FLOOR_X0, 1560, -CAVE_Y[0] + 1, CAVE_Y[0] - 1, zf - 4, 820, ct=10)  # HVAC case (evaporator / PTC core / blend doors)
    _add(P, "alu", diff(hv, [cuts[0]]))
    for y in REFRIG_Y:                                                       # refrigerant ports on the front face
        _add(P, "refrigerant", cbox(HVAC_X - 18, HVAC_X + 5, y - 18, y + 18, zf - 4, 530, ct=4))
    _add(P, "motor", cyl((1450, 310, zf - 4), (1450, 310, 640), 125, seg=48), cbox(1340, 1540, 200, 440, 640, 730, ct=12),
         box(1400, 1500, 140, 200, zf - 4, 600))                             # blower scroll, fresh-air box, duct
    for s in (1, -1):                                                        # foot-well ducts on the floor
        _add(P, "inverter", box(1460, 1560, min(s*150, s*240), max(s*150, s*240), zf - 4, 470))
    # pedals (floor-hinged organ type) and dead pedal
    for y0, y1, xb, xt, zt in ((-330, -270, 1460, 1405, 585), (-480, -380, 1475, 1415, 610), (-760, -660, 1440, 1385, 560)):
        _add(P, "alu" if y0 > -700 else "trim", slab_xz([(xb, zf - 4), (xb + 34, zf - 4), (xt + 26, zt), (xt, zt)], y0, y1))
    # steering column shroud (48+ deg underside), socket for the steering wheel peg, stalks
    a = np.array([math.cos(math.radians(SW_TILT)), 0, math.sin(math.radians(SW_TILT))])
    c = np.array([SW_C[0], -SW_C[1], SW_C[2]])
    hub = c - a*45
    ipe = c - a*120
    pts = section_pts(hub, a, 42, False, 20) + section_pts(ipe, a, 55, False, 20) + [(1735, c[1] + dy, 905) for dy in (-60, 60)]
    shroud = Manifold.hull_points(np.array(pts))
    sock = Manifold.hull_points(np.array(section_pts(hub + a*2, a, M(SW_PEG/2 + GAP/2), False, 20)
                                         + section_pts(hub - a*M(4.4), a, M(SW_PEG/2 + GAP/2), False, 20)))
    _add(P, "trim", shroud)
    holes.append(sock)
    for dy in (-45.0, 45.0):                                                 # indicator / wiper stalks, 51-deg rise
        _add(P, "lv", tube([hub - a*28 + (0, dy, -6), hub - a*28 + (0, dy + math.copysign(64, dy), 80)], 9, tear=False))
    # instrument cluster and infotainment display: light boxes with LED channels to the IP front face
    for x0, x1, y0, y1, zb, zt, lean in ((1665, 1762, -540, -240, 1046, 1198, 0.12), (1625, 1730, -175, 175, 1072, 1222, 0.30)):
        pod, cav, win = light_box(x0, x1, y0, y1, zb, zt, lean, FLOOR_X0)
        _add(P, "screen", diff(pod, [cav, win]))
    # ---- centre console: floating bridge with an under-pass bay, wireless pad, cup holders, armrest
    con = cbox(1560, 2700, -CONSOLE_HW, CONSOLE_HW, zf - 4, 680, ct=18)
    bay = prism(Polygon([(1830, zf - 10), (1830, 440), (1970, 440 + 140*TAN50), (2110, 440), (2110, zf - 10)]),
                "xz", -CONSOLE_HW - 10, CONSOLE_HW + 10)                       # pass-through, 50-deg gable roof
    cups = [cyl((x, 0, 620), (x, 0, 700), 40, seg=32) for x in (2020, 2110)]
    pad = box(1800, 1945, -85, 85, 680 - M(0.3), 700)
    seam = box(2350 - M(0.35), 2350 + M(0.35), -CONSOLE_HW - 1, CONSOLE_HW + 1, 640, 760)
    _add(P, "trim", diff(union([con, cbox(2350, 2700, -CONSOLE_HW, CONSOLE_HW, 670, 735, ct=16)]), [bay, pad, seam] + cups))
    _add(P, "alu", cyl((2195, 0, 675), (2195, 0, 705), 28, seg=28))
    P["holes"] = holes
    return P

SW_PEG = 2.0                          # steering-wheel peg diameter (model mm)
CAVE_Y = (150.0, 790.0)               # footwell tents between the HVAC case and the kick panels

def steering_wheel():
    """Separate part, printed face-down (flat driver-side face on the bed), peg up.
    Rim 370 mm, D-section; 3 spokes; airbag hub.  Local frame: wheel plane = xy, peg along +z."""
    R, rr = SW_R, 17.0
    rim = revolve_axis([(R - rr, 0), (R + rr, 0), (R + rr, 12), (R + 6, 30), (R - 6, 30), (R - rr, 12)], (0, 0, 0), (0, 0, 1), 96)
    hub = revolve_axis([(0, 0), (78, 0), (78, 20), (60, 44), (0, 44)], (0, 0, 0), (0, 0, 1), 64)
    spokes = []
    for ang in (-90.0, 30.0, 150.0):
        t = math.radians(ang); u = np.array([math.cos(t), math.sin(t)])
        n = np.array([-u[1], u[0]])
        poly = [tuple(70*u + 24*n), tuple((R - 5)*u + 18*n), tuple((R - 5)*u - 18*n), tuple(70*u - 24*n)]
        spokes.append(extrude_xy(poly, 0, 22))
    peg = cyl((0, 0, 43), (0, 0, 44 + M(4.0)), M(SW_PEG/2), seg=24)
    w = union([rim, hub, peg] + spokes)
    emblem = revolve_axis([(30, -1), (36, -1), (36, M(0.3)), (30, M(0.3))], (0, 0, 0), (0, 0, 1), 48)
    return w - emblem

# =====================================================================
#  VOXEL TOOLS: occupancy by slicing, printable support fill, printable cavities
# =====================================================================
from scipy import ndimage
from PIL import Image, ImageDraw
from skimage.measure import marching_cubes

class Grid:
    """Anisotropic voxel grid: hxy in plane, hz per layer (hz = 2 hxy makes a 50-deg step
    1.68 px per layer, so ramps can be checked and built one layer at a time)."""
    def __init__(self, lo, hi, hxy, hz=None):
        self.lo = np.asarray(lo, float); self.hxy = float(hxy); self.hz = float(hz or 2*hxy)
        self.sp = np.array([self.hxy, self.hxy, self.hz])
        self.n = np.maximum(1, np.ceil((np.asarray(hi, float) - self.lo)/self.sp).astype(int))
    def zc(self, k):
        return self.lo[2] + (k + 0.5)*self.hz

def voxelize(man, g):
    """Occupancy of a manifold at voxel centres, one slice per layer (winding rule)."""
    nx, ny, nz = g.n
    occ = np.zeros((nx, ny, nz), bool)
    for k in range(nz):
        occ[:, :, k] = raster_polys(man.slice(g.zc(k)).to_polygons(), g.lo, g.hxy, (nx, ny))
    return occ

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

def support_fill(occ, g, angle=50.0, keep_out=None):
    """Material to add under every ceiling so each layer lies within r = hz/tan(angle) of
    the layer below.  Top-down: pixels that the layer below cannot reach are advected by r
    toward their nearest support (EDT feature transform), which grows 50-deg ramps and
    arches from the walls instead of hanging bridges.  Returns the added voxels."""
    nx, ny, nz = occ.shape
    r = g.hz/math.tan(math.radians(angle))/g.hxy          # allowed step in pixels per layer
    R = occ.copy()
    for k in range(nz - 1, 0, -1):
        S = R[:, :, k - 1]
        if not R[:, :, k].any():
            continue
        if not S.any():
            R[:, :, k - 1] |= R[:, :, k]
            continue
        D, (ix, iy) = ndimage.distance_transform_edt(~S, return_indices=True)
        U = R[:, :, k] & (D > r)
        if not U.any():
            continue
        px, py = np.nonzero(U)
        dx, dy = ix[px, py] - px, iy[px, py] - py
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        tx, ty = px + dx*np.minimum(r/d, 1), py + dy*np.minimum(r/d, 1)
        best = np.full(len(px), np.inf); bx = px.copy(); by = py.copy()
        for qx in (np.floor(tx), np.ceil(tx)):
            for qy in (np.floor(ty), np.ceil(ty)):
                qx_ = np.clip(qx.astype(int), 0, nx - 1); qy_ = np.clip(qy.astype(int), 0, ny - 1)
                ok = np.hypot(qx_ - px, qy_ - py) <= r + 1e-6
                dq = np.where(ok, D[qx_, qy_], np.inf)
                better = dq < best
                best = np.where(better, dq, best); bx = np.where(better, qx_, bx); by = np.where(better, qy_, by)
        F = np.zeros((nx, ny), bool)
        F[bx, by] = True
        if keep_out is not None:                      # never fill wheel houses: their flat crowns stay short bridges
            F &= ~keep_out[:, :, k - 1]
        R[:, :, k - 1] |= F
    return R & ~occ

def printable_cavity(K, g, angle=50.0):
    """Smallest cavity containing K whose ceilings are >= angle (bottom-up erosion)."""
    r = g.hz/math.tan(math.radians(angle))/g.hxy
    rr = int(math.ceil(r))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = (xx*xx + yy*yy) <= r*r + 1e-6
    C = K.copy()
    for k in range(1, K.shape[2]):
        C[:, :, k] |= ndimage.binary_erosion(C[:, :, k - 1], structure=disk, border_value=0)
    return C

def max_step(V, g):
    """Largest horizontal reach (pixels) of any layer beyond the layer below (printability)."""
    worst = 0.0
    for k in range(1, V.shape[2]):
        if V[:, :, k].any() and V[:, :, k - 1].any():
            worst = max(worst, float(ndimage.distance_transform_edt(~V[:, :, k - 1])[V[:, :, k]].max()))
    return worst

def voxels_to_manifold(V, g, smooth=0.6, dilate=0):
    """Smooth closed surface around a voxel set (EDT signed distance, Gaussian, marching cubes)."""
    if dilate:
        V = ndimage.binary_dilation(V, iterations=dilate)
    if not V.any():
        return Manifold()
    Vp = np.pad(V, 2)
    sd = ndimage.distance_transform_edt(~Vp, sampling=g.sp) - ndimage.distance_transform_edt(Vp, sampling=g.sp)
    if smooth:
        sd = ndimage.gaussian_filter(sd.astype(np.float32), smooth)
    v, f, _, _ = marching_cubes(sd, level=0.0, spacing=tuple(g.sp), allow_degenerate=False)
    v = v + g.lo - 1.5*g.sp
    man = v4._to_manifold(v, f)
    if man.volume() < 0:
        man = v4._to_manifold(v, f[:, ::-1])
    return man

# =====================================================================
#  BODY KIT: the v4 skin split into chassis skirt, front clip, rear clip (+ liftgate),
#  four doors, hood and greenhouse canopy.  All in full-size mm.
# =====================================================================
S_GAP = M(SPLIT)
VOX = (3.0, 7.5)                      # fill / cavity voxels: 3 mm in plane, 7.5 mm layers (51-deg steps)

def cavity_shape_kit(xa):
    """Arch opening + wheel house: circle with a gable roof clipped flat.  Front: 46-deg gable at
    CAV_TOP (the hood sits just above), 11.9 mm flat bridge.  Rear: 47-deg gable 30 mm higher
    (10 mm bridge, more bump travel)."""
    r = v4.R_OPEN + 4
    top = v4.CAV_TOP + (30.0 if xa == X_RA else 0.0)
    b = math.radians(47 if xa == X_RA else 46)
    pts = [(xa + r*math.sin(b), HUB_Z + r*math.cos(b)), (xa, HUB_Z + r/math.cos(b)), (xa - r*math.sin(b), HUB_Z + r*math.cos(b))]
    shape = unary_union([Point(xa, HUB_Z).buffer(r, resolution=48), Polygon(pts)])
    return shape.intersection(sbox(xa - 2*r, GC - 80, xa + 2*r, top))

def kit_wells():
    """Wheel houses: open inboard above the skirt (the awning roof stays), and the undertray
    reaches to |y| = 690 below it so the knuckles and lower ball joints have a floor."""
    cut = []
    for xa, win in ((X_FA, v4.WELL_IN_F), (X_RA, v4.WELL_IN)):
        sh = cavity_shape_kit(xa)
        for s in (1, -1):
            up = prism(sh, "xz", win, 905.0).intersect(box(-100, 5000, -1200, 1200, Z_SKIRT - 5, 2000))
            lo = prism(sh, "xz", 690.0, 905.0).intersect(box(-100, 5000, -1200, 1200, 0, Z_SKIRT + 5))
            m = union([up, lo])
            cut.append(m if s > 0 else m.mirror([0, 1, 0]))
    return union(cut)

def kit_liners():
    """Wheel-house awnings (shell around the cavity, open inboard), above the skirt."""
    t = M(T_SHELL)
    out = []
    for xa, win in ((X_FA, v4.WELL_IN_F), (X_RA, v4.WELL_IN)):
        sh = cavity_shape_kit(xa)
        m = prism(sh.buffer(t, join_style=2), "xz", win, 905.0 + t) - prism(sh, "xz", win - 1, 905.0)
        m = m.intersect(box(-100, 5000, -1200, 1200, Z_SKIRT, 2000))
        out.append(symy(m))
    return union(out)

def _edge_poly(edge, left, s, z0=250.0, z1=1250.0):
    """xz region left (x below) or right of a door edge polyline, offset by s."""
    e = [(x, z) for x, z in edge]
    e = [(e[0][0], z0)] + e + [(e[-1][0], z1)]
    if left:
        return Polygon([(-500, z0)] + [(x - s, z) for x, z in e] + [(-500, z1)])
    return Polygon([(x + s, z) for x, z in e] + [(6000, z1), (6000, z0)])

def door_edges():
    zb = v4.Z_BELT - 8
    fe = v4._sm([(1442, 345), (1442, 820), (1430, 960), (1418, zb)], 80)
    xc, zc0, rho = X_RA, HUB_Z, 572.0
    Q = np.array([3505.0, zb])
    th = np.linspace(0, math.pi/2, 2000)
    T = np.c_[xc - rho*np.cos(th), zc0 + rho*np.sin(th)]
    i1 = int(np.argmin(np.abs(np.einsum("ij,ij->i", Q - T, T - np.array([xc, zc0])))))
    re = [(xc - rho, 345.0)] + [tuple(T[i]) for i in np.linspace(0, i1, 40).astype(int)] + [tuple(Q)]
    return [tuple(p) for p in fe], re

def liftgate_region(s):
    """Liftgate panel behind the roof hinge line and inside the D-pillar shut lines."""
    zs = np.linspace(1240, 1630, 30)
    dl = [(v4.y_backlight(z) - 34, z) for z in zs]
    tg = [(690, 880), (690, 1100), (0.5*(690 + dl[0][0]) + 10, 1185), dl[0]]
    left = v4._sm(tg + dl[1:], 120)
    top = left[-1]
    poly = [(y - s, z) for y, z in left] + [(top[0] - s, 2000), (-top[0] + s, 2000)] + [(-(y - s), z) for y, z in left[::-1]]
    yz = prism(Polygon(poly).buffer(0), "yz", 3800, 5200)
    y_end = dl[-1][0]
    x_end = float(v4.REAR(dl[-1][1]) - v4.bow_rear(y_end))
    ys = np.linspace(-1, 1, 81)*y_end
    xh = [4236 - 30*(y/600)**2 + (x_end - 4236 + 30*(y_end/600)**2)*float(v4.smoothstep(0.80*y_end, y_end, abs(y))) for y in ys]
    xy = Polygon([(x + s, y) for x, y in zip(xh, ys)] + [(6000, y_end + 300), (6000, -y_end - 300)]).buffer(0)
    xy = unary_union([xy, sbox(x_end + s, -1200, 6000, 1200)])
    return yz.intersect(prism(xy, "xy", 800, 2200))

def hood_loop():
    hhw = v4.hood_halfwidth
    hl = [(158 + float(v4.bow_front(y)), y) for y in np.linspace(-1, 1, 41)*float(hhw(160))]
    sd = [(x, float(hhw(x))) for x in np.linspace(160, float(v4.x_ws_base(hhw(1150))) - 40, 50)]
    cw = [(float(v4.x_ws_base(y)) - 40, y) for y in np.linspace(1, -1, 41)*float(hhw(1200))]
    return Polygon(hl + sd + cw + [(x, -y) for x, y in sd[::-1]]).buffer(0)

def build_body_kit(h=6.0, mirrors="mirror", reduce=0.75, log=print):
    """v4 skin + engraving, kit wheel houses, and the inset skins used for shells and doors.
    Returns dict(base, body, inset_shell, inset_door, inset_clear, tags)."""
    t0 = time.time()
    vol, org, h = v4.sample(h, mirrors, log)
    base = v4.mesh_level(vol, org, h, 0.0, reduce).as_original()
    tags = {base.original_id(): "paint"}
    D = v4.details()
    depths = sorted({d for items in D.values() for _, d in items})
    insets = {d: v4.mesh_level(vol, org, h, -M(d), reduce) for d in depths}
    extra = {k: v4.mesh_level(vol, org, h, -M(d), reduce) for k, d in
             (("shell", T_SHELL), ("door", D_DOOR), ("fill", 1.2))}
    del vol
    log(f"  skin + {len(depths) + 3} insets ({time.time() - t0:.0f} s)")
    cutters = []
    for cat, items in D.items():
        groups = {}
        for man, d in items:
            groups.setdefault(d, []).append(man)
        for d, mans in groups.items():
            p = Manifold.batch_boolean(mans, OpType.Add).as_original()
            fl = insets[d].as_original()
            tags[p.original_id()] = cat + ":wall"
            tags[fl.original_id()] = cat
            cutters.append(p - fl)
    wells = kit_wells().as_original()
    tags[wells.original_id()] = "well"
    body = Manifold.batch_boolean([base, wells] + cutters, OpType.Subtract)
    log(f"  engraved kit body {body.num_tri():,} tri ({time.time() - t0:.0f} s)")
    return dict(base=base, body=body, inset_shell=extra["shell"], inset_door=extra["door"],
                inset_fill=extra["fill"], wells=wells, tags=tags)

def hood_bottom():
    """Flat hood underside: 1.5 mm (model) below the lowest point of the hood skin along its shut line."""
    loop = hood_loop()
    pts = np.asarray(loop.buffer(-40).exterior.coords)[::4]
    zs = np.linspace(850, 1100, 501)
    low = 1e9
    for x, y in pts:
        f = v4.field(np.array([x])[:, None, None], np.array([y])[None, :, None], zs[None, None, :])[0, 0]
        i = np.where(f <= 0)[0]
        if len(i):
            low = min(low, zs[i[-1]])
    return math.floor(low - M(1.5))

def kit_regions(z_hb, sx=None):
    """sx: offset of the door-edge cut planes (defaults to the split gap)."""
    s = S_GAP
    sx = s if sx is None else sx
    big = lambda z0, z1: box(-500, 6000, -1500, 1500, z0, z1)
    fe, re = door_edges()
    xzp = lambda poly: prism(poly, "xz", -1500, 1500)
    band = big(Z_SKIRT + s, Z_CAN - s)
    loop = hood_loop()
    hood_out = prism(loop.buffer(s), "xy", z_hb, 2200)
    L_in, L_out = liftgate_region(s), liftgate_region(-s)
    R = dict(skirt=big(-100, Z_SKIRT - s),
             front=band.intersect(xzp(_edge_poly(fe, True, sx))) - hood_out,
             door_fr=band.intersect(xzp(_edge_poly(fe, False, s))).intersect(box(-500, 2520 - s, 0, 1500, 0, 2200)),
             door_fl=band.intersect(xzp(_edge_poly(fe, False, s))).intersect(box(-500, 2520 - s, -1500, 0, 0, 2200)),
             door_rr=band.intersect(box(2520 + s, 6000, 0, 1500, 0, 2200)).intersect(xzp(_edge_poly(re, True, s))),
             door_rl=band.intersect(box(2520 + s, 6000, -1500, 0, 0, 2200)).intersect(xzp(_edge_poly(re, True, s))),
             rear=union([band.intersect(xzp(_edge_poly(re, False, sx))), L_in.intersect(big(Z_CAN - s - 1, 2200))]),
             canopy=big(Z_CAN + s, 2200) - L_out,
             hood=prism(loop.buffer(-s), "xy", z_hb, 2200))
    return R, fe, re

def fe_x(fe, z):
    zz = [p[1] for p in fe]; xx = [p[0] for p in fe]
    return float(np.interp(z, zz, xx))

def magnet(p, axis):
    """Pocket for a 3 x 2 mm magnet; p on the mating face, axis into the part."""
    return teardrop_hole(p, axis, M(MAG_D), M(MAG_H))

def largest(m):
    """Keep the main body of a part (drops voxel crumbs and boolean slivers)."""
    comps = m.decompose()
    if len(comps) <= 1:
        return m, 0.0
    comps.sort(key=lambda c: -c.volume())
    return comps[0], sum(c.volume() for c in comps[1:])

def decimate(m, reduce=0.8):
    if fast_simplification is None or m.is_empty():
        return m
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64); f = np.asarray(mesh.tri_verts).astype(np.int64)
    v2, f2 = fast_simplification.simplify(v, f, target_reduction=reduce, agg=5)
    m2 = v4._to_manifold(v2, f2)
    return m2 if m2.status().name == "NoError" and abs(m2.volume() - m.volume()) < 0.01*abs(m.volume()) else m

def support_filled(part, bounds, inner, region, keep_out=None, log=print, name=""):
    """Voxelize a hollow part, compute the printable support fill, merge it back.  The fill is
    clipped to `inner` (the skin offset 1.2 mm inward, deeper than any engraving) so it can
    only grow inside the shell and never refills the exterior detail."""
    g = Grid(bounds[0], bounds[1], *VOX)
    t = time.time()
    occ = voxelize(part, g)
    ko = None
    if keep_out is not None:                          # 0.5 mm clearance around chassis / interior parts
        ko = ndimage.binary_dilation(voxelize(keep_out, g), structure=np.ones((3, 3, 1), bool), iterations=int(round(M(0.8)/g.hxy)))
        ko = ndimage.binary_dilation(ko, structure=np.ones((1, 1, 3), bool), iterations=1)
    F = support_fill(occ, g, keep_out=ko)
    fill = decimate(voxels_to_manifold(F, g, smooth=0.7, dilate=1), 0.85)
    fill = fill.intersect(inner).intersect(region)
    out, crumbs = largest(union([part, fill]))
    log(f"    {name}: grid {tuple(g.n)}, fill {fill.volume()/8e6:.1f} cm3 at 1:20, crumbs dropped {crumbs/8e3:.1f} mm3 ({time.time() - t:.0f} s)")
    return out, fill

# ---------------------------------------------------------------------
#  fastener and wiring hard points (full size)
# ---------------------------------------------------------------------
CLIP_M2 = [(360.0, 770.0), (4480.0, 740.0)]      # front / rear bumper corners (each side)
CLIP_M3 = [(1395.0, 915.0), (3330.0, 915.0)]     # rocker ends (each side)
HOOD_MAG = [(420.0, 560.0), (1150.0, 640.0)]      # rear pair clear of the wheel-house gable, so the fill carries the pads
DOOR_PINS = [(1650.0, 2300.0), (2750.0, 3150.0)]
DOOR_MAG_Z = (560.0, 960.0)
TUB_SCREWS = [(XMEMBERS[2] + 18, 770.0)]         # M2 through the cabin floor into the pack side frame
LED_PASS = [(1300.0, 420.0), (1300.0, -420.0), (3420.0, 0.0)]   # LED wire pass-throughs in the undertray

def _pts_inside(poly, pts, margin):
    return [p for p in pts if poly.buffer(-margin).contains(Point(p))]

def screw_column(x, y, z0, z1, size):
    return cbox(x - 60, x + 60, y - 60, y + 60, z0, z1)

def screw_hole_from_below(x, y, z_top, spec):
    """Clearance hole + counterbore entered from the bed face (z = GC)."""
    return union([cyl((x, y, GC - 1), (x, y, z_top + 1), M(spec["clear"]/2), seg=24),
                  cyl((x, y, GC - 1), (x, y, GC + M(spec["head_h"])), M(spec["head"]/2), seg=24),
                  cyl((x, y, GC + M(spec["head_h"]) - 0.01), (x, y, GC + M(spec["head_h"]) + M(spec["head"]/2)), M(spec["head"]/2), 0.0, seg=24)])

def pilot_hole_up(x, y, z0, depth, spec):
    """Blind pilot hole entered from a part's bed face, cone roof (prints without support)."""
    return teardrop_hole((x, y, z0 - 1), (0, 0, 1), M(spec["pilot"]), M(depth) + 1)

def chassis_kit(B, R, log=print):
    """Skirt from the v4 skin (rockers solid) + all chassis CSG, fasteners, pins, pegs."""
    P = {}
    pack_parts(P); crash_parts(P); drive_unit(P, True); drive_unit(P, False)
    suspension_parts(P); thermal_parts(P); harness_parts(P)
    t = time.time()
    # fastener columns where the skirt is only a skin
    for x, y in CLIP_M2:
        _add(P, "chassis", symy(screw_column(x, y, 234, Z_SKIRT - S_GAP, M2)))
    # door locating pins on the rocker tops, cabin-floor pegs on the pack front wall
    for xs in DOOR_PINS:
        for x in xs:
            yp = float(skin_y(x, [Z_SKIRT])[0]) - M(D_DOOR)/2
            pin = union([cyl((x, yp, Z_SKIRT - 5), (x, yp, Z_SKIRT + M(1.6)), M(0.75), seg=20),
                         cyl((x, yp, Z_SKIRT + M(1.6) - 0.01), (x, yp, Z_SKIRT + M(2.0)), M(0.75), M(0.45), seg=20)])
            _add(P, "chassis", symy(pin))
    for x, y in TUB_PEGS:
        _add(P, "alu", union([cyl((x, y, Z_PACK_TOP - 5), (x, y, Z_PACK_TOP + M(2.0)), M(1.0), seg=24),
                              cyl((x, y, Z_PACK_TOP + M(2.0) - 0.01), (x, y, Z_PACK_TOP + M(2.3)), M(1.0), M(0.7), seg=24)]))
    G = {tag: union(ms) for tag, ms in P.items()}
    csg = union(list(G.values())).intersect(B["base"])
    hollow = B["inset_shell"].intersect(box(-500, 6000, -1500, 1500, GC + M(T_FLOOR) + 1, Z_SKIRT + 10))
    hollow = hollow - symy(box(1340, 3390, PACK_HW - 10, 1300, 0, Z_SKIRT + 20))       # solid rockers
    skirt = B["body"].intersect(R["skirt"]) - hollow
    cuts = [axle_bores()]
    for x, y in CLIP_M2:
        cuts += [screw_hole_from_below(x, s*y, Z_SKIRT, M2) for s in (1, -1)]
    for x, y in CLIP_M3:
        cuts += [screw_hole_from_below(x, s*y, Z_SKIRT, M3) for s in (1, -1)]
    for x, y in TUB_SCREWS:
        cuts += [teardrop_hole((x, s*y, Z_PACK_TOP + 1), (0, 0, -1), M(M2["pilot"]), M(7.0)) for s in (1, -1)]
    for x, y in CARGO_POSTS:
        cuts += [teardrop_hole((x, s*y, Z_CARGO[0] + 1), (0, 0, -1), M(MAG_D), M(MAG_H) + 1) for s in (1, -1)]
    for x, y in LED_PASS:
        cuts.append(cyl((x, y, GC - 1), (x, y, 300), M(2.0), seg=24))
    chassis = diff(union([skirt, csg]), cuts)
    log(f"  chassis: {chassis.num_tri():,} tri, {chassis.volume()/8e6:.1f} cm3 at 1:20 ({time.time() - t:.0f} s)")
    G["skin"] = skirt
    return chassis, G

def inner_y_solid(x0, x1, z0, z1, off, nx=56, nz=36):
    """Solid {|y| <= skin_y(x, z) - off} over a box of (x, z): the field is only a true distance
    near the skin, so deep offsets (door cards, cabin envelope) are measured along y instead."""
    xs = np.linspace(x0, x1, nx); zs = np.linspace(z0, z1, nz)
    Y = np.array([skin_y(x, zs) for x in xs]) - off
    Y = np.clip(np.nan_to_num(Y, nan=1.0), 1.0, None)
    idx = lambda i, k, s: (i*nz + k)*2 + s                 # s = 0: +y sheet, 1: -y sheet
    V = np.zeros((nx*nz*2, 3))
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            V[idx(i, k, 0)] = (x, Y[i, k], z); V[idx(i, k, 1)] = (x, -Y[i, k], z)
    F = []
    for i in range(nx - 1):
        for k in range(nz - 1):
            a, b, c, d = idx(i, k, 0), idx(i + 1, k, 0), idx(i + 1, k + 1, 0), idx(i, k + 1, 0)
            F += [(a, c, b), (a, d, c)]
            a, b, c, d = idx(i, k, 1), idx(i + 1, k, 1), idx(i + 1, k + 1, 1), idx(i, k + 1, 1)
            F += [(a, b, c), (a, c, d)]
    rim = [(i, 0) for i in range(nx - 1)] + [(nx - 1, k) for k in range(nz - 1)] + \
          [(i, nz - 1) for i in range(nx - 1, 0, -1)] + [(0, k) for k in range(nz - 1, 0, -1)]
    for (i, k), (j, l) in zip(rim, rim[1:] + rim[:1]):
        a, b, c, d = idx(i, k, 0), idx(j, l, 0), idx(j, l, 1), idx(i, k, 1)
        F += [(a, b, c), (a, c, d)]
    m = v4._to_manifold(V, np.array(F))
    return m if m.volume() > 0 else v4._to_manifold(V, np.array(F)[:, ::-1])

def shell_part(B, reg):
    """Body inside a region, hollowed to a T_SHELL skin, with the wheel-house awnings."""
    if "shell" not in B:
        B["shell"] = B["body"] - B["inset_shell"]
    return union([B["shell"].intersect(reg), kit_liners().intersect(B["base"]).intersect(reg)])

def clip_kit(B, R, which, z_hb, extra, cuts, keep=None, log=print, fill_region=None):
    reg = R[which]
    t0 = time.time()
    part = union([shell_part(B, reg)] + [m.intersect(B["base"]).intersect(reg) for m in extra])
    lo, hi = np.array(reg.bounding_box()[:3]), np.array(reg.bounding_box()[3:])
    bb = B["base"].bounding_box()
    lo = np.maximum(lo, np.array(bb[:3]) - 10); hi = np.minimum(hi, np.array(bb[3:]) + 10)
    lo[2] = reg.bounding_box()[2]
    part, fill = support_filled(part, (lo, hi), B["inset_fill"], fill_region or reg, union([B["wells"]] + ([keep] if keep else [])), log, which)
    part, _ = largest(diff(part, cuts))
    log(f"  {which}: {part.num_tri():,} tri, {part.volume()/8e6:.1f} cm3 at 1:20 ({time.time() - t0:.0f} s)")
    return part, fill

def door_kit(B, R, which, extra, cuts):
    reg = R[which]
    bb = reg.bounding_box()
    inner = inner_y_solid(max(bb[0], 1300) - 40, min(bb[3], 3600) + 40, Z_SKIRT - 20, Z_CAN + 20, M(D_DOOR))
    d = B["body"].intersect(reg) - inner
    d = union([d] + [m.intersect(B["base"]).intersect(reg) for m in extra])
    return largest(diff(d, cuts))[0]

def canopy_kit(B, R, interior, log=print):
    """Greenhouse/roof: solid (glass is painted, as on v4) with tent-roofed cavities over every
    interior part that rises above the beltline, so it prints upright on its flat underside.
    interior: everything that must fit under the canopy (cabin part + steering wheel)."""
    reg = R["canopy"]
    t0 = time.time()
    solid = B["body"].intersect(reg)
    bb = solid.bounding_box()
    g = Grid((bb[0] - 10, bb[1] - 10, Z_CAN + S_GAP), (bb[3] + 10, bb[4] + 10, bb[5] + 10), *VOX)
    K = voxelize(interior.intersect(box(-500, 6000, -1500, 1500, Z_CAN - 50, 2200)), g)
    clr = M(1.0)                                            # 1 mm clearance around the interior
    K = ndimage.binary_dilation(K, structure=np.ones((3, 3, 1), bool), iterations=int(round(clr/g.hxy)))
    K = ndimage.binary_dilation(K, structure=np.ones((1, 1, 3), bool), iterations=max(1, int(round(clr/g.hz))))
    C = printable_cavity(K, g)
    cav = voxels_to_manifold(C, g, smooth=0.6)
    thin = (cav - B["inset_shell"]).volume()/8e6
    cav = cav.intersect(B["inset_shell"])                  # never thinner than the 1.6 mm skin
    can = solid - cav
    log(f"  canopy: cavity {cav.volume()/8e6:.1f} cm3, breaks skin by {thin*1000:.1f} mm3 ({time.time() - t0:.0f} s)")
    return can, cav

def cargo_deck():
    x0, x1, hw = 3650.0, 4560.0, 640.0
    deck = cbox(x0, x1, -hw, hw, Z_CARGO[0], Z_CARGO[1], ct=6)
    grooves = [box(x0 + 40, x1 - 40, y - M(0.3), y + M(0.3), Z_CARGO[1] - M(0.6), Z_CARGO[1] + 1) for y in np.arange(-560, 561, 80)]
    rings = [cyl((x, s*560, Z_CARGO[1] - M(0.6)), (x, s*560, Z_CARGO[1] + 1), 28, seg=24) for x in (3720, 4480) for s in (1, -1)]
    handle = box(4400, 4520, -60, 60, Z_CARGO[1] - M(0.8), Z_CARGO[1] + 1)
    towers = [cyl((3966, s*552, Z_CARGO[0] - 5), (3966, s*552, Z_CARGO[1] + 5), 90, seg=40) for s in (1, -1)]
    mags = [teardrop_hole((x, s*y, Z_CARGO[0] - 1), (0, 0, 1), M(MAG_D), M(MAG_H) + 1) for x, y in CARGO_POSTS for s in (1, -1)]
    return diff(deck, grooves + rings + [handle] + mags + towers)

def interior_solid(P):
    return diff(union([union(v) for k, v in P.items() if k != "holes"]), P.get("holes", []))

def place_steering_wheel(w):
    """Local wheel frame (peg along +z, face on z = 0) -> car position (driver, LHD)."""
    t = math.radians(SW_TILT)
    zl = -np.array([math.cos(t), 0, math.sin(t)])
    xl = np.array([-math.sin(t), 0, math.cos(t)])
    yl = np.cross(zl, xl)
    c = np.array([SW_C[0], -SW_C[1], SW_C[2]])          # face at the wheel centre, hub back on the shroud
    return w.transform(np.c_[xl, yl, zl, c])

# =====================================================================
#  KIT FEATURES: fasteners, door cards, LED light channels, solar wire tunnels
#  body_features() -> (extra, cuts): {part: [manifold]} to add before the support fill
#  (so it is supported) and to cut afterwards (holes, channels, pockets).
# =====================================================================
def face_x(y, z, front=True):
    """x of the outer skin along a ray in x at (y, z) (front or rear face)."""
    xs = np.linspace(-40, 1200, 2481) if front else np.linspace(3600, 4840, 2481)
    f = v4.field(xs[:, None, None], np.array([float(y)])[None, :, None], np.array([float(z)])[None, None, :])[:, 0, 0]
    i = np.where(f <= 0)[0]
    if len(i) == 0:
        return float("nan")
    return float(xs[i[0]] if front else xs[i[-1]])

def edge_normal(edge, z):
    """Unit (nx, nz) normal of a door-edge polyline at height z, pointing toward +x."""
    e = np.asarray(edge, float)
    k = int(np.clip(np.searchsorted(e[:, 1], z), 1, len(e) - 1))
    d = e[k] - e[k - 1]; n = np.array([d[1], -d[0]]); n /= np.linalg.norm(n)
    return n if n[0] > 0 else -n

def best_point(poly, xr, yr, margin, step=10.0):
    """Point of poly (shapely) inside the box xr x yr farthest from the boundary (>= margin)."""
    best, bp = margin, None
    for x in np.arange(xr[0], xr[1] + 1, step):
        for y in np.arange(yr[0], yr[1] + 1, step):
            pt = Point(x, y)
            if poly.contains(pt):
                d = poly.exterior.distance(pt) if poly.geom_type == "Polygon" else poly.boundary.distance(pt)
                if d > best:
                    best, bp = d, (float(x), float(y))
    return bp

def slice_poly(man, z):
    """Horizontal section of a manifold as a shapely geometry (outer contours minus holes)."""
    outer, holes = [], []
    for p in man.slice(z).to_polygons():
        p = np.asarray(p, float)
        if len(p) < 3:
            continue
        a = 0.5*np.sum(p[:, 0]*np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1)*p[:, 1])
        (outer if a > 0 else holes).append(Polygon(p).buffer(0))
    return unary_union(outer).difference(unary_union(holes)) if outer else Polygon()

def door_card(side, front, s):
    """Armrest (48-deg underside), map pocket, speaker, handle recess, trim pocket.
    Returns (extra, cuts) in car coordinates for the door on side s (+1 right, -1 left)."""
    yin = lambda x, z: float(skin_y(x, [z])[0]) - M(D_DOOR)
    ext, cut = [], []
    xa, xb, zt = (1795.0, 2400.0, 935.0) if front else (2690.0, 3180.0, 950.0)
    pts = []
    for x in (xa, xb):
        yt = yin(x, zt); zl = zt - 12 - 80*1.11; yl = yin(x, zl)
        pts += [(x, yt + 12, zt), (x, yt - 70, zt), (x, yt - 70, zt - 12), (x, yl + 12, zl)]
    arm = Manifold.hull_points(np.array([(x, s*y, z) for x, y, z in pts]))
    ext.append(arm)
    xc = (2150.0, 2290.0) if front else (2960.0, 3080.0)
    yt = yin(xc[0], zt)
    cut.append(box(xc[0], xc[1], min(s*(yt - 50), s*(yt - 22)), max(s*(yt - 50), s*(yt - 22)), zt - 30, zt + 5))
    xp = (1560.0, 2350.0) if front else (2620.0, 3120.0)
    z0 = 480.0                                     # above the cabin-floor sill trims (z <= 468)
    for x0, x1 in ((xp[0], xp[1]),):
        q, w = [], []
        for x in (x0, x1):
            y0, y1 = yin(x, z0), yin(x, z0 + 65*1.11)
            q += [(x, s*(y0 + 10), z0), (x, s*(y1 + 10), z0 + 65*1.11 + 18), (x, s*(y1 - 55), z0 + 65*1.11 + 18), (x, s*(y1 - 55), z0 + 65*1.11)]
            w += [(x, s*(y1 - 55), z0 + 65*1.11), (x, s*(y1 - 35), z0 + 65*1.11), (x, s*(y1 - 55), 640.0), (x, s*(y1 - 35), 640.0)]
        ext += [Manifold.hull_points(np.array(q)), Manifold.hull_points(np.array(w))]
    xs_, zs_ = (1600.0, 760.0) if front else (2700.0, 770.0)
    ys = yin(xs_, zs_)
    c0 = np.array([xs_, s*(ys - 6), zs_]); ax = np.array([0, s*1.0, 0])
    cut.append(revolve_axis([(58, 0), (70, 0), (70, M(0.8) + 6), (58, M(0.8) + 6)], c0, ax, 48))
    cut.append(revolve_axis([(0, 0), (48, 0), (48, M(0.3) + 6), (0, M(0.3) + 6)], c0, ax, 40))
    xh = (1490.0, 1590.0) if front else (2600.0, 2680.0)
    yh = yin(xh[0], 985)
    cut.append(box(xh[0], xh[1], min(s*(yh - 6), s*(yh + M(0.8))), max(s*(yh - 6), s*(yh + M(0.8))), 965, 1005))
    xt = (1610.0, 2380.0) if front else (2700.0, 3200.0)
    for x0 in np.arange(xt[0], xt[1], 80.0):                       # trim insert pocket above the armrest, piecewise along the card
        q = []
        for x in (x0, min(x0 + 82, xt[1])):
            for z in (1025.0, 1075.0):
                yy = yin(x, z); q += [(x, s*(yy - 6), z), (x, s*(yy + M(0.3)), z)]
        cut.append(Manifold.hull_points(np.array(q)))
    return ext, cut

def gabled_slit(xface, ya, yb, z0, z1, inward, ramp=False, depth=None):
    """Light slit through a vertical skin (x-facing) between ya and yb: open z0..z1 at the face,
    with a 50-deg roof across the wall depth so its ceiling never bridges along the slit:
    a gable (blind slits) or, with ramp=True, a roof rising all the way to the centre line of
    the LED channel behind it (depth = skin to channel centre), where it meets the channel's
    upward tip, so no knife edge is left hanging between slit and channel."""
    depth = depth or M(T_SHELL) + 8
    t50 = math.tan(math.radians(50))
    pts = []
    for y in (ya, yb):
        xf = xface(y)
        xo, xi = xf - inward*20, xf + inward*depth
        xm = 0.5*(xf + xi)
        pts += [(xo, y, z0), (xo, y, z1 - abs(xo - xf)*t50), (xf, y, z1), (xi, y, z0)]
        pts += [(xi, y, z1 + depth*t50)] if ramp else [(xi, y, z1), (xm, y, z1 + 0.5*depth*t50)]
    return Manifold.hull_points(np.array(pts))

def body_features(B, R, fe, re, z_hb, log=print):
    s = S_GAP
    E = {k: [] for k in R}; C = {k: [] for k in R}
    zt = Z_CAN - s; zb = Z_SKIRT + s
    t0 = time.time()
    # ---- clip <-> chassis screws (pilot holes in bosses on the clip bed face)
    for (x, y), part, spec in ((CLIP_M2[0], "front", M2), (CLIP_M3[0], "front", M3), (CLIP_M2[1], "rear", M2), (CLIP_M3[1], "rear", M3)):
        for sg in (1, -1):
            E[part].append(cbox(x - 58, x + 58, sg*y - 58, sg*y + 58, zb, zb + M(9)))
            C[part].append(pilot_hole_up(x, sg*y, zb, 8.0, spec))
    # ---- door edge magnets: bosses in the clips, pockets in the doors
    for z in DOOR_MAG_Z:
        for sg, dside in ((1, "r"), (-1, "l")):
            xf = fe_x(fe, z) - s
            ym = float(skin_y(xf, [z])[0]) - M(D_DOOR)/2
            E["front"].append(box(xf - 120, xf + 1, sg*ym - 55, sg*ym + 55, z - 55, z + 55))
            C["front"].append(magnet((xf, sg*ym, z), (-1, 0, 0)))
            C["door_f" + dside].append(magnet((xf + 2*s, sg*ym, z), (1, 0, 0)))
            yb_ = float(skin_y(2520.0, [z])[0]) - M(D_DOOR)/2
            C["door_f" + dside].append(magnet((2520 - s, sg*yb_, z), (-1, 0, 0)))
            C["door_r" + dside].append(magnet((2520 + s, sg*yb_, z), (1, 0, 0)))
            zz = [p[1] for p in re]; xx = [p[0] for p in re]
            xr_ = float(np.interp(z, zz, xx)) + s
            n = edge_normal(re, z)
            yr_ = float(skin_y(xr_, [z])[0]) - M(D_DOOR)/2
            p = np.array([xr_, sg*yr_, z]); ax = np.array([n[0], 0, n[1]])
            E["rear"].append(Manifold.hull_points(np.array([p + ax*a + (0, dy, dz) for a in (-1, 120) for dy in (-55, 55) for dz in (-55, 55)])))
            C["rear"].append(magnet(p, ax))
            C["door_r" + dside].append(magnet(p - ax*2*s, -ax))
    # ---- door bottoms: pin sockets; door tops: canopy alignment pins
    for xs_, dpart in ((DOOR_PINS[0], "door_f"), (DOOR_PINS[1], "door_r")):
        for x in xs_:
            yp = float(skin_y(x, [Z_SKIRT])[0]) - M(D_DOOR)/2
            for sg, dside in ((1, "r"), (-1, "l")):
                C[dpart + dside].append(teardrop_hole((x, sg*yp, zb - 1), (0, 0, 1), M(1.5 + 2*GAP), M(2.2) + 1))
    for xs_, dpart in (((1800.0, 2380.0), "door_f"), ((2700.0, 3250.0), "door_r")):
        for x in xs_:
            yp = float(skin_y(x, [Z_CAN - 10])[0]) - M(D_DOOR)/2
            for sg, dside in ((1, "r"), (-1, "l")):
                E[dpart + dside].append(union([cyl((x, sg*yp, zt - 5), (x, sg*yp, zt + M(1.6)), M(0.75), seg=20),
                                              cyl((x, sg*yp, zt + M(1.6) - 0.01), (x, sg*yp, zt + M(2.0)), M(0.75), M(0.45), seg=20)]))
                C["canopy"].append(teardrop_hole((x, sg*yp, Z_CAN + s - 1), (0, 0, 1), M(1.5 + 2*GAP), M(2.2) + 1))
    # ---- door cards
    for dpart, front in (("door_f", True), ("door_r", False)):
        for sg, dside in ((1, "r"), (-1, "l")):
            e, c = door_card(sg, front, sg)
            E[dpart + dside] += e; C[dpart + dside] += c
    # ---- canopy magnets over the rear clip's C-pillars (the front clip only offers the thin
    #      windscreen band at the belt plane, so the canopy front is located by the door-top pins)
    can_sl = slice_poly(B["body"].intersect(R["canopy"]), Z_CAN + s + 1)
    for part, xr in (("rear", (3560.0, 3900.0)),):
        bp = best_point(can_sl, xr, (835.0, 880.0), M(1.8), step=5.0)
        if bp is None:
            log(f"  ! no canopy magnet site over the {part} clip"); continue
        for sg in (1, -1):
            x, y = bp[0], sg*bp[1]
            E[part].append(cbox(x - 60, x + 60, min(y - sg*60, sg*1200), max(y - sg*60, sg*1200), zt - M(5), zt))
            C[part].append(teardrop_hole((x, y, zt + 1), (0, 0, -1), M(MAG_D), M(MAG_H) + 1))
            C["canopy"].append(teardrop_hole((x, y, Z_CAN + s - 1), (0, 0, 1), M(MAG_D), M(MAG_H) + 1))
        log(f"    canopy magnets over the {part} clip at x {bp[0]:.0f}, |y| {bp[1]:.0f}")
    # ---- hood magnets, ledge, PV pigtail socket
    loop = hood_loop()
    E["front"].append(prism(loop.buffer(-s).difference(loop.buffer(-s - M(2.2))), "xy", z_hb - M(1.6), z_hb - s) - B["wells"])
    for x, y in HOOD_MAG:
        for sg in (1, -1):
            ye = float(v4.hood_halfwidth(x))
            E["front"].append(cbox(x - 60, x + 60, min(sg*(y - 60), sg*ye), max(sg*(y - 60), sg*ye), z_hb - M(4.5), z_hb - s))
            C["front"].append(teardrop_hole((x, sg*y, z_hb - s + 1), (0, 0, -1), M(MAG_D), M(MAG_H) + 1))
            C["hood"].append(teardrop_hole((x, sg*y, z_hb - 1), (0, 0, 1), M(MAG_D), M(MAG_H) + 1))
    C["hood"].append(teardrop_hole((725.0, 230.0, z_hb - 1), (0, 0, 1), M(WIRE_D), M(3.0)))
    # ---- headlight light box: channel along the lamp band, LED windows, blade slit, wire exit
    zc = 905.0
    ys = np.linspace(-840, 840, 43)
    xf = np.array([face_x(y, zc) for y in ys])
    rc = M(LED_W)/2
    path = [(x + M(T_SHELL) + rc, y, zc) for x, y in zip(xf, ys)]
    E["front"].append(tube(path, rc + M(T_SHELL), tear=True))
    C["front"].append(tunnel(path, rc))
    C["front"].append(tunnel([(xf[21] + M(T_SHELL) + rc, 0.0, zc), (xf[21] + 300, 0.0, zc)], M(1.5)))
    xfi = lambda y: float(np.interp(y, ys, xf))
    for sg in (1, -1):
        for i in range(6):
            y = sg*(470 + i*58)
            C["front"].append(tunnel([(xfi(y) - 20, y, 910.0), (xfi(y) + M(T_SHELL) + rc, y, 910.0)], M(0.5)))
    for a, b in zip(ys[1:-2], ys[2:-1]):
        if abs(a) > 800 or abs(b) > 800:
            continue
        C["front"].append(gabled_slit(xfi, a, b, 941.0, 947.0, +1))
    # ---- tail light bar on the liftgate (rear clip): channel, slit, wire exit
    zc = 1192.0
    ys = np.linspace(-640, 640, 33)
    xr = np.array([face_x(y, zc, front=False) for y in ys])
    path = [(x - M(T_SHELL) - rc, y, zc) for x, y in zip(xr, ys)]
    E["rear"].append(tube(path, rc + M(T_SHELL), tear=True))
    C["rear"].append(tunnel(path, rc))
    C["rear"].append(tunnel([(xr[16] - M(T_SHELL) - rc, 0.0, zc), (xr[16] - 300, 0.0, zc)], M(1.5)))
    xri = lambda y: float(np.interp(y, ys, xr))
    for a, b in zip(ys[:-1], ys[1:]):
        C["rear"].append(gabled_slit(xri, a, b, 1188.0, 1196.0, -1, ramp=True, depth=M(T_SHELL) + rc))
    # ---- tail light corner units (canopy): channel behind the corner unit, LED windows, exit down
    for sg in (1, -1):
        yy = np.linspace(690, 780, 7)
        xc_ = [face_x(sg*y, 1160.0, front=False) for y in yy]
        path = [(x - M(T_SHELL) - rc - 10, sg*y, 1160.0) for x, y in zip(xc_, yy)]
        C["canopy"].append(tunnel(path, rc))
        px, py, _ = path[3]
        C["canopy"].append(tunnel([(px, py, 1160.0), (px, py, Z_CAN - 5)], M(1.5)))
        E["rear"].append(cbox(px - 60, px + 60, min(py - sg*60, sg*1200), max(py - sg*60, sg*1200), zt - M(5), zt))
        C["rear"].append(teardrop_hole((px, py, zt + 1), (0, 0, -1), M(3.0), M(6.0)))
        for y, x in zip(yy[1:-1:2], xc_[1:-1:2]):
            C["canopy"].append(tunnel([(x + 20, sg*y, 1160.0), (x - M(T_SHELL) - rc - 10, sg*y, 1160.0)], M(0.5)))
    # ---- solar strings: roof junction -> rails -> A-pillars -> front clip -> rocker-front connectors;
    #      tailgate string along the roof centre line from the liftgate hinge
    rw = M(WIRE_D)/2
    for sg in (1, -1):
        pts = []
        for z in np.linspace(1640, Z_CAN + 40, 12):
            xa = v4.x_apillar(z); ya = float(v4.y_glass(xa, z))
            pts.append((xa + 35, sg*(ya - 60), z - 40))
        xe, ye = 1398.0, 745.0
        roof = [(2400.0, 0.0, 1655.0), (2350.0, sg*420.0, 1640.0), (2330.0, sg*620.0, 1615.0)]
        C["canopy"].append(tunnel(roof + pts + [(xe, sg*ye, Z_CAN + 30), (xe, sg*ye, Z_CAN - 5)], rw))
        yk = INLET_Y if sg < 0 else INLET_Y - 70
        C["front"].append(tunnel([(xe, sg*ye, Z_CAN + 5), (xe, sg*ye, 1000.0), (PACK_X0 - 56, sg*yk, 600.0), (PACK_X0 - 56, sg*yk, Z_SKIRT - 5)], rw))
    xh0 = 4236.0 - 2*s
    cen = [(x, 0.0, float(v4.ROOF(x)) - 70) for x in np.linspace(xh0 + 10, 2400, 14)]
    C["canopy"].append(tunnel(cen, rw))
    # ---- charge inlet: socket housing behind the charge-port door, harness tunnel to the rocker
    yi = float(skin_y(1258.0, [958.0])[0])
    E["front"].append(cbox(1205, 1312, yi - M(T_SHELL) - 70, yi - 2, 915, 1000))
    C["front"].append(tunnel([(1258.0, yi - M(T_SHELL) - 40, 930.0), (1300.0, yi - 70, 700.0), (PACK_X0 - 56, INLET_Y, 400.0),
                              (PACK_X0 - 56, INLET_Y, Z_SKIRT - 5)], rw))
    log(f"  features ({time.time() - t0:.0f} s)")
    return E, C

# =====================================================================
#  BUILD + EXPORT
# =====================================================================
PRINT_PARTS = ["chassis", "interior", "front", "rear", "door_fl", "door_fr", "door_rl", "door_rr", "hood", "canopy",
               "cargo_deck", "steering_wheel"]
FILE = dict(chassis="kit_01_chassis", interior="kit_02_cabin_floor", front="kit_03_front_clip", rear="kit_04_rear_clip",
            door_fl="kit_05_door_front_left", door_fr="kit_06_door_front_right", door_rl="kit_07_door_rear_left",
            door_rr="kit_08_door_rear_right", hood="kit_09_hood", canopy="kit_10_roof_canopy", cargo_deck="kit_11_cargo_deck",
            steering_wheel="kit_12_steering_wheel")

def build_kit(h=6.0, mirrors="mirror", log=print):
    """All kit parts in car coordinates (full-size mm).  Returns (parts, info)."""
    t0 = time.time()
    B = build_body_kit(h, mirrors, log=log)
    z_hb = hood_bottom()
    R, fe, re = kit_regions(z_hb)
    E, C = body_features(B, R, fe, re, z_hb, log)
    chassis, CG = chassis_kit(B, R, log)
    chassis, crumbs = largest(chassis)
    env = inner_y_solid(1250, 3700, 370, 1130, M(D_DOOR) + 25)
    IP = interior_parts(envelope=env)
    interior = interior_solid(IP)
    wheel_local = steering_wheel()
    wheel = place_steering_wheel(wheel_local)
    deck = cargo_deck()
    keep = union([v for k, v in CG.items() if k != "skin"] + [interior, wheel, deck])
    parts = dict(chassis=chassis, interior=interior)
    RF, _, _ = kit_regions(z_hb, sx=S_GAP + 6.0)                  # fill stays 6 mm off the door-edge cut planes
    for which in ("front", "rear"):
        parts[which], _ = clip_kit(B, R, which, z_hb, E[which], C[which], keep=keep, log=log, fill_region=RF[which])
    for which in ("door_fl", "door_fr", "door_rl", "door_rr"):
        parts[which] = door_kit(B, R, which, E[which], C[which])
    parts["hood"] = largest(diff(B["body"].intersect(R["hood"]), C["hood"]))[0]
    can, cav = canopy_kit(B, R, union([interior, wheel]), log)
    parts["canopy"] = largest(diff(can, C["canopy"]))[0]
    parts["cargo_deck"] = deck
    parts["steering_wheel_placed"] = wheel
    info = dict(z_hood=z_hb, tags=B["tags"], chassis_groups=CG, interior_groups={k: union(v) for k, v in IP.items() if k != "holes"},
                wheel_local=wheel_local, time=time.time() - t0)
    log(f"kit built in {info['time']:.0f} s")
    return parts, info

def to_print(m):
    """Car coordinates (full size) -> model mm, bed face at z = 0."""
    m = m.scale([1/SCALE]*3)
    return m.translate([0, 0, -m.bounding_box()[2]])


def separate_pinches(v, f, eps=0.002):
    """STL has no topology: readers merge vertices by position.  Where two sheets of a valid
    manifold touch at a vertex or an edge (distinct vertex indices, same position) that merge
    would create non-manifold edges.  Each such vertex is nudged eps (model mm) toward the
    centroid of its own incident faces, so positions become unique and topology is kept."""
    v = np.asarray(v, np.float64).copy(); f = np.asarray(f, np.int64)
    key = v.astype(np.float32)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    dup = np.nonzero(cnt[inv.ravel()] > 1)[0]
    if len(dup) == 0:
        return v, 0
    cen = v[f].mean(axis=1)
    acc = np.zeros_like(v); n = np.zeros(len(v))
    for k in range(3):
        np.add.at(acc, f[:, k], cen); np.add.at(n, f[:, k], 1)
    d = acc[dup]/np.maximum(n[dup], 1)[:, None] - v[dup]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    v[dup] += eps*d
    return v, len(dup)

def write_part(m, path, name="solar_suv_v4_kit"):
    """Car coordinates (full size) -> 1:20, bed face at z = 0 -> binary STL with unique vertex positions."""
    p = to_print(m)
    mesh = p.to_mesh()
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    v, nd = separate_pinches(np.asarray(mesh.vert_properties)[:, :3], f)
    f, dropped = drop_slivers(v, f)
    v, f, repaired = verify_and_repair(v, f)
    tri = v[f].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())
    write_part.last = dict(pinches_separated=int(nd), slivers_dropped=len(dropped), sliver_mm3=round(float(sum(dropped)), 4),
                           repaired_faces=int(repaired))
    return p, nd

def drop_slivers(v, f, min_mm3=0.05):
    """Remove closed pieces that share no edge with the part (boolean slivers, < min_mm3)."""
    import trimesh
    tm = trimesh.Trimesh(v, f, process=False)
    comps = trimesh.graph.connected_components(tm.face_adjacency, nodes=np.arange(len(f)), min_len=1)
    if len(comps) <= 1:
        return f, []
    tri = v[f]
    vol = lambda idx: abs(np.einsum("ij,ij->i", tri[idx, 0], np.cross(tri[idx, 1], tri[idx, 2])).sum())/6
    vols = [vol(c) for c in comps]
    big = int(np.argmax(vols))
    keep = [c for i, c in enumerate(comps) if i == big or vols[i] >= min_mm3]
    dropped = [vols[i] for i in range(len(comps)) if not (i == big or vols[i] >= min_mm3)]
    return f[np.sort(np.concatenate(keep))], dropped

# =====================================================================
#  PACKAGE METRICS (full size), measured on the v4 surfaces and the kit layout
# =====================================================================
def _ztop(x, y):
    zs = np.linspace(200, 1800, 3201)
    f = v4.field(np.array([float(x)])[:, None, None], np.array([float(y)])[None, :, None], zs[None, None, :])[0, 0]
    i = np.where(f <= 0)[0]
    return float(zs[i[-1]]) if len(i) else float("nan")

def package_metrics():
    M_ = {}
    roof_struct = 70.0                      # roof skin + PV laminate + bows + headliner
    def headroom(hp, back=8.0):
        t = math.radians(back)
        for L in np.arange(600, 1300, 2.0):
            x, z = hp[0] + L*math.sin(t), hp[2] + L*math.cos(t)
            if z >= _ztop(x, -hp[1]) - roof_struct:
                return L + 102.0            # SAE effective headroom = line length + 102 mm
        return float("nan")
    M_["H61_front_headroom"] = round(headroom(HP_F), 0)
    M_["H63_rear_headroom"] = round(headroom(HP_R), 0)
    M_["H30_front_seat_height"] = HP_F[2] - AHP[2]
    M_["L53_hpoint_to_heel"] = HP_F[0] - AHP[0]
    ank = (AHP[0] - 50.0, AHP[2] + 95.0)
    M_["L34_front_legroom"] = round(math.hypot(HP_F[0] - ank[0], HP_F[2] - ank[1]) + 254.0, 0)
    M_["Hpoint_front_ground"] = HP_F[2]; M_["Hpoint_rear_ground"] = HP_R[2]
    M_["L50_couple_distance"] = HP_R[0] - HP_F[0]
    M_["eye_point"] = [EYE[0], -EYE[1], EYE[2]]
    xe, ye, ze = EYE[0], -EYE[1], EYE[2]
    down = min(math.degrees(math.atan2(ze - _ztop(x, ye), xe - x)) for x in np.arange(150, 1400, 10))
    M_["down_vision_deg"] = round(down, 1)
    hdr = (2230.0, ye)
    zh = v4.Z_WS0 + (hdr[0] - float(v4.x_ws_base(abs(ye))))*v4.TAN_WS - 37
    M_["up_vision_deg"] = round(math.degrees(math.atan2(zh - ze, xe - hdr[0])), 1)
    M_["shoulder_room_front"] = round(2*(float(skin_y(2450.0, [Z_CAN])[0]) - M(D_DOOR)) - 60, 0)
    M_["hip_room_front"] = round(2*(float(skin_y(2400.0, [HP_F[2]])[0]) - M(D_DOOR) - 70), 0)
    M_["hip_room_rear"] = round(2*min(float(skin_y(3260.0, [HP_R[2]])[0]) - M(D_DOOR) - 70, v4.WELL_IN - 20), 0)
    M_["step_in_height"] = 468.0
    M_["front_door_opening_length"] = 2520.0 - 1442.0
    M_["rear_door_opening_length_at_belt"] = 3505.0 - 2520.0
    lip_f = (v4.XCH_F, GC)
    M_["approach_deg"] = round(math.degrees(math.atan2(v4.Z_LIP_F, X_FA)), 1)
    M_["departure_deg"] = round(math.degrees(math.atan2(v4.Z_LIP_R, v4.L - X_RA)), 1)
    M_["breakover_deg"] = round(2*math.degrees(math.atan2(GC, (X_RA - X_FA)/2)), 1)
    n = 2*8*MOD_CELLS
    kwh = n*CELL["ah"]*CELL["v"]/1000
    M_["pack"] = dict(config="112s2p", cells=n, modules=16, cell_mm=[CELL["l"], CELL["t"], CELL["h"]], nominal_V=round(112*CELL["v"], 0),
                      V_range=[round(112*3.0, 0), round(112*4.2, 0)], Ah=2*CELL["ah"], gross_kWh=round(kwh, 1), usable_kWh=round(kwh*0.95, 1),
                      envelope_mm=[PACK_X1 - PACK_X0, 2*PACK_HW, Z_PACK_TOP - GC], cell_mass_kg=round(n*CELL["kg"], 0),
                      est_pack_mass_kg=round(n*CELL["kg"]*1.32, 0), x_range=[PACK_X0, PACK_X1], module_rows_x=[round(x) for x in ROWS])
    M_["range_km_at_19kWh"] = round(kwh*0.95/0.19, 0)
    M_["cargo_floor_height"] = Z_CARGO[1]
    return M_

def split_t_junctions(tm, tol=1e-4):
    """Close zero-area slits: where a boundary edge (u, w) has another boundary vertex b lying on
    it, split the face that owns (u, w) at b.  Returns the number of faces split."""
    import trimesh
    from collections import Counter
    n_split = 0
    for _ in range(8):
        e = tm.edges_sorted
        cnt = Counter(map(tuple, e))
        bnd = [i for i, k in enumerate(map(tuple, e)) if cnt[k] == 1]
        if not bnd:
            break
        bv = np.unique(e[bnd])
        faces = tm.faces.copy(); new = []; drop = set()
        for i in bnd:
            fi = tm.edges_face[i]
            if fi in drop:
                continue
            u, w = tm.edges[i]                          # directed as in the face
            pu, pw = tm.vertices[u], tm.vertices[w]
            d = pw - pu; L2 = float(d @ d)
            if L2 == 0:
                continue
            t = (tm.vertices[bv] - pu) @ d/L2
            dist = np.linalg.norm(tm.vertices[bv] - (pu + np.outer(t, d)), axis=1)
            ok = (t > 1e-6) & (t < 1 - 1e-6) & (dist < tol) & (bv != u) & (bv != w)
            if not ok.any():
                continue
            b = int(bv[np.flatnonzero(ok)[np.argmin(t[ok])]])  # nearest to u; later passes take the rest
            a, bb, c = faces[fi]
            x = [q for q in (a, bb, c) if q not in (u, w)][0]
            k = list(faces[fi]).index(u)
            if faces[fi][(k + 1) % 3] == w:                     # face order u -> w -> x
                new += [(u, b, x), (b, w, x)]
            else:                                               # face order w -> u -> x
                new += [(w, b, x), (b, u, x)]
            drop.add(fi); n_split += 1
        if not drop:
            break
        keep = np.ones(len(faces), bool); keep[list(drop)] = False
        tm = trimesh.Trimesh(tm.vertices, np.vstack([faces[keep], np.array(new, np.int64)]), process=False)
    return tm, n_split

def verify_and_repair(v, f):
    """Reload as an STL reader would (merge by position).  If that breaks watertightness, drop
    only faces whose corners merged (repeated indices) and exact duplicates, close zero-area
    T-junction slits, and keep the main body.  Needle faces with distinct corners are kept:
    removing them would open the slit they close."""
    import trimesh
    tm = trimesh.Trimesh(np.asarray(v, np.float32).astype(np.float64), f, process=True)
    if tm.is_watertight and len(tm.split(only_watertight=False)) == 1:
        return v, f, 0
    n0 = len(tm.faces)
    F = tm.faces
    tm.update_faces((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]))
    tm.update_faces(tm.unique_faces()); tm.remove_unreferenced_vertices()
    if not tm.is_watertight:
        tm, _ = split_t_junctions(tm)
    if not tm.is_watertight:
        trimesh.repair.fill_holes(tm)
    big = max(tm.split(only_watertight=False), key=lambda p: len(p.faces))
    return np.asarray(big.vertices), np.asarray(big.faces, np.int64), n0 - len(big.faces)

# =====================================================================
#  KIT ENTRY POINT (called by solar_suv_v4_1to20.py --part kit)
# =====================================================================
BOM = [
    ("M2 x 10 pan-head self-tapping screw", 4, "front clip x2, rear clip x2 (bumper corners), through the chassis from below"),
    ("M2 x 8 pan-head self-tapping screw", 2, "cabin floor into the pack side frames (rear footwells)"),
    ("M3 x 10 pan-head self-tapping screw", 4, "front and rear clips at the rocker ends, through the chassis from below"),
    ("3 x 2 mm N52 disc magnet", 44, "doors 24 (6 pairs per side), hood 8, roof canopy 4, cargo deck 8; mind the polarity of each pair"),
    ("3 mm steel or carbon rod, 88.25 mm", 2, "axles: slide through knuckle - drive unit - knuckle, then press the wheels on"),
    ("0603 SMD LED with 0.1 mm magnet wire (optional)", 20, "6+6 headlight windows, light blade, tail bar, corner units, cluster and centre display"),
    ("Printed locating pins (moulded on)", 18, "8 door pins on the rockers, 8 canopy pins on the door tops, 2 cabin-floor pegs"),
]

PRINT_NOTES = {
    "chassis": "flat underside on the bed, no supports, 0.12 mm layers, 3 walls",
    "interior": "cabin floor on the bed, no supports, 0.08-0.12 mm layers",
    "front": "upright on its bottom edge, no supports, bridging on (wheel-house roofs)",
    "rear": "upright on its bottom edge, no supports, bridging on",
    "door_fl": "upright on the bottom edge", "door_fr": "upright on the bottom edge",
    "door_rl": "upright on the bottom edge", "door_rr": "upright on the bottom edge",
    "hood": "flat underside on the bed", "canopy": "flat underside (belt plane) on the bed; 15 % gyroid infill",
    "cargo_deck": "flat, ribbed face up", "steering_wheel": "driver face down, peg up, 0.08 mm layers",
}

def assembly_preview(parts, info, path, max_tri=90000):
    """Colour GLB of the assembled kit (each node decimated to <= max_tri, for viewing and
    for tools/render_kit_figs.py).  Model mm, x rearward, z up from the print bed."""
    import trimesh
    sc = trimesh.Scene()
    def add(name, m, col):
        if m is None or m.is_empty():
            return
        m = decimate(m, 1.0 - max_tri/m.num_tri()) if m.num_tri() > max_tri else m
        mesh = m.scale([1/SCALE]*3).translate([0, 0, -GC/SCALE]).to_mesh()
        tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)
        tm.visual.face_colors = np.tile(np.r_[np.array(col)*255, 255].astype(np.uint8), (len(tm.faces), 1))
        sc.add_geometry(tm, node_name=name)
    for tag, m in info["chassis_groups"].items():
        add("chassis_" + tag, m, COL.get(tag, COL["chassis"]) if tag != "skin" else COL["body"])
    for tag, m in info["interior_groups"].items():
        add("cabin_" + tag, m, COL.get(tag, COL["trim"]))
    add("steering_wheel", parts["steering_wheel_placed"], (0.15, 0.15, 0.15))
    for k in ("front", "rear", "door_fl", "door_fr", "door_rl", "door_rr", "hood", "canopy", "cargo_deck"):
        add(k, parts[k], COL["body"] if k != "cargo_deck" else COL["trim"])
    sc.export(path)

def kit_main(out, res=6.0, mirrors="mirror", log=print):
    import os, trimesh, importlib.util, sys as _s
    os.makedirs(out, exist_ok=True)
    parts, info = build_kit(res, mirrors, log)
    files, checks = {}, {}
    for k in PRINT_PARTS:
        m = info["wheel_local"] if k == "steering_wheel" else parts[k]
        path = os.path.join(out, FILE[k] + ".stl")
        write_part(m, path)
        tm = trimesh.load(path)
        n, a, c = tm.face_normals, tm.area_faces, tm.triangles_center
        bed = (c[:, 2] < tm.bounds[0, 2] + 1e-3) & (n[:, 2] < -0.99)
        checks[FILE[k]] = dict(triangles=len(tm.faces), watertight=bool(tm.is_watertight), shells=len(tm.split(only_watertight=False)),
                               volume_cm3=round(tm.volume/1000, 2), size_mm=np.round(tm.bounds[1] - tm.bounds[0], 1).tolist(),
                               bed_contact_mm2=round(float(a[bed].sum()), 0), print=PRINT_NOTES[k], export=write_part.last)
        files[k] = path
        log(f"  {FILE[k]}: {checks[FILE[k]]['triangles']:,} tri, watertight {checks[FILE[k]]['watertight']}, shells {checks[FILE[k]]['shells']}")
    # layer-by-layer support check (tools/print_check.py)
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("print_check", os.path.join(here, "tools", "print_check.py"))
    pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
    for k, path in files.items():
        try:
            checks[FILE[k]]["support"] = pc.check(pc.load(path))["summary"]
        except ValueError as e:
            checks[FILE[k]]["support"] = dict(error=str(e))
            log(f"  ! support check skipped: {e}")
    # assembly interference (model mm3)
    names = [k for k in parts if k != "steering_wheel_placed"] + ["steering_wheel_placed"]
    inter = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = parts[names[i]], parts[names[j]]
            ba, bb = np.array(a.bounding_box()), np.array(b.bounding_box())
            if np.any(ba[3:] < bb[:3]) or np.any(bb[3:] < ba[:3]):
                continue
            v = a.intersect(b).volume()/SCALE**3
            if v > 0.01:
                inter[f"{names[i]} x {names[j]}"] = round(v, 2)
    checks["_interference_mm3"] = inter
    json.dump(checks, open(os.path.join(out, "kit_checks.json"), "w"), indent=1)
    pk = package_metrics()
    pk["hard_points_full_size_mm"] = dict(front_H_point=list(HP_F), rear_H_point=list(HP_R), heel_point=list(AHP), eye_point=list(EYE),
                                          steering_wheel_centre=list(SW_C), front_axle_x=X_FA, rear_axle_x=X_RA, wheel_centre_z=HUB_Z,
                                          floor_top_z=Z_FLOOR, pack_top_z=Z_PACK_TOP, split_planes_z=dict(chassis_body=Z_SKIRT, body_canopy=Z_CAN),
                                          hood_underside_z=info["z_hood"])
    pk["bill_of_materials"] = [dict(item=a, qty=b, where=c) for a, b, c in BOM]
    json.dump(pk, open(os.path.join(out, "packaging.json"), "w"), indent=1)
    assembly_preview(parts, info, os.path.join(out, "kit_assembly_preview.glb"))
    log(f"kit written to {os.path.abspath(out)}")
    return parts, info, checks
