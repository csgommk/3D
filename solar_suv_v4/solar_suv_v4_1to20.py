#!/usr/bin/env python3
# =====================================================================
#  SOLAR SUV v4 - PREMIUM ELECTRIC SUV WITH CURVED BODY-INTEGRATED PV
#  1:20 scale, watertight, 3D-printable.  Python CAD generator.
#
#  WHY PYTHON INSTEAD OF OPENSCAD
#  v1-v3 were OpenSCAD hulls of straight-sided cross-sections, so every
#  panel came out as a flat facet with 45/90-degree joints.  v4 builds the
#  body the way surface modellers do:
#    1. B-spline profile curves (side profile, plan, section, roof, glass).
#    2. An implicit "class-A" body: each surface family (lower body sides,
#       hood/deck, greenhouse sides, windscreen/roof, front and rear faces)
#       is a signed field driven by those curves; families are joined by
#       blended booleans, which act as constant-radius fillet sweeps along
#       every intersection (shoulder, roof rails, A/B/C/D pillars, header,
#       spoiler crest, bumper corners, beltline, cowl).
#    3. The field is meshed (marching cubes) at 0.3 mm model resolution and reduced by
#       quadric decimation (deviation well under 0.01 mm on the model), so the
#       triangles stay well shaped and reflections stay continuous.
#    4. Detail is engraved by manifold booleans at a CONSTANT DEPTH below the
#       curved skin (inset level sets of the same field), so solar cell gaps,
#       panel lines, glass and lamps follow the crowned hood, roof and
#       tailgate instead of sitting on flat planes.
#
#  Geometry is defined in full-size millimetres (x rearward from the front
#  bumper, y toward the car's right-hand side, z up from the road) and scaled by
#  1/SCALE on output.
#  Print tolerances are given in model millimetres.
#
#  Wheels are built separately as solids of revolution: 275/45 R22 tyre with a
#  printable tread, concave five-Y-spoke rim, slotted disc and sculpted caliper.
#
#  Requires: numpy, scipy, scikit-image, manifold3d, shapely; fast-simplification (optional)
#  Usage:    python3 solar_suv_v4_1to20.py --out stl              all parts (about 2 minutes)
#            python3 solar_suv_v4_1to20.py --res 12 --part body   quick draft of the body
#            --part all|body|split|hollow|wheel|pins|axles     --mirrors mirror|camera|none
#  Outputs:  body_1to20.stl (one piece), body_front/rear_1to20.stl (B-pillar split, 3 pins),
#            body_resin_hollow_1to20.stl (2.5 mm wall, drained), wheel_1to20.stl,
#            wheels_x4_1to20.stl, pins_x3.stl, axles_x2_88.25mm.stl
# =====================================================================
import argparse, math, os, time
import numpy as np
from scipy.interpolate import make_interp_spline
from skimage.measure import marching_cubes
from manifold3d import Manifold, Mesh, CrossSection, OpType, FillRule
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
try:                                   # quadric decimation keeps triangles well shaped (smooth shading)
    import fast_simplification
except ImportError:                    # optional: without it the full-resolution mesh is written
    fast_simplification = None

# ---------------------------------------------------------------------
#  PARAMETERS
# ---------------------------------------------------------------------
SCALE = 20                    # 1:20
# package (full-size mm)
L, H = 4800.0, 1720.0         # length, roof height
X_FA, X_RA = 890.0, 3840.0    # axles: 2950 mm wheelbase
GC = 210.0                    # ground clearance (flat floor)
TRACK = 1670.0                # tyre centre to tyre centre
TYRE = (275, 45, 22)          # 275/45 R22
TYRE_D = TYRE[2]*25.4 + 2*TYRE[0]*TYRE[1]/100.0
TYRE_R, TYRE_W, RIM_R = TYRE_D/2, float(TYRE[0]), TYRE[2]*25.4/2
HUB_Z = TYRE_R
TYRE_OUT = TRACK/2 + TYRE_W/2                 # outer face of the tyre
R_OPEN = TYRE_R + 46.0                        # arch opening radius (46 mm tyre clearance)
WS_ANGLE = 30.0                               # windscreen rake from horizontal
T46 = math.tan(math.radians(46))              # printable overhang slope

# print tolerances (model mm)
GROOVE_W   = 0.60   # solar cell gap width
CELL_D     = 1.10   # cell gap depth below the skin
POCKET_D   = 0.30   # PV laminate / glass / paint-mask pocket depth
SEAM_W     = 0.70   # panel line / door gap width
SEAM_D     = 1.00   # panel line / door gap / glass seal depth
LAMP_D     = 0.80   # lamp module and intake depth
AXLE_D, AXLE_CLEAR, AXLE_SNAP, HUB_PRESS = 3.0, 0.15, 0.25, 0.10
WELL_GAP   = 0.8
SPLIT_X    = 2520.0  # full-size x of the B-pillar split for small beds
PIN_D, PIN_CLEAR = 3.0, 0.15

def M(v):            # model mm -> full-size mm
    return v*SCALE

# ---------------------------------------------------------------------
#  CURVE TOOLS (cubic B-splines through design stations)
# ---------------------------------------------------------------------
class Curve:
    """Cubic B-spline y(x) through stations, with optional end slopes.
    Outside the station range the end tangents are continued linearly."""
    def __init__(self, pts, d0=None, d1=None):
        p = np.asarray(pts, float)
        bc = None
        if d0 is not None or d1 is not None:
            bc = ([(1, d0)] if d0 is not None else [(2, 0.0)],
                  [(1, d1)] if d1 is not None else [(2, 0.0)])
        self.s = make_interp_spline(p[:, 0], p[:, 1], k=3, bc_type=bc if bc else "natural")
        self.x0, self.x1 = p[0, 0], p[-1, 0]
        self.y0, self.y1 = float(self.s(self.x0)), float(self.s(self.x1))
        self.g0, self.g1 = float(self.s(self.x0, 1)), float(self.s(self.x1, 1))
    def __call__(self, x):
        x = np.asarray(x, float)
        xc = np.clip(x, self.x0, self.x1)
        y = self.s(xc)
        y = np.where(x < self.x0, self.y0 + (x - self.x0)*self.g0, y)
        y = np.where(x > self.x1, self.y1 + (x - self.x1)*self.g1, y)
        return y

def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0)/(e1 - e0), 0.0, 1.0)
    return t*t*(3 - 2*t)

def gauss(t, mu, sig):
    return np.exp(-0.5*((np.asarray(t, float) - mu)/sig)**2)

def smax(a, b, k):   # blended intersection = fillet of radius ~k
    h = np.clip(0.5 - 0.5*(b - a)/k, 0.0, 1.0)
    return b + (a - b)*h + k*h*(1 - h)

def smin(a, b, k):   # blended union = concave fillet of radius ~k
    h = np.clip(0.5 + 0.5*(b - a)/k, 0.0, 1.0)
    return b + (a - b)*h - k*h*(1 - h)

# ---------------------------------------------------------------------
#  DESIGN CURVES (full-size mm)
# ---------------------------------------------------------------------
# floor with 46-degree approach / departure chamfers (printable, also the skid plates)
Z_LIP_F, Z_LIP_R = 430.0, 560.0
XCH_F = (Z_LIP_F - GC)/T46
XCH_R = L - (Z_LIP_R - GC)/T46
def z_floor(x):
    return GC + np.maximum(0.0, XCH_F - x)*T46 + np.maximum(0.0, x - XCH_R)*T46

# front face profile x(z) at the centreline: lower lip, bumper nose, raked upper fascia
FRONT = Curve([(380, 66), (430, 40), (490, 14), (560, 3), (640, 0), (720, 2), (800, 10),
               (870, 25), (930, 48), (970, 80), (1000, 120), (1040, 190), (1100, 300)])
def bow_front(y):   # plan bow of the nose
    u = np.abs(y)/800.0
    return 60*u**2 + 45*u**4

# rear face profile x(z): diffuser lip, bumper, load ledge, tailgate, backlight (59 deg)
REAR = Curve([(380, 4735), (560, 4772), (640, 4794), (720, 4800), (790, 4794), (835, 4774),
              (862, 4748), (900, 4736), (1000, 4728), (1100, 4716), (1170, 4698), (1215, 4668),
              (1300, 4604), (1400, 4532), (1500, 4461), (1600, 4390), (1660, 4347), (1760, 4277)])
def bow_rear(y):
    u = np.abs(y)/800.0
    return 70*u**2 + 30*u**4

# lower-body section: offset of the side from the waist line, as a function of z
# (46-deg sill chamfer, tucked rocker, waist, tumblehome into the shoulder)
_SIDE = Curve([(250, -24), (300, -11), (360, -6.5), (440, -4.5), (540, -2), (660, 0),
               (770, 0), (870, -4), (960, -12), (1040, -26), (1100, -44), (1150, -68), (1210, -105)],
              d0=1.0/1.042)
def side_offset(z):
    z = np.asarray(z, float)
    s = _SIDE(z)
    return np.where(z < 250, -24 - (250 - z)*(1/1.042), s)   # 46.2 deg chamfer below 250

HW_WAIST = 973.0
def y_lower(x, z):
    """Half-width of the lower body at (x, z): waist + section + sculpting."""
    taper = 55*(1 - smoothstep(0, 700, x)) + 35*smoothstep(4150, 4800, x)
    flare_f = 16*gauss(x, X_FA, 520)*gauss(z, 760, 230)        # front fender flare
    haunch  = 30*gauss(x, X_RA, 600)*gauss(z, 880, 260)        # rear shoulder haunch
    zc = 560 + 70*smoothstep(1500, 3300, x)                      # rising lower-door light-catcher
    catcher = -10*gauss(z, zc, 48)*smoothstep(1350, 1650, x)*(1 - smoothstep(3150, 3450, x))
    return HW_WAIST - taper + flare_f + haunch + catcher + side_offset(z)

# hood centre and shoulder (fender top / beltline) heights
HOOD_C = Curve([(0, 975), (120, 990), (400, 1010), (700, 1028), (1000, 1045), (1150, 1052), (1500, 1060)])
SHOULDER = Curve([(0, 950), (140, 968), (450, 1003), (800, 1048), (1150, 1090), (1450, 1108),
                  (2500, 1118), (3400, 1128), (3900, 1146), (4300, 1152), (4650, 1150), (4850, 1140)])
def t_lower(x, y):
    """Top of the lower body: crowned, sculpted hood between rising fender lines; flat deck aft."""
    ay = np.abs(y)
    zc = HOOD_C(x)*(1 - smoothstep(1100, 1400, x)) + SHOULDER(x)*smoothstep(1100, 1400, x)
    w = smoothstep(380, 860, ay)
    dome = 9*gauss(ay, 0, 230)*smoothstep(250, 500, x)*(1 - smoothstep(950, 1150, x))
    return zc*(1 - w) + SHOULDER(x)*w - 16*(ay/900.0)**2 + dome

# greenhouse
Z_BELT = 1118.0
Z_WS0, X_WS0 = 1052.0, 1150.0
def x_ws_base(y):                     # windscreen base curves back toward the A-pillars
    return X_WS0 + 150*(np.abs(y)/760.0)**2
def t_ws(x, y):
    return Z_WS0 + (x - x_ws_base(y))*math.tan(math.radians(WS_ANGLE))
ROOF = Curve([(1900, 1690), (2300, 1712), (2700, 1720), (3200, 1713), (3700, 1693),
              (4100, 1668), (4300, 1652), (4600, 1630)])
def t_roof(x, y):
    return ROOF(x) - 50*(np.abs(y)/760.0)**2
GLASS_BASE = Curve([(1100, 845), (1600, 852), (2500, 856), (3300, 844), (3900, 812),
                    (4300, 772), (4750, 700)])
TUMBLE = math.tan(math.radians(15))
def y_glass(x, z):
    return GLASS_BASE(x) - (z - Z_BELT)*TUMBLE

# fillet radii (full-size mm)
K_PLAN_F, K_PLAN_R = 220.0, 200.0     # plan corner radii, front / rear
K_SHOULDER = 60.0                    # shoulder line and hood edges
K_HEADER = 60.0                      # windscreen-to-roof header
K_CREST = 42.0                       # roof spoiler crest
K_RAIL = 78.0                        # roof rails, A- and D-pillars
K_BELT = 42.0                        # concave beltline and cowl fillet
K_MOULD = 14.0                       # arch moulding blend into the body

# arch mouldings: a radial band that follows the sculpted side, inner edge on the arch line
MOULD_W, MOULD_P, MOULD_K = 72.0, 14.0, 16.0   # band width, protrusion, edge round
K_ARCH = 16.0                                  # rolled lip on the wheel-arch opening

# mirrors (door mounted).  The housing sits on a stem whose underside is a 46-degree
# plane rising out of the door, so the head needs no support when the body prints upright.
MIRROR = dict(c=(1615, 988, 1245), half=(80, 88, 60), r=30.0,
              stem=((1588, 930, 1150), (40, 70, 110), 26.0),
              root_y=930.0, root_z=1103.0)

# ---------------------------------------------------------------------
#  SIGNED FIELD (negative inside), evaluated on broadcast grids
# ---------------------------------------------------------------------
def _box_round(px, py, pz, c, half, r):
    qx = np.abs(px - c[0]) - (half[0] - r)
    qy = np.abs(py - c[1]) - (half[1] - r)
    qz = np.abs(pz - c[2]) - (half[2] - r)
    out = np.sqrt(np.maximum(qx, 0)**2 + np.maximum(qy, 0)**2 + np.maximum(qz, 0)**2)
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r

def _capsule(px, py, pz, a, b, r):
    a = np.asarray(a, float); b = np.asarray(b, float); ab = b - a
    t = ((px - a[0])*ab[0] + (py - a[1])*ab[1] + (pz - a[2])*ab[2])/ab.dot(ab)
    t = np.clip(t, 0, 1)
    return np.sqrt((px - a[0] - t*ab[0])**2 + (py - a[1] - t*ab[1])**2 + (pz - a[2] - t*ab[2])**2) - r

def field(x, y, z, mirrors="mirror"):
    """x: (nx,1,1), y: (1,ny,1), z: (1,1,nz) full-size mm -> (nx,ny,nz) field."""
    ay = np.abs(y)
    # --- lower body
    yl = y_lower(x, z)
    f_side  = ay - yl
    f_top   = z - t_lower(x, y)
    f_front = (FRONT(z) + bow_front(y)) - x
    f_rear  = x - (REAR(z) - bow_rear(y))
    lb = smax(f_side, f_front, K_PLAN_F)
    lb = smax(lb, f_rear, K_PLAN_R)
    lb = smax(lb, f_top, K_SHOULDER)
    # --- greenhouse
    g_top = z - smin(t_ws(x, y), t_roof(x, y), K_HEADER)
    gh = smax(g_top, f_rear, K_CREST)
    gh = smax(gh, ay - y_glass(x, z), K_RAIL)
    gh = np.maximum(gh, (Z_BELT - 110.0) - z)
    body = smin(lb, gh, K_BELT)
    # --- arch mouldings: band around each arch, standing MOULD_P proud of the sculpted side
    for xa in (X_FA, X_RA):
        rho = np.sqrt((x - xa)**2 + (z - HUB_Z)**2)
        band = np.abs(rho - (R_OPEN - 3 + MOULD_W/2)) - MOULD_W/2
        mould = smax(band, lb - MOULD_P, MOULD_K)            # lower body grown by MOULD_P
        body = smin(body, mould, K_MOULD)
    # --- door mirrors / camera pods
    if mirrors != "none":
        m = MIRROR
        c, half = m["c"], m["half"]
        if mirrors == "camera":
            c = (c[0], c[1] - 25, c[2] - 40); half = (half[0]*0.8, half[1]*0.7, half[2]*0.7)
        # housing: rounded superellipsoid, longer (in x) at the outer end, flat-ish glass face at the rear
        t = np.clip((ay - (c[1] - half[1]))/(2*half[1]), 0, 1)
        ax = half[0]*(0.78 + 0.22*t)
        ux = np.where(x > c[0], (x - c[0])/(ax*0.82), (x - c[0])/ax)
        e = 3.2
        sq = (np.abs(ux)**e + np.abs((ay - c[1])/half[1])**e + np.abs((z - c[2])/half[2])**e)**(1/e)
        hs = (sq - 1.0)*min(half)
        sc, sh, sr = m["stem"]
        stem = _box_round(x, ay, z, sc, sh, sr)
        # fill under the inner half of the housing so its underside is the 46-deg plane, not a flat ceiling
        fill = _box_round(x, ay, z, (c[0] - 8, c[1] - 0.25*half[1], c[2] - 90),
                          (0.82*half[0], 0.75*half[1], 90 + 0.6*half[2]), 26.0)
        mir = smin(smin(hs, stem, 30.0), fill, 26.0)
        plane = (m["root_z"] + (ay - m["root_y"])*T46) - z     # keep only above a 46-deg plane
        mir = smax(mir, plane, 10.0)
        body = smin(body, mir, 34.0)
    # --- wheel-arch openings with a rolled lip (the wheel house behind is cut by manifold)
    for xa, win in ((X_FA, WELL_IN_F), (X_RA, WELL_IN)):
        rho = np.sqrt((x - xa)**2 + (z - HUB_Z)**2)
        hole = np.maximum(rho - R_OPEN, (win + 60.0) - ay)    # inner end lies inside the cavity
        body = smax(body, -hole, K_ARCH)
    return body

def floor_field(x, z):
    """Flat floor + 46-deg chamfers as an exact (unit-gradient) distance."""
    ch = (x < XCH_F) | (x > XCH_R)
    return (z_floor(x) - z)/np.where(ch, math.sqrt(1 + T46*T46), 1.0)

# ---------------------------------------------------------------------
#  SAMPLING + MESHING
# ---------------------------------------------------------------------
BOUNDS = (-40.0, 4840.0, -1150.0, 1150.0, 196.0, 1764.0)

def sample(h, mirrors="mirror", log=print):
    x0, x1, y0, y1, z0, z1 = BOUNDS
    xs = np.arange(x0, x1 + h/2, h); ys = np.arange(y0, y1 + h/2, h); zs = np.arange(z0, z1 + h/2, h)
    nx, ny, nz = len(xs), len(ys), len(zs)
    log(f"  grid {nx} x {ny} x {nz} = {nx*ny*nz/1e6:.1f} M samples at {h} mm")
    vol = np.empty((nx, ny, nz), np.float32)
    Y = ys[None, :, None]; Z = zs[None, None, :]
    step = max(4, int(4e6//(ny*nz)))
    for i in range(0, nx, step):
        j = min(nx, i + step)
        vol[i:j] = field(xs[i:j, None, None], Y, Z, mirrors).astype(np.float32)
    # first-order distance correction: divide by |grad| so inset levels are true depths
    g = np.empty_like(vol)
    for i in range(0, nx, step):
        a, b = max(0, i - 1), min(nx, i + step + 1)
        blk = vol[a:b].astype(np.float32)
        gx, gy, gz = np.gradient(blk, h)
        gm = np.sqrt(gx*gx + gy*gy + gz*gz)
        g[i:min(nx, i + step)] = gm[i - a:i - a + min(nx, i + step) - i]
    g = np.maximum(g, 0.25)
    vol /= g
    del g
    for i in range(0, nx, step):          # hard floor after normalising: exact plane at z = GC
        j = min(nx, i + step)
        vol[i:j] = np.maximum(vol[i:j], floor_field(xs[i:j, None, None], Z).astype(np.float32))
    return vol, (xs[0], ys[0], zs[0]), h

def _to_manifold(v, f):
    mesh = Mesh(vert_properties=np.ascontiguousarray(v, np.float32),
                tri_verts=np.ascontiguousarray(f, np.uint32))
    man = Manifold(mesh)
    if man.status().name != "NoError":
        mesh.merge(); man = Manifold(mesh)
    return man

def mesh_level(vol, origin, h, level=0.0, reduce=0.75):
    """Marching cubes on one level set, then quadric decimation (removes `reduce` of the
    triangles; deviation stays below 0.05 full-size mm on these smooth surfaces)."""
    v, f, _, _ = marching_cubes(vol, level=level, spacing=(h, h, h), allow_degenerate=False,
                                gradient_direction="ascent")
    v = v + np.asarray(origin)[None, :]
    man = _to_manifold(v, f)
    if man.volume() < 0:
        f = f[:, ::-1]; man = _to_manifold(v, f)
    if reduce and fast_simplification is not None:
        v2, f2 = fast_simplification.simplify(v.astype(np.float64), f.astype(np.int64),
                                              target_reduction=reduce, agg=5)
        m2 = _to_manifold(v2, f2)
        if m2.status().name == "NoError" and abs(m2.volume() - man.volume()) < 1e-3*abs(man.volume()):
            man = m2
    return man

# ---------------------------------------------------------------------
#  MANIFOLD HELPERS
# ---------------------------------------------------------------------
def _frame(m, U, V, W, O):
    """Map local (u, v, w) of a manifold to world: p = O + u*U + v*V + w*W."""
    T = np.array([[U[0], V[0], W[0], O[0]],
                  [U[1], V[1], W[1], O[1]],
                  [U[2], V[2], W[2], O[2]]], float)
    return m.transform(T)

def cs_from_shapely(geom):
    polys = []
    gs = geom.geoms if hasattr(geom, "geoms") else [geom]
    for g in gs:
        if g.is_empty or g.geom_type != "Polygon":
            continue
        g = orient(g, 1.0)
        polys.append(np.asarray(g.exterior.coords)[:-1])
        for r in g.interiors:
            polys.append(np.asarray(r.coords)[:-1])
    return CrossSection(polys, FillRule.Positive) if polys else CrossSection()

def prism(geom, plane, w0, w1):
    """Extrude a 2D shapely shape between w0 and w1 along the plane normal.
    plane 'xy': (u, v) = (x, y), extrude along z
    plane 'xz': (u, v) = (x, z), extrude along y
    plane 'yz': (u, v) = (y, z), extrude along x"""
    cs = cs_from_shapely(geom)
    if cs.is_empty():
        return Manifold()
    m = Manifold.extrude(cs, w1 - w0)
    if plane == "xy":
        return _frame(m, (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, w0))
    if plane == "xz":
        return _frame(m, (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, w0, 0))
    if plane == "yz":
        return _frame(m, (0, 1, 0), (0, 0, 1), (1, 0, 0), (w0, 0, 0))
    raise ValueError(plane)

def cyl_y(xc, zc, r, y0, y1, seg=160):
    m = Manifold.cylinder(y1 - y0, r, r, seg)
    return _frame(m, (1, 0, 0), (0, 0, 1), (0, 1, 0), (xc, y0, zc))

# ---------------------------------------------------------------------
#  WHEEL HOUSES, ARCHES, AXLE SLOTS (full-size mm)
# ---------------------------------------------------------------------
WELL_IN = TYRE_OUT - TYRE_W - M(WELL_GAP)       # inner wheel-house wall (rear)
WELL_IN_F = WELL_IN - 40.0                      # front: room for steering lock
CAV_TOP = HUB_Z + R_OPEN + 80.0                 # wheel-house roof: ~100 mm bump travel

def cavity_shape(xa):
    """Arch opening (circle) + wheel-house section: 45-deg legs, flat top (short print bridge)."""
    c = Point(xa, HUB_Z).buffer(R_OPEN + 4, resolution=48)
    r = R_OPEN + 4
    t = r*math.sqrt(0.5)                                     # 45-deg tangent points on the circle
    roof = Polygon([(xa - t, HUB_Z + t), (xa, HUB_Z + r*math.sqrt(2)), (xa + t, HUB_Z + t)])   # gable
    shape = unary_union([c, roof]).intersection(box(xa - 2*r, GC - 80, xa + 2*r, CAV_TOP))
    return shape

def arch_cutters():
    """Wheel houses behind the skin.  The visible openings (with their rolled lips) are
    already part of the surface field, so only the hidden cavities are cut here."""
    cut = []
    for xa, win in ((X_FA, WELL_IN_F), (X_RA, WELL_IN)):
        for sgn in (1, -1):
            ya, yb = (win, 905.0) if sgn > 0 else (-905.0, -win)
            cut.append(prism(cavity_shape(xa), "xz", ya, yb))
    return Manifold.batch_boolean(cut, OpType.Add)

def axle_slots():
    cut = []
    for xa, win in ((X_FA, WELL_IN_F), (X_RA, WELL_IN)):
        a = M(AXLE_D/2 + AXLE_CLEAR)
        cut.append(Manifold.cube([2*a, 2*win + 40, HUB_Z + M(AXLE_D/2) - GC - M(0.8)])
                   .translate([xa - a, -win - 20, GC + M(0.8)]))
        b = M(AXLE_D - AXLE_SNAP)/2
        cut.append(Manifold.cube([2*b, 2*win + 40, 20 + M(0.81)]).translate([xa - b, -win - 20, GC - 20]))
    return Manifold.batch_boolean(cut, OpType.Add)

# ---------------------------------------------------------------------
#  SURFACE PROBES (scalar solvers on the design curves)
# ---------------------------------------------------------------------
TAN_WS = math.tan(math.radians(WS_ANGLE))
def t_gh(x, y):
    return smin(t_ws(x, y), t_roof(x, y), K_HEADER)
def y_top_edge(x):
    """Lateral position where the windscreen/roof meets the glass side plane."""
    y = 800.0
    for _ in range(30):
        y = float(y_glass(x, t_gh(x, y)))
    return y
def z_rail(x):
    z = 1600.0
    for _ in range(30):
        z = float(t_gh(x, y_glass(x, z)))
    return z
def x_apillar(z):
    x = 1800.0
    for _ in range(40):
        y = float(y_glass(x, z))
        x = float(x_ws_base(y) + (z - Z_WS0)/TAN_WS)
    return x
def x_dpillar(z):
    x = 4400.0
    for _ in range(40):
        x = float(REAR(z) - bow_rear(y_glass(x, z)))
    return x
def y_backlight(z):
    return float(y_glass(x_dpillar(z), z))

def _sm(pts, n=400, k=3):
    """Smooth a polyline with a parametric cubic B-spline through its points."""
    p = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = make_interp_spline(d, p, k=min(k, len(p) - 1))
    return s(np.linspace(0, d[-1], n))

def strip(coords, w, cap=2):
    return LineString(coords).buffer(w/2, cap_style=cap, join_style=1, resolution=8)

def ring(poly, w):
    """Groove of width w centred on the outline (overlaps the pocket it frames, so no thin fins)."""
    return poly.buffer(w/2, join_style=1).difference(poly.buffer(-w/2, join_style=1))

def flow_grid(outline, ulines, vlines, gw):
    """Curvilinear cell gaps: lines clipped to the outline plus the perimeter."""
    g = [ln for ln in (strip(c, gw) for c in ulines + vlines)]
    inner = unary_union(g).intersection(outline.buffer(gw/2))
    return unary_union([inner, outline.buffer(gw/2).difference(outline.buffer(-gw/2))])

# ---------------------------------------------------------------------
#  DETAIL GEOMETRY  (2D shapes in full-size mm -> constant-depth cutters)
# ---------------------------------------------------------------------
def hood_halfwidth(x):        # hood / fender shut line in plan
    return np.interp(x, [100, 300, 700, 1000, 1200], [700, 742, 790, 812, 820])

def details():
    """Return {category: [(prism_manifold, depth_mm_model), ...]}"""
    D = {}
    def add(cat, man, depth):
        D.setdefault(cat, []).append((man, depth))
    gw_cell, gw_seam = M(GROOVE_W), M(SEAM_W)

    # ---------------- HOOD solar array (z-prism, follows the crowned, sculpted hood) -------------
    def hood_xf(y):  return 330 + bow_front(y)*0.9 + 40*(y/700.0)**2
    def hood_xr(y):  return x_ws_base(y) - 95
    def hood_hw(x):  return hood_halfwidth(x) - 50
    # outline: front edge, side, rear edge, side
    yy = np.linspace(-1, 1, 61)
    yf = yy*hood_hw(hood_xf(0)); front = [(hood_xf(y), y) for y in yf]
    yr = yy*hood_hw(hood_xr(0)); rear = [(hood_xr(y), y) for y in yr]
    def side_pts(sgn):
        pts = []
        for x in np.linspace(hood_xf(hood_hw(400)), hood_xr(hood_hw(1100)), 60):
            pts.append((x, sgn*hood_hw(x)))
        return pts
    hood_poly = Polygon(front + side_pts(1) + rear[::-1] + side_pts(-1)[::-1]).buffer(0)
    hood_poly = hood_poly.buffer(-30).buffer(30)          # soft corners
    nu, nv = 6, 12
    ul = []
    for i in range(1, nu):
        u = i/nu
        ul.append([(hood_xf(y) + u*(hood_xr(y) - hood_xf(y)), y) for y in np.linspace(-900, 900, 90)])
    vl = []
    for j in range(1, nv):
        v = -1 + 2*j/nv
        vl.append([(x, v*hood_hw(x)) for x in np.linspace(250, 1300, 90)])
    add("pv", prism(hood_poly, "xy", 900, 1200), POCKET_D)
    add("pvgap", prism(flow_grid(hood_poly, ul, vl, gw_cell), "xy", 900, 1200), CELL_D)

    # ---------------- ROOF solar array (z-prism, crowned roof, tapering aft) ----------------------
    xr0 = 2330.0
    def roof_hw(x):  return y_top_edge(x) - 112
    def roof_xf(y):  return xr0 + 110*(y/650.0)**2
    def roof_xr(y):  return 4170 - 70*(y/600.0)**2
    xs = np.linspace(xr0, 4170, 60)
    hw = np.array([roof_hw(x) for x in xs])
    hwf = lambda x: np.interp(x, xs, hw)
    front = [(roof_xf(y), y) for y in np.linspace(-1, 1, 61)*hwf(roof_xf(600))]
    rear = [(roof_xr(y), y) for y in np.linspace(-1, 1, 61)*hwf(roof_xr(500))]
    side = lambda s: [(x, s*hwf(x)) for x in np.linspace(roof_xf(hwf(2400)), roof_xr(hwf(4100)), 80)]
    roof_poly = Polygon(front + side(1) + rear[::-1] + side(-1)[::-1]).buffer(0).buffer(-60).buffer(60)
    nu, nv = 14, 10
    ul = [[(roof_xf(y) + (i/nu)*(roof_xr(y) - roof_xf(y)), y) for y in np.linspace(-900, 900, 90)] for i in range(1, nu)]
    vl = [[(x, (-1 + 2*j/nv)*hwf(x)) for x in np.linspace(2200, 4300, 90)] for j in range(1, nv)]
    add("pv", prism(roof_poly, "xy", 1580, 1800), POCKET_D)
    add("pvgap", prism(flow_grid(roof_poly, ul, vl, gw_cell), "xy", 1580, 1800), CELL_D)

    # ---------------- TAILGATE solar array (x-prism from behind, wraps with the tailgate) --------
    def tg_hw(z): return np.interp(z, [905, 1150], [600, 640])
    zz = np.linspace(905, 1148, 40)
    tg_poly = Polygon([(tg_hw(z), z) for z in zz] + [(-tg_hw(z), z) for z in zz[::-1]]).buffer(-25).buffer(25)
    ul = [[(y, 905 + (1148 - 905)*f + 8*(y/600.0)**2) for y in np.linspace(-700, 700, 60)] for f in (1/3, 2/3)]
    vl = [[(v*tg_hw(z), z) for z in np.linspace(880, 1170, 20)] for v in np.linspace(-1, 1, 15)[1:-1]]
    add("pv", prism(tg_poly, "yz", 4550, 4900), POCKET_D)
    add("pvgap", prism(flow_grid(tg_poly, ul, vl, gw_cell), "yz", 4550, 4900), CELL_D)

    # ---------------- GLASS: windscreen, side DLOs, rear window + seals --------------------------
    mg = 70.0
    xs = np.linspace(x_ws_base(0) + 40, 2230, 50)
    ws_side = [(x, y_top_edge(x) - mg) for x in xs]
    ws_poly = Polygon([(x_ws_base(y) + 40, y) for y in np.linspace(-1, 1, 41)*ws_side[0][1]][::-1]
                      + [(x, -y) for x, y in ws_side] + [(2230 - 30*(y/600)**2, y) for y in np.linspace(-1, 1, 41)*ws_side[-1][1]]
                      + [(x, y) for x, y in ws_side[::-1]]).buffer(0).buffer(-40).buffer(40)
    add("glass", prism(ws_poly, "xy", 1040, 1800), POCKET_D)
    add("seal", prism(ring(ws_poly, gw_seam*0.9), "xy", 1040, 1800), SEAM_D)

    z_lo = Z_BELT + 38
    zs = np.linspace(z_lo, 1700, 60)
    a_edge = [(x_apillar(z) + mg, z) for z in zs]
    d_edge = [(x_dpillar(z) - mg - 30, z) for z in zs]
    xs = np.linspace(1350, 4500, 120)
    top = [(x, z_rail(x) - K_RAIL - 8) for x in xs]
    topf = lambda x: np.interp(x, xs, [p[1] for p in top])
    xa = lambda z: np.interp(z, [p[1] for p in a_edge], [p[0] for p in a_edge])
    xd = lambda z: np.interp(z, [p[1] for p in d_edge], [p[0] for p in d_edge])
    # daylight opening: beltline, roof rail and either a fixed x or the A-/D-pillar edge at each end
    def dlo(x0, x1, front_fn=None, rear_fn=None):
        pts = []
        xa0 = front_fn(z_lo) if front_fn else x0
        pts.append((xa0, z_lo))
        xb0 = rear_fn(z_lo) if rear_fn else x1
        pts.append((xb0, z_lo))
        if rear_fn:
            for z in np.linspace(z_lo, 1700, 50):
                if z < topf(rear_fn(z)):
                    pts.append((rear_fn(z), z))
        else:
            pts.append((x1, topf(x1)))
        for x in np.linspace(pts[-1][0], x0 if not front_fn else 1400, 60):
            if front_fn and x < front_fn(topf(x)):
                break
            pts.append((x, topf(x)))
        if front_fn:
            for z in np.linspace(min(topf(x0 + 300), 1700), z_lo, 50):
                if z <= topf(front_fn(z)):
                    pts.append((front_fn(z), z))
        else:
            pts.append((x0, topf(x0)))
        return Polygon(pts).buffer(0).buffer(-30).buffer(30)
    panes = [dlo(1400, 2465, front_fn=xa), dlo(2575, 3455), dlo(3590, 4300, rear_fn=xd)]
    band = dlo(1400, 4300, front_fn=xa, rear_fn=xd)
    pillars = band.difference(unary_union(panes).buffer(-3.0))     # overlaps the panes: no shared walls
    pillars = unary_union([g for g in getattr(pillars, "geoms", [pillars]) if g.area > 2000.0])
    for sgn in (1, -1):
        y0, y1 = (560, 1150) if sgn > 0 else (-1150, -560)     # well inside the tumblehome; the windscreen
                                                                # always lies ahead of the A-pillar line
        for pn in panes:
            add("glass", prism(pn, "xz", y0, y1), POCKET_D)
            add("seal", prism(ring(pn, gw_seam*0.9), "xz", y0, y1), SEAM_D)
        add("pillar", prism(pillars, "xz", y0, y1), POCKET_D)   # gloss-black B/C pillar appliques
    zs = np.linspace(1252, 1605, 40)
    rw = [(y_backlight(z) - mg - 18, z) for z in zs]
    rw_poly = Polygon([(y, z) for y, z in rw] + [(-y, z) for y, z in rw[::-1]]).buffer(-35).buffer(35)
    add("glass", prism(rw_poly, "yz", 4250, 4900), POCKET_D)
    add("seal", prism(ring(rw_poly, gw_seam*0.9), "yz", 4250, 4900), SEAM_D)

    # ---------------- DOOR, HOOD, LIFTGATE, CHARGE-PORT LINES ------------------------------------
    zb = Z_BELT - 8
    front_edge = [(1442, 345), (1442, 820), (1430, 960), (1418, zb)]
    b_line = [(2520, 345), (2520, zb)]
    xc, zc0, rho = X_RA, HUB_Z, 572.0                  # 54 mm outside the arch moulding
    Q = np.array([3505.0, zb])
    th = np.linspace(0, math.pi/2, 2000)
    T = np.c_[xc - rho*np.cos(th), zc0 + rho*np.sin(th)]
    tang = np.abs(np.einsum("ij,ij->i", Q - T, T - np.array([xc, zc0])))
    i1 = int(np.argmin(tang))                          # tangent point of the line from the belt
    arc = [tuple(T[i]) for i in np.linspace(0, i1, 40).astype(int)]
    rear_edge = [(xc - rho, 345.0)] + arc + [tuple(Q)]
    bottom = [(1442, 352), (X_RA - 572.0, 352)]
    side_lines = [strip(_sm(front_edge, 80), gw_seam), strip(b_line, gw_seam),
                  strip(rear_edge, gw_seam), strip(bottom, gw_seam)]
    for sgn in (1, -1):
        y0, y1 = (650, 1150) if sgn > 0 else (-1150, -650)
        add("seam", prism(unary_union(side_lines), "xz", y0, y1), SEAM_D)
        # flush pop-out handles: outline + finger pocket
        for hx in (2290, 3080):
            hz = 980 if hx < 2500 else 992
            hp = box(hx, hz - 20, hx + 160, hz + 20).buffer(14).buffer(-14)
            add("seam", prism(ring(hp, gw_seam*0.85), "xz", y0, y1), SEAM_D)
            add("trim", prism(box(hx + 112, hz - 12, hx + 148, hz + 12), "xz", y0, y1), POCKET_D)
        # sill cladding paint mask between the arch mouldings (starts 0.8 mm above the bed so the
        # prism never reaches the floor's own skin layer)
        add("trim", prism(box(1428, 226, 3290, 330).buffer(18).buffer(-18), "xz", y0, y1), POCKET_D)
    # charge-port door (right front fender, +y)
    cp = box(1200, 925, 1316, 991).buffer(16).buffer(-16)
    add("seam", prism(ring(cp, gw_seam*0.85), "xz", 650, 1150), SEAM_D)

    # hood shut lines: front edge just behind the leading-edge round, fender lines, cowl
    hx_f = lambda y: 158 + bow_front(y)
    hl = [(hx_f(y), y) for y in np.linspace(-1, 1, 41)*hood_halfwidth(160)]
    sd = [(x, hood_halfwidth(x)) for x in np.linspace(160, x_ws_base(hood_halfwidth(1150)) - 40, 50)]
    cw = [(x_ws_base(y) - 40, y) for y in np.linspace(1, -1, 41)*hood_halfwidth(1200)]
    loop = hl + sd + cw + [(x, -y) for x, y in sd[::-1]]
    add("seam", prism(LineString(loop + [loop[0]]).buffer(gw_seam/2, join_style=1), "xy", 900, 1200), SEAM_D)

    # liftgate: D-pillar lines, sides of the tailgate, bumper line (x-prism) + roof hinge line (z-prism)
    zs = np.linspace(1240, 1630, 30)
    dl = [(y_backlight(z) - 34, z) for z in zs]
    tg_side = [(690, 880), (690, 1100), (0.5*(690 + dl[0][0]) + 10, 1185), dl[0]]
    left = _sm(tg_side + dl[1:], 120)
    lines = [strip(left, gw_seam), strip(np.c_[-left[:, 0], left[:, 1]], gw_seam),
             strip([(-690, 880), (690, 880)], gw_seam)]
    add("seam", prism(unary_union(lines), "yz", 4300, 4900), SEAM_D)
    # roof hinge line, turning rearward at its ends to meet the tops of the D-pillar lines
    y_end = dl[-1][0]
    x_end = float(REAR(dl[-1][1]) - bow_rear(y_end))
    hinge = [(4236 - 30*(y/600)**2 + (x_end - 4236 + 30*(y_end/600)**2)*float(smoothstep(0.80*y_end, y_end, abs(y))), y)
             for y in np.linspace(-1, 1, 81)*y_end]
    add("seam", prism(strip(_sm(hinge, 200), gw_seam), "xy", 1580, 1800), SEAM_D)

    # ---------------- FRONT: black-mask lamp band, light blade, matrix LEDs, bumper, intake ---------
    mask = Polygon([(-864, 900), (-838, 876), (838, 876), (864, 900), (856, 942), (812, 962),
                    (-812, 962), (-856, 942)]).buffer(-8).buffer(8)
    add("lamphousing", prism(mask, "yz", -20, 420), POCKET_D)                 # gloss-black mask
    add("lamp", prism(box(-800, 938, 800, 950).buffer(5).buffer(-5), "yz", -20, 420), LAMP_D)   # light blade
    for sgn in (1, -1):
        mods = [box(sgn*(470 + i*58) - 17, 894, sgn*(470 + i*58) + 17, 926) for i in range(6)]
        add("lamp", prism(unary_union(mods), "yz", -20, 420), LAMP_D)          # matrix LED modules
    # bumper cover split line under the lamp band, wrapping round the corners to the front arches
    add("seam", prism(strip([(-890, 852), (890, 852)], gw_seam), "yz", -20, 420), SEAM_D)
    for sgn in (1, -1):
        y0, y1 = (820, 1150) if sgn > 0 else (-1150, -820)
        add("seam", prism(strip([(150, 852), (615, 852)], gw_seam), "xz", y0, y1), SEAM_D)
    intake = Polygon([(-650, 452), (650, 452), (575, 604), (-575, 604)]).buffer(40).buffer(-40)
    hexes = []
    r_hex, web = 24.0, 18.0
    dx, dz = math.sqrt(3)*(r_hex + web/1.732), 1.5*(r_hex + web/1.732)
    for j in range(-1, 8):
        for i in range(-18, 19):
            cx, cz = i*dx + (dx/2 if j % 2 else 0), 452 + j*dz
            hexes.append(Polygon([(cx + r_hex*math.cos(math.radians(a)), cz + r_hex*math.sin(math.radians(a)))
                                  for a in range(30, 390, 60)]))
    mesh_holes = unary_union(hexes).intersection(intake.buffer(-22))
    add("intake", prism(mesh_holes, "yz", -20, 420), SEAM_D)
    add("grille", prism(intake, "yz", -20, 420), POCKET_D)                    # black mesh webs
    add("seam", prism(ring(intake, gw_seam*0.9), "yz", -20, 420), SEAM_D)
    for sgn in (1, -1):                                         # air curtains at the bumper corners
        cur = Polygon([(sgn*700, 470), (sgn*742, 470), (sgn*760, 720), (sgn*718, 720)]).buffer(14).buffer(-14)
        add("intake", prism(cur, "yz", -20, 420), SEAM_D)
    add("skid", prism(box(-560, 236, 560, 410).buffer(40).buffer(-40), "yz", -20, 400), POCKET_D)   # skid plate

    # ---------------- REAR: wraparound light bar, plate, reflectors, diffuser ---------------------
    bar = box(-905, 1174, 905, 1210)
    corner_units = [box(700, 1110, 905, 1210), box(-905, 1110, -700, 1210)]
    tail = unary_union([bar] + corner_units).buffer(10).buffer(-10)
    add("tail", prism(tail, "yz", 4520, 4900), LAMP_D)
    for sgn in (1, -1):                                         # wrap onto the rear quarters
        y0, y1 = (560, 1150) if sgn > 0 else (-1150, -560)
        add("tail", prism(box(4585, 1174, 4840, 1210).buffer(10).buffer(-10), "xz", y0, y1), LAMP_D)
        add("tail", prism(box(sgn*610 - 70, 600, sgn*610 + 70, 620), "yz", 4600, 4900), LAMP_D)  # reflectors
    add("plate", prism(box(-262, 646, 262, 756).buffer(10).buffer(-10), "yz", 4600, 4900), POCKET_D)
    add("trim", prism(box(-640, 236, 640, 540).buffer(50).buffer(-50), "yz", 4400, 4900), POCKET_D)
    fins = unary_union([box(c - 20, 250, c + 20, 515) for c in (-320, -160, 0, 160, 320)])
    add("trimgroove", prism(fins, "yz", 4400, 4900), LAMP_D)
    return D

# ---------------------------------------------------------------------
#  BODY ASSEMBLY
# ---------------------------------------------------------------------
def build_body(h=6.0, mirrors="mirror", reduce=0.75, detail=True, hollow_wall=None, log=print):
    """Mesh the body skin, cut the wheel houses and axle slots, and engrave every detail at
    a constant depth below the curved skin.  Returns (manifold in model mm, {original_id: tag}, cavity)
    where cavity is the skin offset `hollow_wall` model-mm inward (for the resin version) or None."""
    t0 = time.time()
    vol, org, h = sample(h, mirrors, log)
    base = mesh_level(vol, org, h, 0.0, reduce).as_original()
    tags = {base.original_id(): "paint"}
    log(f"  skin: {base.num_tri():,} triangles ({time.time() - t0:.0f} s)")
    build_body.last_skin = base.translate([0, 0, -GC]).scale([1/SCALE]*3)   # kept for depth checks
    D = details() if detail else {}
    depths = sorted({d for items in D.values() for _, d in items})
    insets = {}
    for d in depths:                                   # the skin offset inward by d (model mm)
        insets[d] = mesh_level(vol, org, h, -M(d), reduce)
    cavity = mesh_level(vol, org, h, -M(hollow_wall), 0.9) if hollow_wall else None
    del vol
    log(f"  inset skins at {', '.join(f'{d:.1f}' for d in depths)} mm ({time.time() - t0:.0f} s)")
    cutters = []
    for cat, items in D.items():
        groups = {}
        for man, d in items:
            groups.setdefault(d, []).append(man)
        for d, mans in groups.items():
            p = Manifold.batch_boolean(mans, OpType.Add).as_original()
            floor_ = insets[d].as_original()
            tags[p.original_id()] = cat + ":wall"
            tags[floor_.original_id()] = cat
            cutters.append(p - floor_)                 # constant-depth recess under the curved skin
    wells = Manifold.batch_boolean([arch_cutters(), axle_slots()], OpType.Add).as_original()
    tags[wells.original_id()] = "well"
    body = Manifold.batch_boolean([base, wells] + cutters, OpType.Subtract)
    body = body.translate([0, 0, -GC]).scale([1/SCALE]*3)          # model mm, floor at z = 0
    log(f"  body: {body.num_tri():,} triangles, volume {body.volume()/1000:.0f} cm3 ({time.time() - t0:.0f} s)")
    return body, tags, cavity

def hollow_body(body, cavity):
    """Resin version: one connected cavity inside the skin (from build_body), kept away from the
    wheel houses, axle slots and mirrors, drained by two 3.2 mm holes through the floor."""
    guard = []
    for xa, win in ((X_FA, WELL_IN_F), (X_RA, WELL_IN)):
        x0, x1 = xa - R_OPEN - 130, xa + R_OPEN + 130
        for sg in (1, -1):                            # around each wheel house
            y0 = win - 130 if sg > 0 else -1200.0
            guard.append(Manifold.cube([x1 - x0, 1200 - (win - 130), CAV_TOP - GC + 140])
                         .translate([x0, y0, GC - 10]))
        guard.append(Manifold.cube([300, 2400, HUB_Z + 170 - GC]).translate([xa - 150, -1200, GC - 10]))  # axle slot
    for sg in (1, -1):                                # mirror stems and housings stay solid
        y0 = 820.0 if sg > 0 else -1200.0
        guard.append(Manifold.cube([300, 380, 330]).translate([1460, y0, 1020]))
    cav = Manifold.batch_boolean([cavity] + guard, OpType.Subtract)
    cav = cav.translate([0, 0, -GC]).scale([1/SCALE]*3)
    drains = [Manifold.cylinder(8.0, 1.6, 1.6, 32).translate([x/SCALE, 0, -1]) for x in (1900.0, 3250.0)]
    return Manifold.batch_boolean([body, cav] + drains, OpType.Subtract)

# ---------------------------------------------------------------------
#  SPLIT BODY (model mm) + PINS + AXLES
# ---------------------------------------------------------------------
SPLIT_PINS = [(-420.0, 600.0), (420.0, 600.0), (0.0, 1400.0)]   # full-size (y, z) on the split face

def teardrop(cx, cz, d):
    r = d/2
    return unary_union([Point(cx, cz).buffer(r, resolution=24),
                        Polygon([(cx - r*math.sqrt(0.5), cz + r*math.sqrt(0.5)), (cx, cz + r*math.sqrt(2)),
                                 (cx + r*math.sqrt(0.5), cz + r*math.sqrt(0.5))])])

def split_body(body):
    xs = SPLIT_X/SCALE
    front = body.trim_by_plane([-1, 0, 0], -xs)        # keep x <= xs
    rear = body.trim_by_plane([1, 0, 0], xs)           # keep x >= xs
    holes = []
    for y, z in SPLIT_PINS:
        td = teardrop(y/SCALE, (z - GC)/SCALE, PIN_D + 2*PIN_CLEAR)
        holes.append(prism(td, "yz", xs - 6.5, xs + 6.5))
    holes = Manifold.batch_boolean(holes, OpType.Add)
    return front - holes, rear - holes

def pins():
    """Three 3 mm alignment pins, 12.4 mm long, printed lying on a 0.4 mm flat."""
    out = []
    for i in range(3):
        p = Manifold.cylinder(12.4, PIN_D/2, PIN_D/2, 32).rotate([0, 90, 0]).translate([0, 0, PIN_D/2 - 0.4])
        p = p ^ Manifold.cube([20, 2*PIN_D, PIN_D]).translate([-4, -PIN_D, 0])
        out.append(p.translate([0, i*(PIN_D + 3), 0]))
    return Manifold.batch_boolean(out, OpType.Add)

def axles():
    """Two 3 mm axles printed lying on a 0.4 mm flat (or use 3 mm steel / carbon rod cut to length)."""
    out = []
    for i in range(2):
        a = Manifold.cylinder(AXLE_LEN, AXLE_D/2, AXLE_D/2, 32).rotate([0, 90, 0]).translate([0, 0, AXLE_D/2 - 0.4])
        a = a ^ Manifold.cube([AXLE_LEN + 2, 2*AXLE_D, AXLE_D]).translate([-1, -AXLE_D, 0])
        out.append(a.translate([0, i*(AXLE_D + 4), 0]))
    return Manifold.batch_boolean(out, OpType.Add)

# =====================================================================
#  WHEEL  (model mm; axis on z, inner face on the bed at z = 0, outer face up)
#  275/45 R22 tyre: crowned tread, four circumferential grooves with a 45-degree upper
#  flank, slanted shoulder blocks, rib sipes, rounded shoulders and a bulged sidewall.
#  22-inch concave five-Y-spoke alloy with bevelled spoke edges, a slotted brake disc,
#  a sculpted four-piston caliper behind the spokes, five lug nuts and a domed centre cap.
#  Everything below the face is solid, so the wheel prints without supports.
# =====================================================================
WR, WW, WRR = TYRE_R/SCALE, TYRE_W/SCALE, RIM_R/SCALE      # 20.16, 13.75, 13.97 mm
TREAD_D = 0.70                     # tread groove depth (model mm)
HUB_DEPTH = WW - 4.25              # blind press-fit bore from the inner face (+45-deg cone top)
AXLE_LEN = round(2*((TYRE_OUT - TYRE_W)/SCALE + HUB_DEPTH - 0.25), 2)
N_PITCH = 64                       # tread pitches around the tyre

def _bez(*P, n=12):
    P = [np.asarray(p, float) for p in P]
    t = np.linspace(0, 1, n)[:, None]
    deg = len(P) - 1
    out = 0
    for i, p in enumerate(P):
        out = out + math.comb(deg, i)*(1 - t)**(deg - i)*t**i*p
    return [tuple(map(float, q)) for q in out]

def revolve_rz(pts, seg=256):
    poly = orient(Polygon(pts).buffer(0), 1.0)
    cs = CrossSection([np.asarray(poly.exterior.coords)[:-1]], FillRule.Positive)
    return Manifold.revolve(cs, seg)

def extrude_xy(geom, z0, z1):
    return prism(geom, "xy", z0, z1)

def tyre_r(z):
    u = (np.asarray(z, float) - WW/2)/(WW/2 - 1.6)
    return WR - 0.20*np.clip(np.abs(u), 0, 1.3)**4

FACE = Curve([(3.3, WW - 2.45), (5.0, WW - 2.25), (6.5, WW - 2.02), (9.0, WW - 1.58),
              (11.5, WW - 1.14), (WRR - 0.55, WW - 0.80)])          # concave dish of the rim face

def tyre_profile():
    R, W, rr = WR, WW, WRR
    z_in, z_out = 2.2, W - 2.0
    r_in, r_out = float(tyre_r(z_in)), float(tyre_r(z_out))
    pts = [(rr - 0.3, 0.0), (r_in - 1.30, 0.0)]
    pts += _bez((r_in - 1.30, 0.0), (r_in - 0.08, 1.22), (r_in, z_in), n=10)[1:]        # 45-deg bed shoulder
    pts += [(float(tyre_r(z)), float(z)) for z in np.linspace(z_in, z_out, 40)[1:]]      # crowned tread
    pts += _bez((r_out, z_out), (r_out + 0.02, W - 0.55), (R - 1.0, W - 0.08), (R - 2.3, W - 0.02), n=14)[1:]
    pts += _bez((R - 2.3, W - 0.02), (R - 3.6, W + 0.04), (rr + 2.2, W - 0.05), (rr + 1.05, W - 0.70), n=16)[1:]
    pts += [(rr - 0.3, W - 0.70)]
    return pts

def rim_profile():
    rr, W = WRR, WW
    pts = [(0.0, 0.0), (rr, 0.0), (rr, W - 1.25), (rr + 0.95, W - 1.05)]
    pts += _bez((rr + 0.95, W - 1.05), (rr + 1.35, W - 0.70), (rr + 1.05, W - 0.30), (rr + 0.55, W - 0.28), n=10)[1:]
    pts += [(rr - 0.10, W - 0.36), (rr - 0.42, W - 0.62)]
    pts += [(float(r), float(FACE(r))) for r in np.linspace(rr - 0.55, 3.3, 44)]
    pts += [(3.05, W - 2.62), (2.78, W - 2.62), (2.62, W - 2.30)]                        # cap ring groove
    pts += [(float(r), float(W - 2.30 + 0.28*(1 - (r/2.62)**2))) for r in np.linspace(2.62, 0, 12)[1:]]
    return pts

def spokes_2d():
    """Five Y-spokes in plan: a stem from the hub that splits into two arms at r = 7.4."""
    r_hub, r_split, r_rim, spread = 2.6, 7.4, WRR - 0.2, math.radians(9.5)
    parts = [Point(0, 0).buffer(5.2, resolution=64)]
    for k in range(5):
        ph = math.radians(90 + 72*k)
        e = np.array([math.cos(ph), math.sin(ph)]); n = np.array([-e[1], e[0]])
        parts.append(Polygon([tuple(r_hub*e + 1.65*n), tuple(r_split*e + 1.40*n), tuple((r_split + 1.0)*e),
                              tuple(r_split*e - 1.40*n), tuple(r_hub*e - 1.65*n)]))
        s = (r_split - 0.4)*e
        for sg in (1, -1):
            a = ph + sg*spread
            p = np.array([math.cos(a), math.sin(a)])*r_rim
            d = (p - s)/np.linalg.norm(p - s); m = np.array([-d[1], d[0]])
            parts.append(Polygon([tuple(s + 1.05*m), tuple(p + 0.92*m), tuple(p + 2.0*d + 0.92*m),
                                  tuple(p + 2.0*d - 0.92*m), tuple(p - 0.92*m), tuple(s - 1.05*m)]))
    return unary_union(parts)

def caliper_sdf(X, Y, Z, phc, rc=11.15, hr=1.75, half_arc=3.95, z0=None, z1=None, rnd=0.45):
    r = np.sqrt(X*X + Y*Y)
    th = np.arctan2(Y, X) - phc
    th = (th + np.pi) % (2*np.pi) - np.pi
    zc, hz = 0.5*(z0 + z1), 0.5*(z1 - z0)
    q = np.stack([np.abs(r - rc) - (hr - rnd), np.abs(th)*rc - (half_arc - rnd), np.abs(Z - zc) - (hz - rnd)])
    d = np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(axis=0), 0) - rnd
    # recessed pad-window panel on the face and a bulge over the outer pistons
    qp = np.stack([np.abs(r - rc) - 0.55, np.abs(th)*rc - 2.6, np.abs(Z - z1) - 0.22])
    pad = np.linalg.norm(np.maximum(qp, 0), axis=0) + np.minimum(qp.max(axis=0), 0) - 0.15
    d = smax(d, -pad, 0.12)
    return d

def sdf_solid(f, lo, hi, h):
    xs = np.arange(lo[0], hi[0] + h, h); ys = np.arange(lo[1], hi[1] + h, h); zs = np.arange(lo[2], hi[2] + h, h)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = f(X, Y, Z).astype(np.float32)
    vol[0], vol[-1], vol[:, 0], vol[:, -1], vol[:, :, 0], vol[:, :, -1] = (1.0,)*6
    return mesh_level(vol, (xs[0], ys[0], zs[0]), h, 0.0, 0.6)

def build_wheel(caliper_angle=126.0):
    """Returns (wheel manifold in model mm, {original_id: tag})."""
    tags = {}
    W, rr = WW, WRR
    def tag(m, t):
        m = m.as_original(); tags[m.original_id()] = t; return m
    # ---- tyre
    tyre = tag(revolve_rz(tyre_profile()), "tyre")
    cut = []
    groove_l = [2.55, 5.35, 7.55, 10.35]
    for zl in groove_l:
        R0 = float(tyre_r(zl + 0.4))
        cut.append(revolve_rz([(R0 - TREAD_D, zl), (R0 + 0.8, zl), (R0 + 0.8, zl + 0.15 + TREAD_D + 0.8),
                               (R0 - TREAD_D, zl + 0.15)]))
    pitch = 360.0/N_PITCH
    def slot(z0, z1, depth, w, shear, phase):
        zc = 0.5*(z0 + z1)
        s = Manifold.cube([depth + 1.2, w, z1 - z0]).translate([WR - depth, -w/2, z0])
        s = s.transform(np.array([[1, 0, 0, 0], [0, 1, shear, -shear*zc], [0, 0, 1, 0]], float))
        return Manifold.batch_boolean([s.rotate([0, 0, phase + i*pitch]) for i in range(N_PITCH)], OpType.Add)
    cut.append(slot(0.65, groove_l[0] + 0.2, 0.60, 0.44, 0.25, 0.0))                 # inner shoulder blocks
    cut.append(slot(groove_l[3] + 0.6, W + 0.6, 0.60, 0.44, -0.25, 0.0))             # outer shoulder blocks
    for z0, z1 in ((groove_l[0] + 0.7, groove_l[1] + 0.1), (groove_l[1] + 0.7, groove_l[2] + 0.1),
                   (groove_l[2] + 0.7, groove_l[3] + 0.1)):
        cut.append(slot(z0, z1, 0.45, 0.30, -0.35, pitch/2))                           # rib sipes
    cut = tag(Manifold.batch_boolean(cut, OpType.Add), "tread")
    tyre = tyre - cut
    # ---- rim: concave face, Y-spoke windows, bevelled spoke edges
    rim = tag(revolve_rz(rim_profile()), "rim")
    win = Point(0, 0).buffer(rr - 0.55, resolution=128).difference(spokes_2d())
    win = win.buffer(-0.45, join_style=1).buffer(0.45, join_style=1)
    win_cut = tag(extrude_xy(win, W - 5.6, W + 1.0), "rimwall")
    skin = revolve_rz([(0.0, W + 2.0), (0.0, float(FACE(3.3)) - 0.25)] +
                      [(float(r), float(FACE(r)) - 0.25) for r in np.linspace(3.3, rr - 0.3, 40)] +
                      [(rr - 0.3, W + 2.0)])
    bevel = tag(extrude_xy(win.buffer(0.20, join_style=1), W - 5.6, W + 1.0) ^ skin, "rim")
    rim = Manifold.batch_boolean([rim, win_cut, bevel], OpType.Subtract)
    # ---- brake disc behind the spokes: friction ring with ten curved slots, hat groove
    disc = Point(0, 0).buffer(12.35, resolution=128).difference(Point(0, 0).buffer(5.0, resolution=64))
    disc = tag(extrude_xy(disc, W - 6.0, W - 4.31), "disc")
    dcut = [extrude_xy(Point(0, 0).buffer(6.65, resolution=96).difference(Point(0, 0).buffer(6.35, resolution=96)),
                       W - 4.52, W - 4.0)]
    for i in range(10):
        a0 = math.radians(i*36.0)
        arc = [(r*math.cos(a0 + 0.055*(r - 7.6)), r*math.sin(a0 + 0.055*(r - 7.6))) for r in np.linspace(7.7, 11.8, 12)]
        dcut.append(extrude_xy(LineString(arc).buffer(0.17, cap_style=1), W - 4.55, W - 4.0))
    disc = disc - tag(Manifold.batch_boolean(dcut, OpType.Add), "discwall")
    # ---- caliper (sculpted SDF block) in one window, top 0.45 mm behind the spoke faces
    phc = math.radians(caliper_angle)
    zc0, zc1 = W - 5.9, W - 3.05
    c = (11.15*math.cos(phc), 11.15*math.sin(phc))
    lo = (c[0] - 4.8137, c[1] - 4.8071, zc0 - 0.2213); hi = (c[0] + 4.8, c[1] + 4.8, zc1 + 0.2)
    cal = tag(sdf_solid(lambda X, Y, Z: caliper_sdf(X, Y, Z, phc, z0=zc0, z1=zc1), lo, hi, 0.06), "caliper")
    # ---- lug nuts and hub bore
    nuts = []
    for k in range(5):
        a = math.radians(90 + 36 + 72*k)
        zf = float(FACE(4.1))
        nuts.append(Manifold.cylinder(0.62, 0.50, 0.40, 6).translate([4.1*math.cos(a), 4.1*math.sin(a), zf - 0.3]))
    nuts = tag(Manifold.batch_boolean(nuts, OpType.Add), "rim")
    rb = (AXLE_D - HUB_PRESS)/2
    bore = Manifold.batch_boolean([Manifold.cylinder(HUB_DEPTH + 1, rb, rb, 32).translate([0, 0, -1]),
                                   Manifold.cylinder(rb, rb, 0.0, 32).translate([0, 0, HUB_DEPTH])], OpType.Add)
    wheel = Manifold.batch_boolean([tyre, rim, disc, cal, nuts], OpType.Add) - bore
    return wheel, tags

def wheels_x4(w):
    d = 2*WR + 4.0
    return Manifold.batch_boolean([w.translate([i*d, j*d, 0]) for i in (0, 1) for j in (0, 1)], OpType.Add)

# ---------------------------------------------------------------------
#  OUTPUT
# ---------------------------------------------------------------------
def write_stl(man, path, name="solar_suv_v4"):
    m = man.to_mesh()
    v = np.asarray(m.vert_properties)[:, :3].astype(np.float32)
    f = np.asarray(m.tri_verts).astype(np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())

def main(argv=None):
    ap = argparse.ArgumentParser(description="Solar SUV v4, 1:20 printable model generator")
    ap.add_argument("--out", default="stl", help="output folder")
    ap.add_argument("--res", type=float, default=6.0, help="field grid in full-size mm (6 = 0.3 mm on the model)")
    ap.add_argument("--part", default="all", choices=["all", "body", "split", "hollow", "wheel", "pins", "axles"])
    ap.add_argument("--mirrors", default="mirror", choices=["mirror", "camera", "none"])
    ap.add_argument("--no-detail", action="store_true", help="plain skin without engraved detail")
    ap.add_argument("--reduce", type=float, default=0.75, help="fraction of marching-cubes triangles removed by quadric decimation")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out = lambda n: os.path.join(a.out, n)
    t0 = time.time()
    if a.part in ("all", "body", "split", "hollow"):
        print(f"body at {a.res} mm grid ({a.res/SCALE:.2f} mm on the model)")
        body, _, cav = build_body(a.res, a.mirrors, a.reduce, not a.no_detail,
                                  hollow_wall=2.5 if a.part in ("all", "hollow") else None)
        if a.part in ("all", "body"):
            write_stl(body, out("body_1to20.stl"))
        if a.part in ("all", "split"):
            fr, rr_ = split_body(body)
            write_stl(fr, out("body_front_1to20.stl")); write_stl(rr_, out("body_rear_1to20.stl"))
        if a.part in ("all", "hollow"):
            write_stl(hollow_body(body, cav), out("body_resin_hollow_1to20.stl"))
    if a.part in ("all", "wheel"):
        w, _ = build_wheel()
        write_stl(w, out("wheel_1to20.stl")); write_stl(wheels_x4(w), out("wheels_x4_1to20.stl"))
    if a.part in ("all", "pins"):
        write_stl(pins(), out("pins_x3.stl"))
    if a.part in ("all", "axles"):
        write_stl(axles(), out(f"axles_x2_{AXLE_LEN:.2f}mm.stl"))
    print(f"done in {time.time() - t0:.0f} s -> {os.path.abspath(a.out)}")

if __name__ == "__main__":
    main()
