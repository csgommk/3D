#!/usr/bin/env python3
"""Power and thermal flow schematic for the Solar SUV v4 kit (writes SVG; PNG via render_webgl's Chromium).

Two lanes share the colour code of the CAD harness:
  orange = HV DC (400 V class), dark orange = 3-phase AC, yellow = solar strings (< 60 V SELV),
  black = 12 V / CAN, blue = coolant (cold), red = coolant (hot), green = refrigerant (R1234yf)."""
import sys
from types import SimpleNamespace

W, H = 1900, 1270
C = dict(hv="#ff7a00", ac="#c2410c", sol="#e0b000", lv="#222222", cold="#1593c8", hot="#e0402a", ref="#4caf32",
         box="#ffffff", edge="#30343a", text="#1d2125", muted="#5c6570", band="#f3f5f7")
out = []


def box(x, y, w, h, title, lines=(), fill=C["box"], edge=C["edge"]):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{edge}" stroke-width="2"/>')
    out.append(f'<text x="{x + w/2}" y="{y + 24}" text-anchor="middle" font-weight="700" font-size="17" fill="{C["text"]}">{title}</text>')
    for i, t in enumerate(lines):
        out.append(f'<text x="{x + w/2}" y="{y + 46 + 19*i}" text-anchor="middle" font-size="14" fill="{C["muted"]}">{t}</text>')
    return SimpleNamespace(x=x, y=y, w=w, h=h, cx=x + w/2, cy=y + h/2, l=x, r=x + w, t=y, b=y + h)


def wire(pts, col, label=None, width=4, dash=None, arrow=True, lab_at=0.5, lab_dy=-8):
    d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#a{col[1:]})"' if arrow else ""
    out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}"{da}{mk} stroke-linejoin="round"/>')
    if label:
        i = max(0, min(len(pts) - 2, int(lab_at*(len(pts) - 1))))
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        out.append(f'<text x="{(x0 + x1)/2}" y="{(y0 + y1)/2 + lab_dy}" text-anchor="middle" font-size="13" '
                   f'fill="{col if col != C["sol"] else "#8a6d00"}" font-weight="600">{label}</text>')


def text(x, y, t, col, size=13, weight=600, anchor="middle"):
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
               f'fill="{col if col != C["sol"] else "#8a6d00"}">{t}</text>')


def lane(y, h, title):
    out.append(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="14" fill="{C["band"]}"/>')
    out.append(f'<text x="40" y="{y + 30}" font-size="20" font-weight="700" fill="{C["text"]}">{title}</text>')


def build():
    out.clear()
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
               f'font-family="DejaVu Sans, Arial, sans-serif">')
    out.append("<defs>" + "".join(
        f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in C.values()) + "</defs>")
    out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    out.append(f'<text x="40" y="44" font-size="26" font-weight="700" fill="{C["text"]}">Solar SUV v4 - power and thermal flow</text>')
    out.append(f'<text x="40" y="70" font-size="15" fill="{C["muted"]}">Energy from the PV skin and the grid to the wheels and the cabin; '
               f'coolant and refrigerant from the front-end module to the battery, e-axles and HVAC. Values are full-size design figures.</text>')
    # ------------------------------------------------------------------ ENERGY LANE
    lane(90, 650, "1  Energy: solar + grid -> battery -> e-axles -> wheels (regen braking reverses the orange path)")
    hood = box(50, 150, 230, 78, "Hood PV", ["1.13 m2, 72 cells, ~40 V"])
    roof = box(50, 242, 230, 78, "Roof PV", ["2.11 m2, 2 x 70 cells, ~40 V"])
    tail = box(50, 334, 230, 78, "Tailgate PV", ["0.33 m2, 42 cells, ~24 V"])
    mppt = box(420, 210, 230, 120, "MPPT controller", ["3 inputs, 800 W, 97 %", "boost to 336-470 V", "front bay, right carrier"])
    grid = box(50, 470, 230, 92, "Grid / charger", ["AC 3-phase 11 kW", "DC fast 200 kW"])
    inlet = box(420, 470, 230, 92, "Charge inlet", ["right front fender", "AC + DC pins"])
    obc = box(760, 400, 230, 100, "OBC + DC-DC + PDU", ["11 kW AC -> DC", "2.5 kW 400 V -> 14 V"])
    bjb = box(1090, 230, 240, 120, "Battery junction box", ["2 main contactors", "pre-charge, 630 A pyro fuse", "current shunt"])
    pack = box(1090, 420, 240, 130, "Battery pack", ["112s2p, 224 NMC cells", "98.6 kWh gross / 94 net", "411 V nom., BMS + 16 CMU"])
    finv = box(1500, 150, 170, 90, "Front inverter", ["SiC, 450 A rms"])
    rinv = box(1700, 150, 170, 90, "Rear inverter", ["SiC, 600 A rms"])
    fmot = box(1500, 400, 170, 94, "Front e-axle", ["ASM 150 kW", "9.0 : 1, open diff"])
    rmot = box(1700, 400, 170, 94, "Rear e-axle", ["PMSM 250 kW", "9.3 : 1, open diff"])
    fwh = box(1500, 590, 170, 70, "Front wheels", ["275/45 R22"])
    rwh = box(1700, 590, 170, 70, "Rear wheels", ["275/45 R22"])
    lv12 = box(760, 560, 230, 82, "12 V LFP battery", ["40 Ah, ECUs, lights"])
    comp = box(1090, 600, 240, 82, "HV loads", ["e-compressor 6.5 kW", "coolant heater 5 kW"])
    for b_ in (hood, roof, tail):
        wire([(b_.r, b_.cy), (360, b_.cy), (360, mppt.cy), (mppt.l, mppt.cy)], C["sol"], None, 4)
    text(300, 140, "solar strings (SELV, < 60 V): A-pillar wire channels", "#8a6d00")
    wire([(mppt.r, mppt.cy - 20), (bjb.l, mppt.cy - 20)], C["hv"], None, 6); text(870, mppt.cy - 30, "MPPT output, 400 V class", C["hv"])
    wire([(grid.r, grid.cy), (inlet.l, grid.cy)], C["hv"], None, 6); text(350, grid.cy - 10, "cable", C["hv"])
    wire([(inlet.r, inlet.cy - 20), (obc.l, inlet.cy - 20)], C["ac"], None, 5); text(705, inlet.cy - 30, "AC 3-ph", C["ac"])
    wire([(inlet.r, inlet.cy + 25), (700, inlet.cy + 25), (700, 375), (1040, 375), (1040, bjb.b - 18), (bjb.l, bjb.b - 18)], C["hv"], None, 7)
    text(870, 365, "DC fast charge, 2 x 95 mm2", C["hv"])
    wire([(obc.r, obc.cy - 20), (1060, obc.cy - 20), (1060, bjb.b - 48), (bjb.l, bjb.b - 48)], C["hv"], None, 6)
    text(1020, obc.cy - 28, "10 mm2", C["hv"])
    wire([(obc.cx, obc.b), (obc.cx, lv12.t)], C["lv"], None, 4); text(obc.cx + 30, obc.b + 34, "14 V", C["lv"])
    wire([(bjb.cx, bjb.b), (bjb.cx, pack.t)], C["hv"], None, 8, arrow=False); text(bjb.cx + 70, bjb.b + 40, "busbars", C["hv"])
    wire([(bjb.r, bjb.t + 30), (1420, bjb.t + 30), (1420, finv.cy), (finv.l, finv.cy)], C["hv"], None, 7)
    text(1415, bjb.t + 20, "2 x 50 mm2", C["hv"], anchor="end")
    wire([(bjb.r, bjb.b - 30), (1440, bjb.b - 30), (1440, 280), (rinv.cx, 280), (rinv.cx, rinv.b)], C["hv"], None, 7)
    text(1445, bjb.b - 38, "2 x 70 mm2", C["hv"], anchor="start")
    wire([(finv.cx, finv.b), (finv.cx, fmot.t)], C["ac"], None, 6)
    wire([(rinv.cx + 30, rinv.b), (rinv.cx + 30, rmot.t)], C["ac"], None, 6)
    text(1685, 355, "3-phase U/V/W", C["ac"])
    wire([(fmot.cx, fmot.b), (fmot.cx, fwh.t)], C["edge"], None, 5)
    wire([(rmot.cx, rmot.b), (rmot.cx, rwh.t)], C["edge"], None, 5)
    text(1685, 545, "halfshafts", C["muted"])
    wire([(pack.cx, pack.b), (pack.cx, comp.t)], C["hv"], None, 5); text(pack.cx + 60, pack.b + 32, "PDU branch", C["hv"])
    text(40, 720, "Daily PV yield 2.3-2.6 kWh = 12-14 km at 19 kWh/100 km.  Pack 93.7 kWh usable = ~490 km.  "
                  "System peak 400 kW (150 + 250 kW).  Regen recovers up to 0.3 g into the pack.", C["muted"], size=15, weight=400, anchor="start")
    # ------------------------------------------------------------------ THERMAL LANE
    y0 = 760
    lane(y0, 490, "2  Thermal: front-end module, heat pump, battery and e-axle loops, cabin HVAC")
    fem = box(50, y0 + 60, 250, 110, "Front-end module", ["LT radiator + condenser", "2 x 300 mm fans"])
    valve = box(400, y0 + 60, 250, 110, "8-way coolant valve", ["2 pumps (25 / 20 L/min)", "reservoir, left carrier"])
    chill = box(760, y0 + 60, 240, 110, "Chiller", ["refrigerant / coolant", "battery cooling 8 kW"])
    batt = box(1100, y0 + 60, 240, 110, "Battery cold plates", ["16 modules, 18-35 C", "lines in the centre duct"])
    compb = box(400, y0 + 250, 250, 100, "e-compressor", ["34 cc scroll, R1234yf", "accumulator, EXVs"])
    wcc = box(760, y0 + 250, 240, 100, "Water-cooled condenser", ["heat-pump heat to coolant"])
    pt = box(1100, y0 + 250, 240, 100, "E-axles + inverters", ["water jackets, <= 65 C"])
    hvac = box(1440, y0 + 60, 400, 170, "Cabin HVAC module", ["blower 400 W, evaporator, heater core", "blend doors; defrost, 4 face, 2 foot vents",
                                                              "behind the IP centre stack", "modes: A/C, heat pump, battery chill,", "waste-heat recovery, pre-conditioning"])
    wire([(fem.r, fem.cy - 22), (valve.l, fem.cy - 22)], C["cold"], None, 5); text(350, fem.cy - 30, "cold", C["cold"])
    wire([(valve.l, fem.cy + 22), (fem.r, fem.cy + 22)], C["hot"], None, 5); text(350, fem.cy + 42, "hot", C["hot"])
    wire([(valve.r, valve.cy - 22), (chill.l, valve.cy - 22)], C["cold"], None, 5)
    wire([(chill.r, chill.cy - 22), (batt.l, chill.cy - 22)], C["cold"], None, 5); text(1050, chill.cy - 30, "supply", C["cold"])
    wire([(batt.cx, batt.b), (batt.cx, y0 + 195), (valve.cx + 40, y0 + 195), (valve.cx + 40, valve.b)], C["hot"], None, 5)
    text(1080, y0 + 189, "pack return", C["hot"], anchor="end")
    wire([(valve.cx - 40, valve.b), (valve.cx - 40, y0 + 215), (1060, y0 + 215), (1060, pt.t + 20), (pt.l, pt.t + 20)], C["cold"], None, 5)
    text(700, y0 + 209, "e-axle loop", C["cold"])
    wire([(pt.l, pt.cy + 20), (wcc.r, pt.cy + 20)], C["hot"], None, 5); text(1050, pt.cy + 42, "waste heat", C["hot"])
    wire([(compb.r, compb.cy), (wcc.l, compb.cy)], C["ref"], None, 5); text(705, compb.cy - 10, "hot gas", C["ref"])
    wire([(compb.l, compb.cy), (175, compb.cy), (175, fem.b)], C["ref"], None, 5); text(185, compb.cy - 10, "to front condenser (A/C)", C["ref"], anchor="start")
    wire([(chill.l + 30, chill.b), (chill.l + 30, y0 + 235), (compb.cx, y0 + 235), (compb.cx, compb.t)], C["ref"], None, 4, dash="10 6")
    text(700, y0 + 248, "suction", C["ref"])
    wire([(wcc.cx - 40, wcc.b), (wcc.cx - 40, y0 + 400), (hvac.cx - 60, y0 + 400), (hvac.cx - 60, hvac.b)], C["ref"], None, 5)
    text(1300, y0 + 394, "liquid line -> EXV -> evaporator", C["ref"])
    wire([(wcc.cx + 40, wcc.b), (wcc.cx + 40, y0 + 375), (hvac.cx + 60, y0 + 375), (hvac.cx + 60, hvac.b)], C["hot"], None, 5)
    text(1300, y0 + 369, "heater core loop", C["hot"])
    lx, ly = 60, y0 + 440
    for i, (col, name) in enumerate(((C["hv"], "HV DC"), (C["ac"], "3-phase AC"), (C["sol"], "solar strings"), (C["lv"], "12 V / CAN"),
                                     (C["cold"], "coolant cold"), (C["hot"], "coolant hot"), (C["ref"], "refrigerant"))):
        x = lx + (i % 4)*150; y = ly + (i//4)*26
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 26}" y2="{y}" stroke="{col}" stroke-width="5"/>')
        out.append(f'<text x="{x + 31}" y="{y + 5}" font-size="13" fill="{C["text"]}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def to_png(svg_path, png_path, scale=1.5):
    """Rasterise with the local headless Chromium (pip install playwright; no network needed)."""
    import glob
    from playwright.sync_api import sync_playwright
    chrome = (sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")) or [None])[-1]
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox", "--no-proxy-server"])
        pg = b.new_page(viewport=dict(width=W, height=H), device_scale_factor=scale)
        pg.set_content(f'<html><body style="margin:0">{open(svg_path).read()}</body></html>')
        pg.screenshot(path=png_path, full_page=False)
        b.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "power_thermal_flow.svg"
    open(path, "w").write(build())
    print("wrote", path)
    if "--png" in sys.argv:
        to_png(path, path[:-4] + ".png")
        print("wrote", path[:-4] + ".png")
