# Solar SUV v4: 1:20 modular cutaway kit (print and assembly guide)

`python3 solar_suv_v4_1to20.py --out stl --part kit` writes the kit to `stl/kit/`:
12 part STLs, `kit_checks.json`, `packaging.json` and a colour `kit_assembly_preview.glb`.
It also writes the wheel and axle files to `stl/`. A full build takes about 6 minutes; `--res 12`
gives a quick draft in about 4 minutes. Check every export with:

```
python3 tools/check_stl.py stl/kit/*.stl stl/*.stl      # watertight, 2-manifold edges, shells, bed contact
python3 tools/print_check.py stl/kit/*.stl              # layer-by-layer bridges / cantilevers / islands
```

The engineering background (packaging, harness, thermal system and suspension) is in
[ENGINEERING_REPORT.md](ENGINEERING_REPORT.md).

## Parts

Volumes and sizes come from the full-resolution build (`stl/kit/kit_checks.json`).

| # | File | Part | Print orientation |
|---|---|---|---|
| 01 | `kit_01_chassis.stl` | Skateboard chassis: undertray, solid rockers, battery tray with 16 modules, BJB + BMS, crash rails and subframes, both e-axles with inverters, suspension, steering, front-end cooling module, heat-pump module, charging stack, HV harness, coolant and refrigerant lines | Flat underside on the bed |
| 02 | `kit_02_cabin_floor.stl` | Removable cabin: floor, 5 seats, IP with cluster and centre-display light boxes, steering column and stalks, pedals, HVAC case and blower, console, B-pillar trims with belt retractors | Floor on the bed |
| 03 | `kit_03_front_clip.stl` | Front body clip: bumper, lamp band, fenders, A-pillar bases, wheel-house awnings, hood ledge | Upright on its bottom edge |
| 04 | `kit_04_rear_clip.stl` | Rear body clip: quarters, C/D-pillar bases, liftgate with the tail bar, rear wheel houses | Upright on its bottom edge |
| 05-08 | `kit_05`...`kit_08` | Four doors with door cards (armrest, map pocket, speaker, handle recess, trim inlay) | Upright on the bottom edge |
| 09 | `kit_09_hood.stl` | Hood with the 72-cell PV array | Flat underside on the bed |
| 10 | `kit_10_roof_canopy.stl` | Greenhouse and roof with the 140-cell PV array; solid with printable cavities over the cabin | Belt plane on the bed |
| 11 | `kit_11_cargo_deck.stl` | Removable cargo floor over the rear e-axle | Flat |
| 12 | `kit_12_steering_wheel.stl` | Steering wheel with a 2 mm locating peg | Driver face down |
| - | `stl/wheel_1to20.stl` (x4), `stl/axles_x2_88.25mm.stl` | 275/45 R22 wheels with 400 mm rotors and 6-piston calipers; 3 mm axles | As v4 |

Every part is a watertight single shell with a flat bed face. Undersides are built at 46-50
degrees, or as bridges of about 12 mm or less, so no part needs supports. Turn bridge detection
on. The few remaining local overhangs of 2.15 mm or less are listed in the verification section of
the engineering report.

## Hardware

| Item | Qty | Where |
|---|---|---|
| M2 x 10 pan-head self-tapping screw | 4 | front and rear clips at the bumper corners, through the chassis from below |
| M2 x 8 pan-head self-tapping screw | 2 | cabin floor into the battery side frames (rear footwells) |
| M3 x 10 pan-head self-tapping screw | 4 | front and rear clips at the rocker ends, through the chassis from below |
| 3 x 2 mm N52 disc magnet | 44 | doors 24 (front clip, B-line and rear clip edges), hood 8, roof canopy 4, cargo deck 8 |
| 3 mm steel or carbon rod, 88.25 mm | 2 | axles (or print `axles_x2_88.25mm.stl`) |
| 0603 SMD LED with 0.1 mm enamelled wire (optional) | 20 | 6 + 6 headlight windows, light blade, tail bar, 2 corner units, cluster, centre display |

Printed pins are moulded on: 8 door pins on the rocker tops, 8 canopy pins on the door tops,
and 2 cabin-floor pegs on the battery front wall.

**Tolerances (model mm).**

| Feature | Size |
|---|---|
| Split gap between body parts | 0.10 mm per face |
| Sliding clearance on pins and pegs | 0.15 mm |
| Magnet pocket | 3.2 mm diameter x 2.1 mm, teardrop roof when horizontal |
| M2 hole | pilot 1.6 mm, clearance 2.3 mm, head counterbore 4.0 x 1.6 mm |
| M3 hole | pilot 2.5 mm, clearance 3.3 mm, head counterbore 5.8 x 2.0 mm |
| Door and canopy pins | 1.5 mm diameter x 2.0 mm, chamfered tip; sockets 1.8 mm diameter |
| Cabin-floor pegs | 2.0 mm diameter; floor holes 2.3 mm diameter |
| Steering-wheel peg | 2.0 mm diameter; column socket 2.15 mm diameter |

Glue each magnet with a drop of CA. Check the polarity of every pair against its mate before the glue sets.

## Assembly

1. **Chassis.** Press the four wheels onto the axles (as on v4). Slide each axle through
   knuckle, drive unit and knuckle.
2. **Cabin.** Drop the cabin floor onto the chassis: the two front holes go over the pegs on the
   battery front wall. Fix it with 2 x M2 x 8 in the rear footwells. Push the steering wheel
   peg into the column socket.
3. **Clips.** Lower the front and rear clips onto the chassis. Screw each from below with
   2 x M2 x 10 at the bumper corners and 2 x M3 x 10 at the rocker ends.
4. **Doors.** Set each door onto its two rocker pins. The edge magnets pull it against the clips
   and the B-line.
5. **Hood and canopy.** Place the hood on its ledge (4 magnet pairs). Set the canopy on the
   eight door-top pins; the rear magnets hold it on the rear clip.
6. **Cargo deck.** Drop it onto the four magnet posts.

## Cutaway levels (disassembly)

| Level | Remove | Reveals |
|---|---|---|
| 1 | Roof canopy: lift off | Cabin from above: seats, console, IP, displays, belt retractors |
| 2 | Doors: lift off their pins | Door cards and the cabin side view |
| 3 | Hood: lift off | Front bay: cooling module, heat-pump module, charging stack, MPPT, front inverter |
| 4 | Clips: 4 screws each from below | The whole body shell is off. The suspension, steering and harness are exposed |
| 5 | Cabin floor (2 screws) and cargo deck | Battery cell tray, BJB, BMS, centre duct busbars, coolant and refrigerant runs, rear e-axle |

## LED lighting (optional)

All channels are 4 mm teardrop tunnels with self-supporting roofs.

- **Headlights.** A channel runs behind the lamp band (front clip). Six 1.0 mm windows per side
  feed the matrix modules. The light blade is a row of gabled slits. A 3 mm wire exit leads back
  along the centre line.
- **Tail bar.** A channel runs behind the full-width bar in the liftgate (rear clip), with
  light slits that open into it, and a 3 mm exit forward.
- **Corner lamps.** These sit in the roof canopy, each with 1.0 mm windows. A 3 mm drop meets a
  3 mm socket in the rear clip.
- **Cluster and centre display.** Each is a light box behind a see-through window in the IP pod.
  A 4 mm channel runs down and forward to the IP front face.
- **Wire exits.** Three 4 mm pass-throughs in the undertray at the front corners and the rear
  centre lead the wires out under the model.

Thread 0.1 mm enamelled wire before the clips go on. Power the LEDs from outside the model
through the undertray exits, with a 100-150 ohm series resistor per LED on 3-5 V. There is no
room for a battery inside.
