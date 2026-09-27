# Telekinesis CV

A real-time computer vision playground that turns hand gestures into virtual
telekinesis. Grab, throw, resize and rotate virtual crystals over your webcam
feed, using explicit landmark geometry and a lightweight physics simulation.

Built with **Python, OpenCV, MediaPipe and NumPy**. The project focuses on
understanding the mathematics after landmark detection: smoothing, gesture
thresholds, velocity estimation, coordinate transforms and collision response.
**No physical objects, browser, backend or physics engine are required.**

## Features

- Mirrored webcam feed with up to two tracked hands and 21 landmarks per hand.
- Smoothed fingertip pointers and scale-normalized pinch detection with hysteresis.
- Offset-preserving grabs, velocity-based throws and independent object physics.
- Gravity, damped bounces, floor friction and circular collision bounds.
- Two-hand scaling/rotation with outlines, tethers, trails and interaction effects.
- Optional palm push, fist freeze and experimental apparent-size depth control.
- Resizable/fullscreen preview, live FPS, debug overlays and 15 deterministic tests.

## Start

Tested on **Windows with 64-bit Python 3.13**. A working webcam is required.
Clone the repository, then run these commands in PowerShell:

```powershell
git clone https://github.com/MelekCreed/Telekinesis-CV.git
cd Telekinesis-CV
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\START.cmd
```

After setup, double-click **START.cmd** to launch again. The model is downloaded
from Google's model storage on first run. The local environment and model are
not committed to the repository.

Quit any previous camera preview first. **Q**, **Esc**, or closing the window
releases the webcam. The normal launcher has no automatic timeout.
The camera window opens up to twice as large, fitted to your screen. Drag its
edges to resize it or press **F** to toggle fullscreen. Camera inference stays
at the original resolution, preserving tracking performance.

Only three direct dependencies: **MediaPipe 0.10.35**, **OpenCV contrib 4.13.0.92**,
**NumPy 2.3.5**. Their transitive dependencies are installed automatically.
Use only this OpenCV distribution, not additional `opencv-python` or headless
variants in the same environment. The model downloads from Google if missing.
Camera frames are processed locally; this application does not record or upload them.

## Play

1. Show an open hand. Four colored crystals fall onto a visible virtual floor.
2. Put your index reticle on a crystal; it gets an outline. Then touch thumb and
   index together to grab. The fingertip must be on the crystal when pinch starts.
3. Keep pinching and move your hand. The original grab offset is preserved.
4. Open your fingers to release. Move quickly before opening to throw; stop
   first and it drops. Gravity, bounces, friction and collisions continue.
5. To resize/rotate, keep holding with one hand, put the other index reticle
   on the **same crystal**, then pinch with that hand. Spread the hands apart
   to enlarge, bring them together to shrink, and twist their connecting line
   to rotate. The white vertex marks orientation. Let go with one hand to
   continue holding with the other. Each hand can hold only one object.

Open your fingers once after tracking returns before grabbing again. Lost
tracking drops an object without a throw impulse, avoiding accidental launches.
Keep hands separated enough that both remain visible; overlapping hands can
occlude landmarks and confuse identity tracking.

## Controls

| Key | Action |
| --- | --- |
| R | Reset all four crystals |
| F | Toggle fullscreen; the normal window can also be resized |
| G | Toggle gravity for easier floating practice |
| A | Enable/disable experimental palm push and fist freeze |
| Z | Enable/disable the relative depth/size experiment |
| D | Debug: IDs, geometry, velocities, timings and relative z |
| L | Show/hide hand skeletons |
| [ / ] | Less/more pointer smoothing, in 5 ms steps |
| Q / Esc | Exit |

Palm/fist gestures and depth start **off** to keep ordinary grabbing predictable.
Two-hand scaling/rotation is always available with the default two-hand tracker.

With **A on**, sweep an open palm quickly beside crystals to push nearby objects.
Rapid apparent palm growth toward the camera can also trigger a push. Make a fist
near a crystal and hold about 0.4 seconds to toggle freeze. Open your hand before
making another fist to toggle it again. A frozen crystal shows LOCK; grabbing it
also unfreezes it. These are geometric heuristics, not a trained gesture classifier.

With **Z on**, grab a crystal with one hand, then move toward/away from the camera
while keeping the palm orientation steady. Apparent palm growth enlarges the
crystal; shrinking reduces it. Two-hand transforms take precedence.

## Understand the complete pipeline

    camera frame -> mirrored RGB image -> pretrained hand landmarks
    -> our coordinate/gesture math -> grab/throw/transform -> physics -> drawing

### 1. Webcam and landmarks

OpenCV requests 640x480 at 30 FPS, using Windows DirectShow with Media Foundation
fallback. The actual frame size is used for geometry. Mirroring happens before
inference so the predictions already align with your mirror image.
MediaPipe returns 21 landmarks per hand. It does not decide pinch, fist, grab,
throw or object motion for us. We use VIDEO mode with increasing timestamps so
each prediction matches the displayed frame without a separate callback queue.

### 2. Coordinates and pointer smoothing

For image width W, height H and normalized coordinates x,y:

    pixel_x = x * W
    pixel_y = y * H

The top-left corner is (0,0); y increases downward. A normalized point (0.25,0.5)
on 640x480 maps to (160,240). Use each axis's own dimension. Keep floats until
drawing; premature rounding introduces quantization jitter.

The index tip (landmark 8) drives the pointer. For time constant tau=0.035 seconds:

    alpha = 1 - exp(-dt / tau)
    smoothed = previous + alpha * (measurement - previous)

This low-pass filter suppresses prediction jitter. Larger tau is steadier but
lags more. Time-based alpha keeps a consistent response across frame rates.
After tracking loss or a gap over 250 ms, reset to the new fingertip directly.

### 3. Pinch geometry and hysteresis

    distance(a,b) = sqrt((a.x-b.x)^2 + (a.y-b.y)^2)
    palm_size = (distance(wrist[0], middle_knuckle[9])
                 + distance(index_knuckle[5], pinky_knuckle[17])) / 2
    ratio = distance(thumb_tip[4], index_tip[8]) / palm_size

As apparent hand size scales by k, both numerator and denominator scale by k,
which cancels. This is more robust than a fixed pixel gap. Perspective rotation
and occlusion still affect projected geometry. Gesture math uses raw landmarks
from the same frame, independently of pointer smoothing.

Start pinch at ratio <=0.30; release at >=0.45. Between those thresholds, keep
the previous state. This hysteresis prevents rapid toggling near one boundary.
The indicator is hollow when OPEN, has an inner dot when APPROACHING, and is
filled green when PINCHED. The ratio is visible next to it.

### 4. Grab and hit testing

A crystal uses a circular hit region: distance(pointer,center) <= radius+8 pixels.
If regions overlap, choose the nearest center. Grab on a fresh pinch transition.
Store `offset = center - pointer`; while held, `center = pointer + offset`.
This avoids snapping the center unnaturally onto the fingertip.

### 5. Throw velocity

Store recent `(time,x,y)` pointer samples in a small history. Fit a line over
the last 140 ms instead of subtracting only two noisy frames:

    v = sum((t-mean(t)) * (p-mean(p))) / sum((t-mean(t))^2)
    released_velocity = v * throw_gain

This is the least-squares slope in pixels/second, applied to x and y separately.
The default throw multiplier is 1.25. Exclude the opening frame so extending
the index to release does not create a fake final impulse. Stale/insufficient
samples give zero velocity; very small release speeds are treated as stationary.
Speeds are limited to 1800 pixels/second. Debug arrows show the estimate.

### 6. Physics

    velocity_y += gravity * dt
    velocity *= exp(-air_damping * dt)
    position += velocity * dt

Defaults: gravity 650 px/s^2, restitution 0.72, air damping 0.18/s, floor friction
4/s. At a wall, clamp penetration and reverse the incoming velocity component
with restitution. Floor friction slows horizontal motion; tiny bounces settle.
The visible floor sits above the bottom of the frame so your wrist can remain
visible while you point at a resting object.

Use substeps no larger than 1/240 second to reduce fast-object tunneling. Frame
gaps above 100 ms are capped for simulation stability (time beyond that is
discarded rather than replayed). This deliberately trades elapsed-time accuracy
after a stall for safe interaction. Damping/friction depend on seconds, not frames.

### 7. Multiple objects and collisions

Crystals are polygons visually, with **circular collision bounds**, not exact
polygon contact. Overlap occurs when center distance < sum of radii. Move apart
along the contact normal, weighted by inverse mass. Mass is proportional to r^2.
For approaching bodies, apply impulse:

    j = -(1+restitution) * relative_normal_velocity / (inverse_mass_a+inverse_mass_b)

Held and frozen objects have zero inverse mass: other objects can bounce off
them without displacing the hand-controlled object. Three separation passes per
substep handle simple stacks. This is intentionally a small, approximate solver.

### 8. Telekinesis feedback

Target outlines, stronger grabbed outlines, hand-object tethers, short trails,
and fading ring bursts communicate interaction. All use inexpensive OpenCV
drawing; the camera remains the main visual. A bright vertex reveals rotation.

### 9. Advanced interactions

Hand identity is matched across frames by the lowest-cost wrist assignment plus
a soft handedness preference. This avoids assuming model output order is stable.
It is still imperfect when hands cross or occlude one another.

Finger bending uses the dot product of the two vectors at each finger's PIP
joint. A straight finger has cosine near -1; a bent finger has a larger cosine.
Compare fingertip/wrist and joint/wrist distances as a second geometric cue.
Four extended fingers identify an open palm; at least three curled identify a fist.

Palm push requires a stable open palm, then speed >800 px/s or apparent-size
growth rate >1.8/s. Nearby unfrozen/unheld objects get a distance-weighted outward
impulse. A 0.7-second cooldown prevents repeated impulses from one sweep.
Fist freeze requires a stable target within 130 pixels for 0.4 seconds and triggers
only once until the fist opens again.

For two hands, save their initial separation d0 and line angle theta0:

    scale = current_separation / d0
    angle = original_object_angle + atan2(dy,dx) - theta0

Translation follows the hand midpoint, while the original center offset rotates
and scales with it. Radius is clamped to 16-95 pixels. On releasing one hand,
recompute its partner's offset so the object does not jump.

### Optional depth: what z really means

MediaPipe's image-landmark z is relative to the wrist, with magnitude roughly
on the normalized x scale. Smaller means closer to the camera relative to the
wrist. It is **not absolute camera distance**, and moving the whole hand toward
the camera need not change these relative z values in a useful way.
Debug displays the raw index z for inspection. The Z experiment instead uses
`current_palm_size / palm_size_at_grab` as a relative depth cue. Palm rotation,
finger pose and perspective can fool it. This is an apparent-size illusion,
not calibrated monocular 3D reconstruction.

## Files and tuning

- `main.py`: camera loop, drawing, keyboard controls.
- `hand_tracker.py`: model inference, coordinates, smoothing, gesture geometry, velocity.
- `interaction.py`: per-hand state, grabbing, throwing, two-hand transforms and advanced gestures.
- `physics.py`: object state and explicit integration/collision math.
- `test_core.py`: deterministic tests using synthetic landmarks and trajectories.

Examples, run from this folder:

```powershell
.\START.cmd --hands 1 --objects 1
.\START.cmd --throw-gain 1.6 --gravity 500 --restitution 0.8
.\START.cmd --smoothing-ms 20 --pinch-threshold 0.30 --release-threshold 0.45
.\START.cmd --advanced --depth
.\START.cmd --camera 1 --backend msmf
.\.venv\Scripts\python.exe -m unittest -v test_core
```

For camera failure, close other camera apps, check the shutter and enable
Windows Settings > Privacy & security > Camera > Let desktop apps access your camera.
For lower latency, improve lighting, try `--hands 1`, and inspect D's inference
time. The two-hand model does more work; 30 FPS depends on hardware and capture.

## Verification and limits

Manual development checks confirmed landmark alignment and pointer responsiveness.
Single-hand live runs on the development laptop measured 23-26 FPS. The two-hand preview
recorded grabs, throws and two-hand transforms; sampled active-tracking rates
were about 15-17 FPS. A separate headless check encountered approximately 1-second
camera reads, so it is not evidence of real-time performance. The 30 FPS target
has not been reached consistently on this machine.

Automated tests verify velocity fitting, frame-rate-independent smoothing,
normalized pinch/hysteresis, grab offsets, stationary release, lost/invalid
tracking, two-hand transforms and identity ordering, depth scaling, palm push,
fist dwell, elastic collision behavior, floor settling and boundary handling.
Synthetic tests establish mathematical behavior, not physical gesture reliability.

Physical checks still worth doing: pinch each crystal; throw gently then quickly;
hold still before release; hide a grabbing hand; scale/rotate with both hands;
release one hand; enable A and test palm push and fist freeze; enable Z and compare
moving toward the camera with merely rotating the palm. Advanced gestures and
depth need additional real-world validation across users, lighting and hand poses.

Reference: [Google Hand Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).
