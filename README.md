# Telekinesis CV — Reality Manipulation

Touch a **real object** in the mirrored webcam image with your fingertip → its actual
silhouette highlights → **pinch** to lift its extracted appearance → move, throw, scale,
rotate, hide, restore, reset, duplicate. The physical object never moves; only its
appearance in the video does, and its original spot is painted over with reconstructed
background. No object classes, no synthetic assets. Compact local Python + OpenCV.

## Start

Windows, Python 3.13 (tested on an i7-1355U, Iris Xe, no CUDA):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\START.cmd              # or: .\run.ps1 --debug
```

First launch downloads three revision-pinned, SHA-256-checked models (~46 MB total):
EdgeSAM encoder + decoder (segmentation) and MediaPipe's selfie segmenter (person mask).
The hand landmark model is downloaded as before. Webcam frames never leave the machine;
nothing is recorded. **F** toggles fullscreen; the window is resizable.

## How to use it (gestures)

| You do | What happens |
| --- | --- |
| **Touch** an object in the image with your index fingertip and hold still ~¼ s | A ring fills, then "Finding the object…"; a thin cyan outline shows the silhouette |
| Keep pointing at the outline ~1.6 s | It steps to the next plausible outline (part ↔ whole). Pinch when the right one shows |
| **Pinch** (thumb + index) | Locks **and** grabs the object; its original spot is reconstructed |
| Move the pinched hand | The object follows, keeping your grab offset (no snapping) |
| Open fingers slowly | Placed: it floats where you left it ("frozen in space") |
| Flick, then open fingers | Thrown: gravity, bounces, damping, lands on the bottom edge |
| Pinch a floating or flying object | Grab/catch it again |
| **Second hand pinches** while the first holds | Two-hand mode: spread/close = scale (0.25×–4×), turn the hand line = rotate |
| Twist the holding hand (>20°) | One-hand rotation beyond a dead zone (T toggles) |
| **Fist** held 0.6 s (nothing held) | Hide / show the active object (fades; background stays) |
| **Fist** of the other hand while one hand holds | Duplicate (experimental) |
| **Open palm**, fingers spread, held still 1.5 s | Reset: original position, scale, rotation, visible |

Progress rings show fist/palm actions before they fire. A single noisy frame never
triggers anything. The mouse is a fallback "hand": hover to aim, left-button = pinch,
drag = move, fast drag + release = throw, right-click = next outline, wheel = scale,
Ctrl+wheel = rotate. Use it to tell segmentation failures apart from hand-tracking ones.

| Key | Action |
| --- | --- |
| Esc | Release the active object back to reality (original visible again) |
| R | Reset active object · H hide/show · C duplicate · Z freeze/unfreeze (drop) |
| M | Next candidate outline · Tab next object · X release all |
| B | Switch background fill (memory/plate · smooth fill · FSR texture guess) |
| P | Capture a **clean plate** (only valid if you physically removed the objects) |
| O | Occlusion in front of objects: hands → whole person → off |
| D | Debug overlay · K key legend · F fullscreen · E retry model after an error · Q quit |

Options: `--debug`, `--camera 1`, `--backend msmf`, `--hands auto|1|2`,
`--min-scale/--max-scale`, `--max-area 0.30`, `--threads 4`, `--no-person`,
`--headless --seconds 20` (metrics only).

## What happens under the hood (and where)

**Coordinates.** Every frame is mirrored once (`main.run`), before hand tracking,
segmentation and drawing, so landmarks, prompts, masks and display share one pixel
system. MediaPipe's normalized x/y are scaled by width and height separately
(`hand_tracker.py`).

**Detection vs segmentation vs tracking.** *Detection* finds things (here: MediaPipe
detects hands; no object detector or class list is used). *Segmentation* decides which
exact pixels belong to an object: EdgeSAM, a promptable SAM-family model, answers "what
coherent object is at this point?" (`segmentation.py`). It runs **once per selection**,
in a background thread (~0.3–1.1 s encode, ~80–340 ms decode measured here). *Tracking*
follows that same object afterwards, cheaply, every frame: `scene.ObjectTracker` matches
the object's own appearance (masked normalized cross-correlation) in a small window around
its last position. It can say OCCLUDED/LOST and hold still; it never searches for, or
switches to, another object.

**Point prompts and the hand.** The prompt is placed slightly *ahead* of your fingertip
along the finger (`selection.aim_point`), because the fingertip pixel is skin. The
snapshot being segmented is the live frame *with your hand in it*, so points on your hand
are sent as **negative** prompts and candidates overlapping the hand region are rejected.
Transforms are explicit: BGR→RGB, SAM normalization, resize longest side to 1024,
pad right/bottom; the 256² mask logits are upsampled, **the padding is cropped, then**
resized back (`restore_logits`). Skipping the crop would stretch every mask.

**Choosing the whole object.** SAM returns several candidates (part / object / bigger).
Its own quality score favours small crisp parts (a door over the whole truck), so
`rank_candidates` drops implausible ones (huge, surfaces touching 2+ image borders,
covering the hand), then prefers the **largest** candidate whose score and *stability*
(does the mask change if the logit threshold moves ±1?) are close to the best. Stability
is what rejects a merge of two neighbouring objects (their seam is uncertain).

**Binary vs soft masks, alpha compositing.** A binary mask is 0/1 per pixel. We blur it
slightly (`feathered_alpha`) into a *soft* alpha 0…1 so edges blend instead of looking
cut out. The extracted sprite is `RGBA = camera pixels + alpha`. Compositing uses
premultiplied colour: `out = src·α + dst·(1−α)`; premultiplying before warping avoids
dark fringes (`manipulation.render_sprite`).

**Transforms.** Each object has position, scale and 2D angle. `ManipulatedObject.matrix`
builds one affine matrix `T(position)·R(angle)·S(scale)·T(−anchor)`, and the pixels and
alpha are warped with the **same** matrix. This is image-plane rotation: it cannot show
the back of a mug.

**Hiding the original (reconstruction).** The mug physically stays, so its region is
painted over (`scene.Reconstruction`). Best source first: a **clean plate** (P, captured
with objects physically removed), or **scene memory** — a frame from the last ~40 s in
which that spot looked different while its surroundings matched (only if the object was
placed or moved after the app started). Otherwise **inpainting invents** the pixels: an
instant smooth pyramid fill, plus OpenCV-contrib FSR computed in the background (B
switches). A frame that contains the object is *not* a background plate. The patch is
moved with the tracker and re-lit using the surrounding ring's colour drift. Expect a
blurry/smeared patch with inpainting, and a visible seam if lighting changes a lot.

**Occlusion.** MediaPipe's selfie segmenter gives a real per-pixel person mask (~5–35 ms
here). Pixels that are *person* and near a *hand* are drawn in front of manipulated
objects; the whole person can be put in front with O. The webcam gives no depth, so the
layering is a rule (hands in front), not measured depth. Landmark hulls are used only to
decide *which* person pixels count as hands, never as the segmentation itself.

**Temporal gesture logic.** `PinchGesture` is a state machine
`OPEN → PINCH_CANDIDATE → PINCHED → RELEASE_CANDIDATE → OPEN` with two thresholds
(hysteresis), minimum durations, and arming (fingers must be seen open first). Better
than `if distance < threshold` because one noisy frame can neither grab nor drop.
Held objects follow the **palm centre** (not the thumb–index midpoint), so opening the
fingers does not jerk the object; throw velocity is a least-squares fit over timestamped
samples *before* the fingers opened. A holding hand may vanish for 0.25 s without
dropping; longer loss places the object where it is (never a throw).

**Keeping it real-time.** One running + one replaceable pending segmentation job (no
queue); every job carries snapshot/prompt ids so late answers for an abandoned aim are
discarded. Encodings are reused for new aim points while fresh and not hidden by the
hand. MediaPipe runs its palm detector every frame while it sees fewer hands than it is
asked for, so the app tracks **one** hand while pointing and switches to two only while
an object is held. Person segmentation runs only when objects exist.

## Honest limitations

- Segmentation quality varies with contrast, clutter, thin/transparent/reflective objects
  and lighting. When the default outline is wrong, keep pointing (carousel) or press M.
- Pinch/aim use 2D landmarks; bad lighting or motion blur still causes misses.
- Reconstruction without a clean plate or memory is a guess (blurred). Shadows cast by the
  real object are outside the mask and stay visible. Big camera moves break alignment;
  tracking is translation-only (no rotation/scale of the physical object) and holds its
  last pose when unsure.
- The sprite is a snapshot of the object; later lighting changes are not applied to it.
- Two-hand manipulation costs more CPU (the second hand needs palm detection).
- Duplication is experimental. No 3D, no depth, no novel views.

## Files and verification

`main.py` (camera loop, `App.step` per-frame pipeline, UI), `hand_tracker.py`
(landmarks, smoothing, velocity), `selection.py` (gesture state machines, aim, selection
states), `segmentation.py` (EdgeSAM, transforms, ranking, worker), `scene.py` (person
segmentation, tracking, reconstruction), `manipulation.py` (sprites, grab/throw/scale/
rotate, physics, rendering). `tools/` has an evaluation sheet on public SAM demo photos
and a full-sequence demo on a synthetic desk.

```powershell
.\.venv\Scripts\python.exe -m unittest test_core test_selection test_pipeline
.\.venv\Scripts\python.exe tools\fixture_demo.py demo.jpg
.\.venv\Scripts\python.exe tools\eval_selection.py eval.jpg
.\run.ps1 --headless --seconds 20
```

`test_pipeline.py` uses the real model on a synthetic desk (irregular mug with handle,
phone, bottle) with a drawn hand whose landmarks drive the app: selection with the hand in
frame, stale-result rejection, grab offset, reconstruction, hide/show, reset, duplicate,
release, throw/landing, hand loss, two-hand scale/rotation, camera shift and scene change.
These prove the pipeline mechanics, **not** how it feels with your real hands and objects.
