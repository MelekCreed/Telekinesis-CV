# Telekinesis CV — Reality Manipulation

Point at a **real object** in a mirrored webcam image and preview its actual
segmentation silhouette. This compact Python CV learning project currently
implements **Phase 3: touch selection + point-prompted segmentation**. No virtual
crystals are spawned. Movement, tracking, extraction and disappearance come later.

## Start

Tested on Windows, Python 3.13, Intel Core i7-1355U, Intel Iris Xe and 24 GB RAM.
From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\START.cmd
```

On the configured machine, simply double-click **START.cmd**. The large camera
window is resizable; **F** toggles fullscreen. Models download automatically on
first use (internet required). Later runs work locally with cached models.
Webcam frames stay on your computer; the program does not record or upload them.

## Try this milestone

1. Keep the camera fixed. Place a distinct real object in view with reasonable
   lighting and contrast. Leave the object there and lower your hand briefly.
2. The program captures a hand-free **reference scene containing the object**.
   Give the first scene preparation several seconds. The live camera continues.
3. Visually place your index fingertip **over the object in the displayed image**.
   Hold roughly steady for 0.35 seconds. A cyan outline/tint previews a silhouette.
   This is touch selection, not a ray projected in your finger's direction.
4. If the model selected only a part, press **M** to cycle available masks.
5. Keep fingers open briefly, then pinch thumb/index for about 0.2 seconds.
   Move your hand aside if asked; the outline turns green after visible-scene
   verification. This confirms a **static reference mask**, not a tracked object.
6. **R** or **Esc** clears the selection. **B** captures a new reference after
   moving the object/camera. **Q** quits.

| Key | Action |
| --- | --- |
| B | Recapture the scene; keep objects visible and lower your hand |
| M | Cycle plausible silhouette candidates |
| R / Esc | Clear selection and aim again |
| D | Show landmarks, reference thumbnail, binary mask, model timing and gesture state |
| F | Fullscreen / large window |
| [ / ] | Less / more pointer smoothing |
| Q | Quit |

The prototype tracks one controlling hand. A candidate stays stable while the
pointer moves within its silhouette and a small margin. Moving away changes the
prompt; losing the hand clears an unconfirmed preview. After the pinch is
accepted, moving the hand away preserves the candidate for verification.

## Why this model

[**SlimSAM-77-uniform**](https://huggingface.co/nielsr/slimsam-77-uniform) is a
compressed SAM-family point-prompt model. We use the
[Xenova ONNX export](https://huggingface.co/Xenova/slimsam-77-uniform): an 8.9 MB
quantized image encoder plus a 16.6 MB prompt/mask decoder, running on CPU with
ONNX Runtime. Downloads are revision-pinned and SHA-256 verified. No PyTorch,
CUDA, backend or fixed mug/phone/bottle class list is required.

An initial public-image benchmark on this laptop took **5.4 seconds to encode**
and **120 ms to decode**. Another point on the same cached image took **160 ms**
for decoding without re-encoding. These are individual measurements, not a
live-webcam FPS guarantee. The model runs in one background worker with two CPU
threads. It encodes each reference once and only decodes when you hold a point;
one pending request replaces older pending requests. Obsolete results cannot
revive cleared selections. The display reports actual live FPS.

The hand-free reference avoids asking the model to segment your finger when it
covers the object. It is **not a clean background plate**: it still contains the
object and cannot reveal the desk behind it. Large visible scene changes or
changes in the visible target invalidate the reference. This comparison is a
simple grayscale heuristic, not optical flow, tracking or depth. It tolerates
small uniform lighting changes and excludes an approximate hand/forearm region.

## The CV concepts

- **Detection** finds objects, often as boxes and class labels. This milestone
  uses no object-class detector; MediaPipe does locate hand landmarks.
- **Segmentation** labels individual pixels. A point prompt asks the model which
  coherent region contains that location. It can return an object, a part, or
  the wrong surrounding region; estimated IoU ranks candidates but is not proof.
- **Tracking** follows the same object across time. It is not implemented here;
  a stationary mask over a stationary scene must not be called tracking.

The mirrored image is the shared coordinate system for landmarks, prompt and
mask. We convert BGR camera pixels to normalized RGB, resize the longest side
to 1024, and pad the other side. The point uses the same scale. The model returns
low-resolution mask logits; we upsample, remove padding, then restore the camera
size before thresholding. Skipping that crop would stretch the silhouette.

A **binary mask** is 1 on selected pixels and 0 elsewhere. The debug thumbnail
shows these as white and black. A **soft mask** has intermediate values from
0 to 1, useful as fractional edge coverage. Later, mask values will become an
RGBA alpha channel: `output = alpha * object + (1-alpha) * background`.
Currently the mask only drives an outline and subtle tint; no pixels are
extracted or erased. The model's probability is not perfect transparency matting.

Gesture recognition uses time: open fingers arm confirmation, continuous
low pinch distance confirms it, and a release threshold separates open/closed
states. One noisy low-distance frame followed by ambiguous readings cannot
complete a pinch. Candidate changes, hand loss and long gaps require rearming.

## Limitations and next steps

- Promptable does not mean universally accurate. Thin/transparent objects,
  clutter, low contrast and small details can fail; no rectangle is substituted.
  Empty, tiny, whole-scene and low-score masks are rejected.
- Keep the scene mostly still. Body motion, shadows and lighting changes can
  force recapture. Hidden motion cannot be verified; confirmation waits for
  sufficient visible target area, but the check is heuristic.
- First preparation is slow on this CPU; it does not block the camera loop.
- No tracking, RGBA extraction, manipulation, inpainting, depth, occlusion,
  scaling, rotation or throwing is active in this milestone.
- Next: validate desk-object silhouettes and alignment with the webcam, then
  implement Phase 4 tracking and Phase 5 RGBA extraction. Only after that should
  movement and reconstruction make the original object appear removed.

## Files and verification

`main.py` handles the webcam and preview; `hand_tracker.py` handles landmarks;
`segmentation.py` keeps model and coordinate transforms explicit; `selection.py`
contains temporal selection and scene checks. Existing `interaction.py` and
`physics.py` retain reusable geometry/physics from the previous version but are
not imported by the active app.

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_selection test_core
.\run.ps1 --debug
.\run.ps1 --headless --seconds 20
```

The 32 deterministic tests cover coordinates/padding, stale results, bounded
requests, gesture noise, candidate changes, confirmation visibility, scene
changes and prior reusable math. Model inference has also been exercised on
public photographs; physical pointing and mask alignment still need a person
at the webcam. Try a mug, phone and one irregular object, cycle masks, clear,
recapture and deliberately move the target. Record failures rather than assuming
model score or detection count means visual correctness.

Latest verification (2026-09-27): 32 tests passed. A 25-second GUI webcam run
completed at 5.7 FPS; a separate 15-second headless run with detected hands
averaged 27.4 FPS. These are different conditions, not an isolated speedup test.
Detailed timing found window presentation itself cheap (~2 ms); hand/scene
processing varied with load. Previous responsiveness is not yet confirmed for
this milestone. No physical silhouette confirmation was observed in these runs.
Cancellation of running model preparation returned in 0.001 s in a local check.

A fresh read-only Codex CLI review found shutdown cancellation and camera-gap
dwell issues. The implementation author fixed both and reran all 32 tests.

Final installed GUI smoke check: 20 seconds, 313 frames, 15.6 FPS average; hand detected throughout, so no hand-free reference was captured during that run. Interactive mask alignment and confirmation remain unverified.
