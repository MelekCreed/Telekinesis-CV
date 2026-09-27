2026-09-27: Implement Phase 3 only per latest brief. Reuse camera and hand code; remove crystal rendering from active main. Keep prior reusable physics dormant. Use CPU SlimSAM ONNX based on measured hardware and 4.5–5.4s image encoding with ~100–160ms cached decoding. Capture a hand-free scene containing target objects, not a background-removal plate. Delay static-mask confirmation until hand clears enough of target. Track reference/prompt versions and reject obsolete work. No video recording/upload and no GitHub push for this local experimental milestone.

2026-09-27 (Claude Code) — Redesign after failed usability test:
- Model: EdgeSAM-3x ONNX replaces SlimSAM (measured encoder 0.3-1.1 s vs 3.3-5.1 s; decoder 80-340 ms). S-Lab non-commercial licence (fine for this learning project).
- Selection from the LIVE frame (hand-free reference + manual recapture removed): prompt ahead of fingertip, hand landmarks as negative prompts, hand-overlap filter; cached encoding reused while fresh.
- Root causes fixed: pinch motion retargeted/cancelled prompts (preview now latched while inside outline and frozen once the pinch starts); score-only ranking chose parts (now largest steady plausible candidate; stability rejects merges; border-touching surfaces rejected); slow encode.
- Tracking: translation-only masked NCC template tracking; holds pose when occluded/lost; never re-discovers.
- Reconstruction: clean plate > scene memory > push-pull fill (+ FSR in background, B toggles).
- Held objects follow the palm centre; throw velocity from samples before release; 0.25 s dropout grace.
- Adaptive hand count (1 while pointing, 2 while holding): MediaPipe palm detection runs every frame when fewer hands than num_hands are visible.
- Removed interaction.py/physics.py (crystal era, preserved in git history f0ce5e8).
