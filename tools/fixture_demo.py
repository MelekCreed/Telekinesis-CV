r"""Run the full interaction on the synthetic desk fixture and save a contact sheet.

Real EdgeSAM, real app logic, drawn hand + synthetic landmarks (no webcam imagery).
    .venv\Scripts\python.exe tools\fixture_demo.py sheet.jpg
"""
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import App, parse_args  # noqa: E402
from segmentation import SegmentationWorker  # noqa: E402
from test_core import make_hand  # noqa: E402
from test_pipeline import desk_scene, draw_hand  # noqa: E402

scene, _ = desk_scene()
worker = SegmentationWorker(threads=4)
worker.warm_up()
app = App(640, 480, worker, parse_args(["--debug"] if "--debug" in sys.argv else []))
state = {"t": 0.0, "out": None}
shots = []


def step(hands=()):
    state["t"] += 1 / 25
    frame = scene.copy()
    person = np.zeros((480, 640), np.float32)
    for points, _ in hands:
        person[draw_hand(frame, points)] = 1
    state["out"] = app.step(frame, state["t"], 1 / 25,
                            [{"points": p, "z": np.zeros(21), "label": lab} for p, lab in hands], person)
    time.sleep(.005)


def shot(title):
    image = state["out"].copy()
    cv2.putText(image, title, (10, 470), 0, .6, (0, 0, 0), 4)
    cv2.putText(image, title, (10, 470), 0, .6, (255, 255, 255), 1)
    shots.append(image)


def hand(tip, pinch=False, label="Right"):
    return (make_hand(tip, palm=70, pinch=pinch), label)


while worker.model is None:
    step()
shot("1 raw scene (nothing selected)")
while app.selector.state != "PREVIEW":
    step([hand((165, 320))])
shot("2 touch mug -> outline preview")
for _ in range(4):
    step([hand((165, 320), True)])
for i in range(1, 26):
    step([hand((165 + 10 * i, 320 - 4 * i), True)])
shot("3 pinch + drag: original region reconstructed")
for _ in range(6):
    step([hand((415, 220), True)])
for _ in range(4):
    step([hand((415, 220))])
for _ in range(40):
    step()
shot("4 released: floats; hand-free refinement")
obj = app.manip.objects[0]
tip = obj.position + (0, 60)
for _ in range(4):
    step([hand(tip)])
for _ in range(4):
    step([hand(tip, True)])
b = tip + (110, 0)
for _ in range(4):
    step([hand(tip, True), hand(b, False, "Left")])
for _ in range(4):
    step([hand(tip, True), hand(b, True, "Left")])
for i in range(1, 21):
    step([hand(tip + (-3 * i, 3 * i), True), hand(b + (3 * i, -3 * i), True, "Left")])
shot("5 two hands: scale + rotate")
for _ in range(4):
    step([hand(tip + (-60, 60)), hand(b + (60, -60), False, "Left")])
app.key(ord("h"), state["t"])
for _ in range(10):
    step()
shot("6 hidden (H / fist)")
app.key(ord("h"), state["t"])
app.key(ord("c"), state["t"])
for _ in range(10):
    step()
shot("7 shown + duplicated (C)")
app.key(ord("r"), state["t"])
for _ in range(10):
    step()
shot("8 reset (R / open palm): back at origin")
rows = [np.hstack(shots[i:i + 4]) for i in range(0, 8, 4)]
out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "fixture_demo.jpg"
cv2.imwrite(out, np.vstack(rows))
print("wrote", out, "| background:", obj.group.reconstruction.source, "|", obj.group.refine)
app.close()
worker.close()
