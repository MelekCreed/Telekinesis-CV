"""Point-prompt SlimSAM inference. RGB/padding/point/mask transforms stay explicit."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from threading import Event
from urllib.request import urlopen

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).with_name("models") / "slimsam"
REVISION = "5850ab45f587c112167512ffef949107115e26a0"
MODEL_FILES = {
    "vision_encoder_quantized.onnx": "cce23c7b2e5d4f330932738fb67ba518e04b0d99ccdd1cccd22a7da4e01f2971",
    "prompt_encoder_mask_decoder.onnx": "f4514391764fbd56e08e119060d874ecd7d52994bfb1968af159e12d4943b5bb",
}


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Segmentation preparation cancelled")


def ensure_models(directory, cancel=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in MODEL_FILES.items():
        check_cancelled(cancel)
        path = directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        temporary = path.with_suffix(".download")
        url = f"https://huggingface.co/Xenova/slimsam-77-uniform/resolve/{REVISION}/onnx/{name}"
        try:
            print(f"Downloading {name}...", flush=True)
            download_started = time.monotonic()
            # Bound each network read, and bound the entire file transfer too.
            with urlopen(url, timeout=5) as response, temporary.open("wb") as output:
                while True:
                    check_cancelled(cancel)
                    if time.monotonic() - download_started > 120:
                        raise TimeoutError(f"Model download exceeded 120 seconds: {name}")
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            check_cancelled(cancel)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Model checksum failed: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def preprocess(frame):
    height, width = frame.shape[:2]
    scale = 1024 / max(height, width)
    resized_size = (int(height * scale + 0.5), int(width * scale + 0.5))
    new_h, new_w = resized_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    # The export expects normalized RGB with zero padding on the right/bottom.
    tensor = np.zeros((1, 3, 1024, 1024), np.float32)
    tensor[0, :, :new_h, :new_w] = rgb.transpose(2, 0, 1)
    return tensor, resized_size


def restore_logits(low_resolution, resized_size, original_size):
    new_h, new_w = resized_size
    height, width = original_size
    square = cv2.resize(low_resolution, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # Undo padding FIRST, then resize. Direct 256->camera resizing stretches masks.
    return cv2.resize(square[:new_h, :new_w], (width, height), interpolation=cv2.INTER_LINEAR)


@dataclass
class MaskResult:
    reference_id: int
    prompt_id: int
    point: tuple
    masks: list
    scores: list
    encoder_ms: float
    decoder_ms: float
    total_ms: float


class SlimSAM:
    def __init__(self, directory=MODEL_DIR, threads=2, cancel=None, run_options=None):
        self.cancel = cancel
        self.run_options = run_options or ort.RunOptions()
        ensure_models(directory, cancel)
        check_cancelled(cancel)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.encoder = ort.InferenceSession(str(Path(directory) / "vision_encoder_quantized.onnx"),
                                           options, providers=["CPUExecutionProvider"])
        check_cancelled(cancel)
        self.decoder = ort.InferenceSession(str(Path(directory) / "prompt_encoder_mask_decoder.onnx"),
                                           options, providers=["CPUExecutionProvider"])
        self.reference_id = None
        self.embeddings = None
        self.resized_size = None

    def segment(self, frame, point, reference_id=0, prompt_id=0):
        check_cancelled(self.cancel)
        started = time.perf_counter()
        height, width = frame.shape[:2]
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Point prompt must be inside the image.")
        encoder_ms = 0.0
        # A reference ID describes ONE immutable camera image, not a live stream.
        if self.reference_id != reference_id:
            encoding_started = time.perf_counter()
            tensor, self.resized_size = preprocess(frame)
            check_cancelled(self.cancel)
            outputs = self.encoder.run(None, {"pixel_values": tensor}, self.run_options)
            self.embeddings = dict(zip([o.name for o in self.encoder.get_outputs()], outputs))
            self.reference_id = reference_id
            encoder_ms = (time.perf_counter() - encoding_started) * 1000
        new_h, new_w = self.resized_size
        points = np.array([[[[x * new_w / width, y * new_h / height]]]], np.float32)
        feed = dict(self.embeddings, input_points=points, input_labels=np.ones((1, 1, 1), np.int64))
        decoding_started = time.perf_counter()
        check_cancelled(self.cancel)
        outputs = dict(zip([o.name for o in self.decoder.get_outputs()],
                           self.decoder.run(None, feed, self.run_options)))
        decoder_ms = (time.perf_counter() - decoding_started) * 1000
        masks, scores = [], []
        for low, score in zip(outputs["pred_masks"][0, 0], outputs["iou_scores"][0, 0]):
            logits = restore_logits(low, self.resized_size, (height, width))
            # A logit of 0 is probability 0.5; threshold after interpolation.
            mask = logits > 0
            area = int(mask.sum())
            ix, iy = int(round(x)), int(round(y))
            near_point = mask[max(0, iy - 5):min(height, iy + 6), max(0, ix - 5):min(width, ix + 6)].any()
            # Reject implausible masks, but don't replace failures with a rectangle.
            if near_point and 80 <= area <= height * width * .65 and float(score) >= .45:
                masks.append(mask)
                scores.append(float(score))
        order = np.argsort(scores)[::-1]
        return MaskResult(reference_id, prompt_id, tuple(point), [masks[i] for i in order],
                          [scores[i] for i in order], encoder_ms, decoder_ms,
                          (time.perf_counter() - started) * 1000)


class SegmentationWorker:
    """One running inference plus one replaceable pending job, never a frame queue."""
    def __init__(self, directory=MODEL_DIR, threads=2):
        self.directory, self.threads = directory, threads
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentation")
        self.future = None
        self.pending = None
        self.model = None
        self.error = None
        self.cancel = Event()
        self.run_options = ort.RunOptions()

    @property
    def busy(self):
        return self.future is not None or self.pending is not None

    def request(self, frame, point, reference_id, prompt_id):
        # Own a copy; drawing and the next capture cannot mutate a worker's input.
        self.pending = (frame.copy(), tuple(point), reference_id, prompt_id)

    def discard_pending(self):
        self.pending = None

    def _run(self, request):
        check_cancelled(self.cancel)
        if self.model is None:
            self.model = SlimSAM(self.directory, self.threads, self.cancel, self.run_options)
        check_cancelled(self.cancel)
        return self.model.segment(*request)

    def poll(self):
        result = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
            self.future = None
        if self.future is None and self.pending is not None:
            self.error = None
            self.future = self.pool.submit(self._run, self.pending)
            self.pending = None
        return result

    def close(self):
        self.pending = None
        self.cancel.set()
        self.run_options.terminate = True  # Interrupt running ONNX work between operators.
        self.pool.shutdown(wait=True, cancel_futures=True)
