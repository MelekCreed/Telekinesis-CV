"""Selection safety and coordinate tests; no model downloads or webcam needed."""
from concurrent.futures import CancelledError, Future
from threading import Event
import unittest
from unittest.mock import Mock

import numpy as np

from segmentation import MaskResult, SegmentationWorker, check_cancelled, preprocess, restore_logits
from selection import PinchConfirmation, Selection, compare_scene


def candidate(selection, masks=None):
    if masks is None:
        mask = np.zeros((120, 200), bool)
        mask[30:90, 50:150] = True
        masks = [mask]
    return MaskResult(selection.reference_id, selection.prompt_id, (75, 60), masks,
                      [.9] * len(masks), 0, 0, 0)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.selection = Selection()
        self.selection.set_reference(np.zeros((120, 200, 3), np.uint8))

    def test_reference_owns_immutable_copy(self):
        frame = np.zeros((120, 200, 3), np.uint8)
        self.selection.set_reference(frame)
        frame[:] = 255
        self.assertFalse(self.selection.reference.any())

    def test_dwell_and_moving_to_another_object_rejects_old_result(self):
        s = self.selection
        self.assertIsNone(s.aim((75, 60), 0))
        self.assertIsNone(s.aim((78, 60), .2))
        self.assertIsNotNone(s.aim((78, 60), .4))
        old = candidate(s)
        s.aim((190, 100), .5)
        self.assertFalse(s.accept(old))

    def test_reference_change_rejects_old_result(self):
        old = candidate(self.selection)
        self.selection.invalidate()
        self.assertFalse(self.selection.accept(old))

    def test_camera_gap_restarts_unsubmitted_aim_and_verification(self):
        s = self.selection
        s.aim((75, 60), 0)
        s.clear_since = .1
        s.reset_observation_dwell(1)
        self.assertIsNone(s.aim((75, 60), 1))
        self.assertIsNone(s.clear_since)
        self.assertIsNotNone(s.aim((75, 60), 1.4))

    def test_movement_within_preview_retains_selection(self):
        s = self.selection
        s.aim((75, 60), 0)
        s.accept(candidate(s))
        version = s.prompt_id
        s.aim((130, 70), .5)
        self.assertEqual(s.prompt_id, version)
        self.assertIsNotNone(s.mask)
        s.aim(None, .6)
        self.assertIsNone(s.mask)

    def test_candidate_switch_resets_gesture(self):
        s = self.selection
        mask = candidate(s).masks[0]
        s.accept(candidate(s, [mask, mask.copy()]))
        for t in (0, .15, .20, .35):
            s.update_confirmation(.6 if t < .2 else .1, t, .1, False)
        s.cycle()
        s.update_confirmation(.1, .45, 1, False)
        self.assertFalse(s.pending_confirmation)
        self.assertFalse(s.confirmed)

    def test_confirmation_waits_for_visible_unchanged_target(self):
        s = self.selection
        s.accept(candidate(s))
        for ratio, t in ((.6, 0), (.6, .15), (.1, .2), (.1, .35), (.1, .45)):
            s.update_confirmation(ratio, t, .1, False)
        self.assertTrue(s.pending_confirmation)
        s.aim(None, .5)  # Moving the hand away must preserve a pending mask.
        self.assertIsNotNone(s.mask)
        self.assertFalse(s.update_confirmation(None, .6, 1, True))
        self.assertFalse(s.update_confirmation(None, .7, .5, False))
        self.assertFalse(s.update_confirmation(None, .8, 1, False))
        self.assertTrue(s.update_confirmation(None, 1.1, 1, False))

    def test_pending_confirmation_times_out(self):
        s = self.selection
        s.accept(candidate(s))
        s.pending_confirmation, s.pending_since = True, 0
        self.assertFalse(s.update_confirmation(None, 7, 1, False))
        self.assertFalse(s.pending_confirmation)


class GestureTests(unittest.TestCase):
    def test_continuous_close_fires_once(self):
        g = PinchConfirmation()
        events = [g.update(r, t, 1) for r, t in
                  ((.6, 0), (.6, .15), (.1, .2), (.1, .35), (.1, .45), (.1, .6))]
        self.assertEqual(events, [False, False, False, False, True, False])

    def test_single_low_frame_and_hysteresis_band_never_confirm(self):
        g = PinchConfirmation()
        for r, t in ((.6, 0), (.6, .15), (.1, .2), (.38, .3), (.38, .45), (.38, .6)):
            self.assertFalse(g.update(r, t, 1))

    def test_hand_loss_time_gap_or_changed_target_requires_rearming(self):
        for kind in ('loss', 'gap', 'target'):
            g = PinchConfirmation()
            for r, t in ((.6, 0), (.6, .15), (.1, .2)):
                g.update(r, t, 1)
            if kind == 'loss':
                g.update(None, .25, 1)
            self.assertFalse(g.update(.1, .6 if kind == 'gap' else .4,
                                      2 if kind == 'target' else 1))


class GeometryTests(unittest.TestCase):
    def test_rgb_normalization_and_padding_on_non_square_frames(self):
        for h, w in ((120, 200), (200, 120)):
            frame = np.zeros((h, w, 3), np.uint8)
            frame[:, :, 2] = 255
            tensor, (nh, nw) = preprocess(frame)
            self.assertEqual(tensor.shape, (1, 3, 1024, 1024))
            np.testing.assert_allclose(tensor[0, :, 0, 0],
                                       [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-6)
            self.assertEqual(max(nh, nw), 1024)
            self.assertFalse(tensor[:, :, nh:, :].any())
            self.assertFalse(tensor[:, :, :, nw:].any())

    def test_mask_inverse_transform_preserves_edges_and_aspect(self):
        for shape, resized in (((120, 200), (614, 1024)), ((200, 120), (1024, 614))):
            # Linear coordinate ramps expose stretching or padding crop errors.
            y, x = np.mgrid[:256, :256].astype(np.float32)
            for low, axis, extent in ((x, 1, resized[1]), (y, 0, resized[0])):
                restored = restore_logits(low, resized, shape)
                self.assertEqual(restored.shape, shape)
                end = restored[-1, -1]
                self.assertAlmostEqual(float(end), extent / 4 - .5, delta=1)
                self.assertLess(float(restored[0, 0]), 1)

    def test_scene_change_lighting_and_hand_exclusion(self):
        old = np.full((120, 200, 3), 70, np.uint8)
        excluded = np.zeros((120, 200), bool)
        self.assertFalse(compare_scene(old, old + 12, excluded)[0])
        new = old.copy()
        new[20:100, 20:130] = 240
        self.assertTrue(compare_scene(old, new, excluded)[0])
        excluded[12:108, 12:138] = True
        self.assertFalse(compare_scene(old, new, excluded)[0])

    def test_local_target_change_detected_below_global_threshold(self):
        old = np.zeros((120, 200, 3), np.uint8)
        new = old.copy()
        target = np.zeros(old.shape[:2], bool)
        target[40:70, 80:110] = True
        new[target] = 255
        scene, exposed, changed = compare_scene(old, new, np.zeros_like(target), target)
        self.assertFalse(scene)
        self.assertTrue(changed)
        self.assertEqual(exposed, 1)


class WorkerTests(unittest.TestCase):
    def test_cancel_prevents_model_preparation_and_terminates_onnx(self):
        worker = SegmentationWorker()
        worker.close()
        self.assertTrue(worker.run_options.terminate)
        with self.assertRaises(CancelledError):
            worker._run(None)
        cancel = Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            check_cancelled(cancel)

    def test_only_latest_pending_job_and_owned_image_are_submitted(self):
        worker = SegmentationWorker()
        worker.pool.shutdown()
        worker.pool = Mock()
        worker.future = Future()  # Controlled in-flight work, no model downloads.
        frame = np.zeros((10, 10, 3), np.uint8)
        try:
            worker.request(frame, (1, 1), 1, 1)
            worker.request(frame, (2, 2), 1, 2)
            frame[:] = 255
            worker.poll()
            worker.pool.submit.assert_not_called()
            worker.future.set_result('old')
            self.assertEqual(worker.poll(), 'old')
            request = worker.pool.submit.call_args.args[1]
            self.assertEqual(request[3], 2)
            self.assertFalse(request[0].any())
            self.assertIsNone(worker.pending)
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
