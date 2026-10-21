import unittest

import cv2
import numpy as np

from scripts.detect_lines import blue_cyan_mask, detect_all


class DashedCurveDetectionTests(unittest.TestCase):
    @staticmethod
    def make_scan(with_curve: bool) -> np.ndarray:
        image = np.full((500, 500, 3), 245, dtype=np.uint8)
        cv2.line(image, (440, 0), (440, 499), (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(image, (0, 250), (499, 250), (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "D2",
            (350, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

        if with_curve:
            for y in range(18, 481, 26):
                y2 = min(y + 15, 490)
                x1 = round(80 + 0.4 * y + 8 * np.sin(y / 60))
                x2 = round(80 + 0.4 * y2 + 8 * np.sin(y2 / 60))
                cv2.line(image, (x1, y), (x2, y2), (15, 15, 15), 4, cv2.LINE_AA)
        return image

    def test_links_dark_oblique_dashes_without_following_grid(self) -> None:
        x_offset, y_offset = 500, 100
        image = self.make_scan(with_curve=True)
        detections = detect_all(image, x_offset, y_offset)
        dashed = [line for line in detections if line[6] == "dashed_curve"]

        self.assertGreaterEqual(len(dashed), 8)
        for line in dashed:
            for global_x, global_y in ((line[0], line[1]), (line[2], line[3])):
                local_x = global_x - x_offset
                local_y = global_y - y_offset
                expected_x = 80 + 0.4 * local_y + 8 * np.sin(local_y / 60)
                self.assertLess(abs(local_x - expected_x), 20)
                self.assertGreater(abs(local_x - 440), 50)

    def test_blue_mask_preserves_dark_ink_but_clears_saturated_blue(self) -> None:
        image = np.full((500, 500, 3), (240, 230, 210), dtype=np.uint8)
        cv2.line(image, (50, 50), (450, 450), (10, 10, 10), 3, cv2.LINE_AA)
        cv2.line(image, (450, 50), (50, 450), (255, 0, 0), 4, cv2.LINE_AA)

        mask = blue_cyan_mask(image)
        self.assertEqual(mask[150, 150], 0)
        self.assertEqual(mask[200, 300], 255)

        detections = detect_all(image, 0, 0)
        neutral_ink_lines = [
            line
            for line in detections
            if line[6] in ("straight", "curved")
            and max(abs(line[0] - line[1]), abs(line[2] - line[3])) < 20
        ]
        blue_overlay_lines = [
            line
            for line in detections
            if line[6] in ("straight", "curved")
            and max(abs(line[0] + line[1] - 500), abs(line[2] + line[3] - 500)) < 20
        ]
        self.assertTrue(neutral_ink_lines)
        self.assertFalse(blue_overlay_lines)

    def test_grid_and_text_alone_do_not_form_a_dashed_curve(self) -> None:
        image = self.make_scan(with_curve=False)
        detections = detect_all(image, 0, 0)
        self.assertFalse(any(line[6] == "dashed_curve" for line in detections))


if __name__ == "__main__":
    unittest.main()
