import unittest

import cv2
import numpy as np

from scripts.detect_lines import blue_cyan_mask, detect_all, detect_small_circles


class DashedCurveDetectionTests(unittest.TestCase):
    @staticmethod
    def make_scan(
        with_curve: bool, amplitude: float = 8.0, period: float = 60.0
    ) -> np.ndarray:
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
                x1 = round(80 + 0.4 * y + amplitude * np.sin(y / period))
                x2 = round(80 + 0.4 * y2 + amplitude * np.sin(y2 / period))
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

    def test_traces_a_dashed_u_curve_that_reverses_in_y(self) -> None:
        image = self.make_scan(with_curve=False)
        x_start, x_end = 20.0, 320.0
        y_base, amplitude = 120.0, 90.0
        for t in np.arange(0.04, 0.97, 0.08):
            x = x_start + (x_end - x_start) * t
            y = y_base + amplitude * (1.0 - np.cos(2.0 * np.pi * t))
            slope = (
                amplitude
                * 2.0
                * np.pi
                * np.sin(2.0 * np.pi * t)
                / (x_end - x_start)
            )
            tangent_length = float(np.hypot(1.0, slope))
            half_dash_length = 8.0
            dx = half_dash_length / tangent_length
            dy = half_dash_length * slope / tangent_length
            start = (round(x - dx), round(y - dy))
            end = (round(x + dx), round(y + dy))
            cv2.line(image, start, end, (15, 15, 15), 4, cv2.LINE_AA)

        detections = detect_all(image, 0, 0)
        dashed = [line for line in detections if line[6] == "dashed_curve"]

        self.assertGreaterEqual(len(dashed), 8)
        endpoints = [
            (x, y)
            for line in dashed
            for x, y in ((line[0], line[1]), (line[2], line[3]))
        ]
        for x, y in endpoints:
            t = (x - x_start) / (x_end - x_start)
            expected_y = y_base + amplitude * (1.0 - np.cos(2.0 * np.pi * t))
            self.assertLess(abs(y - expected_y), 20)
            self.assertLess(x, 340)

        self.assertLess(min(x for x, _ in endpoints), 60)
        self.assertGreater(max(x for x, _ in endpoints), 260)

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

    def test_thin_wavy_ink_is_detected_on_cyan_background(self) -> None:
        image = np.full((300, 300, 3), (240, 230, 210), dtype=np.uint8)
        points = np.array(
            [
                (round(130 + 25 * np.sin(y / 32)), y)
                for y in range(12, 290, 2)
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.line(image, (0, 145), (299, 145), (25, 25, 25), 1, cv2.LINE_AA)

        detections = detect_all(image, 0, 0)
        curves = [line for line in detections if line[6] == "curved"]
        self.assertGreaterEqual(len(curves), 8)
        for line in curves:
            for x, y in ((line[0], line[1]), (line[2], line[3])):
                expected_x = 130 + 25 * np.sin(y / 32)
                self.assertLess(abs(x - expected_x), 15)

    def test_grid_and_text_alone_do_not_form_a_dashed_curve(self) -> None:
        image = self.make_scan(with_curve=False)
        detections = detect_all(image, 0, 0)
        self.assertFalse(any(line[6] == "dashed_curve" for line in detections))


class SmallCircleDetectionTests(unittest.TestCase):
    def test_finds_closed_ring_symbols_and_rejects_specks(self) -> None:
        image = np.full((500, 500, 3), 220, dtype=np.uint8)
        expected_centers = [(80, 100), (220, 260), (390, 400), (60, 360)]
        for center, radius in zip(expected_centers, (8, 7, 9, 4)):
            cv2.circle(image, center, radius, (10, 10, 10), 2, cv2.LINE_AA)

        # Grid ink, solid specks, and a ring too small to be a target symbol.
        cv2.line(image, (0, 200), (499, 200), (20, 20, 20), 2, cv2.LINE_AA)
        for center in ((130, 120), (300, 330), (450, 80)):
            cv2.circle(image, center, 1, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(image, (320, 100), 3, (10, 10, 10), 1, cv2.LINE_AA)

        x_offset, y_offset = 500, 100
        detections = detect_all(image, x_offset, y_offset)
        circles = [item for item in detections if item[6] == "small_circle"]
        self.assertEqual(len(circles), len(expected_centers))

        actual_centers = sorted((item[0], item[1]) for item in circles)
        expected_global = sorted(
            (x + x_offset, y + y_offset) for x, y in expected_centers
        )
        for actual, expected in zip(actual_centers, expected_global):
            self.assertLess(np.hypot(actual[0] - expected[0], actual[1] - expected[1]), 2)

        for circle in circles:
            self.assertEqual(circle[0], circle[2])
            self.assertEqual(circle[1], circle[3])
            self.assertGreater(circle[11], 4)
            self.assertLessEqual(circle[11], 12)

    def test_rejects_dark_center_even_when_contour_has_a_hole(self) -> None:
        gray = np.full((80, 80), 220, dtype=np.uint8)
        binary = np.zeros_like(gray)
        cv2.circle(binary, (40, 40), 9, 255, 3, cv2.LINE_8)
        cv2.circle(gray, (40, 40), 9, 10, 3, cv2.LINE_8)
        cv2.circle(gray, (40, 40), 5, 80, -1, cv2.LINE_8)

        self.assertEqual(detect_small_circles(gray, binary), [])

        # The same annulus is retained when its interior is genuinely light.
        cv2.circle(gray, (40, 40), 5, 220, -1, cv2.LINE_8)
        circles = detect_small_circles(gray, binary)
        self.assertEqual(len(circles), 1)
        self.assertLess(np.hypot(circles[0][0] - 40, circles[0][1] - 40), 1)

    def test_detects_a_ring_touching_a_short_stroke(self) -> None:
        image = np.full((250, 250, 3), 220, dtype=np.uint8)
        cv2.circle(image, (100, 100), 6, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.line(image, (105, 100), (115, 100), (10, 10, 10), 2, cv2.LINE_AA)

        detections = detect_all(image, 0, 0)
        circles = [item for item in detections if item[6] == "small_circle"]
        self.assertEqual(len(circles), 1)
        self.assertLess(np.hypot(circles[0][0] - 100, circles[0][1] - 100), 2)
        self.assertLessEqual(circles[0][11], 7)


if __name__ == "__main__":
    unittest.main()
