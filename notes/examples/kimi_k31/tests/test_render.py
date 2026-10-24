import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from notes.examples.kimi_k31.geometry import DEFAULT_SPEC, build_plate
from notes.examples.kimi_k31.render import render_mesh


class RenderTests(unittest.TestCase):
    def test_flat_top_is_not_occluded_by_deeper_triangles(self):
        root = Path(__file__).resolve().parents[4]
        (root / ".artifacts").mkdir(exist_ok=True)
        mesh = build_plate(DEFAULT_SPEC)
        azimuth, elevation = math.radians(25), math.radians(58)
        rz = np.array(
            [
                [math.cos(azimuth), -math.sin(azimuth), 0],
                [math.sin(azimuth), math.cos(azimuth), 0],
                [0, 0, 1],
            ]
        )
        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(elevation), -math.sin(elevation)],
                [0, math.sin(elevation), math.cos(elevation)],
            ]
        )
        rotation = rx @ rz
        viewed = mesh.vertices @ rotation.T
        extent = np.ptp(viewed[:, :2], axis=0)
        scale = min(920 / extent[0], 470 / extent[1])
        center = (viewed[:, :2].min(axis=0) + viewed[:, :2].max(axis=0)) / 2
        light = np.array([0.2, -0.3, 0.93])
        light /= np.linalg.norm(light)
        normal = np.array([0, 0, 1]) @ rotation.T
        amount = 0.35 + 0.65 * float(normal @ light)
        expected = tuple(round(channel * amount) for channel in (78, 164, 247))
        with tempfile.TemporaryDirectory(
            prefix="test-render-", dir=root / ".artifacts"
        ) as directory:
            path = Path(directory) / "plate.png"
            render_mesh(mesh, DEFAULT_SPEC, path, "Offline reference")
            with Image.open(path) as image:
                for x in (-40, -15, 0, 15, 40):
                    for y in (-20, 0, 20):
                        projected = np.array([x, y, 4]) @ rotation.T
                        pixel = (projected[:2] - center) * (scale, -scale) + (550, 405)
                        with self.subTest(x=x, y=y):
                            self.assertEqual(
                                image.getpixel(tuple(np.floor(pixel).astype(int))),
                                expected,
                            )


if __name__ == "__main__":
    unittest.main()
