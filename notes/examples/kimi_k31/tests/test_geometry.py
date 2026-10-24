import io
import math
import unittest
from dataclasses import replace

import numpy as np
import trimesh

from notes.examples.kimi_k31.geometry import (
    DEFAULT_SPEC,
    GeometryError,
    PlateSpec,
    build_plate,
    validate_revision,
    verify_plate,
)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = build_plate(DEFAULT_SPEC)

    def test_real_geometry_has_two_through_holes(self):
        report = verify_plate(self.mesh, DEFAULT_SPEC)
        self.assertTrue(report["accepted"], report)
        self.assertEqual(self.mesh.euler_number, -2)
        self.assertTrue(self.mesh.is_watertight)
        expected = 100 * 60 * 8 - 2 * math.pi * 3**2 * 8
        self.assertAlmostEqual(self.mesh.volume, expected, delta=1.0)
        np.testing.assert_allclose(
            self.mesh.bounds, [[-50, -30, -4], [50, 30, 4]], atol=0.001
        )

    def test_two_revisions_preserve_other_requirements(self):
        enlarged = replace(DEFAULT_SPEC, hole_diameter_mm=8)
        wider = replace(enlarged, width_mm=70)
        validate_revision(DEFAULT_SPEC, enlarged, {"hole_diameter_mm": 8})
        validate_revision(enlarged, wider, {"width_mm": 70})
        for spec in (enlarged, wider):
            with self.subTest(spec=spec):
                self.assertTrue(verify_plate(build_plate(spec), spec)["accepted"])

    def test_revision_rejects_unrequested_changes(self):
        changed = replace(DEFAULT_SPEC, hole_diameter_mm=8, hole_pitch_mm=58)
        with self.assertRaises(GeometryError):
            validate_revision(DEFAULT_SPEC, changed, {"hole_diameter_mm": 8})
        with self.assertRaises(GeometryError):
            validate_revision(DEFAULT_SPEC, DEFAULT_SPEC, {"hole_diameter_mm": 8})

    def test_missing_holes_are_rejected(self):
        plain = trimesh.creation.box(extents=(100, 60, 8))
        result = verify_plate(plain, DEFAULT_SPEC)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["through_holes"])

    def test_wrong_pitch_fails_despite_matching_volume(self):
        wrong = build_plate(replace(DEFAULT_SPEC, hole_pitch_mm=50))
        self.assertAlmostEqual(wrong.volume, self.mesh.volume, delta=0.01)
        result = verify_plate(wrong, DEFAULT_SPEC)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["checks"]["volume"])
        self.assertFalse(result["checks"]["through_holes"])

    def test_enlarged_holes_fail_wall_probe(self):
        wrong = build_plate(replace(DEFAULT_SPEC, hole_diameter_mm=8))
        result = verify_plate(wrong, DEFAULT_SPEC)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["checks"]["through_holes"])
        self.assertFalse(result["checks"]["hole_walls"])

    def test_blind_holes_fail_through_probe(self):
        base = trimesh.creation.box(extents=(100, 60, 8))
        cuts = []
        for x in (-30, 30):
            cut = trimesh.creation.cylinder(radius=3, height=6, sections=96)
            cut.apply_translation((x, 0, 3))
            cuts.append(cut)
        blind = trimesh.boolean.difference([base, *cuts], engine="manifold")
        result = verify_plate(blind, DEFAULT_SPEC)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["through_holes"])

    def test_volume_compensated_counterbores_fail_wall_probe(self):
        radius = 2.96
        outer = 4.5
        depth = (3**2 - radius**2) * 8 / (outer**2 - radius**2)
        body = trimesh.creation.box(extents=(100, 60, 8))
        cuts = []
        for x in (-30, 30):
            through = trimesh.creation.cylinder(radius=radius, height=10, sections=96)
            through.apply_translation((x, 0, 0))
            mouth = trimesh.creation.cylinder(
                radius=outer, height=depth + 1, sections=96
            )
            mouth.apply_translation((x, 0, 4 + (1 - depth) / 2))
            cuts.extend([through, mouth])
        wrong = trimesh.boolean.difference([body, *cuts], engine="manifold")
        reloaded = trimesh.load_mesh(
            io.BytesIO(wrong.export(file_type="stl")), file_type="stl", process=True
        )
        for candidate in (wrong, reloaded):
            report = verify_plate(candidate, DEFAULT_SPEC)
            self.assertTrue(report["checks"]["volume"])
            self.assertTrue(report["checks"]["through_holes"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["checks"]["hole_walls"])

    def test_wrong_bounds_or_translation_fail(self):
        for wrong in (
            build_plate(replace(DEFAULT_SPEC, width_mm=61)),
            self.mesh.copy(),
        ):
            if np.allclose(wrong.extents, self.mesh.extents):
                wrong.apply_translation((1, 0, 0))
            result = verify_plate(wrong, DEFAULT_SPEC)
            self.assertFalse(result["accepted"])
            self.assertFalse(result["checks"]["bounds"])

    def test_nonwatertight_mesh_fails_before_boolean_probes(self):
        broken = self.mesh.copy()
        broken.update_faces(np.arange(len(broken.faces) - 1))
        result = verify_plate(broken, DEFAULT_SPEC)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["valid_volume"])
        self.assertNotIn("through_holes", result["checks"])

    def test_export_reload_checks_geometry_not_file_bytes(self):
        binary = self.mesh.export(file_type="stl")
        reloaded = trimesh.load_mesh(io.BytesIO(binary), file_type="stl", process=True)
        self.assertTrue(verify_plate(reloaded, DEFAULT_SPEC)["accepted"])
        fresh = build_plate(DEFAULT_SPEC)
        np.testing.assert_allclose(fresh.bounds, reloaded.bounds, atol=0.001)
        self.assertAlmostEqual(fresh.volume, reloaded.volume, delta=0.1)

    def test_invalid_parameters_are_rejected(self):
        base = DEFAULT_SPEC.to_dict()
        invalid = [
            {**base, "width_mm": True},
            {**base, "width_mm": float("nan")},
            {**base, "thickness_mm": -1},
            {**base, "length_mm": 100000},
            {**base, "length_mm": 10**1000},
            {**base, "hole_diameter_mm": 60},
            {**base, "hole_pitch_mm": 99},
            {**base, "hole_pitch_mm": 6},
            {**base, "command": "not a parameter"},
            {key: value for key, value in base.items() if key != "width_mm"},
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(GeometryError):
                PlateSpec.from_dict(item)


if __name__ == "__main__":
    unittest.main()
