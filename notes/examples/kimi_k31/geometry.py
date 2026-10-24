import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import trimesh


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class PlateSpec:
    length_mm: float
    width_mm: float
    thickness_mm: float
    hole_diameter_mm: float
    hole_pitch_mm: float

    def __post_init__(self):
        limits = {
            "length_mm": (20, 200),
            "width_mm": (20, 200),
            "thickness_mm": (1, 20),
            "hole_diameter_mm": (1, 20),
            "hole_pitch_mm": (1, 198),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not low <= value <= high
                or not math.isfinite(value)
            ):
                raise GeometryError(f"{name} is outside the supported finite range")
        if self.hole_pitch_mm - self.hole_diameter_mm < 2:
            raise GeometryError("holes need at least 2 mm between their edges")
        if self.length_mm - self.hole_pitch_mm - self.hole_diameter_mm < 2:
            raise GeometryError(
                "holes need at least 1 mm of material at the plate ends"
            )
        if self.width_mm - self.hole_diameter_mm < 2:
            raise GeometryError(
                "holes need at least 1 mm of material at the plate sides"
            )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            field.name for field in fields(cls)
        }:
            raise GeometryError("expected exactly the five plate parameters")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


DEFAULT_SPEC = PlateSpec(100, 60, 8, 6, 60)
SECTIONS = 96


def validate_revision(previous, revised, changes):
    known = previous.to_dict()
    if not isinstance(changes, dict) or not changes or set(changes) - set(known):
        raise GeometryError("invalid requested revision")
    expected = PlateSpec.from_dict({**known, **changes})
    for name, value in expected.to_dict().items():
        if not math.isclose(value, getattr(revised, name), rel_tol=0, abs_tol=1e-8):
            raise GeometryError(f"revision did not preserve {name}")


def build_plate(spec):
    body = trimesh.creation.box(
        extents=(spec.length_mm, spec.width_mm, spec.thickness_mm)
    )
    cuts = []
    for x in (-spec.hole_pitch_mm / 2, spec.hole_pitch_mm / 2):
        cut = trimesh.creation.cylinder(
            radius=spec.hole_diameter_mm / 2,
            height=spec.thickness_mm + 2,
            sections=SECTIONS,
        )
        cut.apply_translation((x, 0, 0))
        cuts.append(cut)
    return trimesh.boolean.difference([body, *cuts], engine="manifold")


def _components(mesh):
    parent = list(range(len(mesh.vertices)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in mesh.edges_unique:
        parent[root(int(a))] = root(int(b))
    return len({root(int(index)) for index in np.unique(mesh.faces)})


def _intersection_volume(mesh, probe):
    result = trimesh.boolean.intersection([mesh, probe], engine="manifold")
    return 0.0 if result.is_empty else float(result.volume)


def verify_plate(mesh, spec):
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError("expected a triangular mesh")
    checks = {
        "nonempty": not mesh.is_empty,
        "bounded_mesh_size": len(mesh.faces) <= 20000,
        "finite_vertices": bool(np.isfinite(mesh.vertices).all()),
    }
    measurements = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}

    def report():
        return {
            "accepted": all(checks.values()),
            "checks": checks,
            "measurements": measurements,
            "failures": [name for name, passed in checks.items() if not passed],
        }

    if not all(checks.values()):
        return report()
    checks["valid_volume"] = bool(mesh.is_volume)
    if not checks["valid_volume"]:
        return report()
    checks["single_body"] = _components(mesh) == 1
    checks["two_handle_topology"] = mesh.euler_number == -2
    half = np.array([spec.length_mm, spec.width_mm, spec.thickness_mm], dtype=float) / 2
    checks["bounds"] = bool(
        np.allclose(mesh.bounds, np.array([-half, half]), atol=0.001, rtol=0)
    )
    removed = 2 * math.pi * (spec.hole_diameter_mm / 2) ** 2 * spec.thickness_mm
    full = spec.length_mm * spec.width_mm * spec.thickness_mm
    expected_volume = full - removed
    volume_tolerance = max(0.005, removed * 0.002, full * 5e-7)
    checks["volume"] = abs(float(mesh.volume) - expected_volume) <= volume_tolerance
    measurements.update(
        {
            "bounds_mm": mesh.bounds.tolist(),
            "volume_mm3": float(mesh.volume),
            "expected_volume_mm3": expected_volume,
            "volume_tolerance_mm3": volume_tolerance,
        }
    )
    through, walls, probes = [], [], []
    for x in (-spec.hole_pitch_mm / 2, spec.hole_pitch_mm / 2):
        radius = spec.hole_diameter_mm / 2
        inside = trimesh.creation.cylinder(
            radius=radius - 0.05, height=spec.thickness_mm + 2, sections=SECTIONS
        )
        inside.apply_translation((x, 0, 0))
        blocked = _intersection_volume(mesh, inside)
        ring = trimesh.creation.annulus(
            r_min=radius + 0.05,
            r_max=radius + 0.10,
            height=spec.thickness_mm,
            sections=SECTIONS,
        )
        ring.apply_translation((x, 0, 0))
        retained = _intersection_volume(mesh, ring)
        tolerance = max(0.0001, float(ring.volume) * 0.0005)
        through.append(blocked <= 0.0001)
        walls.append(abs(retained - float(ring.volume)) <= tolerance)
        probes.append(
            {
                "center_x_mm": x,
                "blocked_volume_mm3": blocked,
                "wall_probe_volume_mm3": float(ring.volume),
                "retained_wall_volume_mm3": retained,
            }
        )
    checks["through_holes"] = all(through)
    checks["hole_walls"] = all(walls)
    measurements["hole_probes"] = probes
    return report()
