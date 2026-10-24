import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_mesh(mesh, spec, path, label):
    image = Image.new("RGB", (1100, 760), "#111827")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25)
    small = ImageFont.load_default(size=18)
    draw.text((40, 32), label, fill="#f9fafb", font=font)
    dimensions = f"{spec.length_mm:g} x {spec.width_mm:g} x {spec.thickness_mm:g} mm | hole diameter {spec.hole_diameter_mm:g} mm | pitch {spec.hole_pitch_mm:g} mm"
    draw.text((40, 78), dimensions, fill="#cbd5e1", font=small)
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
    vertices = mesh.vertices @ rotation.T
    extent = np.ptp(vertices[:, :2], axis=0)
    scale = min(920 / extent[0], 470 / extent[1])
    center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    screen = (vertices[:, :2] - center) * (scale, -scale) + (550, 405)
    normals = mesh.face_normals @ rotation.T
    light = np.array([0.2, -0.3, 0.93])
    light /= np.linalg.norm(light)
    pixels = np.array(image)
    depth = np.full((760, 1100), -np.inf)
    for index, face in enumerate(mesh.faces):
        points = screen[face]
        x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int), (0, 0))
        x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int), (1099, 759))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = points
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-9:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        xx, yy = xx + 0.5, yy + 0.5
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        wc = 1 - wa - wb
        z = (
            wa * vertices[face[0], 2]
            + wb * vertices[face[1], 2]
            + wc * vertices[face[2], 2]
        )
        saved = depth[y0 : y1 + 1, x0 : x1 + 1]
        visible = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9) & (z > saved)
        amount = 0.35 + 0.65 * max(0, float(normals[index] @ light))
        pixels[y0 : y1 + 1, x0 : x1 + 1][visible] = [
            round(channel * amount) for channel in (78, 164, 247)
        ]
        saved[visible] = z[visible]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.text(
        (40, 690),
        "Projection of the generated STL geometry. Not a manufacturing certificate.",
        fill="#94a3b8",
        font=small,
    )
    draw.text((970, 724), "YUFENG", fill="#cbd5e1", font=small)
    image.save(path)
