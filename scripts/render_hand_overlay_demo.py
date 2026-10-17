#!/usr/bin/env python3
"""Render saved, unaligned DexYCB predictions over the original RGB remotely.

This does not run a model or optimize a pose. The source's disclosed GT crop
is used only for a common display zoom; prediction geometry always uses its
own beta and the original calibrated camera. No GT/PA position correction is
permitted. Rendering is double sided with a perspective depth buffer, and
does not model occlusion by real objects in the RGB image.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes its declared root: ' + str(path))
    return path


class VideoWriter:
    """Raw BGR -> browser-compatible H264/yuv420p, without frame buffering."""

    def __init__(self, encoder, path, width, height, fps, threads=2):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.stem + '.partial.mp4')
        if self.path.exists() or self.temporary.exists():
            raise FileExistsError('Use a fresh output: ' + str(self.path))
        if width % 2 or height % 2:
            raise ValueError('H264/yuv420p dimensions must both be even')
        self.width, self.height, self.count = width, height, 0
        self.log_path = self.path.with_suffix('.ffmpeg.log')
        self.log = self.log_path.open('wb')
        self.command = [str(encoder), '-hide_banner', '-loglevel', 'warning', '-nostdin',
                        '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                        '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
                        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        '-threads', str(threads), '-y', str(self.temporary)]
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=self.log)

    def append(self, frame):
        import numpy as np
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError('Video frame dimensions/type changed')
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        result = self.process.wait(timeout=60)
        self.log.close()
        if result:
            raise RuntimeError('H264 encoding failed; see ' + str(self.log_path))
        self.temporary.replace(self.path)
        return {'path': str(self.path), 'frames': self.count, 'width': self.width,
                'height': self.height, 'sha256': sha(self.path), 'command': self.command}

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        if not self.log.closed:
            self.log.close()


def raster_mesh_legacy(vertices_camera, pixels, faces, shape, color, alpha):
    """Double-sided triangle rasterization with perspective-correct z buffer."""
    import numpy as np
    height, width = shape
    depth = np.full((height, width), np.inf, dtype=np.float32)
    shade = np.zeros((height, width), dtype=np.float32)
    accepted = 0
    for face in faces:
        xyz = vertices_camera[face]
        uv = pixels[face]
        if (not np.isfinite(xyz).all() or not np.isfinite(uv).all()
                or np.any(xyz[:, 2] <= 1e-5)):
            continue
        low = np.maximum(np.floor(uv.min(0)).astype(np.int64), [0, 0])
        high = np.minimum(np.ceil(uv.max(0)).astype(np.int64), [width - 1, height - 1])
        if np.any(high < low):
            continue
        x0, y0 = uv[0]
        x1, y1 = uv[1]
        x2, y2 = uv[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-8:
            continue
        xx = np.arange(low[0], high[0] + 1, dtype=np.float32)[None] + .5
        yy = np.arange(low[1], high[1] + 1, dtype=np.float32)[:, None] + .5
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1 - w0 - w1
        contained = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        inverse_z = w0 / xyz[0, 2] + w1 / xyz[1, 2] + w2 / xyz[2, 2]
        distance = 1 / np.maximum(inverse_z, 1e-12)
        region = depth[low[1]:high[1]+1, low[0]:high[0]+1]
        take = contained & (inverse_z > 0) & (distance < region)
        if not take.any():
            continue
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        # Absolute normal term intentionally illuminates both face windings.
        intensity = .65 + .35 * abs(float(normal[2]))
        region[take] = distance[take]
        shade[low[1]:high[1]+1, low[0]:high[0]+1][take] = intensity
        accepted += 1
    mask = np.isfinite(depth)
    colors = np.asarray(color, dtype=np.float32)[None, None] * shade[..., None]
    return colors, mask, {'rasterized_triangles': accepted, 'visible_mesh_pixels': int(mask.sum()),
                          'alpha': alpha, 'double_sided': True}


def display_wrist_cap(faces):
    """Close the unique MANO wrist boundary using existing vertices only.

    These fan triangles belong to the display mesh, never the predicted mesh
    used for metric decoding. The cycle follows original face winding; the
    cap reverses it. No vertex is added, moved, smoothed, or thickened.
    """
    import numpy as np
    faces = np.asarray(faces, dtype=np.int64)
    counts, directed = {}, {}
    for face in faces:
        for start, end in zip(face, np.roll(face, -1)):
            edge = (min(int(start), int(end)), max(int(start), int(end)))
            counts[edge] = counts.get(edge, 0)+1
            directed[edge] = (int(start), int(end))
    boundary = [directed[edge] for edge, count in counts.items() if count == 1]
    if not boundary:
        raise ValueError('Expected one open wrist boundary, but MANO has no boundary')
    outgoing, incoming = {}, {}
    for start, end in boundary:
        if start in outgoing or end in incoming:
            raise ValueError('MANO wrist boundary is not a consistently oriented cycle')
        outgoing[start], incoming[end] = end, start
    if set(outgoing) != set(incoming):
        raise ValueError('MANO wrist boundary is not closed')
    first = min(outgoing)
    cycle, current = [], first
    while current not in cycle:
        cycle.append(current)
        current = outgoing[current]
    if current != first or len(cycle) != len(boundary) or len(cycle) < 3:
        raise ValueError('Expected exactly one simple MANO wrist boundary cycle')
    cap = np.asarray([(cycle[0], cycle[index+1], cycle[index])
                      for index in range(1, len(cycle)-1)], dtype=np.int64)
    return np.concatenate((faces, cap), axis=0), {
        'original_faces': int(len(faces)), 'added_display_faces': int(len(cap)),
        'boundary_vertex_indices': cycle, 'new_vertices': 0,
        'vertex_positions_changed': False, 'participates_in_metrics': False}


def raster_mesh(vertices_camera, pixels, faces, shape, color, alpha, cap_face_start=None):
    """Smooth, perspective-correct normals on unchanged predicted vertices.

    Area-weighted vertex normals remove triangle facets without moving a
    vertex. Camera-space studio lights and a restrained Blinn-Phong highlight
    are identical for both methods. A z buffer handles self occlusion only.
    """
    import numpy as np
    vertices = np.asarray(vertices_camera, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1]-triangles[:, 0],
                            triangles[:, 2]-triangles[:, 0])
    normals = np.zeros_like(vertices)
    skin_faces = faces if cap_face_start is None else faces[:cap_face_start]
    skin_normals = face_normals if cap_face_start is None else face_normals[:cap_face_start]
    for corner in range(3):
        np.add.at(normals, skin_faces[:, corner], skin_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    cap_normal = None
    if cap_face_start is not None:
        # One soft planar normal for the display cap, independent of skin
        # normals. This adds neither a thick cuff nor a faceted end surface.
        cap_normal = face_normals[cap_face_start:].sum(axis=0)
        cap_normal /= max(float(np.linalg.norm(cap_normal)), 1e-12)
    lights = np.asarray([[-.40, -.62, -.68], [.72, -.12, -.55], [-.22, .72, -.42]],
                        dtype=np.float32)
    lights /= np.linalg.norm(lights, axis=-1, keepdims=True)
    base = np.asarray(color, dtype=np.float32)
    height, width = shape
    depth = np.full((height, width), np.inf, dtype=np.float32)
    colors = np.zeros((height, width, 3), dtype=np.float32)
    accepted = 0
    for face_index, face in enumerate(faces):
        xyz, uv = vertices[face], pixels[face]
        if (not np.isfinite(xyz).all() or not np.isfinite(uv).all()
                or np.any(xyz[:, 2] <= 1e-5)):
            continue
        low = np.maximum(np.floor(uv.min(0)).astype(np.int64), [0, 0])
        high = np.minimum(np.ceil(uv.max(0)).astype(np.int64), [width-1, height-1])
        if np.any(high < low):
            continue
        x0, y0 = uv[0]
        x1, y1 = uv[1]
        x2, y2 = uv[2]
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(float(denominator)) < 1e-8:
            continue
        xx = np.arange(low[0], high[0]+1, dtype=np.float32)[None]+.5
        yy = np.arange(low[1], high[1]+1, dtype=np.float32)[:, None]+.5
        w0 = ((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denominator
        w1 = ((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denominator
        w2 = 1-w0-w1
        contained = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        inverse_z = w0/xyz[0, 2]+w1/xyz[1, 2]+w2/xyz[2, 2]
        distance = 1/np.maximum(inverse_z, 1e-12)
        region = depth[low[1]:high[1]+1, low[0]:high[0]+1]
        take = contained & (inverse_z > 0) & (distance < region)
        if not take.any():
            continue
        # Perspective-correct interpolation, including normal/view direction.
        weights = np.stack((w0[take]/xyz[0, 2], w1[take]/xyz[1, 2],
                            w2[take]/xyz[2, 2]), axis=-1)/inverse_z[take, None]
        normal = (weights @ normals[face] if cap_normal is None or face_index < cap_face_start
                  else np.broadcast_to(cap_normal, (len(weights), 3)).copy())
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        view = -(weights @ xyz)
        view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
        normal *= np.where(np.sum(normal*view, axis=-1, keepdims=True) < 0, -1., 1.)
        diffuse = np.maximum(normal @ lights.T, 0)
        intensity = .34 + diffuse @ np.asarray([.48, .20, .08], dtype=np.float32)
        halfway = view+lights[0]
        halfway /= np.maximum(np.linalg.norm(halfway, axis=-1, keepdims=True), 1e-12)
        highlight = .22*np.maximum(np.sum(normal*halfway, axis=-1), 0)**32
        # The soft grazing highlight clarifies curved fingertips and silhouette.
        grazing = .055*(1-np.clip(np.sum(normal*view, axis=-1), 0, 1))**3
        shaded = base[None]*intensity[:, None] + 255*(highlight+grazing)[:, None]
        colors[low[1]:high[1]+1, low[0]:high[0]+1][take] = np.clip(shaded, 0, 255)
        region[take] = distance[take]
        accepted += 1
    mask = np.isfinite(depth)
    return colors, mask, {'rasterized_triangles': accepted, 'visible_mesh_pixels': int(mask.sum()),
                          'alpha': alpha, 'double_sided': True}


def project(points, camera_to_world, intrinsics):
    import numpy as np
    rotation = camera_to_world[:3, :3].T
    camera = (points - camera_to_world[:3, 3]) @ rotation.T
    homogeneous = camera @ intrinsics.T
    pixels = homogeneous[:, :2] / np.maximum(camera[:, 2:3], 1e-5)
    return camera, pixels


def overlay(image, vertices, joints, faces, c2w, intrinsics, bbox, alpha,
            style='polished', supersample=2, show_skeleton=False, cap_face_start=None):
    import cv2
    import numpy as np
    # Both methods share material, lighting, opacity, and sampling settings.
    color = (235, 210, 40) if style == 'legacy' else (226, 190, 96)
    camera, uv = project(vertices, c2w, intrinsics)
    camera_joints, joint_uv = project(joints, c2w, intrinsics)
    sample = 1 if style == 'legacy' else supersample
    height, width = image.shape[:2]
    render_shape = (height*sample, width*sample)
    if style == 'legacy':
        colors, high_mask, diagnostics = raster_mesh_legacy(camera, uv*sample, faces, render_shape, color, alpha)
    else:
        colors, high_mask, diagnostics = raster_mesh(camera, uv*sample, faces, render_shape,
                                                     color, alpha, cap_face_start)
    result = image.copy()
    if sample != 1:
        # Downsample only premultiplied foreground and fractional coverage.
        # Resampling the RGB background would alter pixels outside the mesh.
        foreground = cv2.resize(colors*high_mask[..., None], (width, height), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(high_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        mask = coverage > 0
        result[mask] = np.clip(image[mask]*(1-alpha*coverage[mask, None])+
                               alpha*foreground[mask], 0, 255).astype(np.uint8)
        diagnostics['visible_mesh_pixels_supersampled'] = diagnostics['visible_mesh_pixels']
        diagnostics['visible_mesh_pixels'] = int(mask.sum())
    else:
        mask = high_mask
        result[mask] = np.clip((1-alpha)*image[mask]+alpha*colors[mask], 0, 255).astype(np.uint8)
    positive = camera_joints[:, 2] > 1e-5
    in_image = positive & (joint_uv[:, 0] >= 0) & (joint_uv[:, 0] < width)
    in_image &= (joint_uv[:, 1] >= 0) & (joint_uv[:, 1] < height)
    if show_skeleton or style == 'legacy':
        for base in (1, 5, 9, 13, 17):
            chain = (0, base, base+1, base+2, base+3)
            for a, b in zip(chain[:-1], chain[1:]):
                if positive[a] and positive[b]:
                    points = np.clip(joint_uv[[a, b]], -100000, 100000).round().astype(np.int32)
                    cv2.line(result, tuple(points[0]), tuple(points[1]), (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.line(result, tuple(points[0]), tuple(points[1]), color, 1, cv2.LINE_AA)
        for point in joint_uv[in_image].round().astype(np.int32):
            cv2.circle(result, tuple(point), 2, (255, 255, 255), -1, cv2.LINE_AA)
    x0, y0, x1, y1 = bbox
    yy, xx = np.nonzero(mask)
    inside_box = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    diagnostics.update(positive_depth_vertices=int((camera[:, 2] > 1e-5).sum()),
        total_vertices=len(camera), positive_depth_joints=int(positive.sum()),
        in_image_joints=int(in_image.sum()), total_joints=len(camera_joints),
        material=style, supersample=sample, skeleton=show_skeleton or style == 'legacy',
        added_display_wrist_cap_faces=0 if cap_face_start is None else int(len(faces)-cap_face_start),
        mesh_pixels_inside_input_bbox=int(inside_box.sum()),
        mesh_pixel_fraction_inside_input_bbox=float(inside_box.mean()) if len(xx) else None)
    return result, diagnostics, joint_uv, positive


def shared_zoom_box(boxes, ratio):
    """Common zoom from input boxes alone, never from predicted geometry."""
    import numpy as np
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    center = (low + high) / 2
    size = np.maximum((high - low) * 1.35, 16)
    size[0] = max(size[0], size[1] * ratio)
    size[1] = max(size[1], size[0] / ratio)
    return np.concatenate((center - size/2, center + size/2))


def zoom_transform(box, width, height):
    """Display-only affine transform; never operates on predicted 3D states."""
    import numpy as np
    x0, y0, x1, y1 = box
    return np.asarray([[width/(x1-x0), 0, -x0*width/(x1-x0)],
                       [0, height/(y1-y0), -y0*height/(y1-y0)],
                       [0, 0, 1]], dtype=np.float32)


def zoom(image, box, width, height, background=(247, 246, 243)):
    import cv2
    import numpy as np
    transform = zoom_transform(box, width, height)[:2]
    return cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=background)


def compose_legacy(images, box, width, full_height, zoom_height, title, frame_id):
    import cv2
    import numpy as np
    header, zoom_header, footer = 44, 30, 48
    height = header + full_height + zoom_header + zoom_height + footer
    canvas = np.full((height, width*3, 3), 20, dtype=np.uint8)
    for column, (image, label) in enumerate(zip(images, ('Original RGB', 'HaWoR baseline', 'HandLoop best4200'))):
        left = column * width
        cv2.putText(canvas, label, (left+14, 29), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (245, 245, 245), 2, cv2.LINE_AA)
        full_panel = cv2.resize(image, (width, full_height))
        # This shared frame is a display crop marker, not a pose adjustment.
        scale_x, scale_y = width/image.shape[1], full_height/image.shape[0]
        x0, y0, x1, y1 = box
        cv2.rectangle(full_panel, (int(round(x0*scale_x)), int(round(y0*scale_y))),
                      (int(round(x1*scale_x)), int(round(y1*scale_y))),
                      (215, 215, 215), 1, cv2.LINE_AA)
        canvas[header:header+full_height, left:left+width] = full_panel
        cv2.putText(canvas, 'Shared input crop - same scale and position',
                    (left+10, header+full_height+21), cv2.FONT_HERSHEY_SIMPLEX, .44,
                    (210, 210, 210), 1, cv2.LINE_AA)
        start = header + full_height + zoom_header
        canvas[start:start+zoom_height, left:left+width] = zoom(image, box, width, zoom_height, (18, 18, 18))
    start = height - footer
    cv2.putText(canvas, f'{title} | source frame {frame_id} | nominal 30 fps',
                (12, start+18), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Unaligned prediction | Same cyan overlay / alpha | Object occlusion unavailable (see-through overlay)',
                (12, start+38), cv2.FONT_HERSHEY_SIMPLEX, .45, (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def compose(images, box, width, full_height, zoom_height, title, frame_id,
            zoom_images=None, error_summary=None, selected=False, show_skeleton=False):
    """Quiet, bright comparison cards with a large, independently rasterized detail."""
    import cv2
    import numpy as np
    header, zoom_header, footer = 100, 34, 74
    height = header+full_height+zoom_header+zoom_height+footer
    canvas = np.full((height, width*3, 3), (247, 246, 243), dtype=np.uint8)
    ink, secondary, divider = (54, 48, 43), (113, 105, 96), (220, 216, 208)

    def text_line(text, x, y, size=.55, color=ink, thickness=1, max_width=None):
        if max_width is not None:
            size = min(size, max_width/max(cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX,
                                                         1., thickness)[0][0], 1))
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thickness, cv2.LINE_AA)

    text_line(title, 18, 29, .72, thickness=2, max_width=width*3-36)
    labels = ('Original RGB', 'HaWoR', 'HandLoop')
    summary = error_summary or {}
    metric = summary.get('metric', 'Absolute MPJPE')
    unit = summary.get('unit', 'mm')
    for column, (image, label) in enumerate(zip(images, labels)):
        left = column*width
        text_line(label, left+18, 64, .76, thickness=2)
        value = summary.get(('baseline', 'handloop')[column-1]) if column else None
        subtitle = (f'Segment mean {metric}: {float(value):.1f} {unit}' if value is not None
                    else 'Calibrated held-out test video' if column == 0 else 'Saved unaligned prediction')
        text_line(subtitle, left+18, 87, .45, secondary, max_width=width-36)
        # Fit, rather than stretch, the context RGB into the short context row.
        scale = min(width/image.shape[1], full_height/image.shape[0])
        fitted_width = int(round(image.shape[1]*scale))
        fitted_height = int(round(image.shape[0]*scale))
        offset_x, offset_y = (width-fitted_width)//2, (full_height-fitted_height)//2
        context = cv2.resize(image, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = box
        cv2.rectangle(context, (int(round(x0*scale)), int(round(y0*scale))),
                      (int(round(x1*scale)), int(round(y1*scale))), (232, 227, 212), 1, cv2.LINE_AA)
        canvas[header+offset_y:header+offset_y+fitted_height,
               left+offset_x:left+offset_x+fitted_width] = context
        text_line('HAND DETAIL  /  same crop and scale', left+18,
                  header+full_height+23, .45, secondary, max_width=width-36)
        start = header+full_height+zoom_header
        detail = zoom_images[column] if zoom_images is not None else zoom(image, box, width, zoom_height)
        canvas[start:start+zoom_height, left:left+width] = detail
        if column:
            cv2.line(canvas, (left, 44), (left, height-footer), divider, 1, cv2.LINE_AA)
    start = height-footer
    cv2.line(canvas, (0, start), (width*3, start), divider, 1, cv2.LINE_AA)
    choice = 'Selected qualitative test case' if selected else 'Held-out test segment'
    text_line(f'{choice} | source frame {frame_id} | nominal 30 fps', 18, start+24,
              .47, secondary, max_width=width*3-36)
    skeleton_note = ' | X-ray skeleton diagnostic' if show_skeleton else ''
    text_line('Same material and opacity | No 3D alignment | Object occlusion unavailable (see-through overlay)'+skeleton_note,
              18, start+48, .45, secondary, max_width=width*3-36)
    text_line('Controlled GT-crop evaluation; HaWoR official baseline and HandLoop best4200',
              18, start+65, .37, secondary, max_width=width*3-36)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-root', 'evaluation-root', 'selection', 'mano-models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--encoder', type=Path, help='FFmpeg executable with libx264')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--decode-batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--panel-width', type=int, default=640)
    parser.add_argument('--zoom-height', type=int, default=440)
    parser.add_argument('--full-context-height', type=int, default=300,
                        help='Polished layout fits original RGB without stretching into this short context row')
    parser.add_argument('--zoom-mode', choices=('per-frame', 'segment'), default='per-frame',
                        help='Shared input-bbox zoom per frame; segment preserves the v1 stable union crop')
    parser.add_argument('--render-style', choices=('polished', 'legacy'), default='polished',
                        help='Polished smooth normals/material and bright detail-first layout; legacy preserves faceted X-ray presentation')
    parser.add_argument('--supersample', type=int, choices=(1, 2, 3), default=2,
                        help='Mesh edge antialiasing scale; ignored by legacy')
    parser.add_argument('--show-skeleton', action='store_true',
                        help='Add the optional X-ray skeleton diagnostic; polished default is mesh only')
    parser.add_argument('--open-wrist', action='store_true',
                        help='Keep the original open MANO wrist; polished default adds a display-only cap from existing vertices (legacy is always open)')
    parser.add_argument('--alpha', type=float, default=.76)
    args = parser.parse_args()
    if not Path('/sciclone/data10/jchen57').is_dir():
        parser.error('Run image/MANO rendering only inside the authorized remote compute Pod')
    if not 0 < args.alpha < 1 or min(args.decode_batch_size, args.cpu_threads) < 1:
        parser.error('Invalid alpha/batch/thread count')
    if args.panel_width < 320 or args.panel_width % 2 or args.zoom_height < 64 or args.zoom_height % 2:
        parser.error('Use even panel/zoom dimensions with panel width >=320')
    if args.full_context_height < 64 or args.full_context_height % 2:
        parser.error('Use an even full context height >=64')
    root = args.root.resolve()
    for name in ('source_root', 'evaluation_root', 'mano_models'):
        path = getattr(args, name).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            parser.error(name + ' must be an existing directory inside --root')
        setattr(args, name, path)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metadata.json').exists():
        raise FileExistsError('Use a fresh demo output folder')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.cpu_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(args.cpu_threads))
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from handloop.visual.codec import MANO21_THUMB_FIRST_ORDER, MANOStateCodec
    cv2.setNumThreads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    encoder = args.encoder or shutil.which('ffmpeg')
    if encoder is None:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError('Supply --encoder or install imageio-ffmpeg remotely') from exc
    encoder = str(Path(encoder).resolve())
    encoder_version = subprocess.run([encoder, '-version'], check=True, capture_output=True,
                                     text=True).stdout.splitlines()[0]
    selection = read(args.selection)
    entries = selection.get('entries', [])
    if not entries:
        raise ValueError('Selection must contain a nonempty entries list')
    started = time.monotonic()
    report = {'schema': 'handloop.hand_overlay_demo.v2', 'phase': 'initializing',
        'started_utc': utc(), 'selection_path': str(args.selection.resolve()),
        'selection_sha256': sha(args.selection), 'script_sha256': sha(__file__),
        'encoder_version': encoder_version, 'device': args.device, 'segments': [],
        'rendering': {'alpha': args.alpha,
            'color_bgr': [235, 210, 40] if args.render_style == 'legacy' else [226, 190, 96],
            'style': args.render_style, 'supersample': 1 if args.render_style == 'legacy' else args.supersample,
            'skeleton': args.show_skeleton or args.render_style == 'legacy',
            'normals': 'flat face' if args.render_style == 'legacy' else 'area-weighted vertex normals, perspective-correct pixel interpolation',
            'material': 'legacy diffuse' if args.render_style == 'legacy' else 'camera-space three-light soft diffuse + restrained Blinn-Phong specular',
            'geometry_modification': ('prediction vertices/joints and metric geometry unmodified; display-only wrist cap triangles appended using existing boundary vertices'
                                      if args.render_style != 'legacy' and not args.open_wrist else
                                      'none; only normals, shading and display sampling change'),
            'wrist_cap': args.render_style != 'legacy' and not args.open_wrist,
            'wrist_cap_policy': 'unique original boundary cycle; fan triangulation using existing vertices only; independent planar cap normal; excluded from metric geometry',
            'background_sampling': 'original target RGB preserved outside mesh coverage; foreground-only supersampling (optional skeleton is a diagnostic overlay)',
            'detail_rendering': 'zoom of native overlay' if args.render_style == 'legacy' else 'direct rasterization with display affine @ original K',
            'full_context_height': args.full_context_height,
            'zoom_height': args.zoom_height,
            'double_sided': True, 'self_depth_buffer': True, 'object_occlusion_available': False,
            'camera': 'original full-resolution calibrated K and inverse(camera_to_world)',
            'predicted_shape': 'each model own saved beta; left_shapedirs_fix=True',
            'alignment': 'none: no PA, GT wrist translation, rotation or scale correction',
            'zoom_mode': args.zoom_mode,
            'zoom': ('same per-frame crop in all three columns derived only from current common input GT bbox'
                     if args.zoom_mode == 'per-frame' else
                     'same stable segment crop derived only from union of common input GT bboxes'),
            'protocol': 'saved GT-crop full-test predictions, bidirectional16 blocks / reset32 feedback; not causal detector demo'},
        'fps': 30, 'fps_source': 'DexYCB nominal 30Hz source frame index, not measured timestamps',
        'diagnostics_definition': {
            'mesh_pixel_fraction_inside_input_bbox': 'visible raster mesh pixels inside common input GT bbox / visible mesh pixels; NOT segmentation IoU',
            'projected_joint_error_px': 'predicted21 vs source-provided21 projection with real K, positive-depth pairs only; display geometry is never adjusted'},
        'mano_sha256': {side: sha(args.mano_models/f'MANO_{side}.pkl') for side in ('RIGHT', 'LEFT')}}
    write(args.output/'status.json', report)
    device = torch.device(args.device)
    codec = MANOStateCodec(args.mano_models/'MANO_RIGHT.pkl', args.mano_models/'MANO_LEFT.pkl',
                           left_shapedirs_fix=True).to(device).eval()
    plans = []
    for number, entry in enumerate(entries):
        key = entry['key']
        if not key or '/' in key or '\\' in key or key in ('.', '..'):
            raise ValueError('Unsafe shard key')
        manifest_path = inside(args.source_root, entry['manifest'])
        manifest = read(manifest_path)
        if manifest.get('phase') != 'complete' or manifest.get('shard_id') != key or len(manifest['records']) != 1:
            raise ValueError('Expected one completed selected shard')
        record = manifest['records'][0]
        if record['split'] != 'test':
            raise ValueError('Only held-out test video is allowed for this demo')
        reference_path = inside(manifest_path.parent, record['reference_file'])
        if sha(reference_path) != record['reference_sha256']:
            raise ValueError('Source reference changed')
        with np.load(reference_path, allow_pickle=False) as source:
            # GT beta/target states are deliberately never loaded.
            arrays = {name: source[name].copy() for name in
                      ('frame_ids', 'timestamps', 'camera_intrinsics', 'camera_to_world', 'handedness', 'bbox_xyxy')}
            if 'joint_3d' in source:
                arrays['diagnostic_reference_joints'] = source['joint_3d'].copy()
        ids = arrays['frame_ids']
        if len(ids) != record['frames'] or len(record['image_paths']) != len(ids):
            raise ValueError('Source frame/image membership changed')
        take = np.ones(len(ids), dtype=bool)
        if 'start_frame_id' in entry:
            take &= ids >= int(entry['start_frame_id'])
        if 'end_frame_id' in entry:
            take &= ids <= int(entry['end_frame_id'])
        positions = np.flatnonzero(take)
        if 'max_frames' in entry:
            if int(entry['max_frames']) < 1:
                raise ValueError('max_frames must be positive')
            positions = positions[:int(entry['max_frames'])]
        if not len(positions) or (len(positions) > 1 and not np.all(np.diff(ids[positions]) == 1)):
            raise ValueError('Each selected segment must have consecutive source frame IDs, without gaps')
        summary = entry.get('error_summary')
        if summary is not None:
            if not isinstance(summary, dict):
                raise ValueError('error_summary must be an object with metric/unit/baseline/handloop')
            for label in ('baseline', 'handloop'):
                if label not in summary or not np.isfinite(float(summary[label])) or float(summary[label]) < 0:
                    raise ValueError('error_summary must contain finite, nonnegative baseline/handloop values')
            if 'frames' in summary and int(summary['frames']) != len(positions):
                raise ValueError('error_summary frames must describe exactly the selected segment')
        if not np.allclose(np.diff(arrays['timestamps'][positions]), 1/30, rtol=0, atol=1e-9):
            raise ValueError('Selected timestamps disagree with nominal 30Hz')
        if not np.all(arrays['handedness'] == arrays['handedness'][0]):
            raise ValueError('Hand identity changed within source shard')
        predictions, prediction_meta = {}, {}
        for label, branch in (('baseline', 'raw'), ('best', 'final')):
            path = inside(args.evaluation_root, f'{label}/shards/{key}/predictions.npz')
            result_path = path.parent/'result.json'
            result = read(result_path)
            if result['phase'] != 'complete' or result['source_manifest_sha256'] != sha(manifest_path):
                raise ValueError('Predictions came from another source manifest')
            digest = sha(path)
            if result['predictions_sha256'] != digest:
                raise ValueError('Saved prediction archive changed')
            with np.load(path, allow_pickle=False) as archive:
                if not np.array_equal(archive['frame_ids'], ids):
                    raise ValueError('Saved predictions are not exactly frame-ID aligned')
                state, beta, available = (archive[name][positions].copy()
                                          for name in (branch, 'beta', 'availability'))
            if state.shape != (len(positions), 99) or beta.shape != (len(positions), 10):
                raise ValueError('Invalid predicted state/beta shape')
            if not np.isfinite(state).all() or not np.isfinite(beta).all() or not available.all():
                raise ValueError('Selected prediction geometry is unavailable/nonfinite')
            predictions[label] = (state, beta)
            prediction_meta[label] = {'path': str(path), 'sha256': digest, 'branch': branch,
                                      'model_identity': result['identity']}
        plans.append((entry, manifest_path, record, arrays, positions, predictions, prediction_meta))
    total_frames = sum(len(plan[4]) for plan in plans)
    sample_indices = {0: 'first', total_frames//2: 'middle', total_frames-1: 'last'}
    writers, contact, segment_contact_images = [], {}, []
    global_index = 0
    image_identity = hashlib.sha256()
    try:
        for segment_index, (entry, manifest_path, record, arrays, positions, predictions, prediction_meta) in enumerate(plans):
            decode_start = time.monotonic()
            handedness = arrays['handedness'][positions].astype(bool)
            side = 'right' if handedness[0] else 'left'
            faces = codec.layers[side].faces.copy()
            cap_face_start, cap_details = None, None
            if args.render_style != 'legacy' and not args.open_wrist:
                cap_face_start = len(faces)
                faces, cap_details = display_wrist_cap(faces)
            geometry = {}
            for label, (state, beta) in predictions.items():
                vertex_parts, joint_parts = [], []
                for begin in range(0, len(positions), args.decode_batch_size):
                    end = min(begin+args.decode_batch_size, len(positions))
                    with torch.inference_mode():
                        decoded = codec.geometry(torch.as_tensor(state[begin:end], device=device),
                            torch.as_tensor(beta[begin:end], device=device),
                            torch.as_tensor(handedness[begin:end], device=device))
                    vertices, joints = decoded['vertices'].cpu().numpy(), decoded['joints'].cpu().numpy()
                    tips = [745, 317, 444 if handedness[0] else 445, 556, 673]
                    points = np.concatenate((joints, vertices[:, tips]), axis=1)
                    vertex_parts.append(vertices)
                    joint_parts.append(points[:, MANO21_THUMB_FIRST_ORDER])
                geometry[label] = {'vertices': np.concatenate(vertex_parts), 'joints': np.concatenate(joint_parts)}
            decode_seconds = time.monotonic() - decode_start
            box = shared_zoom_box(arrays['bbox_xyxy'][positions], args.panel_width/args.zoom_height)
            segment = {'index': segment_index+1, 'key': entry['key'], 'manifest': str(manifest_path),
                'manifest_sha256': sha(manifest_path), 'reference_sha256': record['reference_sha256'],
                'physical_recording': record.get('physical_recording'), 'hand': side,
                'frame_id_first': int(arrays['frame_ids'][positions[0]]),
                'frame_id_last': int(arrays['frame_ids'][positions[-1]]), 'frames': len(positions),
                'frame_ids_sha256': hashlib.sha256(arrays['frame_ids'][positions].astype('<i8').tobytes()).hexdigest(),
                'nominal_fps': 30, 'timestamp_source': record['timestamp_source'],
                'prediction_sources': prediction_meta, 'zoom_mode': args.zoom_mode,
                'title': entry.get('title'), 'error_summary': entry.get('error_summary'),
                'selection_reason': entry.get('selection_reason'),
                'display_wrist_cap': cap_details,
                'segment_union_zoom_box_xyxy': box.tolist(),
                'decode_seconds': decode_seconds, 'frame_diagnostics': []}
            render_start = time.monotonic()
            for local_index, position in enumerate(positions):
                frame_id = int(arrays['frame_ids'][position])
                relative = record['image_paths'][position]
                image_path = inside(manifest_path.parent, relative)
                image_sha = sha(image_path)
                if image_sha != record['image_sha256'][relative]:
                    raise ValueError('Source RGB changed: ' + str(image_path))
                image_identity.update(f'{entry["key"]}/{frame_id}:{image_sha}\n'.encode())
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError('Cannot decode source RGB: ' + str(image_path))
                height, width = image.shape[:2]
                full_height = (int(round(height*args.panel_width/width/2))*2
                               if args.render_style == 'legacy' else args.full_context_height)
                if not writers:
                    comparison_height = ((44+full_height+30+args.zoom_height+48)
                                         if args.render_style == 'legacy' else
                                         (100+full_height+34+args.zoom_height+74))
                    writers = [VideoWriter(encoder, args.output/'comparison.mp4', args.panel_width*3, comparison_height, 30),
                               VideoWriter(encoder, args.output/'original.mp4', width, height, 30),
                               VideoWriter(encoder, args.output/'handloop_overlay.mp4', width, height, 30)]
                    report['native_image_dimensions'] = [width, height]
                if [width, height] != report['native_image_dimensions']:
                    raise ValueError('All selected source clips must have matching original RGB dimensions')
                frame_box = (shared_zoom_box(arrays['bbox_xyxy'][position:position+1],
                                             args.panel_width/args.zoom_height)
                             if args.zoom_mode == 'per-frame' else box)
                images, diagnostic = [image], {'source_frame_id': frame_id,
                                               'common_zoom_box_xyxy': frame_box.tolist()}
                zoom_images = None
                if args.render_style == 'polished':
                    zoom_images = [zoom(image, frame_box, args.panel_width, args.zoom_height)]
                    display_affine = zoom_transform(frame_box, args.panel_width, args.zoom_height)
                    detail_intrinsics = display_affine @ arrays['camera_intrinsics'][position]
                    x0, y0, x1, y1 = arrays['bbox_xyxy'][position]
                    detail_bbox_points = np.asarray([[x0, y0, 1.], [x1, y1, 1.]]) @ display_affine.T
                    detail_bbox = detail_bbox_points[:, :2].reshape(-1)
                for label in ('baseline', 'best'):
                    rendered, stats, joint_uv, positive = overlay(image,
                        geometry[label]['vertices'][local_index], geometry[label]['joints'][local_index], faces,
                        arrays['camera_to_world'][position], arrays['camera_intrinsics'][position],
                        arrays['bbox_xyxy'][position], args.alpha, args.render_style,
                        args.supersample, args.show_skeleton, cap_face_start)
                    if stats['positive_depth_vertices'] != 778 or stats['visible_mesh_pixels'] == 0:
                        raise ValueError(f'Projected mesh has nonpositive depth or zero visible pixels: {entry["key"]}/{frame_id}/{label}')
                    if 'diagnostic_reference_joints' in arrays:
                        gt_camera, gt_uv = project(arrays['diagnostic_reference_joints'][position],
                            arrays['camera_to_world'][position], arrays['camera_intrinsics'][position])
                        valid = positive & (gt_camera[:, 2] > 1e-5) & np.isfinite(gt_uv).all(-1)
                        stats['projected_joint_error_px'] = float(np.linalg.norm(joint_uv[valid]-gt_uv[valid], axis=-1).mean()) if valid.any() else None
                        stats['projected_joint_error_pairs'] = int(valid.sum())
                    diagnostic[label] = stats
                    images.append(rendered)
                    if zoom_images is not None:
                        detail, detail_stats, _, _ = overlay(zoom_images[0],
                            geometry[label]['vertices'][local_index], geometry[label]['joints'][local_index], faces,
                            arrays['camera_to_world'][position], detail_intrinsics,
                            detail_bbox, args.alpha, args.render_style, args.supersample,
                            args.show_skeleton, cap_face_start)
                        zoom_images.append(detail)
                        stats['detail_visible_mesh_pixels'] = detail_stats['visible_mesh_pixels']
                title = (str(entry['title']) if entry.get('title') else
                         f'Segment {segment_index+1}/{len(plans)} | {side} hand')
                if args.render_style == 'legacy':
                    comparison = compose_legacy(images, frame_box, args.panel_width, full_height,
                                                args.zoom_height, title, frame_id)
                else:
                    comparison = compose(images, frame_box, args.panel_width, full_height, args.zoom_height,
                        title, frame_id, zoom_images, entry.get('error_summary'),
                        selected=bool(selection.get('selection_uses_test_error') or entry.get('error_summary')),
                        show_skeleton=args.show_skeleton)
                for writer, frame in zip(writers, (comparison, image, images[-1])):
                    writer.append(frame)
                if global_index in sample_indices:
                    name = sample_indices[global_index]
                    path = args.output/f'comparison_{name}.png'
                    if not cv2.imwrite(str(path), comparison):
                        raise RuntimeError('Contact image write failed')
                    contact[global_index] = comparison.copy()
                if local_index == len(positions)//2:
                    path = args.output/f'comparison_segment{segment_index+1}_middle.png'
                    if not cv2.imwrite(str(path), comparison):
                        raise RuntimeError('Segment contact image write failed')
                    segment_contact_images.append(str(path))
                    contact[global_index] = comparison.copy()
                segment['frame_diagnostics'].append(diagnostic)
                global_index += 1
                if global_index % 30 == 0:
                    write(args.output/'status.json', {'phase': 'rendering', 'frames': global_index,
                        'total_frames': total_frames, 'elapsed_seconds': time.monotonic()-started,
                        'updated_utc': utc()})
                    print(json.dumps({'phase': 'rendering', 'frames': global_index, 'total_frames': total_frames}), flush=True)
            segment['render_and_encode_seconds'] = time.monotonic()-render_start
            report['segments'].append(segment)
        if global_index != total_frames:
            raise ValueError('Output frame coverage changed')
        report['videos'] = [writer.close() for writer in writers]
        writers = []
        # Keep the normal-speed artifact unchanged. This explicitly named
        # derivative doubles presentation timestamps and repeats frames to
        # retain a standard 30fps H264 stream; geometry/pixels are not rerun.
        slow_path = args.output/'comparison_slow_0p5x.mp4'
        if slow_path.exists():
            raise FileExistsError('Use a fresh slow-motion output')
        slow_command = [encoder, '-hide_banner', '-loglevel', 'warning', '-nostdin',
            '-i', str(args.output/'comparison.mp4'), '-vf', 'setpts=2.0*PTS',
            '-r', '30', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2', str(slow_path)]
        slow_result = subprocess.run(slow_command, capture_output=True, text=True, timeout=60)
        if slow_result.returncode:
            raise RuntimeError('Slow-motion encoding failed: ' + slow_result.stderr[-2000:])
        report['slow_motion'] = {'path': str(slow_path), 'sha256': sha(slow_path),
            'playback_speed': .5, 'source_unique_frames': total_frames,
            'nominal_duration_seconds': total_frames/15,
            'frame_policy': 'setpts=2.0*PTS, output30fps repeats source frames; no interpolated geometry',
            'command': slow_command}
        snapshots = [contact[index] for index in sorted(contact)]
        sheet_width = 1200
        sheet = np.concatenate([cv2.resize(frame, (sheet_width, int(round(frame.shape[0]*sheet_width/frame.shape[1])))) for frame in snapshots], axis=0)
        if not cv2.imwrite(str(args.output/'contact_sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError('Combined contact sheet write failed')
        report.update(phase='complete', finished_utc=utc(), frames=global_index,
            duration_seconds=global_index/30, elapsed_seconds=time.monotonic()-started,
            source_rgb_identity_sha256=image_identity.hexdigest(),
            contact_images=[str(args.output/f'comparison_{name}.png') for name in sample_indices.values()],
            segment_contact_images=segment_contact_images,
            contact_sheet=str(args.output/'contact_sheet.jpg'))
        write(args.output/'metadata.json', report)
        write(args.output/'status.json', {'phase': 'complete', 'frames': global_index,
            'duration_seconds': global_index/30, 'finished_utc': utc()})
        print(json.dumps({'phase': 'complete', 'frames': global_index, 'output': str(args.output)}), flush=True)
    except BaseException as exc:
        for writer in writers:
            writer.abort()
        report.update(phase='failed', error=repr(exc), failed_utc=utc(),
                      frames=global_index, elapsed_seconds=time.monotonic()-started)
        write(args.output/'metadata_failed.json', report)
        write(args.output/'status.json', {'phase': 'failed', 'error': repr(exc), 'frames': global_index})
        raise


if __name__ == '__main__':
    main()
