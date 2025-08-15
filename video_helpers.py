import math
import random
from typing import Tuple

from moviepy.editor import ImageClip, CompositeVideoClip, vfx, afx


def compute_crossfade(durations: list[float]) -> float:
    if not durations:
        return 0.5
    return min(0.5, max(0.1, min(durations) / 4))


def make_kenburns_with_random_pan_clip(
    image_path: str,
    audio_clip,
    target_size: Tuple[int, int] = (1920, 1080),
    zoom_start: float = 1.05,
    zoom_end: float = 1.15,
    fade_seconds: float = 0.5,
    pan_max_px: float = 60.0,
    seed: int | None = None,
):
    """
    Create a Ken Burns-style clip (slow zoom) with a subtle random pan direction.

    - image is scaled to cover `target_size`
    - zooms from `zoom_start` to `zoom_end`
    - pans up to `pan_max_px` pixels across the clip duration in a random 360° direction
    - applies fade in/out to both video and audio
    """
    duration = max(0.01, getattr(audio_clip, "duration", 0.01) or 0.01)
    fade_seconds = min(fade_seconds, duration / 3)

    if seed is not None:
        random.seed(seed)

    # Random pan direction and distance
    theta = random.uniform(0.0, 2.0 * math.pi)
    pan_distance = random.uniform(0.4, 1.0) * pan_max_px
    pan_dx_final = math.cos(theta) * pan_distance
    pan_dy_final = math.sin(theta) * pan_distance

    base_img = ImageClip(image_path)
    base_w, base_h = base_img.w, base_img.h
    target_w, target_h = target_size

    # Scale to fully cover frame, add small overscale to avoid edge reveal while panning
    cover_scale = max(target_w / base_w, target_h / base_h)
    overscale = 1.04

    # Compute frame sizes at start and end of the zoom to constrain pan so edges never reveal
    min_scale = cover_scale * overscale * zoom_start
    max_scale = cover_scale * overscale * zoom_end
    frame_w_start = base_w * min_scale
    frame_h_start = base_h * min_scale
    frame_w_end = base_w * max_scale
    frame_h_end = base_h * max_scale
    # Use the smallest frame across the zoom range as the worst case for edge reveal
    min_frame_w = min(frame_w_start, frame_w_end)
    min_frame_h = min(frame_h_start, frame_h_end)

    # How far we can move from center without exposing borders at the END (worst case for final offset)
    # Keep a tiny margin to avoid rounding artifacts
    edge_margin = 2.0
    max_pan_x_allowed = max(0.0, (min_frame_w - target_w) / 2.0 - edge_margin)
    max_pan_y_allowed = max(0.0, (min_frame_h - target_h) / 2.0 - edge_margin)

    if max_pan_x_allowed <= 0.0 and max_pan_y_allowed <= 0.0:
        # No panning possible; force zero
        pan_dx_final = 0.0
        pan_dy_final = 0.0
    else:
        # Scale down the random pan so it fits within allowed bounds on both axes
        scale_limit = 1.0
        if abs(pan_dx_final) > 1e-6:
            scale_limit = min(scale_limit, max_pan_x_allowed / abs(pan_dx_final))
        # If dx is ~0, y bound alone matters
        if abs(pan_dy_final) > 1e-6:
            scale_limit = min(scale_limit, max_pan_y_allowed / abs(pan_dy_final))
        scale_limit = max(0.0, min(1.0, scale_limit))
        pan_dx_final *= scale_limit
        pan_dy_final *= scale_limit

    def scale_at(t: float) -> float:
        z = zoom_start + (zoom_end - zoom_start) * (t / duration)
        return cover_scale * overscale * z

    def position_at(t: float) -> tuple[float, float]:
        s = scale_at(t)
        frame_w = base_w * s
        frame_h = base_h * s
        center_x = (target_w - frame_w) / 2.0
        center_y = (target_h - frame_h) / 2.0
        k = (t / duration)
        x = center_x + pan_dx_final * k
        y = center_y + pan_dy_final * k

        # Clamp position so the scaled image always fully covers the frame
        min_x = target_w - frame_w
        max_x = 0.0
        min_y = target_h - frame_h
        max_y = 0.0

        if x < min_x:
            x = min_x
        elif x > max_x:
            x = max_x

        if y < min_y:
            y = min_y
        elif y > max_y:
            y = max_y

        return (x, y)

    zoomed = (
        base_img
        .resize(lambda t: scale_at(t))
        .set_duration(duration)
        .set_position(lambda t: position_at(t))
    )

    visual = CompositeVideoClip([zoomed], size=target_size).set_duration(duration)
    visual = visual.fx(vfx.fadein, fade_seconds).fx(vfx.fadeout, fade_seconds)
    visual = visual.set_audio(audio_clip.fx(afx.audio_fadein, fade_seconds).fx(afx.audio_fadeout, fade_seconds))

    return visual

