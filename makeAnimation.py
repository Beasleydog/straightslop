import subprocess
import tempfile
import os
import random
import math
import hashlib

# Constants for Ken Burns effect
ZOOM_MIN = 1.05
ZOOM_MAX = 1.15
PAN_MAX_RATIO = 0.1  # Maximum pan as ratio of image dimension
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FPS = 60

CACHE_DIR = "cache/animatedimages"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(image_path: str, total_frames: int, target_width: int, target_height: int, fps: int):
    """Generate cache path based on inputs so different sizes/fps don't collide."""
    key_src = (
        f"{image_path}|frames={total_frames}|{ZOOM_MIN}|{ZOOM_MAX}|{PAN_MAX_RATIO}|{target_width}|{target_height}|{fps}".encode("utf-8")
    )
    return os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".mp4")


def ken_burns_effect_ffmpeg(
    image_path: str,
    duration: float,
    total_frames_override: int | None = None,
    target_size: tuple[int, int] | None = None,
    fps_override: int | None = None,
) -> str:
    """
    Apply Ken Burns effect to an image using ffmpeg with random pan direction and zoom direction.
    
    Args:
        image_path: Path to input image
        duration: Duration of the effect in seconds
        
    Returns:
        Path to the cached video file
    """
    # Determine exact frame count (CFR) for accuracy
    total_frames = (
        int(total_frames_override)
        if total_frames_override is not None and int(total_frames_override) > 1
        else max(2, int(round(duration * FPS)))
    )
    # Resolve output geometry and fps
    out_w, out_h = target_size if target_size else (TARGET_WIDTH, TARGET_HEIGHT)
    out_fps = int(fps_override) if (fps_override and int(fps_override) > 0) else FPS

    # Check cache first
    cache_path = _cache_path(image_path, total_frames, out_w, out_h, out_fps)
    if os.path.exists(cache_path):
        print(f"Ken Burns: Using cached animation: {cache_path}")
        return cache_path
    
    print(f"Ken Burns: Creating animation for: {image_path} ({duration}s, {total_frames}f)...")
    
    # Random pan direction (0 to 2*pi radians)
    pan_angle = random.uniform(0, 2 * math.pi)
    
    # Random zoom direction - either zoom in or zoom out
    if random.choice([True, False]):
        zoom_start = ZOOM_MIN
        zoom_end = ZOOM_MAX
    else:
        zoom_start = ZOOM_MAX
        zoom_end = ZOOM_MIN
    
    # Calculate zoom progression
    zoom_range = zoom_end - zoom_start
    denom_frames = max(1, total_frames - 1)
    
    # Scale image to accommodate zoom and pan
    scale_factor = 2.5  # Overscan to allow for zoom and pan
    scaled_width = int(TARGET_WIDTH * scale_factor)
    
    # Calculate pan distances based on angle
    pan_x_max = out_w * PAN_MAX_RATIO * math.cos(pan_angle)
    pan_y_max = out_h * PAN_MAX_RATIO * math.sin(pan_angle)
    
    # Build the filter complex
    filter_complex = (
        f"[0:v]"
        f"scale={scaled_width}:-2:flags=lanczos,setsar=1,"
        f"zoompan="
        f"z='{zoom_start} + {zoom_range}*(on/{denom_frames})':"
        f"x='iw/2 - (iw/zoom/2) + {pan_x_max}*(on/{denom_frames})':"
        f"y='ih/2 - (ih/zoom/2) + {pan_y_max}*(on/{denom_frames})':"
        f"d={total_frames}:s={out_w}x{out_h}:fps={out_fps}[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        # "-vsync", "cfr",
        # "-r", str(FPS),
        "-frames:v", str(total_frames),
        "-movflags", "+faststart",
        cache_path
    ]
    
    print(f"Running ffmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with return code: {result.returncode}")
    
    return cache_path