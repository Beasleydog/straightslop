import subprocess
import math
import random
import os
import tempfile
from typing import Tuple, List
from PIL import Image
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

def compute_crossfade(durations: List[float]) -> float:
    """Compute appropriate crossfade duration based on section durations."""
    if not durations:
        return 0.5
    return min(0.5, max(0.1, min(durations) / 4))

def create_kenburns_with_ffmpeg(
    image_path: str,
    duration: float,
    output_path: str,
    target_size: Tuple[int, int] = (1920, 1080),
    zoom_start: float = 1.05,
    zoom_end: float = 1.15,
    pan_max_px: float = 60.0,  # treated as a soft cap; geometry still wins
    fps: int = 30,
    seed: int = None,
    total_frames_override: int | None = None
) -> str:
    """
    Create a Ken Burns effect video using ffmpeg, with unbiased random start/end
    positions and randomized zoom direction to avoid repeated center/diagonal bias.
    Keeps high-quality scaling, gblur anti-shimmer, 4:4:4 filtering, and CFR output.
    """
    if seed is not None:
        random.seed(seed)

    # Get image dimensions - handle both local files and URLs
    if image_path.startswith(('http://', 'https://')):
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0", image_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dimensions = result.stdout.strip().split('x')
            img_w, img_h = int(dimensions[0]), int(dimensions[1])
        except Exception:
            img_w, img_h = 1024, 1024
    else:
        with Image.open(image_path) as img:
            img_w, img_h = img.size

    target_w, target_h = target_size

    # Calculate scale to cover the frame, then overscale for pan margin
    cover_scale = max(target_w / img_w, target_h / img_h)
    overscale = 1.25

    # Randomize zoom direction half the time (zoom-out shot variety)
    if random.random() < 0.5:
        zoom_start, zoom_end = zoom_end, zoom_start

    # Effective scales at ends
    scale_start = cover_scale * overscale * zoom_start
    scale_end = cover_scale * overscale * zoom_end

    # Pre-scale base plane once for zoompan
    base_scale = cover_scale * overscale
    base_w = int(2 * round((img_w * base_scale) / 2))
    base_h = int(2 * round((img_h * base_scale) / 2))

    # Compute smallest framed dimensions across the shot (avoid edge reveal)
    scaled_w_start = img_w * scale_start
    scaled_h_start = img_h * scale_start
    scaled_w_end = img_w * scale_end
    scaled_h_end = img_h * scale_end
    min_frame_w = min(scaled_w_start, scaled_w_end)
    min_frame_h = min(scaled_h_start, scaled_h_end)

    # Safe half-pan range from center at all times (source-space pixels)
    max_pan_x = max(0.0, (min_frame_w - target_w) / 2.0 - 2.0)
    max_pan_y = max(0.0, (min_frame_h - target_h) / 2.0 - 2.0)

    # Optional additional cap based on requested screen-pixel travel
    if pan_max_px is not None and pan_max_px > 0:
        cap_x = pan_max_px * (min_frame_w / max(1.0, target_w))
        cap_y = pan_max_px * (min_frame_h / max(1.0, target_h))
        max_pan_x = min(max_pan_x, cap_x)
        max_pan_y = min(max_pan_y, cap_y)

    # Random start/end offsets uniformly inside the safe box
    def rand_offset() -> tuple[float, float]:
        ox = random.uniform(-max_pan_x, max_pan_x) if max_pan_x > 0 else 0.0
        oy = random.uniform(-max_pan_y, max_pan_y) if max_pan_y > 0 else 0.0
        return ox, oy

    start_x, start_y = rand_offset()
    tries = 0
    min_travel = 0.25 * math.hypot(max_pan_x, max_pan_y)
    if min_travel == 0:
        min_travel = 10.0
    while True:
        end_x, end_y = rand_offset()
        if math.hypot(end_x - start_x, end_y - start_y) >= min_travel or tries > 8:
            break
        tries += 1
    if tries > 8 and (max_pan_x + max_pan_y) > 0:
        end_x, end_y = -start_x, -start_y

    # Frame/easing setup
    total_frames = (
        int(total_frames_override)
        if total_frames_override is not None and int(total_frames_override) > 1
        else max(2, int(round(fps * duration)))
    )
    denom_frames = total_frames - 1
    p_expr = f"(on/{denom_frames})"
    ease_expr = f"({p_expr}*{p_expr}*(3-2*{p_expr}))"  # smoothstep

    # Eased lerp for offsets; divide by zoom to keep screen-pixel consistency
    x_num = f"(({start_x})*(1-{ease_expr}) + ({end_x})*{ease_expr})"
    y_num = f"(({start_y})*(1-{ease_expr}) + ({end_y})*{ease_expr})"

    # Multiplicative zoom progression (smooth exponential from start->end)
    z_expr = f"({zoom_start}*pow({zoom_end}/{zoom_start}, on/{denom_frames}))"

    vf = (
        "format=rgba,"
        f"scale={base_w}:{base_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
        "gblur=sigma=0.30,"
        f"zoompan=z='{z_expr}':"
        f"x='(iw-{target_w}/zoom)/2+({x_num})/zoom':"
        f"y='(ih-{target_h}/zoom)/2+({y_num})/zoom':"
        f"d={total_frames}:s={target_w}x{target_h}:fps={fps},"
        "format=yuv444p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-vsync", "cfr", "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        # Fallback: no pan, only zoom (still random zoom direction)
        simple_vf = (
            "format=rgba,"
            f"scale={base_w}:{base_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
            "gblur=sigma=0.30,"
            f"zoompan=z='{z_expr}':"
            f"x='(iw-{target_w}/zoom)/2':y='(ih-{target_h}/zoom)/2':"
            f"d={total_frames}:s={target_w}x{target_h}:fps={fps},"
            "format=yuv444p"
        )
        simple_cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-vf", simple_vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-vsync", "cfr", "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(simple_cmd, check=True)
        return output_path

def create_section_video_with_ffmpeg(
    section_data: dict,
    output_path: str,
    target_size: Tuple[int, int] = (1920, 1080),
    zoom_start: float = 1.05,
    zoom_end: float = 1.15,
    pan_max_px: float = 60.0,
    fps: int = 30,
    silence_padding: float = 0.0
) -> str:
    """
    Create a complete section video with multiple images timed to audio using ffmpeg.
    """
    audio_path = section_data["audio_path"]
    items = section_data["items"]
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        video_clips = []
        
        # Compute precise integer frame counts per item using largest remainder
        raw_durations = [max(0.0, it["end"] - it["start"]) for it in items]
        if raw_durations and silence_padding > 0:
            raw_durations[-1] += silence_padding
        exact_frames_f = [dur * fps for dur in raw_durations]
        base_frames = [int(math.floor(f)) for f in exact_frames_f]
        remainders = [f - b for f, b in zip(exact_frames_f, base_frames)]
        total_target_frames = int(round(sum(exact_frames_f)))
        deficit = total_target_frames - sum(base_frames)
        # Distribute leftover frames to largest remainders
        order = sorted(range(len(base_frames)), key=lambda k: remainders[k], reverse=True)
        for idx in order[:max(0, deficit)]:
            base_frames[idx] += 1

        # Create individual video clips with exact frame counts
        for i, item in enumerate(tqdm(items, desc="Clips", unit="clip")):
            frames_for_item = max(2, base_frames[i]) if i < len(base_frames) else max(2, int(round((item["end"] - item["start"]) * fps)))
            clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")

            create_kenburns_with_ffmpeg(
                image_path=item["image"],
                duration=frames_for_item / fps,
                output_path=clip_path,
                target_size=target_size,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                pan_max_px=pan_max_px,
                fps=fps,
                seed=i,
                total_frames_override=frames_for_item,
            )

            video_clips.append((clip_path, frames_for_item))
        
        # Create filter complex for concatenating and timing clips
        # Concatenate clips by stream-copy to preserve exact frames
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for clip_path, _frames in video_clips:
                f.write(f"file '{clip_path}'\n")

        concat_out = os.path.join(temp_dir, "video_section.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            concat_out
        ]
        subprocess.run(concat_cmd, check=True)

        # Mux with audio without changing video timing
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_out,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",  # trim audio if video shorter
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return output_path

def concatenate_videos_fast(
    video_paths: List[str],
    output_path: str
) -> str:
    """
    Concatenate videos using simple, fast concatenation (no crossfade).
    This is MUCH faster than complex filter chains.
    """
    if len(video_paths) == 1:
        # Single video, just copy
        subprocess.run(["ffmpeg", "-y", "-i", video_paths[0], "-c", "copy", output_path], check=True)
        return output_path
    
    # Create concat file for simple concatenation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_paths:
            # Use forward slashes for ffmpeg concat (works on Windows too)
            normalized_path = video_path.replace('\\', '/')
            f.write(f"file '{normalized_path}'\n")
        concat_file = f.name
    
    try:
        # Simple concat - blazing fast!
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",  # Copy streams without re-encoding = FAST
            output_path
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(concat_file)
    
    return output_path