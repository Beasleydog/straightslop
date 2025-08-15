import os
import subprocess
from typing import Tuple


def _probe_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Return (width, height) of the first video stream using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)


def addOverlay(video_path: str, max_duration_s: float | None = None) -> str:
    """
    Overlay `overlay.mp4` across the entire input video at 10% opacity, looping
    the overlay (10s source) for the full duration of the base video. The
    original audio is preserved. Writes to `<INPUTNAME>_overlay.mp4` next to the
    input and returns the output path.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    base_dir, base_filename = os.path.split(video_path)
    name, _ext = os.path.splitext(base_filename)
    output_path = os.path.join(base_dir or ".", f"{name}_overlay.mp4")

    overlay_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay.mp4")
    if not os.path.isfile(overlay_path):
        raise FileNotFoundError(f"Overlay video not found: {overlay_path}")

    base_w, base_h = _probe_video_dimensions(video_path)

    # Build filter graph using blend for reliable constant-opacity compositing:
    # - Scale overlay to base dimensions
    # - Blend 10% overlay over the base
    saturation_mult = 1.30  # constant factor to boost overlay saturation
    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS,format=yuv420p[base];"
        f"[1:v]setpts=PTS-STARTPTS,scale={base_w}:{base_h},format=yuv420p,eq=saturation={saturation_mult}[ov];"
        f"[base][ov]blend=all_mode=normal:all_opacity=0.50[outv]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        # Base video
        "-i",
        video_path,
        # Loop overlay infinitely so it covers the whole base video duration
        "-stream_loop",
        "-1",
        "-i",
        overlay_path,
        # Compose
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        # Keep original audio if present (the ? makes it optional)
        "-map",
        "0:a?",
        # Encoding params
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        # Copy audio to preserve original
        "-c:a",
        "copy",
        # Ensure output stops with the video (in case audio is longer)
        "-shortest",
    ]

    # If limiting duration for testing, add -t
    if max_duration_s is not None and max_duration_s > 0:
        cmd.extend(["-t", str(float(max_duration_s))])

    cmd.append(output_path)
    

    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    import sys

    # When executed directly, render only the first 10 seconds for quick testing
    # Pass a path argument or fall back to output.mp4 if present
    input_path = None
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    elif os.path.isfile("output.mp4"):
        input_path = "output.mp4"
    else:
        print("Usage: python addOverlay.py <path_to_video>")
        raise SystemExit(2)

    print(addOverlay(input_path))


