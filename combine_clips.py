import os
import tempfile
import subprocess
from typing import List


def combine_clips(video_paths: List[str], output_path: str) -> str:
    """
    Combine multiple MP4 clips into a single MP4 using ffmpeg concat demuxer.

    - Expects all inputs to have identical codec/size/fps/format.
    - Performs stream copy for speed.
    - Returns the output path.
    """
    if not video_paths:
        raise ValueError("combine_clips: video_paths is empty")

    if len(video_paths) == 1:
        # Fast path: single clip, just copy
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_paths[0],
            "-c", "copy",
            output_path,
        ], check=True)
        return output_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            abs_path = os.path.abspath(p)
            normalized = abs_path.replace('\\', '/')
            f.write(f"file '{normalized}'\n")
        list_path = f.name

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ], check=True)
    finally:
        try:
            os.unlink(list_path)
        except Exception:
            pass

    return output_path

