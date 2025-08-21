import os
import subprocess
import tempfile
from typing import List


def _probe_duration_seconds(path: str) -> float:
    """Return media duration in seconds using ffprobe. Falls back to 0.0 on error."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        return 0.0


def _normalize_path_for_concat(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _probe_nb_frames(path: str) -> int:
    """Return number of video frames using ffprobe. Fallbacks to 0 on error."""
    try:
        r1 = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        txt = r1.stdout.strip()
        if txt and txt != "N/A":
            val = int(txt)
            if val > 0:
                return val
    except Exception:
        pass

    try:
        r2 = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        txt2 = r2.stdout.strip()
        if txt2 and txt2 != "N/A":
            val2 = int(txt2)
            if val2 > 0:
                return val2
    except Exception:
        pass

    return 0


def _cut_encode_segment_by_frames(src: str, dst: str, start_frame: int, num_frames: int, fps: int) -> None:
    """Cut a segment by exact frame indices and re-encode to CFR fps."""
    end_frame = start_frame + max(0, num_frames - 1)
    vf = (
        f"select='between(n,{start_frame},{end_frame})',"
        f"settb=AVTB,setpts=N/({fps}*TB),fps={fps},format=yuv420p"
    )
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-profile:v",
        "baseline",
        "-level:v",
        "4.2",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        "-frames:v",
        str(max(1, num_frames)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-video_track_timescale",
        "15360",
        dst,
    ]
    subprocess.run(cmd, check=True)


def _create_crossfade_segment(
    a_path: str,
    b_path: str,
    a_nb_frames: int,
    fade_frames: int,
    fps: int,
    dst: str,
) -> None:
    """Re-encode only the overlapping tails/heads with an xfade.

    Encodes with the same general settings as ken_burns_effect_ffmpeg to keep
    parameters compatible for concat stream copy.
    """
    # Tail frames of A and head frames of B
    start_a_frame = max(0, a_nb_frames - fade_frames)
    end_a_frame = max(0, a_nb_frames - 1)
    end_b_frame = max(0, fade_frames - 1)
    fade_seconds = max(1, fade_frames) / float(fps)
    filter_complex = (
        f"[0:v]select='between(n,{start_a_frame},{end_a_frame})',settb=AVTB,setpts=N/({fps}*TB),fps={fps}[va];"
        f"[1:v]select='between(n,0,{end_b_frame})',settb=AVTB,setpts=N/({fps}*TB),fps={fps}[vb];"
        f"[va][vb]xfade=transition=fade:duration={fade_seconds:.9f}:offset=0,format=yuv420p[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        a_path,
        "-i",
        b_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "20",
        "-profile:v",
        "baseline",
        "-level:v",
        "4.2",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        # let filtergraph dictate the exact number of frames for stability
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-video_track_timescale",
        "15360",
        dst,
    ]
    subprocess.run(cmd, check=True)


def combine_clips_with_fades(
    video_paths: List[str],
    output_path: str,
    fade_seconds: float,
    fps: int = 60,
    pair_fade_frames_override: list[int] | None = None,
) -> str:
    """
    Concatenate clips with crossfades, re-encoding only the overlapping regions.

    Strategy for N clips: A, B, C, ...
    - Pre of A: [0, dur(A)-f0]
    - CF AB: last f0 of A crossfaded with first f0 of B (re-encode)
    - Mid of interior clips i (1..N-2): [f_{i-1}, dur(i)-f_i]
    - CF i->i+1 for each adjacent pair
    - Post of last: [f_{N-2}, end]

    Where f_i = min(fade_seconds, dur(i), dur(i+1)).
    Non-overlap segments are copied with -c copy, crossfades are re-encoded.
    Temporary files are cleaned after final concat.
    """
    if not video_paths:
        raise ValueError("combine_clips_with_fades: video_paths is empty")

    if len(video_paths) == 1 or fade_seconds <= 0:
        # Fast path: single clip or no fade requested
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i",
            video_paths[0],
            "-c",
            "copy",
            output_path,
        ], check=True)
        return output_path

    # Probe durations and frames
    durations = [_probe_duration_seconds(p) for p in video_paths]
    nb_frames = [_probe_nb_frames(p) for p in video_paths]

    # Compute pair-wise fade frames
    requested_fade_frames = max(1, int(round(fade_seconds * fps)))
    if pair_fade_frames_override is not None and len(pair_fade_frames_override) == max(0, len(video_paths) - 1):
        pair_fade_frames: List[int] = [max(1, int(v)) for v in pair_fade_frames_override]
    else:
        pair_fade_frames = []
        for i in range(len(video_paths) - 1):
            ai = nb_frames[i] if nb_frames[i] > 0 else int(round(durations[i] * fps))
            bi = nb_frames[i + 1] if nb_frames[i + 1] > 0 else int(round(durations[i + 1] * fps))
            f_frames = max(1, min(requested_fade_frames, ai, bi))
            pair_fade_frames.append(f_frames)

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_paths: List[str] = []

        # Pre of first clip frames
        f0_frames = pair_fade_frames[0]
        n0 = nb_frames[0] if nb_frames[0] > 0 else int(round(durations[0] * fps))
        pre_frames = max(0, n0 - f0_frames)
        if pre_frames > 0:
            pre_path = os.path.join(temp_dir, f"seg_pre_000.mp4")
            _cut_encode_segment_by_frames(video_paths[0], pre_path, start_frame=0, num_frames=pre_frames, fps=fps)
            segment_paths.append(pre_path)

        # Process each adjacent pair
        for i in range(len(video_paths) - 1):
            fade_i_frames = pair_fade_frames[i]
            a_path = video_paths[i]
            b_path = video_paths[i + 1]
            a_n = nb_frames[i] if nb_frames[i] > 0 else int(round(durations[i] * fps))

            if fade_i_frames > 0:
                cf_path = os.path.join(temp_dir, f"seg_xfade_{i:03d}.mp4")
                _create_crossfade_segment(a_path, b_path, a_n, fade_i_frames, fps, cf_path)
                segment_paths.append(cf_path)

            # Mid of next clip (if interior)
            if 0 < (i + 1) < (len(video_paths) - 1):
                # For clip j = i+1, mid frames are [f_{j-1}, n_j - f_j)
                f_prev_frames = pair_fade_frames[i]
                f_next_frames = pair_fade_frames[i + 1]
                b_n = nb_frames[i + 1] if nb_frames[i + 1] > 0 else int(round(durations[i + 1] * fps))
                start_mid_frame = max(0, f_prev_frames)
                mid_frames = max(0, b_n - f_prev_frames - f_next_frames)
                if mid_frames > 0:
                    mid_path = os.path.join(temp_dir, f"seg_mid_{i+1:03d}.mp4")
                    _cut_encode_segment_by_frames(b_path, mid_path, start_frame=start_mid_frame, num_frames=mid_frames, fps=fps)
                    segment_paths.append(mid_path)

        # Post of last clip frames
        f_last_frames = pair_fade_frames[-1]
        n_last = nb_frames[-1] if nb_frames[-1] > 0 else int(round(durations[-1] * fps))
        post_frames = max(0, n_last - f_last_frames)
        if post_frames > 0:
            post_path = os.path.join(temp_dir, f"seg_post_{len(video_paths)-1:03d}.mp4")
            _cut_encode_segment_by_frames(video_paths[-1], post_path, start_frame=f_last_frames, num_frames=post_frames, fps=fps)
            segment_paths.append(post_path)

        # Concat all segments with stream copy
        if not segment_paths:
            # Fallback: nothing produced; copy first
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i",
                video_paths[0],
                "-c",
                "copy",
                output_path,
            ], check=True)
            return output_path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in segment_paths:
                f.write(f"file '{_normalize_path_for_concat(p)}'\n")
            list_path = f.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-c",
                    "copy",
                    output_path,
                ],
                check=True,
            )
        finally:
            try:
                os.unlink(list_path)
            except Exception:
                pass

    return output_path


def _probe_duration_seconds_strict(path: str) -> float:
    """Like _probe_duration_seconds but raise if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(0.0, float(result.stdout.strip()))
    except Exception as exc:
        raise RuntimeError(f"Could not probe duration for {path}: {exc}")


def combine_two_clips_with_fades_av(
    clip_a_path: str,
    clip_b_path: str,
    output_path: str,
    fade_seconds: float,
    fps: int = 60,
) -> str:
    """
    Combine two clips with a crossfade that preserves both video and audio.

    - Video: xfade transition for the specified duration, starting at
      offset = duration(A) - fade_seconds.
    - Audio: acrossfade for the same duration.
    - Outputs CFR H.264 + AAC suitable for later concat with -c copy.
    """
    if fade_seconds <= 0:
        # Fast path: no fade, simple concat re-encode to ensure uniform params
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"file '{_normalize_path_for_concat(clip_a_path)}'\n")
            f.write(f"file '{_normalize_path_for_concat(clip_b_path)}'\n")
            list_path = f.name
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "20",
                "-profile:v", "baseline",
                "-level:v", "4.2",
                "-fps_mode", "cfr",
                "-r", str(int(fps)),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart",
                "-video_track_timescale", "15360",
                output_path,
            ], check=True)
        finally:
            try:
                os.unlink(list_path)
            except Exception:
                pass
        return output_path

    dur_a = _probe_duration_seconds_strict(clip_a_path)
    dur_b = _probe_duration_seconds_strict(clip_b_path)
    if dur_a <= 0.0 or dur_b <= 0.0:
        raise RuntimeError("Invalid input durations for crossfade")

    fade = max(0.0, min(float(fade_seconds), dur_a, dur_b))
    if fade <= 0.0:
        # Fallback to simple concat encode
        return combine_two_clips_with_fades_av(clip_a_path, clip_b_path, output_path, 0.0, fps=fps)

    offset = max(0.0, dur_a - fade)

    filter_complex = (
        f"[0:v][1:v]xfade=transition=fade:duration={fade:.9f}:offset={offset:.9f},fps={int(fps)},format=yuv420p[v];"
        f"[0:a][1:a]acrossfade=d={fade:.9f}:c1=tri:c2=tri[a]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", clip_a_path,
        "-i", clip_b_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-profile:v", "baseline",
        "-level:v", "4.2",
        "-fps_mode", "cfr",
        "-r", str(int(fps)),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-video_track_timescale", "15360",
        output_path,
    ]

    subprocess.run(cmd, check=True)
    return output_path
