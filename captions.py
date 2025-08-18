import os
import subprocess
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faster_whisper import WhisperModel
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip


# ===== Caption Config =====
CAPTION_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font.ttf")
CAPTION_FONT_SIZE = 72
CAPTION_STROKE_WIDTH = 8
CAPTION_COLOR = "white"
CAPTION_STROKE_COLOR = "black"
CAPTION_HIGHLIGHT_COLOR = "#FFD60A"
CAPTION_HIGHLIGHT_BOLD_OFFSET_PX = 1

CAPTION_Y_RATIO = 0.75  # 3/4 down the screen
CAPTION_MAX_WIDTH_RATIO = 0.90
CAPTION_MAX_LINES = 2
CAPTION_WINDOW_SIZE = 5  # number of words per window; window advances only after last word

# How much context around the current word to show in the window
CAPTION_WORDS_BEFORE = 3
CAPTION_WORDS_AFTER = 3

# Control how long each snippet is visible around the word timing
CAPTION_PRE_ROLL_S = 0.02
CAPTION_POST_ROLL_S = 0.10

# ===== Whisper Config =====
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"  # 'cpu' keeps it simple and works everywhere
WHISPER_COMPUTE_TYPE = "int8"


def _transcribe_words(video_path: str) -> List[dict]:
    """
    Transcribe video with faster-whisper and return a flat list of words
    with start/end timestamps: [{"text": str, "start": float, "end": float}].
    """
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    segments, _info = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        language=None,
    )

    words: List[dict] = []
    for seg in segments:
        if not getattr(seg, "words", None):
            # fallback to segment-level text if words missing
            words.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "end": float(seg.end),
            })
            continue
        for w in seg.words:
            text = (w.word or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": float(w.start),
                "end": float(w.end),
            })
    return words


def _probe_video_nb_frames(path: str) -> int:
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


def _probe_video_fps(path: str) -> float:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rate = r.stdout.strip()  # like '60/1'
        if rate and rate != "N/A":
            if "/" in rate:
                num, den = rate.split("/", 1)
                num_f = float(num)
                den_f = float(den)
                if den_f != 0:
                    return num_f / den_f
            else:
                return float(rate)
    except Exception:
        pass
    return 0.0


def _probe_video_stream_duration_seconds(path: str) -> float:
    # Prefer frame count / fps to avoid container-level audio duration
    nb = _probe_video_nb_frames(path)
    fps = _probe_video_fps(path)
    if nb > 0 and fps > 0:
        return nb / fps
    # Fallback to container duration
    try:
        r = subprocess.run(
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
        return max(0.0, float(r.stdout.strip()))
    except Exception:
        return 0.0


def _measure_text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int) -> Tuple[int, int]:
    # Returns (width, height)
    if not text:
        return 0, 0
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return x1 - x0, y1 - y0


def _wrap_words_to_lines(
    words: List[str],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    stroke_width: int,
    max_width_px: int,
    max_lines: int,
) -> Tuple[List[List[str]], List[int]]:
    """
    Greedy word-wrap into up to max_lines. Returns (lines, line_widths).
    lines is List[List[str]] where each sublist is words for that line.
    """
    lines: List[List[str]] = [[]]
    line_widths: List[int] = [0]
    for w in words:
        if not lines[-1]:
            candidate = w
        else:
            candidate = " ".join(lines[-1] + [w])
        cand_w, _ = _measure_text_bbox(draw, candidate, font, stroke_width)
        if cand_w <= max_width_px or not lines[-1]:
            # fits or forced into empty line
            lines[-1] = candidate.split(" ")
            line_widths[-1] = cand_w
        else:
            if len(lines) >= max_lines:
                # truncate extra words into last line
                # force append with space even if it exceeds, to avoid dropping text
                lines[-1].append(w)
                line_widths[-1], _ = _measure_text_bbox(draw, " ".join(lines[-1]), font, stroke_width)
            else:
                lines.append([w])
                w_width, _ = _measure_text_bbox(draw, w, font, stroke_width)
                line_widths.append(w_width)
    return lines, line_widths


def _render_caption_image(
    snippet_words: List[str],
    highlight_index: int,
    video_size: Tuple[int, int],
) -> Image.Image:
    """
    Render a transparent image of the caption snippet with the current word highlighted.
    Center horizontally within full video width for easy positioning.
    """
    vw, vh = video_size
    max_text_width = int(vw * CAPTION_MAX_WIDTH_RATIO)

    # Create a temporary small canvas for measurement
    tmp_img = Image.new("RGBA", (max(10, max_text_width), 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp_img)
    font = ImageFont.truetype(CAPTION_FONT_PATH, CAPTION_FONT_SIZE)

    lines, line_widths = _wrap_words_to_lines(
        snippet_words, draw, font, CAPTION_STROKE_WIDTH, max_text_width, CAPTION_MAX_LINES
    )

    # Compute line height and total height
    ascent, descent = font.getmetrics()
    base_line_height = ascent + descent
    interline_px = int(round(CAPTION_FONT_SIZE * 0.28))
    total_h = base_line_height * len(lines) + interline_px * (len(lines) - 1)

    # Final image spans full video width to simplify centering; height fits text
    img = Image.new("RGBA", (vw, total_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Precompute where the highlight word lands (line index and x offset)
    # Map highlight_index within flattened snippet to line/offset
    flat_words: List[Tuple[int, str]] = []  # (line_idx, word)
    for li, line in enumerate(lines):
        for w in line:
            flat_words.append((li, w))

    if 0 <= highlight_index < len(flat_words):
        hl_line_index, _hl_word = flat_words[highlight_index]
    else:
        hl_line_index = 0

    # Draw lines (base layer: white text with black stroke)
    y_cursor = 0
    per_word_x_offsets: List[List[int]] = []
    for li, line in enumerate(lines):
        text_line = " ".join(line)
        line_w = line_widths[li]
        x_start = (vw - line_w) // 2
        # Track x offsets per word for highlight placement
        x_offsets: List[int] = []
        accum = ""
        for wi, w in enumerate(line):
            before = (accum + (" " if accum else "")) + w
            w_w, _ = _measure_text_bbox(d, before, font, CAPTION_STROKE_WIDTH)
            # offset for current word is width of previous accum
            if accum:
                prev_w, _ = _measure_text_bbox(d, accum, font, CAPTION_STROKE_WIDTH)
            else:
                prev_w = 0
            x_offsets.append(x_start + prev_w)
            accum = before

        d.text(
            (x_start, y_cursor),
            text_line,
            font=font,
            fill=CAPTION_COLOR,
            stroke_width=CAPTION_STROKE_WIDTH,
            stroke_fill=CAPTION_STROKE_COLOR,
            align="center",
        )
        per_word_x_offsets.append(x_offsets)
        y_cursor += base_line_height + interline_px

    # Draw the highlight word on top in its color (simulate bold by overdraw)
    if 0 <= highlight_index < len(flat_words):
        li, _w = flat_words[highlight_index]
        # find word index within that line
        # compute index of highlight within line
        words_before = sum(len(ln) for ln in lines[:li])
        wi_in_line = highlight_index - words_before
        if 0 <= wi_in_line < len(per_word_x_offsets[li]):
            # compute y for that line
            y_for_line = li * (base_line_height + interline_px)
            x_for_word = per_word_x_offsets[li][wi_in_line]
            the_word = lines[li][wi_in_line]
            # Overdraw several times for a bolder look
            offsets = [(0, 0), (CAPTION_HIGHLIGHT_BOLD_OFFSET_PX, 0), (0, CAPTION_HIGHLIGHT_BOLD_OFFSET_PX)]
            for dx, dy in offsets:
                d.text(
                    (x_for_word + dx, y_for_line + dy),
                    the_word,
                    font=font,
                    fill=CAPTION_HIGHLIGHT_COLOR,
                    stroke_width=CAPTION_STROKE_WIDTH,
                    stroke_fill=CAPTION_STROKE_COLOR,
                )

    return img


def add_tiktok_captions(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Transcribe the input video and burn TikTok-style captions.
    Returns the output path written to disk.
    """
    words = _transcribe_words(video_path)
    if not words:
        return video_path

    # Durations
    video_stream_duration = float(_probe_video_stream_duration_seconds(video_path) or 0.0)
    base_raw = VideoFileClip(video_path)
    try:
        audio_duration = float(base_raw.audio.duration) if base_raw.audio else 0.0
    except Exception:
        audio_duration = 0.0
    container_duration = float(base_raw.duration or 0.0)
    target_duration = max(video_stream_duration, audio_duration, container_duration)

    # Freeze last frame after the video stream ends, so video spans full audio
    eps = 1.0 / float(base_raw.fps or 30.0) / 2.0
    safe_last_t = max(0.0, (video_stream_duration or container_duration) - eps)
    last_frame_img = base_raw.get_frame(safe_last_t)

    def _frozen_frame_fn(t: float):
        if t <= (video_stream_duration - eps):
            # Clamp inside available stream range
            tt = max(0.0, min(t, video_stream_duration - eps))
            return base_raw.get_frame(tt)
        return last_frame_img

    base = VideoClip(frame_function=_frozen_frame_fn).with_duration(target_duration)
    base.fps = base_raw.fps
    base.size = base_raw.size
    if base_raw.audio is not None:
        base.audio = base_raw.audio
    vw, vh = base.w, base.h
    clips: List[ImageClip] = []

    total_dur = float(target_duration)
    window_size = max(1, int(CAPTION_WINDOW_SIZE))
    prev_end = 0.0

    # --- Split into sentences by terminal punctuation on words ---
    sentence_boundaries: List[Tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
    s_start = 0
    n = len(words)
    terminal_chars = set([".", "!", "?", "…"])
    for i, wobj in enumerate(words):
        wtxt = str(wobj.get("text", ""))
        ends_sentence = any(ch in wtxt for ch in terminal_chars)
        is_last = (i == n - 1)
        if ends_sentence or is_last:
            end_idx = i + 1
            if end_idx > s_start:
                sentence_boundaries.append((s_start, end_idx))
            s_start = end_idx
    if s_start < n:
        sentence_boundaries.append((s_start, n))

    # --- Build clips: window within each sentence, highlight word-by-word ---
    for (s_start, s_end) in sentence_boundaries:
        idx = s_start
        while idx < s_end:
            win_start = idx
            win_end = min(s_end, win_start + window_size)
            snippet = [words[i]["text"] for i in range(win_start, win_end)]

            for wi in range(win_start, win_end):
                local_index = wi - win_start
                w = words[wi]

                if wi == win_start:
                    desired_start = max(0.0, float(w["start"]) - CAPTION_PRE_ROLL_S)
                    start_time = max(prev_end, desired_start)
                else:
                    start_time = prev_end

                if wi < win_end - 1:
                    next_start = float(words[wi + 1]["start"])  # until next word begins
                else:
                    # last word in window: hold until next window OR next sentence OR end
                    if win_end < s_end:
                        next_start = float(words[win_end]["start"])  # next window in same sentence
                    elif (sentence_boundaries and (sentence_boundaries[-1] != (s_start, s_end))):
                        # next sentence exists
                        cur_idx = sentence_boundaries.index((s_start, s_end))
                        if cur_idx + 1 < len(sentence_boundaries):
                            next_sentence_first = sentence_boundaries[cur_idx + 1][0]
                            next_start = float(words[next_sentence_first]["start"]) if next_sentence_first < n else total_dur
                        else:
                            next_start = total_dur
                    else:
                        next_start = total_dur
                    next_start = min(next_start, float(w["end"]) + CAPTION_POST_ROLL_S, total_dur)

                end_time = min(total_dur, next_start)
                if end_time < start_time:
                    end_time = start_time

                duration = end_time - start_time
                if duration <= 0:
                    continue

                img = _render_caption_image(snippet, local_index, (vw, vh))
                np_img = np.array(img)
                cap_clip = ImageClip(np_img).with_start(start_time).with_duration(duration)

                y_center = int(vh * CAPTION_Y_RATIO)
                y_pos = int(y_center - (cap_clip.h // 2))
                cap_clip = cap_clip.with_position(("center", y_pos))
                clips.append(cap_clip)

                prev_end = end_time

            idx = win_end

    # Ensure composite has a defined duration matching the target timeline (audio end)
    comp = CompositeVideoClip([base] + clips, size=(vw, vh)).with_duration(total_dur)

    if output_path is None:
        base_dir, base_name = os.path.split(video_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(base_dir or ".", f"{name}_captions{ext}")

    # Avoid writing to the same file being read
    in_abs = os.path.abspath(video_path)
    out_abs = os.path.abspath(output_path)
    same_path = in_abs == out_abs
    if same_path:
        base_dir, base_name = os.path.split(in_abs)
        name, ext = os.path.splitext(base_name)
        target_path = os.path.join(base_dir or ".", f"{name}_captions{ext}")
    else:
        target_path = out_abs

    comp.write_videofile(
        target_path,
        codec="libx264",
        audio_codec="aac",
        fps=base.fps or 30,
        preset="medium",
        threads=0,
        logger=None,
    )

    base.close()
    base_raw.close()
    comp.close()
    for c in clips:
        c.close()

    return target_path


__all__ = ["add_tiktok_captions"]

