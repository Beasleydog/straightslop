import os
import hashlib
import subprocess
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from generateImage import generateImage

# PIL compatibility shim for deprecated constants
try:
    LANCZOS = Image.Resampling.LANCZOS
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    LANCZOS = getattr(Image, "LANCZOS", Image.ANTIALIAS)
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache/section_cards"
TARGET_SIZE: Tuple[int, int] = (1920, 1080)
BLUR_RADIUS: int = 22
PADDING_PX: int = 300
MAX_FONT_SIZE: int = 528
MIN_FONT_SIZE: int = 18
TEXT_FILL = (255, 255, 255)
VERTICAL_OFFSET_RATIO: float = -0.06  # negative lifts text slightly for optical vertical centering
DARKEN_ALPHA: float = 0.22  # 0..1, blend with black after blur

# Default duration for section card videos (seconds)
SECTION_CARD_VIDEO_LENGTH_SECONDS: float = 5.0

# Background audio for section cards
BACKGROUND_AUDIO_PATH: str = "backgroundsection.mp3"

# Optional override via environment variable.
ENV_FONT_PATH = os.getenv("SECTION_CARD_FONT")


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(image_prompt: str, title: str) -> str:
    key_src = f"{image_prompt}|{title}".encode("utf-8")
    return os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".png")


def _video_cache_path(image_prompt: str, title: str, duration_s: float, fps: int) -> str:
    # Include a marker to distinguish builds with background audio
    key_src = f"{image_prompt}|{title}|dur={duration_s:.3f}|fps={fps}|bg=1".encode("utf-8")
    return os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".mp4")


def _pick_font(preferred_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Priority: explicit env var, then OS-common fonts, then Pillow default.
    candidate_paths = [
        "font.ttf"
    ]

    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, preferred_size)
        except Exception:
            continue
    # Fallback: default bitmap font (small)
    return ImageFont.load_default()


def _resize_cover(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    tw, th = target_size
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return img.convert("RGB").resize((tw, th))
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), resample=LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _best_fit_font(title: str, max_w: int, max_h: int) -> ImageFont.ImageFont:
    # Binary search the largest font size that fits within max_w x max_h
    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_font = _pick_font(MIN_FONT_SIZE)
    # If truetype font not available, fallback immediately
    if not isinstance(best_font, ImageFont.FreeTypeFont):
        return best_font

    draw = ImageDraw.Draw(Image.new("RGB", (max(1, max_w), max(1, max_h))))
    while low <= high:
        mid = (low + high) // 2
        font_try = _pick_font(mid)
        w, h = _measure_text(draw, title, font_try)
        if w <= max_w and h <= max_h:
            best_font = font_try
            low = mid + 1
        else:
            high = mid - 1
    return best_font


def makeSectionCard(image_prompt: str, title: str) -> str:
    """
    Create a blurred section card from a generated image and overlay a centered title.
    - Ensures title fits on a single line within padding by auto-scaling font size.
    - Caches result based on (image_prompt, title) and returns the cached path.
    """
    _ensure_cache_dir()

    # Cache check first
    out_path = _cache_path(image_prompt, title)
    if os.path.exists(out_path):
        return out_path

    # Generate or load the base image
    image_path = generateImage(image_prompt)
    base = Image.open(image_path).convert("RGB")

    # Prepare blurred background (cover + blur)
    bg = _resize_cover(base, TARGET_SIZE)
    bg = bg.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    # Slightly darken for better title contrast
    if DARKEN_ALPHA > 0.0:
        black = Image.new("RGB", bg.size, (0, 0, 0))
        bg = Image.blend(bg, black, max(0.0, min(1.0, DARKEN_ALPHA)))

    draw = ImageDraw.Draw(bg)

    # Find the largest font size that fits in the padded area without wrapping
    target_w, target_h = TARGET_SIZE
    max_w = max(1, target_w - 2 * PADDING_PX)
    max_h = max(1, target_h - 2 * PADDING_PX)

    font: ImageFont.ImageFont = _best_fit_font(title, max_w, max_h)

    # Measure with final font and center
    text_w, text_h = _measure_text(draw, title, font)
    x = (target_w - text_w) // 2
    y = (target_h - text_h) // 2 + int(text_h * VERTICAL_OFFSET_RATIO)

    # Draw centered title (no stroke)
    draw.text((x, y), title, font=font, fill=TEXT_FILL)

    # Save to cache and return path
    bg.save(out_path, format="PNG")
    return out_path


def makeSectionCardVideo(
    image_prompt: str,
    title: str,
    duration_seconds: float = SECTION_CARD_VIDEO_LENGTH_SECONDS,
    fps: int = 60,
) -> str:
    """
    Return a cached MP4 path for a section card video:
    - Builds or uses the cached PNG from makeSectionCard
    - Creates an H.264 CFR video at the requested fps
    - Fades in from black at the start and out to black at the end
    - Uses backgroundsection.mp3 as background audio with fade in/out
    """
    _ensure_cache_dir()

    # Prepare inputs and cache path
    img_path = makeSectionCard(image_prompt, title)
    out_path = _video_cache_path(image_prompt, title, float(duration_seconds), int(fps))
    if os.path.exists(out_path):
        return out_path

    # Build video fade in/out and audio fades using filter_complex
    fade_dur = min(1.0, max(0.0, duration_seconds / 3.0))
    fade_out_start = max(0.0, duration_seconds - fade_dur)

    vf = (
        f"[0:v]format=rgba,"
        f"fade=t=in:st=0:d={fade_dur:.3f},"
        f"fade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f},"
        f"fps={int(fps)},format=yuv420p[v]"
    )

    af = (
        f"[1:a]atrim=0:{duration_seconds:.3f},"
        f"afade=t=in:st=0:d={fade_dur:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_dur:.3f},"
        f"aformat=sample_rates=48000:channel_layouts=stereo[a]"
    )

    filter_complex = f"{vf};{af}"

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{duration_seconds:.3f}", "-i", img_path,
        "-stream_loop", "-1", "-t", f"{duration_seconds:.3f}", "-i", BACKGROUND_AUDIO_PATH,
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
        out_path,
    ]

    subprocess.run(cmd, check=True)
    return out_path

if __name__ == "__main__":
    path = makeSectionCardVideo(
        "A stark, black silhouette of a traditional Japanese torii gate stands against a deeply unsettling sky. The sky is a gradient of bruised purples at the top, bleeding down into a violent, blood-red horizon. Wisps of dark, moody clouds drift slowly, partially obscuring a pale, setting sun. The overall atmosphere is one of ominous beauty and impending dread, with no people or specific objects visible, just the setting itself.",
        "Forged in Fear",
        duration_seconds=5.0,
    )
    print(path)