import time
from getPlan import getPlan
from getTimestamps import getTimestamps
from getTTS import getTTS
from generateImage import generateImage
from makeAnimation import ken_burns_effect_ffmpeg
from combine_clips import combine_clips
from combine_with_fades import combine_clips_with_fades, combine_two_clips_with_fades_av
from makeSectionCard import makeSectionCardVideo
from addOverlay import addOverlay
from getMeta import getMeta
from upload_video import publish_simple
from makeIntro import makeIntro

import subprocess
import tempfile
import os
import math

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Read first line from next_ideas.txt and move it to done_ideas.txt
def update_default_title():
    next_ideas_file = "next_ideas.txt"
    done_ideas_file = "done_ideas.txt"
    
    if os.path.exists(next_ideas_file):
        with open(next_ideas_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if lines:
            # Get first line and strip whitespace
            first_line = lines[0].strip()
            
            # Write remaining lines back to next_ideas.txt
            with open(next_ideas_file, 'w', encoding='utf-8') as f:
                f.writelines(lines[1:])
            
            # Append first line to done_ideas.txt
            with open(done_ideas_file, 'a', encoding='utf-8') as f:
                f.write(first_line + '\n')
            
            return first_line
        else:
            raise Exception("No ideas available in next_ideas.txt")
    else:
        raise Exception("next_ideas.txt file not found")
# Update DEFAULT_TITLE with first line from next_ideas.txt
DEFAULT_TITLE = update_default_title()

# Config
# DEFAULT_TITLE = "The Most Brutal Rituals of Samurai Initiation."
LAST_IMAGE_EXTRA_SECONDS = 1.0
FIRST_IMAGE_EXTRA_SECONDS = 1.0
CROSSFADE_SECONDS = 0.5
SECTION_CARD_LENGTH_SECONDS = 5.0

def mux_audio(video_in: str, audio_in: str, output_path: str, audio_offset_seconds: float = 0.0) -> str:
    """Mux audio onto a video; optionally delay audio by audio_offset_seconds with inserted silence."""
    if audio_offset_seconds and audio_offset_seconds > 0:
        delay_ms = int(round(audio_offset_seconds * 1000))
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", audio_in,
            "-filter_complex", f"[1:a]adelay={delay_ms}:all=1[a]",
            "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", audio_in,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            output_path,
        ]
    subprocess.run(cmd, check=True)
    return output_path


def main() -> None:
    start_time = time.time()
    
    # Plan the video
    vo_sections, plan = getPlan(DEFAULT_TITLE, model="gemini-2.5-flash")


    print(vo_sections)
    section_outputs: list[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each section independently
        for i, section_obj in enumerate(tqdm(vo_sections, desc="Sections", unit="sec")):
            prev_text = vo_sections[i - 1] if i > 0 else None

            # 1) Generate TTS for the section (use the 'script' field from the section object)
            prev_script = prev_text["script"] if isinstance(prev_text, dict) else None
            audio_path = getTTS(section_obj["script"], previous_text=prev_script)

            # 2) Generate images and compute timestamps within the section
            items = []
            for plan_item in plan[i]:
                image_path = generateImage(plan_item["description"])
                ts = getTimestamps(audio_path, plan_item["vo"])
                items.append({
                    "image": image_path,
                    "start": float(ts["start"]),
                    "end": float(ts["end"]),
                })

            if not items:
                continue

            # Compute frame-accurate cut boundaries based on START times to avoid early cuts
            fps = 60
            starts = [float(it["start"]) for it in items]
            ends = [float(it["end"]) for it in items]

            boundaries: list[int] = [0]
            for k in range(1, len(items)):
                # Align cut at or after the next VO's start time
                next_start_frames = math.ceil(starts[k] * fps)
                boundaries.append(max(boundaries[-1] + 1, next_start_frames))
            # Last boundary: end of last VO + extra hold
            last_end_frames = math.ceil((ends[-1] + float(LAST_IMAGE_EXTRA_SECONDS)) * fps)
            boundaries.append(max(boundaries[-1] + 1, last_end_frames))

            # Extend the first image by FIRST_IMAGE_EXTRA_SECONDS by shifting all subsequent
            # boundaries forward. This preserves all other per-item durations.
            extra_frames_first = int(math.ceil(float(FIRST_IMAGE_EXTRA_SECONDS) * fps))
            if extra_frames_first > 0:
                for idx in range(1, len(boundaries)):
                    boundaries[idx] += extra_frames_first

            # Convert boundaries to per-item frame counts
            frames_per_item: list[int] = []
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                frames_per_item.append(max(2, b - a))

            # 3) Animate each image for its duration
            clip_paths: list[str] = []
            for j, (it, nframes) in enumerate(zip(items, frames_per_item)):
                clip_path = ken_burns_effect_ffmpeg(it["image"], nframes / fps, total_frames_override=int(nframes))
                clip_paths.append(clip_path)

            # 4) Combine animated image clips for this section (with crossfades)
            combined_images = os.path.join(temp_dir, f"section_{i}_visual.mp4")
            combine_clips_with_fades(clip_paths, combined_images, fade_seconds=CROSSFADE_SECONDS, fps=fps)

            # 5) Add section audio to the combined visuals (delay audio by FIRST_IMAGE_EXTRA_SECONDS)
            section_out = os.path.join(temp_dir, f"section_{i}.mp4")
            mux_audio(combined_images, audio_path, section_out, audio_offset_seconds=float(FIRST_IMAGE_EXTRA_SECONDS))

            # 6) For the first section, skip the section card. For subsequent sections, build and prepend section card
            if i == 0:
                # First section: use the section directly without a card
                section_outputs.append(section_out)
            else:
                # Subsequent sections: build cached section card video and prepend to section
                card_clip = makeSectionCardVideo(
                    image_prompt=section_obj["backgroundImage"],
                    title=section_obj["title"],
                    duration_seconds=SECTION_CARD_LENGTH_SECONDS,
                    fps=fps,
                )
                section_with_card = os.path.join(temp_dir, f"section_{i}_with_card.mp4")
                # Preserve and crossfade both video and audio between the card and the section
                combine_two_clips_with_fades_av(card_clip, section_out, section_with_card, fade_seconds=CROSSFADE_SECONDS, fps=fps)
                section_outputs.append(section_with_card)

        # 7) Combine all sections into one final video
        if not section_outputs:
            raise RuntimeError("No sections were produced.")

        print("Concatenating sections...")
        combine_clips(section_outputs, "output.mp4")
        
        # 8) Generate intro and combine with main video
        # Combine all scripts for intro generation
        combined_script = "\n".join([section["script"] for section in vo_sections])
        
        print("Generating intro...")
        intro_path = makeIntro(combined_script, "intro.mp4")
        
        print("Combining intro with main video...")
        video_with_intro = os.path.join(temp_dir, "video_with_intro.mp4")
        combine_two_clips_with_fades_av(intro_path, "output.mp4", video_with_intro, fade_seconds=CROSSFADE_SECONDS, fps=60)
        
        # Add overlay to the combined video (intro + main)
        print("Adding overlay...")
        overlay_video_path = addOverlay(video_with_intro)
        print(f"Overlay complete! Output: {overlay_video_path}")
        
        # Generate metadata and thumbnail
        print("Generating metadata and thumbnail...")
        meta_data = getMeta(combined_script)
        print(f"Metadata generated - Title: {meta_data['title']}")
        print(f"Thumbnail saved: {meta_data['thumbnail_path']}")
        
        # Upload video to YouTube
        print("Uploading to YouTube...")
        watch_url = publish_simple(
            title=meta_data['title'],
            file_path=overlay_video_path,
            description=meta_data['description'],
            thumbnail_path=meta_data['thumbnail_path'],
            category="27",  # Education category
            keywords=meta_data['keywords']
        )
        print(f"Upload complete! Video URL: {watch_url}")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60
    
    print("Video generation complete! Output: output.mp4")
    print(f"Overlay output: {overlay_video_path if 'overlay_video_path' in locals() else 'N/A'}")
    print(f"YouTube URL: {watch_url if 'watch_url' in locals() else 'N/A'}")
    print(f"Total elapsed time: {minutes}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
