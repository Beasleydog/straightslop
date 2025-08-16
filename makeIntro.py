from getMeta import ask_gemini
from getTTS import getTTS
from generateImage import generateImage
from getTimestamps import getTimestamps
from makeAnimation import ken_burns_effect_ffmpeg
from combine_with_fades import combine_clips_with_fades
import json
import os
import tempfile
import math

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Video generation configuration
CROSSFADE_SECONDS = 0.5
FIRST_IMAGE_EXTRA_SECONDS = 0.0
LAST_IMAGE_EXTRA_SECONDS = 4.0

def mux_audio(video_in: str, audio_in: str, output_path: str, audio_offset_seconds: float = 0.0) -> str:
    """Mux audio onto a video; optionally delay audio by audio_offset_seconds with inserted silence."""
    import subprocess
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

VO_PROMPT = """Output a 10 second script (just paragraph of text) for the intro of this video.

NEVER USE ANY MARKDOWN SYNTAX IN THE SCRIPT. JUST STRAIGHT TEXT.

It must hook the viewer in, teasing the ending and encouraging them to stay.
It should be a bit meta, welcoming the viewer to the start of a new vid. Adress the history viewer as somebody who likes history. 
At the end, it should encourage them to sit back/relax/settle in/etc and enjoy the entire thing.

It should make the watcher feel the following:
- Excited, eager to learn about cool content
- Welcome, like they are wanted
- Curious, like theres stuff about the topic that they don't know
- Wanting to watch the entire length of the video

Script:
{script}
"""

GET_MEDIA_PROMPT = """You've been given the intro section and the full script of a video.

You must choose media for the VO SECTION.

Output a json array like this:
{{
"description": "the image that should be showing",
"vo": "the portion of the script"
}},{{
"description": "the image that should be showing",
"vo": "the portion of the script"
}}

Description Guidelines:
- Describe a static image, no animation or video
- Describe the image in detail, including the background, the main subject, and any other details
    - GREAT FUCKING DETAILS
- The image should NEVER be complex like a point on a map or a graph
- If theres no clear visual for the script portion, just describe a more general image that fits in with the rest of the script context
- Pretty much NEVER use text in the image
- Never make an image to represent any of the following:
    - Subscribe button
    - Leaving comments on the video
- Make quick cuts, never have the same image for a long period of time
    - This means we should use lots of media, each just showing for a short phrase/sentence of VO

Intro VO:
{intro_vo}
"""



def makeIntro(script, output_filename="intro.mp4"):
    """Generate an intro video from the script."""
    print("Generating intro script...")
    response = ask_gemini(VO_PROMPT.format(script=script), model="gemini-2.5-pro")
    print(f"Intro script: {response}")
    
    print("Getting media descriptions...")
    with_images = ask_gemini(GET_MEDIA_PROMPT.format(intro_vo=response, script=script), model="gemini-2.5-pro")
    print(f"Media descriptions: {with_images}")

    # Parse the JSON response
    try:
        # Clean up the response if it has markdown code blocks
        clean_json = with_images.strip()
        if clean_json.startswith('```json'):
            clean_json = clean_json[7:]
        if clean_json.endswith('```'):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()
        
        # Parse as JSON array
        media_items = json.loads(clean_json)
        if not isinstance(media_items, list):
            raise ValueError("Expected JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response was: {with_images}")
        return None

    print(f"Parsed {len(media_items)} media items")

    # Generate TTS for the intro script
    print("Generating TTS audio...")
    audio_path = getTTS(response)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate images and get timestamps for each media item
        items = []
        for i, plan_item in enumerate(tqdm(media_items, desc="Processing media items")):
            print(f"Generating image for: {plan_item['description']}")
            image_path = generateImage(plan_item["description"])
            
            print(f"Getting timestamps for: {plan_item['vo']}")
            ts = getTimestamps(audio_path, plan_item["vo"])
            
            items.append({
                "image": image_path,
                "start": float(ts["start"]),
                "end": float(ts["end"]),
            })

        if not items:
            print("No items to process")
            return None

        # Compute frame-accurate cut boundaries based on START times
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

        # Extend the first image by FIRST_IMAGE_EXTRA_SECONDS
        extra_frames_first = int(math.ceil(float(FIRST_IMAGE_EXTRA_SECONDS) * fps))
        if extra_frames_first > 0:
            for idx in range(1, len(boundaries)):
                boundaries[idx] += extra_frames_first

        # Convert boundaries to per-item frame counts
        frames_per_item: list[int] = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            frames_per_item.append(max(2, b - a))

        # Animate each image for its duration
        print("Creating animated clips...")
        clip_paths: list[str] = []
        for j, (it, nframes) in enumerate(zip(items, frames_per_item)):
            clip_path = ken_burns_effect_ffmpeg(it["image"], nframes / fps, total_frames_override=int(nframes))
            clip_paths.append(clip_path)

        # Combine animated image clips with crossfades
        print("Combining clips with crossfades...")
        combined_images = os.path.join(temp_dir, "intro_visual.mp4")
        combine_clips_with_fades(clip_paths, combined_images, fade_seconds=CROSSFADE_SECONDS, fps=fps)

        # Add audio to the combined visuals (delay audio by FIRST_IMAGE_EXTRA_SECONDS)
        print("Muxing audio with video...")
        mux_audio(combined_images, audio_path, output_filename, audio_offset_seconds=float(FIRST_IMAGE_EXTRA_SECONDS))

        print(f"Intro video complete: {output_filename}")
        return output_filename


if __name__ == "__main__":
    print(makeIntro("""Before they ever drew a blade in battle,
before they earned the right to bear the
formidable katana, aspiring samurai
faced a journey far more terrifying than
any battlefield. A series of initiation
rituals so profoundly brutal, so utterly
unforgiving. They transcended mere
training to become a crucible of the
soul. This wasn't just about learning to
fight. It was about systematically
dismantling the boy and forging through
unimaginable hardship an unyielding
warrior ready to embrace a life where
death was a constant companion and
loyalty an absolute law. From the moment
they stepped onto this path, these young
men were plunged into a world where pain
was a teacher, fear a test, and survival
itself the ultimate lesson, preparing
them for the harsh realities that
awaited Japan's elite warrior class.
So why inflict such agonizing trials?
This wasn't merely about cruelty. It was
a deliberate, brutal forge. Every moment
of pain, every instance of deprivation
was meticulously designed to strip away
weakness, to obliterate individual will,
and to replace it with an ironclad
discipline. It was about etching
absolute, unquestioning loyalty into
their very bones, ensuring that when the
moment of truth arrived on the
battlefield, there would be no
hesitation, no fear, only an unyielding
spirit devoted entirely to their lord
and the warrior's path. These were not
just ceremonies. They were an extreme
education in survival, crafting warriors
whose minds and bodies were tempered to
face death headon without a flinch.
The Crucible truly began with Genpuku,
the coming of age ceremony that was less
a joyous celebration and more a stark,
unforgiving induction into the brutal
realities of samurai life. Typically
performed between the ages of 10 and 15,
this wasn't merely about dawning adult
clothes or a new haircut. It was a
profound symbolic shedding of childhood
marked by the receiving of one's first
adult name and often the dawning of a
real katana. From that moment forward,
the young man was no longer a boy, but a
full-fledged warrior, instantly burdened
with the absolute responsibilities of a
samurai. There was no gentle easing in.
The expectation was immediate.
Unflinching loyalty, readiness for
combat, and an unwavering commitment to
the warrior code, often demanding
sacrifice and hardship just moments
after his childhood name was
ceremonially shed.
Beyond the ceremonial robes and symbolic
haircuts, the path to becoming a samurai
plunged initiates into a relentless
gauntlet of physical torment. Designed
not just to strengthen the body, but to
shatter and rebuild the spirit. They
endured prolonged fasting, their
stomachs gnawing, not for days, but
often for weeks, pushing the very limits
of human endurance to teach absolute
self-control over primal urges. Sleep
became a luxury, then a forgotten dream.
As continuous drills and meditations
frayed their minds, stripping away
weakness and cultivating unshakable
mental fortitude, recruits were
subjected to the brutal whims of nature
itself, left exposed to the biting
blizzards of winter or the suffocating
heat of summer, forcing them to find
resilience in the face of environmental
extremes. And perhaps most daunting,
they cultivated an almost inhuman pain
tolerance, learning to ignore cuts,
bruises, and exhaustion. For in the
crucible of battle, a samurai spirit
could never break, even if their body
might.
But the trials of the samurai extended
far beyond the physical to forge truly
unshakable warriors. Their minds had to
be as tempered as their blades. This
meant confronting fear itself, often
through harrowing psychological
conditioning. Aspiring samurai would be
sent to meditate in the most chilling of
locations. Forgotten temples shrouded in
mist, ancient graveyards where the
whispers of the departed seemed to hang
in the air, or even sights of past
battles where the echoes of bloodshed
lingered. Here, amidst the palpable
presence of death, they were forced to
sit with their own mortality to confront
the deepest human fears head on. This
wasn't just about bravery. It was about
internalizing death as a constant
compion, stripping away all fear of it,
making them utterly fearless in the face
of combat. The goal was an unyielding
spirit, a stoicism so profound that even
the spectre of oblivion could not break
their resolve.
Beyond the grueling physical and mental
conditioning, the true crucible of the
samurai spirit lay in the dojo. This
wasn't merely practice. It was a brutal
proving ground of skill and survival.
Young warriors engaged in live sparring,
often wielding not blunted training
implements, but actual potentially
lethal weapons. Imagine the clang of
steel, the whistle of a near miss, the
searing pain of a blade making contact.
Injuries, from deep cuts to broken
bones, were not uncommon. Yet, they were
never seen as failures. Instead, each
scar was a visceral lesson, a testament
to resilience, and a brutal badge of
honor. This constant exposure to danger,
this dance with death, was the ultimate
test, forging the instinctual reactions
and unyielding fortitude essential for a
life lived on the edge of a blade.
But beyond the grueling physical and
mental conditioning, the true
inescapable binding of an aspiring
samurai occurred through the solemn act
of oathtaking.
These weren't mere promises. They were
sacred life or death vows, often sworn
before a deity or ancestral spirits,
etching an unbreakable commitment into
their very soul. This ultimate pledge
solidified absolute allegiance not just
to their chosen lord, but to the
rigorous, all-encompassing tenets of the
samurai code itself. To break such an
oath was to invite not only a swift,
brutal end, but eternal dishonor for
oneself and one's family, ensuring these
commitments were the unyielding
cornerstone of their existence,
demanding unwavering loyalty until their
final breath.
So after enduring trials of mind, body
and spirit, the prolonged fasting, the
agonizing sleepless nights, the
terrifying meditations among the dead,
the brutal live sparring combat, and the
sacred unbreakable oaths. What emerged
from this relentless crucible of pain
and purpose? Not merely a soldier, but a
samurai perfected. These were not just
training exercises. They were alchemical
processes systematically burning away
weakness, forging an indomitable will
and tempering the spirit until only
absolute discipline remained. Every fear
was confronted and conquered, every
doubt purged, every vestage of
self-preservation
replaced by an unshakable fanatical
loyalty. The ideal samurai, therefore,
was not simply born. They were made
through an ordeal so profound it
transmuted human flesh and bone into a
living instrument of war, utterly
fearless in the face of death and
devoted beyond measure to their lord and
their code, ready to strike, unflinching
at a moment's notice.
So, as we've journeyed through the
crucible of their coming of age, it
becomes undeniably clear these weren't
merely rituals, but profound rights of
passage designed to shatter the ordinary
and forge the extraordinary.
Every gruelling trial from the solemn
Genuku ceremony to the mindbending
psychological conditioning and the
brutal livesparring combat drills
stripped away fear, doubt, and ego,
leaving behind an unyielding warrior, a
paragon of discipline, absolute loyalty,
and an almost supernatural stoicism. The
samurai's character honed in this fire
created an enduring ethos that continues
to echo through history, a testament to
the power of self-mastery and an
unwavering commitment to one's code.
We've seen how their foundations were
built on such extreme principles, but
what truly resonates with Aou about the
samurai spirit? Do you think such
rigorous training could still forge such
character today? Share your insights and
reflections in the comments below. Your
thoughts are truly valued. And if you
found this deep dive into the samurai's
world as compelling as we did, please
hit that like button, subscribe for more
historical explorations, and ring the
notification bell so you don't miss our
next adventure. Yeah."""))