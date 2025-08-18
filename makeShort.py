from gemini import ask_gemini
from getTTS import getTTS
from generateImage import generateImage
from getTimestamps import getTimestamps
from makeAnimation import ken_burns_effect_ffmpeg
from combine_with_fades import combine_clips_with_fades
from captions import add_tiktok_captions
import json, os, tempfile, math, subprocess

# Visual pacing for shorts
CROSSFADE_SECONDS = 0.4
FIRST_IMAGE_EXTRA_SECONDS = 0.0
LAST_IMAGE_EXTRA_SECONDS = 1.0

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

SCRIPT_PROMPT="""You've been given a full youtube video script. Output a 30 second VO for a short form video that will also be posted to promote and direct the audience to the longer short.
The short should open with exclaiming one word (the main topic of the video) to hook the audience. (use exclamation points)Then, it should go into a very abbreviated description of the video. It should leave the viewer with enough intrigue and then direct them to check out the newest video on the channel to watch the full thing.
Note that throughout all language should be simple. Don't over describe things, it can be slightly informal, it shouldn't feel like word salad.
Just return one paragraph for the VO, nothing else.

The script is:
{script}"""


GET_MEDIA_PROMPT = """You've been given the script for a short form video.

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

Script:
{script}
"""
def _parse_media_items(raw: str) -> list[dict]:
	clean = raw.strip()
	if clean.startswith('```json'):
		clean = clean[7:]
	if clean.endswith('```'):
		clean = clean[:-3]
	clean = clean.strip()
	data = json.loads(clean)
	if not isinstance(data, list):
		raise ValueError("Expected JSON array of media items")
	out: list[dict] = []
	for it in data:
		if not isinstance(it, dict):
			continue
		desc = it.get("description") or it.get("image") or ""
		vo = it.get("vo") or it.get("text") or ""
		if desc and vo:
			out.append({"description": str(desc), "vo": str(vo)})
	return out


def makeShort(script: str, output_filename: str = "short.mp4") -> str | None:
	print("Generating short VO text...")
	vo_text = ask_gemini(SCRIPT_PROMPT.format(script=script), model="gemini-2.5-pro")
	print(f"Short VO: {vo_text}")

	print("Planning visuals...")
	with_images_raw = ask_gemini(GET_MEDIA_PROMPT.format(script=vo_text), model="gemini-2.5-pro")

	try:
		media_items = _parse_media_items(with_images_raw)
	except Exception as e:
		print(f"Error parsing media JSON: {e}")
		print(with_images_raw)
		return None

	if not media_items:
		print("No media items returned")
		return None

	print(f"Parsed {len(media_items)} media items")

	print("Generating TTS...")
	audio_path = getTTS(vo_text)

	with tempfile.TemporaryDirectory() as temp_dir:
		items: list[dict] = []
		for it in media_items:
			print(f"Image: {it['description']}")
			img_path = generateImage(it["description"],image_size="portrait_16_9")
			ts = getTimestamps(audio_path, it["vo"])  # {start, end}
			items.append({
				"image": img_path,
				"start": float(ts["start"]),
				"end": float(ts["end"]),
			})

		if not items:
			print("No items to process")
			return None

		fps = 60
		fade_frames = max(1, int(round(CROSSFADE_SECONDS * fps)))
		starts = [float(it["start"]) for it in items]
		ends = [float(it["end"]) for it in items]

		# Boundaries by VO start (ceil), preserve holds like index/makeIntro
		boundaries: list[int] = [0]
		for k in range(1, len(items)):
			next_start_frames = math.ceil(starts[k] * fps)
			boundaries.append(max(boundaries[-1] + 1, next_start_frames))

		last_end_frames = math.ceil((ends[-1] + float(LAST_IMAGE_EXTRA_SECONDS)) * fps)
		boundaries.append(max(boundaries[-1] + 1, last_end_frames))

		extra_frames_first = int(math.ceil(float(FIRST_IMAGE_EXTRA_SECONDS) * fps))
		if extra_frames_first > 0:
			for idx in range(1, len(boundaries)):
				boundaries[idx] += extra_frames_first

		# Convert boundaries to planned spans per item (in frames)
		s_spans: list[int] = []
		for a, b in zip(boundaries[:-1], boundaries[1:]):
			s_spans.append(max(1, b - a))

		# Backward fade-compensated allocation to align xfade starts with VO starts:
		# For i < N-1: choose n_i such that n_i - f_i = s_i, where f_i = min(F, n_{i+1})
		# For last: n_last = s_last
		F = max(1, int(round(CROSSFADE_SECONDS * fps)))
		N = len(s_spans)
		if N == 1:
			frames_per_item = [max(2, s_spans[0])]
		else:
			n_values = [0] * N
			# last clip
			n_values[N - 1] = max(2, s_spans[N - 1])
			# backward pass
			for i in range(N - 2, -1, -1):
				f_i = min(F, n_values[i + 1])
				n_values[i] = max(2, s_spans[i] + f_i)
			frames_per_item = n_values

		print("Animating images (9:16, fast path)...")
		clip_paths: list[str] = []
		for it, nframes in zip(items, frames_per_item):
			clip_paths.append(
				ken_burns_effect_ffmpeg(
					it["image"],
					nframes / fps,
					total_frames_override=int(nframes),
					target_size=(1080, 1920),
					fps_override=fps,
				)
			)

		print("Combining clips with crossfades...")
		combined_path = os.path.join(temp_dir, "short_visual.mp4")
		combine_clips_with_fades(clip_paths, combined_path, fade_seconds=CROSSFADE_SECONDS, fps=fps)

		print("Muxing audio...")
		mux_audio(combined_path, audio_path, output_filename, audio_offset_seconds=float(FIRST_IMAGE_EXTRA_SECONDS))

		print("Adding captions...")

		output_captions_filename = "short_captions.mp4"
		# Burn TikTok-style captions onto the final short, overwrite same file
		add_tiktok_captions(output_filename, output_captions_filename)

		print(f"Short video complete: {output_filename}")
		return output_captions_filename

if __name__ == "__main__":
    print(makeShort("""Before they ever drew a blade in battle,
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