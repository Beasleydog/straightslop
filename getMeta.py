import os
import json
import hashlib
import re
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from gemini import ask_gemini
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
CACHE_DIR = "cache/thumbnails"
TARGET_SIZE: Tuple[int, int] = (1920, 1080)
PADDING_PX: int = 100  # Padding from bottom and sides
MAX_FONT_SIZE: int = 400
MIN_FONT_SIZE: int = 40
TEXT_FILL = (255, 255, 255)  # White text
TEXT_STROKE_WIDTH = 4  # Black border width
TEXT_STROKE_FILL = (0, 0, 0)  # Black border
GRADIENT_HEIGHT_RATIO: float = 0.5  # Gradient covers bottom 40% of image

meta_prompt = """Developer: You receive a script as input. Generate a single JSON object containing video thumbnail information, a catchy title, an SEO description, and keywords, using the structure defined below. All output must be plain text—do not use markdown, special formatting, or additional syntax in any field.

Instructions:
- 'videoTitle': Write an attention-grabbing title that subverts expectations or popular beliefs, related but not identical to the thumbnail text.
- 'image': Create a detailed, click-maximizing prompt for an SFW thumbnail image. The image must not depict harm.
- 'text': Compose a very brief, enticing, easy-to-read phrase for thumbnail overlay, directly linked to the script and image. Use pronouns to reference the image subject, avoid complex grammar, keep it broad, and spark curiosity without giving away the video's content.
- 'description': Write an informative, compelling, SEO-optimized description (150-160 characters), including relevant keywords.
- 'keywords': Provide 5-8 relevant search terms as a comma-separated string, no spaces after commas.
- Output fields must be ordered: videoTitle, image, text, description, keywords.

Output Format:
Only return the JSON object below:
{{
  "videoTitle": "string",
  "image": "string",
  "text": "string",
  "description": "string",
  "keywords": "string"
}}

If the script is missing or empty, return all fields as null.
Script:
{script}
"""

def getMeta(script):
    """
    Get video metadata including title, thumbnail path, SEO description, and keywords
    
    Args:
        script (str): The video script
        
    Returns:
        dict: Contains 'title', 'thumbnail_path', 'description', and 'keywords'
    """
    # Get metadata from AI
    response = ask_gemini(meta_prompt.format(script=script))
    
    # Parse the response
    try:
        meta_data = _parse_meta_response(response)
        video_title = meta_data['videoTitle']
        image_prompt = meta_data['image']
        text = meta_data['text']
        description = meta_data['description']
        keywords = meta_data['keywords']
    except ValueError as e:
        raise ValueError(f"Failed to parse meta response: {e}\nResponse: {response}")
    
    # Generate thumbnail
    thumbnail_path = _generate_thumbnail_from_meta(script, image_prompt, text)
    
    return {
        'title': video_title,
        'thumbnail_path': thumbnail_path,
        'description': description,
        'keywords': keywords
    }

def _parse_meta_response(response: str) -> dict:
    """
    Parse meta response with robust handling of markdown and malformed JSON
    
    Args:
        response (str): Raw response from getMeta
        
    Returns:
        dict: Parsed metadata with videoTitle, image, and text
        
    Raises:
        ValueError: If parsing fails
    """
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```(?:json)?\s*', '', response.strip())
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    # Try to extract JSON from the response
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = cleaned
    
    try:
        # Parse JSON
        meta_data = json.loads(json_str)
        
        # Validate required fields
        if not isinstance(meta_data, dict):
            raise ValueError("Response is not a JSON object")
            
        required_fields = ["videoTitle", "image", "text", "description", "keywords"]
        for field in required_fields:
            if field not in meta_data:
                raise ValueError(f"Missing '{field}' field in response")
            
            if not isinstance(meta_data[field], str):
                raise ValueError(f"'{field}' field must be a string")
                
            if not meta_data[field].strip():
                raise ValueError(f"'{field}' field cannot be empty")
        
        return {
            'videoTitle': meta_data['videoTitle'].strip(),
            'image': meta_data['image'].strip(),
            'text': meta_data['text'].strip(),
            'description': meta_data['description'].strip(),
            'keywords': meta_data['keywords'].strip()
        }
        
    except json.JSONDecodeError as e:
        # Try to extract using regex as fallback
        title_match = re.search(r'"videoTitle"\s*:\s*"([^"]*)"', response, re.DOTALL)
        image_match = re.search(r'"image"\s*:\s*"([^"]*)"', response, re.DOTALL)
        text_match = re.search(r'"text"\s*:\s*"([^"]*)"', response, re.DOTALL)
        description_match = re.search(r'"description"\s*:\s*"([^"]*)"', response, re.DOTALL)
        keywords_match = re.search(r'"keywords"\s*:\s*"([^"]*)"', response, re.DOTALL)
        
        if title_match and image_match and text_match and description_match and keywords_match:
            return {
                'videoTitle': title_match.group(1).strip(),
                'image': image_match.group(1).strip(),
                'text': text_match.group(1).strip(),
                'description': description_match.group(1).strip(),
                'keywords': keywords_match.group(1).strip()
            }
        
        raise ValueError(f"Failed to parse JSON: {e}\nCleaned response: {json_str}")

def _generate_thumbnail_from_meta(script: str, image_prompt: str, text: str) -> str:
    """
    Generate thumbnail using provided image prompt and text
    
    Args:
        script (str): Original script (for caching)
        image_prompt (str): AI-generated image prompt
        text (str): Text to overlay on image
        
    Returns:
        str: Path to generated thumbnail
    """
    _ensure_cache_dir()
    
    # Check cache first
    out_path = _cache_path(script)
    if os.path.exists(out_path):
        print("Using cached thumbnail")
        return out_path
    
    # If we got a JPEG path but it doesn't exist, try PNG path for new generation
    if out_path.endswith('.jpg'):
        hash_name = os.path.splitext(os.path.basename(out_path))[0]
        out_path = os.path.join(CACHE_DIR, hash_name + ".png")
    
    print("Generating new thumbnail...")
    
    # Generate the base image
    print(f"Generating image with prompt: {image_prompt[:100]}...")
    image_path = generateImage(image_prompt, image_size="landscape_16_9")
    base_img = Image.open(image_path).convert("RGB")
    
    # Resize to target size
    img = _resize_cover(base_img, TARGET_SIZE)
    
    # For very high-resolution source images, ensure we don't create unnecessarily large files
    if img.size[0] * img.size[1] > TARGET_SIZE[0] * TARGET_SIZE[1]:
        img = img.resize(TARGET_SIZE, LANCZOS)
    
    # Create gradient overlay
    gradient_overlay = _create_gradient_overlay(TARGET_SIZE)
    
    # Apply gradient overlay
    img_with_gradient = Image.new('RGBA', TARGET_SIZE)
    img_with_gradient.paste(img, (0, 0))
    img_with_gradient = Image.alpha_composite(img_with_gradient, gradient_overlay)
    
    # Convert back to RGB for text drawing
    img_final = img_with_gradient.convert('RGB')
    draw = ImageDraw.Draw(img_final)
    
    # Calculate text area (bottom area with padding)
    target_w, target_h = TARGET_SIZE
    text_area_w = target_w - 2 * PADDING_PX
    text_area_h = int(target_h * GRADIENT_HEIGHT_RATIO) - PADDING_PX
    
    # Find best fitting font
    font = _best_fit_font(text, text_area_w, text_area_h)
    
    # Measure final text dimensions
    text_w, text_h = _measure_text(draw, text, font)
    
    # Position text at bottom center
    x = (target_w - text_w) // 2
    y = target_h - text_h - PADDING_PX - 50
    
    # Draw text with black stroke (border)
    draw.text(
        (x, y), 
        text, 
        font=font, 
        fill=TEXT_FILL, 
        stroke_width=TEXT_STROKE_WIDTH, 
        stroke_fill=TEXT_STROKE_FILL
    )
    
    # Save to cache with optimization for YouTube's 2MB limit
    img_final.save(out_path, format="PNG", optimize=True)
    
    # Check file size and reduce quality if needed
    file_size = os.path.getsize(out_path)
    max_size = 2 * 1024 * 1024  # 2MB in bytes
    
    if file_size > max_size:
        print(f"Thumbnail too large ({file_size / 1024 / 1024:.1f}MB), converting to JPEG...")
        # Convert to JPEG with quality adjustment
        jpeg_path = out_path.replace('.png', '.jpg')
        
        try:
            quality = 85
            success = False
            while quality > 50:
                img_final.save(jpeg_path, format="JPEG", quality=quality, optimize=True)
                new_size = os.path.getsize(jpeg_path)
                if new_size <= max_size:
                    # Remove the PNG and use JPEG
                    os.remove(out_path)
                    out_path = jpeg_path
                    print(f"Thumbnail optimized to {new_size / 1024 / 1024:.1f}MB")
                    success = True
                    break
                quality -= 10
            
            if not success:
                print("Warning: Could not reduce thumbnail below 2MB")
                # Clean up failed JPEG attempt and keep PNG
                if os.path.exists(jpeg_path):
                    os.remove(jpeg_path)
                    
        except Exception as e:
            print(f"Error optimizing thumbnail: {e}")
            # Clean up failed JPEG attempt and keep PNG
            if os.path.exists(jpeg_path):
                os.remove(jpeg_path)
    
    print(f"Thumbnail saved to: {out_path}")
    return out_path

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(script: str) -> str:
    """Generate cache file path based on script"""
    key_src = script.encode("utf-8")
    hash_name = hashlib.md5(key_src).hexdigest()
    
    # Check if JPEG version exists first (from previous optimization)
    jpeg_path = os.path.join(CACHE_DIR, hash_name + ".jpg")
    if os.path.exists(jpeg_path):
        return jpeg_path
    
    # Default to PNG
    return os.path.join(CACHE_DIR, hash_name + ".png")

def _pick_font(preferred_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pick the best available font"""
    candidate_paths = [
        "font.ttf"
    ]

    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, preferred_size)
        except Exception:
            continue
    # Fallback: default bitmap font
    return ImageFont.load_default()

def _resize_cover(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize image to cover target size (crop to fit)"""
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
    """Measure text dimensions"""
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=TEXT_STROKE_WIDTH)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def _best_fit_font(text: str, max_w: int, max_h: int) -> ImageFont.ImageFont:
    """Find the largest font size that fits within the given dimensions"""
    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_font = _pick_font(MIN_FONT_SIZE)
    
    # If truetype font not available, fallback immediately
    if not isinstance(best_font, ImageFont.FreeTypeFont):
        return best_font

    # Create a temporary draw object for measuring
    temp_img = Image.new("RGB", (max(1, max_w), max(1, max_h)))
    draw = ImageDraw.Draw(temp_img)
    
    while low <= high:
        mid = (low + high) // 2
        font_try = _pick_font(mid)
        w, h = _measure_text(draw, text, font_try)
        if w <= max_w and h <= max_h:
            best_font = font_try
            low = mid + 1
        else:
            high = mid - 1
    return best_font

def _create_gradient_overlay(size: Tuple[int, int]) -> Image.Image:
    """Create a black-to-transparent gradient overlay"""
    width, height = size
    gradient_height = int(height * GRADIENT_HEIGHT_RATIO)
    
    # Create gradient image
    gradient = Image.new('RGBA', (width, gradient_height), (0, 0, 0, 0))
    
    for y in range(gradient_height):
        # Alpha goes from 0 (top) to 180 (bottom) for a nice gradient effect
        alpha = int((y / gradient_height) * 180)
        for x in range(width):
            gradient.putpixel((x, y), (0, 0, 0, alpha))
    
    # Create full-size overlay with gradient at bottom
    overlay = Image.new('RGBA', size, (0, 0, 0, 0))
    overlay.paste(gradient, (0, height - gradient_height))
    
    return overlay

def _parse_thumbnail_response(response: str) -> Tuple[str, str]:
    """
    Parse thumbnail response with robust handling of markdown and malformed JSON
    
    Args:
        response (str): Raw response from makeThumbnail
        
    Returns:
        Tuple[str, str]: (image_prompt, text)
        
    Raises:
        ValueError: If parsing fails
    """
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```(?:json)?\s*', '', response.strip())
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    # Try to extract JSON from the response
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = cleaned
    
    try:
        # Parse JSON
        thumbnail_data = json.loads(json_str)
        
        # Validate required fields
        if not isinstance(thumbnail_data, dict):
            raise ValueError("Response is not a JSON object")
            
        if "image" not in thumbnail_data:
            raise ValueError("Missing 'image' field in response")
            
        if "text" not in thumbnail_data:
            raise ValueError("Missing 'text' field in response")
            
        image_prompt = thumbnail_data["image"]
        text = thumbnail_data["text"]
        
        # Validate field types
        if not isinstance(image_prompt, str):
            raise ValueError("'image' field must be a string")
            
        if not isinstance(text, str):
            raise ValueError("'text' field must be a string")
            
        # Validate content
        if not image_prompt.strip():
            raise ValueError("'image' field cannot be empty")
            
        if not text.strip():
            raise ValueError("'text' field cannot be empty")
            
        return image_prompt.strip(), text.strip()
        
    except json.JSONDecodeError as e:
        # Try to extract using regex as fallback
        image_match = re.search(r'"image"\s*:\s*"([^"]*)"', response, re.DOTALL)
        text_match = re.search(r'"text"\s*:\s*"([^"]*)"', response, re.DOTALL)
        
        if image_match and text_match:
            return image_match.group(1).strip(), text_match.group(1).strip()
        
        raise ValueError(f"Failed to parse JSON: {e}\nCleaned response: {json_str}")


if __name__ == "__main__":
    # Test with a sample script
    test_script = """
    Forget what you think you know aboutsamurai. We often romanticize them asnoble, honorable masters of the blade.But the true path to becoming one wasnot just rigorous. It was a soulccrushing, mindbending, brutally extremegauntlet designed to break all but themost unyielding. This wasn't merelyabout learning how to wield a katana. Itwas about stripping away fear, pain, andindividuality. transforming a humanbeing into an emotionless weapon in aterrifying crucible of discipline anddeath. From their earliest childhood,every breath was a step deeper into thisrelentless ordeal.So, what truly happened when a younghopeful dared to walk the path of thesamurai? Prepare to discover theunimaginable trials that forgedhistory's most legendary warriors.[Music]Before we delve into the sheer brutalityof samurai initiation, it's crucial tounderstand why such extremity was notonly accepted but demanded. The samuraiwere not merely warriors. They formedthe elite ruling class of feudal Japan.Their very existence predicated onunwavering loyalty, marshall prowess,and an unshakable readiness to die fortheir lord. This wasn't a job. It was asacred calling where the fate of entiredomains often rested on their blades.Such an immense burden necessitated anequally immense commitment to training,forging not just skilled fighters, butparagonss of discipline and honor. Thisfoundational philosophy codified asbushido or the way of the warrior wentfar beyond physical combat. Itencompassed an entire code of ethicsdemanding absolute loyalty, courage inthe face of death, unwaveringself-control, and a stoic acceptance ofone's fate. It was this profoundsocietal role and the tenets of Bushidothat provided the bedrock for theintensity of their training,transforming mere individuals intoliving weapons imbued with anunbreakable spirit.It wasn't a journey they embarked uponas adults. The crucible of samuraitraining began almost from the momentthey could walk. From the tenderest age,these children were subjected to anintense psychological conditioningmeticulously crafted to forge unshakablediscipline and an iron will. Everylesson, every game, every interactionwas a deliberate step towards instillingunwavering obedience, an uncannytolerance for pain, and perhaps mostcrucially, the absolute suppression offear. They learned not to flinch, not tocry out, and certainly not to retreat.their young minds being systematicallyrewired to view discomfort as a lessonand danger as merely another challengeto be overcome long before they everheld a real katana.Beyond the initial grueling years, thepivotal moment for a young samuraiarrived with the genpuku, the coming ofage ceremony. While often romanticallysimplified to a mere haircut and thebestowyl of a new adult name for thesenent warriors, this was far from aquaint ritual. This wasn't just aboutexchanging a boy's top knot for awarrior's shaved pate. It was aprofound, often physically rigoroustransition. Picture, if you will, thesymbolic, yet intensely demandingcomponents. Hours spent in icy coldbaths for spiritual purification orenduring long periods of kneeling insilent contemplation, pushing the veryboundaries of physical discomfort andmental resolve. The dawning of new adultrobes, often presented by a godfather,symbolized not just a change inappearance, but a solemn vow to upholdthe Bushidto code, a lifealteringcommitment to a path of unwaveringdiscipline and potential death on thebattlefield. This was the precise momenta boy officially shed his childhood,stepping into the formidable boots of afullyfledged warrior.Once their minds were being molded,their bodies were subjected to an evenmore relentless forge. Imagine daysblurring into nights, not from revalry,but from deliberate, agonizing sleepdeprivation, perhaps through endlessmeditation or ritualistic chantingdesigned to push the very limits ofhuman consciousness.Then picture them confronting the raw,unyielding power of nature, meditatingstoically beneath frigid winterwaterfalls, their bodies shiveringuncontrollably or standing for hoursunder the merciless glare of the summersun, their skin baking. These weren'tmerely exercises. They were crucibles ofendurance, stripping away every comfortand exposing the raw essence of theirwill. Every ache, every shiver, everygnawing pang of hunger was a deliberatelesson in resilience. Transforming theirphysical weakness into an unyielding,unbreakable resolve, forging a body assharp and indomitable as their blade.Beyond mere physical endurance, samuraiinitiates were deliberately plunged intothe crucible of pain, not as punishment,but as a forge for their minds. Thiswasn't about gratuitous suffering, butabout controlled, self-inflicteddiscomfort, designed to cultivate anunbreakable mental fortitude. Pictureyoung warriors perhaps holdingincredibly strenuous stances for hourson end until muscles screamed, orenduring extremes of heat and cold withonly their willpower as a shield. Someaccounts suggest disciplined repetitivestrikes to areas of the body, not tocause injury, but to desensitize andaccustom them to the shocks of battle.The objective was clear. To master theirown physiological responses, to embracediscomfort, and to systematicallydismantle the natural human urge torecoil from pain, transforming it into atool for unwavering focus and anunyielding will.Beyond the grueling physical demands,samurai initiates faced psychologicaltrials designed to rip fear from theirvery souls. Imagine a young recruitalone spending a desolate night in asilent moonlit graveyard, surrounded bythe whispers of the departed. Thiswasn't merely a macabra tale. It was aprofound immersion into death's domain,forcing them to confront their deepestanxieties about mortality and theunknown, transforming primal dread intoresolute calm. Even more harrowing werethe combat simulations, not with bluntedtraining weapons, but often with realsharpened blades, where the line betweenpractice and peril blurred dangerouslyclose. These weren't just drills. Theywere meticulously orchestratedencounters engineered to inflict intensepsychological pressure and real physicalrisk, pushing initiates to overcome theprimal instinct to flee and insteadembrace the chaos and danger of truebattle, proving their unwavering couragewhen faced with actual harm.But the path to becoming a true samuraiextended far beyond mere physicalendurance. Once the body had been pushedto its limits, the mind became theultimate battleground. These were thepsychological and loyalty tests.Cunningly designed not just to testresolve, but to systematically unravelany personal will or independentthought. breaking down individualidentity to forge absolute devotion totheir lord and the sacred samurai code.Recruits faced a gauntlet of calculateddeception, moral dilemmas engineered toforce impossible choices, and evenstaged betrayals among their peers. Allmeticulously observed to see who wouldfalter, who would uphold the hierarchy,and who would prove their unwaveringfeelalty. The aim was nothing less thantotal psychological re-engineering,molding their very spirit into anunyielding instrument of their do'swill, ensuring that when the time came,their loyalty would be as sharp andunbending as their katana.After enduring years of unimaginablephysical and psychological torment, fromthe frigid mountains to the hauntedgraveyards, a young samurai's initiationculminated in one final brutal provingground. Their first taste of realcombat. This wasn't merely a skirmish.It was the ultimate test where everylesson learned, every fear suppressed,every ounce of discipline honed was putto the most unforgiving trial. The FirstBlood wasn't just about survival orvictory. It was about demonstratingunwavering courage in the face of death,upholding the samurai code under extremeduress, and proving unequivocally theirworthiness of the warrior class. It wasthe moment the theoretical becameterrifyingly real, solidifying theirtransformation from a mere student ofthe sword into a true battlehhardardened samurai ready to lay downtheir life for their lord at a moment'snotice.So after witnessing the relentlessgauntlet these young samurai enduredfrom childhood conditioning to facingdeath itself, the crucial questionremains. Why? The ultimate purposebehind every single brutal test, everymoment of deprivation, everyconfrontation with fear was singular andprofound. It wasn't just about creatingskilled fighters. It was about forgingunyielding warriors, individuals whosediscipline was ironclad, whose mindswere utterly fearless in the face of anythreat, and whose spirits embraced aprofound readiness for death, not as anend, but as an honorable inevitability.These rituals systematically strippedaway every weakness, every shred ofhesitation, leaving behind only thepure, distilled essence of a samurai. Aperfect weapon bound by honor, preparedfor anything, and ultimately ready todie for their lord without a moment'spause.And so, we've journeyed through thecrucible that forged the samurai. Itwasn't just battle prowess or a sharpkatana that defined these legendarywarriors. It was the unyielding spirit,the iron discipline, and the absolutereadiness for death, meticulouslychiseled into their very being by thesebrutal, mindshattering, and bodybuakinginitiation rights. This isn't merelyhistory. It's the very bedrock of thesamurai mystique. The hidden storybehind every epic tale of their courageand honor, reminding us that truelegends are not born, but brutallyforged. What part of their journeysurprised you the most or perhapsinstilled the deepest respect? Let meknow in the comments below. Smash thatlike button if you found this deep divefascinating, and don't forget tosubscribe for more journeys into theextreme corners of history. Until nexttime, stay vigilant."""
    
    # Test the new getMeta function
    result = getMeta(test_script)
    print(f"Video Title: {result['title']}")
    print(f"Thumbnail Path: {result['thumbnail_path']}")
    print(f"Description: {result['description']}")
    print(f"Keywords: {result['keywords']}")
