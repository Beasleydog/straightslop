
from google import genai
from google.genai import types
import base64
import os
import hashlib
from gemini import ask_gemini

from dotenv import load_dotenv
load_dotenv()

# Cache setup
CACHE_DIR = "cache/googleimages"
os.makedirs(CACHE_DIR, exist_ok=True)

# Prompt template preserved from original
# PROMPT_TEMPLATE = """
# ```json
# {{"image": "{prompt}", "style": "vector icon", "background": "white"}}
# ```
# """
PROMPT_TEMPLATE = """In the style of surreal, hand-drawn, vintage etching illustrations. Use extremely detailed, fine ink-like linework with precise cross-hatching and stippling for shading. Maintain consistent thin-to-medium line weights, with meticulous contour lines defining shapes and textures. The drawing should fill the entire frame, with no large empty backgrounds, and be fully colored. The composition should have strong outlines, intricate detail work, and minimal gradients, resembling a classic engraved illustration with a surreal or symbolic twist in subject matter.
{prompt}"""

def _cache_path(prompt, aspect_ratio="16:9", image_size="2K", seed=None):
    """Generate cache file path based on parameters"""
    key_src = f"{prompt}|{aspect_ratio}|{image_size}|{seed}".encode("utf-8")
    return os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".png")

def _map_image_size_to_aspect_ratio(image_size):
    """Map old FLUX image_size parameter to Google Imagen aspect ratios"""
    size_mapping = {
        "square": "1:1",
        "square_hd": "1:1", 
        "portrait_4_3": "3:4",
        "portrait_16_9": "9:16",
        "landscape_4_3": "4:3",
        "landscape_16_9": "16:9"
    }
    return size_mapping.get(image_size, "16:9")

# ---------------------------------------------------------------------------
# Prompt sanitisation helper
# ---------------------------------------------------------------------------

def _sanitize_prompt(prompt: str, strength: int, max_strength: int, model: str = "gemini-2.5-flash") -> str:
    """
    Rewrite *prompt* using a lightweight Gemini model so that it avoids
    content that would be blocked by Google Imagen safety filters while
    preserving as much descriptive detail as possible.

    The assistant should:
    1. Remove or soften any references that could be flagged as child,
       hateful, violent, sexual, dangerous or otherwise disallowed.
    2. Maintain the artistic intent, style and key visual details.
    3. Return **only** the rewritten prompt, with no additional commentary.

    If rewriting fails for any reason, the original prompt is returned.
    """
    try:
        # We instruct Gemini very explicitly to return a single-line prompt.
        system_instruction = (
            f"You are a prompt-safety rewriter. "
            f"Sanitize at strength {strength}/{max_strength} - the higher the strength, the more aggressive the sanitization should be. "
            f"Rewrite the user's image-generation prompt so that it is unquestionably "
            f"safe and inoffensive whilst preserving as much detail and intent as possible. "
            f"Avoid any references that might trigger Google Imagen safety filters such as "
            f"child exploitation, hate, violence, sexual or dangerous content. "
            f"At strength {strength}/{max_strength}, be {'extremely conservative and safe. MAKE ABSOLUTELY SURE IT IS SAFE AND DOESNT SHOW ANY HARM OR ANYTHNIG BAD IN THE SLIGHTEST' if strength > max_strength // 2 else 'moderately cautious'}. "
            f"Return ONLY the rewritten prompt text."
        )
        full_prompt = f"{system_instruction}\n\nUSER PROMPT:\n{prompt}"
        rewritten = ask_gemini(full_prompt, model=model).strip()
        # In rare cases the model may prepend explanatory text — attempt to
        # detect and drop it by taking the last non-empty line.
        if "\n" in rewritten:
            rewritten = [line.strip() for line in rewritten.splitlines() if line.strip()][-1]
        return rewritten or prompt
    except Exception as ex:
        print(f"[sanitize_prompt] Fallback to original prompt due to error: {ex}")
        return prompt

def generateImage(prompt, image_size="16:9", guidance_scale=3.5, num_inference_steps=4, seed=None):
    """
    Generate an image using Google Imagen 4 Fast model with caching
    
    Args:
        prompt (str): The prompt to generate an image from
        image_size (str): Size of the generated image. Valid options: 
                          'square_hd', 'square', 'portrait_4_3', 'portrait_16_9', 
                          'landscape_4_3', 'landscape_16_9' (default: "square")
        guidance_scale (float): CFG scale (ignored for Imagen, kept for compatibility)
        num_inference_steps (int): Number of inference steps (ignored for Imagen, kept for compatibility)
        seed (int, optional): Seed for reproducible generation
    
    Returns:
        str: Path to the generated/cached image file
    """
    user_prompt_raw = prompt.strip()
    if not user_prompt_raw:
        raise ValueError("generateImage: prompt is empty.")

    print(f"Google Imagen Generation: Processing prompt: {prompt[:80]!r}...")

    # Map image_size to aspect ratio for Imagen
    aspect_ratio = _map_image_size_to_aspect_ratio(image_size)
    imagen_size = "2K"
    
    cache_path = _cache_path(prompt, aspect_ratio, imagen_size, seed)
    
    # Return cached image if it exists
    if os.path.exists(cache_path):
        print("Using cached image")
        return cache_path

    # Generate new image using Google Imagen
    try:
        generated_path = generate_image_google_imagen(prompt, cache_path, seed, aspect_ratio)
        print(f"Image generated and saved to: {generated_path}")
        return generated_path

    except Exception as e:
        print(f"Primary image generation failed: {e}\nAttempting to sanitize prompt and retry…")
        
        # Try sanitizing with increasing strength
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            print(f"Sanitization attempt {attempt}/{max_attempts}")
            
            # Sanitize the *original* user prompt, not the templated one
            safe_user_prompt = _sanitize_prompt(user_prompt_raw, attempt, max_attempts)

            # If sanitization did not modify the prompt on first attempt, continue to stronger sanitization
            if attempt == 1 and safe_user_prompt == user_prompt_raw:
                print("First sanitization attempt made no changes, trying stronger sanitization...")
                continue

            safe_prompt_full = safe_user_prompt
            safe_cache_path = _cache_path(safe_prompt_full, aspect_ratio, imagen_size, seed)

            # Return cached sanitized version if available
            if os.path.exists(safe_cache_path):
                print(f"Using cached image (sanitized prompt, attempt {attempt})")
                return safe_cache_path

            try:
                generated_path = generate_image_google_imagen(safe_prompt_full, safe_cache_path, seed, aspect_ratio)
                print(f"Image generated with sanitized prompt (attempt {attempt}) and saved to: {generated_path}")
                return generated_path
            except Exception as e2:
                print(f"Sanitization attempt {attempt} failed: {e2}")
                if attempt == max_attempts:
                    raise RuntimeError(f"Google Imagen generation failed after {max_attempts} sanitization attempts: {e2}") from e2
                # Continue to next attempt with stronger sanitization

def generate_image_google_imagen(prompt, output_path=None, seed=None, aspect_ratio="16:9"):
    """Generate image using Google Imagen 4 Fast model"""
    
    PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    LOCATION = os.getenv("GOOGLE_LOCATION")
    
    try:
        # ----- AUTHENTICATION -----
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        # Initialize client with explicit creds (avoids bad cached tokens)
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            credentials=creds,
        )
        
        # generation_model = "imagen-4.0-fast-generate-preview-06-06"
        # generation_model = "imagen-4.0-ultra-generate-preview-06-06"
        # generation_model = "imagen-4.0-generate-preview-06-06"
        # generation_model = "imagen-3.0-generate-002"
        generation_model = "imagen-4.0-fast-generate-001"

        prompt = PROMPT_TEMPLATE.format(prompt=prompt)

        print(f"Generating image with prompt: {prompt}")
        
        # Generate image
        result = client.models.generate_images(
            model=generation_model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=1,
                negative_prompt="text, captions, subtitles, labels, letters, words",
                image_size="2K",
                safety_filter_level="BLOCK_ONLY_HIGH",
                person_generation="ALLOW_ALL",
                enhance_prompt=False,
                add_watermark=False,
            ),
        )
        print(result)
        if not result or not result.images:
            raise RuntimeError("No images generated")
            
        # Get the first image
        image_data = result.images[0]
        
        # Get the image bytes directly (they're already binary, not base64)
        image_bytes = image_data.image_bytes
        
        # Generate output path if not provided
        if output_path is None:
            import hashlib
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            output_path = f"generated_image_{prompt_hash}.png"
            
        # Save image
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"Image saved to: {output_path}")
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Google Imagen generation failed: {e}") from e
