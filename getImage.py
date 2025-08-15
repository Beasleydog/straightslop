import os
import json
import hashlib
import time
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Simple on-disk cache for image generation responses
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "imagecache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(params: dict) -> str:
    """Return a stable SHA256 hash for *params* dict."""
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_cache(key: str) -> Optional[str]:
    """Load cached response text if available, otherwise *None*."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("response")
    except Exception as e:
        print(f"Warning: failed to read cache {path}: {e}")
        return None


def _save_cache(key: str, params: dict, response: str) -> None:
    """Persist *response* (and *params* for debugging) to the cache."""
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"params": params, "response": response, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to write cache {path}: {e}")


def getImage(
    prompt: str,
    api_key: Optional[str] = None,
    style: str = "digital_illustration/antiquarian",
    image_size: str | dict = "1024x576",
) -> str:
    """
    Generate an image using the Recraft V3 API.
    
    Args:
        prompt: The text prompt to generate an image from
        api_key: Optional FAL API key (defaults to FAL_KEY environment variable)
        style: The style of the generated image (default: "realistic_image")
        image_size: The size of the generated image (default: "square_hd")
        
    Returns:
        URL of the generated image
    """
    api_key = api_key or os.getenv('FAL_KEY')
    if not api_key:
        raise ValueError("FAL_KEY environment variable not set. Please create a .env file with FAL_KEY=your_api_key_here")

    # ---- Cache lookup ----
    _cache_params = {"prompt": prompt, "style": style, "image_size": image_size}
    _cache_key_val = _cache_key(_cache_params)
    _cached = _load_cache(_cache_key_val)
    if _cached is not None:
        return _cached

    # Make API request (use correct model route)
    url = "https://fal.run/fal-ai/recraft-v3"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }

    # Normalize image_size: accept enum string, WxH string, or dict {width, height}
    normalized_size: str | dict
    if isinstance(image_size, dict):
        normalized_size = {
            "width": int(image_size.get("width", 0)),
            "height": int(image_size.get("height", 0)),
        }
    elif isinstance(image_size, str):
        parts = image_size.lower().replace("×", "x").split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            normalized_size = {"width": int(parts[0]), "height": int(parts[1])}
        else:
            normalized_size = image_size  # assume enum
    else:
        normalized_size = "square_hd"

    payload = {
        "prompt": prompt,
        "style": style,
        "image_size": normalized_size,
        # Keep defaults explicit for clarity and predictable behavior
        "enable_safety_checker": False,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except requests.HTTPError as http_err:
            # Bubble up detailed server message to help debug schema issues
            detail = None
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise Exception(f"HTTP {response.status_code}: {detail}") from http_err
        
        result = response.json()
        if "images" not in result or not result["images"]:
            raise ValueError("No images returned from API")
        
        image_url = result["images"][0]["url"]
        
        # Save to cache
        _save_cache(_cache_key_val, _cache_params, image_url)
        
        return image_url
        
    except Exception as e:
        raise Exception(f"Failed to generate image: {str(e)}")
