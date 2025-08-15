import os
import time
import io
from typing import Optional, Union
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
import hashlib
import json

def ask_openai(prompt: str, model: str = "gpt-4.1") -> str:
    return "lol"

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Simple on-disk cache for Gemini responses
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "geminicache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(params: dict) -> str:
    """Return a stable SHA256 hash for *params* dict.*"""
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

# ---------------------------------------------------------------------------
# Re-use a single Client instance so we do not create new gRPC pools for every
# clip.  This keeps memory stable across long sessions.
# ---------------------------------------------------------------------------

_CLIENT: Optional[genai.Client] = None


def _get_client(api_key: str) -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

# ---------------------------------------------------------------------------
# Prompt templates (edit here to customize behavior)
# ---------------------------------------------------------------------------

REWRITE_PROMPT_TEMPLATE = (
    "Rewrite the following prompt slightly differently while maintaining as much "
    "of the original content and meaning as possible:\n\n"
    "{prompt}\n",
    "Make sure the prompt is safe for work and does not contain any explicit content.\n",
    "Return ONLY the rewritten prompt text."
)

#NOTE THAT WE SHOULD ALWAYS BE USING GEMINI-2.5-FLASH, THIS IS NOT A TYPO.
def ask_gemini(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
    prefill: Optional[str] = None,
    allow_rewrite: bool = True,
) -> str:
    """
    Ask Gemini a text-only question using the official Google Gemini package.
    
    Args:
        prompt: The text prompt to send to Gemini
        api_key: Optional API key (defaults to GEMINI_API_KEY environment variable)
        model: The Gemini model to use (default: gemini-2.5-flash)
        max_retries: Maximum number of retries if response is empty (default: 3)
        prefill: Optional **partial response** used to pre-seed the model.
        
    Returns:
        Gemini's response as a string
    """
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file with GEMINI_API_KEY=your_api_key_here")

    # ---- Cache lookup ----
    _cache_params = {"prompt": prompt, "model": model, "prefill": prefill}
    _cache_key_val = _cache_key(_cache_params)
    _cached = _load_cache(_cache_key_val)
    if _cached is not None:
        return _cached
    
    client = _get_client(api_key)
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Build `contents` list if a prefilled response is requested
            # (see Gemini API role-based prefill)
            if prefill:
                contents = [
                    {"role": "user", "parts": [{"text": prompt}]},
                    {"role": "model", "parts": [{"text": prefill}]},
                ]
            else:
                contents = prompt

            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            
            # Check if response is empty after trimming
            response_text = response.text.strip() if response.text else ""
            if response_text:
                if prefill:
                    response_text = prefill + response_text

                _save_cache(_cache_key_val, _cache_params, response_text)
                return response_text
            else:
                print(f"Attempt {attempt + 1}/{max_retries}: Received empty response, retrying...")
                print(response)
                # Print relevant response info for debugging
                print(f"Response object type: {type(response)}")
                print(f"Response text: {repr(response.text)}")
                if hasattr(response, 'candidates'):
                    print(f"Number of candidates: {len(response.candidates) if response.candidates else 0}")
                    for i, candidate in enumerate(response.candidates or []):
                        print(f"Candidate {i}: {candidate}")
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'usage_metadata'):
                    print(f"Usage metadata: {response.usage_metadata}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retrying
                    
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)
    
    # If we get here, all attempts failed (either empty responses or exceptions)
    if allow_rewrite:
        try:
            rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(prompt=prompt)
            rewritten = ask_gemini(rewrite_prompt, model="gemini-2.5-flash").strip()
            print("Prompt rewrite generated, retrying Gemini with rewritten prompt...")
            return ask_gemini(
                rewritten,
                api_key=api_key,
                model=model,
                max_retries=max_retries,
                prefill=prefill,
                allow_rewrite=False,
            )
        except Exception as rewrite_error:
            print(f"Failed to rewrite prompt with OpenAI: {rewrite_error}")
    if last_exception:
        raise Exception(f"Gemini API request failed: {last_exception}")
    print(f"Warning: All {max_retries} attempts returned empty responses")
    return ""


def wait_for_file_activation(client, file_name: str, max_wait_time: int = 300) -> bool:
    """
    Wait for a file to become ACTIVE before using it.
    
    Args:
        client: Gemini client instance
        file_name: Name of the uploaded file
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
        
    Returns:
        True if file becomes active, False if timeout or failed
    """
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            file_info = client.files.get(name=file_name)
            if file_info.state == "ACTIVE":
                return True
            elif file_info.state == "FAILED":
                print(f"File {file_name} failed to activate, deleting it...")
                try:
                    client.files.delete(name=file_name)
                    print(f"Successfully deleted failed file: {file_name}")
                except Exception as delete_error:
                    print(f"Warning: Failed to delete failed file {file_name}: {delete_error}")
                return False
            print(f"Waiting for file {file_name} to activate... Current state: {file_info.state}")
            time.sleep(5)  # Wait 5 seconds before checking again
        except Exception as e:
            print(f"Error checking file state: {e}")
            time.sleep(5)
    
    return False


def ask_gemini_with_video(video_path: str, prompt: str, api_key: Optional[str] = None, max_upload_retries: int = 3, max_content_retries: int = 3, model: str = "gemini-2.5-flash") -> str:
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file with GEMINI_API_KEY=your_api_key_here")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ---- Cache lookup ----
    _vid_cache_params = {"prompt": prompt, "model": model, "video_path": video_path}
    _vid_cache_key = _cache_key(_vid_cache_params)
    _vid_cached = _load_cache(_vid_cache_key)
    if _vid_cached is not None:
        return _vid_cached

    client = _get_client(api_key)
    uploaded_file = None
    
    def generate_content_with_retry(response_func):
        """Helper function to retry content generation if response is empty"""
        for content_attempt in range(max_content_retries):
            try:
                response = response_func()
                response_text = response.text.strip() if response.text else ""
                if response_text:
                    return response_text
                else:
                    print(f"Content attempt {content_attempt + 1}/{max_content_retries}: Received empty response, retrying...")
                    if content_attempt < max_content_retries - 1:
                        time.sleep(2)
            except Exception as e:
                if content_attempt == max_content_retries - 1:
                    raise e
                print(f"Content attempt {content_attempt + 1}/{max_content_retries} failed: {str(e)}, retrying...")
                time.sleep(2)
        
        print(f"Warning: All {max_content_retries} content generation attempts returned empty responses")
        return ""
    
    try:
        # Check file size to determine upload method
        file_size = os.path.getsize(video_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Use resumable upload with manual chunked streaming
        print("Using resumable file upload (chunked streaming)…")

        for attempt in range(max_upload_retries):
            try:
                print(f"Upload attempt {attempt + 1}/{max_upload_retries}")

                file_name = upload_file_resumable(video_path, api_key)

                # Wait for file to become ACTIVE before we can use it
                print("Waiting for file to activate…")
                if not wait_for_file_activation(client, file_name):
                    raise RuntimeError("File failed to activate after upload")

                uploaded_file = client.files.get(name=file_name)

                def file_api_response():
                    return client.models.generate_content(
                        model=model,
                        contents=[uploaded_file, prompt]
                    )

                _vid_result = generate_content_with_retry(file_api_response)
                _save_cache(_vid_cache_key, _vid_cache_params, _vid_result)
                return _vid_result

            except Exception as upload_error:
                print(f"Upload attempt {attempt + 1} failed: {upload_error}")

                # Best-effort cleanup (no need to fail if already deleted)
                try:
                    if uploaded_file:
                        client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

                uploaded_file = None

                if attempt < max_upload_retries - 1:
                    print("Retrying upload in 5 seconds…")
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to upload and activate file after {max_upload_retries} attempts: {upload_error}")
        
        # This should never be reached, but just in case
        return ""
        
    except Exception as e:
        raise Exception(f"Failed to analyze video {video_path}: {str(e)}")
    finally:
        # Clean up uploaded file if it exists
        if uploaded_file:
            try:
                print(f"Cleaning up file: {uploaded_file.name}")
                client.files.delete(name=uploaded_file.name)
                print("File cleanup successful")
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup file {uploaded_file.name}: {cleanup_error}")


def ask_gemini_with_images(image_paths: list[str], prompt: str, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", max_retries: int = 3) -> str:
    """
    Ask Gemini by supplying multiple local image files alongside a text *prompt*.

    Args:
        image_paths: List of local file paths to images that will be sent to Gemini.
        prompt: The text prompt to accompany the images.
        api_key: Optional Gemini API key (falls back to GEMINI_API_KEY env var).
        model: Gemini model to use (default: gemini-2.5-flash).
        max_retries: Maximum number of retries if Gemini returns an empty response.

    Returns:
        The textual response from Gemini.
    """
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file with GEMINI_API_KEY=your_api_key_here")

    # ---- Cache lookup ----
    _img_paths_sorted = sorted([str(p) for p in image_paths])
    _img_cache_params = {"prompt": prompt, "model": model, "image_paths": _img_paths_sorted}
    _img_cache_key = _cache_key(_img_cache_params)
    _img_cached = _load_cache(_img_cache_key)
    if _img_cached is not None:
        return _img_cached

    client = _get_client(api_key)

    # Build the list of Part objects: one image Part, then the prompt Part
    parts: list[types.PartUnion] = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: image not found: {path}. Skipping.")
            continue

        mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        try:
            with open(path, "rb") as f:
                parts.append(
                    types.Part.from_bytes(
                        data=f.read(),
                        mime_type=mime_type,
                    )
                )
        except Exception as e:
            print(f"Failed to load image {path}: {e}")

    # Finally add the text prompt
    parts.append(types.Part.from_text(text=prompt))

    # The SDK will auto‑wrap this list into a single UserContent
    contents = parts

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            response_text = response.text.strip() if response.text else ""
            if response_text:
                _save_cache(_img_cache_key, _img_cache_params, response_text)
                return response_text
            else:
                print(f"Attempt {attempt + 1}/{max_retries}: Received empty response, retrying…")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Gemini API request failed: {str(e)}")
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}, retrying…")

        time.sleep(2)

    print(f"Warning: All {max_retries} attempts returned empty responses")
    return ""


def upload_file_resumable(file_path: str, api_key: str, chunk_size: int = 5*8 * 1024 * 1024) -> str:
    """Upload *file_path* to Gemini using a resumable, chunked upload.

    This avoids loading the full video into RAM by streaming the file in
    ``chunk_size`` byte pieces. Returns the server-generated file *name*
    (e.g. ``files/abcd1234``) which can be passed to other File API calls.
    """

    SESSION_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"

    total_size = os.path.getsize(file_path)

    print(f"[UPLOAD] Starting resumable upload → {os.path.basename(file_path)}  "
          f"{total_size / (1024 * 1024):.2f} MB in chunks of {chunk_size / (1024 * 1024):.2f} MB")

    # 1. Initiate a resumable upload session
    init_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Type": "video/mp4",
        "X-Goog-Upload-Header-Content-Length": str(total_size),
        "Content-Type": "application/json",
    }

    init_payload = {
        "file": {
            "display_name": os.path.basename(file_path)
        }
    }

    init_resp = requests.post(
        f"{SESSION_URL}?uploadType=resumable&key={api_key}",
        headers=init_headers,
        json=init_payload,
        timeout=120,
    )

    if init_resp.status_code not in {200, 201}:
        raise RuntimeError(f"Could not initiate upload session: {init_resp.text}")

    upload_url = init_resp.headers.get("X-Goog-Upload-URL") or init_resp.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Upload URL not returned by Gemini API")

    print(f"[UPLOAD] Resumable session URL received")

    offset = 0

    with open(file_path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break

            headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(len(buf)),
                "X-Goog-Upload-Offset": str(offset),
                "X-Goog-Upload-Command": "upload, finalize" if offset + len(buf) == total_size else "upload",
                "X-Goog-Upload-Protocol": "resumable",
            }

            put_resp = requests.put(upload_url, headers=headers, data=io.BytesIO(buf), timeout=300)

            if put_resp.status_code not in {200, 201, 308}:
                raise RuntimeError(
                    f"Chunk upload failed at offset {offset}: {put_resp.status_code} – {put_resp.text}"
                )

            offset += len(buf)

            status_type = "FINAL" if headers["X-Goog-Upload-Command"].startswith("upload, finalize") else "CHUNK"
            print(f"[UPLOAD] {status_type} OK – bytes {offset-len(buf)}-{offset-1}  "
                  f"({offset / total_size * 100:.1f}% done)  → HTTP {put_resp.status_code}")

    print("[UPLOAD] All chunks uploaded – waiting for server to finalise file resource …")

    # --- Parse the final response ----------------------------
    # The Gemini Files API may respond with either of these shapes:
    # 1. {"file": {"name": "files/abc-123", ...}}
    # 2. {"name": "files/abc-123", ...}
    # Accept both to stay forward-compatible.

    try:
        file_info_raw = put_resp.json()
    except ValueError:
        # Service returned non-JSON – surface the raw text to help debugging.
        raise RuntimeError(
            f"Upload finished but response was not valid JSON: {put_resp.text}"
        )

    file_resource = file_info_raw["file"] if "file" in file_info_raw else file_info_raw

    name = file_resource.get("name")
    if not name:
        raise RuntimeError(
            f"Upload finished but no file name in response: {file_info_raw}"
        )

    print(f"[UPLOAD] Upload complete – file name: {name}")
    return name
