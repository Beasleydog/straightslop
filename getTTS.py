import os, hashlib
from gradio_client import Client

CACHE_DIR = "cache/tts"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(text, voice="en-GB-RyanNeural - en-GB (Male)", rate=0, pitch=0):
    key_src = f"{voice}|{rate}|{pitch}|{text}".encode("utf-8")
    return os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".wav")

# def getTTS(text, voice="en-GB-RyanNeural - en-GB (Male)", rate=0, pitch=0):
#     text = text.strip()
#     if not text:
#         raise ValueError("getTTS: text is empty.")

#     print(f"TTS: Processing text: {text[:80]!r}...")  # trim for logs

#     cache_path = _cache_path(text, voice=voice, rate=rate, pitch=pitch)
    
#     if os.path.exists(cache_path):
#         return cache_path

#     # call gradio client
#     try:
#         client = Client("keshav6936/GENAI-TTS-Text-to-Speech")
#         result = client.predict(
#             text=text,
#             voice=voice,
#             rate=rate,
#             pitch=pitch,
#             api_name="/predict"
#         )
#     except Exception as e:
#         raise RuntimeError(f"TTS request failed: {e}") from e

#     if not result or len(result) < 2:
#         raise RuntimeError(f"Unexpected TTS result: {result}")

#     audio_file_path, status = result
    
#     if not audio_file_path:
#         raise RuntimeError(f"TTS generation failed: {status}")

#     # copy to cache
#     import shutil
#     shutil.copy2(audio_file_path, cache_path)

#     return cache_path

def getTTS(text, voice="Callum", previous_text=None):
    import requests
    import json
    import time
    from dotenv import load_dotenv
    
    load_dotenv()
    
    text = text.strip()
    if not text:
        raise ValueError("getTTS: text is empty.")

    print(f"TTS: Processing text: {text[:80]!r}...")

    # Update cache path for new parameters
    key_src = f"{voice}|{previous_text or ''}|{text}".encode("utf-8")
    cache_path = os.path.join(CACHE_DIR, hashlib.md5(key_src).hexdigest() + ".mp3")
    
    if os.path.exists(cache_path):
        print(f"TTS: Using cached audio: {cache_path}")
        return cache_path

    api_key = os.getenv('FAL_KEY')
    if not api_key:
        raise ValueError("FAL_KEY environment variable not set")

    url = "https://fal.run/fal-ai/elevenlabs/tts/turbo-v2.5"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice": voice,
    }
    
    if previous_text:
        payload["previous_text"] = previous_text

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "audio" not in result or "url" not in result["audio"]:
            raise ValueError("No audio URL returned from API")
        
        audio_url = result["audio"]["url"]
        
        # Download and cache the audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        
        with open(cache_path, "wb") as f:
            f.write(audio_response.content)
        
        return cache_path
        
    except Exception as e:
        raise RuntimeError(f"TTS request failed: {e}") from e
