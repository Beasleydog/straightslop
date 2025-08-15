import os
import json
import hashlib
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from faster_whisper import WhisperModel


_MODEL: WhisperModel | None = None
_WHISPER_MODEL_ID = "base"

# Reuse the same cache directory as TTS so artifacts stay co-located
CACHE_DIR = "cache/tts"
os.makedirs(CACHE_DIR, exist_ok=True)


def _model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(_WHISPER_MODEL_ID)
    return _MODEL


def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _extract_words(audio_path: str) -> List[Tuple[str, float, float]]:
    # 1) Try sidecar JSON next to the audio (e.g., somefile.whisper.json)
    base_no_ext, _ = os.path.splitext(audio_path)
    sidecar_path = f"{base_no_ext}.whisper.json"
    if os.path.exists(sidecar_path):
        print(f"Timestamps: Using cached sidecar: {sidecar_path}")
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            cached: List[Tuple[str, float, float]] = []
            for item in arr:
                if isinstance(item, list) and len(item) == 3:
                    token = _norm_token(str(item[0]))
                    start = float(item[1])
                    end = float(item[2])
                    if token:
                        cached.append((token, start, end))
            if cached:
                return cached
        except Exception:
            # Ignore corrupt cache and fall through to recompute
            pass

    # 2) Try content-addressed cache in cache/tts based on audio bytes + model id
    def _hashed_cache_path() -> str:
        md5 = hashlib.md5()
        md5.update(_WHISPER_MODEL_ID.encode("utf-8"))
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        return os.path.join(CACHE_DIR, md5.hexdigest() + ".whisper.json")

    hashed_path = _hashed_cache_path()
    if os.path.exists(hashed_path):
        try:
            with open(hashed_path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            cached: List[Tuple[str, float, float]] = []
            for item in arr:
                if isinstance(item, list) and len(item) == 3:
                    token = _norm_token(str(item[0]))
                    start = float(item[1])
                    end = float(item[2])
                    if token:
                        cached.append((token, start, end))
            if cached:
                return cached
        except Exception:
            # Ignore corrupt cache and fall through to recompute
            pass

    # 3) Cache miss: transcribe and persist as [[token, start, end], ...]
    segments, _ = _model().transcribe(audio_path, word_timestamps=True)
    words: List[Tuple[str, float, float]] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            text = (
                getattr(w, "word", None)
                or getattr(w, "text", None)
                or (w.get("word") if isinstance(w, dict) else None)
                or ""
            )
            start = getattr(w, "start", None)
            if start is None and isinstance(w, dict):
                start = w.get("start")
            end = getattr(w, "end", None)
            if end is None and isinstance(w, dict):
                end = w.get("end")
            if start is None or end is None:
                continue
            tok = _norm_token(str(text))
            if tok:
                words.append((tok, float(start), float(end)))

    # Persist to hashed cache
    try:
        serializable = [[t, s, e] for (t, s, e) in words]
        with open(hashed_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception:
        # Non-fatal if we cannot write cache
        pass

    # Also persist a sidecar next to the audio if possible
    try:
        serializable = [[t, s, e] for (t, s, e) in words]
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception:
        pass

    return words


def getTimestamps(audio_path: str, text: str) -> Dict[str, float]:
    """
    Transcribe audio and find the time span of the given text (flexible match).

    Returns a dict: {"start": float, "end": float}
    """
    target_tokens = [_norm_token(t) for t in re.findall(r"[A-Za-z0-9']+", text)]
    target_tokens = [t for t in target_tokens if t]
    if not target_tokens:
        raise ValueError("Target text yields no tokens after normalization.")

    words = _extract_words(audio_path)
    if not words:
        raise RuntimeError("No words extracted from audio.")

    vocab = [t for t, _, _ in words]
    m = len(target_tokens)

    # 1) Exact match on normalized tokens
    for i in range(0, len(vocab) - m + 1):
        if vocab[i : i + m] == target_tokens:
            return {"start": words[i][1], "end": words[i + m - 1][2]}

    # 2) Approximate match: try windows around length m and pick best ratio
    def join_tokens(tokens: List[str]) -> str:
        return " ".join(tokens)

    target_join = join_tokens(target_tokens)
    best_ratio = -1.0
    best_span = (0, 0)

    min_len = max(1, m - 2)
    max_len = min(len(vocab), m + 2)
    for L in range(min_len, max_len + 1):
        for i in range(0, len(vocab) - L + 1):
            cand = join_tokens(vocab[i : i + L])
            ratio = SequenceMatcher(None, cand, target_join).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (i, i + L - 1)

    i0, i1 = best_span
    return {"start": words[i0][1], "end": words[i1][2]}


