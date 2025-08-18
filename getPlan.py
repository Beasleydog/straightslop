# Start of Selection
import json
import os
import re
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini import ask_gemini


MAX_WORKERS = int(os.getenv("GEMINI_MAX_WORKERS", "6"))

"""
Prompt templates (edit these to change model instructions)
"""

PLAN_SECTIONS_PROMPT_TEMPLATE = """Plan out a history video narrative youtube video script for this
{topic_title}
Output bullet point sections, each with a description of what content is included (just description, no exact vo or media plan)
NEVER INCLUDE AN INTRO SECTION.
END WITH A SHORT OUTRO, WRAPPING UP THE VIDEO AND ENCOURAGING THE VIEWER TO LIKE/SUBSCRIBE/COMMENT THEIR THOUGHTS.
Minimal output, no yap, just list.
Output JSON, like this:
[
"description one",
"description two",
etc
]
Return only valid JSON, no extra text."""

WRITE_SECTION_SCRIPT_PROMPT_TEMPLATE = """
Write a full script for the given section of the YouTube video. Consider the full attached plan for context on the video.

Don't output plans for the media; only write one cohesive paragraph for the script.
Focus on making the script smooth and entertaining — it must be THE HIGHEST OF QUALITY.

You must also include:
- a very short title for the script
- a detailed description of a background image that will show when the title appears

Background image guidance:
- The image should be designed as a subtle backdrop, not a focal illustration
- It should show a scene/setting, not specific objects or attention-grabbing elements
- Avoid text and avoid specific items; aim for mood, tone, and atmosphere

Output just JSON in exactly this format:
{{
  "script": "string",
  "title": "string",
  "backgroundImage": "string"
}}

Section: {section}
Full Context: {context_json}
Return only valid JSON, no extra text."""

SEGMENT_SCRIPT_TO_DESCRIPTION_VO_PROMPT_TEMPLATE = """You've been given a portion of a video.
Consider the portion along with the context of the over all video.
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

VO Guidelines:
- The vo must be EXACTLY the same as the script portion
- The vo can range from a few words to one sentence
    - It should never feel like the image is dragging on, but it also shouldn't feel like the image is too short
    - The visuals should really align with the VO, don't be afraid to split one sentence into multiple images to really show the viewer the content

You've been given the:
{section_label}
Script portion:
{script_portion}
Script context:
{context_json}
Return only valid JSON, with keys exactly 'description' and 'vo'."""



def _extract_json_array(raw_text: str) -> Any:
    """Best-effort extraction of a JSON array from a model response.

    - Strips markdown code fences like ```json ... ```
    - Attempts to parse the first [...] block
    - Falls back to parsing the cleaned full string
    """
    print(f"[DEBUG] _extract_json_array received raw_text: {raw_text[:100]}...")
    if not raw_text:
        raise ValueError("Empty response from model; cannot parse JSON.")
    cleaned = _strip_markdown_fences(raw_text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            result = json.loads(candidate)
            print(f"[DEBUG] _extract_json_array extracted JSON via candidate: {type(result)}")
            return result
        except Exception as e:
            print(f"[DEBUG] _extract_json_array candidate parse failed: {e}")
            # Try removing trailing commas
            candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                result = json.loads(candidate2)
                print(f"[DEBUG] _extract_json_array extracted after cleanup: {type(result)}")
                return result
            except Exception as e2:
                print(f"[DEBUG] _extract_json_array cleanup parse failed: {e2}")
                pass
    # Fallback: try full cleaned string
    cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned)
    result = json.loads(cleaned2)
    print(f"[DEBUG] _extract_json_array fallback parse: {type(result)}")
    return result


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response.

    - Strips markdown code fences
    - Attempts to parse the first {...} block
    - Falls back to parsing the cleaned full string
    """
    print(f"[DEBUG] _extract_json_object received raw_text: {raw_text[:100]}...")
    if not raw_text:
        raise ValueError("Empty response from model; cannot parse JSON object.")
    cleaned = _strip_markdown_fences(raw_text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            result = json.loads(candidate)
            if not isinstance(result, dict):
                raise ValueError("Parsed JSON is not an object.")
            print("[DEBUG] _extract_json_object extracted JSON via candidate: object")
            return result
        except Exception as e:
            print(f"[DEBUG] _extract_json_object candidate parse failed: {e}")
            # Try removing trailing commas
            candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                result = json.loads(candidate2)
                if not isinstance(result, dict):
                    raise ValueError("Parsed JSON is not an object.")
                print("[DEBUG] _extract_json_object extracted after cleanup: object")
                return result
            except Exception as e2:
                print(f"[DEBUG] _extract_json_object cleanup parse failed: {e2}")
                pass
    # Fallback: try full cleaned string
    cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned)
    result = json.loads(cleaned2)
    if not isinstance(result, dict):
        raise ValueError("Parsed JSON is not an object (fallback).")
    print("[DEBUG] _extract_json_object fallback parse: object")
    return result


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (``` or ```json) and return trimmed content."""
    t = text.strip()
    # Remove opening fence line like ```json\n or ```\n
    if t.startswith("```"):
        newline_idx = t.find("\n")
        if newline_idx != -1:
            t = t[newline_idx + 1 :]
        else:
            t = t.lstrip("`")
    # Remove trailing closing fence
    if t.endswith("```"):
        t = t[:-3]
    # Remove stray fences using regex
    t = re.sub(r"^```[a-zA-Z]*\n", "", t)
    t = re.sub(r"\n```$", "", t)
    return t.strip()


def plan_sections(topic_title: str, model: str = "gemini-2.5-flash") -> List[str]:
    print(f"[DEBUG] plan_sections called with topic_title: {topic_title}")
    prompt = PLAN_SECTIONS_PROMPT_TEMPLATE.format(topic_title=topic_title)
    raw = ask_gemini(prompt, model=model)
    print(f"[DEBUG] plan_sections received raw response: {raw[:100]}...")
    data = _extract_json_array(raw)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Plan output was not a JSON array of strings.")
    print(f"[DEBUG] plan_sections returning {len(data)} sections")
    return data


def write_section_script(section: str, full_context: List[str], model: str = "gemini-2.5-flash") -> Dict[str, str]:
    print(f"[DEBUG] write_section_script called for section: {section}")
    context_json = json.dumps(full_context, ensure_ascii=False)
    prompt = WRITE_SECTION_SCRIPT_PROMPT_TEMPLATE.format(
        section=section,
        context_json=context_json,
    )
    raw = ask_gemini(prompt, model=model)
    print(f"[DEBUG] write_section_script received raw response: {raw[:100]}...")
    obj = _extract_json_object(raw)
    # Validate expected keys
    for key in ("script", "title", "backgroundImage"):
        if key not in obj:
            raise ValueError(f"Section script JSON missing key: {key}")
        if not isinstance(obj[key], str):
            raise ValueError(f"Section script JSON field '{key}' must be a string")
    print(
        f"[DEBUG] write_section_script returning object with script length: {len(obj['script'])}, title length: {len(obj['title'])}"
    )
    return obj  # type: ignore[return-value]


def segment_script_to_description_vo(
    section_label: str,
    script_portion: str,
    script_context: List[str],
    model: str = "gemini-2.5-flash",
) -> List[Dict[str, str]]:
    print(f"[DEBUG] segment_script_to_description_vo called for section: {section_label}")
    context_json = json.dumps(script_context, ensure_ascii=False)
    prompt = SEGMENT_SCRIPT_TO_DESCRIPTION_VO_PROMPT_TEMPLATE.format(
        section_label=section_label,
        script_portion=script_portion,
        context_json=context_json,
    )
    raw = ask_gemini(prompt, model=model)
    print(f"[DEBUG] segment_script_to_description_vo received raw response: {raw[:100]}...")
    data = _extract_json_array(raw)
    if not isinstance(data, list):
        raise ValueError("Segmentation output was not a JSON array.")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each segmentation item must be an object.")
        if "description" not in item or "vo" not in item:
            raise ValueError("Segmentation items must contain 'description' and 'vo'.")
        if not isinstance(item["description"], str) or not isinstance(item["vo"], str):
            raise ValueError("'description' and 'vo' must be strings.")
    print(f"[DEBUG] segment_script_to_description_vo returning {len(data)} segments")
    return data  # type: ignore[return-value]


def getPlan(
    topic_title: str,
    model: str = "gemini-2.5-flash",
) -> tuple[List[Dict[str, str]], List[List[Dict[str, str]]]]:
    """End-to-end: plan → write each section → segment → return section objects and plan arrays.

    Returns:
        tuple: (sections, plan) where:
            - sections: List of objects for each section with keys 'script', 'title', 'backgroundImage'
            - plan: 2D array where plan[i] contains segments for sections[i]['script']
    """
    print(f"[DEBUG] getPlan called with topic_title: {topic_title}")
    sections = plan_sections(topic_title, model=model)

    def process_one(index: int, section_text: str) -> tuple[int, Dict[str, str], List[Dict[str, str]]]:
        print(f"[DEBUG] [worker] start section {index+1}/{len(sections)}")
        section_obj = write_section_script(section_text, sections, model=model)
        segments = segment_script_to_description_vo(
            section_text,
            section_obj["script"],
            sections,
            model=model,
        )
        print(f"[DEBUG] [worker] done section {index+1}: {len(segments)} segments")
        return index, section_obj, segments

    results: Dict[int, tuple[Dict[str, str], List[Dict[str, str]]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_one, i, sec): i for i, sec in enumerate(sections)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            index, section_obj, segments = future.result()
            results[index] = (section_obj, segments)

    # Build sections and plan arrays in original order
    vo_sections: List[Dict[str, str]] = []
    plan: List[List[Dict[str, str]]] = []
    for i in range(len(sections)):
        section_obj, segs = results.get(i, ({"script": "", "title": "", "backgroundImage": ""}, []))
        vo_sections.append(section_obj)
        plan.append(segs)

    print(f"[DEBUG] getPlan returning {len(vo_sections)} vo_sections and {len(plan)} plan arrays")
    return vo_sections, plan
