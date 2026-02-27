"""Answer extraction and prompt formatting utilities."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def format_options(options: Dict[str, str]) -> str:
    """Format MCQ options as a readable string."""
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def build_multimodal_prompt(
    text: str,
    image_details: Optional[List[Dict]] = None,
    table_details: Optional[List[Dict]] = None,
    *,
    use_url: bool = False,
    image_root: Optional[Path] = None,
) -> Tuple[str, List[Union[Path, str]]]:
    """
    Build prompt text and collect image sources.

    Returns (prompt_text, image_sources) where image_sources are
    Path objects (for local mode) or URL strings (for url mode).
    """
    parts = [text]
    images: List[Union[Path, str]] = []

    if image_details:
        captions: List[str] = []
        for img in image_details:
            caption = img.get("caption_prefix", "")
            if not caption:
                continue

            if use_url:
                url = img.get("url")
                if url:
                    captions.append(caption)
                    images.append(url)
            else:
                rel_path = img.get("file", "")
                if rel_path and image_root:
                    abs_path = (image_root / Path(str(rel_path))).resolve()
                    if abs_path.is_file():
                        captions.append(caption)
                        images.append(abs_path)

        if captions:
            parts.append("* Figures:")
            parts.extend(captions)

    if table_details:
        tables: List[str] = []
        for tbl in table_details:
            prefix = tbl.get("caption_prefix", "")
            caption = tbl.get("caption", "")
            content = tbl.get("content", "")
            if content:
                tables.append(f"{prefix} {caption}\n{content}".strip())
        if tables:
            parts.append("* Tables:")
            parts.extend(tables)

    return "\n".join(parts), images


ANSWER_FORMAT = (
    "Please provide the letter of the correct option, "
    "formatted as \\boxed{LETTER} (e.g., \\boxed{A})."
)


def build_scenario_prompt(
    scenario: str,
    first_mcq: Dict,
    scenario_images: Optional[List[Dict]] = None,
    scenario_tables: Optional[List[Dict]] = None,
    *,
    use_url: bool = False,
    image_root: Optional[Path] = None,
) -> Tuple[str, List[Union[Path, str]]]:
    """Build the first-turn prompt: scenario + first MCQ."""
    scenario_text, scenario_imgs = build_multimodal_prompt(
        f"* Scenario: {scenario}",
        scenario_images,
        scenario_tables,
        use_url=use_url,
        image_root=image_root,
    )

    q_text = first_mcq.get("question", "")
    q_opts = format_options(first_mcq.get("options", {}))
    mcq_text, mcq_imgs = build_multimodal_prompt(
        f"* Question: {q_text}\n* Options:\n{q_opts}\n\n{ANSWER_FORMAT}",
        first_mcq.get("image_details"),
        first_mcq.get("table_details"),
        use_url=use_url,
        image_root=image_root,
    )

    prompt = f"{scenario_text}\n\n{mcq_text}"
    all_images = scenario_imgs + mcq_imgs
    return prompt, all_images


def build_followup_prompt(
    mcq: Dict,
    *,
    use_url: bool = False,
    image_root: Optional[Path] = None,
) -> Tuple[str, List[Union[Path, str]]]:
    """Build a follow-up turn prompt for MCQ index >= 1."""
    q_text = mcq.get("question", "")
    q_opts = format_options(mcq.get("options", {}))
    return build_multimodal_prompt(
        f"* Question: {q_text}\n* Options:\n{q_opts}\n\n{ANSWER_FORMAT}",
        mcq.get("image_details"),
        mcq.get("table_details"),
        use_url=use_url,
        image_root=image_root,
    )


def extract_answer(response: str) -> Optional[str]:
    """Extract an answer letter (A-J) from a model response."""
    if not response:
        return None

    # 1. \boxed{X}
    m = re.search(r"\\boxed{\s*([A-J])\s*}", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2. Explicit "answer is X" patterns
    m = re.search(
        r"(?:answer|option|choice|selected option)\s*[:\-is\s]*\s*([A-Ja-j])\b",
        response,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # 3. Last line that is just a single letter
    for line in reversed(response.strip().split("\n")):
        trimmed = line.strip()
        if re.match(r"^[A-Ja-j][\s.)]*$", trimmed, re.IGNORECASE):
            return trimmed[0].upper()

    # 4. Last standalone capital letter A-J
    matches = re.findall(r"\b([A-J])\b", response)
    if matches:
        return matches[-1]

    # 5. Single character response
    if len(response.strip()) == 1 and response.strip().upper() in "ABCDEFGHIJ":
        return response.strip().upper()

    return None
