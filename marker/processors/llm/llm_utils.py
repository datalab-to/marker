import re

NO_CORRECTION_PHRASES = (
    "no_corrections",
    "no corrections",
    "no corrections needed",
    "no correction needed",
    "no correction required",
    "no corrections required",
    "no errors detected",
    "no errors found",
    "no changes needed",
    "no change needed",
    "looks good",
)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def string_indicates_no_corrections(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in NO_CORRECTION_PHRASES)


def get_analysis_prompt_parts(style: str | None) -> tuple[str, str]:
    """
    Get the instruction and schema parts for the analysis prompt based on the style.

    Parameters
    ----------
    style : str | None
        The style of analysis prompt to generate.
        Can be "summary", "deep", "auto" or None for default.
        Defaults to "auto" if None.

    Returns
    -------
    tuple[str, str]
        A tuple containing the instruction and schema parts for the analysis prompt.
    """
    normalized = (style or "auto").strip().lower()
    if normalized == "summary":
        return (
            "Write a short analysis.",
            "- `analysis`: short string",
        )
    if normalized == "deep":
        return (
            "Write a deep and detailed analysis. Include step-by-step reasoning and thorough explanations.",
            "- `analysis`: detailed string with step-by-step reasoning",
        )
    return (
        "Write an analysis. Keep it concise unless a longer reasoning process is necessary. "
        "If the task is difficult, you must enter a detailed reasoning process and think step by step.",
        "- `analysis`: string. Keep it concise unless a longer reasoning process is necessary. "
        "If the task is difficult, you must enter a detailed reasoning process and think step by step.",
    )


def inject_analysis_prompt(prompt: str, style: str | None) -> str:
    instruction, schema = get_analysis_prompt_parts(style)
    return (
        prompt.replace("{{analysis_instruction}}", instruction).replace(
            "{{analysis_schema}}", schema
        )
    )
