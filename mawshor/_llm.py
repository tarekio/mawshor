import requests

_SYSTEM_PROMPT = """
You are an expert Arabic copyeditor specialized in cleaning OCR output.
Output ONLY the corrected text. No preamble. No commentary.

Tasks:
1. Fix low confidence words by inferring the most likely intended word based on context and common OCR errors.
2. Split merged words and join fragments into valid Arabic words.
3. Clean up irregular spacing and fix punctuation placement.

Constraints:
- Do not add new factual information or change the meaning.
- Fix only low confidence words, leaving high confidence words unchanged (except for splitting/merging).
- Direct output only.
"""


def correct_text(
    text: str,
    low_confidence_words: list[str] | None = None,
    endpoint: str = "http://localhost:11434/v1",
    model: str = "qwen3.5:4b",
    api_key: str = "ollama",
) -> str:
    lc_list = ", ".join(low_confidence_words) if low_confidence_words else "None"
    content = (
        f"/no_think\n"
        f"Please correct the following OCR text:\n\n{text}\n\n"
        f"Low confidence words: {lc_list}"
    )

    response = requests.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "stream": False,
            "think": False,
            "temperature": 0.3,
            "max_tokens": 1500,
            "frequency_penalty": 1.3,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
