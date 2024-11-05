# narratorx/llm.py

from typing import List

from litellm import completion

from narratorx.utils import split_text_into_chunks


def llm_process_text(text: str, language: str, model_name: str = "gpt-4o") -> List[str]:
    """Processes the text by chunking and using llms to fix the text."""
    # Chunk text using `semchunk`
    chunks = split_text_into_chunks(text, max_tokens=4000, model_name=model_name)
    fixed_chunks = []

    with open("src/narratorx/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    with open("src/narratorx/prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    system_prompt = system_prompt_template.format(language=language)

    # Process each chunk individually
    for chunk in chunks:
        user_prompt = user_prompt_template.format(content=chunk)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "<fixed_text>"},
        ]
        response_text = completion(model=model_name, messages=messages, stop=["</fixed_text>"])[
            "choices"
        ][0]["message"]["content"]

        fixed_chunks.append(response_text)

    return "\n\n".join(fixed_chunks)
