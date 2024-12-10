# narratorx/llm.py

from litellm import completion
import litellm

from narratorx.utils import split_text_into_chunks
from pydantic import BaseModel, Field

import json


class FixedTextResponse(BaseModel):
    thinking: str
    fixed_text: str


def llm_process_text(
    text: str,
    language: str,
    model_name: str = "gpt-4o-mini",
    max_chars: int = 8000,
    max_tokens: int = 4000,
) -> str:
    """Processes the text by chunking and using llms to fix the text."""
    chunks = split_text_into_chunks(text, max_chars=max_chars, model_name=model_name)
    fixed_chunks = []

    with open("src/narratorx/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
        user_prompt_template = f.read()
    with open("src/narratorx/prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    system_prompt = system_prompt_template.format(language=language)

    litellm.enable_json_schema_validation = True

    # Process each chunk individually
    for chunk in chunks:
        user_prompt = user_prompt_template.format(content=chunk, do_pages_have_page_numbers=True, do_pages_have_headers_or_footers=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_text = completion(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            response_format=FixedTextResponse,
        )
        json_res = response_text.choices[0].message.content
        print("Response text:", json_res)
        parsed_res = json.loads(json_res)
        fixed_chunks.append(parsed_res["fixed_text"])

    return "\n\n".join(fixed_chunks)
