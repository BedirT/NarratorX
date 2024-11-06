# narratorx/llm.py

from litellm import completion

from narratorx.utils import split_text_into_chunks


def llm_process_text(
    text: str,
    language: str,
    model_name: str = "gpt-4o",
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

    # Process each chunk individually
    for chunk in chunks:
        user_prompt = user_prompt_template.format(content=chunk)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "<fixed_text>"},
        ]
        response_text = completion(
            model=model_name,
            messages=messages,
            stop=["</fixed_text>"],
            max_tokens=max_tokens,
        )["choices"][0]["message"]["content"]

        # Get the text after the fixed_text tag
        if "<fixed_text>" in response_text:
            new_response_text = response_text.split("<fixed_text>")[1].strip()
            if not new_response_text:
                new_response_text = response_text.split("<fixed_text>")[0].strip()
        else:
            new_response_text = response_text

        fixed_chunks.append(new_response_text)

    return "\n\n".join(fixed_chunks)
