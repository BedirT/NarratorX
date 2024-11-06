# narratorx/utils.py

from typing import List

from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import NarrativeText


def split_text_into_chunks(
    text: str, max_chars: int = 8000, model_name: str = "gpt-4o"
) -> List[str]:
    """Uses `semchunk` to split text into chunks that fit within a specified token limit,
    respecting sentence boundaries."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer.")

    # Create a NarrativeText element with the input text
    element = NarrativeText(text)

    # Chunk the text using unstructured
    chunks = chunk_elements(
        elements=[element],
        max_characters=max_chars,
        overlap=0,
    )

    return chunks
