# narratorx/utils.py

import os
from typing import List

import semchunk
import tiktoken


def split_text_into_chunks(
    text: str, max_tokens: int = 4000, model_name: str = "gpt-4o"
) -> List[str]:
    """Uses `semchunk` to split text into chunks that fit within a specified token limit,
    respecting sentence boundaries."""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")

    # Initialize the tokenizer using `tiktoken` for the specified model
    tokenizer = tiktoken.encoding_for_model(model_name)

    # Create a `semchunk` chunker using the tokenizer and token limit
    chunker = semchunk.chunkerify(tokenizer, max_tokens)

    # Perform chunking
    chunks = chunker(text, processes=os.cpu_count(), progress=False)  # Set progress=False for tests

    return chunks
