# tests/test_utils.py

import unittest

import tiktoken

from narratorx.utils import split_text_into_chunks


class TestSplitTextIntoChunks(unittest.TestCase):

    def setUp(self):
        self.model_name = "gpt-4o"  # Adjust to match the models supported by tiktoken
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.max_tokens = 50  # Set a small token limit for testing purposes

    def test_split_text_simple(self):
        """Test splitting simple text that should not require chunking."""
        text = "This is a simple sentence."
        chunks = split_text_into_chunks(
            text, max_tokens=self.max_tokens, model_name=self.model_name
        )
        self.assertEqual(len(chunks), 1, "Text shorter than max_tokens should result in one chunk.")
        self.assertEqual(chunks[0], text, "Chunk content should match the original text.")

    def test_split_text_long(self):
        """Test splitting long text that requires chunking."""
        text = "This is a test sentence. " * 100  # Create a long text
        chunks = split_text_into_chunks(
            text, max_tokens=self.max_tokens, model_name=self.model_name
        )
        self.assertGreater(len(chunks), 1, "Long text should be split into multiple chunks.")
        for idx, chunk in enumerate(chunks):
            num_tokens = len(self.tokenizer.encode(chunk))
            self.assertLessEqual(
                num_tokens, self.max_tokens, f"Chunk {idx} exceeds max token limit."
            )

    def test_split_text_sentence_boundaries(self):
        """Ensure chunks end at sentence boundaries when possible."""
        text = "Sentence one. Sentence two? Sentence three! Sentence four."
        max_tokens = 50  # Adjusted to accommodate full sentences

        chunks = split_text_into_chunks(text, max_tokens=max_tokens, model_name=self.model_name)
        tokenizer = tiktoken.encoding_for_model(self.model_name)

        for idx, chunk in enumerate(chunks):
            num_tokens = len(tokenizer.encode(chunk))
            if num_tokens < max_tokens:
                # If the chunk does not reach max_tokens, it should end at a sentence boundary
                self.assertTrue(
                    chunk.strip().endswith((".", "?", "!")),
                    f"Chunk {idx} does not end at a sentence boundary: {chunk}",
                )
            else:
                # If the chunk reaches max_tokens, we cannot guarantee it ends at sentence boundary
                pass  # Acceptable due to token constraints

    def test_split_text_empty(self):
        """Test handling of empty text."""
        text = ""
        chunks = split_text_into_chunks(
            text, max_tokens=self.max_tokens, model_name=self.model_name
        )
        self.assertEqual(len(chunks), 0, "Empty text should result in zero chunks.")

    def test_split_text_none(self):
        """Test handling of None as input."""
        text = None
        with self.assertRaises(TypeError):
            split_text_into_chunks(text, max_tokens=self.max_tokens, model_name=self.model_name)

    def test_split_text_special_characters(self):
        """Test handling of text with special characters."""
        text = "This is a sentence with emojis 😊 and symbols ©, ™, ✨."
        chunks = split_text_into_chunks(
            text, max_tokens=self.max_tokens, model_name=self.model_name
        )
        self.assertGreater(
            len(chunks), 0, "Text with special characters should be split into chunks."
        )
        self.assertIn("😊", chunks[0], "Special characters should be preserved in the chunks.")

    def test_split_text_large_max_tokens(self):
        """Test behavior when max_tokens is larger than text length."""
        text = "This is a sentence." * 10
        max_tokens = 1000  # Larger than the text's token count
        chunks = split_text_into_chunks(text, max_tokens=max_tokens, model_name=self.model_name)
        self.assertEqual(len(chunks), 1, "Text should be in one chunk when max_tokens is large.")

    def test_split_text_invalid_max_tokens(self):
        """Test handling of invalid max_tokens values."""
        text = "This is a sentence."
        with self.assertRaises(ValueError):
            split_text_into_chunks(text, max_tokens=0, model_name=self.model_name)
        with self.assertRaises(ValueError):
            split_text_into_chunks(text, max_tokens=-10, model_name=self.model_name)

    def test_split_text_non_string_input(self):
        """Test handling of non-string input."""
        text = 12345  # Non-string input
        with self.assertRaises(TypeError):
            split_text_into_chunks(text, max_tokens=self.max_tokens, model_name=self.model_name)

    def test_split_text_with_different_models(self):
        """Test chunking with different model names."""
        models = ["gpt-3.5-turbo", "gpt-4o"]
        text = "This is a test sentence." * 20
        for model in models:
            chunks = split_text_into_chunks(text, max_tokens=self.max_tokens, model_name=model)
            tokenizer = tiktoken.encoding_for_model(model)
            for idx, chunk in enumerate(chunks):
                num_tokens = len(tokenizer.encode(chunk))
                self.assertLessEqual(
                    num_tokens,
                    self.max_tokens,
                    f"Chunk {idx} exceeds max token limit for model {model}.",
                )


if __name__ == "__main__":
    unittest.main()
