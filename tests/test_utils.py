# tests/test_utils.py

import unittest

from narratorx.utils import split_text_into_chunks


class TestSplitTextIntoChunks(unittest.TestCase):

    def setUp(self):
        self.model_name = "gpt-4o"
        self.max_chars = 50  # Set a small token limit for testing purposes

    def test_split_text_long(self):
        """Test splitting long text that requires chunking."""
        text = "This is a test sentence. " * 100  # Create a long text
        chunks = split_text_into_chunks(
            text, max_chars=self.max_chars, model_name=self.model_name
        )
        self.assertGreater(len(chunks), 1, "Long text should be split into multiple chunks.")

    def test_split_text_empty(self):
        """Test handling of empty text."""
        text = ""
        chunks = split_text_into_chunks(
            text, max_chars=self.max_chars, model_name=self.model_name
        )
        self.assertEqual(len(chunks), 0, "Empty text should result in zero chunks.")

    def test_split_text_none(self):
        """Test handling of None as input."""
        text = None
        with self.assertRaises(TypeError):
            split_text_into_chunks(text, max_chars=self.max_chars, model_name=self.model_name)

    def test_split_text_special_characters(self):
        """Test handling of text with special characters."""
        text = "This is a sentence with emojis 😊 and symbols ©, ™, ✨."
        chunks = split_text_into_chunks(
            text, max_chars=self.max_chars, model_name=self.model_name
        )
        self.assertGreater(
            len(chunks), 0, "Text with special characters should be split into chunks."
        )
        self.assertIn("😊", chunks[0].text, "Special characters should be preserved in the chunks.")

    def test_split_text_large_max_tokens(self):
        """Test behavior when max_chars is larger than text length."""
        text = "This is a sentence." * 10
        max_chars = 1000  # Larger than the text's token count
        chunks = split_text_into_chunks(text, max_chars=max_chars, model_name=self.model_name)
        self.assertEqual(len(chunks), 1, "Text should be in one chunk when max_chars is large.")

    def test_split_text_invalid_max_tokens(self):
        """Test handling of invalid max_chars values."""
        text = "This is a sentence."
        with self.assertRaises(ValueError):
            split_text_into_chunks(text, max_chars=0, model_name=self.model_name)
        with self.assertRaises(ValueError):
            split_text_into_chunks(text, max_chars=-10, model_name=self.model_name)

    def test_split_text_non_string_input(self):
        """Test handling of non-string input."""
        text = 12345  # Non-string input
        with self.assertRaises(TypeError):
            split_text_into_chunks(text, max_chars=self.max_chars, model_name=self.model_name)


if __name__ == "__main__":
    unittest.main()
