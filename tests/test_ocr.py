# tests/test_ocr.py

import os
import unittest

from narratorx.ocr import get_valid_languages, process_pdf


class TestOCR(unittest.TestCase):
    def setUp(self):
        # Paths to sample PDF files for Turkish and English tests
        self.sample_pdf_tr = "tests/sample_tr.pdf"
        self.sample_pdf_en = "tests/sample_en.pdf"
        self.language_tr = "tr"
        self.language_en = "en"

        # Ensure both sample PDF files exist
        if not os.path.exists(self.sample_pdf_tr):
            raise FileNotFoundError(f"Turkish sample PDF not found at {self.sample_pdf_tr}")
        if not os.path.exists(self.sample_pdf_en):
            raise FileNotFoundError(f"English sample PDF not found at {self.sample_pdf_en}")

    def test_process_pdf_turkish(self):
        """Test OCR processing with Turkish language option and Turkish PDF."""
        result = process_pdf(self.sample_pdf_tr, self.language_tr)

        # Assert that the result is a non-empty string
        self.assertIsInstance(result, str, "The output should be a string.")
        self.assertTrue(len(result.strip()) > 0, "The output string should not be empty.")

        # Assert that the result contains expected Turkish text
        to_look_for = [
            "hayat tarzları da benzer",  # Example Turkish phrase
            "Jared Diamond",
        ]
        for text in to_look_for:
            self.assertIn(text, result, f"The output should contain '{text}'.")

    def test_process_pdf_english(self):
        """Test OCR processing with English language option and English PDF."""
        result = process_pdf(self.sample_pdf_en, self.language_en)

        # Assert that the result is a non-empty string
        self.assertIsInstance(result, str, "The output should be a string.")
        self.assertTrue(len(result.strip()) > 0, "The output string should not be empty.")

        # Assert that the result contains expected English text
        to_look_for = [
            "The Little Prince",
            "I pondered deeply",
            "Oh! That is funny!",
        ]
        for text in to_look_for:
            self.assertIn(text, result, f"The output should contain '{text}'.")

    def test_process_pdf_invalid_path(self):
        """Test OCR with an invalid PDF path, expecting an exception."""
        with self.assertRaises(Exception):
            process_pdf("invalid/path/to/sample.pdf", self.language_tr)

    def test_process_pdf_invalid_language(self):
        """Test OCR with an unsupported language code, expecting an exception."""
        with self.assertRaises(Exception):
            process_pdf(self.sample_pdf_tr, "invalid_lang_code")

    # def test_optional_manual_investigation(self):
    #     """Optional: Manually inspect the output to verify OCR results."""
    #     result = process_pdf(self.sample_pdf_tr, self.language_tr)

    #     # Save the output to a file for manual inspection
    #     output_file = "tests/narratorx_ocr_test_output.txt"
    #     with open(output_file, "w", encoding="utf-8") as f:
    #         f.write(result)

    #     print(f"Output saved to '{output_file}' for manual inspection.")
    #     print("Please inspect the output to ensure the OCR process is working correctly.")

    def test_get_valid_languages(self):
        """Test the get_valid_languages function."""
        valid_languages = get_valid_languages()
        self.assertIsInstance(valid_languages, list, "The output should be a list.")
        self.assertTrue(len(valid_languages) > 0, "The list should not be empty.")
        self.assertIn(self.language_tr, valid_languages, "Turkish should be a valid language.")
        self.assertIn(self.language_en, valid_languages, "English should be a valid language.")


if __name__ == "__main__":
    unittest.main()
