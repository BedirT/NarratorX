# tests/test_ocr.py

import unittest
import os
from narratorx.ocr import process_pdf

class TestOCR(unittest.TestCase):
    def setUp(self):
        # Path to a sample PDF file for testing
        self.sample_pdf = 'tests/sample.pdf'
        # Ensure the sample PDF exists
        if not os.path.exists(self.sample_pdf):
            raise FileNotFoundError(f"Sample PDF not found at {self.sample_pdf}")
        # Specify the language for OCR
        self.language = 'tr'

    def test_process_pdf(self):
        # Call the process_pdf function
        result = process_pdf(self.sample_pdf, self.language)

        # Assert that the result is a non-empty string
        self.assertIsInstance(result, str, "The output should be a string.")
        self.assertTrue(len(result.strip()) > 0, "The output string should not be empty.")

        # Assert that the result contains the expected text
        to_look_for = [
            "hayat tarzları da benzer",
            "Jared Diamond",
        ]
        for text in to_look_for:
            self.assertIn(text, result, f"The output should contain '{text}'.")

    def test_process_pdf_invalid_path(self):
        # Test with an invalid PDF path and expect an exception
        with self.assertRaises(Exception):
            process_pdf('invalid/path/to/sample.pdf', self.language)

    def test_process_pdf_invalid_language(self):
        # Test with an unsupported language code and expect an exception
        with self.assertRaises(Exception):
            process_pdf(self.sample_pdf, 'invalid_lang_code')

    def test_optional_manual_investigation(self):
        # Optional: Manually inspect the output
        result = process_pdf(self.sample_pdf, self.language)

        # save the output to a file
        with open('narratorx_test_output.txt', 'w', encoding='utf-8') as f:
            f.write(result)

        print("Output saved to 'narratorx_test_output.txt' for manual inspection.")
        print("Please inspect the output to ensure the OCR process is working correctly.")
        print("You can also compare the output with the sample PDF file.")
        print("If the output is correct, you can mark this test as passed.")

        # Assert that the result is a non-empty string
        self.assertIsInstance(result, str, "The output should be a string.")
        self.assertTrue(len(result.strip()) > 0, "The output string should not be empty.")

if __name__ == '__main__':
    unittest.main()
