# tests/test_cli.py

import os
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from narratorx.cli import main


class TestCLI(unittest.TestCase):

    @patch("narratorx.cli.process_pdf")
    @patch("narratorx.cli.llm_process_text")
    @patch("narratorx.cli.text_to_speech")
    def test_cli_success(self, mock_tts, mock_llm_process_text, mock_process_pdf):
        # Mock the functions
        mock_process_pdf.return_value = "Extracted text"
        mock_llm_process_text.return_value = "Processed text"
        mock_tts.return_value = None

        # Set environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "tests/docs/sample_en.pdf",
                    "--output",
                    "output.wav",
                    "--language",
                    "en",
                    "--model",
                    "gpt-4o",
                    "--log-level",
                    "INFO",
                ],
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIn("Starting NarratorX...", result.output)
            self.assertIn("OCR processing completed.", result.output)
            self.assertIn("LLM text processing completed.", result.output)
            self.assertIn("Text-to-speech synthesis completed.", result.output)
            self.assertIn("Audio saved to output.wav", result.output)

    @patch("narratorx.cli.process_pdf")
    def test_cli_missing_env_var(self, mock_process_pdf):
        # Do not set the required environment variable
        with patch.dict(os.environ, {}, clear=True):
            runner = CliRunner()
            result = runner.invoke(main, ["tests/docs/sample_en.pdf"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Missing required environment variables: OPENAI_API_KEY", result.output)

    def test_cli_invalid_log_level(self):
        # Set environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            runner = CliRunner()
            result = runner.invoke(main, ["tests/sample.pdf", "--log-level", "INVALID"])

            # Check that the exit code is 2, indicating a CLI argument error
            self.assertEqual(result.exit_code, 2)
            self.assertIn("Invalid value for '--log-level'", result.output)
            self.assertIn(
                "'INVALID' is not one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'",
                result.output,
            )

    @patch("narratorx.cli.process_pdf")
    @patch("narratorx.cli.llm_process_text")
    def test_cli_ocr_failure(self, mock_llm_process_text, mock_process_pdf):
        # Mock process_pdf to raise an exception
        mock_process_pdf.side_effect = Exception("OCR error")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            runner = CliRunner()
            result = runner.invoke(main, ["tests/docs/sample_en.pdf", "--log-level", "DEBUG"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("An error occurred: OCR error", result.output)
            self.assertIn("Traceback (most recent call last):", result.output)

    @patch("narratorx.cli.process_pdf")
    @patch("narratorx.cli.llm_process_text")
    def test_cli_llm_failure(self, mock_llm_process_text, mock_process_pdf):
        # Mock functions
        mock_process_pdf.return_value = "Extracted text"
        mock_llm_process_text.side_effect = Exception("LLM processing error")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            runner = CliRunner()
            result = runner.invoke(main, ["tests/docs/sample_en.pdf"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("An error occurred: LLM processing error", result.output)

    @patch("narratorx.cli.process_pdf")
    @patch("narratorx.cli.llm_process_text")
    @patch("narratorx.cli.text_to_speech")
    def test_cli_tts_failure(self, mock_tts, mock_llm_process_text, mock_process_pdf):
        # Mock functions
        mock_process_pdf.return_value = "Extracted text"
        mock_llm_process_text.return_value = "Processed text"
        mock_tts.side_effect = Exception("TTS error")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            runner = CliRunner()
            result = runner.invoke(main, ["tests/docs/sample_en.pdf"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("An error occurred: TTS error", result.output)

    @patch("narratorx.cli.process_pdf")
    def test_cli_help(self, mock_process_pdf):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Convert a PDF to an audiobook.", result.output)
        self.assertIn("--output", result.output)
        self.assertIn("--language", result.output)
        self.assertIn("--model", result.output)
        self.assertIn("--max-tokens", result.output)
        self.assertIn("--max-characters", result.output)
        self.assertIn("--log-level", result.output)
        self.assertIn("--log-file", result.output)


if __name__ == "__main__":
    unittest.main()
