# tests/test_llm.py

import unittest
from unittest.mock import call, patch

from narratorx.llm import llm_process_text
from narratorx.utils import split_text_into_chunks


class TestFixTextOpenAI(unittest.TestCase):

    def setUp(self):
        self.text = "This is a test sentence. " * 50  # Create a long text
        self.language = "en"
        self.model_name = "gpt-4o"

        # Prepare prompts
        with open("src/narratorx/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
            self.user_prompt_template = f.read()
        with open("src/narratorx/prompts/system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()
        self.system_prompt = self.system_prompt_template.format(language=self.language)

    @patch("narratorx.llm.completion")
    def test_llm_process_text_success(self, mock_completion):
        """Test that llm_process_text returns the expected concatenated result on success."""
        # Prepare mock response
        mock_response_content = "Fixed text chunk."
        mock_completion.return_value = {
            "choices": [{"message": {"content": mock_response_content}}]
        }

        # Call the function
        result = llm_process_text(self.text, self.language, model_name=self.model_name)

        # Calculate expected result
        chunks = split_text_into_chunks(self.text, max_tokens=4000, model_name=self.model_name)
        expected_result = "\n".join([mock_response_content for _ in chunks])

        # Assertions
        self.assertEqual(result, expected_result, "Result should be concatenation of fixed chunks.")

    @patch("narratorx.llm.completion")
    def test_llm_process_text_exception(self, mock_completion):
        """Test that llm_process_text raises an exception when the API call fails."""
        # Configure the mock to raise an exception
        mock_completion.side_effect = Exception("API Error")

        # Assertions
        with self.assertRaises(Exception) as context:
            llm_process_text(self.text, self.language, model_name=self.model_name)
        self.assertIn(
            "API Error", str(context.exception), "Exception message should contain 'API Error'."
        )

    @patch("narratorx.llm.completion")
    def test_llm_process_text_empty_text(self, mock_completion):
        """Test that llm_process_text returns an empty string when given empty text."""
        # Call the function with empty text
        result = llm_process_text("", self.language, model_name=self.model_name)

        # Assertions
        self.assertEqual(result, "", "Result should be an empty string when input text is empty.")
        mock_completion.assert_not_called()

    @patch("narratorx.llm.completion")
    def test_llm_process_text_none_text(self, mock_completion):
        """Test that llm_process_text raises TypeError when given None as text."""
        # Assertions
        with self.assertRaises(TypeError):
            llm_process_text(None, self.language, model_name=self.model_name)
        mock_completion.assert_not_called()

    @patch("narratorx.llm.completion")
    def test_llm_process_text_invalid_language(self, mock_completion):
        """Test that llm_process_text proceeds when given an invalid language."""
        # Prepare mock response
        mock_response_content = "Fixed text chunk."
        mock_completion.return_value = {
            "choices": [{"message": {"content": mock_response_content}}]
        }

        # Call the function with an invalid language
        result = llm_process_text(self.text, "", model_name=self.model_name)

        # Assertions
        self.assertIsNotNone(result, "Result should not be None even with an invalid language.")
        mock_completion.assert_called()

    @patch("narratorx.llm.completion")
    def test_llm_process_text_calls_with_correct_parameters(self, mock_completion):
        """Test that llm_process_text calls the completion function with correct parameters."""
        # Prepare mock response
        mock_response_content = "Fixed text chunk."
        mock_completion.return_value = {
            "choices": [{"message": {"content": mock_response_content}}]
        }

        # Call the function
        llm_process_text(self.text, self.language, model_name=self.model_name)

        # Prepare expected calls
        chunks = split_text_into_chunks(self.text, max_tokens=4000, model_name=self.model_name)
        expected_calls = []
        for chunk in chunks:
            user_prompt = self.user_prompt_template.format(content=chunk)
            expected_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "<fixed_text>"},
            ]
            expected_calls.append(
                call(model=self.model_name, messages=expected_messages, stop=["</fixed_text>"])
            )

        # Assertions
        mock_completion.assert_has_calls(expected_calls, any_order=False)
        self.assertEqual(
            mock_completion.call_count, len(chunks), "completion should be called once per chunk."
        )

    @patch("narratorx.llm.completion")
    def test_llm_process_text_multiple_chunks(self, mock_completion):
        """Test that llm_process_text correctly concatenates multiple fixed chunks."""

        # Mock the completion function to return different outputs for each call
        def mock_completion_side_effect(model, messages, stop):
            content = messages[1]["content"]
            fixed_content = content.replace("<fixed_text>", "").replace("</fixed_text>", "").upper()
            return {"choices": [{"message": {"content": fixed_content}}]}

        mock_completion.side_effect = mock_completion_side_effect

        # Call the function
        result = llm_process_text(self.text, self.language, model_name=self.model_name)

        # Since we transformed the content to uppercase,
        # result should be uppercase version of the text
        expected_result = "\n".join(
            [
                self.user_prompt_template.format(content=chunk)
                .replace("<fixed_text>", "")
                .replace("</fixed_text>", "")
                .upper()
                for chunk in split_text_into_chunks(
                    self.text, max_tokens=4000, model_name=self.model_name
                )
            ]
        )

        # Assertions
        self.assertEqual(
            result, expected_result, "Result should be concatenation of fixed uppercase chunks."
        )


if __name__ == "__main__":
    unittest.main()
