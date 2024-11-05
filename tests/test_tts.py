# tests/test_tts.py

import unittest
import numpy as np
import soundfile as sf
from unittest.mock import MagicMock, patch, call
from narratorx.tts import text_to_speech
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import NarrativeText

class TestTextToSpeech(unittest.TestCase):

    @patch("narratorx.tts.TTS")
    @patch("narratorx.tts.sf.write")
    def test_text_to_speech_success(self, mock_sf_write, mock_tts_class):
        """Test that text_to_speech generates audio correctly with chunking."""
        # Mock TTS model and audio generation
        mock_tts_instance = MagicMock()
        mock_tts_instance.to.return_value = mock_tts_instance  # Ensure .to() returns the mock instance
        mock_tts_instance.tts.return_value = np.array([0.0, 1.0, -1.0])  # Mock audio waveform
        mock_tts_class.return_value = mock_tts_instance

        # Define test parameters
        text = "Sentence one. Sentence two. Sentence three." * 10  # Long text to enforce chunking
        language = "en"
        output_path = "output.wav"
        max_characters = 50  # Force multiple chunks

        # Call the function
        text_to_speech(text, language, output_path, max_characters=max_characters)

        # Generate expected chunks for comparison
        element = NarrativeText(text)
        expected_chunks = chunk_elements(elements=[element], max_characters=max_characters, overlap=0)
        expected_chunk_count = len(expected_chunks)
        print(f"Expected number of chunks: {expected_chunk_count}")

        # Assertions
        # Check that TTS was called for each chunk
        actual_call_count = mock_tts_instance.tts.call_count
        self.assertEqual(
            actual_call_count,
            expected_chunk_count,
            f"TTS should be called once per chunk. Expected {expected_chunk_count}, got {actual_call_count}."
        )

        # Verify that each TTS call had the correct text for each chunk
        for idx, (call_args, chunk) in enumerate(zip(mock_tts_instance.tts.call_args_list, expected_chunks)):
            print(f"Chunk {idx} text: {chunk.text.strip()}")
            args, kwargs = call_args
            self.assertEqual(kwargs["text"], chunk.text.strip(), "Each chunk should be processed as text.")
            self.assertEqual(kwargs["language"], language)
            self.assertEqual(kwargs["speaker"], "Asya Anara")
            self.assertEqual(kwargs["split_sentences"], False)

        # Check that the concatenated audio data is written to the output path
        mock_sf_write.assert_called_once()
        sf_write_args, sf_write_kwargs = mock_sf_write.call_args
        self.assertEqual(sf_write_args[0], output_path, "Audio should be written to the specified output path.")

    @patch("narratorx.tts.TTS")
    @patch("narratorx.tts.sf.write")
    def test_text_to_speech_empty_text(self, mock_sf_write, mock_tts_class):
        """Test text_to_speech with empty text input."""
        text = ""
        language = "en"
        output_path = "output.wav"

        # Call the function and check for ValueError due to empty content
        with self.assertRaises(ValueError):
            text_to_speech(text, language, output_path)

        # Ensure TTS and soundfile.write were never called
        mock_tts_class.return_value.tts.assert_not_called()
        mock_sf_write.assert_not_called()

    @patch("narratorx.tts.TTS")
    @patch("narratorx.tts.sf.write")
    def test_text_to_speech_invalid_text_type(self, mock_sf_write, mock_tts_class):
        """Test text_to_speech raises TypeError with non-string text input."""
        text = 12345  # Invalid input type
        language = "en"
        output_path = "output.wav"

        # Check that TypeError is raised
        with self.assertRaises(TypeError):
            text_to_speech(text, language, output_path)

        # Ensure TTS and soundfile.write were never called
        mock_tts_class.return_value.tts.assert_not_called()
        mock_sf_write.assert_not_called()

    @patch("narratorx.tts.TTS")
    @patch("narratorx.tts.sf.write")
    def test_text_to_speech_no_audio_generated(self, mock_sf_write, mock_tts_class):
        """Test that text_to_speech raises an error if no audio data was generated."""
        # Mock the TTS instance
        mock_tts_instance = MagicMock()
        mock_tts_instance.to.return_value = mock_tts_instance  # Ensure .to() returns the mock instance

        # Set tts to return an empty numpy array to simulate no audio generation
        mock_tts_instance.tts.return_value = np.array([])  # Empty audio data
        mock_tts_class.return_value = mock_tts_instance

        # Define test parameters
        text = "This text will be processed, but no audio will be generated."
        language = "en"
        output_path = "output.wav"

        # Call the function and expect ValueError due to empty audio data
        with self.assertRaises(ValueError):
            text_to_speech(text, language, output_path)

        # Ensure that TTS was called
        mock_tts_instance.tts.assert_called()

        # Ensure soundfile.write was not called since no audio data should be written
        mock_sf_write.assert_not_called()

    @patch("narratorx.tts.TTS")
    @patch("narratorx.tts.sf.write")
    def test_text_to_speech_single_chunk(self, mock_sf_write, mock_tts_class):
        """Test text_to_speech processes a single chunk correctly."""
        # Mock TTS model
        mock_tts_instance = MagicMock()
        mock_tts_instance.to.return_value = mock_tts_instance
        mock_tts_instance.tts.return_value = np.array([0.0, 1.0, -1.0])  # Mock audio waveform
        mock_tts_class.return_value = mock_tts_instance

        # Define short text to ensure a single chunk
        text = "Short sentence that fits in one chunk."
        language = "en"
        output_path = "output_single.wav"

        # Call the function
        text_to_speech(text, language, output_path, max_characters=1000) # Large enough to avoid splitting

        # Assertions
        mock_tts_instance.tts.assert_called_once_with(
            text=text,
            language=language,
            speaker="Asya Anara",
            split_sentences=False
        )
        mock_sf_write.assert_called_once()

        # Verify audio output path
        sf_write_args, sf_write_kwargs = mock_sf_write.call_args
        self.assertEqual(sf_write_args[0], output_path, "Audio should be written to the specified output path.")

if __name__ == "__main__":
    unittest.main()
