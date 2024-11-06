# narratorx/tts.py

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from TTS.api import TTS
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import NarrativeText


def text_to_speech(text, language, output_path, max_characters=290):
    # Validate that text is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("xtts_v2.0.2").to(device)

    # Create a NarrativeText element with the input text
    element = NarrativeText(text)

    # Chunk the text using unstructured
    chunks = chunk_elements(
        elements=[element],
        max_characters=max_characters,
        overlap=0,
    )

    # Initialize a list to store audio data
    audio_data = []

    # Process each chunk
    for idx, chunk in enumerate(tqdm(chunks, desc="Synthesizing speech")):
        chunk_text = chunk.text.strip()
        if not chunk_text:
            continue

        # Generate speech for the chunk
        wav = tts_model.tts(
            text=chunk_text, language=language, speaker="Asya Anara", split_sentences=False
        )

        # Append the audio data only if it contains data
        if len(wav) > 0:
            audio_data.append(wav)

    # Check if audio_data has content before concatenating
    if not audio_data:
        raise ValueError("No audio data was generated; the input text may be empty or invalid.")

    # Concatenate all audio chunks
    combined_audio = np.concatenate(audio_data)

    # Save the combined audio to the output file
    sf.write(output_path, combined_audio, samplerate=tts_model.synthesizer.output_sample_rate)
