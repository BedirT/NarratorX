import numpy as np
import soundfile as sf
import streamlit as st
import torch
from tqdm import tqdm
from TTS.api import TTS
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import NarrativeText


@st.cache_resource
def load_tts_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("xtts_v2.0.2").to(device)
    return tts_model


def text_to_speech(
    text,
    language,
    output_path,
    max_characters=290,
    tts_model=None,
    use_streamlit=False,
    streamlit_container=None,
):
    # Validate that text is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if tts_model is None:
        # Load the model if not provided
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

    # Set up progress bar depending on the environment
    if use_streamlit:
        with streamlit_container:
            progress_bar = st.progress(0)
    else:
        progress_bar = tqdm(total=len(chunks), desc="Synthesizing speech")

    # Process each chunk
    for idx, chunk in enumerate(chunks):
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

        # Update progress bar
        if use_streamlit:
            with streamlit_container:
                progress_bar.progress((idx + 1) / len(chunks))
        else:
            progress_bar.update(1)

    # Check if audio_data has content before concatenating
    if not audio_data:
        raise ValueError("No audio data was generated; the input text may be empty or invalid.")

    # Concatenate all audio chunks
    combined_audio = np.concatenate(audio_data)

    # Save the combined audio to the output file
    sf.write(output_path, combined_audio, samplerate=tts_model.synthesizer.output_sample_rate)

    # Finalize the progress bar
    if use_streamlit:
        progress_bar.empty()
    else:
        progress_bar.close()
