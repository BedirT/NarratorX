import numpy as np
import soundfile as sf
import streamlit as st
import torch
from tqdm import tqdm
import os
from TTS.api import TTS
import nltk
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from litellm import completion
from typing import List
import json


nltk.download('punkt_tab')

# Define language-specific breakpoints
language_breakpoints = {
    'en': [' and ', ' but ', ' or '],
    'es': [' y ', ' pero ', ' o '],
    'fr': [' et ', ' mais ', ' ou '],
    'de': [' und ', ' aber ', ' oder '],
    'it': [' e ', ' ma ', ' o '],
    'pt': [' e ', ' mas ', ' ou '], # Portuguese
    'pl': [' i ', ' ale ', ' lub '], # Polish
    'tr': [' ve ', ' ama ', ' veya '], # Turkish
    'ru': [' и ', ' но ', ' или '], # Russian
    'nl': [' en ', ' maar ', ' of '], # Dutch
    'cz': [' a ', ' ale ', ' nebo '], # Czech
    'ar': [' و ', ' لكن ', ' أو '], # Arabic
    'cn': [' 和 ', ' 但是 ', ' 或者 '], # Chinese
    'jp': [' そして ', ' しかし ', ' または '], # Japanese
    'hu': [' és ', ' de ', ' vagy '], # Hungarian
    'kr': [' 그리고 ', ' 그러나 ', ' 또는 '], # Korean
}

with open("src/narratorx/prompts/splitter_user_prompt.txt", "r", encoding="utf-8") as f:
    user_prompt_template = f.read()
with open("src/narratorx/prompts/splitter_system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_template = f.read()

convert_language_code = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'pl': 'polish',
    'tr': 'turkish',
    'ru': 'russian',
    'nl': 'dutch',
    'cz': 'czech',

    'ar': 'english', # there is no arabic in nltk
    'cn': 'english', # there is no chinese in nltk
    'jp': 'english', # there is no japanese in nltk
    'hu': 'english', # there is no hungarian in nltk
    'kr': 'english', # there is no korean in nltk
}

class LLMChunkSplitter(BaseModel):
    chunks: List[str]

def split_with_llm(model_name, text, max_chars, language='en'):
    user_prompt = user_prompt_template.format(content=text, max_chars=max_chars)
    messages = [
        {"role": "system", "content": system_prompt_template},
        {"role": "user", "content": user_prompt},
    ]
    response_text = completion(
        model=model_name,
        messages=messages,
        max_tokens=4000, # hard coded for now
        response_format=LLMChunkSplitter,
    )
    json_res = response_text.choices[0].message.content
    parsed_res = json.loads(json_res)
    return parsed_res["chunks"]

def split_by_natural_breakpoints(text, max_chars, language='en', model_name="gpt-4o-mini"):
    breakpoints = language_breakpoints.get(language, []) + [",", ";", ":"]
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        tentative = (current_chunk + " " + word).strip() if current_chunk else word
        if len(tentative) <= max_chars:
            current_chunk = tentative
        else:
            # If we can split on a breakpoint
            if any(bp in current_chunk for bp in breakpoints):
                for bp in breakpoints:
                    if bp in current_chunk:
                        parts = current_chunk.rsplit(bp, 1)
                        # Append part before the breakpoint
                        if not parts[0].endswith(bp):
                            parts[0] += bp
                        chunks.append(parts[0].strip())
                        # Start new chunk after breakpoint + current word
                        current_chunk = (parts[1] + " " + word).strip()
                        break
            else:
                # No breakpoint found; finalize this chunk and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    # At this point, we have chunks split by breakpoints, but they might still be too long.
    # We involve LLM to split them further if needed.
    final_chunks = []
    for c in chunks:
        if len(c) > max_chars:
            new_cs = split_with_llm(model_name, c, max_chars, language)
            final_chunks.extend(new_cs)
        else:
            final_chunks.append(c)
    return final_chunks

def split_text_into_chunks(text, max_chars, language='en', model_name="gpt-4o-mini"):
    # Split text into paragraphs
    paragraphs = text.strip().split("\n")
    all_chunks = []

    for para in paragraphs:
        para = para.strip()
        # Split into sentences
        sentences = sent_tokenize(para, language=convert_language_code[language])
        for sent in sentences:
            if len(sent) <= max_chars:
                all_chunks.append(sent.strip())
            else:
                # Try splitting by natural breakpoints
                chunks = split_by_natural_breakpoints(sent, max_chars, language, model_name)
                all_chunks.extend(chunks)

    return all_chunks

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
    model_name="gpt-4o-mini",
):
    # Validate that text is a string
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if tts_model is None:
        # Load the model if not provided
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = TTS("xtts_v2.0.2").to(device)

    # Use the custom splitting method instead of unstructured
    chunks = split_text_into_chunks(text, max_characters, language=language, model_name=model_name)

    # Initialize a list to store audio data
    audio_data = []

    # Set up progress bar depending on the environment
    total_chunks = len(chunks)
    if use_streamlit:
        with streamlit_container:
            progress_bar = st.progress(0)
    else:
        progress_bar = tqdm(total=total_chunks, desc="Synthesizing speech")

    # Process each chunk
    for idx, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
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
                progress_bar.progress((idx + 1) / total_chunks)
        else:
            progress_bar.update(1)

    # Check if audio_data has content before concatenating
    if not audio_data:
        raise ValueError("No audio data was generated; the input text may be empty or invalid.")

    # Concatenate all audio chunks
    combined_audio = np.concatenate(audio_data)

    # Save the combined audio to the output file
    if '/' in output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, combined_audio, samplerate=tts_model.synthesizer.output_sample_rate)

    # Finalize the progress bar
    if use_streamlit:
        progress_bar.empty()
    else:
        progress_bar.close()
