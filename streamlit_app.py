import streamlit as st
import os
import tempfile

from narratorx.ocr import process_pdf
from narratorx.llm import llm_process_text
from narratorx.tts import text_to_speech, load_tts_model
from narratorx.utils import get_valid_languages

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Page configuration
st.set_page_config(
    page_title="NarratorX - PDF to Audiobook Converter",
    layout="centered",
    initial_sidebar_state="expanded",
)

# st.image("img/logo.png")
st.title("NarratorX - PDF to Audiobook Converter")
st.write("Convert your PDFs into audiobooks using OCR, LLMs, and TTS technologies.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file to upload", type=["pdf"], accept_multiple_files=False)

# Check file size
MAX_FILE_SIZE_MB = 20
if uploaded_file is not None and uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB.")
    uploaded_file = None

# Language selection
valid_languages = get_valid_languages()
language = st.selectbox("Select Language", options=valid_languages, index=valid_languages.index("en"))

# Model selection
model_options = ["gpt-4o-mini", "llama3.1", "gpt-4o"]
model = st.selectbox("Select LLM Model", options=model_options)
if model not in ["gpt-4o", "gpt-4o-mini"]:
    model = "ollama/" + model

# Advanced Settings
with st.expander("Advanced Settings"):
    max_characters_llm = st.number_input(
        "Maximum characters per LLM chunk",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )
    max_tokens = st.number_input(
        "Maximum tokens per LLM call",
        min_value=100,
        max_value=8000,
        value=4000,
        step=100,
    )
    max_characters_tts = st.number_input(
        "Maximum characters per TTS chunk",
        min_value=100,
        max_value=1000,
        value=250,
        step=50,
    )


if st.button("Convert to Audiobook", use_container_width=True):
    if uploaded_file is not None:
        if "OPENAI_API_KEY" not in os.environ:
            st.error("Missing OpenAI API key. Please set the OPENAI_API_KEY environment variable.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.spinner("Processing...")
                container = st.container()

            try:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_pdf_path = tmp_file.name

                # Step 1: OCR Processing
                expander = st.expander("Task Logs", expanded=True, icon=":material/web_stories:")
                with expander:
                    st.info("Processing PDF with OCR...")
                text = process_pdf(tmp_pdf_path, language)
                with expander:
                    st.success("OCR processing completed.")

                # Step 2: LLM Text Processing
                with expander:
                    st.info("Processing text with LLM...")
                fixed_text = llm_process_text(
                    text,
                    language,
                    model_name=model,
                    max_chars=max_characters_llm,
                    max_tokens=max_tokens,
                )
                with expander:
                    st.success("Text processing completed.")

                # Step 3: Text-to-Speech Synthesis
                with expander:
                    st.info("Synthesizing speech...")
                output_audio_path = os.path.join(tempfile.gettempdir(), "output.wav")
                tts_model = load_tts_model()
                text_to_speech(
                    fixed_text,
                    language,
                    output_audio_path,
                    max_characters_tts,
                    tts_model=tts_model,
                    use_streamlit=True,
                    streamlit_container=container,
                )
                with expander:
                    st.success("Speech synthesis completed.")

                # Display audio player
                audio_file = open(output_audio_path, "rb")
                audio_bytes = audio_file.read()
                with container:
                    st.audio(audio_bytes, format="audio/wav")

                    # Provide download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="audiobook.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

            except Exception as e:
                with container:
                    st.error(f"An error occurred: {e}")
            finally:
                # Clean up temporary files
                if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)
                if output_audio_path and os.path.exists(output_audio_path):
                    os.remove(output_audio_path)
    else:
        st.warning("Please upload a PDF file to proceed.")
