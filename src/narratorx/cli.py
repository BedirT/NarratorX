# narratorx/cli.py


import logging
import logging.handlers
import os
import sys

import click
import colorlog

from narratorx.llm import llm_process_text
from narratorx.ocr import process_pdf
from narratorx.tts import text_to_speech


def setup_logging(log_level, log_file=None):
    """Sets up logging with color formatting for console output and optional file logging."""
    logger = logging.getLogger("narratorx")
    numeric_level = getattr(logging, log_level.upper(), None)
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers = []

    # Create colorful console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Define colorized format
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # File handler for logs (no color)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="output.wav", help="Output audio file path.")
@click.option("--language", "-l", default="en", help="Language code (e.g., en, tr).")
@click.option("--model", "-m", default="ollama/llama3.1", help="LLM model name.")
@click.option("--max-characters-llm", default=1000, help="Maximum characters per LLM chunk.")
@click.option("--max-tokens", default=4000, help="Maximum output tokens for the LLM call.")
@click.option(
    "--max-characters-tts",
    default=250,
    help="Maximum characters per TTS chunk. This is enforced by the TTS model, please refer to the model documentation for the exact limit.",  # noqa: E501
)
@click.option(
    "--log-level",
    default="ERROR",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging level.",
)
@click.option("--log-file", default=None, help="Path to log file.")
def main(
    pdf_path,
    output,
    language,
    model,
    max_characters_llm,
    max_tokens,
    max_characters_tts,
    log_level,
    log_file,
):
    """
    NarratorX: Convert a PDF to an audiobook.
    """

    try:
        # Set up logging
        logger = setup_logging(log_level, log_file)

        # Check for required environment variables
        required_env_vars = ["OPENAI_API_KEY"]
        missing_env_vars = [var for var in required_env_vars if var not in os.environ]
        if missing_env_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_env_vars)}")
            sys.exit(1)

        logger.info("Starting NarratorX...")

        # Step 1: OCR processing
        logger.info("Starting OCR processing...")
        text = process_pdf(pdf_path, language)
        logger.info("OCR processing completed.")

        # Step 2: LLM text processing
        logger.info("Starting LLM text processing...")
        fixed_text = llm_process_text(
            text, language, model_name=model, max_chars=max_characters_llm, max_tokens=max_tokens
        )
        logger.info("LLM text processing completed.")

        # Step 3: Text-to-speech synthesis
        logger.info("Starting text-to-speech synthesis...")
        text_to_speech(fixed_text, language, output, max_characters_tts)
        logger.info(f"Text-to-speech synthesis completed. Audio saved to {output}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
