# runner.py

import click

from narratorx.cli import main

# Set necessary environment variables
# os.environ["OPENAI_API_KEY"] = "your_openai_key_here"  # Replace with your actual key

# Prepare command-line arguments
pdf_path = "tests/docs/sample_en.pdf"  # Replace with the path to your PDF
output = "output.wav"
language = "en"
model = "gpt-4o-mini"
log_level = "ERROR"
log_file = "narratorx.log"
max_tokens = 4000
max_llm_characters = 1000
max_stt_characters = 290


# Simulate running the CLI by directly invoking `main()`
@click.command()
@click.pass_context
def run_cli(ctx):
    """
    Run NarratorX CLI for debugging purposes.
    """
    # Call the main function with the simulated CLI parameters
    ctx.invoke(
        main,
        pdf_path=pdf_path,
        output=output,
        language=language,
        model=model,
        max_characters_llm=max_llm_characters,
        max_tokens=max_tokens,
        max_characters_tts=max_stt_characters,
        log_level=log_level,
        log_file=log_file,
    )


# Run the CLI runner
if __name__ == "__main__":
    run_cli()
