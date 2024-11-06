
<p align="center">
    <img src="img/banner.png" alt="NarratorX Banner" />
</p>

<p align="center">
    <em>Transform your PDFs into engaging audiobooks with the power of OCR, LLMs, and TTS in 16 languages.</em>
</p>

<p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License" /></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version" /></a>
</p>


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Supported Languages](#supported-languages-)
- [Roadmap](#roadmap)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Streamlit Web Application](#streamlit-web-application)
- [Contributing](#contributing)
  - [Style Guidelines](#style-guidelines)
  - [Testing](#testing)
- [License](#license)

## Introduction

🎉 Welcome to NarratorX! 🎉 For the longest time, I found myself saying, "Ah shucks, there's no audiobook for this one… guess I'll have to wait or squeeze in some reading when I have free time." If you’re like me and love listening to audiobooks while doing chores 🧹, commuting 🚶, or just relaxing 😌, then I guess this tool was made for you!

📚 With NarratorX, you can freely convert the PDF of that book that’s been lingering on your to-read list into a beautifully narrated, non-robotic audiobook. Say hello to NarratorX – savior of the busy and the lazy! By harnessing advanced OCR 🔍, powerful LLMs 🤖, and state-of-the-art TTS 🎙️, NarratorX transforms your PDFs into immersive audiobooks, ready to accompany you wherever you go. And all for free, my favorite kinda payment method...

NarratorX is a passion project, specially crafted for my book club 📖💬. With support for 16 languages 🌍, it’s perfect for readers worldwide. I hope it brings you joy too! So go on, read away and listen away! 🎧✨

## Features

| 🚀 **Feature**                           | ✨ **Description** |
|-----------------------------------------|--------------------|
| **Seamless PDF to Audiobook Conversion** 🎧 | Effortlessly convert your PDFs into high-quality audiobooks using OCR 🔍, LLMs 🤖, and TTS 🎙️, fully automating the process from text extraction to speech generation. |
| **Multi-Language Support** 🌍             | Supports 16 languages, including English, Spanish, French, German, Turkish, and more, making NarratorX accessible to a global audience. |
| **Open-Source LLM Integration** 🔓        | Leverage the Ollama library for local, open-source LLMs. OpenAI is optional, offering flexibility for users with different needs. |
| **User-Friendly Interfaces** 🖥️           | Use either the command-line interface for quick, scriptable conversions or the Streamlit web app for a visual, interactive experience. |
| **Customizable Settings** 🎛️              | Fine-tune your experience by adjusting chunk sizes, selecting models, and choosing language preferences to suit each document. |
| **Robust Error Handling and Logging** 🛠️  | With comprehensive error handling and logging, NarratorX ensures smooth operation and easy troubleshooting if issues arise. |


---

## Demo



https://github.com/user-attachments/assets/eeb3157d-8d2f-423f-97dd-da30439c7bf2




🎧 **Sample Audiobook Conversions**
Here are some sample audiobook conversions created with NarratorX! (Not cherry-picked – just pure NarratorX magic ✨)

| 🌐 **Language** | 📚 **Book Title** | 🔊 **Sample Audio** |
|----------------|--------------------|---------------------|
| 🇺🇸 **English** | *"The Little Prince"* by Antoine de Saint-Exupéry | [Listen 🎶](https://github.com/user-attachments/assets/59179257-19aa-431f-b7ba-2419a234e3bc) |
| 🇪🇸 **Spanish** | *"Don Quijote"* by Miguel de Cervantes | [Escuchar 🎶](https://github.com/user-attachments/assets/3605936c-2644-4ddf-888f-6128f9adab10) |
| 🇫🇷 **French** | *"Les Misérables"* by Victor Hugo | [Écouter 🎶](https://github.com/user-attachments/assets/c5aacc01-333b-4d47-94a3-ab4d53fa83f7) |
| 🇹🇷 **Turkish** | *"Tüfek, Mikrop ve Çelik"* by Jared Diamond | [Dinle 🎶](https://github.com/user-attachments/assets/7b0dfcf0-8c16-4f5a-b925-d965d6c3b033) |

---

## Supported Languages 🌍

NarratorX supports a wide array of languages, making it accessible for many readers around the world! 🌐 The currently supported languages are:

- 🇺🇸 **English** (en)
- 🇪🇸 **Spanish** (es)
- 🇫🇷 **French** (fr)
- 🇩🇪 **German** (de)
- 🇮🇹 **Italian** (it)
- 🇵🇹 **Portuguese** (pt)
- 🇵🇱 **Polish** (pl)
- 🇹🇷 **Turkish** (tr)
- 🇷🇺 **Russian** (ru)
- 🇳🇱 **Dutch** (nl)
- 🇨🇿 **Czech** (cs)
- 🇸🇦 **Arabic** (ar)
- 🇨🇳 **Chinese** (zh-cn)
- 🇯🇵 **Japanese** (ja)
- 🇭🇺 **Hungarian** (hu)
- 🇰🇷 **Korean** (ko)

With NarratorX, the world of audiobooks is just a language selection away! 🌎📖🎶

## Roadmap

Here's what's on the horizon:

- [ ] **Enhanced LLM Options**: Integration with additional open-source LLMs and libraries.
- [ ] **Custom TTS Voices**: Support for custom TTS voices and accents.
- [ ] **Enhanced Chunking and OCR**: I am planning to move to [open-parse](https://github.com/Filimoa/open-parse) for better OCR results.
- [ ] **Support for additional file formats**: EPUB, DOCX, Image, and plain text support.
- [ ] **Automated Chunk Size Optimization**: Automatically adjust chunk sizes based on model capabilities and language requirements.
- [ ] **Improved UI/UX**: Enhancements to the Streamlit app for a more intuitive user experience.
  - Add a stopping option.
  - Add step-wise processing - OCR, LLM, TTS.

I welcome all contributions! If you have ideas, suggestions, or want to contribute, feel free to reach out.

---

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Poetry**: A tool for dependency management.
  ```bash
  pip install poetry
  ```
- **CUDA (Optional)**: For GPU acceleration with CUDA-compatible GPUs. This can significantly speed up processing times.

### Steps

1. **Clone the Repository**

   Start by cloning the NarratorX repository to your local machine:

   ```bash
   git clone https://github.com/bedirt/narratorx.git
   cd narratorx
   ```

2. **Install Dependencies**

   Use Poetry to install all required dependencies:

   ```bash
   poetry install
   ```

3. **Set Up Environment Variables**

   NarratorX can work with both OpenAI's GPT models and local open-source LLMs via the Ollama library.

   - **If you want to use OpenAI Models**:

     Obtain your OpenAI API key and set it as an environment variable:

     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

4. **(Optional - Just for Contributing) Install Pre-commit Hooks**

   If you plan to contribute to NarratorX, set up pre-commit hooks to maintain code quality:

   ```bash
   pre-commit install
   ```

---

## Usage

I implemented two ways to use NarratorX. You can choose between the command-line interface for quick operations or the Streamlit web app for a more interactive experience.

### Command-Line Interface

Convert your PDF to an audiobook directly from the terminal:

```bash
narratorx path/to/yourfile.pdf --output output.wav --language en --model gpt-4o --log-level INFO
```

**Parameters Explained:**

- `path/to/yourfile.pdf`: The path to the PDF you wish to convert.
- `--output, -o`: (Optional) The path where the output audio file will be saved. Defaults to `output.wav`.
- `--language, -l`: (Optional) The language code of your PDF content (e.g., `en` for English, `tr` for Turkish).
- `--model, -m`: (Optional) The LLM model to use. Options include `gpt-4o`, `gpt-4o-mini`, or any model supported by the Ollama library like `llama3.1`. You can see ollama models [here](https://ollama.com/library). Do not forget to use `ollama/` prefix for ollama models.
- `--max-characters-llm`: (Optional) Maximum characters per LLM chunk. Adjust based on model capabilities. 2-4k is a good starting point.
- `--max-tokens`: (Optional) Maximum tokens per LLM call. Adjust based on model capabilities. 2-4k is a good starting point again.
- `--max-characters-tts`: (Optional) Maximum characters per TTS chunk. This value should be changed based on the language you are using, if you get a warning `Warning: The text length exceeds the character limit of 239 for language 'es', this might cause truncated audio.` you should decrese this value to be the same as the warning message. (I will automate this soon, lazy at the moment :))
- `--log-level`: (Optional) Set the logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

**Example:**

Using an open-source LLM with Ollama:

```bash
narratorx path/to/yourfile.pdf --output output.wav --language en --model ollama/llama3.1 --log-level INFO
```

### Streamlit Web Application

For a more user-friendly interface, use the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Open your web browser and navigate to `http://localhost:8501`. Here, you can:

- Upload your PDF file (up to 200MB).
- Select the language and preferred LLM model.
- Adjust advanced settings such as chunk sizes (optional).
- Monitor the progress of each processing step.
- Listen to a preview and download the final audiobook.

---

## Contributing

Love contributions! Whether it's bug fixes, new features, or documentation improvements, your help is invaluable, and I appreciate it.

### Style Guidelines

To maintain code quality and consistency, please adhere to the following guidelines:

- **Code Formatting**: Use `black` for code formatting with a line length of 100 characters. This ensures that all code follows a consistent style.

  ```bash
  black --line-length 100 .
  ```

- **Import Sorting**: Use `isort` with the `black` profile to sort imports correctly.

  ```bash
  isort --profile black .
  ```

- **Linting**: Use `flake8` to lint your code and catch potential issues.

  ```bash
  flake8
  ```

- **Pre-commit Hooks**: We have pre-commit hooks set up to automate these checks before each commit. Install them with:

  ```bash
  pre-commit install
  ```

  **Pre-commit Configuration:**

  ```yaml
  repos:
    - repo: local
      hooks:
        - id: black
          name: black
          entry: poetry run black --line-length 100 --exclude '.venv/*'
          language: system
          types: [python]
        - id: flake8
          name: flake8
          entry: poetry run flake8 --exclude .venv --config .flake8
          language: system
          types: [python]
        - id: isort
          name: isort
          entry: poetry run isort --profile black --skip .venv
          language: system
          types: [python]
  ```

  You can ignore all others, and just install pre-commit hooks with the command above, and (since I already have the configuration file) you can use `pre-commit run --all-files` to run all checks.

### Testing

Ensure your changes don't break existing functionality by running tests:
1. **Install Test Dependencies**

    ```bash
    poetry install --with dev
    ```

2. **Run Tests**

    ```bash
    python -m unittest discover tests/
    ```

3. **Test Specific Modules**

    ```bash
    python -m unittest tests/test_llm.py
    ```

### How to Contribute

1. **Fork the Repository**

   Click the "Fork" button at the top right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/narratorx.git
   cd narratorx
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Write code, fix bugs, or improve documentation.

5. **Format and Lint Your Code**

   ```bash
   black --line-length 100 .
   isort --profile black .
   flake8
   ```

   or simply:

   ```bash
   pre-commit run --all-files
   ```

6. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add your descriptive commit message here"
   ```

7. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**

   Go to the original repository and open a pull request. Provide a clear description of your changes.

---

## License

NarratorX is licensed under the **GNU General Public License v3.0**. By contributing to this project, you agree that your contributions will be licensed under its GPLv3.

See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

I made use of the following libraries and tools to build NarratorX and am grateful for all their contributions:
- [OpenAI](https://openai.com)
- [Ollama](https://ollama.com)
- [Coqui/TTS](https://github.com/coqui-ai/TTS)
- [Streamlit](https://streamlit.io)
- [Surya-OCR](https://github.com/VikParuchuri/surya)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [LiteLLM](https://github.com/BerriAI/litellm)

---
Thank you for using NarratorX! I am excited to see how you transform your reading experience. If you have any questions, suggestions, or need assistance, feel free to open an issue or reach out.

You can add me on goodreads [here](https://www.goodreads.com/bedirt).

Happy listening! 🎧📖
