# AccentGuard Pro 🗣️🕵️‍♀️

Use this link to access the app "https://rnstu08-english-accent-analyzer-app-pvvmta.streamlit.app/"

**Challenge Submission for REMWaste AI Agent Solutions Engineer**

AccentGuard Pro is a tool designed to analyze spoken English accents from video URLs or uploaded audio files. It leverages speaker diarization to distinguish between different speakers in an audio track, identifies the language spoken by each, and then classifies the English accent for segments identified as English. This tool aims to assist in evaluating spoken English for hiring purposes by providing nuanced insights into a candidate's speech.

**Live Demo Link:** `[You will insert your deployed Streamlit Cloud app link here]`

## Features

* **Flexible Input:** Accepts public video URLs (YouTube, Loom, direct MP4 links) or direct audio file uploads (WAV, MP3, M4A, etc.).
* **Audio Extraction & Standardization:** Automatically extracts audio from videos and standardizes all audio inputs to 16kHz mono WAV for optimal AI model performance.
* **Speaker Diarization:** Identifies different speaker segments within an audio file ("who spoke when").
* **Language Identification (LID):** Detects the language spoken in each identified segment before attempting accent classification. This ensures English accent analysis is only performed on English speech.
* **English Accent Classification:** For segments identified as English, it classifies the accent (e.g., American, British, Indian, Australian, Canadian) using an open-source AI model.
* **Confidence Scoring:** Provides confidence scores for both language identification and accent classification.
* **Detailed Results:** Outputs a breakdown per speaker segment, including language, accent (if English), confidence, and a summary from the models.
* **User-Friendly Web Interface:** Built with Streamlit for ease of use.
* **Handles Multi-lingual Inputs:** Correctly identifies non-English segments and skips English accent classification for them.

## What REMWaste Was Looking For & How AccentGuard Pro Addresses It

* **Practicality – Can you build something that actually works?**
    * Yes, AccentGuard Pro is a fully functional Streamlit application that processes audio from URLs/uploads and provides diarized language and accent analysis.
* **Creativity – Did you come up with a smart or resourceful solution?**
    * The solution integrates multiple open-source AI models:
        * Speaker diarization (`pyannote/speaker-diarization-3.1`) to handle real-world multi-speaker audio.
        * Language identification (`facebook/mms-lid-126`) to robustly pre-filter for English speech, preventing misclassification of non-English audio.
        * Accent classification (`dima806/english_accents_classification`) for English segments.
    * This multi-stage pipeline demonstrates a resourceful approach to a complex problem.
* **Technical Execution – Is it clean, testable, and logically structured?**
    * The project is organized into modular Python scripts (`audio_extractor.py`, `audio_processor.py`, `diarizer.py`, `language_identifier.py`, `accent_analyzer_os.py`, `app.py`).
    * Each module has a specific responsibility and can be understood/tested independently.
    * The Streamlit application (`app.py`) orchestrates these modules.
    * Industry-standard practices like virtual environments and a `requirements.txt` file are used.

## Technology Stack

* **Python 3.9+**
* **Streamlit:** For the web application UI.
* **yt-dlp:** For downloading videos and extracting audio from URLs.
* **pydub:** For audio file manipulation and standardization.
* **pyannote.audio:** For speaker diarization (requires PyTorch).
* **Hugging Face `transformers` library:** For Language Identification and Accent Classification pipelines (requires PyTorch).
* **PyTorch:** As the backend for the AI models.
* **Core AI Models:**
    * Speaker Diarization: `pyannote/speaker-diarization-3.1`
    * Language Identification: `facebook/mms-lid-126`
    * English Accent Classification: `dima806/english_accents_classification`

## Prerequisites for Running Locally

Before you can run AccentGuard Pro, you'll need the following installed on your system:

1.  **Python:** Version 3.9 or newer.
2.  **FFmpeg:** This is a crucial dependency for `yt-dlp` and `pydub` to process audio/video files.
    * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) (e.g., gyan.dev builds), extract, and add the `bin` folder to your system's PATH environment variable.
    * **macOS:** `brew install ffmpeg`
    * **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
3.  **Git:** For cloning the repository.
4.  **Hugging Face Account and Authentication (IMPORTANT for Diarization Model):**
    * The speaker diarization model (`pyannote/speaker-diarization-3.1`) and its dependent segmentation model (`pyannote/segmentation-3.0`) are "gated" on Hugging Face Hub. This means you need to:
        * Have a free Hugging Face account ([huggingface.co](https://huggingface.co)).
        * Visit the model pages and accept their terms/user conditions while logged into your Hugging Face account:
            * [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
            * [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
        * Authenticate your environment to Hugging Face Hub. The recommended way is using the Hugging Face Command Line Interface (CLI):
            * Generate an Access Token from your Hugging Face settings ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) with `read` permissions.
            * In your terminal (after activating the virtual environment), run `huggingface-cli login` and paste your token when prompted. This will allow the application to download these specific models.
    * *Note: The other models (Language ID, Accent Classification) used in this project are not gated and do not strictly require this token for download, but being logged in is good practice.*

## Setup and Running the Application

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd AccentGuardPro
    ```
    *(Replace `<repository_url>` with the actual URL of the Git repository where this code is hosted.)*

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # Windows (Command Prompt):
    # venv\Scripts\activate.bat
    # macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install Streamlit, pyannote.audio, transformers, PyTorch, and other necessary packages.)*

4.  **Hugging Face Authentication (if not already done globally):**
    Ensure you have completed step #4 in the "Prerequisites" section (accept model terms and run `huggingface-cli login`).

5.  **(Optional) Configure Model Cache Location:**
    The AI models can be large. By default, Hugging Face libraries cache models in `C:\Users\<YourUser>\.cache\huggingface` (Windows) or `~/.cache/huggingface` (macOS/Linux). If you wish to change this (e.g., to a larger drive), set the `HF_HOME` environment variable before running the application:
    * Example (Windows PowerShell): `$env:HF_HOME = "D:\HuggingFaceCache"`
    * Example (macOS/Linux): `export HF_HOME="/path/to/your/cache"`
    * Ensure the specified directory exists.

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will start the application, and it should open in your default web browser (usually at `http://localhost:8501`). The first time you run it, the AI models will be downloaded, which may take some time depending on your internet speed. Subsequent runs will load the models from your local cache (`HF_HOME`).

## How AccentGuard Pro Works

The application follows a multi-stage pipeline:

1.  **Input:** The user provides a video URL or uploads an audio file via the Streamlit interface.
2.  **Audio Extraction/Processing (`audio_extractor.py`, `audio_processor.py`):**
    * If a URL is provided, `yt-dlp` extracts the audio.
    * All input audio (from URL or upload) is standardized by `pydub` to a consistent format (16kHz mono WAV), which is optimal for the subsequent AI models. Basic validation (duration) is also performed.
3.  **Speaker Diarization (`diarizer.py`):**
    * The standardized audio file is processed by `pyannote.audio`'s speaker diarization pipeline (`pyannote/speaker-diarization-3.1`).
    * This stage identifies segments of speech and attributes them to different speakers (e.g., `SPEAKER_00`, `SPEAKER_01`).
4.  **Language Identification & Accent Classification per Segment (`language_identifier.py`, `accent_analyzer_os.py`, `app.py`):**
    * For each significant speaker segment identified by the diarizer:
        * The segment is first passed to a Language Identification (LID) model (`facebook/mms-lid-126`) to determine the language spoken.
        * **If the language is confidently identified as English:** The segment is then passed to an English Accent Classification model (`dima806/english_accents_classification`). This model predicts the English accent (e.g., American, British, Indian, Australian, Canadian) and provides a confidence score.
        * **If the language is identified as non-English:** The detected language is reported, and English accent classification is skipped for that segment.
5.  **Results Display (`app.py`):**
    * The Streamlit application displays the results in a structured way, showing each speaker segment, its identified language, and (if applicable) the classified English accent with its confidence score and a summary from the model.
    * Audio players are provided for the main processed audio and for individual analyzed segments.

## Limitations & Considerations

* **AI Model Accuracy:** The accuracy of diarization, language identification, and accent classification is subject to the capabilities and training data of the underlying AI models. Results are probabilistic and not guaranteed to be 100% correct.
* **Accent Model Coverage:** The current English accent classification model (`dima806/english_accents_classification`) is primarily trained on American (US), British (England), Indian, Australian, and Canadian English. Its performance on other English accents or highly nuanced/mixed accents may vary.
* **Language ID Model Coverage:** While `facebook/mms-lid-126` covers many languages, its accuracy can vary, especially for very short segments or languages less represented in its training.
* **Audio Quality:** The quality of the input audio (background noise, microphone clarity, speaker volume, overlapping speech) significantly impacts the performance of all AI models.
* **Processing Time:** Analyzing audio, especially with speaker diarization and multiple model inferences, can be computationally intensive and may take time, particularly for longer audio files when running on a CPU.
* **Segment Duration:** Very short speech segments (e.g., less than 1.5-2 seconds) are often too brief for reliable language or accent classification and are typically skipped or marked as N/A.

## Meeting Evaluation Criteria

* **Functional Script:** fully operational Streamlit application is provided.
* **Logical Approach:** a multi-stage pipeline using established open-source AI methods for diarization, LID, and accent scoring is implemented.
* **Accent Handling (English):** the tool specifically identifies English speech first and then attempts to classify English accents.
* **Bonus: Confidence Scoring:** confidence scores are provided for both language identification and accent classification.


---
