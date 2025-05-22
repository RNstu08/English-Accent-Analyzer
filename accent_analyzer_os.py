# accent_analyzer_os.py
# I'll use this module to classify the English accent from an audio file
# or an audio segment using a Hugging Face transformers model.

import os
from transformers import pipeline # The main workhorse from Hugging Face
import torch # To specify device (CPU/GPU)
import time

# I'm selecting a model fine-tuned for English accent classification.
# 'dima806/english_accents_classification' seems suitable.
# Its expected labels are: 'us', 'england', 'indian', 'australia', 'canada'
ACCENT_MODEL_NAME = "dima806/english_accents_classification"

# I'll use a global variable for the classifier pipeline to load it only once.
ACCENT_CLASSIFIER_PIPELINE = None

# I can create a mapping for more user-friendly accent names.
ACCENT_LABEL_MAP = {
    "us": "American English (US)",
    "england": "British English (England)", # Or just "British English"
    "indian": "Indian English",
    "australia": "Australian English",
    "canada": "Canadian English",
    # I should add more if the model supports other labels.
    # For now, these are based on the likely output of the chosen model.
}

def load_accent_classifier_pipeline():
    """
    Loads the Hugging Face audio-classification pipeline for accent detection.
    Caches it in the global ACCENT_CLASSIFIER_PIPELINE variable.
    """
    global ACCENT_CLASSIFIER_PIPELINE
    if ACCENT_CLASSIFIER_PIPELINE is None:
        print(f"Loading accent classification pipeline: {ACCENT_MODEL_NAME}...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device for accent classification: {device}")

            ACCENT_CLASSIFIER_PIPELINE = pipeline(
                "audio-classification", # This is the task type
                model=ACCENT_MODEL_NAME,
                device=device # device can be 0 for cuda:0, 1 for cuda:1, or -1 for CPU with older transformers.
                              # Newer versions prefer torch.device object.
            )
            print("Accent classification pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading accent classification pipeline: {e}")
            ACCENT_CLASSIFIER_PIPELINE = None # Ensure it's None if loading failed
            raise # Re-raise the exception so the calling code knows it failed
    return ACCENT_CLASSIFIER_PIPELINE

def classify_accent(audio_path: str, top_k: int = 3) -> dict | None:
    """
    Classifies the English accent from an audio file.

    Args:
        audio_path (str): Path to the input audio file (should be 16kHz mono WAV).
        top_k (int): Number of top predictions to return.

    Returns:
        dict | None: A dictionary containing the top accent prediction:
                     {'accent': 'Friendly Name', 'confidence_percent': SCORE, 'raw_label': 'model_label', 'summary': '...'}
                     or None if classification fails.
                     The summary might include other top predictions.
    """
    classifier = load_accent_classifier_pipeline()
    if classifier is None:
        print("Accent classification pipeline not available.")
        return {"accent": "Error", "confidence_percent": 0, "raw_label": "N/A", "summary": "Accent classification model not loaded."}


    if not os.path.exists(audio_path):
        print(f"Audio file not found for accent classification: {audio_path}")
        return {"accent": "Error", "confidence_percent": 0, "raw_label": "N/A", "summary": f"Audio file not found: {audio_path}"}
    if os.path.getsize(audio_path) == 0:
        return {"accent": "Error", "confidence_percent": 0, "raw_label": "N/A", "summary": "Audio file is empty."}


    print(f"Starting accent classification for: {audio_path}")
    start_time = time.time()

    try:
        # The audio-classification pipeline can usually take a file path directly.
        # It handles loading and resampling if needed, but our audio is already 16kHz.
        predictions = classifier(audio_path, top_k=top_k)
        # Example output of 'predictions':
        # [{'score': 0.99, 'label': 'us'}, {'score': 0.005, 'label': 'england'}, ...]
    except Exception as e:
        print(f"Error during accent classification for {audio_path}: {e}")
        return {"accent": "Error", "confidence_percent": 0, "raw_label": "N/A", "summary": f"Error during model inference: {str(e)}"}

    end_time = time.time()
    print(f"Accent classification completed in {end_time - start_time:.2f} seconds.")

    if not predictions:
        print("No predictions returned by the accent classifier.")
        return {"accent": "Undetermined", "confidence_percent": 0, "raw_label": "N/A", "summary": "Model returned no predictions."}

    # The first prediction in the list is the one with the highest score.
    top_prediction = predictions[0]
    raw_label = top_prediction['label']
    confidence = top_prediction['score']

    friendly_accent_name = ACCENT_LABEL_MAP.get(raw_label, raw_label.capitalize()) # Get friendly name, or capitalize raw label

    summary_lines = [f"Top predicted accent: {friendly_accent_name} ({raw_label}) with {confidence*100:.2f}% confidence."]
    if len(predictions) > 1:
        summary_lines.append("Other possibilities include:")
        for pred in predictions[1:]:
            alt_label = ACCENT_LABEL_MAP.get(pred['label'], pred['label'].capitalize())
            summary_lines.append(f"  - {alt_label} ({pred['label']}): {pred['score']*100:.2f}%")

    return {
        "accent": friendly_accent_name,
        "confidence_percent": round(confidence * 100, 2),
        "raw_label": raw_label,
        "summary": " ".join(summary_lines)
    }

# --- Main block for testing this module directly ---
if __name__ == '__main__':
    print("Testing Accent Classification Module...")

    # I'll use a processed audio file from 'temp_processor_output' for testing.
    # This ensures it's in the correct 16kHz mono WAV format.
    processed_audio_dir = "temp_processor_output"
    test_audio_file_for_accent = None

    if os.path.exists(processed_audio_dir):
        for f in os.listdir(processed_audio_dir):
            # I'll pick a file that was likely the dummy stereo converted, as it's short
            # or any reasonably sized file.
            if f.endswith(".wav") and os.path.getsize(os.path.join(processed_audio_dir, f)) > 10 * 1024: # >10KB
                test_audio_file_for_accent = os.path.join(processed_audio_dir, f)
                print(f"Found test audio file for accent classification: {test_audio_file_for_accent}")
                break

    if not test_audio_file_for_accent:
        print(f"No suitable processed audio file found in '{processed_audio_dir}'.")
        print("Please run audio_processor.py successfully first to generate test files.")
    else:
        # --- Test Case 1: Classify accent from the found audio file ---
        print(f"\n--- Test Case 1: Classifying accent for {test_audio_file_for_accent} ---")
        result = classify_accent(test_audio_file_for_accent)

        if result:
            print("\n--- Classification Result ---")
            print(f"Detected Accent: {result.get('accent')}")
            print(f"Confidence: {result.get('confidence_percent')}%")
            print(f"Raw Model Label: {result.get('raw_label')}")
            print(f"Summary: {result.get('summary')}")
        else:
            print("[FAILURE] Accent classification did not return a result.")

    print("\n--- Accent Classification Module Testing Complete ---")