# language_identifier.py
# I will use this module to identify the language of an audio segment.

import os
from transformers import pipeline
import torch
import time
import traceback # For more detailed error printing during testing

# Using facebook/mms-lid-126, a robust LID model
LID_MODEL_NAME = "facebook/mms-lid-126"

# I'll use a global variable for the pipeline to load it only once.
LID_PIPELINE = None

# This map will be populated or verified based on the model's actual labels.
# MMS models typically output ISO 639-3 codes (e.g., 'eng', 'hin', 'kan').
LANGUAGE_CODE_MAP = {
    "eng": "English",
    "hin": "Hindi",
    "kan": "Kannada",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "rus": "Russian",
    "zho": "Chinese", # Example, mms-lid-126 uses 'zho_Hans' or 'zho_Hant' typically
    "jpn": "Japanese",
    # This map will likely be updated by the load_lid_pipeline function
}

def load_lid_pipeline():
    """
    Loads the Hugging Face audio-classification pipeline for Language ID.
    Caches it in the global LID_PIPELINE variable.
    Also attempts to update LANGUAGE_CODE_MAP with actual model labels.
    """
    global LID_PIPELINE
    global LANGUAGE_CODE_MAP # Allow modification of the global map
    if LID_PIPELINE is None:
        print(f"Loading Language Identification pipeline: {LID_MODEL_NAME}...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device for Language ID: {device}")

            LID_PIPELINE = pipeline(
                "audio-classification",
                model=LID_MODEL_NAME,
                device=device,
                use_auth_token=True # Good practice if logged in, though this model is public
            )
            
            print("Language Identification pipeline loaded successfully.")

            # Dynamically update LANGUAGE_CODE_MAP from the loaded model's config
            if hasattr(LID_PIPELINE.model.config, 'id2label'):
                print("\nModel labels (id2label from config):")
                # Temporarily store new mappings to avoid modifying dict while iterating if necessary
                new_mappings = {}
                for i, label_code in sorted(LID_PIPELINE.model.config.id2label.items()):
                    print(f"  ID {i}: {label_code}")
                    # If the code isn't already mapped to a friendly name,
                    # we can add it or decide on a default naming convention.
                    # For now, just ensure the code itself is in the map if not already there with a better name.
                    if label_code not in LANGUAGE_CODE_MAP:
                        new_mappings[label_code] = label_code.capitalize() # Default: Capitalize the code
                    elif label_code not in ['eng', 'hin', 'kan', 'spa', 'fra', 'deu', 'rus', 'zho', 'jpn']: # If it's a code we already defined better
                        if LANGUAGE_CODE_MAP[label_code] == label_code: # and if current map is just the code itself
                             new_mappings[label_code] = label_code.capitalize()


                if new_mappings:
                    LANGUAGE_CODE_MAP.update(new_mappings)
                    print(f"\nUpdated LANGUAGE_CODE_MAP based on model (new entries or capitalized codes): {LANGUAGE_CODE_MAP}")
                else:
                    print("\nLANGUAGE_CODE_MAP already covers known model labels or uses specific friendly names.")

            else:
                print("Warning: Could not retrieve id2label from model config. LANGUAGE_CODE_MAP may be incomplete.")

        except Exception as e:
            print(f"Error loading Language ID pipeline: {e}")
            LID_PIPELINE = None # Ensure it's None if loading failed
            raise # Re-raise so the calling code knows it failed
    return LID_PIPELINE

def identify_language(audio_path: str, top_k: int = 1) -> dict | None:
    """
    Identifies the language(s) spoken in an audio file.
    """
    pipeline_instance = load_lid_pipeline() # Ensures pipeline is loaded
    if pipeline_instance is None:
        return {"language_code": "err", "language_name": "Error", "confidence_percent": 0, "summary": "LID model not loaded."}

    if not os.path.exists(audio_path):
        return {"language_code": "err", "language_name": "Error", "confidence_percent": 0, "summary": f"Audio file not found: {audio_path}"}
    if os.path.getsize(audio_path) == 0:
         return {"language_code": "err", "language_name": "Error", "confidence_percent": 0, "summary": "Audio file is empty."}

    print(f"Starting language identification for: {audio_path}")
    start_time = time.time()

    try:
        # Forcing a 16kHz sample rate as MMS models are often trained on it.
        # The pipeline should handle this, but we can be explicit if issues arise.
        predictions = pipeline_instance(audio_path, top_k=top_k, framework="pt") # framework="pt" for PyTorch
    except Exception as e:
        print(f"Error during language identification for {audio_path}: {e}")
        # traceback.print_exc() # Uncomment for more detailed error during debugging
        return {"language_code": "err", "language_name": "Error", "confidence_percent": 0, "summary": f"Error during LID model inference: {str(e)}"}

    end_time = time.time()
    print(f"Language identification completed in {end_time - start_time:.2f} seconds.")

    if not predictions:
        return {"language_code": "unk", "language_name": "Unknown", "confidence_percent": 0, "summary": "LID Model returned no predictions."}

    top_prediction = predictions[0]
    # The 'label' from facebook/mms-lid-126 should be an ISO 639-3 code like 'eng', 'kan', 'hin'
    raw_model_label_code = top_prediction['label']
    confidence = top_prediction['score']
    
    # Use the raw model label code for our internal logic (e.g., "eng" check)
    simple_lang_code = raw_model_label_code
    
    # Get a more user-friendly name from our map
    lang_name_from_map = LANGUAGE_CODE_MAP.get(simple_lang_code, simple_lang_code.capitalize()) # Fallback to capitalized code

    summary_parts = [f"Top predicted language: {lang_name_from_map} (Code: {simple_lang_code}) with {confidence*100:.2f}% confidence."]
    if top_k > 1 and len(predictions) > 1:
        summary_parts.append("Other possibilities:")
        for pred in predictions[1:]:
            pred_code = pred['label']
            alt_name = LANGUAGE_CODE_MAP.get(pred_code, pred_code.capitalize())
            summary_parts.append(f"  - {alt_name} (Code: {pred_code}): {pred['score']*100:.2f}%")
    
    return {
        "language_code": simple_lang_code, # This should be 'eng', 'kan', etc.
        "language_name": lang_name_from_map,
        "raw_model_label": raw_model_label_code, # Redundant if same as simple_lang_code, but clear
        "confidence_percent": round(confidence * 100, 2),
        "all_predictions": predictions, # Store all raw predictions from the model
        "summary": " ".join(summary_parts)
    }

# --- Main block for testing this module directly ---
if __name__ == '__main__':
    print("Testing Language Identification Module...")
    # Ensure HF_HOME is set correctly if running standalone and model needs to be downloaded
    print(f"HF_HOME is set to: {os.environ.get('HF_HOME')}")
    
    processed_audio_dir = "temp_processor_output"
    test_audio_file_for_lid = None

    # Attempt to find a suitable test file
    if os.path.exists(processed_audio_dir):
        # Try to find a file that might not be predominantly English to test LID
        # This is heuristic, you might need to place a specific test file.
        found_files = [os.path.join(processed_audio_dir, f) for f in os.listdir(processed_audio_dir) if f.endswith(".wav") and os.path.getsize(os.path.join(processed_audio_dir, f)) > 50 * 1024]
        if found_files:
            test_audio_file_for_lid = found_files[0] # Take the first one found
            # If you have multiple, you might pick one specifically, e.g., the output from the Kannada video.
            # For now, just using the first available one from previous processing.
            print(f"Found test audio file for Language ID: {test_audio_file_for_lid}")
    
    if not test_audio_file_for_lid:
        print(f"No suitable processed audio file found in '{processed_audio_dir}'.")
        print("Please run audio_processor.py successfully first (e.g., by running app.py with a URL)"
              " to generate test files in 'temp_processor_output', or place a test audio file there.")
    else:
        print(f"\n--- Test Case: Identifying language for {os.path.basename(test_audio_file_for_lid)} (top_k=3) ---")
        try:
            # This will trigger model download if not already cached.
            # Ensure you are logged in via `huggingface-cli login`
            load_lid_pipeline() 
            result = identify_language(test_audio_file_for_lid, top_k=3)

            if result and result.get("language_code") != "err":
                print("\n--- Language ID Result ---")
                print(f"Simplified Language Code (for app logic): {result.get('language_code')}")
                print(f"Detected Language Name: {result.get('language_name')}")
                print(f"Raw Model Label from pipeline: {result.get('raw_model_label')}")
                print(f"Confidence: {result.get('confidence_percent')}%")
                print(f"Summary: {result.get('summary')}")
                
                print("\nAll predictions from model pipeline:")
                if result.get('all_predictions'):
                    for pred_idx, pred_info in enumerate(result['all_predictions']):
                        pred_code = pred_info['label']
                        pred_name = LANGUAGE_CODE_MAP.get(pred_code, pred_code.capitalize())
                        print(f"  {pred_idx+1}. Label Code: {pred_code} (Name: {pred_name}), Score: {pred_info['score']:.4f}")
                else:
                    print("  No 'all_predictions' data in result.")
            elif result: # Error case from identify_language
                print(f"[FAILURE] Language identification returned an error: {result.get('summary')}")
            else:
                print("[FAILURE] Language identification did not return any structured result.")
        except Exception as e:
            print(f"An error occurred during standalone LID test: {e}")
            traceback.print_exc()

    print("\n--- Language Identification Module Testing Complete ---")