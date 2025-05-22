# diarizer.py
# I'm going to use this module to figure out who spoke when in an audio file.
# This uses the pyannote.audio library.

import os
import torch # pyannote.audio uses PyTorch models
from pyannote.audio import Pipeline
from pydub import AudioSegment # I'll use this to potentially segment audio later
import time # For timing how long diarization takes

# I'll define the model name here. It's good practice to use current, robust models.
# pyannote/speaker-diarization-3.1 is a good choice as of early 2024/2025.
# Ensure I've accepted user conditions on Hugging Face Hub for this model
# and for pyannote/segmentation-3.0
DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1"
# DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization@2.1" # Alternative if 3.1 has issues / for older examples

# I'll use a global variable for the pipeline so it's loaded only once.
# This is important because model loading can be slow.
DIARIZATION_PIPELINE = None
HF_AUTH_TOKEN = True # Set to True if you've logged in via huggingface-cli,
                     # or set to your actual token string if preferred (less secure for committed code)
                     # If True, it expects the token to be found by huggingface_hub.

def load_diarization_pipeline(use_auth_token=HF_AUTH_TOKEN):
    """
    Loads the pyannote.audio diarization pipeline.
    Caches it in the global DIARIZATION_PIPELINE variable.
    """
    global DIARIZATION_PIPELINE
    if DIARIZATION_PIPELINE is None:
        print(f"Loading speaker diarization pipeline: {DIARIZATION_MODEL_NAME}...")
        try:
            # Check if CUDA (GPU) is available, otherwise it will use CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # If HF_AUTH_TOKEN is True, it assumes token is available via CLI login or env var
            # If it's a string, it uses that string as the token.
            
         
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                DIARIZATION_MODEL_NAME,
                use_auth_token=use_auth_token # Important for gated models
            ).to(device)
            print("Diarization pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading diarization pipeline: {e}")
            print("Please ensure you have accepted the model's terms on Hugging Face Hub:")
            print(f"- {DIARIZATION_MODEL_NAME}")
            print("- pyannote/segmentation-3.0 (or the segmentation model used by your chosen diarization model)")
            print("And that you are logged in via 'huggingface-cli login' with a valid token.")
            DIARIZATION_PIPELINE = None # Ensure it's None if loading failed
            # We might want to raise the error to stop the app from proceeding without diarization
            raise
    return DIARIZATION_PIPELINE

def diarize_audio(audio_path: str, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None) -> list[dict] | None:
    """
    Performs speaker diarization on an audio file.

    Args:
        audio_path (str): Path to the input audio file (should be 16kHz mono WAV).
        num_speakers (int, optional): Exact number of speakers to detect.
        min_speakers (int, optional): Minimum number of speakers.
        max_speakers (int, optional): Maximum number of speakers.

    Returns:
        list[dict] | None: A list of speaker segments, where each segment is a dictionary
                            like {'speaker': 'SPEAKER_XX', 'start_ms': S, 'end_ms': E}.
                            Returns None if diarization fails.
    """
    pipeline = load_diarization_pipeline()
    if pipeline is None:
        print("Diarization pipeline not available. Cannot diarize.")
        return None

    if not os.path.exists(audio_path):
        print(f"Audio file not found for diarization: {audio_path}")
        return None

    print(f"Starting diarization for: {audio_path}")
    start_time = time.time()

    try:
        # The pipeline directly processes the audio file path.
        # It expects the audio to be in a format it can handle (WAV is good).
        # We can provide speaker count hints if known.
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        if params:
            print(f"Applying diarization with parameters: {params}")
            diarization_result = pipeline(audio_path, **params)
        else:
            print("Applying diarization with automatic speaker count detection.")
            diarization_result = pipeline(audio_path) # Let pipeline determine number of speakers

    except Exception as e:
        print(f"Error during diarization processing for {audio_path}: {e}")
        # This can happen due to various reasons, including issues with the audio file itself
        # or specific model constraints.
        return None

    end_time = time.time()
    print(f"Diarization completed in {end_time - start_time:.2f} seconds.")

    segments = []
    # The 'diarization_result' object is a pyannote.core.Annotation
    # It can be iterated to get turns and speakers.
    # Each turn has a start time, end time, and a speaker label.
    for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
        segment_info = {
            "speaker": speaker_label,  # e.g., SPEAKER_00, SPEAKER_01
            "start_ms": int(turn.start * 1000), # Convert seconds to milliseconds
            "end_ms": int(turn.end * 1000)      # Convert seconds to milliseconds
        }
        segments.append(segment_info)
        # print(f"  Found segment: Speaker {speaker_label} from {segment_info['start_ms']}ms to {segment_info['end_ms']}ms")

    if not segments:
        print("No speaker segments found by the diarization pipeline.")
        # This might happen for very short audio or audio with no detectable speech.
        return [] # Return empty list rather than None if pipeline ran but found nothing

    # I can optionally merge consecutive segments from the same speaker here if needed,
    # but pyannote's pipeline usually handles this well.
    # For now, I'll return the raw segments.
    return segments

# --- Main block for testing this module directly ---
if __name__ == '__main__':
    print("Testing Speaker Diarization Module...")

    # I'll try to use the processed audio from the previous phase for testing.
    # For example, the longer YouTube audio file.
    processed_youtube_audio_dir = os.path.join("temp_processor_output")
    test_audio_file = None

    # Find a processed .wav file to test with
    if os.path.exists(processed_youtube_audio_dir):
        for f in os.listdir(processed_youtube_audio_dir):
            if f.endswith(".wav") and os.path.getsize(os.path.join(processed_youtube_audio_dir, f)) > 100 * 1024: # Find a reasonably sized one
                test_audio_file = os.path.join(processed_youtube_audio_dir, f)
                print(f"Found test audio file from 'temp_processor_output': {test_audio_file}")
                break

    if not test_audio_file:
        print("No suitable processed audio file found in 'temp_processor_output'.")
        print("Please run audio_processor.py successfully first to generate test files,")
        print("or place a 16kHz mono WAV file in 'temp_processor_output' for testing.")
    else:
        # --- Test Case 1: Diarize with automatic speaker count ---
        print("\n--- Test Case 1: Automatic speaker count ---")
        segments_auto = diarize_audio(test_audio_file)
        if segments_auto is not None:
            print(f"[SUCCESS] Diarization (auto count) found {len(segments_auto)} segments.")
            # I'll print the first few segments for brevity
            for i, seg in enumerate(segments_auto[:5]):
                duration_ms = seg['end_ms'] - seg['start_ms']
                print(f"  Segment {i+1}: Speaker {seg['speaker']}, Start: {seg['start_ms']}ms, End: {seg['end_ms']}ms, Duration: {duration_ms/1000.0:.2f}s")
            if not segments_auto:
                 print("  No speech segments detected by diarization.")
        else:
            print("[FAILURE] Diarization (auto count) failed.")

        # --- Test Case 2: Diarize with a hint for number of speakers (e.g., if I know there are 2) ---
        # This depends on the content of test_audio_file. For a generic test, this might not always be accurate.
        # For now, I'll just demonstrate how to pass the parameter.
        # You would adjust num_speakers based on actual knowledge of the test audio.
        # For the BigBuckBunny or a typical YouTube monologue, num_speakers=1 is more likely.
        # If it were an interview, num_speakers=2 would be appropriate.
        # Let's assume for testing we are expecting 1 or 2 speakers.
        print("\n--- Test Case 2: Diarize with num_speakers hint (e.g., 2) ---")
        # segments_hinted = diarize_audio(test_audio_file, num_speakers=2) # Example for 2 speakers
        segments_hinted_min_max = diarize_audio(test_audio_file, min_speakers=1, max_speakers=2)

        if segments_hinted_min_max is not None:
            print(f"[SUCCESS] Diarization (min_speakers=1, max_speakers=2) found {len(segments_hinted_min_max)} segments.")
            unique_speakers = set(seg['speaker'] for seg in segments_hinted_min_max)
            print(f"  Unique speakers detected: {len(unique_speakers)} ({', '.join(sorted(list(unique_speakers)))})")
            for i, seg in enumerate(segments_hinted_min_max[:5]):
                duration_ms = seg['end_ms'] - seg['start_ms']
                print(f"  Segment {i+1}: Speaker {seg['speaker']}, Start: {seg['start_ms']}ms, End: {seg['end_ms']}ms, Duration: {duration_ms/1000.0:.2f}s")
        else:
            print("[FAILURE] Diarization (min_speakers=1, max_speakers=2) failed.")

    print("\n--- Speaker Diarization Module Testing Complete ---")