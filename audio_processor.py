# audio_processor.py
# I'll use this module to handle audio files, whether they come from an upload
# or from our audio_extractor. It will standardize them and perform some basic checks.

import os
from pydub import AudioSegment, exceptions as pydub_exceptions # I'm importing AudioSegment for audio manipulation
                                                              # and exceptions for error handling.
import uuid # For unique naming if I need to save processed files temporarily

# Constants for our desired audio format
TARGET_SAMPLE_RATE = 16000  # 16kHz, common for speech models
TARGET_CHANNELS = 1         # Mono
TARGET_FORMAT = "wav"       # We'll work with WAV files internally

MIN_AUDIO_DURATION_MS = 2000 # Minimum 2 seconds of audio to be considered for analysis
MAX_AUDIO_DURATION_MS = 20 * 60 * 1000 # Maximum 20 minutes, to prevent overly long processing

def process_uploaded_audio(
        input_audio_path: str,
        output_dir: str = "temp_processed_audio"
    ) -> tuple[str | None, str | None]:
    """
    Processes an uploaded audio file:
    1. Converts it to a standardized WAV format (16kHz, mono).
    2. Performs basic validation (duration, readability).
    3. Saves the processed file to a unique path in the output directory.

    Args:
        input_audio_path (str): Path to the originally uploaded audio file.
        output_dir (str): Directory to save the processed audio file.

    Returns:
        tuple[str | None, str | None]: (path_to_processed_wav, error_message)
                                       The path will be None if processing fails.
    """
    # I need to make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # I'll check if the input file actually exists first
    if not os.path.exists(input_audio_path):
        return None, f"Input audio file not found: {input_audio_path}"

    try:
        # I'll load the audio file using pydub. It can handle various formats.
        print(f"Loading audio from: {input_audio_path}")
        audio = AudioSegment.from_file(input_audio_path)
        print(f"Successfully loaded. Original duration: {len(audio) / 1000.0:.2f}s, Channels: {audio.channels}, Frame rate: {audio.frame_rate}Hz")

    except pydub_exceptions.CouldntDecodeError:
        return None, "Could not decode the audio file. It might be corrupted or in an unsupported format."
    except FileNotFoundError: # Should be caught by os.path.exists, but good to have
         return None, f"Input audio file not found (pydub): {input_audio_path}"
    except Exception as e: # Catch any other pydub loading errors
        return None, f"Error loading audio file with pydub: {str(e)}"

    # Basic Validation: Duration
    duration_ms = len(audio)
    if duration_ms < MIN_AUDIO_DURATION_MS:
        return None, f"Audio is too short ({duration_ms/1000.0:.2f}s). Minimum required: {MIN_AUDIO_DURATION_MS/1000.0:.2f}s."
    if duration_ms > MAX_AUDIO_DURATION_MS:
        return None, f"Audio is too long ({duration_ms/1000.0:.2f}s). Maximum allowed: {MAX_AUDIO_DURATION_MS/(60*1000.0):.0f} minutes."

    # Standardization: Convert to target sample rate and mono
    # I'll only change if necessary to avoid re-processing identical files.
    changed_format = False
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        print(f"Resampling from {audio.frame_rate}Hz to {TARGET_SAMPLE_RATE}Hz.")
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        changed_format = True

    if audio.channels != TARGET_CHANNELS:
        print(f"Converting channels from {audio.channels} to {TARGET_CHANNELS} (mono).")
        audio = audio.set_channels(TARGET_CHANNELS)
        changed_format = True

    # Saving the processed audio file
    # I want a unique name for this processed file.
    unique_filename = f"{str(uuid.uuid4())}.{TARGET_FORMAT}"
    processed_audio_path = os.path.join(output_dir, unique_filename)

    try:
        print(f"Exporting processed audio to: {processed_audio_path}")
        # Parameters: format is "wav". Pydub handles sample width, etc., automatically for WAV.
        audio.export(processed_audio_path, format=TARGET_FORMAT)
    except Exception as e:
        return None, f"Could not export processed audio file: {str(e)}"

    # Final check: ensure the file was created and is not empty
    if not os.path.exists(processed_audio_path) or os.path.getsize(processed_audio_path) == 0:
        return None, "Processed audio file was not created or is empty."

    print(f"Audio processed and saved to: {processed_audio_path}")
    return processed_audio_path, None # No error


# --- Main block for testing this module directly ---
if __name__ == '__main__':
    print("Testing Audio Processor Module...")

    # I need a sample audio file for testing.
    # For this test, I'll create a dummy MP3 file if it doesn't exist.
    # In a real scenario, I'd use actual sample MP3, M4A, or WAV files.

    # Let's assume I have a test MP3 file from the audio_extractor.py output (BigBuckBunny after increasing limit)
    # Or, I can create a small dummy one if needed for structural testing.
    # For a proper test, replace 'sample_test_audio.mp3' with a real audio file you have.

    # Example: Using an output from audio_extractor (make sure it ran successfully first)
    # This path assumes audio_extractor.py created it. Adjust if needed.
    # Ensure you've run audio_extractor.py with a successful extraction first for this to work.
    sample_audio_path_from_extractor = os.path.join(
        "temp_module_outputs", "mp4_test", # Assuming BigBuckBunny was extracted here
        # I'll need to find the actual UUID-based filename here or use a fixed test file
    )

    # To make this test self-contained, let's try to create a dummy MP3 if one specific path doesn't exist.
    # But for proper testing, you should use real audio files.
    # For now, I will use a file from the youtube extraction since that was successful
    # and is already in wav format. This will test the path where minimal processing is needed.

    # Find the first .wav file in the youtube_test directory
    youtube_test_dir = os.path.join("temp_module_outputs", "youtube_test")
    first_youtube_wav = None
    if os.path.exists(youtube_test_dir):
        for f in os.listdir(youtube_test_dir):
            if f.endswith(".wav"):
                first_youtube_wav = os.path.join(youtube_test_dir, f)
                break

    test_files_to_process = []
    if first_youtube_wav and os.path.exists(first_youtube_wav):
        print(f"Found extracted YouTube WAV for testing: {first_youtube_wav}")
        test_files_to_process.append({"path": first_youtube_wav, "desc": "Pre-processed WAV (from YouTube extraction)"})
    else:
        print("Could not find a WAV file from YouTube extraction test. "
              "Run audio_extractor.py successfully first or provide other test files.")

    # I should also test with a file that *needs* conversion, e.g., an MP3 or a WAV with different sample rate/channels.
    # For now, I'll create a dummy stereo, 44.1kHz WAV file for testing conversion.
    dummy_stereo_44100hz_wav = os.path.join("temp_module_outputs", "dummy_stereo_44100hz.wav")
    try:
        # Create a 2-second stereo, 44.1kHz dummy WAV for testing conversion
        # This uses a silent AudioSegment and exports it.
        silence_for_dummy = AudioSegment.silent(duration=2000, frame_rate=44100) # 2 seconds
        silence_for_dummy = silence_for_dummy.set_channels(2) # stereo
        os.makedirs("temp_module_outputs", exist_ok=True)
        silence_for_dummy.export(dummy_stereo_44100hz_wav, format="wav")
        print(f"Created dummy stereo 44.1kHz WAV for testing: {dummy_stereo_44100hz_wav}")
        test_files_to_process.append({"path": dummy_stereo_44100hz_wav, "desc": "Dummy Stereo 44.1kHz WAV"})
    except Exception as e:
        print(f"Could not create dummy stereo WAV for testing: {e}. Some conversion tests might be skipped.")


    # --- Test Case: Process various audio files ---
    output_processing_dir = "temp_processor_output" # Where processed files will go

    for test_item in test_files_to_process:
        test_audio_file = test_item["path"]
        description = test_item["desc"]
        print(f"\n--- Testing processing for: {description} ({test_audio_file}) ---")

        processed_path, error = process_uploaded_audio(test_audio_file, output_dir=output_processing_dir)

        if error:
            print(f"[FAILURE] Error processing {test_audio_file}: {error}")
        elif processed_path:
            print(f"[SUCCESS] Processed audio saved to: {processed_path}")
            print(f"File size: {os.path.getsize(processed_path) / 1024:.2f} KB")
            # I can verify its properties using pydub again
            try:
                loaded_processed = AudioSegment.from_file(processed_path)
                print(f"Verified properties: Duration={len(loaded_processed)/1000.0:.2f}s, Channels={loaded_processed.channels}, Rate={loaded_processed.frame_rate}Hz")
                assert loaded_processed.frame_rate == TARGET_SAMPLE_RATE
                assert loaded_processed.channels == TARGET_CHANNELS
                print("Verified: Audio is 16kHz Mono WAV.")
            except Exception as e:
                print(f"Error verifying processed file: {e}")
        else:
            print(f"[FAILURE] Unknown issue processing {test_audio_file}, no path and no error returned (should not happen).")

    # Test with a non-existent file
    print("\n--- Testing with a non-existent file ---")
    non_existent_file = "non_existent_audio_file.mp3"
    _, error_non_existent = process_uploaded_audio(non_existent_file, output_dir=output_processing_dir)
    if error_non_existent:
        print(f"[SUCCESS] Correctly handled non-existent file: {error_non_existent}")
    else:
        print("[FAILURE] Did not correctly handle non-existent file.")

    # Test with a very short file
    print("\n--- Testing with a very short audio file ---")
    short_audio_path = os.path.join("temp_module_outputs", "short_test.wav")
    try:
        short_audio = AudioSegment.silent(duration=MIN_AUDIO_DURATION_MS - 500) # 0.5s shorter than min
        short_audio.export(short_audio_path, format="wav")
        _, error_short = process_uploaded_audio(short_audio_path, output_dir=output_processing_dir)
        if error_short and "too short" in error_short.lower():
            print(f"[SUCCESS] Correctly handled short audio: {error_short}")
        else:
            print(f"[FAILURE] Did not correctly handle short audio. Error: {error_short}")
    except Exception as e:
        print(f"Could not create short audio for testing: {e}")


    print("\n--- Audio Processor Module Testing Complete ---")
    print(f"Check the '{os.path.join(os.getcwd(), output_processing_dir)}' directory for processed files.")