# audio_extractor.py
import subprocess # For running command-line tools like yt-dlp
import os         # For path manipulations and file checks
import uuid       # For generating unique filenames
import json       # For parsing yt-dlp's JSON output

def extract_audio_from_url(video_url: str, output_dir: str = "temp_audio_extraction") -> str | None:
    """
    Downloads audio from a public video URL using yt-dlp, saves as a 16kHz mono WAV file.

    Args:
        video_url (str): The public URL of the video.
        output_dir (str): The directory to save the extracted audio file.

    Returns:
        str | None: The path to the extracted WAV file if successful, None otherwise.
                    The filename will be unique (UUID based).
    """
    print(f"Attempting to extract audio from URL: {video_url}")
    print(f"Output directory for extraction: {output_dir}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)# creates the output directory if it doesn't already exist

    # Generate a unique base filename (without extension)
    unique_id = str(uuid.uuid4())
    # yt-dlp -o option expects a template, it will add the .wav extension itself based on --audio-format
    output_filename_template = os.path.join(output_dir, f"{unique_id}")
    # The final expected output path after yt-dlp processes it
    expected_output_path = f"{output_filename_template}.wav"


    ## yt-dlp command arguments:
    # -x or --extract-audio: Extract audio.
    # --audio-format wav: Convert audio to WAV format.
    # --audio-quality 0: Best audio quality (for WAV, this means uncompressed).
    # -o TEMPLATE: Output filename template.
    # --no-playlist: If URL is a playlist, download only the video specified.
    # --socket-timeout 30: Timeout for network connections.
    # --max-filesize 180M: Limit download size to 180MB (can be adjusted).
    #    This helps prevent processing overly long files in this PoC.
    # --ppa "ffmpeg:-ac 1 -ar 16000": Post-processor arguments for ffmpeg.
    #    -ac 1: Set audio channels to 1 (mono).
    #    -ar 16000: Set audio sampling rate to 16000 Hz.
    # --print-json: Print metadata as JSON to stdout. This helps confirm details.
    # --no-warnings: Suppress yt-dlp warnings to keep output clean, errors will still show.
    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'wav',
        '--audio-quality', '0', # Best quality for WAV
        '-o', f"{output_filename_template}.%(ext)s", # yt-dlp appends correct extension
        '--no-playlist',
        '--socket-timeout', '60', # Increased timeout
        '--max-filesize', '180M', # Change limit if required
        '--postprocessor-args', 'ffmpeg:-ac 1 -ar 16000',
        # '--print-json', # We can capture this if needed for more robust error checking or info
        '--no-warnings',
        video_url
    ]

    print(f"Executing yt-dlp command: {' '.join(cmd)}")

    try:
        # Execute the command
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=300) # 5-minute timeout for download/conversion

        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')

        # print(f"yt-dlp stdout:\n{stdout_str}") # For debugging
        # print(f"yt-dlp stderr:\n{stderr_str}") # For debugging


        # Checking if the expected output file exists and is not empty
        # yt-dlp should have created output_filename_template.wav if successful
        actual_created_file = None
        # Sometimes yt-dlp might output to a slightly different name or extension before post-processing,
        # but with explicit WAV output and template, it should be predictable.
        # We look for a file starting with unique_id and ending with .wav
        for f_name in os.listdir(output_dir):
            if f_name.startswith(unique_id) and f_name.endswith(".wav"):
                potential_path = os.path.join(output_dir, f_name)
                if os.path.exists(potential_path) and os.path.getsize(potential_path) > 0:
                    actual_created_file = potential_path
                    if actual_created_file != expected_output_path:
                         os.rename(actual_created_file, expected_output_path)
                    break

        if actual_created_file is None or not (os.path.exists(expected_output_path) and os.path.getsize(expected_output_path) > 0) :
            # If the file wasn't created or is empty, something went wrong.
            # stderr from yt-dlp often contains useful error messages.
            error_message = f"Audio extraction failed. yt-dlp stderr: {stderr_str.strip()}"
            if not stderr_str.strip() and stdout_str.strip(): # Sometimes errors go to stdout
                 error_message = f"Audio extraction failed. yt-dlp stdout: {stdout_str.strip()}"
            print(error_message)
            raise RuntimeError(error_message)

        print(f"Audio extracted successfully to: {expected_output_path}")
        return expected_output_path

    except subprocess.TimeoutExpired:
        print("Audio extraction timed out (5 minutes). The video might be too long or the connection too slow.")
        # Clean up potentially incomplete file
        if os.path.exists(expected_output_path):
            try: os.remove(expected_output_path)
            except OSError: pass
        return None
    except RuntimeError as e: # Catch our specific RuntimeError
        print(f"RuntimeError during audio extraction: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(expected_output_path): # Check before removing
            try: os.remove(expected_output_path)
            except OSError: pass
        return None

# --- Main block for testing this module directly ---
if __name__ == '__main__':
    print("Testing Audio Extractor Module...")
    # This one is a test video from Google.
    test_url_mp4 = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    
    # Testing with a YouTube URL (yt-dlp handles YouTube well)
    test_url_youtube = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Short, well-known video
    
    # Testing with a Loom URL (replace with a REAL, PUBLIC Loom video URL for actual testing)
    # For now, I am using an example one, you can replace it with yours.
    #test_url_loom_placeholder = "https://www.loom.com/share/YOUR_PUBLIC_LOOM_VIDEO_ID_HERE"
    test_url_loom_placeholder = "https://www.loom.com/share/e00c8856f48049519ca6bece165b449a"


    # --- Test Case 1: MP4 URL ---
    print(f"\n--- Testing MP4 URL: {test_url_mp4} ---")
    output_directory_mp4 = os.path.join("temp_module_outputs", "mp4_test")
    extracted_file_mp4 = extract_audio_from_url(test_url_mp4, output_dir=output_directory_mp4)
    if extracted_file_mp4:
        print(f"[SUCCESS] MP4 audio saved to: {extracted_file_mp4}")
        print(f"File size: {os.path.getsize(extracted_file_mp4) / 1024:.2f} KB")
    else:
        print("[FAILURE] Failed to extract audio from MP4 URL.")

    # --- Test Case 2: YouTube URL ---
    print(f"\n--- Testing YouTube URL: {test_url_youtube} ---")
    output_directory_youtube = os.path.join("temp_module_outputs", "youtube_test")
    extracted_file_youtube = extract_audio_from_url(test_url_youtube, output_dir=output_directory_youtube)
    if extracted_file_youtube:
        print(f"[SUCCESS] YouTube audio saved to: {extracted_file_youtube}")
        print(f"File size: {os.path.getsize(extracted_file_youtube) / 1024:.2f} KB")
    else:
        print("[FAILURE] Failed to extract audio from YouTube URL.")

    # --- Test Case 3: Loom URL ---
    print(f"\n--- Testing Loom URL (Placeholder): {test_url_loom_placeholder} ---")
    print("NOTE: For Loom testing, replace the placeholder URL with a valid, public Loom video ID.")
    if "YOUR_PUBLIC_LOOM_VIDEO_ID_HERE" in test_url_loom_placeholder:
        print("Skipping Loom test as placeholder URL is used.")
    else:
        # replace with your URL
        output_directory_loom = os.path.join("temp_module_outputs", "loom_test")
        extracted_file_loom = extract_audio_from_url(test_url_loom_placeholder, output_dir=output_directory_loom)
        if extracted_file_loom:
            print(f"[SUCCESS] Loom audio saved to: {extracted_file_loom}")
            print(f"File size: {os.path.getsize(extracted_file_loom) / 1024:.2f} KB")
        else:
            print("[FAILURE] Failed to extract audio from Loom URL.")

    print("\n--- Audio Extractor Module Testing Complete ---")
    print(f"Check the '{os.path.join(os.getcwd(), 'temp_module_outputs')}' directory for extracted files (if any).")
    print("Remember to manually delete this directory after checking if you wish.")