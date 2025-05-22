# app.py
# This is the main Streamlit application file for AccentGuard Pro.
# I'll orchestrate the UI and the backend modules from here.

import streamlit as st
import os
import shutil # For cleaning up temporary directories
import time   # For simulating delays or measuring performance
import traceback # For printing detailed error tracebacks during debugging

# I need to import my custom modules AND specific variables/classes
from audio_extractor import extract_audio_from_url
from audio_processor import process_uploaded_audio, TARGET_FORMAT
from diarizer import diarize_audio, load_diarization_pipeline
from accent_analyzer_os import classify_accent, load_accent_classifier_pipeline
from language_identifier import identify_language, load_lid_pipeline # << IMPORTED LID
from pydub import AudioSegment

# --- Application Constants and Configuration ---
TEMP_PRIMARY_AUDIO_DIR = "temp_app_audio" # For audio from URL or direct upload before diarization
TEMP_SEGMENTS_DIR = "temp_speaker_segments" # For storing individual speaker segments
LID_ENGLISH_CONFIDENCE_THRESHOLD = 50 # Minimum confidence (percent) to consider a segment English

# --- Helper Functions ---
def cleanup_temp_directories():
    """Removes temporary directories used by the app."""
    print("Cleaning up temporary directories...")
    # App-specific temp directories
    for app_temp_dir in [TEMP_PRIMARY_AUDIO_DIR, TEMP_SEGMENTS_DIR, "temp_url_extraction"]:
        if os.path.exists(app_temp_dir):
            try:
                shutil.rmtree(app_temp_dir)
                print(f"Removed: {app_temp_dir}")
            except Exception as e:
                print(f"Could not remove app temp dir {app_temp_dir}: {e}")

    # Module-specific test output directories (if they exist from direct module runs)
    module_test_dirs = ["temp_audio_extraction", "temp_processor_output", "temp_module_outputs"]
    for m_dir in module_test_dirs:
        if os.path.exists(m_dir):
            try:
                shutil.rmtree(m_dir)
                print(f"Removed module test output dir: {m_dir}")
            except Exception as e:
                print(f"Could not remove module test output dir {m_dir}: {e}")

def create_temp_directories():
    """Creates necessary temporary directories."""
    os.makedirs(TEMP_PRIMARY_AUDIO_DIR, exist_ok=True)
    os.makedirs(TEMP_SEGMENTS_DIR, exist_ok=True)

# --- Load AI Models at App Startup (using st.cache_resource) ---
@st.cache_resource
def get_diarization_pipeline_cached():
    """Loads and caches the diarization pipeline."""
    print("Attempting to load diarization pipeline (cached)...")
    try:
        return load_diarization_pipeline(use_auth_token=True) # Assumes CLI login
    except Exception as e:
        st.error(f"Fatal Error: Could not load Speaker Diarization model: {e}")
        st.error("Ensure model terms are accepted on Hugging Face and `huggingface-cli login` was successful.")
        return None

@st.cache_resource
def get_accent_classifier_pipeline_cached():
    """Loads and caches the accent classification pipeline."""
    print("Attempting to load accent classification pipeline (cached)...")
    try:
        return load_accent_classifier_pipeline()
    except Exception as e:
        st.error(f"Fatal Error: Could not load Accent Classification model: {e}")
        return None

@st.cache_resource
def get_lid_pipeline_cached():
    """Loads and caches the language identification pipeline."""
    print("Attempting to load language identification pipeline (cached)...")
    try:
        return load_lid_pipeline()
    except Exception as e:
        st.error(f"Fatal Error: Could not load Language Identification model: {e}")
        return None

# --- Main Application UI and Logic ---
def main():
    st.set_page_config(page_title="AccentGuard Pro", page_icon="🗣️", layout="wide")
    st.title(" AccentGuard Pro: English Accent Analyzer 🕵️‍♀️🗣️")
    st.markdown("""
        This tool analyzes spoken English accents from video URLs or uploaded audio files.
        It uses speaker diarization to identify speakers and language identification before classifying English accents.
    """)

    # Load models
    diarization_pipeline = get_diarization_pipeline_cached()
    accent_classifier_pipeline = get_accent_classifier_pipeline_cached()
    lid_pipeline = get_lid_pipeline_cached()

    if not all([diarization_pipeline, accent_classifier_pipeline, lid_pipeline]):
        st.warning("One or more AI models could not be loaded. Please check console logs. Functionality will be limited.")
        # Consider st.stop() if critical models fail

    # Sidebar for input
    st.sidebar.header("🎙️ Audio Input")
    input_method = st.sidebar.radio("Choose input method:", ("YouTube/Video URL", "Upload Audio File"), key="input_method_radio")

    # Initialize session state variables if they don't exist
    if 'processed_audio_path' not in st.session_state:
        st.session_state.processed_audio_path = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'run_analysis' not in st.session_state: # Flag to trigger analysis
        st.session_state.run_analysis = False


    processed_audio_path_for_current_run = None

    if input_method == "YouTube/Video URL":
        video_url = st.sidebar.text_input("Enter Video URL:", key="video_url_input")
        if st.sidebar.button("Process URL", key="process_url_button", type="primary", disabled=not video_url):
            if video_url:
                create_temp_directories()
                with st.spinner("Extracting audio from URL..."):
                    # Using a dedicated temp dir for URL extraction before it's processed further
                    url_extraction_temp_dir = "temp_url_extraction_app"
                    os.makedirs(url_extraction_temp_dir, exist_ok=True)
                    raw_extracted_audio_path = extract_audio_from_url(video_url, output_dir=url_extraction_temp_dir)

                if raw_extracted_audio_path:
                    st.success(f"Audio extracted from URL: {os.path.basename(raw_extracted_audio_path)}")
                    with st.spinner("Standardizing extracted audio..."):
                        final_audio_path, error_msg = process_uploaded_audio(
                            raw_extracted_audio_path, output_dir=TEMP_PRIMARY_AUDIO_DIR
                        )
                    if error_msg:
                        st.error(f"Error standardizing extracted audio: {error_msg}")
                    else:
                        processed_audio_path_for_current_run = final_audio_path
                        st.session_state.processed_audio_path = final_audio_path
                        st.session_state.analysis_results = None # Clear old results
                        st.session_state.run_analysis = True # Flag to run analysis
                    if os.path.exists(url_extraction_temp_dir): # Clean up temp URL extraction dir
                        shutil.rmtree(url_extraction_temp_dir)
                else:
                    st.error("Failed to extract audio from URL.")
            else:
                st.sidebar.warning("Please enter a URL.")

    elif input_method == "Upload Audio File":
        uploaded_file = st.sidebar.file_uploader("Choose audio file:", type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac'], key="file_uploader")
        if st.sidebar.button("Process Uploaded File", key="process_upload_button", type="primary", disabled=not uploaded_file):
            if uploaded_file:
                create_temp_directories()
                temp_upload_path = os.path.join(TEMP_PRIMARY_AUDIO_DIR, f"upload_{uploaded_file.name}")
                with open(temp_upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info(f"File '{uploaded_file.name}' uploaded. Processing...")

                with st.spinner("Processing and standardizing uploaded audio..."):
                    final_audio_path, error_msg = process_uploaded_audio(
                        temp_upload_path, output_dir=TEMP_PRIMARY_AUDIO_DIR
                    )
                if error_msg:
                    st.error(f"Error processing uploaded file: {error_msg}")
                else:
                    processed_audio_path_for_current_run = final_audio_path
                    st.session_state.processed_audio_path = final_audio_path
                    st.session_state.analysis_results = None # Clear old results
                    st.session_state.run_analysis = True # Flag to run analysis
                if os.path.exists(temp_upload_path): # Clean up initial upload
                    os.remove(temp_upload_path)
            else:
                st.sidebar.warning("Please upload a file.")

    # Use path from current run if available, else from session state (for reruns without new processing)
    final_processed_audio_path = processed_audio_path_for_current_run or st.session_state.processed_audio_path

    # --- Perform Analysis ---
    if final_processed_audio_path and os.path.exists(final_processed_audio_path) and st.session_state.run_analysis:
        st.header("🎧 Processed Audio Sample")
        try:
            with open(final_processed_audio_path, 'rb') as audio_file_obj:
                st.audio(audio_file_obj.read(), format=f'audio/{TARGET_FORMAT}')
        except Exception as e:
            st.warning(f"Could not display audio player for the main file: {e}")

        if not all([diarization_pipeline, accent_classifier_pipeline, lid_pipeline]):
            st.error("AI models not loaded. Cannot perform full analysis.")
        else:
            with st.spinner("Performing speaker diarization... (This can take time for long audio)"):
                speaker_segments = diarize_audio(final_processed_audio_path)

            analysis_results_list = []
            if speaker_segments is None:
                st.error("Speaker diarization failed.")
            elif not speaker_segments:
                st.warning("No speaker segments detected by diarization.")
            else:
                st.success(f"Diarization found {len(speaker_segments)} speech segments from {len(set(s['speaker'] for s in speaker_segments))} unique speaker(s).")
                
                try:
                    main_audio_for_clipping = AudioSegment.from_file(final_processed_audio_path, format=TARGET_FORMAT)
                except Exception as e:
                    st.error(f"Could not load main audio for clipping segments: {e}")
                    main_audio_for_clipping = None

                if main_audio_for_clipping:
                    st.subheader("🗣️ Language & Accent Analysis per Segment:")
                    segment_progress = st.progress(0)
                    for i, seg_info in enumerate(speaker_segments):
                        speaker = seg_info['speaker']
                        start_ms, end_ms = seg_info['start_ms'], seg_info['end_ms']
                        duration_ms = end_ms - start_ms
                        
                        current_segment_data = {
                            "speaker": speaker, "start_ms": start_ms, "end_ms": end_ms,
                            "duration_ms": duration_ms, "segment_audio_path": None,
                            "language_result": None, "accent_result": None
                        }

                        if duration_ms < 1500: # Min duration for analysis
                            current_segment_data["language_result"] = {"language_name": "N/A", "summary": "Segment too short for analysis."}
                            current_segment_data["accent_result"] = {"accent": "N/A", "summary": "Segment too short."}
                        else:
                            seg_file_path = os.path.join(TEMP_SEGMENTS_DIR, f"{speaker}_seg_{i}_{start_ms}_{end_ms}.{TARGET_FORMAT}")
                            try:
                                clipped_audio = main_audio_for_clipping[start_ms:end_ms]
                                clipped_audio.export(seg_file_path, format=TARGET_FORMAT)
                                current_segment_data["segment_audio_path"] = seg_file_path

                                with st.spinner(f"Segment {i+1}: Identifying language ({speaker})..."):
                                    lang_res = identify_language(seg_file_path, top_k=1)
                                current_segment_data["language_result"] = lang_res

                                if lang_res and lang_res.get("language_code") == "eng" and lang_res.get("confidence_percent", 0) >= LID_ENGLISH_CONFIDENCE_THRESHOLD:
                                    with st.spinner(f"Segment {i+1}: Classifying English accent ({speaker})..."):
                                        accent_res = classify_accent(seg_file_path)
                                    current_segment_data["accent_result"] = accent_res
                                else:
                                    current_segment_data["accent_result"] = {"accent": "N/A", "summary": f"Not classified as English or low confidence (Lang: {lang_res.get('language_name', 'Unknown') if lang_res else 'LID Error'})."}
                            except Exception as e:
                                print(f"Error processing segment {i+1}: {e}")
                                current_segment_data["language_result"] = {"language_name": "Error", "summary": str(e)}
                                current_segment_data["accent_result"] = {"accent": "Error", "summary": "Processing error."}
                        
                        analysis_results_list.append(current_segment_data)
                        segment_progress.progress((i + 1) / len(speaker_segments))
                    segment_progress.empty()
            st.session_state.analysis_results = analysis_results_list
        st.session_state.run_analysis = False # Reset flag after analysis is done or attempted

    # --- Display Results ---
    if st.session_state.analysis_results is not None:
        results_to_display = st.session_state.analysis_results
        if not results_to_display and st.session_state.processed_audio_path: # Analysis run but no results
            st.info("Analysis complete. No actionable segments found or processed.")
        elif results_to_display:
            st.header("📊 Analysis Results")
            for i, res_item in enumerate(results_to_display):
                speaker_id = res_item['speaker']
                start_s = res_item['start_ms'] / 1000.0
                end_s = res_item['end_ms'] / 1000.0
                duration_s = res_item['duration_ms'] / 1000.0
                
                lang_res = res_item.get("language_result", {})
                accent_res = res_item.get("accent_result", {})

                lang_name = lang_res.get('language_name', 'N/A')
                lang_conf = lang_res.get('confidence_percent', 0)
                
                accent_name = accent_res.get('accent', 'N/A')
                accent_conf = accent_res.get('confidence_percent', 0)

                expander_title = f"🗣️ Segment {i+1}: {speaker_id} ({start_s:.2f}s - {end_s:.2f}s, Dur: {duration_s:.2f}s)"
                if lang_name != 'N/A' and lang_name != 'Error':
                    expander_title += f" - Language: {lang_name} ({lang_conf:.1f}%)"
                if accent_name != 'N/A' and accent_name != 'Error' and (not lang_name or lang_name == "English"): # Show accent if relevant
                     expander_title += f" - Accent: {accent_name} ({accent_conf:.1f}%)"


                with st.expander(expander_title, expanded=i < 3): # Expand first few
                    st.markdown(f"**Language:** {lang_name} (`{lang_res.get('language_code', 'N/A')}`) - Confidence: {lang_conf:.2f}%")
                    if lang_res.get('summary') and lang_name != 'N/A': st.caption(f"LID Summary: {lang_res.get('summary')}")

                    if lang_res.get("language_code") == "eng" and lang_conf >= LID_ENGLISH_CONFIDENCE_THRESHOLD:
                        st.markdown(f"**English Accent:** {accent_name} - Confidence: {accent_conf:.2f}%")
                        if accent_res.get('summary'): st.info(f"Accent Model: {accent_res.get('summary')}")
                    elif accent_name != 'N/A': # If accent was processed for some reason or there's an error message
                        st.markdown(f"**English Accent:** {accent_name}")
                        if accent_res.get('summary'): st.warning(f"Note: {accent_res.get('summary')}")


                    seg_audio_path = res_item.get("segment_audio_path")
                    if seg_audio_path and os.path.exists(seg_audio_path):
                        try:
                            with open(seg_audio_path, 'rb') as saf:
                                st.audio(saf.read(), format=f'audio/{TARGET_FORMAT}')
                        except Exception as e:
                            st.caption(f"Could not play segment audio: {e}")
    
    # --- Sidebar Bottom ---   
    st.sidebar.markdown("---") # Separator
    
    with st.sidebar.expander("ℹ️ About AccentGuard Pro & Model Info", expanded=False):
        st.markdown(
            "This tool performs automated analysis of spoken audio using several AI models."
        )
        st.subheader("Core AI Models Used:")
        st.markdown(
            """
            * **Speaker Diarization:** `pyannote/speaker-diarization-3.1`
                * Identifies *who* spoke *when*.
            * **Language Identification (LID):** `facebook/mms-lid-126`
                * Detects the language spoken in each segment. Covers 126 languages.
            * **English Accent Classification:** `dima806/english_accents_classification`
                * Classifies English accents if the LID model identifies the segment as English.
                * Primary accents covered: American (US), British (England), Indian, Australian, Canadian.
            """
        )
        st.subheader("Important Considerations & Disclaimers:")
        st.warning(
            """
            * **AI is Not Perfect:** Results are based on probabilistic models and may not always be 100% accurate.
            * **Accent Model Limitations:** The English accent classifier is trained on specific accents. Predictions for other English accents or highly nuanced speech may vary in accuracy or show mixed results.
            * **Audio Quality Matters:** Background noise, microphone quality, and unclear speech can significantly impact the accuracy of all models.
            * **Use as a Guide:** This tool should be used as one of several factors in any evaluation process, not as the sole determinant.
            """
        )
        st.caption("Version 1.0 | AccentGuard Pro")


    st.sidebar.markdown("---") # Separator before cleanup button
    if st.sidebar.button("🧹 Clear Temporary Files & Reset", key="cleanup_button"):
        cleanup_temp_directories()
        # Clear all relevant session state keys
        for key in list(st.session_state.keys()):
            if key not in ['input_method_radio', 'video_url_input', 'file_uploader']: # Preserve input widgets state
                del st.session_state[key]
        st.session_state.run_analysis = False # Explicitly reset this flag
        st.success("Temporary files and session state cleared.")
        st.rerun()

if __name__ == "__main__":
    main()