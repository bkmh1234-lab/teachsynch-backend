import os
from flask import Flask, request, jsonify
from pyannote.audio import Pipeline
import whisper
import torch

# ==============================================================================
# 1. SERVER AND MODEL SETUP
# ==============================================================================

app = Flask(__name__)

# --- Load Diarization Model ---
# Securely get the token from the environment variable set on Render
hf_token = os.environ.get('HUGGING_FACE_TOKEN')

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-crdova-ami",
        use_auth_token=hf_token
    )
    print("Diarization pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading diarization pipeline: {e}")
    pipeline = None

# --- Load Transcription Model ---
try:
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None


# ==============================================================================
# 2. API ENDPOINTS
# ==============================================================================

@app.route('/')
def index():
    return "Hello! Your audio processing server is running."


@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not pipeline or not whisper_model:
        return jsonify({"error": "A required model is not loaded. Check server logs."}), 500

    temp_path = "temp_audio_file.wav"
    audio_file.save(temp_path)

    try:
        # --- 1. Speaker Diarization ---
        diarization = pipeline(temp_path, num_speakers=2)

        # --- 2. Audio Transcription ---
        transcription = whisper_model.transcribe(temp_path)

        # --- 3. Combine Diarization and Transcription ---
        final_transcript = []
        for segment in transcription["segments"]:
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()

            speaker_times = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(segment_start, turn.start)
                overlap_end = min(segment_end, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration

            dominant_speaker = max(speaker_times, key=speaker_times.get) if speaker_times else "UNKNOWN"

            final_transcript.append({
                "start": segment_start,
                "end": segment_end,
                "text": segment_text,
                "speaker": dominant_speaker
            })

        # --- 4. Identify Teacher and assign roles ---
        speaker_talk_time = {}
        for seg in final_transcript:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            speaker_talk_time[speaker] = speaker_talk_time.get(speaker, 0) + duration

        teacher_label = max(speaker_talk_time, key=speaker_talk_time.get) if speaker_talk_time else "UNKNOWN"

        student_counter = 1
        speaker_map = {}
        for seg in final_transcript:
            original_speaker = seg['speaker']
            if original_speaker == teacher_label:
                seg['speaker_role'] = "Teacher"
            elif original_speaker != "UNKNOWN":
                if original_speaker not in speaker_map:
                    speaker_map[original_speaker] = f"Student {student_counter}"
                    student_counter += 1
                seg['speaker_role'] = speaker_map[original_speaker]
            else:
                seg['speaker_role'] = "Unknown"

        return jsonify(final_transcript)

    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
