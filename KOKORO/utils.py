from .kokoro import normalize_text,phonemize,generate
import re
import librosa
import os
import uuid
from pydub.silence import split_on_silence
from pydub import AudioSegment
import wave
import numpy as np
import torch


def create_audio_dir():
    """Creates the 'kokoro_audio' directory in the root folder if it doesn't exist."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_dir = os.path.join(root_dir, "kokoro_audio")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    else:
        print(f"Directory already exists: {audio_dir}")
    return audio_dir

temp_folder = create_audio_dir()


debug=False
def resplit_strings(arr):
    # Handle edge cases
    if not arr:
        return '', ''
    if len(arr) == 1:
        return arr[0], ''
    # Try each possible split point
    min_diff = float('inf')
    best_split = 0
    # Calculate lengths when joined with spaces
    lengths = [len(s) for s in arr]
    spaces = len(arr) - 1  # Total spaces needed
    # Try each split point
    left_len = 0
    right_len = sum(lengths) + spaces
    for i in range(1, len(arr)):
        # Add current word and space to left side
        left_len += lengths[i-1] + (1 if i > 1 else 0)
        # Remove current word and space from right side
        right_len -= lengths[i-1] + 1
        diff = abs(left_len - right_len)
        if diff < min_diff:
            min_diff = diff
            best_split = i
    # Join the strings with the best split point
    return ' '.join(arr[:best_split]), ' '.join(arr[best_split:])

def recursive_split(text, voice):
    if not text:
        return []
    tokens = phonemize(text, voice, norm=False)
    if len(tokens) < 511:
        return [(text, tokens, len(tokens))] if tokens else []
    if ' ' not in text:
        return []
    for punctuation in ['!.?…', ':;', ',—']:
        splits = re.split(f'(?:(?<=[{punctuation}])|(?<=[{punctuation}]["\'»])|(?<=[{punctuation}]["\'»]["\'»])) ', text)
        if len(splits) > 1:
            break
        else:
            splits = None
    splits = splits or text.split(' ')
    a, b = resplit_strings(splits)
    return recursive_split(a, voice) + recursive_split(b, voice)

def segment_and_tokenize(text, voice, skip_square_brackets=True, newline_split=2):    
    if skip_square_brackets:
        text = re.sub(r'\[.*?\]', '', text)
    texts = [t.strip() for t in re.split('\n{'+str(newline_split)+',}', normalize_text(text))] if newline_split > 0 else [normalize_text(text)]
    segments = [row for t in texts for row in recursive_split(t, voice)]
    return [(i, *row) for i, row in enumerate(segments)]


def large_text(text,VOICE_NAME):
    if len(text) <= 500:
        return [(0, text, len(text))]
    else:
        result=segment_and_tokenize(text, VOICE_NAME[0])
        filtered_result = [(row[0], row[1], row[3]) for row in result]
        return filtered_result
    

def clamp_speed(speed):
    if not isinstance(speed, float) and not isinstance(speed, int):
        return 1
    elif speed < 0.5:
        # return 0.5
        return speed
    elif speed > 2:
        return 2
    return speed

def clamp_trim(trim):
    if not isinstance(trim, float) and not isinstance(trim, int):
        return 0.5
    elif trim <= 0:
        return 0
    elif trim > 1:
        return 0.5
    return trim

def trim_if_needed(out, trim):
    if not trim:
        return out
    a, b = librosa.effects.trim(out, top_db=30)[1]
    a = int(a*trim)
    b = int(len(out)-(len(out)-b)*trim)
    return out[a:b]    

#Above code copied from https://huggingface.co/spaces/hexgrad/Kokoro-TTS/blob/main/app.py

def get_random_file_name(output_file=""):
    global temp_folder
    if output_file=="":
        random_id = str(uuid.uuid4())[:8]
        output_file = f"{temp_folder}/{random_id}.wav"
        return output_file
    # Ensure temp_folder exists 
    if not os.path.exists(output_file):
        return output_file   
    try:
        if output_file and os.path.exists(output_file):
            os.remove(output_file)  # Try to remove the file if it exists
            return output_file      # Return the same name if the file was successfully removed
    except Exception as e:
        # print(f"Error removing file {output_file}: {e}")
        random_id = str(uuid.uuid4())[:8]
        output_file = f"{temp_folder}/{random_id}.wav"
        return output_file
    

def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence) 
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path

# import simpleaudio as sa
# def play_audio(filename):
#     wave_obj = sa.WaveObject.from_wave_file(filename)
#     play_obj = wave_obj.play()
#     play_obj.wait_done()


import re

def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "-": " ",  # Replace hyphen with space
        ":": ",",  # Replace colon with comma
        "**": " ", # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous symbols and pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and map symbols
        r'[\U0001F700-\U0001F77F]|'  # Alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric shapes extended
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental symbols and pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and pictographs extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U0001F1E0-\U0001F1FF]'   # Flags (iOS)
        r'', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text

#copied from F5TTS 😁
import re
def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "af"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                text=clean_text(text)
                segments.append({"voice_name": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments

def podcast(MODEL, device, gen_text, speed=1.0, trim=0.5, pad_between_segments=0, remove_silence=True, minimum_silence=50):
    segments = parse_speechtypes_text(gen_text)
    speed = clamp_speed(speed)
    trim = clamp_trim(trim)
    silence_duration = clamp_trim(pad_between_segments)
    # output_file = get_random_file_name(output_file)
    sample_rate = 24000  # Sample rate of the audio

    # Create a silent audio segment in float32
    silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
    if len(segments)>=1:
        first_line_text=segments[0]["text"]
        output_file=tts_file_name(first_line_text)
    else:
        output_file = get_random_file_name("")
    
    output_file = output_file.replace('\n', '').replace('\r', '')
    # Open a WAV file for writing
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)

        for idx, segment in enumerate(segments):  # Added index `idx` to track position
            voice_name = segment["voice_name"]
            text = segment["text"]
            voice_pack_path = f"./KOKORO/voices/{voice_name}.pt"
            VOICEPACK = torch.load(voice_pack_path, weights_only=True).to(device)

            # Generate audio for the segment
            audio, out_ps = generate(MODEL, text, VOICEPACK, lang=voice_name[0], speed=speed)
            audio = trim_if_needed(audio, trim)

            # Scale audio from float32 to int16
            audio = (audio * 32767).astype(np.int16)

            # Write the audio segment to the WAV file
            wav_file.writeframes(audio.tobytes())

            # Add silence between segments, except after the last segment
            if idx != len(segments) - 1:
                wav_file.writeframes((silence * 32767).astype(np.int16).tobytes())

    # Optionally remove silence from the output file
    if remove_silence:
        output_file = remove_silence_function(output_file, minimum_silence=minimum_silence)

    return output_file
old_voice_pack_path=""
old_VOICEPACK=None
def tts(MODEL,device,text, voice_name, speed=1.0, trim=0.5, pad_between_segments=0.5, output_file="",remove_silence=True,minimum_silence=50):
    global old_voice_pack_path,old_VOICEPACK
    language = voice_name[0]
    voice_pack_path = f"./KOKORO/voices/{voice_name}.pt"
    if voice_name.endswith(".pt"):
        language="a"
        voice_pack_path=voice_name
    text=clean_text(text)
    segments = large_text(text, language)
    if (old_voice_pack_path!=voice_pack_path)or ("weighted_normalised_voices.pt" in voice_pack_path):
        VOICEPACK = torch.load(voice_pack_path, weights_only=True).to(device)
        old_voice_pack_path=voice_pack_path
        old_VOICEPACK=VOICEPACK
        # print("Loaded new voice pack")
    else:
        VOICEPACK=old_VOICEPACK
        # print("Using old voice pack")
    speed = clamp_speed(speed)
    trim = clamp_trim(trim)
    silence_duration = clamp_trim(pad_between_segments)
    output_file=get_random_file_name(output_file)
    if debug:
        print(f'Loaded voice: {voice_pack_path}')
        print(f"Speed: {speed}")
        print(f"Trim: {trim}")
        print(f"Silence duration: {silence_duration}")
    sample_rate = 24000  # Sample rate of the audio

    # Create a silent audio segment in float32
    silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)

    # Open a WAV file for writing
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)

        for i in segments:
            id = i[0]
            text = i[1]
            if debug:
                print(i)
            audio, out_ps = generate(MODEL, text, VOICEPACK, lang=language, speed=speed)
            audio = trim_if_needed(audio, trim)

            # Scale audio from float32 to int16
            audio = (audio * 32767).astype(np.int16)

            # Write the audio segment to the WAV file
            wav_file.writeframes(audio.tobytes())
            
            # Add silence between segments, except after the last segment
            if id != len(segments) - 1:
                wav_file.writeframes((silence * 32767).astype(np.int16).tobytes())
    if remove_silence:
        output_file=remove_silence_function(output_file,minimum_silence=minimum_silence)
    return output_file



def tts_file_name(text):
    global temp_folder
    # Remove all non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Retain only alphabets and spaces
    text = text.lower().strip()             # Convert to lowercase and strip leading/trailing spaces
    text = text.replace(" ", "_")           # Replace spaces with underscores
    
    # Truncate or handle empty text
    truncated_text = text[:25] if len(text) > 25 else text if len(text) > 0 else "empty"
    
    # Generate a random string for uniqueness
    random_string = uuid.uuid4().hex[:8].upper()
    
    # Construct the file name
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name

