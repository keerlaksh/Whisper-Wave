from huggingface_hub import list_repo_files, hf_hub_download
import os
import shutil
import torch
from itertools import combinations
import platform

# Repository ID
repo_id = "hexgrad/Kokoro-82M"
repo_id2="Remsky/kokoro-82m-mirror"
# Set up the cache directory
cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)

# Set up the base model paths
KOKORO_DIR = "./KOKORO"
VOICES_DIR = os.path.join(KOKORO_DIR, "voices")
FP16_DIR = os.path.join(KOKORO_DIR, "fp16")
KOKORO_FILE = "kokoro-v0_19.pth"
FP16_FILE = "fp16/kokoro-v0_19-half.pth"

def download_files(repo_id, filenames, destination_dir, cache_dir):
     # Ensure directories exist
     os.makedirs(destination_dir, exist_ok=True)
     
     for filename in filenames:
        destination = os.path.join(destination_dir, os.path.basename(filename))
        if not os.path.exists(destination):
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
            shutil.copy(file_path, destination)
            print(f"Downloaded and saved: {destination}")
        else:
              print(f"File already exist in: {destination}")
        

def get_voice_models():
    """Downloads missing voice models from the Hugging Face repository."""

    # Create or empty the 'voices' directory
    if os.path.exists(VOICES_DIR):
       shutil.rmtree(VOICES_DIR)
    os.makedirs(VOICES_DIR, exist_ok=True)

    # Get list of files from the repository
    files = list_repo_files(repo_id)
    
    # Filter for voice files
    voice_files = [file.replace("voices/", "") for file in files if file.startswith("voices/")]

    # Get current voice files
    current_voice = os.listdir(VOICES_DIR)

    # Download new voices
    download_voice = [file for file in voice_files if file not in current_voice]
    if download_voice:
        #  print(f"Files to download: {download_voice}")
         pass
    eng_voices = []
    for i in download_voice:
         if i.startswith("a") or i.startswith("b"):
              eng_voices.append(i)        
    download_files(repo_id, [f"voices/{file}" for file in eng_voices], VOICES_DIR, cache_dir)

def download_base_models():
    """Downloads Kokoro base model and fp16 version if missing."""

    download_files(repo_id2, [KOKORO_FILE], KOKORO_DIR, cache_dir)
    download_files(repo_id2, [FP16_FILE], FP16_DIR, cache_dir)

def setup_batch_file():
    """Creates a 'run_app.bat' file for Windows if it doesn't exist."""

    if platform.system() == "Windows":
        bat_file_name = 'run_app.bat'
        if not os.path.exists(bat_file_name):
            bat_content_app = '''@echo off
call myenv\\Scripts\\activate
@python.exe app.py %*
@pause
'''
            with open(bat_file_name, 'w') as bat_file:
                bat_file.write(bat_content_app)
            print(f"Created '{bat_file_name}'.")
        else:
            print(f"'{bat_file_name}' already exists.")
    else:
        print("Not a Windows system, skipping batch file creation.")

def download_ffmpeg():
    """Downloads ffmpeg and ffprobe executables from Hugging Face."""
    print("For Kokoro TTS we don't need ffmpeg, But for Subtitle Dubbing we need ffmpeg")
    os_name=platform.system()
    if os_name == "Windows":
        repo_id = "fishaudio/fish-speech-1"
        filenames = ["ffmpeg.exe", "ffprobe.exe"]
        ffmpeg_dir = "./ffmpeg"
        download_files(repo_id, filenames, ffmpeg_dir, cache_dir)
    elif os_name == "Linux":
         print("Please install ffmpeg using the package manager for your system.")
         print("'sudo apt install ffmpeg' on Debian/Ubuntu")
    else:
        print(f"Manually install ffmpeg for {os_name} from https://ffmpeg.org/download.html")

def mix_all_voices(folder_path=VOICES_DIR):
     """Mix all pairs of voice models and save the new models."""
    # Get the list of available voice packs
     available_voice_pack = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(folder_path)
        if filename.endswith('.pt')
    ]

     # Generate all unique pairs of voices
     voice_combinations = combinations(available_voice_pack, 2)
   
     # Function to mix two voices
     def mix_model(voice_1, voice_2):
          """Mix two voice models and save the new model."""
          new_name = f"{voice_1}_mix_{voice_2}"
          voice_id_1 = torch.load(f'{folder_path}/{voice_1}.pt', weights_only=True)
          voice_id_2 = torch.load(f'{folder_path}/{voice_2}.pt', weights_only=True)

          # Create the mixed model by averaging the weights
          mixed_voice = torch.mean(torch.stack([voice_id_1, voice_id_2]), dim=0)

          # Save the mixed model
          torch.save(mixed_voice, f'{folder_path}/{new_name}.pt')
          print(f"Created new voice model: {new_name}")

    # Create mixed voices for each pair
     for voice_1, voice_2 in voice_combinations:
          print(f"Mixing {voice_1} ❤️ {voice_2}")
          mix_model(voice_1, voice_2)

def save_voice_names(directory=VOICES_DIR, output_file="./voice_names.txt"):
    """
    Retrieves voice names from a directory, sorts them by length, and saves to a file.

    Parameters:
        directory (str): Directory containing the voice files.
        output_file (str): File to save the sorted voice names.

    Returns:
        None
    """
    # Get the list of voice names without file extensions
    voice_list = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(directory)
        if filename.endswith('.pt')
    ]

    # Sort the list based on the length of each name
    voice_list = sorted(voice_list, key=len)

    # Save the sorted list to the specified file
    with open(output_file, "w") as f:
        for voice_name in voice_list:
            f.write(f"{voice_name}\n")

    print(f"Voice names saved to {output_file}")

# --- Main Execution ---
if __name__ == "__main__":
    get_voice_models()
    download_base_models()
    setup_batch_file()
    # mix_all_voices()
    save_voice_names()
    download_ffmpeg()
    print("Setup complete!")
