import os
from pydub import AudioSegment
import librosa

def get_audio_length(audio_file):
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000  # Length in seconds

def print_audio_lengths_in_directory(directory):
    print(f"Directory: {directory}")
    audio_subdir = os.path.join(directory, "Rumble")  # Assuming audio files are in a subdirectory named "audio"
    if os.path.exists(audio_subdir) and os.path.isdir(audio_subdir):
        for file in os.listdir(audio_subdir):
            file_path = os.path.join(audio_subdir, file)
            if os.path.isfile(file_path) and file.endswith(('.mp3', '.wav', '.ogg')):
                length = get_audio_length(file_path)
                print(f"  {file}: {length:.2f} seconds")
                audio, sr = librosa.load(file_path, sr=None)
                print(f"  {file}: {sr}")
    else:
        print("  No audio directory found.")

def main():
    root_directory = '/home/ubuntu/hiruni/noise_identification/data/train/'
    for dirpath, _, _ in os.walk(root_directory):
        print_audio_lengths_in_directory(dirpath)

if __name__ == "__main__":
    main()
