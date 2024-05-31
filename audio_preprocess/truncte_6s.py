import os
import librosa
import numpy as np
from scipy.io.wavfile import write

# Function to pad audio files to 6 seconds duration
def pad_audio(audio_file, target_duration=6):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Calculate current duration
    current_duration = librosa.get_duration(y=audio, sr=sr)
    print(f"Old duration of {audio_file}: {current_duration:.2f} seconds")
    
    # Check if audio duration is between 6 and 8 seconds
    if 6 <= current_duration < 8:
        # Calculate the number of samples to remove from both start and end of the audio
        samples_to_remove = int((current_duration - target_duration) / 2 * sr)
        
        # Truncate audio by removing equal parts from both start and end
        audio = audio[samples_to_remove:-samples_to_remove]
    
    # Truncate audio if duration exceeds 6 seconds
    if librosa.get_duration(y=audio, sr=sr) > target_duration:
        audio = audio[:int(target_duration * sr)]
    
    new_duration = librosa.get_duration(y=audio, sr=sr)
    print(f"New duration of padded audio: {new_duration:.2f} seconds")

    return audio

# Directory containing raw WAV files
input_dir = 'data/train/Trumpet'

# Loop through all WAV files in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(input_dir, file_name)
        
        # Check duration of audio file
        duration = librosa.get_duration(path=audio_file)
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Pad audio if duration is between 6 and 8 seconds
        if 6 < duration < 7:
            padded_audio = pad_audio(audio_file)
            
            # Save padded audio
            output_file = os.path.join(input_dir, f'padded_{file_name}')
            write(output_file, sr, padded_audio) 
