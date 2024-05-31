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
    
    # Check if padding is needed
    if current_duration < target_duration:
        # Calculate number of samples to pad
        # samples_to_pad = int((target_duration - current_duration) * sr)
        samples_to_pad = int(target_duration * sr) - len(audio)
        
        # Pad audio with zeros
        audio_padded = np.pad(audio, (0, samples_to_pad), mode='constant')
    else:
        # If duration is already longer than target, no padding needed
        audio_padded = audio
    
    new_duration = librosa.get_duration(y=audio_padded, sr=sr)
    print(f"New duration of padded audio: {new_duration:.2f} seconds")

    return audio_padded

# Directory containing raw WAV files
input_dir = '/home/ubuntu/hiruni/noise_identification/data/train/Trumpet'

def save_audio(audio, sr, output_dir, file_numbers, prefix='Trumpet'):
    file_number = file_numbers.pop(0)
    output_file = os.path.join(output_dir, f'{prefix}{file_number:02d}.wav')
    write(output_file, sr, audio)

# List to keep track of file numbers
file_numbers = list(range(100, 200))  # Adjust the range as needed


# Loop through all WAV files in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(input_dir, file_name)
        
        # Check duration of audio file
        duration = librosa.get_duration(path=audio_file)
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Pad audio if duration is between 3 and 6 seconds
        if 3.00 < duration < 6:
            padded_audio = pad_audio(audio_file)
            
            # Save padded audio
            # output_file = os.path.join(input_dir, f'padded_{file_name}')
            # # librosa.output.write_wav(output_file, padded_audio, sr)
            # write(output_file, sr, padded_audio) 
            save_audio(padded_audio, sr, input_dir,file_numbers)
