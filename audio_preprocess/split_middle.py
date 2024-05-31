import os
import librosa
import numpy as np
from scipy.io.wavfile import write

# Function to pad audio files to 6 seconds duration
def pad_audio(audio, target_duration=6, sr=44100):
    # Calculate current duration
    current_duration = len(audio) / sr
    
    # Pad audio if duration is less than target duration
    if current_duration < target_duration:
        # Calculate number of samples to pad
        samples_to_pad = int((target_duration - current_duration) * sr)
        
        # Pad audio with zeros
        audio = np.pad(audio, (0, samples_to_pad), mode='constant')
    
    return audio

# Function to split audio file from the middle and repeat to make each segment 6 seconds long
def split_and_repeat_audio(audio, target_duration=6, sr=44100):
    # Split audio file from the middle
    midpoint = len(audio) // 2
    audio1 = audio[:midpoint]
    audio2 = audio[midpoint:]
    
    # Repeat each segment to make it 6 seconds long
    audio1_duration = len(audio1) / sr
    audio2_duration = len(audio2) / sr
    
    # Repeat audio1
    while audio1_duration < target_duration:
        audio1 = np.concatenate((audio1, audio1), axis=None)
        audio1_duration *= 2
    
    # Repeat audio2
    while audio2_duration < target_duration:
        audio2 = np.concatenate((audio2, audio2), axis=None)
        audio2_duration *= 2
    
    # Truncate audio if duration exceeds 6 seconds
    audio1 = audio1[:int(target_duration * sr)]
    audio2 = audio2[:int(target_duration * sr)]
    
    return audio1, audio2

# Path to the audio file
input_audio_file = 'data/validate/Roar/padded_Roar-Rumble -207.wav_1.wav'

# Load audio file
audio, sr = librosa.load(input_audio_file, sr=None)

# Check duration of audio file
duration = librosa.get_duration(y=audio, sr=sr)
print(f"Old duration of {input_audio_file}: {duration:.2f} seconds")

# If duration is greater than 6 seconds, split and repeat audio
if duration >= 6:
    audio1, audio2 = split_and_repeat_audio(audio, sr=sr)
    
    # Save audio segments as separate files
    output_file1 = 'data/validate/Roar/padded_audio_1.wav'
    write(output_file1, sr, audio1)
    
    output_file2 = 'data/validate/Roar/padded_audio_2.wav'
    write(output_file2, sr, audio2)

# If duration is between 3 and 6 seconds, pad the audio if necessary
elif 3 < duration <6:
    padded_audio = pad_audio(audio, sr=sr)
    output_file = 'data/validate/Roar/padded_audio.wav'
    write(output_file, sr, padded_audio)
