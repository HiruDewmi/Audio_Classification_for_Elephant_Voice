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

# Function to segment audio into 6-second chunks
def segment_audio(audio, target_duration=6, sr =44100 ):
    # Calculate number of samples in each segment
    
    print("sr",sr)
    segment_samples = int(target_duration * sr)
    
    # Calculate number of segments
    num_segments = len(audio) // segment_samples
    
    # Segment audio into chunks
    segments = [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(num_segments)]
    
    # Pad the last segment if its duration is less than target duration
    if len(audio) % segment_samples != 0:
        last_segment = audio[num_segments * segment_samples:]
        # Calculate duration of the last segment
        last_segment_duration = len(last_segment) / sr

        while last_segment_duration <= 3:
            last_segment = np.concatenate((last_segment, last_segment), axis=None)
            last_segment_duration *= 2
        
        # If the last segment duration is still less than 6 seconds, pad it
        if last_segment_duration < target_duration:
            last_segment = pad_audio(last_segment, target_duration, sr)

        segments.append(last_segment)
    
    return segments

# Directory containing raw WAV files
input_dir = 'data/train/Trumpet'

# Function to save audio files with sequential names like Rumble01.wav, Rumble02.wav, etc.
def save_audio(audio, sr, output_dir, file_numbers, prefix='Trumpet'):
    file_number = file_numbers.pop(0)
    output_file = os.path.join(output_dir, f'{prefix}{file_number:02d}.wav')
    write(output_file, sr, audio)

# List to keep track of file numbers
file_numbers = list(range(20, 100))  # Adjust the range as needed

# Loop through all WAV files in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(input_dir, file_name)
        
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Check duration of audio file
        duration = librosa.get_duration(y=audio, sr=sr)

        print(f"Old duration of {file_name}: {duration:.2f} seconds")
        
        # If duration is greater than 10 seconds, segment audio into 6-second chunks
        if duration >= 8:
            segments = segment_audio(audio, sr=sr)
            
            # Save each segment as a separate audio file
            for i, segment in enumerate(segments):
                segment_duration = len(segment) / sr
                print(f"Duration of segment {i + 1}: {segment_duration:.2f} seconds")
                
                # output_file = os.path.join(input_dir, f'padded_{file_name}_{i}.wav')
                # write(output_file, sr, segment)
                save_audio(segment, sr, input_dir,file_numbers)
        
        # If duration is between 6 and 10 seconds, pad the audio if necessary
        # elif 6 <= duration <= 10:
        #     padded_audio = pad_audio(audio, sr=sr)
        #     output_file = os.path.join(input_dir, f'padded_{file_name}')
        #     write(output_file, sr, padded_audio) 
