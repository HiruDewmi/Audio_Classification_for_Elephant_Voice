import os
import librosa
import numpy as np
from scipy.io.wavfile import write
import argparse

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
def segment_audio(audio, target_duration=6, sr=44100):
    # Calculate number of samples in each segment
    segment_samples = int(target_duration * sr)
    
    # Calculate number of segments
    num_segments = len(audio) // segment_samples
    
    # Segment audio into chunks
    segments = [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(num_segments)]
    
    # Pad the last segment if its duration is less than target duration
    if len(audio) % segment_samples != 0:
        last_segment = audio[num_segments * segment_samples:]
        last_segment_duration = len(last_segment) / sr
        
        while last_segment_duration <= 3:
            last_segment = np.concatenate((last_segment, last_segment), axis=None)
            last_segment_duration *= 2
        
        if last_segment_duration < target_duration:
            last_segment = pad_audio(last_segment, target_duration, sr)
        
        segments.append(last_segment)
    
    return segments

# Function to save audio files with sequential names
def save_audio(audio, sr, output_dir, file_numbers, prefix='Trumpet'):
    file_number = file_numbers.pop(0)
    output_file = os.path.join(output_dir, f'{prefix}{file_number:02d}.wav')
    write(output_file, sr, audio)

# Main function to process audio files
def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to keep track of file numbers
    file_numbers = list(range(1, 100))  # Adjust the range as needed

    # Loop through all WAV files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            audio_file = os.path.join(input_dir, file_name)
            
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=None)
            
            # Check duration of audio file
            duration = librosa.get_duration(y=audio, sr=sr)
            print(f"Old duration of {file_name}: {duration:.2f} seconds")
            
            # Segment or pad audio based on duration
            if duration >= 8:
                segments = segment_audio(audio, sr=sr)
                for i, segment in enumerate(segments):
                    segment_duration = len(segment) / sr
                    print(f"Duration of segment {i + 1}: {segment_duration:.2f} seconds")
                    save_audio(segment, sr, output_dir, file_numbers)
            elif 3 < duration < 6:
                padded_audio = pad_audio(audio, sr=sr)
                save_audio(padded_audio, sr, output_dir, file_numbers)
            elif duration <= 3:
                print(f"Skipping {file_name} as its duration is less than or equal to 3 seconds")
            else:
                save_audio(audio, sr, output_dir, file_numbers)

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for classification.')
    parser.add_argument('input_dir', type=str, help='Directory containing raw WAV files')
    parser.add_argument('output_dir', type=str, help='Directory to save processed audio files')
    args = parser.parse_args()

    process_audio_files(args.input_dir, args.output_dir)
