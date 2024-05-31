import os
import sys
import librosa
import soundfile as sf
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise

# Define the augmentation parameters
augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    # SpeedChange(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    # TimeShift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def augment_audio(input_dir, target_class, num_augmented_files):
    # Count the number of available audio files
    audio_files = [f for f in os.listdir(os.path.join(input_dir, target_class)) if f.endswith(".wav")]
    num_audio_files = len(audio_files)
    
    # Calculate the number of augmentations per audio file
    augmentations_per_file = num_augmented_files // num_audio_files
    remaining_augmentations = num_augmented_files % num_audio_files

    print("augmentations_per_file",augmentations_per_file)
    
    # Iterate through each audio file in the target class directory
    augmented_files_created = 0
    for file in audio_files:
        # Load the audio file
        # if "_aug_" in file:
        #     continue

        audio_path = os.path.join(input_dir, target_class, file)
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply augmentation and save augmented files
        for i in range(augmentations_per_file):
            augmented_audio = augment(samples=y, sample_rate=sr)
            output_file = os.path.join(input_dir, target_class, f"{file.split('.')[0]}_aug_{i}.wav")
            sf.write(output_file, augmented_audio, sr)
            augmented_files_created += 1
        
        # If there are remaining augmentations, apply them to this audio file
        if remaining_augmentations > 0:
            augmented_audio = augment(samples=y, sample_rate=sr)
            output_file = os.path.join(input_dir, target_class, f"{file.split('.')[0]}_aug_{augmentations_per_file}.wav")
            sf.write(output_file, augmented_audio, sr)
            augmented_files_created += 1
            remaining_augmentations -= 1
        
        # Stop augmenting if the desired number of files is reached
        if augmented_files_created >= num_augmented_files:
            break

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input_dir target_class num_augmented_files")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    target_class = sys.argv[2]
    num_augmented_files = int(sys.argv[3])
    
    augment_audio(input_dir, target_class, num_augmented_files)
