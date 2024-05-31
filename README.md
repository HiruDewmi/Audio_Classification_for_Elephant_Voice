# Audio_Classification_for_Elephant_Voice

# Raw Audio Processing with Machine Learning Models

In this project, we explore various machine learning models suitable for raw audio processing.

## Dataset
Our dataset comprises recordings of three caller types of elephants:
- Rumble
- Roar
- Trumpet

## Used Models
We experimented with the following models:

1. **MobileNet V2:**
   - Utilized pre-trained data and fine-tuned it on our dataset.

2. **YAMNet:**
   - Trained the model from scratch specifically for our datasets.

3. **RawNet:**
   - Trained the model from scratch for our datasets.

4. **ElephantCallerNet based from ACDNet:**
   - Trained the model from scratch for our datasets.
  
## Pre-processing
The models are trained on raw audio files. However, the audio files need to be pre-processed before training. Here are the steps for pre-processing:
1.   **Dataset Division**: The dataset is divided into test, train, and validation datasets, each containing the three classes.
2.   **Audio Duration**: The models accept 6-second audio files.
      - If the waveform is greater than 6 seconds, it is trimmed to 6 seconds.
      - If the audio file is greater than 8 seconds, it is segmented into equal length sizes and padded or trimmed to make each segment 6 seconds.
      - Audio files with a duration less than 2 seconds are avoided.

#Environment Setup
To ensure a consistent environment for running the script, we have included an environment.yml file. This file can be used to set up a conda environment with all necessary dependencies.

##Steps to Set Up the Environment
1. **Install Conda:**
   - If you don't have conda installed, download and install it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Environment:**
   ```bash
   conda env create -f environment.yml
   ```
## Usage Guide

The following steps describe how to process the raw audio files to ensure they meet the requirements of the machine learning models. The script segments, pads, or trims the audio files as necessary to produce 6-second audio files suitable for training.

### Steps to Run the Audio Processing Script

1. **Prepare the Script:**
   - Save the provided script as `process_audio.py`.

2. **Run the Script:**
   - Run the script with the input and output directories as arguments:
   ```bash
   python process_audio.py <input_dir> <output_dir>
   ```
   Replace `<input_dir>` with the path to your directory containing the raw WAV files, and `<output_dir>` with the path where you want to save the processed audio files.

### Example

Suppose you have a directory of raw audio files located at `data/raw_audio` and you want to save the processed audio files in `data/processed_audio`. You would run the script as follows:
```bash
python process_audio.py data/raw_audio data/processed_audio
```

This command processes each audio file in `data/raw_audio`, applies the necessary padding, segmentation, or trimming, and saves the processed files in `data/processed_audio` with appropriate naming.

