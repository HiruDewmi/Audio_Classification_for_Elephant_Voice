import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchaudio
from models import *
from models import MobileNetV2RawAudio, YAMNet, ElephantCallerNet, RawNet
from torch.utils.data import DataLoader
import os
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model.to(device)

# Load the appropriate pre-trained model based on some condition
def load_model_based_on_condition(condition):
    if condition == "mobilenet":
        model_path = "MobileNetV2RawAudio_optim.pt"  
    elif condition == "yamnet":
        model_path = "YAMNETRawAudio_100.pt" 
    elif condition == "rawnet":
        model_path = "rawnet.pt"  
    elif condition == "elephantnet":
        model_path = "adcnet_ep100.pt"  
    else:
        raise ValueError("Invalid condition")
    return load_model(model_path)

def inference_audio_file(model, audio_file_path, classes):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)
    with torch.no_grad():
        output = model(waveform)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class = classes[predicted_class_index]
    return predicted_class, probabilities

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio classification inference")
    parser.add_argument("condition", type=str, help="Condition (e.g., mobilenet, yamnet, elephantnet)")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    args = parser.parse_args()

    # Define classes based on imported model names
    classes = ["Roar", "Rumble", "Trumpet"]

    # Load model based on condition
    model = load_model_based_on_condition(args.condition)

    # Perform inference
    predicted_class, probabilities = inference_audio_file(model, args.audio_file, classes)
    print("Predicted class:", predicted_class)
    print("Class probabilities:", probabilities)