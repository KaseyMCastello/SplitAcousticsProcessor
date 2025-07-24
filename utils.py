"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines useful functions that could be used by any inferencer
"""
import torch
import torchaudio
import numpy as np

def convert_back_to_int16(audio_bytes, num_channels=1):
    # Step 1: Convert bytes to native-endian int16 using NumPy's dtype trick
    audio_int16 = np.frombuffer(audio_bytes, dtype='>i2').astype(np.int16, copy=False)

    # Step 2: Reshape into (samples, channels) — zero-copy if shape already matches
    if audio_int16.size % num_channels != 0:
        raise ValueError(f"Data length {audio_int16.size} not divisible by {num_channels} channels.")
    
    return audio_int16.reshape(-1, num_channels)

# Function to convert single audio chunk to spectrogram
def audio_to_spectrogram(chunk, sr, device, min_hz, max_hz):
    # Use librosa to compute the spectrogram
    S = torch.stft(chunk, n_fft=sr, hop_length=int(sr/10), window=torch.hamming_window(sr).to(device), return_complex=True)
    transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80) #convert to dB and clip at 80dB
    S_dB_all = transform(torch.abs(S))
    S_dB = S_dB_all[0, min_hz:max_hz, :]  # 151 is exclusive, so it includes up to 150 Hz
    return S_dB.cpu().numpy()

   