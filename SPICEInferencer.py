"""
Created on Friday, July 11, 2025

@author: Kasey Castello

Adaptation of the SPICE Detector for real-time packet-based inference.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.fft import fft  
import time
from datetime import timedelta

from InferencerShell import InferencerShell

class SPICEInferencer(InferencerShell):
    """Class to detect/classify odontocete clicks."""
    def __init__(self, buffer_master, duration_ms, model_path, stop_event, sample_rate=200000, bytes_per_sample=2, channels=1, cutoff_SNR = 10, 
                 bandpass_low = 10000, bandpass_high = 90000, click_duration_us_low = 30, click_duration_us_high = 300, 
                 f_low = 16, f_high = 75):
        super().__init__( buffer_master, duration_ms, model_path, stop_event, sample_rate, bytes_per_sample, channels )
        self.name = "SPICE DETECTOR"
        self.packetCount = 0

        self.cutoff_SNR = cutoff_SNR
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.click_duration_us_low = click_duration_us_low
        self.click_duration_us_high = click_duration_us_high
        self.f_low = f_low
        self.f_high = f_high
        self.triggerCounter = 0
        self.detections = []

        self.load_model()
        self.print()

    def print(self):
        print(f"--------------ESTABLISHED {self.name} ------------------")
        print(f"\t INFERENCE WINDOW: {self.duration_ms /1000} s")
        print(f"\t CUTOFF SNR (dB): {self.cutoff_SNR}")
        print(f"\tBANDPASS: {self.bandpass_low} - {self.bandpass_high} Hz")
        print(f"\t CLICK DURATION: {self.click_duration_us_low } - {self.click_duration_us_high} us")
    
    def load_model(self):
        self.model_name = "SPICE_DETECTOR_RT"
        self.model_load_time = 0.0
        return
    
    def process_audio(self, audio_view, start_time):
        self.triggerCounter +=1
        if isinstance(audio_view, tuple):
            audio = np.concatenate(audio_view, axis=0)  # shape: (n_packets * 248, channels)
        else:
            audio = audio_view
        audio = audio.reshape(-1)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        else:
            audio = audio[:, 0] if audio.ndim == 2 else audio  # mono already

        # 3. Bandpass filter (zero-phase)
        sos = butter( 5, [self.bandpass_low, self.bandpass_high], btype='bandpass', fs=self.sample_rate, output='sos' )
        try:
            filtered = sosfiltfilt(sos, audio)
        except ValueError:
            print(f"[{self.name}] ⚠️ Filtering failed, too short: {len(audio)} samples")
            return

        # 4. Envelope and RMS in dB
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        rms = np.sqrt(np.convolve(filtered**2, np.ones(20)/20, mode='same'))
        rms[rms == 0] = 1e-12
        db_spl = 20 * np.log10(rms)

        # 5. Thresholding for candidate clicks
        noise_floor = np.percentile(db_spl, 30)
        threshold_db = noise_floor + self.cutoff_SNR
        above = db_spl > threshold_db
        rising = np.where(np.diff(above.astype(int)) == 1)[0]
        falling = np.where(np.diff(above.astype(int)) == -1)[0]

        # Align edges
        if len(falling) > 0 and (len(rising) == 0 or falling[0] < rising[0]):
            falling = falling[1:]
        if len(rising) > len(falling):
            rising = rising[:len(falling)]

        detections = []
        freq_res = self.sample_rate / 400

        # 6. Loop through candidate pulses
        for st, ed in zip(rising, falling):
            dur_us = (ed - st) / self.sample_rate * 1e6
            if not (self.click_duration_us_low <= dur_us <= self.click_duration_us_high):
                continue

            pulse = filtered[st:ed] * np.hanning(ed - st)
            padded = np.zeros(400)
            padded[:pulse.size] = pulse
            spectrum = np.abs(fft(padded))[:200]
            peak_bin = np.argmax(spectrum)
            peak_freq = peak_bin * freq_res

            if not (self.f_low * 1000 <= peak_freq <= self.f_high * 1000):
                continue

            onset = start_time + timedelta(seconds=st / self.sample_rate)
            detections.append({
                "time": onset,
                "duration_us": dur_us,
                "peak_kHz": peak_freq / 1000,
                "SNR_dB": threshold_db - noise_floor
            })
        
        self.detections.append(detections)
        if(self.triggerCounter >= 30000):
            self.triggerCounter = 0
            print(f"5 Minute Detection Summary: {len(self.detections)} clicks found in last 5 minutes")