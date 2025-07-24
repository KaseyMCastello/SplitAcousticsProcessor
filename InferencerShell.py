"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines an inferencer class that all other inferencers should inherit from. This class shouldn't be used directly.
It provides a basic structure for inferencers to process audio data from a BufferMaster.
"""

import threading
from BufferMaster import BufferMaster
from datetime import datetime, timedelta
import time

class InferencerShell:
    def __init__(self, buffer_master, duration_ms, model_path, stop_event, sample_rate=200000, bytes_per_sample=2, channels=1):
        """
        Initializes the inferencer with the given parameters.
        """
        self.buffer_master = buffer_master
        self.model_path = model_path
        #For Sample Processing
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        #For processing audio data
        self.duration_ms = duration_ms
        self.last_inference_time = 0
        self.next_start_time = None
        self.next_end_time = None
        
        #For executing the inferencer
        self.trigger_event = threading.Event()
        self.stop_event = stop_event or threading.Event()
        self.buffer_master.register_inferencer(self)

    def print(self):
        print(f"--------------ESTABLISHED {self.name} ------------------")
        print(f"\t MODEL: {self.model_name}")
        print(f"\t MODEL LOAD TIME: {self.model_load_time} s")
        print(f"\t INFERENCE WINDOW: {self.duration_ms /1000} s")
        
    def load_model(self):
        """
        Loads the model needed for inference. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def start(self):
        """
        Starts the inferencer thread.
        """
        threading.Thread(target=self.run, daemon=False).start()
    
    def stop(self):
        """
        Stops the inferencer thread.
        """
        self.stop_event.set()
    
    def trigger(self):
        "Condition for running inference met. Trigger code to run"
        self.use_time = self.next_start_time
        self.trigger_event.set()
    
    def process_audio(self, audio_view, start_time):
        """
        Processes a window of audio data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_window_start(self, newTime):
        self.next_start_time = newTime
        self.next_end_time = self.next_start_time + timedelta(milliseconds=self.duration_ms)

    def run(self):
        """
        Runs the inferencer, processing audio data at specified intervals.
        """
        while not self.stop_event.is_set():
            self.trigger_event.wait(timeout=1)  # Wait max 1s to check stop_event
            if self.stop_event.is_set():
                break
            if not self.trigger_event.is_set():
                continue  # Timeout occurred, check loop condition again
            self.trigger_event.clear()
            try: 
                start_idx, end_idx, gen, actual_start, audio_view = self.buffer_master.get_audio_window(self.use_time, self.duration_ms)
                if audio_view is None:
                    raise ValueError(f"Failed to grab window at {self.use_time}")
                self.process_audio(audio_view, actual_start)
                self.buffer_master.release_audio_window(start_idx, end_idx, gen)
            except ValueError as e:
                print(f"{self.name} Window gathering error: {e}")
        print(f'Inference killer recieved for {self.name}, killing pipeline')
        