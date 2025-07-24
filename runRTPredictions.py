# -*- coding: utf-8 -*-
"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module combines all of the inferencers and buffer master classes to conduct real-time predictions on 
any input UDP stream
"""

import os
import yaml
import torch
import threading
import struct
import socket
import time
from datetime import datetime, timedelta

from BufferMaster import BufferMaster
from BFWInferencer import BFWInferencer
from SPICEInferencer import SPICEInferencer
from Listener import Listener

# === Load config ===
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

wav_file_path = ''
txt_file_path = config['txt_file_path']
model_path = config['model_path']
CalCOFI_flag = config['CalCOFI_flag']
listen_address = config['listen_address']
listen_port = config['listen_port']
sample_rate = config['sample_rate']
samples_per_packet = config['samples_per_packet']
bytes_per_sample = config['bytes_per_sample']
channels = config['channels']
header_size = config['header_size']

packet_audio_bytes = samples_per_packet * bytes_per_sample * channels
packet_size = header_size + packet_audio_bytes  

udp_timeout = 20  # minutes
window_size_sec = 60

# === Make output dir ===
os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
with open(txt_file_path, mode='w', encoding='utf-8') as txtfile:
    txtfile.write('\t'.join([ 'source', 'model', 'image_path', 'label', 'score','start_time','end_time', 'min_frequency', 'max_frequency', 'box_x1','box_x2','box_y1','box_y2']) + '\n')

#Global stopping event so that once one process triggers stop they all end
global_stop_event = threading.Event()
    
# === Make BufferMaster ===
buffer_master = BufferMaster( global_stop_event,max_duration_sec = (2 * window_size_sec), packet_audio_bytes= packet_audio_bytes, sample_rate = sample_rate, 
                             bytes_per_sample = bytes_per_sample, channels = channels, samples_per_packet=samples_per_packet)

# === Make Listener ===
listener = Listener( global_stop_event, listen_address = listen_address, listen_port = listen_port, packet_size = packet_size, buffer_master = buffer_master, timeout_duration = udp_timeout)

# === Make Inferencer ===
bfw = BFWInferencer( buffer_master = buffer_master, duration_ms = window_size_sec * 1000, model_path = model_path, stop_event= global_stop_event,
    sample_rate = sample_rate, bytes_per_sample = bytes_per_sample, channels = channels, CalCOFI_flag = CalCOFI_flag, output_file_path = txt_file_path,
    file_output_bool = True )

sp = SPICEInferencer( buffer_master = buffer_master, duration_ms=10, model_path= "", stop_event= global_stop_event)
# === Start ===
print("---------------------------------------------------------------")

import cProfile
import pstats

def main():
    listener.start()
    sp.start()
    bfw.start()
    try:
        listener.stop_event.wait()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down...")
        listener.stop()
        sp.stop()
        bfw.stop()

    print("Shutdown complete.")

def print_profile_in_ms(profiler, top_n=85):
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    print(f"{'ncalls':>10} {'tottime':>15} {'percall':>10} {'cumtime':>15} {'percall':>10} function")
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        print(f"{nc:10} {tt*1000:15.3f} {tt*1000/nc if nc else 0:10.3f} {ct*1000:15.3f} {ct*1000/nc if nc else 0:10.3f} {func[2]}")

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    main()
    #time.sleep(3)
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('cumtime')
    #print_profile_in_ms(profiler)









