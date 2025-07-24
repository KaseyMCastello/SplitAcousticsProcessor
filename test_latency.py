import os
import yaml
import torch
import threading
import time
import cProfile
import pstats
from io import StringIO

from BufferMaster import BufferMaster
from TJMInferencer import TestJoeModelInferencer
from Listener import Listener

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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
udp_timeout = 1  # short timeout for profiling
window_sizes = [1, 2, 10, 100, 1000, 10000, 20000, 60000]  # seconds to test

def run_profile(window_size_sec):
    stop_event = threading.Event()
    buffer_master = BufferMaster(
        stop_event,
        max_duration_sec= 2 * 60,
        packet_audio_bytes=packet_audio_bytes,
        sample_rate=sample_rate,
        bytes_per_sample=bytes_per_sample,
        channels=channels,
        samples_per_packet=samples_per_packet
    )

    listener = Listener(stop_event, listen_address, listen_port, packet_size, buffer_master, udp_timeout)

    inferencer = TestJoeModelInferencer(
        buffer_master=buffer_master,
        duration_ms=window_size_sec,
        model_path="",
        stop_event=stop_event
    )

    inferencer = TestJoeModelInferencer(
        buffer_master=buffer_master,
        duration_ms=window_size_sec*10,
        model_path="",
        stop_event=stop_event
    )

    profiler = cProfile.Profile()
    profiler.enable()

    listener.start()
    inferencer.start()
    stop_event.wait()  # Let it run until timeout
    profiler.disable()

    listener.stop()
    inferencer.stop()

    return profiler

def extract_function_times(profiler, target_funcs=('add_packet', 'get_window')):
    stats = pstats.Stats(profiler)
    stats_dict = {}
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, name = func
        if any(name.endswith(target) for target in target_funcs):
            key = f"{os.path.basename(filename)}:{name}"
            stats_dict[key] = {
                'calls': nc,
                'total_time_us': tt * 1_000_000,
                'avg_time_us': (tt * 1_000_000 / nc) if nc else 0,
                'cum_time_us': ct * 1_000_000,
                'cum_per_call_us': (ct * 1_000_000 / nc) if nc else 0
            }
    return stats_dict

# === Loop through window sizes and print profiling results ===
for window_size in window_sizes:
    print(f"\n=== Profiling for window size {window_size} sec ===")
    profiler = run_profile(window_size)
    times = extract_function_times(profiler)
    print(f"{'Function':35} | {'calls':>5} | {'total(us)':>10} | {'avg(us)':>8} | {'cum(us)':>10} | {'cum/call(us)':>13}")
    print("-" * 90)
    for func_name, info in times.items():
        print(f"{func_name:35} | {info['calls']:5d} | {info['total_time_us']:12.2f} | "
      f"{info['avg_time_us']:10.2f} | {info['cum_time_us']:12.2f} | {info['cum_per_call_us']:15.2f}")
