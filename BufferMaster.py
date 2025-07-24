import threading
import numpy as np
from datetime import datetime, timedelta
import time

from utils import convert_back_to_int16

class RingBuffer:
    """
    Based on reading, I think this will be more efficient that a double-ended queue for getting and adding audio data. 
    
    Ring buffer: pre-allocates array space in a buffer and gives windows of audio data that the buffer can dole to 
    any registered inferencers. **ONLY TO BE DIRECTLY ACCESSED BY THE BM**
    """
    def __init__(self, max_packets: int, packet_audio_bytes: int, packet_duration_ms: float, samples_per_packet, num_channels, steam_sec):
        self.max_packets = max_packets
        self.packet_audio_bytes = packet_audio_bytes
        self.packet_duration_ms = packet_duration_ms
        self.window_duration_sec = steam_sec

        # Audio buffer: shape (max_packets, packet_audio_bytes)
        self.audio_buffer = np.empty((max_packets, samples_per_packet, num_channels), dtype=np.int16)
        self.timestamps = [None] * max_packets  # Separate timestamp per packet

        self.head = 0  # Next write index
        self.tail = 0  # Oldest valid index
        self.size = 0 #Number of packets in the RB
        self.global_gen = 0

        self.lock = threading.Lock()
        # Active inference windows: set of tuples (start_idx, end_idx, gen)
        #Tell whether I can overwrite these indicices
        self.active_windows = set()
        self.print()

    def print(self):
        print(f"--------------ESTABLISHED RING BUFFER ------------------")
        print(f"\t STREAM AVAILABLE IN BUFFER: {self.window_duration_sec} s")
        print(f"\t EXPECTED PACKET DURATION: {self.packet_duration_ms} ms")
        print(f"\t MAX NUMBER OF PACKETS: {self.max_packets} packets")
        print(f"\t BUFFER SHAPE: {self.audio_buffer.shape}")
    
    def index_in_window(self, idx, start, end):
        """Check if idx is inside the window [start, end) modulo size."""
        if start <= end:
            return start <= idx < end
        else:
            return idx >= start or idx < end
        
    def add_packet(self, audio_bytes: bytes, timestamp: datetime):
        '''
        Add packet to the buffer and update head/tail pointers.
        '''
        audio_int16 = convert_back_to_int16(audio_bytes)

        #Write data to buffer
        with self.lock:
            # Check overwrite safety
            for start_idx, end_idx, gen in list(self.active_windows):
                if self.index_in_window(self.head, start_idx, end_idx):
                    print(f"[DEBUG] Overlap detected: head={self.head}, active_window=({start_idx}, {end_idx}, {gen})")
                    raise RuntimeError("Buffer overwrite during active inference!")
            # Write the data into the audio buffer at 'head' index
            self.audio_buffer[self.head, :len(audio_int16)] = audio_int16
            self.timestamps[self.head] = timestamp
            self.global_gen += 1

            # Manage ring buffer pointers and size
            if self.size == self.max_packets:
                # Buffer is full, move tail forward to overwrite oldest packet
                self.tail = (self.tail + 1) % self.max_packets
            else:
                # Buffer not full yet, increment size
                self.size += 1
            # Move head forward (circular)
            self.head = (self.head + 1) % self.max_packets
    
    def get_idx(self, start_time: datetime):
        """
        Returns the index of the last packet with timestamp <= start_time.
        Binary search that accounts for ring buffer wraparound.
        Assumes timestamps are strictly increasing in write order.
        """
        if self.size == 0:
            return None

        low, high = 0, self.size - 1
        result = None

        while low <= high:
            mid = (low + high) // 2
            idx = (self.tail + mid) % self.max_packets
            ts = self.timestamps[idx]

            if ts is None:
                high = mid - 1
            elif ts <= start_time:
                result = idx  # Keep track of best match so far
                low = mid + 1
            else:
                high = mid - 1

        return result

    def get_window(self, start_time: datetime, duration_ms: float):
        """
        Get a view of the window from the start time for duration_ms. 
        NOTE: This method will not return the exact audio window. Instead, it will return a
        view of the packets needed to make the window. So may be up to 1.240 ms extra on the back
        and on the front ends. I did this to minimize buffer lock time. 

        If it fails it will return none, none. Inferencers should check for this.
        """
        with self.lock:
            # Binary search for the first packet with timestamp >= start_time
            start_idx = self.get_idx(start_time)
            if(start_idx == None):
                return None, None
            packets_needed = int(duration_ms // self.packet_duration_ms + 2)
            end_idx = (start_idx + packets_needed) % self.max_packets

            self.active_windows.add((start_idx, end_idx, self.global_gen))

            # Extract view
            if start_idx < end_idx:
                view = self.audio_buffer[start_idx:end_idx]
            else:
                view = (self.audio_buffer[start_idx:], self.audio_buffer[:end_idx])
            return start_idx, end_idx, self.global_gen, self.timestamps[start_idx], view
        
    def release_window(self, start_idx, end_idx, gen):
        with self.lock:
            self.active_windows.discard((start_idx, end_idx, gen))


class BufferMaster:
    """
    Created on Friday, July 11, 2025

    @author: Kasey Castello

    This module defines a BufferMaster class that adds acoustic data packets to a buffer and triggers
    instances of Inferencers that process the buffered data. 
    """
    def __init__(self, global_stop_event, max_duration_sec, packet_audio_bytes, sample_rate, bytes_per_sample, channels, samples_per_packet):
        '''
        Creates the buffermaster. Requires maximum buffer size (seconds), sample rate (Hz), bytes per sample, and number of channels.
        '''
        #For packet handling
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.channels = channels
        
        self.inferenceCheckCounter =0

        #Create the buffer
        self.packet_audio_bytes = packet_audio_bytes
        self.packet_duration_ms = (packet_audio_bytes / (self.sample_rate * self.bytes_per_sample * self.channels)) * 1000
        self.max_packets = int( max_duration_sec *1000/ self.packet_duration_ms) + 1
        self.samples_per_packet = samples_per_packet

        self.ring_buffer = RingBuffer(max_packets=self.max_packets, packet_audio_bytes=self.packet_audio_bytes, packet_duration_ms=self.packet_duration_ms, 
                                      samples_per_packet=samples_per_packet, num_channels=channels, steam_sec=max_duration_sec)
        
        #Threads for Executing Required BM Tasks
        self.inferencers = []
        self.stopper = global_stop_event
        self.trigger_thread = threading.Thread(target = self.trigger_ops, daemon=True)
        self.trigger_thread.start()

    def add_packet(self, audio_bytes: bytes, timestamp: datetime):
        """
        Adds a packet of audio data to the buffer.
        """
        try:
            self.ring_buffer.add_packet(audio_bytes, timestamp)
        except RuntimeError as e:
            print(f"Error: {e} — Killing all inferencers due to buffer overwrite")
            self.kill_all()
            self.stopper.set()
            return
        
    def get_audio_window(self, start_time: datetime, duration_ms: float):
        """
            Retrieves a window view of the packets needed based on the start time and the duration. 

            NOTE: It does not give exact bytes, just a no-copy view of the packets. The inferencer will have to handle 
            if sub-packet level splicing is requitred. 

            I did this to minimize buffer lock time to minimize packet dropoff.
        """
        return self.ring_buffer.get_window(start_time, duration_ms)
    
    def release_audio_window(self, start_idx, end_idx, gen):
        self.ring_buffer.release_window(start_idx, end_idx, gen)
        
    def register_inferencer(self, inferencer):
        self.inferencers.append(inferencer)

    def trigger_inferencers(self):
        for inferencer in self.inferencers:
            oldest = self.ring_buffer.timestamps[self.ring_buffer.tail]
            if inferencer.next_start_time is None:
                if oldest is not None:
                    inferencer.update_window_start(oldest)
                else:
                    continue  # Can't initialize yet
                    
            next_start = inferencer.next_start_time
            duration_ms = inferencer.duration_ms

            # Estimate the time that the window would end
            end_time = next_start + timedelta(milliseconds=duration_ms)

            # Guard: are the required timestamps covered?
            newest_idx = (self.ring_buffer.head - 1 + self.max_packets) % self.max_packets
            newest = self.ring_buffer.timestamps[newest_idx]

            if oldest is None or newest is None:
                continue  # Incomplete buffer
            
            if next_start < oldest or end_time > newest:
                continue  # Requested window isn't fully in buffer

            # Ok to get data
            start_idx = self.ring_buffer.get_idx(next_start)
            if start_idx is None:
                continue
            inferencer.trigger()
            inferencer.update_window_start(next_start + timedelta(milliseconds=duration_ms) )

    def trigger_ops(self):
        while not self.stopper.is_set():
            self.trigger_inferencers()
            time.sleep(0.002)  # Tune for your system

    def kill_all(self):
        for prisoner in self.inferencers:
            prisoner.stop()
        self.stopper.set()
        