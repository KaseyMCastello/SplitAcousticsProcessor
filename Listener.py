# -*- coding: utf-8 -*-
"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines a Listener class that handles UDP packet transmission and queueing
for MBARC realtime acoustic data. It includes methods for starting and stopping the listener, 
and processing incoming packets.
"""
import socket
import threading
import time
from datetime import datetime
import struct

class Listener:
    def __init__(self, global_stop_event, listen_address = '127.0.0.1', listen_port = 5005, packet_size = 0, buffer_master = None, timeout_duration = 20):
        """
        Creates a listening socket for receiving UDP packets.
        """
        #For establishing the socket connection
        self.listen_address = listen_address
        self.listen_port = listen_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        #For processing packets
        self.packet_size = packet_size
        self.buffer_master = buffer_master
        self.last_packet_time = time.time()
        self.first_packet_received = False

        #For killing the class
        self.stop_event = global_stop_event or threading.Event()
        self.timeout_duration = timeout_duration * 60 # Convert minutes to seconds
        self.print()
        self.counter = 0
    
    def print(self):
        print(f"--------------ESTABLISHED LISTENER ------------------")
        print(f"\t LISTENING ON : {self.listen_address} PORT {self.listen_port}")
        print(f"\t EXPECTED PACKET SIZE: {self.packet_size} bytes")
        print(f"\t PACKET TIMEOUT: {self.timeout_duration/60} minutes.")

    def start(self):
        """
        Starts the listener to receive packets.
        """
        threading.Thread(target=self.run, daemon=False).start()
        threading.Thread(target=self.timeout_monitor, args=(self.timeout_duration,), daemon=True).start()
    
    def stop(self):
        """
        Stops the listener.
        """
        self.buffer_master.kill_all()
        self.stop_event.set()

    def timeout_monitor(self, timeout_duration=20*60):
        """
        Monitors the listener for timeouts.
        :param timeout_duration: Duration in seconds to wait before considering a timeout
        """
        while not self.stop_event.is_set():
            time.sleep(10)  # Check every 10 seconds
            time_since_last_packet = time.time() - self.last_packet_time
            if time_since_last_packet > timeout_duration:
                print(f"No packets received in {timeout_duration / 60:.0f} minutes. Stopping Listener...")
                self.stop()
       
    def run(self):
        """
        The main loop for receiving packets.
        """
        #Open the socket and begin listening 
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.socket.bind((self.listen_address, self.listen_port))
        self.socket.settimeout(1/1000)

        #Receive packets until the stop event is set
        while not self.stop_event.is_set():
            try:
                data, _ = self.socket.recvfrom(self.packet_size)
                if len(data) != self.packet_size:
                    print(f"Received packet of unexpected size: {len(data)} bytes. Expect: {self.packet_size} bytes.")
                    continue
                
                # Extract timestamp from header
                year, month, day, hour, minute, second = struct.unpack("BBBBBB", data[0:6])
                microseconds = int.from_bytes(data[6:10], byteorder='big')
                year += 2000
                timestamp = datetime(year, month, day, hour, minute, second, microsecond=microseconds)
                
                if not self.first_packet_received:
                    print(f"First packet received and timestamp set to: {timestamp}")
                    self.first_packet_received = True

                audio_data = data[12:]
                self.buffer_master.add_packet(audio_data, timestamp)
                self.last_packet_time = time.time()

            except socket.timeout:
                time.sleep(0.0005)
                continue
            except socket.error:
                break  # allows clean exit if socket is closed
        
        self.socket.close()
        print("No longer listening. Have a whale of a day!")
        
        
    
