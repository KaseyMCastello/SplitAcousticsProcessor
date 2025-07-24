# SplitAcousticsProcessor
*SplitAcousticsProcessor** is a real-time acoustic inference module designed to process incoming UDP audio packets and apply inference using a suite of pluggable acoustic models. It features:

- A size-variable ring buffer for efficient memory use
- Concurrent window-based model inferencing
- Real-time compatibility with audio packet streams containing timestamped headers

Currently, this system is tuned to handle 12-byte header UDP packets (with embedded timestamps), such as those from acoustic datastreaming firmware.

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KaseyMCastello/SplitAcousticsProcessor.git
cd SplitAcousticsProcessor
pip install -r requirements.txt
```

## 🚀 Running Inference

a. Edit `config.yaml`
This YAML file contains settings for the packet stream and inference behavior. Below is an example configuration:
```bash
txt_file_path: 'C:\Users\kasey\Desktop\TimingTestSimUDP2\output.txt' 
listen_address: '127.0.0.1'
listen_port: 5005
#Datastream Firmware parameters
sample_rate: 200000  # e.g. 200 kHz – adjust as needed
bytes_per_sample: 2 #bytes in one acoustic sample
channels: 1 #number of channels in the audio stream
samples_per_packet: 248 #number of samples in one packet
packet_rate: 1.240 #number of ms per packet
header_size: 12
```

b. Modify `runRTPredictions.py`
In this script, you instantiate and configure your desired inferencers. Add as many inferencers as you like. Currently supported:
* `BFWInferencer` – uses spectrograms for detection. Currently configured to Blue/Fin Whale Detection but any torch model could be added in model_path
* `SPICEInferencer` – implements the SPICE click detector

To add your own custom inferencer:
* Subclass `InferencerShell`
* Implement three required methods:
  * `init(self, ...) ` 
  * `load_model(self)`
  * `process_audio(self, audio_window: np.ndarray, timestamp: datetime)`

**Example:** 
```bash
from InferencerShell import InferencerShell

class NewInferencer(InferencerShell):
    def __init__(self, buffer_master, duration_ms, model_path, stop_event, sample_rate=200000, bytes_per_sample=2, channels=1, extra Variables):
        super().__init__( buffer_master, duration_ms, model_path, stop_event, sample_rate, bytes_per_sample, channels )
        self.name = "New DETECTOR"
        self.packetCount = 0
        #ADD ANY NEW VARIABLES OR FUNCTIONALITY HERE
        self.load_model()
        self.print()
    
    def load_model(self):
        self.model_name = "MODEL_NAME"
        #Load model as necessary
        return
    
    def process_audio(self, audio_view, start_time):
        #Deteremine how packets should be handled for this class of inferencer    
```
Make sure your new inferencer is also included in the start() and stop() logic in runRTPredictions.py. When run, it should come up like this: 
