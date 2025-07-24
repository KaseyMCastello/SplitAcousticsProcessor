"""
Created on Friday, July 11, 2025

@author: Kasey Castello

This module defines an inferencer class for blue and finn whales, using the model and data-processing pipeline developed
by Michaela Alksne (https://github.com/m1alksne/WhaleMoanDetector). (Takes in 60s of audio data, outputs predictions of blue whale
A, B, and D calls and fin whale 40Hz, and 20Hz calls. )
"""
from InferencerShell import InferencerShell
import os
import torch
import time
import torchvision
import numpy as np
import torchvision.ops as ops
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime, timedelta
from utils import convert_back_to_int16, audio_to_spectrogram

class BFWInferencer(InferencerShell):
    """Class to detect/classify specifically blue and fin whale calls."""
    def __init__(self, buffer_master, duration_ms, model_path, stop_event, sample_rate=200000, bytes_per_sample=2, channels=1, 
                 CalCOFI_flag=False, output_file_path = "", file_output_bool = True, stream_name = "BFW_UDP_Stream_"):
        super().__init__( buffer_master, duration_ms, model_path, stop_event, sample_rate, bytes_per_sample, channels)
        self.output_file_path = output_file_path

        self.name = "Blue-Fin Whale Inferencer"

        #Variables for later use in spectrogram generation and filtering
        self.A_thresh=0
        self.B_thresh=0
        self.D_thresh=0
        self.TwentyHz_thresh=0
        self.FourtyHz_thresh=0
        self.window_size = self.duration_ms
        self.overlap_size = 0
        self.time_per_pixel = 0.1  # Since hop_length = sr / 10, this simplifies to 1/10 second per pixel

        #Variables for use in prediction
        self.label_mapping = {'D': 1, '40Hz': 2, '20Hz': 3, 'A': 4, 'B': 5}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.CalCOFI_flag = CalCOFI_flag
        self.file_output_bool = file_output_bool
        self.stream_name = stream_name

        ## Define spectrogram and data parameters
        self.fieldnames = ['source', 'model_no', 'image_file_path', 'label', 'score', 'start_time','end_time',
              'min_frequency', 'max_frequency','box_x1', 'box_x2', 'box_y1', 'box_y2' ]
    
        #Have the model ready for when we get our first packet of audio data.
        self.model_load_time = 0
        self.model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        self.print()

    def load_model(self):
        "Loads the model according to user input. Right now, uses RESNET 50 and the pth is the state dict to be loaded "
        start_time = time.time()
        self.model_name = os.path.basename(self.model_path)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        num_classes = 6 # 5 classes plus background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features # classification score and number of features (1024 in this case)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.model_load_time = time.time() - start_time
    
    def predict_spectrogram(self, spectrogram_data, start_time):
        # Threshold dictionary for easy access by label name
        thresholds = {'A': self.A_thresh, 'B': self.B_thresh, 'D': self.D_thresh,
                    '20Hz': self.TwentyHz_thresh, '40Hz': self.FourtyHz_thresh}
        freq_resolution=1
        
        predictions = []
        # Normalize spectrogram and convert to tensor
        normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data))
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L')
        image = ImageOps.flip(S_dB_img)
        # Convert the image to a numpy array for processing
        img_array = np.array(image)

        if self.CalCOFI_flag:
        # CalCOFI Sonobuoy cleanup for AIS

            threshold_1 = 200  # Threshold for the first 10 pixel block
            threshold_2 = 180  # Threshold for the second 10 pixel block
            threshold_3 = 160  # Lower threshold for the third 10 pixel block
            # Gray value to replace the AIS signal
            gray_value = 128  # Mid-gray
            # Find the vertical white lines and gray them out
            # Loop through each column (time slice) in the spectrogram
            for col in range(img_array.shape[1]):  # Loop through each column
            # Check first 10 pixel block (corresponding to the lowest frequency band)
                if np.sum(img_array[-10:, col]) > threshold_1 * 10:
                    # If the first 10 pixel block passes, check the second 10 pixel block
                    if np.sum(img_array[-20:-10, col]) > threshold_2 * 10:
                        # If the second block passes, check the third block with a lower threshold
                        if np.sum(img_array[-30:-20, col]) > threshold_3 * 10:
                            # If all conditions are met, gray out the entire column
                            img_array[:, col] = gray_value  # Replace the entire column with gray 
        
        final_image = Image.fromarray(img_array)
        # Convert to tensor
        S_dB_tensor = F.to_tensor(final_image).unsqueeze(0).to(self.device)        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(S_dB_tensor)

        # Extract prediction results
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        # Apply Non-Maximum Suppression (NMS)
        keep_indices = ops.nms(boxes, scores, 0.05)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
    
        # Iterate through detections
        for box, score, label in zip(boxes, scores, labels):
            textual_label = self.inverse_label_mapping.get(label.item(), 'Unknown')
            if score.item() < thresholds.get(textual_label, 0):
                continue

            # Get box time offsets
            start_offset_sec = box[0].item() * self.time_per_pixel
            end_offset_sec = box[2].item() * self.time_per_pixel

            start_datetime = start_time + timedelta(seconds=start_offset_sec)
            end_datetime = start_time + timedelta(seconds=end_offset_sec)
        
            predictions.append({
                'label': textual_label,
                'score': round(score.item(), 2),
                'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'min_frequency': round(150 - box[3].item() * freq_resolution),
                'max_frequency': round(150 - box[1].item() * freq_resolution),
                'box_x1': box[0].item(),
                'box_x2': box[2].item(),
                'box_y1': box[1].item(),
                'box_y2': box[3].item()
                })
        
        return predictions
    
    def predict_and_save_spectrogram(self, spectrogram_data, start_time):
        csv_base_dir = os.path.dirname(self.output_file_path)
        freq_resolution=1

        # Threshold dictionary for easy access by label name
        thresholds = {'A': self.A_thresh, 'B': self.B_thresh, 'D': self.D_thresh,
                    '20Hz': self.TwentyHz_thresh, '40Hz': self.FourtyHz_thresh}
        
        predictions = []
        # Normalize spectrogram and convert to tensor
        normalized_S_dB = (spectrogram_data - np.min(spectrogram_data)) / (np.max(spectrogram_data) - np.min(spectrogram_data)) 
        S_dB_img = Image.fromarray((normalized_S_dB * 255).astype(np.uint8), 'L')
        image = ImageOps.flip(S_dB_img)
        # Convert the image to a numpy array for processing
        img_array = np.array(image)

        if self.CalCOFI_flag:
        # CalCOFI Sonobuoy cleanup for AIS

            threshold_1 = 200  # Threshold for the first 10 pixel block
            threshold_2 = 180  # Threshold for the second 10 pixel block
            threshold_3 = 160  # Lower threshold for the third 10 pixel block
            # Gray value to replace the AIS signal
            gray_value = 128  # Mid-gray
            # Find the vertical white lines and gray them out
            # Loop through each column (time slice) in the spectrogram
            for col in range(img_array.shape[1]):  # Loop through each column
            # Check first 10 pixel block (corresponding to the lowest frequency band)
                if np.sum(img_array[-10:, col]) > threshold_1 * 10:
                    # If the first 10 pixel block passes, check the second 10 pixel block
                    if np.sum(img_array[-20:-10, col]) > threshold_2 * 10:
                        # If the second block passes, check the third block with a lower threshold
                        if np.sum(img_array[-30:-20, col]) > threshold_3 * 10:
                            # If all conditions are met, gray out the entire column
                            img_array[:, col] = gray_value  # Replace the entire column with gray 
        
        final_image = Image.fromarray(img_array)

        timestamp_str = start_time.strftime('%Y%m%dT%H%M%S')
        image_filename = f"{self.stream_name}_{timestamp_str}.png"
        image_path = os.path.join(csv_base_dir, image_filename)

        # Convert to tensor
        S_dB_tensor = F.to_tensor(final_image).unsqueeze(0).to(self.device)        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(S_dB_tensor)

        # Extract prediction results
        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        # Apply Non-Maximum Suppression (NMS)
        keep_indices = ops.nms(boxes, scores, 0.05)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
    
        # Iterate through detections
        for box, score, label in zip(boxes, scores, labels):
            textual_label = self.inverse_label_mapping.get(label.item(), 'Unknown')
            if score.item() < thresholds.get(textual_label, 0):
                continue

            # Get box time offsets
            start_offset_sec = box[0].item() * self.time_per_pixel
            end_offset_sec = box[2].item() * self.time_per_pixel

            start_datetime = start_time + timedelta(seconds=start_offset_sec)
            end_datetime = start_time + timedelta(seconds=end_offset_sec)
        
            predictions.append({
                'image_file_path': image_path,
                'label': textual_label,
                'score': round(score.item(), 2),
                'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'min_frequency': round(150 - box[3].item() * freq_resolution),
                'max_frequency': round(150 - box[1].item() * freq_resolution),
                'box_x1': box[0].item(),
                'box_x2': box[2].item(),
                'box_y1': box[1].item(),
                'box_y2': box[3].item()
                })
        
        final_image = Image.fromarray(img_array).convert('RGB')
        final_image = self.plot_one_annotated_spectrogram(final_image, predictions)
        #final_image.show()
        final_image.save(image_path)
        
        return predictions
    
    def plot_one_annotated_spectrogram(self, image, predictions):
        """
        Plot a single annotated spectrogram with bounding boxes and labels.
        
        Parameters:
        - image: PIL Image object representing the spectrogram.
        - predictions: DataFrame containing predictions for the spectrogram.
        """
        # Load the spectrogram image
        #image = Image.open(spectrogram_path)
        draw = ImageDraw.Draw(image)  # Create a drawing context
        #font = ImageFont.truetype("arial.ttf", 8)  # Adjust the font and size as needed
        font = ImageFont.load_default()

        # Plot each bounding box and label for this spectrogram
        for row in predictions:
            xmin, ymin, xmax, ymax = row['box_x1'], row['box_y1'], row['box_x2'], row['box_y2']
            label = row['label']
            score = row['score']
            # Format the label and score together
            label_text = f"{label} ({score:.2f})"

            # Draw rectangle on the image (Color options: https://pillow.readthedocs.io/en/stable/releasenotes/7.1.0.html#added-140-html-color-names)
            #For Kasey knowledge: A,B, D are all from blue whales. We will make them all blue.
            if label == 'D' or label == 'A' or label == 'B':
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='cyan', width=3)
                # Optionally draw the label near the bounding box
                draw.text((xmin, ymin - 17), label_text, fill='cyan', font=font)
            elif label == '40Hz'or label == '20Hz':
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline='pink', width=3)
                draw.text((xmin, ymin - 17), label_text, fill='pink', font=font)
            else:
                print(f"Unknown label: {label}. Skipping drawing for this label.")
                continue

        return image 

    def process_audio(self, audio_view, start_time):
        """
        Get data -> make sg -> inference -> output results
        """
        inference_start_time = time.time()
        #Flatten ring buffer view
        if isinstance(audio_view, tuple):
            audio_np = np.concatenate(audio_view, axis=0)
        else:
            audio_np = audio_view
        audio_np = audio_np.astype(np.int32)
        audio_np = audio_np.reshape(-1)
        #Compute exact offset to get the actual start time you want from the window
        total_samples = int(self.sample_rate * self.duration_ms / 1000)
        sample_offset = int((self.use_time - start_time).total_seconds() * self.sample_rate)
        #Validate bounds
        if sample_offset < 0 or sample_offset + total_samples > len(audio_np):
            print(f"{self.name} Warning: sample slice out of bounds")
            return

        #Slice and prepare input
        sliced_audio = audio_np[sample_offset:sample_offset + total_samples].astype(np.float32)
        audio_tensor = torch.tensor(sliced_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        spectrogram = audio_to_spectrogram(audio_tensor, self.sample_rate, self.device, 10, 150)

        #Predict and output
        if self.file_output_bool:
            predictions = self.predict_and_save_spectrogram(spectrogram, self.use_time)
            with open(self.output_file_path, mode='a', encoding='utf-8') as txtfile:
                for event in predictions:
                    event['source'] = self.stream_name
                    event['model_no'] = self.model_name
                    txtfile.write('\t'.join(str(event[field]) for field in self.fieldnames) + '\n')
        else:
            predictions = self.predict_spectrogram(spectrogram, start_time)

        self.last_inference_time = time.time() - inference_start_time
        print(f"{self.name} Inference complete. Took {self.last_inference_time:.3f}s. Processed {len(predictions)} predictions.")

        if predictions:
            print(f"{'Label':<10}{'Score':<10}{'Start Time':<20}{'End Time'}")
            print("-" * 60)
            for p in predictions:
                print(f"{p['label']:<10}{p['score']:<10}{p['start_time']:<20}{p['end_time']}")
        else:
            print("No predictions for this window.")








        