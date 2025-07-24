import os
from PIL import Image
import numpy as np

# Define folders
rpi_folder = r"C:\Users\Kasey\OneDrive - UC San Diego\Lab Work\RealTimeBaleen\Spectrograms_for_Compare\TimingTest2"
sgs_folder = r"C:\Users\Kasey\Desktop\TestMichaelaProgram\GroundTruthTests\SGs_NoLabels"
output_folder = r"C:\Users\kasey\OneDrive - UC San Diego\Lab Work\RealTimeBaleen\Spectrograms_for_Compare\Subtracted_Output_2kHz_fixed"

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Get and sort file names (ignore 'output' and keep only PNGs)
rpi_files = sorted([f for f in os.listdir(rpi_folder) if f.endswith('.png') and 'output' not in f])
sgs_files = sorted([f for f in os.listdir(sgs_folder) if f.endswith('.png') and 'output' not in f])

# Loop over files in RPI only
for i, rpi_file in enumerate(rpi_files):
    try:
        rpi_path = os.path.join(rpi_folder, rpi_file)
        sgs_path = os.path.join(sgs_folder, sgs_files[i])  # Match by index

        img_rpi = Image.open(rpi_path).convert("L")
        img_sgs = Image.open(sgs_path).convert("L")

        # Convert to numpy arrays
        arr_rpi = np.array(img_rpi)
        arr_sgs = np.array(img_sgs)

        # Pad the smaller one to match dimensions
        h1, w1 = arr_rpi.shape
        h2, w2 = arr_sgs.shape

        target_h = max(h1, h2)
        target_w = max(w1, w2)

        def pad_to_shape(arr, target_h, target_w):
            pad_h = target_h - arr.shape[0]
            pad_w = target_w - arr.shape[1]
            return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        arr_rpi = pad_to_shape(arr_rpi, target_h, target_w)
        arr_sgs = pad_to_shape(arr_sgs, target_h, target_w)

        # Subtract and save
        diff_array = np.abs(arr_rpi.astype(int) - arr_sgs.astype(int)).astype(np.uint8)
        diff_img = Image.fromarray(diff_array)
        diff_img.save(os.path.join(output_folder, f"diff_{rpi_file}"))

    except IndexError:
        print(f"No matching SGs image for {rpi_file}")
        break

print("Done subtracting with padding.")