import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def get_hsv_from_image(img, num_samples=1000):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    ys = np.random.randint(0, h, size=num_samples)
    xs = np.random.randint(0, w, size=num_samples)
    return hsv[ys, xs]

def remove_outliers(samples, z_thresh=2.0):
    # Compute standard deviation along each channel.
    std = np.std(samples, axis=0)
    # If all std values are zero, the samples are constant, so no outliers to remove.
    if np.all(std == 0):
        return samples
    mean = np.mean(samples, axis=0)
    z_scores = np.abs((samples - mean) / std)
    filtered = samples[np.all(z_scores < z_thresh, axis=1)]
    # If filtering removes everything, return the original samples.
    if filtered.size == 0:
        return samples
    return filtered
    
def compute_hsv_range(samples):
    lower = np.min(samples, axis=0)
    upper = np.max(samples, axis=0)
    return tuple(lower), tuple(upper)

# Open file dialog
root = tk.Tk()
root.withdraw()  # Hide main window
file_paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

if not file_paths:
    print("No images selected.")
    exit()

all_samples = []

for path in file_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load image: {path}")
        continue
    samples = get_hsv_from_image(img)
    all_samples.append(samples)

if not all_samples:
    print("No valid images were processed.")
    exit()

# Combine and filter
all_samples = np.vstack(all_samples)
filtered = remove_outliers(all_samples)

# Get HSV bounds
lower_hsv, upper_hsv = compute_hsv_range(filtered)

print("✅ HSV Range (OpenCV format):")
print("lower_hsv =", tuple(int(x) for x in lower_hsv), ", upper_hsv =", tuple(int(x) for x in upper_hsv))
