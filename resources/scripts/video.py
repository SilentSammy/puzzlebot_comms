import cv2
import os
from glob import glob

# Parameters
output_video = 'output_video.mp4'  # Changed to .mp4
fps = 10
image_extension = '*.png'

# Get list of images in the current directory
images = sorted(glob(image_extension))

if not images:
    print("No images found in the current directory.")
    exit()

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add images to the video
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release the video writer
video.release()
print(f"Video saved as {output_video}")
