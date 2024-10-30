# #image_processing.py
# #used to find the number of damaged pixels present in a video, suitable for gamma dosimetry monitoring in cmos cameras

import os
from os.path import exists
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image


def detect_damaged_pixels(video_url, video_filename, plot = False):
    #Download the video file if it doesn't exist
    if not os.path.exists(video_filename):
        print("Downloading video...")
        urlretrieve(video_url, video_filename)
        print("Download complete.")

    # Open the video
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize variables
    damaged_pixel_counts = []
    frames = []
    pixel_history = {}
    #pixel_history = np.zeros((height, width), dtype = np.uint8)

    # Define parameters
    window_size = 2
    min_consecutive_frames = 1
    threshold_multiplier_low = 9  # for low brightness
    threshold_multiplier_high = 1.2  # for higher brightness
    threshold_multiplier_3 = 1.07
    threshold_multiplier_4 = 1.05
    #threshold = 40

    # Read all frames into grayscale for faster processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert directly to grayscale

    cap.release()

    # Process each frame with a sliding window for averaging brightness
    num_frames = len(frames)
    #num_frames = 5
    height, width = frames[0].shape  # Dimensions of the frame
    
    #pixel_history = np.zeros((height, width), dtype = np.uint8)

    for i in range(num_frames):
        current_frame = frames[i]
        damaged_pixel_count = 0
        damaged_pixels = np.zeros_like(current_frame)  # Mask for damaged pixels

        # Calculate adjacent frame brightness using sliding window
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)
        adjacent_brightness = np.mean(frames[start:end], axis=0)  # Average brightness of surrounding frames
        #adjacent_brightness = np.median(frames[start:end], axis = 0)

        # Determine threshold based on average brightness
        threshold_multiplier = np.where(
            adjacent_brightness < 30, threshold_multiplier_low,
            np.where(adjacent_brightness > 190, threshold_multiplier_4,
            np.where(adjacent_brightness > 170, threshold_multiplier_3, threshold_multiplier_high)))
        
        # Identify damaged pixels
        damaged_pixels_mask = current_frame > (adjacent_brightness * threshold_multiplier)
        #damaged_pixel_count = np.sum(damaged_pixels_mask)
        
        # Track pixel history and update damaged pixels mask
        if i not in pixel_history:
            pixel_history[i] = np.zeros_like(current_frame, dtype=np.uint8)
            
        new_damaged_pixels = np.logical_and(damaged_pixels_mask, pixel_history[i] < min_consecutive_frames)
        #new_damaged_pixels = np.logical_and(damaged_pixels_mask, damaged_pixels_mask == 1)
        damaged_pixel_count = np.sum(new_damaged_pixels)
        pixel_history[i] += damaged_pixels_mask.astype(np.uint8)

        # Reset pixels that are damaged for too long
        pixel_history[i][pixel_history[i] > min_consecutive_frames] = 0

        # Append the count for this frame
        damaged_pixel_counts.append(damaged_pixel_count)

#         # Identify damaged pixels
#         damaged_pixels_mask = current_frame > (adjacent_brightness * threshold_multiplier)
        
#         # Update pixel history for damaged pixels
#         pixel_history = np.where(damaged_pixels_mask, pixel_history + 1, 0)
        
#         # Count pixels that have been damaged for the required consecutive frames
#         confirmed_damaged_pixels = pixel_history >= min_consecutive_frames
#         damaged_pixel_count = np.sum(confirmed_damaged_pixels)

#         # Append the count for this frame
#         damaged_pixel_counts.append(damaged_pixel_count)

        # damaged_pixels_mask = current_frame > (adjacent_brightness * threshold_multiplier)
        # pixel_history = np.where(damaged_pixels_mask, pixel_history + 1, 0)
        # confirmed_damaged_pixels = pixel_history >= min_consecutive_frames
        # damaged_pixel_count = np.sum(confirmed_damaged_pixels)
        # damaged_pixel_counts.append(damaged_pixel_count)
        # #pixel_history[confirmed_damaged_pixels] = 0
        
#         pixel_history[i] += damaged_pixels_mask.astype(np.uint8)
#         reset_mask = pixel_history[i] > min_consecutive_frames
#         damaged_pixels_mask[reset_mask] = 0  # Reset pixels that are damaged for too long

#         # Update damaged pixel count and append to list
#         damaged_pixel_counts.append(damaged_pixel_count)

        # For visualization using Matplotlib
        if i < 5:  # Only show the first few frames for speed
            plt.figure(figsize=(15, 8))

            # Original Frame
            plt.subplot(1, 2, 1)
            plt.imshow(current_frame, cmap='gray')
            plt.title(f'Original Frame {i}')
            plt.axis('off')

            # Damaged Pixels Highlighted
            damaged_pixels_colored = np.zeros((height, width, 3), dtype=np.uint8)
            damaged_pixels_colored[damaged_pixels_mask] = [255, 0, 0]  # Red for damaged pixels
            highlighted_frame = cv2.addWeighted(cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR), 1.0, damaged_pixels_colored, 0.4, 0)
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(highlighted_frame, cv2.COLOR_BGR2RGB))
            plt.title(f'Damaged Pixels Highlighted {i}')
            plt.axis('off')
            
            plt.show()

            print(f"Frame {i}: {damaged_pixel_count} damaged pixels detected.")

    # Plot the number of damaged pixels per frame
    if plot:
        plt.figure(figsize=(15, 8))
        plt.plot(damaged_pixel_counts, label='Damaged Pixels Count', color='blue')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Damaged Pixels')
        plt.title('Damaged Pixels Detected Over Time')
        plt.legend()
        plt.show()
    
    return damaged_pixel_counts


def damaged_pixels_per_second(damaged_pixel_counts, fps = 22):
    num_seconds = len(damaged_pixel_counts) // fps
    seconds = np.arange(0, num_seconds)
    print(num_seconds)
    
    mean_damaged_pixels_per_second = [np.mean(damaged_pixel_counts[i * fps: (i+1) * fps]) for i in range(num_seconds)]
    
    print(len(seconds))
    print(len(mean_damaged_pixels_per_second))
    
    linfit = np.polyfit(seconds, mean_damaged_pixels_per_second, 0)
    #fit = linfit[0] * seconds + linfit[1]
    fit = np.full((len(seconds)), linfit[0])
    
    
    plt.figure(figsize = (15, 8))
    plt.scatter(seconds, mean_damaged_pixels_per_second, label = 'Mean Damaged Pixels per Second', color = 'red', marker = '+')
    plt.plot(seconds, fit, color = 'blue')
    plt.title('Mean Damaged Pixels Detected per Second')
    plt.legend()
    plt.show()
    
    return mean_damaged_pixels_per_second


def create_test_video(filename = 'test_video.avi', num_frames = 1000, width = 928, height = 576, white_pixels_per_frame = 25):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 22.0, (width, height), isColor = False)
    
    halfway_point = width // 2
    
    for _ in range(num_frames):
        #start with black frame
        frame = np.full((height, width), 25, dtype = np.uint8)
        
#         #make around half of the pixels brighter for low contrast testing
#         for _ in range(round((width * height) / 2)):
#             low_contrast_x = np.random.randint(0, width)
#             low_contrast_y = np.random.randint(0, height)
#             frame[low_contrast_y, low_contrast_x] = 150
        
#         #randomly select pixel positions and make them white
#         for _ in range(white_pixels_per_frame):
#             x = np.random.randint(0, width)
#             y = np.random.randint(0, height)
#             frame[y, x] = 255

        frame[:, halfway_point:] = 200
        
        # Randomly select unique indices for the white pixels within the frame
        white_indices = np.random.choice(width * height, white_pixels_per_frame, replace=False)
        frame.flat[white_indices] = 255  # Set these pixels to 255
            
        #write frame to video
        out.write(frame)
        
    out.release()