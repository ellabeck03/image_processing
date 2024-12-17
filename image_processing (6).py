# #image_processing.py
# #used to find the number of damaged pixels present in a video, suitable for gamma dosimetry monitoring in cmos cameras

import os
from os.path import exists
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import requests
import sys

from zipfile import ZipFile
from urllib.request import urlretrieve
from collections import deque
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from IPython.display import Image


def download_video_from_url(url, filename):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded video as {filename}")
    else:
        print("Failed to download video")
    return filename

def load_video_frames(filename, frames_start = None, frames_end = None):
    cap = cv2.VideoCapture(filename)
    frames = []
        
    if frames_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_start)
    
    frame_idx = frames_start or 0
    while cap.isOpened():
        if frames_end and frame_idx >= frames_end:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        
        frame_idx += 1
            
    cap.release()
    return frames

def get_video_frames_from_url(url, local_filename='temp_video.avi'):
    download_video_from_url(url, local_filename)
    return load_video_frames(local_filename)


def detect_damaged_pixels(frames, plot=True, consecutive_threshold=2, mse_threshold=100, ssim_threshold = 0.1, brightness_threshold = 100):
    num_frames = len(frames)
    height, width = frames[0].shape  # Dimensions of the frame
    
    damaged_pixel_masks = []
    damaged_pixel_counts = []

    # Define parameters
    threshold_multiplier_low = 10
    threshold_multiplier_high = 1.2
    threshold_multiplier_3 = 1.2
    threshold_multiplier_4 = 1.05
    max_cluster_size = 100  # Adjust as needed
    min_cluster_size = 1

    #for i in range(num_frames):
    for i in tqdm(range(num_frames), desc = 'processing frames', unit = 'frame'):
        current_frame = frames[i]
        
        # Detect global scene changes
        if i > 0:
            ssim_score = compute_ssim(frames[i], frames[i - 1])
            if ssim_score < ssim_threshold:
                print(f'Frame {i} skipped due to global scene change')
                #damaged_pixel_masks.append(np.zeros((height, width), dtype=bool))
                damaged_pixel_masks.append(None)
                damaged_pixel_counts.append(np.nan)
                continue

        # Determine window frames for averaging (exclude the current frame)
        start = max(0, i - 3)
        end = min(num_frames, i + 4)
        window_frames = np.array(frames[start:i] + frames[i+1:end])

        # Calculate the mean per pixel
        background = np.mean(window_frames, axis=0)
        
        # Determine threshold multiplier based on background brightness levels
        threshold_multiplier = np.where(
            background < 30, threshold_multiplier_low,
            np.where(background > 190, threshold_multiplier_4,
            np.where(background > 170, threshold_multiplier_3, threshold_multiplier_high))
        )

        # Detect damaged pixels by comparing to the background
        damaged_pixels = current_frame > (threshold_multiplier * background)
        damaged_pixels_uint8 = damaged_pixels.astype(np.uint8)
        
        # Filter out clusters of damaged pixels
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damaged_pixels_uint8, connectivity=8)
        filtered_damaged_pixels = np.zeros_like(damaged_pixels, dtype=bool)
        
        for label in range(1, num_labels):  # Skip background label (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if min_cluster_size <= area <= max_cluster_size:
                filtered_damaged_pixels[labels == label] = True
                
        #remove detected damaged pixels which lie in bright regions
        bright_background_mask = (background > brightness_threshold)
        for row in range(1, height - 1):  # Skip edge rows
            for col in range(1, width - 1):  # Skip edge columns
                if filtered_damaged_pixels[row, col]:
                    neighbours = [
                        bright_background_mask[row-1, col],  # Above
                        bright_background_mask[row+1, col],  # Below
                        bright_background_mask[row, col-1],  # Left
                        bright_background_mask[row, col+1]   # Right
                    ]
                    if np.any(neighbours):
                        filtered_damaged_pixels[row, col] = False

        # Apply filtered mask to remove large regions from the image
        large_regions = np.zeros_like(current_frame, dtype=current_frame.dtype)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] > max_cluster_size:
                large_regions[labels == label] = current_frame[labels == label]
                current_frame[labels == label] = 0  # Remove large regions by setting to 0
                

        damaged_pixel_masks.append(filtered_damaged_pixels)
            
    # Process consecutive damaged pixels
    filtered_damaged_pixel_counts = filter_consecutive_damaged_pixels(damaged_pixel_masks, consecutive_threshold)
    
    
    bright_area_estimates = []
    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is None:
            bright_area_estimates.append(np.nan)
            continue
        
        high_brightness_mask = (frame > brightness_threshold) & ~mask
        
        if np.sum(high_brightness_mask) > 0:
            estimate = estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, filtered_damaged_pixel_counts)
            bright_area_estimates.append(estimate[i])
        else:
            bright_area_estimates.append(np.nan)
    
    #calculate heatmap of damaged pixels
    heatmap = np.zeros((height, width), dtype = np.float32)
    valid_pixel_counts = np.zeros((height, width), dtype = np.float32)
    
    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is not None:
            heatmap += mask.astype(np.float32)
            high_brightness_mask = (frame > brightness_threshold) & ~mask
            valid_pixel_counts += ~high_brightness_mask.astype(np.float32)

    heatmap = np.divide(heatmap, valid_pixel_counts, out = np.zeros_like(heatmap), where = valid_pixel_counts > 0) * 100 # heatmap in percentage form

    
    #call plotting functions
    visualize_damaged_pixels(frames[0], damaged_pixel_masks[0], filtered_damaged_pixel_counts[0], estimate_count = bright_area_estimates[0])
    plot_heatmap(heatmap, title = "Damaged Pixel Heatmap")
    
    
    #bright_area_estimates = estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, filtered_damaged_pixel_counts)
    
    print(filtered_damaged_pixel_counts[0])
    print(bright_area_estimates[0])
    first_bright_area = (frames[0] > brightness_threshold) & ~damaged_pixel_masks[0]
    area = np.sum(first_bright_area) / frames[0].size
    print(area)
    
    
    total_damaged_pixel_counts = [actual + estimate if not np.isnan(estimate) else actual
                                  for actual, estimate in zip(filtered_damaged_pixel_counts, bright_area_estimates)]
    
    if plot:
        plot_damaged_pixels(total_damaged_pixel_counts)

    return total_damaged_pixel_counts


def compute_ssim(frame1, frame2):
    score, _ = ssim(frame1, frame2, full = True)
    return score


def filter_consecutive_damaged_pixels(damaged_pixel_masks, consecutive_threshold):
    num_frames = len(damaged_pixel_masks)
    height, width = damaged_pixel_masks[0].shape
    filtered_counts = []
    
    consecutive_counts = np.zeros((height, width), dtype=int)

    for mask in damaged_pixel_masks:
        if mask is None:
            filtered_counts.append(np.nan)
            continue
        consecutive_counts[mask] += 1
        consecutive_counts[~mask] = 0
        filtered_mask = mask & (consecutive_counts < consecutive_threshold)
        filtered_counts.append(np.sum(filtered_mask))
    
    return filtered_counts
    
    
def estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, filtered_damaged_pixel_counts, brightness_threshold =100):
    #print('estimation function called')
    estimated_damaged_pixel_counts = []
    
    for i, (frame, mask, filtered_damaged_count) in enumerate(zip(frames, damaged_pixel_masks, filtered_damaged_pixel_counts)):
        if mask is None:
            estimated_damaged_pixel_counts.append(np.nan)
            continue
            
        #identify low and high brightness regions
        low_brightness_mask = frame < brightness_threshold
        low_brightness_mask &~ mask
        high_brightness_mask = ~low_brightness_mask
        low_brightness_damaged_pixels = mask & low_brightness_mask
        
        
        low_brightness_area = np.sum(low_brightness_mask)
        high_brightness_area = np.sum(high_brightness_mask)
        
        if low_brightness_area > 0:
            damaged_pixel_density = np.sum(low_brightness_damaged_pixels) / low_brightness_area
            estimated_high_brightness_damaged_pixels = damaged_pixel_density * high_brightness_area
        else:
            estimated_high_brightness_damaged_pixels = np.nan
            
        estimated_damaged_pixel_counts.append(estimated_high_brightness_damaged_pixels)
        
    return estimated_damaged_pixel_counts


def create_test_video(num_frames = 1000, width = 928, height = 576, damaged_pixel_count = 1000, grid_size = 20):
    frames = []
    
    grid_rows = int(np.sqrt(damaged_pixel_count * height / width))
    grid_cols = int(damaged_pixel_count / grid_rows)

    grid_coordinates = [(y * (height // grid_rows), x * (width // grid_cols)) for y in range(grid_rows) for x in range(grid_cols)]
    
    def generate_damaged_pixel_pattern(offset = 0):
        damaged_pixels = []
        
        for y, x in grid_coordinates:
            damaged_pixels.append((y + offset, x + offset))
        return damaged_pixels
    
    for i in range(num_frames):
        base_brightness = 50 + 10 * np.sin(2 * np.pi * i / 50)
        noise = np.random.normal(loc=0, scale=1, size=(height, width))  # Add Gaussian noise
        frame = np.clip(base_brightness + noise, 0, 255).astype(np.uint8)
        
        frame[: height //  2, : width // 2] = 150
        
        damaged_pixels = generate_damaged_pixel_pattern(offset=0) if i % 2 == 0 else generate_damaged_pixel_pattern(offset=1)
        
        for y, x in damaged_pixels:
            if 0 <= y < height and 0 <= x < width:
                frame[y, x] = 200
                
        frames.append(frame)
        
    return frames, len(damaged_pixels)


def save_frames_to_video(frames, filename='test_video.avi', fps=22):
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

    # Write each frame to the video
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved as {filename}")
    

def visualize_damaged_pixels(frame, damaged_pixels, frame_index, estimate_count = None, bright_threshold = 100):
    height, width = frame.shape
    
    bright_areas = frame > bright_threshold
    highlighted_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    #add true damaged pixels in blue
    damaged_pixels_colored = np.zeros((height, width, 3), dtype=np.uint8)
    damaged_pixels_colored[damaged_pixels] = [255, 0, 0]
    highlighted_frame = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), 1.0, damaged_pixels_colored, 1.0, 0)
    
    #highlight bright pixel mask
    bright_pixels_coloured = np.zeros((height, width, 3), dtype = np.uint8)
    bright_pixels_coloured[bright_areas] = [0, 255, 0]
    highlighted_frame = cv2.addWeighted(highlighted_frame, 1.0, bright_pixels_coloured, 1.0, 0)
    
    #scatter red pixels for estimated damaged pixels
    if estimate_count and bright_areas.any():
        bright_coords = np.column_stack(np.where(bright_areas))
        if len(bright_coords) > estimate_count:
            selected_coords = bright_coords[np.random.choice(len(bright_coords), round(estimate_count), replace = False)]
            
        else:
            selected_coords = bright_coords
            
        for coord in selected_coords:
            highlighted_frame[coord[0], coord[1]] = [0, 0, 255]
            
    plt.figure(figsize = (15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(frame, cmap = 'gray')
    plt.title(f'Original Frame {frame_index}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Damaged Pixels Highlighted {frame_index}')
    plt.axis('off')
    plt.show()
    
    
def plot_heatmap(heatmap, title = "Damaged Pixel Heatmap"):
    plt.figure(figsize = (15, 10))
    plt.ismhow(heatmap, cmap = 'viridis', interpolation ='nearest')
    plt.colorbar(label = "Percentage of frames (%)")
    plt.title(title)
    plt.show()
    
    
def plot_damaged_pixels(damaged_pixel_counts):
    """Plots the count of damaged pixels across frames."""
    plt.figure(figsize=(10, 5))
    plt.plot(damaged_pixel_counts, label='Damaged Pixels Count', color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Damaged Pixels')
    plt.title('Damaged Pixels Detected Over Time')
    plt.legend()
    plt.show()
     
    
def average_per_second(data, frames, fps = 22):
    averages = [sum(data[i:i+fps]) / fps for i in range(0, len(data), fps)]
    times = list(range(len(averages)))
    return averages, times
    