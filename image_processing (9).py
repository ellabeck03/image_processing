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
from numba import njit, prange

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

#@njit(parallel = True)
def detect_damaged_pixels(frames, plot=True, consecutive_threshold=5, mse_threshold=100, ssim_threshold = 0.1, brightness_threshold = 170):
    frames = [np.array(frame) for frame in frames]
    num_frames = len(frames)
    height, width = frames[0].shape[:2]  # Dimensions of the frame
    
    damaged_pixel_masks = []
    damaged_pixel_counts = []

    min_cluster_size = 1
    max_cluster_size = 300

    for i in tqdm(range(num_frames), desc = 'processing frames', unit = 'frame'):
        current_frame = frames[i]
        
        # Detect global scene changes
        if i > 0:
            ssim_score = compute_ssim(frames[i], frames[i - 1])
            if ssim_score < ssim_threshold:
                print(f'Frame {i} skipped due to global scene change')
                damaged_pixel_masks.append(None)
                damaged_pixel_counts.append(np.nan)
                continue

        # Determine window frames for averaging (exclude the current frame)
        start = max(0, i - 3)
        end = min(num_frames, i + 4)
        window_frames = np.array(frames[start:i] + frames[i+1:end])
        
        #exclude potentially damaged pixels from background calculation
        background = find_background(window_frames)
        
        # Calculate the standard deviation for thresholding
        std_dev = np.std(window_frames, axis=0)
        
        # Get damaged pixel mask
        damaged_pixels_uint8, threshold = get_damaged_pixel_mask(current_frame, height, width, background)
        
        # Filter out clusters of damaged pixels
        filtered_damaged_pixels = filter_damaged_pixel_clusters(damaged_pixels_uint8, min_cluster_size, max_cluster_size)
                
        #remove detected damaged pixels which lie in bright regions
        filtered_damaged_pixels = remove_bright_regions(background, threshold, height, width, filtered_damaged_pixels)

        damaged_pixel_masks.append(filtered_damaged_pixels)
            
    # Process consecutive damaged pixels
    filtered_damaged_pixel_counts = filter_consecutive_damaged_pixels(damaged_pixel_masks, consecutive_threshold)
    
    # find bright area estimates
    bright_area_estimates = find_bright_area_estimates(frames, damaged_pixel_masks, np.array(filtered_damaged_pixel_counts), brightness_threshold)

    
    total_damaged_pixel_counts = [actual + estimate if not np.isnan(estimate) else actual
                                  for actual, estimate in zip(filtered_damaged_pixel_counts, bright_area_estimates)]
    
    if plot:
        visualize_damaged_pixels(frames[0], damaged_pixel_masks[0], filtered_damaged_pixel_counts[0], estimate_count = bright_area_estimates[0])
        visualize_damaged_pixels(frames[1], damaged_pixel_masks[1], filtered_damaged_pixel_counts[1], estimate_count = bright_area_estimates[1])
        visualize_damaged_pixels(frames[2], damaged_pixel_masks[2], filtered_damaged_pixel_counts[2], estimate_count = bright_area_estimates[2])
        
        #calculate heatmap of damaged pixels
        # heatmap = find_damaged_pixel_heatmap(height, width, frames, damaged_pixel_masks, threshold)#check this threshold
        # plot_heatmap(heatmap, title = "Damaged Pixel Heatmap")
        
        plot_damaged_pixels(total_damaged_pixel_counts)

    return total_damaged_pixel_counts 


def compute_ssim(frame1, frame2):
    score, _ = ssim(frame1, frame2, full = True, data_range = 255)
    return score


def find_background(frames):
    pixel_means = np.mean(frames, axis = 0)
    pixel_std = np.std(frames, axis = 0)
    background = []
    
    valid_background_pixels = frames <= (pixel_means + (2 * pixel_std))
    result = np.where(valid_background_pixels, frames, np.nan)
    background = np.nanmean(result, axis = 0)
    
    if np.isnan(background).any():
        print(f'background not accurately determined for frame {i}')
        background = np.nan_to_num(background)

    background = np.array(background)
    
    return background


@njit(parallel = True)
def get_damaged_pixel_mask(frame, height, width, background):
    damaged_pixels = np.zeros_like(frame, dtype=np.bool_)
    
    for row in prange(height):
        for col in prange(width):
            #condition 1: pixel brightness should exceed background by a shreshold scaled inversely with background brightness
            threshold = 30 + (background[row, col] / 255) * (255 - 30)
            
            if frame[row, col] > threshold:
                #condition 2: pixel's brightness should exceed mean of its neighbours in a 3x3 kernel
                kernel = frame[max(row - 1, 0) : min(row + 2, height), max(col - 1, 0) : min(col + 1, width)]
                kernel_mean = np.mean(kernel)
                
                if frame[row, col] > kernel_mean:
                    damaged_pixels[row, col] = True
                    
    damaged_pixels_uint8 = damaged_pixels.astype(np.uint8)
    
    return damaged_pixels_uint8, threshold


#@njit(parallel = True)
def filter_damaged_pixel_clusters(damaged_pixel_mask, min_cluster_size, max_cluster_size):
    filtered_damaged_pixels = np.zeros_like(damaged_pixel_mask, dtype = np.bool_)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damaged_pixel_mask, connectivity = 8)
    
    for label in prange(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        
        if min_cluster_size <= area <= max_cluster_size:
            filtered_damaged_pixels[labels == label] = True
            
    return filtered_damaged_pixels


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


@njit(parallel = True)
def remove_bright_regions(background, brightness_threshold, height, width, filtered_damaged_pixels):
    bright_background_mask = (background > brightness_threshold)
    
    for row in prange(1, height - 1):
        for col in prange(1, width - 1):
            if filtered_damaged_pixels[row, col]:
                neighbours = [
                    bright_background_mask[row - 1, col],
                    bright_background_mask[row + 1, col],
                    bright_background_mask[row, col - 1],
                    bright_background_mask[row, col + 1]
                ]
                
                if bright_background_mask[row - 1, col] or \
                   bright_background_mask[row + 1, col] or \
                   bright_background_mask[row, col - 1] or \
                   bright_background_mask[row, col + 1]:

                    filtered_damaged_pixels[row, col] = False
                    
    return filtered_damaged_pixels


@njit(parallel=True)
def estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, filtered_damaged_pixel_counts, brightness_threshold=170):
    #print('estimation function called')
    num_frames = len(frames)
    frame_shape = frames[0].shape
    estimated_damaged_pixel_counts = np.full(num_frames, np.nan, dtype=np.float64)
    
    # Preprocess masks
    processed_masks = np.zeros((num_frames, frame_shape[0], frame_shape[1]), dtype=np.bool_)
    for i in range(num_frames):
        if damaged_pixel_masks[i] is not None:
            processed_masks[i] = damaged_pixel_masks[i]
            
    for i in prange(len(frames)):
        frame = frames[i]
        mask = processed_masks[i]
        
        # Identify low and high brightness regions, excluding existing damaged pixels
        low_brightness_mask = (frame < brightness_threshold) & ~mask
        high_brightness_mask = (frame >= brightness_threshold) & ~mask
        
        # Calculate areas
        low_brightness_area = np.sum(low_brightness_mask)
        high_brightness_area = np.sum(high_brightness_mask)
        #print(low_brightness_area)
        #print(low_brightness_mask)
        
        if low_brightness_area > 0:
            # Density of damaged pixels in low-brightness areas
            damaged_pixel_density = np.sum(mask) / low_brightness_area
            #print(damaged_pixel_density)
            
            # Estimate damaged pixels in high-brightness areas
            estimated_high_brightness_damaged_pixels = round(damaged_pixel_density * high_brightness_area)
        else:
            estimated_high_brightness_damaged_pixels = np.nan
            
        # Assign the estimated count
        estimated_damaged_pixel_counts[i] = estimated_high_brightness_damaged_pixels

    return estimated_damaged_pixel_counts

@njit(parallel = True)
def find_bright_area_estimates(frames, damaged_pixel_masks, damaged_pixel_counts, brightness_threshold):
    bright_area_estimates = np.full(len(frames), np.nan, dtype = np.float64)
    
    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is None:
            bright_area_estimates[i] = np.nan
            continue

        high_brightness_mask = (frame > brightness_threshold) & ~mask
        
        if np.sum(high_brightness_mask) > 0:
            estimate = estimate_damaged_pixels_in_bright_areas(frames, damaged_pixel_masks, damaged_pixel_counts)
            bright_area_estimates[i] = estimate[i]
        else:
            bright_area_estimates[i] = np.nan
            
    return bright_area_estimates

@njit(parallel = True)
def find_damaged_pixel_heatmap(height, width, frames, damaged_pixel_masks, brightness_threshold):
    heatmap = np.zeros((height, width), dtype = np.float32)
    valid_pixel_counts = np.zeros((height, width), dtype = np.float32)
    
    for i, (frame, mask) in enumerate(zip(frames, damaged_pixel_masks)):
        if mask is not None:
            heatmap += mask.astype(np.float32)
            high_brightness_mask = (frame > brightness_threshold) & ~mask
            valid_pixel_counts += (~high_brightness_mask).astype(np.float32)
            
    #heatmap = np.divide(heatmap, valid_pixel_counts, out = np.zeros_like(heatmap), where = valid_pixel_counts > 0) * 100 #(percentage form)
    result = np.zeros_like(heatmap, dtype = np.float64)
    for x in range(height):
        for y in range(width):
            if valid_pixel_counts[x, y] > 0:
                result[x, y] = (heatmap[x, y] / valid_pixel_counts[x, y]) * 100
    
    return result


def create_test_video(num_frames = 1000, width = 928, height = 576, damaged_pixel_count = 1000, grid_size = 20):
    frames = []
    
    grid_rows = int(np.sqrt(damaged_pixel_count * height / width))
    grid_cols = int(damaged_pixel_count / grid_rows)

    grid_coordinates = [(y * (height // grid_rows), x * (width // grid_cols)) for y in range(grid_rows) for x in range(grid_cols)]
    print(len(grid_coordinates))
    
    def generate_damaged_pixel_pattern(offset = 0):
        damaged_pixels = []
        
        for y, x in grid_coordinates:
            damaged_pixels.append((y + offset, x + offset))
        return damaged_pixels
    
    for i in range(num_frames):
        base_brightness = 50 + 10 * np.sin(2 * np.pi * i / 50)
        noise = np.random.normal(loc=0, scale=1, size=(height, width))  # Add Gaussian noise
        frame = np.clip(base_brightness + noise, 0, 255).astype(np.uint8)
        
        #frame[: (height //  2) + 9] = 150 
        
        damaged_pixels = generate_damaged_pixel_pattern(offset=0) if i % 2 == 0 else generate_damaged_pixel_pattern(offset=1)
        
        for y, x in damaged_pixels:
            if 0 <= y < height and 0 <= x < width:
                frame[y, x] = 250
                
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
    plt.imshow(heatmap, cmap = 'viridis', interpolation ='nearest')
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


def create_isotropic_test_video(num_frames=1000, width=928, height=576, damaged_pixel_count=1000):
    frames = []
    for i in range(num_frames):
        frame = np.full((height, width), 0, dtype=np.uint8)  # Specify dtype as uint8
        damaged_pixels = np.random.choice(height * width, damaged_pixel_count, replace=False)
        damaged_coords = np.unravel_index(damaged_pixels, (height, width))
        
        frame[damaged_coords] = 255
        
        frames.append(frame)
        
    return frames


# def create_test_video_2(num_frames = 1000, width = 928, height = 576, damaged_pixel_count = 1000, grid_size = 20):
#     frames = []
    
#     #coordinates = []
    
#     #generate 10 randomly distributed damaged pixels in the light and dark regions
    
#     def generate_damaged_pixel_coords(offset):
#         coordinates = []
#         offset = 0
        
#         for i in range(damaged_pixel_count):
#             # x1 = np.random.choice(np.arange(0, width))
#             # y1 = np.random.choice(np.arange(0, height // 2))

#             x2 = np.random.choice(np.arange(0, width))
#             y2 = np.random.choice(np.arange(height // 2, height))

#             #coordinates.append((y1, x1))
#             coordinates.append((y2, x2))
#         #coordinates.append((289 + offset, 100))
            
#         return coordinates


#     offset = 0
#     for i in range(num_frames):

#         base_brightness = 50
#         #frame = np.clip(base_brightness, 0, 255).astype(np.uint8)
#         frame = np.full((height, width), base_brightness, dtype = np.uint8)
        
#         frame[: (height //  2)] = 180 
#         coords = generate_damaged_pixel_coords(offset)
#         for y, x in coords:
#             if 0 <= y < height and 0 <= x < width:
#                 frame[y, x] = 250
                
#         frames.append(frame)
#         offset += 3
        
#     return frames, damaged_pixel_count


def create_test_video_2(num_frames = 1000, width = 928, height = 576, damaged_pixel_count = 1000, grid_size = 20):
    frames = []
    
    #coordinates = []
    
    #generate 10 randomly distributed damaged pixels in the light and dark regions
    
    def generate_damaged_pixel_coords(offset):
        coordinates = []
        #offset = 0
        
        #for i in range(damaged_pixel_count):
            # x1 = np.random.choice(np.arange(0, width))
            # y1 = np.random.choice(np.arange(0, height // 2))

            #x2 = np.random.choice(np.arange(0, width))
            #y2 = np.random.choice(np.arange(height // 2, height))

            #coordinates.append((y1, x1))
            #coordinates.append((y2, x2))
        #coordinates.append((289 + offset, 100))
#         coordinates.append((303 + offset, 15 + offset))
#         coordinates.append((338 + offset, 50 + offset))
#         coordinates.append((373 + offset, 85 + offset))
#         coordinates.append((408 + offset, 120 + offset))
#         coordinates.append((443 + offset, 155 + offset))
#         coordinates.append((478 + offset, 190 + offset))
#         coordinates.append((513 + offset, 225 + offset))
#         coordinates.append((548 + offset, 260 + offset))
        
#         coordinates.append((388 + offset, 200 + offset))
#         coordinates.append((338 + offset, 100 + offset))

        coordinates.append((289, 15 + offset))
        coordinates.append((289, 50 + offset))
        coordinates.append((289, 85 + offset))
        coordinates.append((289, 120 + offset))
        coordinates.append((289, 155 + offset))
        coordinates.append((289, 190 + offset))
        coordinates.append((289, 225 + offset))
        coordinates.append((289, 260 + offset))
        
        coordinates.append((300, 200 + offset))
        coordinates.append((300, 100 + offset))
            
        return coordinates


    offset = 0
    for i in range(num_frames):

        base_brightness = 50
        #frame = np.clip(base_brightness, 0, 255).astype(np.uint8)
        frame = np.full((height, width), base_brightness, dtype = np.uint8)
        
        frame[: (height //  2)] = 180 
        coords = generate_damaged_pixel_coords(offset)
        for y, x in coords:
            if 0 <= y < height and 0 <= x < width:
                frame[y, x] = 250
                
        frames.append(frame)
        offset += 1
        
    return frames, damaged_pixel_count