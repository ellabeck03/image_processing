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


# def detect_damaged_pixels(video_filename, num_frames=20, threshold_value=50, max_frame_duration=10, min_area=1, max_area=10):
#     # Download video
# #     urlretrieve(video_url, video_filename)

# #     if not os.path.exists(video_filename):
# #         print("File download failed.")
# #         return

# #     cap = cv2.VideoCapture(video_filename)
# #     if not cap.isOpened():
# #         print("Error: Could not open video.")
# #         return

#     cap = cv2.VideoCapture(video_filename)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     hot_pixel_count_history = []
#     grey_frame_history = []

#     # Create a background subtractor for motion tracking
#     subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

#     # Process frames
#     for _ in range(num_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to grayscale
#         grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         grey_frame_history.append(grey_image)

#         # Use background subtractor to detect changes from the background
#         foreground_mask = subtractor.apply(grey_image)

#         # Refine mask using morphological operations to reduce noise
#         kernel = np.ones((3, 3), np.uint8)  # Slightly larger kernel to remove noise
#         foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

#         # Connected components to detect pixel groups
#         num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask)

#         # Filter components by area (removing regions that are too large or too small)
#         for i in range(1, num_labels):
#             area = stats[i, cv2.CC_STAT_AREA]
#             if area < min_area or area > max_area:
#                 foreground_mask[labels_im == i] = 0

#         # Append the refined mask to track across frames
#         hot_pixel_count_history.append(foreground_mask)

#     # Initialize a pixel occurrence tracker
#     pixel_occurrences = np.zeros_like(grey_frame_history[0], dtype=np.uint8)

#     # Track occurrences of pixels detected across frames
#     for mask in hot_pixel_count_history:
#         pixel_occurrences[mask > 0] += 1

#     # Remove pixels that persist for too many frames (likely noise)
#     duration_mask = (pixel_occurrences <= max_frame_duration).astype(np.uint8) * 255

#     # Combine hot pixel masks across frames
#     combined_hot_pixel_mask = np.zeros_like(hot_pixel_count_history[0])
#     for mask in hot_pixel_count_history:
#         combined_hot_pixel_mask = cv2.bitwise_or(combined_hot_pixel_mask, mask)

#     # Final damaged pixel mask (excluding persistent pixels)
#     damaged_pixels_mask = cv2.bitwise_and(combined_hot_pixel_mask, duration_mask)

#     # Count the number of damaged pixels
#     damaged_pixel_count = cv2.countNonZero(damaged_pixels_mask)

#     # Visualize the damaged pixels on the first frame
#     output_image = grey_frame_history[0].copy()
#     output_image_colored = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
#     output_image_colored[damaged_pixels_mask > 0] = [0, 0, 255]  # Highlight damaged pixels in red

#     # # Display the results
#     # plt.figure(figsize=(20, 10))
#     # plt.subplot(1, 2, 1)
#     # plt.title('Original Frame')
#     # plt.imshow(grey_frame_history[0], cmap='gray')
#     # plt.subplot(1, 2, 2)
#     # plt.title('Frame with Detected Damaged Pixels')
#     # plt.imshow(cv2.cvtColor(output_image_colored, cv2.COLOR_BGR2RGB))
#     # plt.show()

#     print(f"Number of damaged pixels detected: {damaged_pixel_count}")
    
#     return damaged_pixel_count, output_image_colored

# def process_all_frames(video_url, video_filename, max_frames=50, threshold_value=50, max_frame_duration=10, min_area=1, max_area=10):
#     # Download video if necessary
#     urlretrieve(video_url, video_filename)

#     if not os.path.exists(video_filename):
#         print("File download failed.")
#         return

#     # Open video
#     cap = cv2.VideoCapture(video_filename)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     frame_damaged_pixel_counts = []
#     frames_with_damaged_pixels = []
    
#     frame_num = 0
#     while frame_num < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Save frame to temporary file and process
#         frame_filename = f"temp_frame_{frame_num}.jpg"
#         cv2.imwrite(frame_filename, frame)

#         damaged_pixel_count, output_image_colored = detect_damaged_pixels(
#             video_filename=frame_filename,
#             num_frames=1,  # Process only the current frame
#             threshold_value=threshold_value,
#             max_frame_duration=max_frame_duration,
#             min_area=min_area,
#             max_area=max_area
#         )
        
#         frame_damaged_pixel_counts.append(damaged_pixel_count)
#         frames_with_damaged_pixels.append(output_image_colored)

#         os.remove(frame_filename)  # Clean up temporary file
#         frame_num += 1

#     cap.release()

#     # Plot results
#     fig, axs = plt.subplots(2, max_frames//2, figsize=(20, 10))
#     for i, ax in enumerate(axs.flat):
#         if i < len(frames_with_damaged_pixels):
#             ax.imshow(cv2.cvtColor(frames_with_damaged_pixels[i], cv2.COLOR_BGR2RGB))
#             ax.set_title(f"Frame {i+1}: Damaged Pixels {frame_damaged_pixel_counts[i]}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

#     # Plot damaged pixel count over time
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(len(frame_damaged_pixel_counts)), frame_damaged_pixel_counts, marker='o')
#     plt.title("Damaged Pixels Over Time")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Damaged Pixel Count")
#     plt.show()
    
    
# def detect_damaged_pixels_over_time(video_url, video_filename, threshold_value=50, max_frame_duration=10, min_area=1, max_area=10):
#     # Download and open video
#     urlretrieve(video_url, video_filename)
#     cap = cv2.VideoCapture(video_filename)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     damaged_pixel_counts = []
#     frame_count = 0

#     # Process each frame in the video
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to grayscale
#         grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply background subtractor to detect damaged pixels
#         damaged_pixel_count = detect_damaged_pixels(
#             video_url, video_filename, num_frames=1, threshold_value=threshold_value,
#             max_frame_duration=max_frame_duration, min_area=min_area, max_area=max_area
#         )

#         # Append count to list
#         damaged_pixel_counts.append(damaged_pixel_count)
#         frame_count += 1

#     # Plot the damaged pixels over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(frame_count), damaged_pixel_counts, color="blue")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Number of Damaged Pixels")
#     plt.title("Damaged Pixels Over Time")
#     plt.show()


# import cv2
# import numpy as np
# from urllib.request import urlretrieve
# import matplotlib.pyplot as plt
# import os

# def detect_damaged_pixels(selected_frame, neighboring_frames, threshold_value=50, min_area=1, max_area=10):
#     # Calculate the average brightness of each pixel from neighboring frames
#     avg_brightness = np.mean(neighboring_frames, axis=0).astype(np.uint8)

#     # Create a mask for damaged pixels where the selected frame has significantly lighter pixels
#     damaged_pixels_mask = cv2.absdiff(selected_frame, avg_brightness) > threshold_value

#     # Filter connected components by area
#     num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(damaged_pixels_mask.astype(np.uint8), connectivity=8)
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area < min_area or area > max_area:
#             damaged_pixels_mask[labels_im == i] = 0

#     # Count the number of damaged pixels
#     damaged_pixel_count = cv2.countNonZero(damaged_pixels_mask.astype(np.uint8))

#     # Return both the count and the mask
#     return damaged_pixel_count, damaged_pixels_mask

# def detect_damaged_pixels_over_time(video_url, video_filename, num_neighbors=5, threshold_value=50, min_area=1, max_area=10):
#     # Download and open video
#     urlretrieve(video_url, video_filename)
#     cap = cv2.VideoCapture(video_filename)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     damaged_pixel_counts = []
#     frame_buffer = []
#     frames_with_damaged_pixels = []  # To store frames with damaged pixels highlighted

#     # Read initial frames to populate the buffer
#     for _ in range(num_neighbors):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

#     # Process each frame in the video
#     frame_count = 0
#     while True:
#         # Read the next frame
#         ret, frame = cap.read()
#         if not ret:
#             break
#         grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Add the new frame to the buffer
#         frame_buffer.append(grey_image)

#         # If we have enough frames, detect damaged pixels for the current frame
#         if len(frame_buffer) >= 2 * num_neighbors + 1:
#             # Select the target frame (middle frame in buffer)
#             selected_frame = frame_buffer[num_neighbors]

#             # Neighboring frames are all other frames in the buffer except the middle one
#             neighboring_frames = [f for i, f in enumerate(frame_buffer) if i != num_neighbors]

#             # Calculate damaged pixels for the selected frame
#             damaged_pixel_count, damaged_pixels_mask = detect_damaged_pixels(selected_frame, neighboring_frames, threshold_value, min_area, max_area)
#             damaged_pixel_counts.append(damaged_pixel_count)

#             # Store the frame with damaged pixels highlighted
#             damaged_highlighted_frame = selected_frame.copy()
#             damaged_highlighted_frame[damaged_pixels_mask > 0] = 255  # Highlight damaged pixels in white
#             frames_with_damaged_pixels.append(damaged_highlighted_frame)

#             # Keep only a few frames for visualization (e.g., 5 frames)
#             if len(frames_with_damaged_pixels) > 5:
#                 frames_with_damaged_pixels.pop(0)

#             # Remove the first frame from the buffer to slide the window
#             frame_buffer.pop(0)

#         frame_count += 1

#     cap.release()

#     # Plot a few frames with damaged pixels highlighted
#     plt.figure(figsize=(15, 8))
#     for i, frame in enumerate(frames_with_damaged_pixels):
#         plt.subplot(1, len(frames_with_damaged_pixels), i + 1)
#         plt.imshow(frame, cmap='gray')
#         plt.title(f"Frame {frame_count - len(frames_with_damaged_pixels) + i}")
#         plt.axis("off")
#     plt.suptitle("Damaged Pixels Highlighted in Sample Frames")
#     plt.show()

#     # Plot the damaged pixels over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(len(damaged_pixel_counts)), damaged_pixel_counts, color="blue")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Number of Damaged Pixels")
#     plt.title("Damaged Pixels Over Time")
#     plt.show()

def detect_damaged_pixels(video_url, video_filename):
    # Download the video file if it doesn't exist
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

    # Read all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    # Define the number of adjacent frames to average
    window_size = 10
    min_consecutive_frames = 2
    threshold_multiplier = 1.5  # Adjust this to define "significantly brighter"

    for i in range(len(frames)):
        current_frame = frames[i]
        height, width = current_frame.shape
        damaged_pixel_count = 0
        damaged_pixels = np.zeros((height, width), dtype=np.uint8)

        # Initialize the pixel history
        if i not in pixel_history:
            pixel_history[i] = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Get the brightness of the current pixel
                current_brightness = current_frame[y, x]

                # Collect adjacent frame brightness
                adjacent_brightness = []

                for j in range(-window_size//2, window_size//2 + 1):
                    adjacent_index = i + j
                    if 0 <= adjacent_index < len(frames) and j != 0:
                        adjacent_brightness.append(frames[adjacent_index][y, x])

                if len(adjacent_brightness) > 0:
                    avg_brightness = np.mean(adjacent_brightness)

                    # Check for damaged pixels with significant brightness
                    if current_brightness > (avg_brightness * threshold_multiplier):
                        damaged_pixel_count += 1
                        damaged_pixels[y, x] = 255  # Highlight the damaged pixel
                        
                        # Track the history of damaged pixels
                        pixel_history[i][y, x] += 1

                # Reset the count if the pixel was previously damaged for too long
                if pixel_history[i][y, x] > min_consecutive_frames:
                    damaged_pixels[y, x] = 0

        damaged_pixel_counts.append(damaged_pixel_count)

        # For visualization using Matplotlib
        plt.figure(figsize=(10, 5))
        
        # Original Frame
        plt.subplot(1, 2, 1)
        plt.imshow(current_frame, cmap='gray')
        plt.title(f'Original Frame {i}')
        plt.axis('off')
        
        # Damaged Pixels Highlighted
        highlighted_frame = cv2.addWeighted(current_frame, 0.7, damaged_pixels, 0.3, 0)
        plt.subplot(1, 2, 2)
        plt.imshow(highlighted_frame, cmap='gray')
        plt.title(f'Damaged Pixels Highlighted {i}')
        plt.axis('off')
        
        plt.show()

    # Plotting the number of damaged pixels per frame
    plt.figure()
    plt.plot(damaged_pixel_counts, label='Damaged Pixels Count')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Damaged Pixels')
    plt.title('Damaged Pixels Detected Over Time')
    plt.legend()
    plt.show()

# Example usage
video_url = 'https://www.dropbox.com/scl/fi/gr4bxigggxjq7ix0ip7co/11_01_H_170726081325_460000.avi?rlkey=5p2y97bixslqnr23bwedp3a21&st=8t2scn3s&dl=1'
video_filename = '11_01_H_170726081325_460000.avi'
detect_damaged_pixels(video_url, video_filename)