#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import math
import datetime

def euc_dist(arr1, point):
    square_sub = (arr1 - point) ** 2
    return np.sqrt(np.sum(square_sub, axis=1))

def preprocess_image(img_path, variables):
    # Read the input image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image:", img_path)
        exit(1)
    # Save original dimensions (unused later since we resize)
    img_ht, img_wd = img.shape[0], img.shape[1]
    # Resize image to desired dimensions
    img = cv2.resize(img, (variables.resize_wd, variables.resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl1 = clahe.apply(img_gray)
    # Gaussian adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    # Save processed images and dimensions in the variables object
    variables.img_ht = variables.resize_ht
    variables.img_wd = variables.resize_wd
    variables.img_gray = img_gray
    variables.img_thresh = img_thresh
    variables.img = img
    return variables

def get_extreme_coordinates(mask):
    indices = np.where(mask == 255)
    if indices[0].size == 0 or indices[1].size == 0:
        return (0, 0), (mask.shape[1], mask.shape[0])
    x = indices[1]
    y = indices[0]
    topleft = (np.min(x), np.min(y))
    bottomright = (np.max(x), np.max(y))
    return topleft, bottomright

def preprocess_hand_image(hand_path, variables):
    """
    This function reads the hand image (assumed to be a PNG with transparency)
    and extracts the alpha channel to create a binary mask.
    """
    # Read hand image with unchanged flag to capture alpha channel if available
    hand_img = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    if hand_img is None:
        print("Failed to load hand image:", hand_path)
        exit(1)
    # Check if the image has an alpha channel (4 channels)
    if hand_img.shape[2] == 4:
        # Separate BGR and alpha channel
        bgr = hand_img[:, :, :3]
        alpha = hand_img[:, :, 3]
        # Threshold alpha to obtain a binary mask
        _, hand_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    else:
        bgr = hand_img
        # Fall back: create a mask from grayscale conversion (less ideal)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, hand_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Crop the hand image and mask to the hand’s region using extreme coordinates
    top_left, bottom_right = get_extreme_coordinates(hand_mask)
    hand_cropped = bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask_cropped = hand_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hand_mask_inv = 255 - hand_mask_cropped
    # Normalize masks to range 0-1 for blending later
    hand_mask_norm = hand_mask_cropped / 255.0
    hand_mask_inv_norm = hand_mask_inv / 255.0
    variables.hand_ht, variables.hand_wd = hand_cropped.shape[0], hand_cropped.shape[1]
    variables.hand = hand_cropped
    variables.hand_mask = hand_mask_norm
    variables.hand_mask_inv = hand_mask_inv_norm
    return variables

def draw_hand_on_img(drawing, hand, drawing_coord_x, drawing_coord_y, hand_mask_inv, hand_ht, hand_wd, img_ht, img_wd):
    remaining_ht = img_ht - drawing_coord_y
    remaining_wd = img_wd - drawing_coord_x
    crop_hand_ht = hand_ht if remaining_ht > hand_ht else remaining_ht
    crop_hand_wd = hand_wd if remaining_wd > hand_wd else remaining_wd
    hand_cropped = hand[:crop_hand_ht, :crop_hand_wd]
    hand_mask_inv_cropped = hand_mask_inv[:crop_hand_ht, :crop_hand_wd]
    
    # Blend the hand image onto the drawing at the given location
    region = drawing[drawing_coord_y:drawing_coord_y+crop_hand_ht, drawing_coord_x:drawing_coord_x+crop_hand_wd]
    for c in range(3):
        region[:, :, c] = region[:, :, c] * hand_mask_inv_cropped + hand_cropped[:, :, c]
    drawing[drawing_coord_y:drawing_coord_y+crop_hand_ht, drawing_coord_x:drawing_coord_x+crop_hand_wd] = region
    return drawing

def draw_masked_object(variables, object_mask=None, skip_rate=5, black_pixel_threshold=10):
    print("Skip Rate:", skip_rate)
    img_thresh_copy = variables.img_thresh.copy()
    if object_mask is not None:
        object_mask_black_ind = np.where(object_mask == 0)
        object_ind = np.where(object_mask == 255)
        img_thresh_copy[object_mask_black_ind] = 255

    selected_ind = 0
    n_cuts_vertical = int(math.ceil(variables.resize_ht / variables.split_len))
    n_cuts_horizontal = int(math.ceil(variables.resize_wd / variables.split_len))
    
    # Split the thresholded image into a grid
    grid_of_cuts = np.array(np.split(img_thresh_copy, n_cuts_horizontal, axis=-1))
    grid_of_cuts = np.array(np.split(grid_of_cuts, n_cuts_vertical, axis=-2))
    print("Grid shape:", grid_of_cuts.shape)
    
    cut_having_black = (grid_of_cuts < black_pixel_threshold) * 1
    cut_having_black = np.sum(np.sum(cut_having_black, axis=-1), axis=-1)
    cut_black_indices = np.array(np.where(cut_having_black > 0)).T

    counter = 0
    while len(cut_black_indices) > 1:
        selected_ind_val = cut_black_indices[selected_ind].copy()
        range_v_start = selected_ind_val[0] * variables.split_len
        range_h_start = selected_ind_val[1] * variables.split_len

        # Create a small drawing patch from the grid piece
        temp_drawing = np.zeros((variables.split_len, variables.split_len, 3), dtype=np.uint8)
        temp_drawing[:, :, 0] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 1] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        temp_drawing[:, :, 2] = grid_of_cuts[selected_ind_val[0]][selected_ind_val[1]]
        
        variables.drawn_frame[range_v_start:range_v_start+variables.split_len,
                               range_h_start:range_h_start+variables.split_len] = temp_drawing
        
        # Position the hand image roughly at the center of the current grid cell
        hand_coord_x = range_h_start + variables.split_len // 2
        hand_coord_y = range_v_start + variables.split_len // 2
        drawn_frame_with_hand = draw_hand_on_img(
            variables.drawn_frame.copy(),
            variables.hand.copy(),
            hand_coord_x,
            hand_coord_y,
            variables.hand_mask_inv.copy(),
            variables.hand_ht,
            variables.hand_wd,
            variables.resize_wd,
            variables.resize_ht,
        )
        
        # Remove the selected grid piece from the list
        cut_black_indices[selected_ind] = cut_black_indices[-1]
        cut_black_indices = cut_black_indices[:-1]
        
        if len(cut_black_indices) == 0:
            break
        
        euc_arr = euc_dist(cut_black_indices, selected_ind_val)
        selected_ind = np.argmin(euc_arr)
        
        counter += 1
        if counter % skip_rate == 0:
            variables.video_object.write(drawn_frame_with_hand)
        
        if counter % 40 == 0:
            print("Remaining grid pieces:", len(cut_black_indices))
    
    # Once done, fill any remaining area with the original image
    variables.drawn_frame[:, :, :] = variables.img
    return

def draw_whiteboard_animation(img_path, hand_path, save_video_path, variables):
    variables = preprocess_image(img_path, variables)
    variables = preprocess_hand_image(hand_path, variables)
    
    start_time = time.time()
    variables.video_object = cv2.VideoWriter(
        save_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        variables.frame_rate,
        (variables.resize_wd, variables.resize_ht)
    )
    
    # Create a white canvas for drawing
    variables.drawn_frame = np.full(variables.img.shape, 255, np.uint8)
    
    # Animate drawing the image in segments
    draw_masked_object(variables, object_mask=None, skip_rate=variables.object_skip_rate)
    
    # Hold the completed image for a few seconds at the end
    for i in range(variables.frame_rate * variables.end_gray_img_duration_in_sec):
        variables.video_object.write(variables.img)
    
    end_time = time.time()
    print("Total animation time:", end_time - start_time, "seconds")
    variables.video_object.release()

class AllVariables:
    def __init__(self, frame_rate, resize_wd, resize_ht, split_len, object_skip_rate, end_gray_img_duration_in_sec):
        self.frame_rate = frame_rate
        self.resize_wd = resize_wd
        self.resize_ht = resize_ht
        self.split_len = split_len
        self.object_skip_rate = object_skip_rate
        self.end_gray_img_duration_in_sec = end_gray_img_duration_in_sec

if __name__ == "__main__":
    # Define file paths for Termux (adjust if necessary)
    img_path = "/storage/emulated/0/Download/images.jpeg"
    hand_path = "/storage/emulated/0/Download/drawing-hand.png"
    
    # Define output directory and create it if it doesn't exist
    output_dir = "/storage/emulated/0/OUTPUT"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    current_date = str(datetime.datetime.now().date())
    img_name = os.path.basename(img_path).split(".")[0]
    video_save_name = f"{img_name}-{current_date}.mp4"
    save_video_path = os.path.join(output_dir, video_save_name)
    
    print("Saving video to:", save_video_path)
    
    # Initialize the variables for animation settings
    variables = AllVariables(
        frame_rate=25,
        resize_wd=1020,
        resize_ht=1020,
        split_len=10,
        object_skip_rate=8,
        end_gray_img_duration_in_sec=3
    )
    
    # Run the whiteboard animation
    draw_whiteboard_animation(img_path, hand_path, save_video_path, variables)
    
    print("Animation complete!")
