import pickle
import json
import re

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from config import *

def extract_images_from_wikipedia_screenshot(image_path, output_folder="extracted_images", debug_results_folder = "output_folder", img_name = "abc"):
    """
    Extract images from Wikipedia screenshots with special handling for tables.
    
    Args:
        image_path: Path to the Wikipedia screenshot image
        output_folder: Folder to save extracted images
        debug_results_folder: Folder to save debug_img for visualization purposes
        img_name: Image name for debug_img
    
    Returns:
        List of paths to extracted images
    """
    # Create output directory and debug_results_folder if they don't exist
    os.makedirs(output_folder, exist_ok = True)
    os.makedirs(debug_results_folder, exist_ok = True)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy for visualization
    debug_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours with hierarchy information
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    min_area = 5000  # Minimum area for general extraction
    min_inner_area = 2000  # Minimum area for images inside tables
    extracted_images = []
    
    # First identify potential tables or large rectangular structures
    tables = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if it looks like a table using multiple methods
        is_table = False
        
        # Method 1: Check if it's rectangular and large
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and area > 50000:
            is_table = True
        
            # Method 2: Check for grid pattern of horizontal and vertical lines
            # Extract the region
            roi = gray[y:y+h, x:x+w]
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal = cv2.erode(roi, horizontal_kernel)
            horizontal = cv2.dilate(horizontal, horizontal_kernel)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical = cv2.erode(roi, vertical_kernel)
            vertical = cv2.dilate(vertical, vertical_kernel)
            
            # Count pixels in horizontal and vertical lines
            h_pixels = np.count_nonzero(horizontal)
            v_pixels = np.count_nonzero(vertical)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Calculate coverage percentage
            line_coverage = (h_pixels + v_pixels) / (2 * total_pixels)  # Divide by 2 as we counted twice
            
            # Check if lines cover a significant portion of the area and have sufficient grid structure
            if line_coverage > 0.15:  # Threshold for line coverage
                # Combine horizontal and vertical lines to detect intersections
                grid = cv2.add(horizontal, vertical)
                
                # Count the number of line intersections as a measure of grid structure
                intersections = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                intersections = cv2.erode(intersections, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                
                # Count intersection points
                num_intersections = cv2.countNonZero(intersections)
                
                # If we have enough intersections, it's likely a table
                if num_intersections > 10:  # Adjust threshold as needed
                    is_table = True
        
        if is_table:
            # Draw red rectangles around potential tables
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            tables.append((x, y, w, h))
    
    # Now process each table to find images inside
    # First, get all potential image cells
    potential_cells = []
    for table_idx, (tx, ty, tw, th) in enumerate(tables):
        # Extract the table region
        table_roi = img[ty:ty+th, tx:tx+tw]
        table_gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find elements within the table
        _, table_thresh = cv2.threshold(table_gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours within the table
        table_contours, _ = cv2.findContours(table_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process potential image cells within the table
        for j, table_contour in enumerate(table_contours):
            cell_area = cv2.contourArea(table_contour)
            
            # Filter by size - not too small, not the whole table
            if cell_area < min_inner_area or cell_area > tw * th * 0.9:
                continue
            
            # Get cell bounding rectangle
            cx, cy, cw, ch = cv2.boundingRect(table_contour)
            
            # Skip if aspect ratio is extreme
            aspect_ratio = cw / max(ch, 1)  # Avoid division by zero
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Extract the potential image cell
            cell_roi = table_roi[cy:cy+ch, cx:cx+cw]
            
            # Apply image detection logic
            is_image = is_likely_image(cell_roi)

            if is_image:
                potential_cells.append((tx + cx, ty + cy, cw, ch, cell_roi))
                # Draw yellow rectangles around potential cells
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Now filter out inner rectangles
    filtered_cells = []
    for i, (cx1, cy1, cw1, ch1, roi1) in enumerate(potential_cells):
        is_inner = False
        for j, (cx2, cy2, cw2, ch2, _) in enumerate(potential_cells):
            if i == j:
                continue
            
            # Check if rectangle 1 is inside rectangle 2
            if (cx1 > cx2 and cy1 > cy2 and 
                cx1 + cw1 < cx2 + cw2 and 
                cy1 + ch1 < cy2 + ch2):
                # Rectangle 1 is inside rectangle 2
                is_inner = True
                break
            
            # Check for significant overlap (e.g., more than 85%)
            overlap_x = max(0, min(cx1 + cw1, cx2 + cw2) - max(cx1, cx2))
            overlap_y = max(0, min(cy1 + ch1, cy2 + ch2) - max(cy1, cy2))
            overlap_area = overlap_x * overlap_y
            area1 = cw1 * ch1
            
            if overlap_area > 0.85 * area1 and area1 < cw2 * ch2:
                # This rectangle has significant overlap with a larger one
                is_inner = True
                break
        
        if not is_inner:
            filtered_cells.append((cx1, cy1, cw1, ch1, roi1))
    
    # Process the filtered cells (outer rectangles only)
    for j, (cx, cy, cw, ch, cell_roi) in enumerate(filtered_cells):
        # Draw green rectangles around identified images within tables
        cv2.rectangle(debug_img, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
        
        # Save the extracted image
        output_path = os.path.join(output_folder, f"table_image_{j}.jpg")
        cv2.imwrite(output_path, cell_roi)
        extracted_images.append(output_path)
    
    # Find contours with hierarchy information - Avoiding RETR_TREE
    # RETR_EXTERNAL would tell OpenCV to only retrieve the outermost 
    # contours and ignore any nested contours inside them. 
    # This would help prevent the duplicate detection you're experiencing.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Also look for standalone images not in tables
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area or area > img.shape[0] * img.shape[1] * 0.8:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)  # Avoid division by zero
        
        # Skip extreme aspect ratios
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue
        
        # Skip if this region is within any of the detected tables
        in_table = False
        for tx, ty, tw, th in tables:
            if x >= tx and y >= ty and x + w <= tx + tw and y + h <= ty + th:
                in_table = True
                break
        
        if in_table:
            continue
        
        # Extract the potential standalone image
        roi = img[y:y+h, x:x+w]
        
        # Check if it's likely an image
        if is_likely_image(roi):
            # Draw blue rectangles around standalone images
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the extracted image
            output_path = os.path.join(output_folder, f"standalone_image_{i}.jpg")
            cv2.imwrite(output_path, roi)
            extracted_images.append(output_path)
    
    # Save the debug image
    cv2.imwrite(os.path.join(debug_results_folder, f"{img_name}.jpg"), debug_img)
    
    return extracted_images

def is_likely_image(roi, text_threshold=10, color_variation_threshold=30):
    """
    Determine if a region is likely an image based on its content characteristics.
    
    Args:
        roi: Region of interest (image patch)
        text_threshold: Threshold for text detection
        color_variation_threshold: Threshold for color variation
        
    Returns:
        bool: True if the region likely contains an image
    """
    # Skip very small regions
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Calculate standard deviation as a measure of pixel variation
    std_dev = np.std(gray)
    
    # Low standard deviation usually means uniform background (text or empty space)
    if std_dev < text_threshold:
        return False
    
    # Check color variation for color images
    if len(roi.shape) == 3:  # Color image
        # Calculate standard deviation for each color channel
        b, g, r = cv2.split(roi)
        color_std = [np.std(b), np.std(g), np.std(r)]
        color_variation = max(color_std)
        
        # High variation in at least one color channel suggests an image
        if color_variation > color_variation_threshold:
            return True
    
    return False


if __name__ == "__main__":
    
    images_folder = PASSAGE_IMAGE_FOLDER

    save_extracted_images_folder = PASSAGE_SCRAPED_FOLDER
    debug_results_folder = DEBUG_FOLDER

    os.makedirs(save_extracted_images_folder, exist_ok = True)
    os.makedirs(debug_results_folder, exist_ok = True)


    image_names = os.listdir(PASSAGE_IMAGE_FOLDER)

    # This will work
    cleaned_image_names = []
    for line in image_names:
        cleaned_line = ""
        for char in line:
            if 0xE000 <= ord(char) <= 0xF8FF:  # Private Use Area range
                cleaned_line += "?"  # or any replacement
            else:
                cleaned_line += char
        cleaned_image_names.append(cleaned_line)

    
    errors = []
            
    for count, img_name in enumerate(cleaned_image_names):

        output_folder = os.path.join(save_extracted_images_folder, img_name)
        
        img_path = os.path.join(images_folder, img_name + ".png")

        try:
            basic_images = extract_images_from_wikipedia_screenshot(img_path, output_folder, debug_results_folder, img_name)
            print("*-"*20)
            print(count, img_name)
            print(f"Found {len(basic_images)} images using basic method.")
        
        except Exception as err:
            print(err)
            errors.append(err)


print("\nExtracted images have been saved to the output folders.")
