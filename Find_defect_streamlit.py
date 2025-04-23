#%% Libraries

import cv2

import numpy as np
import matplotlib.pyplot as plt

#%% Functions

def process_image(image_path, x1, y1, x2, y2, colorspace = 'BW', num_defects_threshold=5, area_ratio_threshold=0.05):
    
    # Load the original image
    Rail_original = image_path

    # Extract the specified rectangle from the original image
    Rail_cropped = Rail_original[y1:y2, x1:x2]
    
    # Contrasting
    Rail_contrasted = cv2.normalize(Rail_cropped, None, 0, 255, cv2.NORM_MINMAX)
    
    # Sharpening
    Rail_sharpened = cv2.addWeighted(Rail_contrasted, 1.5, cv2.GaussianBlur(Rail_contrasted, (0, 0), sigmaX=1), -0.5, 0)
    
    # Binarization
    _,Rail_binary = cv2.threshold(Rail_sharpened,170,255,cv2.THRESH_BINARY)
    
    # Morphological transformations
    kernel = np.ones((2,2),np.uint8)
    Rail_binary = cv2.morphologyEx(Rail_binary, cv2.MORPH_OPEN, kernel)
    
    
    # Determination of gaps on the rail surface
    
    # Searching for labels
    num_labels, labels = cv2.connectedComponents(Rail_binary)
    height, width = Rail_binary.shape
    Full_rail = []
    
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)

        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        
        # Check, if the rail has no gaps (from top to bottom)
        top_touch = np.any(ys == 0)
        bottom_touch = np.any(ys == height - 1)
        
        Full_rail.append(top_touch and bottom_touch)
        
        if top_touch and bottom_touch:
            break
        
        
    if True in Full_rail:
        
        # Continuation of processing
        kernel = np.ones((100,10),np.uint8)
        Rail_binary = cv2.morphologyEx(Rail_binary, cv2.MORPH_CLOSE, kernel)
        
        # Determination of the rail's edges:
        contours, _ = cv2.findContours(Rail_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(contours)

        # Restricting the rail's surface region
        x_coords = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            x_coords.append(x)
            x_coords.append(x + w - 3)

        x_coords = sorted(set(x_coords))
        left_x, right_x = x_coords[0], x_coords[-1]
        
        # Choosing the Region of Interest
        Rail_ROI = Rail_sharpened[:,left_x:right_x]
        _,Rail_binary = cv2.threshold(Rail_ROI,170,255,cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(Rail_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(contours)

        # Metric to determine if there are a large number of defects
        num_defects = len(cnts)
        total_defect_area = sum(cv2.contourArea(cnt) for cnt in cnts)
        total_area = (right_x - left_x) * (y2 - y1)
        
        if total_area == 0:
            Check_YOLO = False

        else:
            # Calculate defect area ratio
            defect_area_ratio = total_defect_area / total_area            
            Check_YOLO = (num_defects > num_defects_threshold) and (defect_area_ratio > area_ratio_threshold/100)
        
    else:
        
        # Sending to YOLO immediately
        num_defects = 100
        defect_area_ratio = 1
        Check_YOLO = True
    
    return Check_YOLO