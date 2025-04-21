#%% Libraries

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
from ultralytics import YOLO

import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("yolov5")

#%% Functions

def process_image(image_path, x1, y1, x2, y2, output_path, colorspace = 'RGB', num_defects_threshold=5, area_ratio_threshold=0.05):
    
    # Load the original image
    Rail_original = cv2.imread(image_path)

    # Extract the specified rectangle from the original image
    Rail_cropped = Rail_original[y1:y2, x1:x2]
    
    # Convert to grayscale
    if colorspace == 'RGB':
        Rail_hsv = cv2.cvtColor(Rail_cropped, cv2.COLOR_RGB2CSV)
        
        
    elif colorspace == 'BW':
        
        # Ignoring other color channels
        Rail_cropped = Rail_cropped[:,:,0]
        
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
            
            # Drawing contours
            # Contour_image = Rail_original.copy()
            # cv2.drawContours(Contour_image, [cnt + [x1, 0] for cnt in cnts], -1, (0, 255, 0), 1)
            
            # plt.imshow(Contour_image, cmap='gray')
            # plt.title('Contours')
            # plt.axis('off')
            # plt.show()

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

            # Draw contours for visualization
            Contour_image = Rail_original.copy()
            cv2.drawContours(Contour_image, [cnt + [x1+left_x, 0] for cnt in cnts], -1, (0, 255, 0), 1)
                            
            plt.imshow(Contour_image, cmap='gray')
            plt.title('Defects')
            plt.axis('off')
            plt.show()

            # Metric to determine if there are a large number of defects
            num_defects = len(cnts)
            total_defect_area = sum(cv2.contourArea(cnt) for cnt in cnts)
            total_area = (right_x - left_x) * (y2 - y1)

            # Calculate defect area ratio
            defect_area_ratio = total_defect_area / total_area            
            Check_YOLO = (num_defects > num_defects_threshold) and (defect_area_ratio > area_ratio_threshold/100)
            
        else:
            
            # Sending to YOLO immediately
            num_defects = 100
            defect_area_ratio = 1
            Check_YOLO = True

    return {
        'image_path': image_path,
        'num_defects': num_defects,
        'defect_area_ratio': defect_area_ratio*100,
        'requires_YOLO': Check_YOLO
    }


# def process_images_in_parallel(image_paths, x1, y1, x2, y2, num_defects_threshold=5, area_ratio_threshold=0.05):
#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_image, path, x1, y1, x2, y2, num_defects_threshold, area_ratio_threshold) for path in image_paths]
#         for future in concurrent.futures.as_completed(futures):
#             results.append(future.result())
#     return results


# Define the list of image paths
image_paths = ['image_1254.jpg', 'image_1279.jpg', 'image_202.jpg', 'image_240.jpg', 'image_270.jpg']

# Define the coordinates of the rectangle (top-left and bottom-right corners)
x1, y1 = 610, 0
x2, y2 = 700, 500

Images_for_YOLO = []

for image in image_paths:
    
    # Measure the time taken to process images in parallel
    start_time = time.time()
    Result = process_image('images/' + image, x1, y1, x2, y2, 'BW', 5, 10)
    end_time = time.time()
    
    print(f"Time taken to process images: {end_time - start_time:.2f} seconds")
    
    print(f"Image: {Result['image_path']}")
    print(f"Number of defects: {Result['num_defects']}")
    print(f"Defect area ratio: {Result['defect_area_ratio']:.2f}")
    print(f"Reuires YOLO: {Result['requires_YOLO']}")
    print("-------")
    
    if Result['requires_YOLO']:
        Images_for_YOLO.append(Result['image_path'])


#%%

Confidence = 0.4
model = YOLO('runs/detect/train5/weights/best.pt')
Results = []

for image_path in Images_for_YOLO:

    # Analysis with YOLO5    
    Result = model(image_path, conf=Confidence)
    Results.append(Result)

    # Visualization of the defects
    boxes = Result[0].boxes
    conf = []
    for box in boxes:
        conf.append(float(box.conf))
    
    if len(conf)>0:
        if max(conf) > Confidence:
    
            Annotated_img = Result[0].plot()
            
            plt.imshow(Annotated_img, cmap='gray')
            plt.title(image_path[image_path.index('/')+1:])
            plt.axis('off')
            plt.show()
            
            output_path = image_path.replace(image_path.split('/')[-1], '') + 'YOLO_' + image_path.split('/')[-1]
            cv2.imwrite(output_path, Annotated_img)
            
        else:
            print(f"{image[image.index('/')+1:]} does not have defects")

#detect.run(weights='runs/detect/train5/weights/best.pt', source='images/' + image_paths[2], conf_thres=0.1)
