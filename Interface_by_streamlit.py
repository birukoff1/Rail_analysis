#%% Libraries
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
from ultralytics import YOLO

import streamlit as st
import tempfile
from Find_defect import process_image

#%% Main
st.title("🎥 Video Analyzer")

uploaded_file = st.file_uploader("Upload your video file", type=["sgb"])

if uploaded_file is not None:
    
    # Save uploaded file to a temp location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Define the coordinates of the rectangle
    x1, y1 = 610, 0
    x2, y2 = 700, 500
    
    output_path = 'cache'

    st.info("Processing video... please wait.")
    Frame_number = 0
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = frame[:,:,0]
        Frame_number+=1
        
        if ret == True:
     
            # Sending the image to preprocessing:
            if process_image(frame, x1, y1, x2, y2, output_path, 'BW', 5, 10):
                
                # Saving the image with indexed seconds
                cv2.imwrite(output_path + f'/rail_{Frame_number}_frame_{Frame_number/fps}_s.jpg', frame)

        # Break the loop
        else: 
            break
     
    cap.release()
    cv2.destroyAllWindows()    

    st.info("Preprocessing is complete!")
    
    
    #%% Analysis of the images by YOLO
    Images_for_YOLO = os.listdir(output_path)

    Confidence = 0.6
    model = YOLO('weights/best.pt')
    Results = []

    for image_path in Images_for_YOLO:
        
        image_path = output_path + '/' + image_path
        
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
                
                cv2.imwrite(image_path.replace('/'+image_path.split('/')[-1], '') + '_YOLO/' + image_path.split('/')[-1], Annotated_img)
                st.image(Annotated_img)
                
            else:
                st.info(f"{image_path[image_path.index('/')+1:]} does not have defects")
    
    st.success("Analysis complete!")