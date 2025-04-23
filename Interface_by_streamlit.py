#%% Libraries
import platform
import streamlit as st
st.write(f"Python version: {platform.python_version()}")

import os

import cv2
from ultralytics import YOLO

import tempfile
from Find_defect_streamlit import process_image

#%% Main
st.title("Анализ рельсового покрытия")

uploaded_file = st.file_uploader("Загрузите Ваше видео", type=["sgb"])

side = st.selectbox(
    "Где расположен рельс на видео?",
    options=["Слева", "Справа"]
)

if uploaded_file is not None and side is not None:
    
    with tempfile.TemporaryDirectory() as temp_images_for_YOLO:
        
        # Save uploaded file to a temp location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    
        # Define the coordinates of the rectangle
        if side == "Слева":
            x1, y1 = 270, 0
            x2, y2 = 380, 500
        else:
            x1, y1 = 610, 0
            x2, y2 = 700, 500
    
        st.info("Шаг 1: Предобработка видео...")
        Frame_number = 0
        while(cap.isOpened()):
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            Frame_number+=1
            
            if ret and Frame_number < 500:
                
                frame = frame[:,:,0]
         
                # Sending the image to preprocessing:
                if process_image(frame, x1, y1, x2, y2, 'BW', 5, 10):
                    
                    # Saving the image with indexed seconds
                    cv2.imwrite(temp_images_for_YOLO + f'/rail_{Frame_number}_frame_{Frame_number/fps}_s.jpg', frame)
    
            # Break the loop
            else: 
                break
        
        st.info("Шаг 1 завершен!")
        
        cap.release()
        cv2.destroyAllWindows()    
        
        
        #%% Analysis of the images by YOLO
        
        st.info("Шаг 2: Классификация дефектов...")
        
        Images_for_YOLO = os.listdir(temp_images_for_YOLO)
    
        Confidence = 0.7
        model = YOLO('weights/best.pt')
        Results = []
    
        for image_path in Images_for_YOLO:
            
            image_path = temp_images_for_YOLO + '/' + image_path
            
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
                    
                    st.image(Annotated_img, caption=f"Кадр: №{image_path.split('_')[-4]}, Метка времени: {image_path.split('_')[-2]}", use_container_width=True)
                    
                # else:
                #     st.info(f"{image_path[image_path.index('/')+1:]} не имеет дефектов")
        
        st.info("Шаг 2 завершен!")
        st.success("Анализ видео успешно завершен!")
        
else:
    st.info("Требования к видео:")
    st.info("1) Формат .sgb")
    st.info("2) В оттенках серого")
