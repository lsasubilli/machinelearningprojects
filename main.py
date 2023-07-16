import streamlit as st
from annotated_text import annotated_text
import cv2 as cv
from PIL import Image
import numpy as np
frozen_model = 'frozen_inference_graph.pb'
st.markdown('Image Detection')
my_list = [
    "Made",
    [
        " by: ",  
    ],
    ("Lalith", "Sasubilli"),
    ".",
]
annotated_text(my_list)
st.markdown('### This is an image detecting software that detects daily common objects used by humans. This model was trained on Coco dataset released in 2020.')
labels_file = '/Users/a2022/Downloads/labels.txt' 
classlabels = []
config_file ='/Users/a2022/Downloads/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model ='/Users/a2022/Downloads/frozen_inference_graph.pb'
with open(labels_file, 'rt') as ftp:
    classlabels = ftp.read().strip('\n').split('\n')
model = cv.dnn.DetectionModel(frozen_model, config_file)
model.setInputSize(300, 300)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)
uploaded_file = st.file_uploader("Upload your file here...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert PIL Image to NumPy array
    img = cv.cvtColor(image, cv.COLOR_RGB2BGR)  
    ClassIndex, Confidence, bbox = model.detect(img, confThreshold=0.36)
    try:
        x= ClassIndex[0]
        st.markdown(classlabels[x-1])
    except TypeError:
        print("Error: cannot add an int and a str")
    for classind, conf, box in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
        cv.rectangle(image, box,(255,0,0),5 )
        cv.putText(image, classlabels[classind-1], (box[0]+10, box[1]+40), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0),5)
    pil_image = Image.fromarray(image)
    st.image(pil_image, caption='OpenCV Output', use_column_width=True)
