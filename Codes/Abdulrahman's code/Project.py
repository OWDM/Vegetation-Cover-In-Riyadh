import streamlit as st
# import geemap.foliumap as geemap
from st_pages import hide_pages
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="U-Net",  page_icon="C:\\Users\\abdul\\Downloads\\Gemini_Generated_Image_9lbfo29lbfo29lbf-removebg-preview.png")
model = load_model('C:\\Users\\abdul\\Desktop\\interface\\unet_segmentation_3 (1).keras')
hide_pages(["geemap","main","auth","Project"])
# Customize the sidebar
markdown = """
Project Members :\n
1- Abdulrahman Alghamdi\n
2- Naif Alayaid\n
3- Mussaed Albaidhani\n
4- Rowaid Sindi\n
5- Azzam Alharbi
"""
# st.sidebar.title("Vegetation Segmentation")
# logo = "https://i.imgur.com/UbOXYAU.png"
logo = "C:\\Users\\abdul\\Downloads\\Gemini_Generated_Image_9lbfo29lbfo29lbf-removebg-preview.png"
st.sidebar.image(logo)
st.sidebar.info(markdown)
st.title("Vegetation Segmentation")

st.markdown("---")

c1, c2, c3 = st.columns([4, 2, 4])

left  = "C:\\Users\\abdul\\Downloads\\RGB_RUH-All-Bands_2023-12-19_sub_576.png"
right = "C:\\Users\\abdul\\Downloads\\true_MASK_RUH-All-Bands_2023-12-19_sub_99.png"





with c1:
    placeholderL = st.empty()
    placeholderL.image(left)

with c2:
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.image("C:\\Users\\abdul\\Downloads\\360_F_787319331_KsNlGp8UVUHWMzVIGQSpAtX1oqOFppSZ-removebg-preview.png",width=130)

with c3:
    placeholderR = st.empty()
    placeholderR.image(right)


img_file_buffer = st.file_uploader('', type=['png','jpg','jpeg'])


if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    placeholderL.image(image)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    # placeholderL.image(image)
    placeholderR.markdown("![Alt Text](https://media.tenor.com/On7kvXhzml4AAAAi/loading-gif.gif)")
    predicted_mask = model.predict(img_array)
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    st.markdown(predicted_mask.shape)
    predicted_mask = predicted_mask[0]  # Remove batch dimension
    st.markdown(predicted_mask.shape)

    # If the result is a single channel (e.g., grayscale), remove the last dimension if needed
    if predicted_mask.shape[-1] == 1:
        st.markdown(predicted_mask.shape)
        predicted_mask = predicted_mask.squeeze(-1)
        st.markdown(predicted_mask.shape)
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    # predicted_mask = np.expand_dims(predicted_mask, axis=3)
    st.markdown(predicted_mask.shape)
    placeholderR.image(predicted_mask)

    st.pyplot(predicted_mask)


