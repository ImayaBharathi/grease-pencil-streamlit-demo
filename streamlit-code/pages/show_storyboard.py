import streamlit as st
import numpy as np
import pandas as pd
import traceback
import requests
from PIL import Image
import io

# @st.cache
def app():

    frame_input = st.text_input("Paste your storyboard frame description")
    st.caption("Tip: Enter the same script name that you given on Upload Page")
    on_sumit = st.button("Get Storyboard")
    if on_sumit:
        try:
            st.caption("Our AI is working")
            url = f"http://20.25.21.167:8000/get_images_for_storyboard_description?storyboard_description={frame_input}"
            response = requests.post(url)
            image_data = io.BytesIO(response.raw.read())
            img = Image.open(image_data)

            print(type(image_data))
            if response:
                st.container.image(img)
            else:
                st.write(f"Server Side Issue : response json is empty - explanation")
        except Exception as e:
            print(traceback.format_exc())
            st.write("Client Side Issue : ", str(e))

    