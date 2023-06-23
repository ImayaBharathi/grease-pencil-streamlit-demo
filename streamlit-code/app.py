import os
import numpy as np
import streamlit as st
from PIL import Image

# Custom imports 
from multipage import MultiPage
from pages import upload_data, show_scenes, show_storyboard, show_frames

app = MultiPage()

data = ""
display = Image.open('Logo.jpeg')
display = np.array(display)
col1, col2 = st.columns(2)
col1.image(display, width = 400)

app.add_page("Upload Data", upload_data.app)
app.add_page("Extracted Scenes", show_scenes.app)
app.add_page("Frames for selected scenes", show_frames.app)
app.add_page("Storyboard Images for selectec Frames", show_storyboard.app)
app.run()
