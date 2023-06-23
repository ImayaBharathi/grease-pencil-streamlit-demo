import streamlit as st
import numpy as np
import pandas as pd
import requests, json, traceback

# @st.cache
def app():
    st.header("Steps")
    st.subheader("1.Copy the Scene from Extracted Scenes page")
    st.subheader("2.Paste Scene")

    text_data = st.text_area("Paste your scene here")
    on_submit = st.button("Submit")
    if on_submit:
        try:
            st.caption("Our AI is working")
            url = f"http://20.25.21.167:8000/get_storyboard_prompts_from_scene"
            headers = {'Content-Type': 'text/plain'}
            payload = str(text_data)
            response = requests.post(url, headers=headers, data=payload)
            response = response.json()        
            if response:
                if response.get("status_code", "") == "201":
                    response_data = json.loads(response.get("data", None))
                    if response_data:
                        st.subheader("The following are the Frame Description")
                        for frame in response_data.get("storyboard_frames",None):
                            st.markdown(f"- {frame}")
                    else:
                        st.write("Server Side Issue: Format of a scene should be Scripting Format")
                else:
                    st.write("Server Side Issue : ",response.get("msg", "None"))
            else:
                st.write("Server Side Issue : Not able to connect to backend")
        except Exception as e:
            print(traceback.format_exc())
            st.write("Client Side Issue : ", str(e))


