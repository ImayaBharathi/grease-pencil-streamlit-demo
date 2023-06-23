import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import json
import traceback

filename = ""
# @st.cache
def app():
    global filename
    st.markdown("## Script Upload [PDF]")
    st.write("\n")
    uploaded_file = st.file_uploader("Choose a file", type = ['pdf'])
    filename = st.text_input("Script Name (should be unique,lowercase,without space)")
    filename = filename.lower()
    on_submit = st.button("Load Script")
    if on_submit:
        try:
            file_type = "pdf"
            url = f"http://20.25.21.167:8000/script_file_upload?file_type={file_type}&file_name={filename}"
            response = requests.post(url, files={"pdfFile": uploaded_file.getvalue()})
            response = response.json()        
            if response:
                if response.get("status_code", "") == "201":
                    response_data = json.loads(response.get("data", None))
                    if response_data:
                        filename = response_data.get("filename","").split(".")[0]
                        st.subheader('File is successfully uploaded to the server')
                        st.subheader('Next Page (Click On): Extracted Scenes')
                        return filename
                    
                    else:
                        st.write("Try Again, File name is causing issues.")
                        st.write("Tip: File name must be in lowercase and without space")
                else:
                    st.write("Server Side Issue : ",response.get("msg", "None"))
            else:
                st.write("Server Side Issue : Not able to connect to backend")
        except Exception as e:
            print(traceback.format_exc())
            st.write("Client Side Issue : ", str(e))




