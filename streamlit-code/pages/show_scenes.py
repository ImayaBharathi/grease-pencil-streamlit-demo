import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import traceback

# @st.cache
def app(filename):
    def make_grid(cols,rows):
        grid = [0]*cols
        for i in range(cols):
            with st.container():
                grid[i] = st.columns(rows)
        return grid
    print(filename)
    filename = st.text_input("Script File Name")
    st.caption("Tip: Enter the same script name that you given on Upload Page")
    extract = st.button("Get Scenes")
    if extract:
        try:
            st.write(f"Our AI is extracting Scenes form {filename} script file")
            url = f"http://20.25.21.167:8000/get_scenes_from_uploaded_file?file_name={filename}"
            response = requests.post(url)
            response = response.json()        
            if response:
                if response.get("status_code", "") == "201":
                    response_data = json.loads(response.get("data", None))
                    scenes_list = response_data.get("scene_list", None)
                    if scenes_list:
                        n_rows = 1 + len(scenes_list) // int(3)
                        rows = [st.container() for _ in range(n_rows)]
                        cols_per_row = [r.columns(3) for r in rows]
                        cols = [column for row in cols_per_row for column in row]
                        for scene_index, scene in enumerate(scenes_list):
                            cols[scene_index].subheader(str(scene[0]))
                            cols[scene_index].text(f"{str(scene[1])}")
                    else:
                        print(f"Scene list is empty, filename= {filename} and response= {str(response)}")
                        st.write("Server Side Issue : No Details Available")
                else:
                    print(f"Scene Extraction Status code is not 201, filename= {filename} and response= {str(response)}")
                    msg = response.get("msg", None)
                    st.write(f"Server Side Issue : {msg} - explanation")
            else:
                print(str(response))
                st.write(f"Server Side Issue : response json is empty - explanation")
        except Exception as e:
            print(traceback.format_exc())
            st.write("Client Side Issue : ", str(e))