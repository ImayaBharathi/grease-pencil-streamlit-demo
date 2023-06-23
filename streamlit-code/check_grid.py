import streamlit as st


scenes_list = [1]*30
n_rows = 1 + len(scenes_list) // int(3)
rows = [st.container() for _ in range(n_rows)]
cols_per_row = [r.columns(3) for r in rows]
cols = [column for row in cols_per_row for column in row]

for image_index, scene in enumerate(scenes_list):
    cols[image_index].subheader(str(scene))
    cols[image_index].text(f"Something - {str(scene)}")

