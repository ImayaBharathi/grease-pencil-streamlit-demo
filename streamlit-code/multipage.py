"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

# Import necessary libraries 
import streamlit as st

# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
        self.file_obj = []
        self.selected_scene = 0
        self.selected_frame_description = 0
        self.file_name_from_backend = ""
        self.scenes_list_from_backend = []
        self.frames_list_from_backend = []
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project

        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            
            func: Python function to render this page in Streamlit
        """

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        # Drodown to select the page to run  
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page.get('title', "")
        )
        print(page)
        if page.get("title", "") == "Upload Data":
            ###run the app function 
            self.file_name_from_backend = page.get('function')()
        elif page.get("title", "") == "Extracted Scenes":
            print("---------------", self.file_name_from_backend)
            page.get('function')(self.file_name_from_backend)
        else: page.get('function')()