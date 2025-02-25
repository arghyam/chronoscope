import streamlit as st
import os
import pandas as pd
import cv2
import numpy as np
import io
import glob
from PIL import Image

def load_or_create_annotation_file(folder_path):
    # Try to load existing CSV file from the folder
    csv_path = os.path.join(folder_path, 'annotations.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Create a new DataFrame if no existing file
    return pd.DataFrame(columns=['image_name', 'annotation', 'annotation_done'])

def main():
    st.title("Image Quality Annotation")

    # Replace file_uploader with text input for folder path
    folder_path = st.sidebar.text_input("Enter the full path to your images folder:")
    
    # Session state initialization
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'annotation_df' not in st.session_state:
        st.session_state.annotation_df = None
    if 'image_files' not in st.session_state:
        st.session_state.image_files = None
    if 'selected_annotation' not in st.session_state:
        st.session_state.selected_annotation = None

    if folder_path and os.path.exists(folder_path):
        # Load or create annotation file
        if st.session_state.annotation_df is None:
            st.session_state.annotation_df = load_or_create_annotation_file(folder_path)

        # Get all image files from the folder
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if image_files:
            # Update session state with image files and set current index to first unannotated image
            if st.session_state.image_files != image_files:
                st.session_state.image_files = image_files
                # Find the first unannotated image
                annotated_images = set(st.session_state.annotation_df['image_name'].values)
                for i, file_path in enumerate(image_files):
                    if os.path.basename(file_path) not in annotated_images:
                        st.session_state.current_image_index = i
                        break

            current_file_path = st.session_state.image_files[st.session_state.current_image_index]
            
            # Display image using cv2 without any orientation changes
            image = cv2.imread(current_file_path)
            # Convert BGR to RGB for proper display in streamlit
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image)
            st.write(f"Image {st.session_state.current_image_index + 1} of {len(st.session_state.image_files)}")

            # Create two rows of three columns for the annotation options
            annotation_options = ['Good', 'Blurry', 'Out of focus', 'Oriented', 'Foggy', 'Poor lighting']
            
            # First row of options
            for i, col in enumerate(st.columns(3)):
                if col.button(annotation_options[i], 
                            key=f"btn_{i}",
                            type="primary" if st.session_state.selected_annotation == annotation_options[i] else "secondary"):
                    st.session_state.selected_annotation = annotation_options[i]
                    
            # Second row of options
            for i, col in enumerate(st.columns(3)):
                if col.button(annotation_options[i+3], 
                            key=f"btn_{i+3}",
                            type="primary" if st.session_state.selected_annotation == annotation_options[i+3] else "secondary"):
                    st.session_state.selected_annotation = annotation_options[i+3]

            # Add some vertical space
            st.write("")
            st.write("")
            st.write("")
            
            # Modify the save and next button section
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("Save and Next", key="save_next", type="secondary"):
                        if st.session_state.selected_annotation is None:
                            st.error("Please select an annotation first!")
                        else:
                            # Save annotation
                            current_file_name = os.path.basename(current_file_path)
                            new_annotation = {
                                'image_name': current_file_name,
                                'annotation': st.session_state.selected_annotation,
                                'annotation_done': 'yes'
                            }
                            
                            # Update DataFrame
                            if current_file_name in st.session_state.annotation_df['image_name'].values:
                                st.session_state.annotation_df.loc[
                                    st.session_state.annotation_df['image_name'] == current_file_name
                                ] = new_annotation
                            else:
                                st.session_state.annotation_df = pd.concat([
                                    st.session_state.annotation_df,
                                    pd.DataFrame([new_annotation])
                                ], ignore_index=True)

                            # Automatically save CSV to the image folder
                            csv_path = os.path.join(folder_path, 'annotations.csv')
                            st.session_state.annotation_df.to_csv(csv_path, index=False)
                            
                            # Move to next image
                            if st.session_state.current_image_index < len(st.session_state.image_files) - 1:
                                st.session_state.current_image_index += 1
                                st.session_state.selected_annotation = None  # Reset selection for next image
                                st.rerun()
                            else:
                                st.success("All images have been annotated!")
                                
        else:
            st.error("No image files found in the specified folder!")
    elif folder_path:
        st.error("The specified folder path does not exist!")

if __name__ == "__main__":
    main()
