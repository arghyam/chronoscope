import csv
import os

import pandas as pd
import streamlit as st
from PIL import Image

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception:
        return None

def load_csv(csv_path):
    try:
        # Use dtype=str to preserve exact string representation
        return pd.read_csv(csv_path, dtype={'digit_sequence': str})
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def save_csv(df, csv_path):
    try:
        # Ensure digit_sequence is saved exactly as is
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return True
    except Exception as e:
        st.error(f"Error saving CSV: {e}")
        return False

def main():
    st.title("BFM Meter Reading Validator")

    # Initialize session state for tracking current image index
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Input for folder path with helper text
    st.write("Please enter the base folder path that contains 'images' and 'csv_files' folders")
    folder_path = st.text_input("Enter the path to base folder:")

    if folder_path and os.path.exists(folder_path):
        # Look for CSV file in the new structure
        csv_file = os.path.join(folder_path, "csv_files", "digit_sequences.csv")
        if not os.path.exists(csv_file):
            st.error("digit_sequences.csv not found in csv_files folder!")
            return

        # Load CSV data with string dtype
        df = load_csv(csv_file)
        if df is None:
            return

        # Display progress
        total_images = len(df)
        st.write(f"Progress: {st.session_state.current_index + 1}/{total_images} images")
        st.progress(st.session_state.current_index / total_images)

        # Get current image
        current_image = df.iloc[st.session_state.current_index]

        # Add a column for missing images if it doesn't exist
        if 'is_missing' not in df.columns:
            df['is_missing'] = False

        # Display current image with updated path
        image_path = os.path.join(folder_path, "images", current_image['image_name'])
        image = load_image(image_path)

        if image is None:
            # Handle missing image
            st.error(f"Image not found: {current_image['image_name']}")
            df.at[st.session_state.current_index, 'is_missing'] = True

            st.warning("This image is missing. You can:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Skip to Next Image"):
                    if st.session_state.current_index < total_images - 1:
                        st.session_state.current_index += 1
                        save_csv(df, csv_file)  # Save the missing image status
                        st.rerun()
                    else:
                        st.success("All images have been processed!")

            with col2:
                if st.button("Mark as Missing and Continue"):
                    if st.session_state.current_index < total_images - 1:
                        st.session_state.current_index += 1
                        save_csv(df, csv_file)
                        st.rerun()
                    else:
                        st.success("All images have been processed!")

            # Show missing image statistics
            total_missing = df['is_missing'].sum()
            st.write(f"Total missing images so far: {total_missing}")

        else:
            # Normal flow for existing images
            st.image(image, caption=current_image['image_name'])

            # Display the exact digit sequence value
            st.write(f"Current digit sequence: {current_image['digit_sequence']}")

            # Input for digit sequence with current value, ensuring exact string preservation
            new_sequence = st.text_input(
                "Edit digit sequence:",
                value=str(current_image['digit_sequence']),
                key=f"digit_input_{st.session_state.current_index}"
            )

            # Radio button for decimal marking
            current_decimal = 'red' if 'has_decimal' in df.columns and df.iloc[st.session_state.current_index]['has_decimal'] == 'red' else 'black'
            has_decimal = st.radio(
                "Does the image contain decimal?",
                ('Red (Has decimal)', 'Black (No decimal)'),
                index=0 if current_decimal == 'red' else 1,
                key=f"decimal_input_{st.session_state.current_index}"
            )

            # Save and Next button
            if st.button("Save and Next"):
                # Update DataFrame with exact string value
                df.at[st.session_state.current_index, 'digit_sequence'] = str(new_sequence)

                # Add or update decimal column
                if 'has_decimal' not in df.columns:
                    df['has_decimal'] = 'black'
                df.at[st.session_state.current_index, 'has_decimal'] = 'red' if has_decimal == 'Red (Has decimal)' else 'black'
                df.at[st.session_state.current_index, 'is_missing'] = False  # Mark as not missing

                # Save changes to CSV
                if save_csv(df, csv_file):
                    # Move to next image if available
                    if st.session_state.current_index < total_images - 1:
                        st.session_state.current_index += 1
                        st.rerun()
                    else:
                        st.success("All images have been processed!")

            # Previous button
            if st.button("Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()

            # Add a sidebar with statistics
            st.sidebar.markdown("### Statistics")
            if 'is_missing' in df.columns:
                total_missing = df['is_missing'].sum()
                total_processed = st.session_state.current_index + 1
                st.sidebar.write(f"Total images processed: {total_processed}/{total_images}")
                st.sidebar.write(f"Missing images: {total_missing}")
                st.sidebar.write(f"Success rate: {((total_processed - total_missing)/total_processed)*100:.2f}%")

            # Display keyboard shortcuts help
            st.sidebar.markdown("""
            ### Keyboard Shortcuts:
            - Enter: Save and move to next image
            - Tab: Move between input fields
            """)

    else:
        if folder_path:
            st.error("Invalid folder path!")

if __name__ == "__main__":
    main()
