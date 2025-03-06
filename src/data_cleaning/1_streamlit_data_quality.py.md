# Image Quality Annotation Tool: Internal Documentation

[Linked Table of Contents](#linked-table-of-contents)

## <a name="linked-table-of-contents"></a>Linked Table of Contents

* [1. Introduction](#1-introduction)
* [2. Modules and Libraries](#2-modules-and-libraries)
* [3. Function: `load_or_create_annotation_file`](#3-function-load_or_create_annotation_file)
* [4. Function: `main`](#4-function-main)
    * [4.1 Session State Management](#4.1-session-state-management)
    * [4.2 Folder Path Handling and File Loading](#4.2-folder-path-handling-and-file-loading)
    * [4.3 Image Display and Annotation Interface](#4.3-image-display-and-annotation-interface)
    * [4.4 Annotation Saving and Navigation](#4.4-annotation-saving-and-navigation)
* [5. Algorithm Details](#5-algorithm-details)


## 1. Introduction

This document provides internal code documentation for the Image Quality Annotation tool, a Streamlit application designed to annotate images with quality labels.  The tool allows users to specify a folder containing images, then iteratively annotate each image with predefined quality categories. Annotations are saved to a CSV file within the image folder.

## 2. Modules and Libraries

The application utilizes several Python libraries:

| Library      | Purpose                                      |
|--------------|----------------------------------------------|
| `streamlit`  | Web application framework                     |
| `os`         | Operating system functionalities              |
| `pandas`     | Data manipulation and analysis                 |
| `cv2`        | OpenCV for image processing                   |
| `numpy`      | Numerical computation                          |
| `io`         | Input/output operations                       |
| `glob`       | Filename pattern matching                      |
| `PIL` (Pillow)|Image manipulation library                    |


## 3. Function: `load_or_create_annotation_file`

This function handles the loading and creation of the annotation CSV file (`annotations.csv`).

```python
def load_or_create_annotation_file(folder_path):
    # Try to load existing CSV file from the folder
    csv_path = os.path.join(folder_path, 'annotations.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Create a new DataFrame if no existing file
    return pd.DataFrame(columns=['image_name', 'annotation', 'annotation_done'])
```

The function first checks if `annotations.csv` exists in the specified `folder_path`. If it does, it loads the CSV into a pandas DataFrame and returns it. Otherwise, it creates a new, empty DataFrame with columns for 'image_name', 'annotation', and 'annotation_done' and returns that.


## 4. Function: `main`

The `main` function orchestrates the application's logic.

### 4.1 Session State Management

The application uses Streamlit's session state to maintain persistent variables across user interactions. This includes:

* `current_image_index`: The index of the currently displayed image.
* `annotation_df`: The pandas DataFrame storing image annotations.
* `image_files`: A list of paths to all image files in the specified folder.
* `selected_annotation`: The annotation currently selected by the user.

These variables are initialized if they don't exist in the session state.


### 4.2 Folder Path Handling and File Loading

The user provides the image folder path via a text input. The application checks if the path is valid and exists. If valid, it calls `load_or_create_annotation_file` to manage the annotation data.  It then identifies all image files in the folder using `glob`, considering common image extensions.


### 4.3 Image Display and Annotation Interface

If image files are found, the application displays the current image using `cv2`.  `cv2.cvtColor` converts the image from BGR (OpenCV's default) to RGB for correct display in Streamlit.  The application then presents annotation options in two rows of buttons.  The selected annotation is highlighted.

### 4.4 Annotation Saving and Navigation

The "Save and Next" button triggers the saving of the current annotation to the `annotation_df`.  If the image is already in the DataFrame, its entry is updated; otherwise, a new row is added. The updated DataFrame is saved to `annotations.csv`. The application then advances to the next unannotated image or displays a success message if all images are annotated.  Error handling is included for cases where no annotation is selected.



## 5. Algorithm Details

The core algorithm involves iterating through images in a specified folder. For each image:

1. **Annotation Loading/Creation:** The application loads existing annotations from `annotations.csv` or creates a new file if one doesn't exist.
2. **Image Display:** The selected image is loaded and displayed using OpenCV and Streamlit.
3. **User Interaction:** The user selects an annotation from a set of predefined options using buttons.
4. **Annotation Saving:** The selected annotation, along with the image name and a 'yes' flag indicating annotation completion are saved in the `annotation_df`.
5. **Data Persistence:** The updated `annotation_df` is written to `annotations.csv`.
6. **Navigation:** The application proceeds to the next unannotated image, automatically handling the case of all images being processed.


The algorithm efficiently manages annotation data using pandas DataFrames and Streamlit's session state, ensuring data persistence and a seamless user experience.  Error handling is implemented at each critical step to prevent unexpected application crashes.
