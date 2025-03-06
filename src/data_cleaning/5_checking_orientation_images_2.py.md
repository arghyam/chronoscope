# Image Orientation Checker Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Module Overview](#2-module-overview)
    * [2.1 `get_image_files` Function](#21-get_image_files-function)
    * [2.2 `check_exif_orientation` Function](#22-check_exif_orientation-function)
    * [2.3 `detect_orientation_using_features` Function](#23-detect_orientation_using_features-function)
    * [2.4 `is_image_rotated` Function](#24-is_image_rotated-function)
    * [2.5 `main` Function](#25-main-function)
* [3. Usage](#3-usage)


## 1. Introduction

This document details the functionality of the Python script designed to detect image rotation and generate a CSV report.  The script analyzes images using two primary methods: EXIF data extraction and feature-based detection.  The results are then compiled into a CSV file for easy review.


## 2. Module Overview

This script uses several libraries including `os`, `csv`, `cv2` (OpenCV), `numpy`, and `PIL` (Pillow).

### 2.1 `get_image_files` Function

This function retrieves a list of all image files within a specified directory and its subdirectories.

| Parameter | Type | Description |
|---|---|---|
| `directory` | `str` | The path to the directory containing images. |
| **Return Value** | `list` | A list of strings, where each string is a full path to an image file. |

**Algorithm:**

The function uses `os.walk` to traverse the directory tree. For each file found, it checks if the file's extension (converted to lowercase) is present in the `image_extensions` list. If it is, the full file path is added to the `image_files` list, which is then returned.


### 2.2 `check_exif_orientation` Function

This function attempts to extract orientation information from an image's EXIF data.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The path to the image file. |
| **Return Value** | `bool` or `None` |  `True` if the image has a non-standard orientation according to EXIF data, `False` if the image has a standard orientation (Orientation tag value 1), and `None` if no orientation tag is found or an error occurs during EXIF data reading. |


**Algorithm:**

The function opens the image using PIL. It then accesses the EXIF data using `_getexif()`. It iterates through the EXIF tags, searching for the "Orientation" tag. If found, it checks the tag's value. A value other than 1 indicates a rotated image. If the "Orientation" tag is not found, or if an error occurs, `None` is returned.


### 2.3 `detect_orientation_using_features` Function

This function detects image orientation using feature detection, specifically by analyzing the prevalence of horizontal versus vertical lines.  This is a heuristic approach.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The path to the image file. |
| **Return Value** | `bool` or `None` | `True` if the image is likely rotated (more vertical lines), `False` if it's likely not rotated (more horizontal lines), and `None` if the detection is uncertain or an error occurs. |

**Algorithm:**

1. **Image Loading and Preprocessing:** The image is loaded using OpenCV (`cv2.imread`). It is then converted to grayscale using `cv2.cvtColor`.

2. **Edge Detection:** The Canny edge detector (`cv2.Canny`) is applied to identify edges in the image.

3. **Line Detection:** The Hough Line Transform (`cv2.HoughLinesP`) is used to detect lines within the edge image.

4. **Line Classification:** Each detected line's angle is calculated. Lines with angles close to 0 or 180 degrees are classified as horizontal, while lines with angles close to 90 degrees are classified as vertical. A tolerance is applied to account for slight deviations from perfectly horizontal or vertical lines.

5. **Orientation Determination:** The counts of horizontal and vertical lines are compared. If the number of horizontal lines significantly exceeds the number of vertical lines (by a factor of 1.5), the image is considered properly oriented (`False`). Conversely, if vertical lines significantly outweigh horizontal lines, the image is considered likely rotated (`True`). If neither condition is met, the function returns `None`, indicating uncertainty.



### 2.4 `is_image_rotated` Function

This function determines if an image is rotated by using both EXIF data and feature-based detection.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The path to the image file. |
| **Return Value** | `bool` | `True` if the image is rotated, `False` otherwise. |

**Algorithm:**

The function first calls `check_exif_orientation`. If EXIF data provides a definitive answer, that result is returned. Otherwise, it calls `detect_orientation_using_features`. If neither method provides a conclusive result, it defaults to assuming the image is not rotated (`False`).


### 2.5 `main` Function

This function handles command-line argument parsing, image processing, and CSV report generation.

| Parameter | Type | Description |
|---|---|---|
| `--input_dir` | `str` |  (Required) Path to the directory containing images. |
| `--output_csv` | `str` | (Required) Path to save the output CSV file. |

**Algorithm:**

1. **Argument Parsing:** The function uses `argparse` to parse command-line arguments for the input directory and output CSV file path.

2. **Image File Retrieval:** It calls `get_image_files` to obtain a list of image files.

3. **Orientation Checking:** It iterates through the image files, calling `is_image_rotated` for each image to determine its orientation.  A progress indicator is printed every 10 images.

4. **CSV Report Generation:** The results (image name and orientation status) are stored in a list of dictionaries. This data is then written to a CSV file using the `csv` module.


## 3. Usage

To use the script, run it from the command line, providing the input directory and desired output CSV file path as arguments:

```bash
python image_orientation_checker.py --input_dir /path/to/images --output_csv output.csv
```

Replace `/path/to/images` with the actual path to your image directory, and `output.csv` with your desired output file name.
