# Image Orientation Checker Documentation

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. `get_image_files` Function](#2-getimagefiles-function)
* [3. `check_exif_orientation` Function](#3-checkexiforientation-function)
* [4. `detect_orientation_using_ocr` Function](#4-detectorientationusingocr-function)
*     * [4.1 Algorithm Details](#41-algorithm-details)
* [5. `is_image_rotated` Function](#5-isimagerotated-function)
* [6. `main` Function](#6-main-function)


## 1. Introduction

This document details the functionality of the Python script designed to detect image rotation and generate a CSV report. The script leverages EXIF data and OCR techniques to determine image orientation.


## 2. `get_image_files` Function

This function retrieves a list of all image files within a specified directory and its subdirectories.

| Parameter | Type | Description |
|---|---|---|
| `directory` | `str` | The path to the directory containing images. |

**Return Value:** A list of strings, where each string is the full path to an image file.  The function supports the following image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.


## 3. `check_exif_orientation` Function

This function checks for orientation information within the EXIF metadata of an image file.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The full path to the image file. |

**Return Value:**
* `True`: if the image's EXIF data indicates rotation.
* `False`: if the image's EXIF data indicates no rotation (orientation tag value is 1).
* `None`: if no EXIF data or orientation tag is found.

The function uses the `PIL` library to access EXIF data.  It specifically looks for the `Orientation` tag.  Any value other than 1 indicates rotation. Error handling is included to gracefully manage potential issues during EXIF data reading.


## 4. `detect_orientation_using_ocr` Function

This function employs Optical Character Recognition (OCR) to infer the orientation of an image.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The full path to the image file. |

**Return Value:**
* `True`: if the image is detected as rotated (best orientation is not 0 degrees).
* `False`: if the image is detected as not rotated (best orientation is 0 degrees).
* `None`: if an error occurs during OCR processing.


### 4.1 Algorithm Details

The algorithm works as follows:

1. **Image Rotation:** The input image is rotated by 0, 90, 180, and 270 degrees.
2. **OCR Processing:**  `pytesseract` performs OCR on each rotated image.  The `image_to_data` function with `output_type=pytesseract.Output.DICT` is used to get both text and confidence scores for each detected text block.
3. **Confidence Score Calculation:** For each rotation, a confidence score is calculated. This score considers both the average confidence of detected text blocks and the number of text blocks with confidence greater than 50%. A higher score indicates a more likely correct orientation.  The formula is: `score = avg_conf * text_blocks`.
4. **Orientation Determination:** The rotation angle with the highest confidence score is chosen as the best orientation. If this angle is not 0, the image is considered rotated.

This approach is robust to images with low quality or limited text.  The combined use of average confidence and the number of text blocks helps to reduce the effect of isolated high-confidence errors.


## 5. `is_image_rotated` Function

This function acts as a wrapper, combining EXIF data and OCR methods to determine if an image is rotated.  It prioritizes EXIF data if available; otherwise, it falls back to OCR.

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | The full path to the image file. |

**Return Value:** `True` if the image is rotated, `False` otherwise.


## 6. `main` Function

This function orchestrates the entire process:

1. **Argument Parsing:** It uses `argparse` to handle command-line arguments for input directory and output CSV file.
2. **Image Processing:** It iterates through image files, using `is_image_rotated` to determine orientation for each.  Progress is printed every 10 images.
3. **CSV Report Generation:**  Results are written to a CSV file with columns for image name and orientation status (`Yes` or `No`).  Error handling is not explicitly included within this function because the underlying functions already handle potential errors.

The `main` function provides a user-friendly interface for batch processing of images and generating a concise report.
