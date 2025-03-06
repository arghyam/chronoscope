# Internal Code Documentation: Duplicate Image Removal

## Table of Contents

* [1. Overview](#1-overview)
* [2. `get_image_hash` Function](#2-get_image_hash-function)
* [3. `remove_duplicate_images` Function](#3-remove_duplicate-images-function)
* [4. Main Execution Block](#4-main-execution-block)


<a name="1-overview"></a>
## 1. Overview

This document details the Python code used to identify and remove duplicate images from a directory. The code utilizes perceptual hashing for efficient comparison of visual similarity, and falls back to MD5 hashing for byte-level comparison if requested.  Unique images are copied to a designated output directory, handling potential filename conflicts.

<a name="2-get_image_hash-function"></a>
## 2. `get_image_hash` Function

This function computes a hash value representing an image, using either perceptual hashing or MD5 hashing.

**Function Signature:**

```python
get_image_hash(image_path, hash_method='phash')
```

**Parameters:**

| Parameter      | Type          | Description                                                              |
|-----------------|-----------------|--------------------------------------------------------------------------|
| `image_path`   | `str`          | Path to the image file.                                                  |
| `hash_method` | `str` (default: 'phash') | Hashing algorithm to use. Options: 'phash', 'dhash', 'ahash', 'md5'. |

**Return Value:**

* A string representing the hash value, or `None` if an error occurs during processing.

**Algorithm:**

* **MD5 Hashing (`hash_method == 'md5'`):**  The function calculates the MD5 hash of the image file's raw bytes. This method is sensitive to even minor changes in the file's content.  It is suitable for ensuring bit-for-bit identical images are identified as duplicates.

* **Perceptual Hashing (otherwise):** The function uses the `imagehash` library to calculate a perceptual hash. This represents the image's visual content, making it robust to minor changes like compression artifacts or slight variations in color. The specific type of perceptual hash is determined by the `hash_method` parameter:
    * `'phash'`: Perceptual hash.
    * `'dhash'`: Difference hash.
    * `'ahash'`: Average hash.

**Error Handling:**

The function includes a `try-except` block to catch and report any errors that may occur during image processing (e.g., file not found, invalid image format).  An error message is printed, and `None` is returned.


<a name="3-remove_duplicate-images-function"></a>
## 3. `remove_duplicate_images` Function

This function identifies and removes duplicate images from a source directory, copying unique images to a destination directory.

**Function Signature:**

```python
remove_duplicate_images(input_folder, output_folder, hash_method='phash')
```

**Parameters:**

| Parameter      | Type          | Description                                                              |
|-----------------|-----------------|--------------------------------------------------------------------------|
| `input_folder` | `str`          | Path to the input folder containing images.                               |
| `output_folder` | `str`          | Path to the output folder where unique images will be copied.           |
| `hash_method`  | `str` (default: 'phash') | Hashing method to use for comparison ('phash', 'dhash', 'ahash', 'md5'). |

**Return Value:**

* A tuple containing:  `(number of unique images, number of duplicates found)`


**Algorithm:**

1. **Initialization:** The function creates the output directory if it doesn't exist. It then gathers all image files within the input directory based on a list of supported extensions.

2. **Hash Calculation:** It iterates through all image files, calculating the hash for each using the `get_image_hash` function.  Hashes are stored in a dictionary (`hash_dict`), with the hash value as the key and image path as the value.  If a hash already exists in the dictionary, the image is considered a duplicate; otherwise it's considered unique.

3. **Duplicate Identification:**  The code keeps track of both unique and duplicate images. Duplicate images are stored in the `duplicates` list.

4. **Unique Image Copying:**  Unique images are copied to the output folder using `shutil.copy2` (preserves metadata). Filename conflicts are handled by appending a counter to the filename.

5. **Return Values:** The function returns the count of unique and duplicate images found.


<a name="4-main-execution-block"></a>
## 4. Main Execution Block

The `if __name__ == "__main__":` block demonstrates the usage of the `remove_duplicate_images` function with hardcoded paths.  It utilizes the `'phash'` hash method, which is generally preferred for image similarity. The results (number of unique and duplicate images) are printed to the console.
