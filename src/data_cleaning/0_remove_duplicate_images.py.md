# Internal Code Documentation: Duplicate Image Removal

[Linked Table of Contents](#linked-table-of-contents)

## Linked Table of Contents

* [1. Overview](#1-overview)
* [2. Function: `remove_duplicate_images`](#2-function-remove_duplicate_images)
    * [2.1 Function Parameters](#21-function-parameters)
    * [2.2 Function Logic and Algorithm](#22-function-logic-and-algorithm)
    * [2.3 Regular Expression Explanation](#23-regular-expression-explanation)


## 1. Overview

This document details the functionality of the Python script designed to identify and remove duplicate images from a specified source folder.  The script leverages regular expressions to detect files with sequential numbering (e.g., `image.jpg`, `image(1).jpg`, `image(2).jpg`), considering only the file without the sequential numbering as the original.  It then copies the original images to a destination folder, effectively removing the duplicates.


## 2. Function: `remove_duplicate_images`

This function is the core of the script. It takes a source folder path and a destination folder path as input, processes the images, and provides a summary of the operation.

### 2.1 Function Parameters

| Parameter Name        | Data Type | Description                                                                 |
|-----------------------|------------|-----------------------------------------------------------------------------|
| `source_folder`      | String     | Path to the folder containing the images to process.                       |
| `destination_folder` | String     | Path to the folder where the non-duplicate images will be copied.          |


### 2.2 Function Logic and Algorithm

The `remove_duplicate_images` function employs the following algorithm:

1. **Destination Folder Creation:** Checks if the destination folder exists; if not, it creates it.

2. **Initialization:** Initializes an empty dictionary `original_files` to store filenames of original images (without sequential numbering) and count variables to track original and final image counts.

3. **Regular Expression Matching:** Defines a regular expression pattern (`pattern`) to identify file names with optional sequential numbering in parentheses.  This pattern is explained in more detail below.

4. **Iteration and Processing:** Iterates through each file in the `source_folder`. For each file with a `.jpg`, `.jpeg`, or `.png` extension:
    * It attempts to match the filename against the regular expression.
    * If a match is found, it extracts the base filename and extension.
    * If the filename does not contain '(', indicating it's the original, it's added to the `original_files` dictionary, and a copy of the file is created in the `destination_folder` using `shutil.copy2` (which preserves metadata).

5. **Counting and Reporting:** After processing all files, it calculates and prints the original number of images, the number of images after removing duplicates, and the number of duplicates removed.


### 2.3 Regular Expression Explanation

The regular expression `r'(.+?)(?:\(\d+\))?\.(jpg|jpeg|png)$'` is used to parse filenames. Let's break it down:

* `(.+?)`: This captures one or more characters (`.+`), non-greedily (`?`). This part captures the base filename.
* `(?:\(\d+\))?`: This is a non-capturing group (`(?: ... )`). It matches an optional part:
    * `\(\d+\)`: Matches an opening parenthesis `\(`, followed by one or more digits `\d+`, and a closing parenthesis `\)`.  This represents the sequential numbering.
    * `?`: Makes the entire group optional.
* `\.`: Matches a literal dot.
* `(jpg|jpeg|png)`: Matches either "jpg", "jpeg", or "png". This is another capturing group, capturing the file extension.
* `$`: Matches the end of the string.  This ensures that the entire filename is matched.

This regex efficiently identifies both original and numbered files, allowing for the extraction of the base filename and extension.
