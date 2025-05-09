import os

import cv2
import numpy as np

def extract_digit_color(digit_image):
    """
    Extracts the dominant color of a digit with robustness for small, blurry images

    Args:
        digit_image: Cropped image of a single digit

    Returns:
        Dictionary with color information
    """
    # Convert to RGB (OpenCV uses BGR)
    rgb_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2RGB)

    # Create mask for the digit
    gray = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphology to improve mask
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Convert to HSV for better color analysis
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Get foreground pixels using mask
    foreground_hsv = hsv_image[binary_mask > 0]

    if len(foreground_hsv) == 0:
        return {'is_red': False, 'confidence': 0.0, 'dominant_rgb': [0, 0, 0], 'hue_value': 0}

    # Calculate histogram of hue for better noise resistance
    hist = cv2.calcHist([hsv_image], [0], binary_mask, [30], [0, 180])
    hist = hist / np.sum(hist)  # Normalize

    # Red hue in OpenCV HSV is around 0-10 and 170-180
    red_bins = np.sum(hist[0:2]) + np.sum(hist[28:30])  # 0-12 and 168-180 degrees

    # Get median saturation and value (more robust than mean for small samples)
    median_sat = np.median(foreground_hsv[:, 1])
    median_val = np.median(foreground_hsv[:, 2])

    # Calculate median hue (considering the circular nature of hue)
    hues = foreground_hsv[:, 0]
    # Handle circular nature of hue for better median calculation
    sin_hue = np.sin(hues * np.pi / 90)  # 180->π
    cos_hue = np.cos(hues * np.pi / 90)
    median_hue = np.arctan2(np.median(sin_hue), np.median(cos_hue)) * 90 / np.pi
    if median_hue < 0:
        median_hue += 180

    # Classify as red with multiple criteria
    is_red = (red_bins > 0.4) and (median_sat > 80)

    # Calculate confidence
    if median_hue <= 15 or median_hue >= 165:
        # Higher confidence near pure red hues
        hue_conf = 1.0 - min(abs(median_hue), abs(180-median_hue)) / 15
    else:
        hue_conf = 0

    # Weight confidence by saturation and value
    confidence = hue_conf * (median_sat/255) * (median_val/255)

    # Get dominant RGB for visualization
    dominant_hsv = np.array([[[median_hue, median_sat, median_val]]], dtype=np.uint8)
    dominant_bgr = cv2.cvtColor(dominant_hsv, cv2.COLOR_HSV2BGR)
    dominant_rgb = dominant_bgr[0, 0, ::-1]

    return {
        'is_red': is_red,
        'confidence': float(confidence),
        'dominant_rgb': dominant_rgb.tolist(),
        'hue_value': float(median_hue)
    }

def classify_last_digit_color(digit_images):
    """
    Determine if the last digit in a sequence is red (decimal) or black

    Args:
        digit_images: List of cropped digit images in reading order

    Returns:
        Dictionary with classification results:
        {
            'is_decimal': bool,       # Whether the last digit is classified as a decimal (red)
            'confidence': float,      # Confidence level (0-1)
            'color_analysis': dict    # Details from extract_digit_color
        }
    """
    if not digit_images or len(digit_images) == 0:
        return {
            'is_decimal': False,
            'confidence': 0.0,
            'color_analysis': None
        }

    # Get the last digit image
    last_digit = digit_images[-1]

    # Extract color information
    color_analysis = extract_digit_color(last_digit)

    return {
        'is_decimal': color_analysis['is_red'],
        'confidence': color_analysis['confidence'],
        'color_analysis': color_analysis
    }

def visualize_color_analysis(digit_images, save_path=None):
    """
    Create a visualization of the color analysis for debugging

    Args:
        digit_images: List of cropped digit images in reading order
        save_path: Optional path to save the visualization

    Returns:
        Visualization image
    """
    if not digit_images or len(digit_images) == 0:
        return None

    # Get the last digit and its color analysis
    last_digit = digit_images[-1]
    color_analysis = extract_digit_color(last_digit)

    # Create a visualization image
    # Resize the last digit for better visibility
    height, width = last_digit.shape[:2]
    display_height = 200
    display_width = int(width * (display_height / height))
    displayed_digit = cv2.resize(last_digit, (display_width, display_height))

    # Create a color sample patch
    dominant_rgb = color_analysis['dominant_rgb']
    color_patch = np.full((display_height, 100, 3),
                          [dominant_rgb[2], dominant_rgb[1], dominant_rgb[0]],
                          dtype=np.uint8)

    # Create a blank area for text
    text_area = np.full((display_height, 400, 3), [255, 255, 255], dtype=np.uint8)

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)  # Black text
    thickness = 2

    # Add color information to the text area
    texts = [
        f"Is Red: {color_analysis['is_red']}",
        f"Confidence: {color_analysis['confidence']:.2f}",
        f"RGB: {color_analysis['dominant_rgb']}",
        f"Hue: {color_analysis['hue_value']:.1f}",
        f"Decimal Point: {color_analysis['is_red']}"
    ]

    # Add each line of text
    for i, text in enumerate(texts):
        position = (10, 40 + i * 30)
        cv2.putText(text_area, text, position, font, font_scale, color, thickness)

    # Combine images horizontally
    visualization = np.hstack([displayed_digit, color_patch, text_area])

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, visualization)

    return visualization

def process_meter_reading_with_decimal(digit_images, digit_classes):
    """
    Generate the full meter reading including decimal point if present

    Args:
        digit_images: List of cropped digit images in reading order
        digit_classes: List of recognized digit classes (0-9) as strings

    Returns:
        Dictionary with:
        {
            'reading': str,           # Full reading with decimal point if applicable
            'has_decimal': bool,      # Whether a decimal point was detected
            'confidence': float,      # Confidence in the decimal classification
            'decimal_analysis': dict  # Full color analysis data
        }
    """
    if not digit_images or len(digit_images) == 0 or not digit_classes:
        return {
            'reading': '',
            'has_decimal': False,
            'confidence': 0.0,
            'decimal_analysis': None
        }

    # Classify if the last digit is a decimal
    decimal_analysis = classify_last_digit_color(digit_images)

    # Construct the reading
    reading = ''.join(digit_classes)
    has_decimal = decimal_analysis['is_decimal']

    # If the last digit is a decimal, add a decimal point before it
    if has_decimal and len(reading) > 1:
        reading = reading[:-1] + '.' + reading[-1]

    return {
        'reading': reading,
        'has_decimal': has_decimal,
        'confidence': decimal_analysis['confidence'],
        'decimal_analysis': decimal_analysis
    }
