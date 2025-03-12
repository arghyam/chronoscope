import os

import cv2
import numpy as np

def straighten_box(image, bbox_points):
    """
    Straighten an oriented bounding box region using perspective transformation

    Args:
        image: Original image (numpy array)
        bbox_points: List/array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    representing the oriented bounding box corners

    Returns:
        Straightened image of the box region
    """
    # Convert points to numpy array
    pts = np.array(bbox_points, dtype=np.float32)

    # Get width and height of the box
    # Calculate width as the maximum of the two possible widths
    width1 = np.sqrt(((pts[1][0] - pts[0][0]) ** 2) + ((pts[1][1] - pts[0][1]) ** 2))
    width2 = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
    max_width = int(max(width1, width2))

    # Calculate height as the maximum of the two possible heights
    height1 = np.sqrt(((pts[2][0] - pts[1][0]) ** 2) + ((pts[2][1] - pts[1][1]) ** 2))
    height2 = np.sqrt(((pts[3][0] - pts[0][0]) ** 2) + ((pts[3][1] - pts[0][1]) ** 2))
    max_height = int(max(height1, height2))

    # Define destination points for the perspective transform
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply perspective transformation
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

# Example usage:
def main():
    # Load your image
    image = cv2.imread('data/refined_data/field_1505.jpeg')

    # Example bbox points (replace with your actual bbox points from YOLO)
    # Points should be in order: top-left, top-right, bottom-right, bottom-left
     #310.73,802.71;702.69,1048.61;667.56,1135.50;262.66,867.42
    # bbox_points = [
    bbox_points = [
        [310.73, 802.71],  # top-left
        [702.69, 1048.61],  # top-right
        [667.56, 1135.50],  # bottom-right
        [262.66, 867.42]    # bottom-left
    ]

    # Straighten the box region
    straightened_image = straighten_box(image, bbox_points)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save results
    cv2.imwrite('output/original.jpg', image)
    cv2.imwrite('output/straightened.jpg', straightened_image)

if __name__ == "__main__":
    main()
