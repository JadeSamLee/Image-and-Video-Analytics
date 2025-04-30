import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_t_pyramid(image_path, levels=3):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    pyramid = [image]
    current = image.copy()
    for _ in range(levels - 1):
        # Downscale using OpenCV's pyrDown (Gaussian pyramid)
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    # Display pyramid levels
    plt.figure(figsize=(10, 5))
    for i, level in enumerate(pyramid):
        plt.subplot(1, levels, i+1)
        plt.imshow(level, cmap='gray')
        plt.title(f'Level {i}')
        plt.axis('off')
    plt.show()
    return pyramid

# Example usage
try:
    pyramid = compute_t_pyramid("cat.jpeg", levels=3)
    print(f"Pyramid levels created: {len(pyramid)}")
except ValueError as e:
    print(e)