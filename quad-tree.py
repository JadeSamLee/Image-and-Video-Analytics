import cv2
import numpy as np

def quadtree(image, threshold=10):
    if image.size <= 1 or np.var(image) < threshold:
        return {"value": np.mean(image), "leaf": True}
    h, w = image.shape
    half_h, half_w = h // 2, w // 2
    return {
        "top_left": quadtree(image[:half_h, :half_w], threshold),
        "top_right": quadtree(image[:half_h, half_w:], threshold),
        "bottom_left": quadtree(image[half_h:, :half_w], threshold),
        "bottom_right": quadtree(image[half_h:, half_w:], threshold),
        "leaf": False
    }

def print_quadtree(tree, level=0):
    if tree["leaf"]:
        print("  " * level + f"Leaf: {tree['value']:.2f}")
    else:
        print("  " * level + "Node:")
        print_quadtree(tree["top_left"], level + 1)
        print_quadtree(tree["top_right"], level + 1)
        print_quadtree(tree["bottom_left"], level + 1)
        print_quadtree(tree["bottom_right"], level + 1)

# Example usage
image = cv2.imread("cat.jpeg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found or unable to load.")
else:
    tree = quadtree(image, threshold=10)
    print_quadtree(tree)