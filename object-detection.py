import cv2
import numpy as np

def object_detection(source_path, template_path):
    source = cv2.imread(source_path)
    template = cv2.imread(template_path)
    if source is None or template is None:
        raise ValueError("Image(s) not found or unable to load.")
    
    result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    h, w = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(source, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("Detected Object", source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return top_left, bottom_right

# Example usage
try:
    tl, br = object_detection("src2.jpg", "temp.jpg")
    print(f"Object detected at: Top-left {tl}, Bottom-right {br}")
except ValueError as e:
    print(e)