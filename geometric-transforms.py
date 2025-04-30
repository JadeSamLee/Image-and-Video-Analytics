import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_transforms(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # (a) Rotation (90 degrees)
    rot_matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), 90, 1)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    
    # (b) Change of scale (resize to half)
    scaled = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # (c) Skewing (shear transformation)
    shear_matrix = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    skewed = cv2.warpPerspective(image, shear_matrix, (int(image.shape[1]*1.5), image.shape[0]))
    
    # (d) Affine transform from three pairs
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    affine = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))
    
    # (e) Bilinear transform (perspective from four pairs)
    pts1 = np.float32([[0, 0], [image.shape[1]-1, 0], [0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1]])
    pts2 = np.float32([[0, 0], [image.shape[1]*0.75, 0], [image.shape[1]*0.25, image.shape[0]], [image.shape[1]*0.75, image.shape[0]]])
    persp_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    bilinear = cv2.warpPerspective(image, persp_matrix, (image.shape[1], image.shape[0]))

    # Display results
    titles = ['Original', 'Rotated', 'Scaled', 'Skewed', 'Affine', 'Bilinear']
    images = [image, rotated, scaled, skewed, affine, bilinear]
    plt.figure(figsize=(15, 10))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# Example usage
try:
    apply_transforms("cat.jpeg")
except ValueError as e:
    print(e)