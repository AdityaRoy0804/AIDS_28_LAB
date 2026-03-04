import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure,data
from skimage.io import imread, imshow
from skimage.transform import resize

#1 . reading image
img = cv2.imread("/content/tiger.jpg")

if img is None:
  raise ValueError("Image not found. Check the image path.")


image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)

# Convert to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

#Apply hog
features, hog_image = hog(
    gray,
    orientations = 9,
    pixels_per_cell = (8,8),
    cells_per_block = (2,2),
    block_norm = 'L2-Hys',
    visualize= True
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,
10))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("HOG Feature Visualization")
plt.imshow(hog_image_rescaled, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
# 4. Print Feature Vector Size
print("HOG feature vector length:", len(features))