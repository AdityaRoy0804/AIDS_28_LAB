import cv2
import matplotlib.pyplot as plt

# read image

image = cv2.imread('cv4.jpg')
# Display original image

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

#convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray

# Create SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
sift_image = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Show results
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Keypoints (Total: {len(keypoints)})")
plt.axis("off")
plt.show()