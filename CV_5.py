from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# load pre trained YOLO model

model = YOLO('yolov8n.pt')

#load input
image_path = "CV5(1).jpg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}. Please ensure the image file is uploaded to Colab.")

image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

results = model(image_rgb)

annotated_image = results[0].plot()

plt.figure(figsize=(10, 8))
plt.imshow(annotated_image)
plt.title("YOLO Object Detection")
plt.axis("off")
plt.show()