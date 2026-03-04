# =============================================
# IMPORT LIBRARIES
# =============================================
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# =============================================
# 1️⃣ DEVICE CONFIGURATION
# =============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =============================================
# 2️⃣ LOAD PRETRAINED MASK R-CNN
# =============================================
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()

print("Mask R-CNN Model Loaded Successfully!")

# =============================================
# 3️⃣ UPLOAD IMAGE
# =============================================
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found! Please upload valid image.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image_rgb.shape

# =============================================
# 4️⃣ SHOW ORIGINAL IMAGE
# =============================================
plt.figure(figsize=(8,6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# =============================================
# 5️⃣ GRID REPRESENTATION (Image as Matrix)
# =============================================
grid_image = image_rgb.copy()

rows = 8
cols = 8

row_step = height // rows
col_step = width // cols

# Horizontal lines
for r in range(rows):
    y = r * row_step
    cv2.line(grid_image, (0, y), (width, y), (0,255,0), 1)

# Vertical lines
for c in range(cols):
    x = c * col_step
    cv2.line(grid_image, (x, 0), (x, height), (0,255,0), 1)

plt.figure(figsize=(8,6))
plt.imshow(grid_image)
plt.title("Grid Representation (Image as Matrix)")
plt.axis("off")
plt.show()

# =============================================
# 6️⃣ CONVERT IMAGE TO TENSOR
# =============================================
image_tensor = F.to_tensor(image_rgb).to(device)

# =============================================
# 7️⃣ RUN MASK R-CNN INFERENCE
# =============================================
with torch.no_grad():
    outputs = model([image_tensor])

# =============================================
# 8️⃣ EXTRACT PREDICTIONS
# =============================================
boxes = outputs[0]["boxes"].cpu().numpy()
scores = outputs[0]["scores"].cpu().numpy()
labels = outputs[0]["labels"].cpu().numpy()
masks = outputs[0]["masks"].cpu().numpy()

coco_labels = weights.meta["categories"]

# Confidence threshold
threshold = 0.5
keep = scores >= threshold

boxes = boxes[keep]
scores = scores[keep]
labels = labels[keep]
masks = masks[keep]

print("\nDetected Objects:", len(boxes))

# =============================================
# 9️⃣ SHOW INDIVIDUAL MASKS
# =============================================
for i in range(len(masks)):
    mask = masks[i, 0]

    plt.figure(figsize=(4,4))
    plt.imshow(mask, cmap='gray')
    plt.title(f"Mask: {coco_labels[labels[i]]} | Score: {scores[i]:.2f}")
    plt.axis("off")
    plt.show()

# =============================================
# 🔟 OVERLAY MASK + BOUNDING BOXES
# =============================================
final_image = image_rgb.copy()

for i in range(len(boxes)):

    x1, y1, x2, y2 = boxes[i].astype(int)

    # Draw Bounding Box
    cv2.rectangle(final_image, (x1,y1), (x2,y2), (0,255,0), 2)

    label_text = f"{coco_labels[labels[i]]}: {scores[i]:.2f}"
    cv2.putText(final_image, label_text,
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2)

    # Process Mask
    mask = masks[i, 0]
    mask = mask > 0.5

    colored_mask = np.zeros_like(final_image)
    colored_mask[mask] = [255, 0, 0]  # Red mask

    final_image = cv2.addWeighted(final_image, 1, colored_mask, 0.5, 0)

# =============================================
# 1️⃣1️⃣ FINAL OUTPUT
# =============================================
plt.figure(figsize=(10,8))
plt.imshow(final_image)
plt.title("Final Output: Mask R-CNN Instance Segmentation")
plt.axis("off")
plt.show()