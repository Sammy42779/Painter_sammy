# To represent the relationship between two images, one being a random image and the other being the corresponding task output image (such as segmentation output), you can use a few metrics or visualizations:

# Pixel-wise comparison: 
    # You can compare the two images pixel by pixel to see how much they differ. 
    # This can be done using metrics like Mean Squared Error (MSE) or Structural Similarity Index (SSIM).
# Overlay: 
    # Overlay the segmentation output on the original image to visualize the result directly. 
    # By doing this, you can get an idea of how well the segmentation matches the original image.
# Side-by-side comparison: 
    # Display the original image and the task output image side by side for a visual comparison. 
    # This can help you see how well the task output aligns with the original image.
# Feature space comparison: 
    # You can use feature extraction techniques like deep learning-based image encoders (e.g., pre-trained CNNs, vision transformers) to obtain feature representations of the images. 
    # Then, you can compute the similarity between these feature representations, such as by calculating the cosine similarity or Euclidean distance.
# Correlation: 
    # Calculate the correlation between the pixel values of the original image and the task output image. 
    # A high correlation would indicate a strong relationship between the two images.
# Remember that the appropriate method to represent the relationship between the two images depends on the specific task and the desired level of detail in the comparison.


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torchvision import models

# Load images (replace 'image_path' and 'seg_path' with actual paths)
image = cv2.imread('image_path')
seg_image = cv2.imread('seg_path')

# Convert images to grayscale if necessary
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
seg_gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)

# Mean Squared Error (MSE)
mse = np.mean((image_gray - seg_gray) ** 2)
print(f'Mean Squared Error: {mse}')

# Structural Similarity Index (SSIM)
ssim_value = ssim(image_gray, seg_gray)
print(f'SSIM: {ssim_value}')

# Overlay segmentation output on the original image
alpha = 0.5  # transparency factor
overlay = cv2.addWeighted(image, alpha, seg_image, 1 - alpha, 0)
cv2.imshow('Overlay', overlay)
cv2.waitKey(0)

# Feature space comparison using a pre-trained CNN (ResNet18)
resnet = models.resnet18(pretrained=True).eval()

def get_image_features(image):
    image = cv2.resize(image, (224, 224))  # Resize to the expected input size
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        features = resnet(image).squeeze()
    return features

image_features = get_image_features(image)
seg_features = get_image_features(seg_image)

# Cosine similarity
cos_sim = F.cosine_similarity(image_features, seg_features, dim=0)
print(f'Cosine Similarity: {cos_sim.item()}')

# Euclidean distance
euclidean_distance = torch.dist(image_features, seg_features)
print(f'Euclidean Distance: {euclidean_distance.item()}')

# Correlation
correlation = np.corrcoef(image_gray.flatten(), seg_gray.flatten())[0, 1]
print(f'Correlation: {correlation}')
