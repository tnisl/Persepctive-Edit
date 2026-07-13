"""
Facial Landmark Detection and Face Segmentation Module
Uses face-alignment library and BiSeNet for face parsing
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import face_alignment

from model import BiSeNet


# Disable torch.compile for compatibility
torch.compile = lambda model, *args, **kwargs: model


def facial_landmark(img, fa):
    """
    Detect facial landmarks using face-alignment library.
    
    Args:
        img: Input image (numpy array, RGB)
        fa: FaceAlignment model instance
    
    Returns:
        preds: List of facial landmark coordinates
    """
    h, w, _ = img.shape
    bbox = [0, 0, w, h]
    preds = fa.get_landmarks_from_image(img, detected_faces=[bbox])
    return preds


def viz(img, preds):
    """
    Visualize facial landmarks on image.
    
    Args:
        img: Input image (numpy array, RGB)
        preds: Facial landmark predictions
    """
    viz_img = img.copy()
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        cv2.circle(viz_img, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(viz_img, str(i), (x + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(viz_img)
    plt.axis('off')


def calculate_centroid(preds, mask):
    """
    Calculate centroid of masked facial landmarks.
    
    Args:
        preds: Facial landmark predictions
        mask: Binary mask image
    
    Returns:
        centroid: (x, y) coordinates of centroid
        bucket: Array of masked landmark points
    """
    bucket = []
    
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        
        if y <= mask.shape[0] and x <= mask.shape[1] and mask[y, x, 0] != 0:
            bucket.append([x, y])
    
    bucket = np.array(bucket)
    centroid = np.mean(bucket, axis=0)
    
    return centroid, bucket


def load_bisenet_model(weight_path='/kaggle/input/datasets/lctrnnguynkhi/bisenet-checkpoint/79999_iter.pth'):
    """
    Load BiSeNet face parsing model.
    
    Args:
        weight_path: Path to model weights
    
    Returns:
        net: BiSeNet model
        device: torch device
    """
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()
    return net, device


def segment_face(img_bgr, net, device):
    """
    Segment face using BiSeNet model.
    
    Args:
        img_bgr: Input image in BGR format (numpy array)
        net: BiSeNet model
        device: torch device
    
    Returns:
        pure_face_mask_orig: Binary mask of face segmentation
        only_face_result: Image with only face region
    """
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Resize and prepare image
    h_orig, w_orig = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor_img = transform(img_rgb).unsqueeze(0)
    
    # Run inference
    tensor_img = tensor_img.to(device)
    
    with torch.no_grad():
        output = net(tensor_img)[0]
        parsing_mask = output.squeeze(0).argmax(0).cpu().numpy()
    
    # Extract face labels: skin(1), eyebrows(2-3), eyes(4-5), nose(10), mouth(11), lips(12-13)
    face_ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]
    pure_face_mask = np.isin(parsing_mask, face_ids).astype(np.uint8) * 255
    
    # Resize back to original dimensions
    pure_face_mask_orig = cv2.resize(pure_face_mask, (w_orig, h_orig))
    
    # Apply mask to image
    mask_3ch = np.repeat(pure_face_mask_orig[:, :, np.newaxis], 3, axis=2) / 255.0
    only_face_result = (img_bgr * mask_3ch).astype(np.uint8)
    
    return pure_face_mask_orig, only_face_result


def align_and_blend(img, rotated_img, mask, preds, rotated_preds):
    """
    Align and blend faces based on landmarks.
    
    Args:
        img: Original image (numpy array, RGB)
        rotated_img: Rotated/transformed image (numpy array, RGB)
        mask: Binary mask
        preds: Landmarks for original image
        rotated_preds: Landmarks for rotated image
    
    Returns:
        result: Blended result image
    """
    # Calculate centroids
    centroid, _ = calculate_centroid(preds, mask)
    rotated_centroid, _ = calculate_centroid(rotated_preds, mask)
    
    # Calculate translation vector
    vector = rotated_centroid - centroid
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, vector[0]],
                                     [0, 1, vector[1]]])
    
    # Apply translation to mask
    moved_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]), 
                                borderValue=(0, 0, 0)) / 255
    
    # Crop rotated image with moved mask
    mask_cropped_rotation = (rotated_img * moved_mask).astype(np.uint8)
    
    # Reverse translation
    reverse_matrix = np.float32([[1, 0, -vector[0]],
                                 [0, 1, -vector[1]]])
    
    mask_result = cv2.warpAffine(mask_cropped_rotation, reverse_matrix, 
                                 (mask.shape[1], mask.shape[0]), borderValue=(0, 0, 0))
    
    # Calculate bounding box center for seamless clone
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    
    if mask_gray.max() <= 1.0:
        mask_gray = (mask_gray * 255).astype(np.uint8)
    else:
        mask_gray = mask_gray.astype(np.uint8)
    
    x, y, w, h = cv2.boundingRect(mask_gray)
    c_bbox = (x + w // 2, y + h // 2)
    
    # Seamless cloning
    result = cv2.seamlessClone(mask_result, img, mask_gray, 
                               (int(c_bbox[0]), int(c_bbox[1])), cv2.NORMAL_CLONE)
    
    return result


def visualize_landmark_matching(img, rotated_img, mask, preds, rotated_preds):
    """
    Visualize landmark matching between two images.
    
    Args:
        img: Original image
        rotated_img: Rotated image
        mask: Binary mask
        preds: Landmarks for original image
        rotated_preds: Landmarks for rotated image
    
    Returns:
        concat_img: Concatenated visualization image
    """
    img_cp = img.copy()
    rotated_img_cp = rotated_img.copy()
    
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        
        if y <= mask.shape[0] and x <= mask.shape[1] and mask[y, x, 0] != 0:
            cv2.circle(img_cp, (x, y), 4, (0, 255, 0), -1)
            dx, dy = int(rotated_preds[0][i][0]), int(rotated_preds[0][i][1])
            cv2.circle(rotated_img_cp, (dx, dy), 4, (255, 0, 0), -1)
    
    concat_img = cv2.hconcat([img_cp, rotated_img_cp])
    
    # Draw connecting lines
    for i, pts in enumerate(preds[0]):
        x, y = int(pts[0]), int(pts[1])
        
        if y <= mask.shape[0] and x <= mask.shape[1] and mask[y, x, 0] != 0:
            dx, dy = int(rotated_preds[0][i][0]), int(rotated_preds[0][i][1])
            cv2.line(concat_img, (x, y), (dx + img.shape[1], dy), (0, 0, 255), 2)
    
    return concat_img