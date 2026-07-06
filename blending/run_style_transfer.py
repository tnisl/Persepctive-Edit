"""
Demo script for running blending style transfer
This script demonstrates how to use the 4 required input variables:
- left_face_mask: Mask image defining the region
- left_face_mask_content: Source content image to blend
- target_image: Background/target image
- style_image: Style reference image
"""

import matplotlib
matplotlib.use("Agg")

from PIL import Image
import matplotlib.pyplot as plt
from blending_style_transfer import run_blending_style_transfer, download_vgg_model


def main():
    # Download VGG model if not present
    print("Checking VGG model...")
    download_vgg_model()
    
    # Load the 4 required input images from Kaggle datasets
    print("\nLoading input images from Kaggle datasets...")
    
    left_face_mask = Image.open('/kaggle/input/datasets/lctrnnguynkhi/material-images/left_face_mask.png')
    left_face_mask_content = Image.open('/kaggle/input/datasets/lctrnnguynkhi/material-images/left_face_mask_content.png')
    target_image = Image.open('/kaggle/input/datasets/cymeu4l0t/project-images/portrait.jpg')
    style_image = Image.open('/kaggle/input/datasets/cymeu4l0t/project-images/style1.jpg')
    
    print(f"Mask size: {left_face_mask.size}")
    print(f"Content size: {left_face_mask_content.size}")
    print(f"Target size: {target_image.size}")
    print(f"Style size: {style_image.size}")
    
    # Run the blending style transfer
    print("\nStarting blending style transfer...")
    result = run_blending_style_transfer(
        source_img=left_face_mask_content,
        mask_img=left_face_mask,
        target_img=target_image,
        style_img=style_image,
        num_steps=300,  # Adjust for quality vs speed tradeoff
        max_side=512
    )
    
    # Save and display result
    print("\nSaving result...")
    result.save('result.jpg')
    
    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Blending Style Transfer Result')
    plt.tight_layout()
    plt.savefig('result_preview.png', dpi=150, bbox_inches='tight')
    print("Result saved to 'result.jpg' and 'result_preview.png'")
    
    # Optionally show the plot
    plt.show()


if __name__ == "__main__":
    main()