import os
from PIL import Image
import albumentations as A
import numpy as np
from tqdm import tqdm
import cv2


# Paths to folders
optical_flow_folder = "/data2/local_datasets/vimeo_all/condition/optical_flow"
encoded_frame_folder = "/data2/local_datasets/vimeo_all/condition/quality_8"
target_folder = "/data2/local_datasets/vimeo_all/images"

# Load image file names
optical_flow_files = sorted([os.path.join(optical_flow_folder, f) for f in os.listdir(optical_flow_folder) if f.endswith('.png')])
encoded_frame_files = sorted([os.path.join(encoded_frame_folder, f) for f in os.listdir(encoded_frame_folder) if f.endswith('.png')])
target_files = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.png')])

# Ensure all folders have the same number of images
assert len(optical_flow_files) == len(encoded_frame_files) == len(target_files), "Mismatch in number of images across folders."

# Define augmentations
augmentations = A.Compose(
    [
        A.RandomCrop(height=512, width=512),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.4,
            p=1.0
        ),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.2, 0.2), p=1.0),
    ],
    additional_targets={"encoded_frame": "image", "target_image": "image"},
)

# Function to preprocess, augment, and save images
def preprocess_augment_and_save(optical_flow_path, encoded_frame_path, target_image_path, size=(512, 512)):
    """Preprocesses, applies augmentations, and saves augmented images."""
    # Load and resize images
    optical_flow = np.array(Image.open(optical_flow_path).resize(size).convert('RGB'))
    encoded_frame = np.array(Image.open(encoded_frame_path).resize(size).convert('RGB'))
    target_image = np.array(Image.open(target_image_path).resize(size).convert('RGB'))

    # Apply augmentations
    augmented = augmentations(image=optical_flow, encoded_frame=encoded_frame, target_image=target_image)

    # Helper function to save images
    def save_image(image_array, save_path):
        """Saves an image to the given path."""
        aug_image = Image.fromarray(image_array)
        aug_image.save(save_path)

    # Generate augmented file paths
    optical_flow_aug_path = os.path.splitext(optical_flow_path)[0] + "_aug.png"
    encoded_frame_aug_path = os.path.splitext(encoded_frame_path)[0] + "_aug.png"
    target_image_aug_path = os.path.splitext(target_image_path)[0] + "_aug.png"

    print(optical_flow_aug_path, encoded_frame_aug_path, target_image_aug_path)

    # Save augmented images
    save_image(augmented["image"], optical_flow_aug_path)
    save_image(augmented["encoded_frame"], encoded_frame_aug_path)
    save_image(augmented["target_image"], target_image_aug_path)

    return optical_flow_aug_path, encoded_frame_aug_path, target_image_aug_path


# Main loop to process and save all images
for optical_flow, encoded_frame, target_image in tqdm(
    zip(optical_flow_files, encoded_frame_files, target_files),
    total=len(optical_flow_files),
):
    augmented_paths = preprocess_augment_and_save(optical_flow, encoded_frame, target_image)
    print(f"Saved augmented images: {augmented_paths}")
