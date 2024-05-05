import numpy as np
import cv2
import pandas as pd

def load_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def create_mask(shape, bbox):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = map(int, bbox)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

def min_max_normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)

def apply_mask(image, mask):
    if image is None or mask is None:
        return None  # Proper handling if the image or mask is not available

    mask_boolean = mask > 0
    expanded_mask = np.repeat(mask_boolean[:, :, np.newaxis], 3, axis=2)

    if image.shape[-1] != 3:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    return image * expanded_mask

def load_data(df):
    images = []
    masks = []
    labels = []
    masked_images = []

    for index, row in df.iterrows():
        img = load_image(row['full_filepath'])
        if img is not None:
            bbox = tuple(row['bbox']) if isinstance(row['bbox'], list) else row['bbox']
            img = min_max_normalize(img)
            mask = create_mask(img.shape, bbox)
            img = resize_image(img)
            mask = resize_image(mask)
            img = img.astype('float32')
            mask = mask.astype('float32')

            masked_img = apply_mask(img, mask)

            images.append(img)
            masks.append(mask)
            masked_images.append(masked_img)
            labels.append(row['category_id'])
        else:
            print(f"Failed to load image at {row['full_filepath']}")

    return np.array(masked_images), np.array(images), np.array(masks), np.array(labels)