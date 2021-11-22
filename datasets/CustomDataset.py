from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from utils import open_image
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage
from skimage import measure
from datasets.DataAugmentations import crop_rectangle, rect_bbx, inside_rect, image_rotate_without_crop, crop_rotated_rectangle, adjustment_center

import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, augmentation, r_dir, gt_dir, labels, pytorch=True):
        super().__init__()
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.labels = labels
        self.pytorch = pytorch
        self.augmentations = augmentation

    def combine_files(self, r_file: Path, gt_dir):
        files = {'samples': r_file,
                 'mask': gt_dir/r_file.name.replace(".jpg","_mask.png")}

        return files

    def __len__(self):
        return len(self.files)

    def augmentation(self, image_path, mask_path):
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        crop_w = 768
        crop_h = 768

        w, h = image.size

        # Perform random scale
        random = np.random.uniform(0.7, 1.1)
        scaled_image = image.resize((int(w*random),int(h*random)))
        scaled_mask = mask.resize((int(w*random),int(h*random)))

        # Get the new size of the images
        img_width, img_height = scaled_image.size

        # Detect annotations
        contours = measure.find_contours(np.array(scaled_mask), level=80)

        # Find cetner
        centers = []
        for con in contours:
            if len(con) > 100:
                #Calculate the center of the contour
                x = [p[0] for p in con]
                y = [p[1] for p in con]
                centroid = (sum(x) / len(con), sum(y) / len(con))
                # Store in list
                centers.append(centroid)

        # Variables
        center = None
        half_width = crop_w * 0.5
        half_height = crop_h * 0.5
        # Determine center point
        if np.random.randint(0, 5) == 3 or len(centers) == 0:
            center = (np.random.randint(half_width, img_width - half_width), np.random.randint(half_height, img_height - half_height))
        else:
            # Select random stomata
            position = centers[np.random.randint(0, len(centers))]
            # Adjust center position if out of bounds
            y = adjustment_center(position[0], half_height, 0.1, img_height)
            x = adjustment_center(position[1], half_width, 0.1, img_width)

            center = (x, y)

        # Find rectangle orientation
        angle =  np.random.randint(low = -35, high = 35)
        increment = (0 - angle) / 10
        while True:
            rect = (center, (crop_w, crop_h), angle)
            if inside_rect(rect = rect, num_cols = img_width, num_rows = img_height):
                break
            # Walk towards angle = 0 if unable to fit inside rect
            angle += increment

        # Perform crop:
        image_crop = crop_rotated_rectangle(image = np.array(scaled_image), rect = rect)
        mask_crop = crop_rotated_rectangle(image = np.array(scaled_mask), rect = rect)

        # If unable to crop image_crop len will be 0
        if len(image_crop) == 0 or len(mask_crop) == 0:
            i = i - 1
            return None

        # Create PIL image for further processing
        image_crop = Image.fromarray(image_crop)
        mask_crop = Image.fromarray(mask_crop)

        # Perform random flip with 20% probability
        if np.random.randint(0, 4) == 3:
            if np.random.randint(2) == 1:
                image_crop = image_crop.transpose(Image.FLIP_LEFT_RIGHT)
                mask_crop = mask_crop.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image_crop = image_crop.transpose(Image.FLIP_TOP_BOTTOM)
                mask_crop = mask_crop.transpose(Image.FLIP_TOP_BOTTOM)

        # Perform random filter with 20% probability
        choice = np.random.randint(10)
        if choice == 1:
            image_crop = image_crop.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        elif choice == 2:
            image_crop = image_crop.filter(ImageFilter.GaussianBlur(radius=np.random.randint(0,2)))
        elif choice == 3:
            enhancer = ImageEnhance.Contrast(image_crop)
            image_crop = enhancer.enhance(np.random.uniform(0.5,2))
        elif choice == 4:
            enhancer = ImageEnhance.Sharpness(image_crop)
            image_crop = enhancer.enhance(np.random.uniform(0.5,2))

        image_crop = np.array(image_crop)

        if len(image_crop.shape) == 2:
            image_crop = np.expand_dims(image_crop, axis=2)

        # HWC to CHW
        image_crop = image_crop.transpose((2, 0, 1))

        # Normalise
        if image_crop.max() > 1:
            image_crop = image_crop / 255

        # Mask to numpy array
        mask_crop = np.asarray(mask_crop)
        # Update labels in mask to be between == len(labels)
        for i in range(len(self.labels)):
            mask_crop = np.where(mask_crop == self.labels[i], i, mask_crop)

        return image_crop, mask_crop

    def no_augmentation(self, image_path, mask_path):
        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        # HWC to CHW
        image = image.transpose((2, 0, 1))

        # Normalise
        if image.max() > 1:
            image = image / 255

        # Update labels in mask to be between == len(labels)
        for i in range(len(self.labels)):
           mask = np.where(mask == self.labels[i], i, mask)

        return image, mask

    def __getitem__(self, idx):
        if self.augmentations == 1:
            x,y = self.augmentation(self.files[idx]['samples'], self.files[idx]['mask'])
        else:
            x,y  = self.no_augmentation(self.files[idx]['samples'], self.files[idx]['mask'])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

    def display_item(self, img, mask):
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img)
        axarr[1].imshow(mask)
        plt.show()
