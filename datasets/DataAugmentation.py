import glob
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import numpy as np
import cv2
import os
import random as r
from scipy import ndimage
from skimage import measure

def crop_rectangle(image, rect):
    # rect has to be upright
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]

    image = image[rect_center_y-rect_height//2:rect_center_y+rect_height-rect_height//2, rect_center_x-rect_width//2:rect_center_x+rect_width-rect_width//2]
    return image

def rect_bbx(rect):
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:,0]))
    x_min = int(np.min(box[:,0]))
    y_max = int(np.max(box[:,1]))
    y_min = int(np.min(box[:,1]))

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return (center, (width, height), angle)

def inside_rect(rect, num_cols, num_rows):
    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    rect_width, rect_height = rect[1]

    rect_angle = rect[2]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:,0]))
    x_min = int(np.min(box[:,0]))
    y_max = int(np.max(box[:,1]))
    y_min = int(np.min(box[:,1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False

def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)

    return rotated_mat

def crop_rotated_rectangle(image, rect):
    # Crop a rotated rectangle from a image
    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect = rect, num_cols = num_cols, num_rows = num_rows):
        print("Proposed rectangle is not fully in the image.")
        return []

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect = rect)
    rect_bbx_upright_image = crop_rectangle(image = image, rect = rect_bbx_upright)

    rotated_rect_bbx_upright_image = image_rotate_without_crop(mat = rect_bbx_upright_image, angle = rotated_angle)

    rect_width = rect[1][0]
    rect_height = rect[1][1]

    crop_center = (rotated_rect_bbx_upright_image.shape[1]//2, rotated_rect_bbx_upright_image.shape[0]//2)

    return rotated_rect_bbx_upright_image[crop_center[1]-rect_height//2 : crop_center[1]+(rect_height-rect_height//2), crop_center[0]-rect_width//2 : crop_center[0]+(rect_width-rect_width//2)]


def adjustment_center(position, half_crop, jitter, upper_bounds):
    # Adjust center position if out of bounds
    if position - (half_crop) <= 0:
        y_low = half_crop
    elif position + (half_crop) >= upper_bounds:
        y_low = upper_bounds - (half_crop)
    else:
        y_low = position
        iteration = 0
        found = False
        while iteration < 50:
            adjustment = (jitter / 50) * iteration
            y_low = y_low * np.random.uniform((1 - jitter) + adjustment, (1 + jitter) - adjustment)
            if y_low - (half_crop) >= 0 and y_low  + (half_crop) <= upper_bounds:
                found = True
                break
            iteration += 1
        if not found:
            y_low = position
    return y_low


if __name__ == "__main__":
    # Variables
    original_imgs = "C:/Users/jonat/Documents/stomata/wheat/"
    sample_dir = original_imgs + "data_samples/"
    mask_dir = original_imgs + "data_mask/"

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    original = glob.glob(original_imgs + "*.jpg")

    crop_w = 768
    crop_h = 768
    scale = True
    number_samples = 30

    # Iterate over images
    for image in original:
        im = Image.open(image).convert("RGB")
        mask = Image.open(image.replace(".jpg", "_mask.png")).convert("L")

        img_name = Path(image).stem
        print(img_name)

        w, h = im.size

        for i in range(0, number_samples):
            original_cropped, mask_cropped = None, None

            if scale:
                random = np.random.uniform(0.6, 1.1)
                scaled_image = im.resize((int(w*random),int(h*random)))
                scaled_mask = mask.resize((int(w*random),int(h*random)))
            else:
                scaled_image = im
                scaled_mask = mask

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

            # Select random stomata
            if len(centers) == 0: continue
            position = centers[np.random.randint(0, len(centers))]

            half_width = crop_w * 0.5
            half_height = crop_h * 0.5
            # Adjust center position if out of bounds
            y = adjustment_center(position[0], half_height, 0.1, img_height)
            x = adjustment_center(position[1], half_width, 0.1, img_width)

            # Determine center point
            if np.random.randint(0, 5) == 3:
                center = (np.random.randint(half_width, img_width - half_width), np.random.randint(half_height, img_height - half_height))
            else:
                center = (x, y)

            # Find rectangle orientation
            while True:
                angle =  np.random.randint(low = -25, high = 25)

                rect = (center, (crop_w, crop_h), angle)
                if inside_rect(rect = rect, num_cols = img_width, num_rows = img_height):
                    break

            # Perform crop:
            image_crop = crop_rotated_rectangle(image = np.array(scaled_image), rect = rect)
            mask_crop = crop_rotated_rectangle(image = np.array(scaled_mask), rect = rect)

            # If unable to crop image_crop len will be 0
            if len(image_crop) == 0 or len(mask_crop) == 0:
                i = i - 1
                continue

            # Create PIL image
            original_cropped = Image.fromarray(image_crop)
            mask_cropped = Image.fromarray(mask_crop)

            # Perform random flip with 20% probability
            if np.random.randint(0, 4) == 3:
                if np.random.randint(2) == 1:
                    original_cropped = original_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_cropped = mask_cropped.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    original_cropped = original_cropped.transpose(Image.FLIP_TOP_BOTTOM)
                    mask_cropped = mask_cropped.transpose(Image.FLIP_TOP_BOTTOM)

            # Perform random filter with 20% probability
            choice = np.random.randint(4)
            if choice == 1:
                original_cropped = original_cropped.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            choice = np.random.randint(4)
            if choice == 2:
                open_cv_image = np.array(original_cropped)
                # Convert RGB to BGR
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                invGamma = 1.0 / np.random.uniform(0.2, 2)

                table = [((i / 255) ** invGamma) * 255 for i in range(256)]
                table = np.array(table, np.uint8)

                open_cv_image = cv2.LUT(open_cv_image, table)

                # Convert back
                original_cropped = Image.fromarray(open_cv_image)

            choice = np.random.randint(4)
            if choice == 2:
                subChoice = np.random.randint(0,4)
                if subChoice == 0:
                    original_cropped = original_cropped.filter(ImageFilter.GaussianBlur(radius=np.random.randint(0,2)))
                else:
                    open_cv_image = np.array(original_cropped)
                    # Convert RGB to BGR
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                    shift = r.randrange(3,9,2)

                    if subChoice == 1:
                        open_cv_image=cv2.blur(open_cv_image,(shift,shift))
                    if subChoice == 2:
                        open_cv_image=cv2.medianBlur(open_cv_image,shift)
                    if subChoice == 3:
                        d = r.randrange(1,60,2)
                        open_cv_image=cv2.bilateralFilter(open_cv_image, d, 75,75)


                    original_cropped = Image.fromarray(open_cv_image)

            choice = np.random.randint(3)
            if choice == 2:
                enhancer = ImageEnhance.Contrast(original_cropped)
                original_cropped = enhancer.enhance(np.random.uniform(0.5,1))

            choice = np.random.randint(3)
            if choice == 2:
                enhancer = ImageEnhance.Sharpness(original_cropped)
                original_cropped = enhancer.enhance(np.random.uniform(0.5,1.5))


            choice = np.random.randint(3)
            choice = 2
            if choice == 2:
                open_cv_image = np.array(original_cropped)
                # Convert RGB to BGR
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

                # Random kernel
                shift = r.randrange(3,10,2)
                kernel = np.ones((shift, shift), np.uint8)

                subChoice = np.random.randint(0,2)
                if subChoice == 0:
                    open_cv_image = cv2.dilate(open_cv_image,kernel,iterations = 1)
                if subChoice == 1:
                    open_cv_image = cv2.erode(open_cv_image,kernel,iterations = 1)
                # Convert back
                original_cropped = Image.fromarray(open_cv_image)




            original_cropped.save(sample_dir + img_name + "_" + str(i) + ".png", subsampling=0, quality=100)
            mask_cropped.save(mask_dir + img_name + "_" + str(i) + ".png", subsampling=0, quality=100)
