import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import math

# Print iterations progress
def progress(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def saveloss(path, train_loss, valid_loss):
    plt.figure(figsize=(10,8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.savefig(path + '/loss.png')

def save_prediction(pred, file, directory_out, annotation, labels):
    unique = np.unique(pred)
    # Get original size and reshape
    img1 = Image.open(file)
    w, h = img1.size
    pred = pred[0:h, 0:w]
    if annotation != "True":
        for i in range(len(labels)):
            pred = np.where(pred == i, labels[i], pred)
        pred = np.where(pred==0, 255, pred)

        pred = np.stack((pred,)*3, axis=-1)
        pred_img = Image.fromarray((pred).astype(np.uint8))
        pred_img.save(directory_out + str(file.stem) + "_mask.png", subsampling=0, quality=100)

        blended = Image.blend(img1, pred_img, alpha=0.6)
        blended.save(directory_out + str(file.stem) + "_blend.jpg", subsampling=0, quality=100)
    else:
        for i in range(len(self.labels)):
            pred = np.where(pred == i, self.labels[i], pred)
        pred = np.where(pred==255, 0, pred)

        pred = np.stack((pred,)*3, axis=-1)
        pred_img = Image.fromarray((pred).astype(np.uint8))
        pred_img.save(directory_out + str(file.stem) + "_mask.png", subsampling=0, quality=100)

        pred.fill(0)
        pred_img = Image.fromarray((pred).astype(np.uint8))
        pred_img.save(directory_out + str(file.stem) + "_color_mask.png", subsampling=0, quality=100)

        pred_img = Image.fromarray((pred).astype(np.uint8))
        pred_img.save(directory_out + str(file.stem) + "_watershed_mask.png", subsampling=0, quality=100)

        img1.save(directory_out + str(file.stem) + ".jpg", subsampling=0, quality=100)

def is_power_of_2_i(n):
    return (n & (n - 1)) == 0

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def open_image(file_path):
    accepted = list([64,128,256,512,756,1024,2048,3072,4096])

    raw_rgb = Image.open(file_path).convert('L')
    width, height = raw_rgb.size

    margin = 0
    new_width = math.ceil(width / 256) * 256
    raw_rgb = add_margin(raw_rgb, 0, new_width-width, 0, 0, margin)
    new_height = math.ceil(height / 256) * 256
    raw_rgb = add_margin(raw_rgb, 0, 0, new_height-height, 0, margin)

    #raw_rgb = raw_rgb.convert('L')
    img_nd = np.array(raw_rgb)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans
