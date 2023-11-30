import os
from sklearn.model_selection import train_test_split
import shutil

def splitData(path):
    masks = path + "\\train_mask\\"
    samples = path + "\\train_samples\\"

    valid_mask = path + "\\valid_mask\\"
    valid_samples = path + "\\valid_samples\\"

    if not os.path.exists(valid_mask):
        os.makedirs(valid_mask)
    if not os.path.exists(valid_samples):
        os.makedirs(valid_samples)


    maskList = os.listdir(masks)

    train, test  = train_test_split(maskList, test_size=0.1)

    for file_name in test:
        shutil.move(os.path.join(masks, file_name), valid_mask + file_name)
        shutil.move(os.path.join(samples, file_name), valid_samples + file_name)
