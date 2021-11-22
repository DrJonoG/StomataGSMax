from pathlib import Path
from skimage import measure
import numpy as np
from PIL import Image

def bounding_box(points):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    return [int(min_x), int(min_y), int(max_x), int(max_y)]


if __name__ == "__main__":

    working_dir = "D:/02.Datasets/Stomata/00.Dataset/Poplar/Annotated/"

    files = list(Path(working_dir).glob('*.png'))

    um_per_pixel = 0.181818
    # Poplar  34, 120
    # Wheat 23, 150
    pore_value = 120
    guard_value = 34

    total_stomata = 0
    total_area = 0

    file_count = len(files)
    current_file = 0
    for file in files:
        # Check if mask exists for file
        file_name = str(file.stem)
        if "_mask" not in file_name: continue

        # Load image and convert to numpy
        mask_img = Image.open(file).convert('L')
        w,h = mask_img.size
        mask_np = np.array(mask_img)
        # convert to white
        mask_np = np.where(mask_np == 0, 255, mask_np)
        # Make whole blobs
        mask_np = np.where(mask_np == pore_value, guard_value, mask_np)
        # Find contours
        contours = measure.find_contours(mask_np, level=80)
        # Count contour
        countour_count = 0
        for cont in contours:
            if len(cont) > 50:
                # Bounding box and padding
                min_x, min_y, max_x, max_y = bounding_box(cont)
                # Check if border, and skip
                if min_x == 0 or min_y == 0: continue
                countour_count += 1


        #print(file_name + "; area: " + str(w*h) + " stomata = " + str(countour_count))
        print('Processing images [%d / %d]\r'%(current_file, file_count), end="")
        # Update variables
        total_area += ((w*um_per_pixel/1000)*(h*um_per_pixel/1000))
        total_stomata += countour_count

        current_file += 1

    # Calculate density
    density = total_stomata / total_area
    # Display results
    print("Area: %.2f, Stomata count: %d, Density:%d"%(total_area, total_stomata, density))
