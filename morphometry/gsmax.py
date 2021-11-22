import os
import PIL.ImageDraw as ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image, ImageFont
from skimage.draw import polygon
from skimage import measure
from pathlib import Path
from skimage.draw import ellipse
from skimage.transform import rotate
from collections import Counter


def gsmax(density, length, gcw1, gcw2, psg, pore_area = 0):
    # Diffusivity of water in air at 25oC
    d = 0.0282
    # Molar volume of water in air
    v = 0.02241
    # Density
    D = density
    # Length of Pore
    L = length
    # Guard cell width
    AVG_GCW = (gcw1 + gcw2) / 2
    # Depth of pore
    l = AVG_GCW / 2
    # PSG (W)
    PSG = psg
    # Area
    if pore_area == 0:
        amax_circular = 0.25 * (np.pi) * np.power(L, 2)
        amax_ellipitical = np.pi * (0.5 * L) * (0.5 * PSG)
    else:
        amax_circular = pore_area
        amax_ellipitical = pore_area
    # gsmax for circular -  conductance to water
    gs_numerator_c = (d / v) * D * amax_circular
    gs_denominator_c = l + (np.pi / 2) * np.sqrt((amax_circular/np.pi))
    circular_mmol = gs_numerator_c / gs_denominator_c
    circular_mol = circular_mmol / 1000
    # gsmax for ellpitical - conductance to water
    gs_numerator_e = (d / v) * D * amax_ellipitical
    gs_denominator_e = l + (np.pi / 2) * np.sqrt((amax_ellipitical/np.pi))
    ellipitical_mmol = gs_numerator_e / gs_denominator_e
    ellipitical_mol = ellipitical_mmol / 1000

    return circular_mol, ellipitical_mol

def morphometry(guard_value, pore_value, mask_crop):
    # Extract morphometry
    g_w = g_w1 = g_h = p_w = p_h = gch1 = gch2 = 0

    # Get center values
    m_h, m_w = mask_crop.shape

    # Get the pore
    pore = np.array(np.where(mask_crop[:,:] == pore_value))

    # Iterate over range to remove artifacts in data
    for i in range(-15, 15):
        # Guard cell measurements
        if int(m_w / 2) + i >= m_w or int(m_h / 2) + i >= m_h: continue
        guard_height = np.where(mask_crop[:, int(m_w / 2) + i] == guard_value)
        guard_width = np.where(mask_crop[int(m_h / 2)  + i,:] == guard_value)

        # Split the guard cell where pore is
        guard_h_split = split_set(guard_height[0], 2)
        guard_w_split = split_set(guard_width[0], 2)

        # Calculate guard width and height
        temp_g_w = sum(len(i) for i in guard_w_split)
        if temp_g_w > g_w:
            g_w = temp_g_w
        temp_g_h = sum(len(i) for i in guard_h_split)
        if temp_g_h > g_h:
            g_h = temp_g_h

        # Check that whole guard cell has been found
        if len(guard_h_split) != 2:
            continue

        # Get maximum value of the guard cell width
        temp_gch1 = (len(guard_h_split[0]))
        if temp_gch1 > gch1:
            gch1 = temp_gch1
        temp_gch2 = (len(guard_h_split[1]))
        if temp_gch2 > gch2:
            gch2 = temp_gch2

        # Pore measurements
        pore_height = np.where(mask_crop[:, int(m_w / 2) + i] == pore_value)
        pore_width = np.where(mask_crop[int(m_h / 2) + i,:] == pore_value)

        # Pore width and height
        temp_p_h = (len(pore_height[0]))
        if temp_p_h > p_h:
            p_h = temp_p_h
        temp_p_w = (len(pore_width[0]))
        if temp_p_w > p_w:
            p_w = temp_p_w

    # Remove any error or noise within data
    if gch2 == 0 or gch1 == 0:
        return None, None, None, None

    # Total width and height
    s_w = p_w + g_w
    s_h = p_h + gch1 + gch2

    # Output to csv file
    PL = (p_w * um_per_pixel)
    GCW_1 = (gch1 * um_per_pixel)
    GCW_2 = (gch2 * um_per_pixel)
    PSG_W = (s_h * um_per_pixel)

    return (PL, GCW_1, GCW_2, PSG_W)

def split_set(arr, space):
    indices = [i + 1 for (x, y, i) in zip(arr, arr[1:], range(len(arr))) if space < abs(x - y)]
    result = [arr[start:end] for start, end in zip([0] + indices, indices + [len(arr)])]
    return result

def contour_area(contour):
    a = 0
    prev = contour[-1]
    for pt in contour:
        #pt = pt * um_per_pixel
        a += (prev[0] * pt[1] - prev[1] * pt[0])
        prev = pt
    a *= 0.5
    return abs(a)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = 0.5 * abs(area)
    return area


def bounding_box(points, padding, w, h):
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)

    if min_x <= padding: min_x = 0
    else: min_x = min_x - padding
    if min_y <= padding: min_y = 0
    else: min_y = min_y - padding
    if max_x + padding >= w: max_x = h
    else: max_x = max_x + padding
    if max_y + padding >= w: max_y = w
    else: max_y = max_y + padding

    return [int(min_x), int(min_y), int(max_x), int(max_y)]

if __name__ == "__main__":
    # Variables
    working_dir = "E:/NewData/Wheat"
    image_dir = "/images/"
    predict_dir = "/prediction/"

    # Poplar  34, 120
    # Wheat gc 23, pore 150
    pore_value = 150
    guard_value = 23
    padding = 2
    artifact_padding = 15
    min_contour = 200
    um_per_pixel = 0.12547
    #poplar 0.181818 wheat  0.12547
    save_individual = False
    density_value = 63
    #68 wheat 277 poplar
    # Empty excel output and declare headers
    excel_output = []
    columns = "ID, Length of Pore, GCW_1, GCW_2, PSG(W), Pore Area, GC Area, gsmax circular, gsmax ellpitical"
    # Iterate through files
    files = list(Path(working_dir + predict_dir).glob('*.png'))
    for file in files:
        if "_stats" in file.name: continue
        # Image variables for averages
        avg_pore_length = avg_gcw_1 = avg_gcw_2 = avg_psg =  avg_pore_area = avg_gc_area = avg_gsmax_c = avg_gsmax_e = avg_operational_gs = 0
        # Check if mask exists for file
        file_name = str(file.stem)

        # Output to excel
        excel_output.append(file_name + ", " + columns)

        # Open images
        print(file_name)
        original_img = Image.open(file)
        mask_img = Image.open(working_dir + predict_dir + file.name).convert('L')

        # Initialise draw
        draw = ImageDraw.Draw(original_img)
        font = ImageFont.truetype("arial", 18)

        # Convert PIL images to numpy arrays
        mask_np = np.array(mask_img)

        # Size
        w, h = original_img.size

        # Find contours
        contours = measure.find_contours(np.where(mask_np == pore_value, guard_value, mask_np), level=60)

        # Select the largest contiguous contour
        contours = sorted(contours, key=lambda x: len(x), reverse=True)
        #contours = [contour for contour in contours if len(contour) >=  min_contour]

        # Variables
        id, total_count, total_pore, total_guard = 0, None, 0, 0
        # Iterate through each of the contours (stomata)
        for contour in contours:
            # If contour is too small: ignore
            if len(contour) < min_contour:
                continue
            # Bounding box and padding
            min_x, min_y, max_x, max_y = bounding_box(contour, padding, w, h)

            # Check if border, and skip
            #if min_x == 0 or min_y == 0 or max_x == w or max_y == h: continue

            # Increment counts
            id += 1

            # Crop the images
            mask_crop = mask_np[min_x:max_x,min_y:max_y]

            # Get coordinates inside poly
            xx, yy = polygon(contour[:, 0], contour[:, 1], mask_np.shape)

            # Calculate eigenvalue and vectors
            x = xx - np.mean(xx)
            y = yy - np.mean(yy)
            coords = np.vstack([x, y])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]

            # Eigenvector
            x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

            # Get angle of rotation
            theta = np.arctan((x_v1)/(y_v1))

            # Convert mask to 3 channels
            mask_multichannel = np.stack((mask_crop,)*3, axis=-1)

            # Rotate the mask and save
            mask = Image.fromarray(mask_multichannel)
            mask_aligned = mask.rotate(57.2958 * theta, expand=True, fillcolor=(255,255,255))

            # Begin counting
            unique, counts = np.unique(mask_crop, return_counts=True)
            local_counts = dict(zip(unique, counts))
            if total_count == None:
                total_count = local_counts
            else:
                total_count = Counter(total_count) + Counter(local_counts)

            # Extract morphometry
            mask_crop = np.asarray(mask_aligned)
            mask_crop = mask_crop[:,:,0]

            PL, GCW_1, GCW_2, PSG_W = morphometry(guard_value, pore_value, mask_crop)
            if PL == None:
                id -= 1
                continue

            # New area estimation for pore
            pore_crop = np.where(mask_crop == guard_value, 255, mask_crop)
            pore_crop = np.where(pore_crop == pore_value, 0, 255)
            pore_contour = measure.find_contours(pore_crop, level=1, fully_connected='high', positive_orientation='high')
            for i in range(0, len(pore_contour[0])):
                pore_contour[0][i] = pore_contour[0][i] * um_per_pixel

            '''for p in pore_contour[0]:
                pore_crop[int(p[0]), int(p[1])] = 5

            # Save mask image
            mask = Image.fromarray(pore_crop).convert('RGB')
            mask.save(working_dir + predict_dir + file_name + "_testing2.jpg", subsampling=0, quality=100)'''

            if len(pore_contour) < 1: continue
            pore_area = (contour_area(np.array(pore_contour[0])))
            # Total Area
            unique, counts = np.unique(mask_crop, return_counts=True)
            total_counts = (dict(zip(unique, counts)))

            # Individual areas
            guard_area = total_counts.get(guard_value, 0.0) * um_per_pixel
            pore_area = total_counts.get(pore_value, 0.0) * um_per_pixel
            pore_area = pore_area * um_per_pixel
            # Expand numpy dimensions
            c = np.expand_dims(pore_contour[0].astype(np.float32), 1)
            # Convert it to UMat object
            c = cv2.UMat(c)
            area2 = cv2.contourArea(c)

            # Total areas
            total_pore += pore_area
            total_guard += guard_area

            # GSMAX calculation returns [circular, ellipitical]
            gsmax_circular_mol, gsmax_ellipitical_mol = gsmax(density_value, PL, GCW_1, GCW_2, PSG_W)
            operational_gs, _ = gsmax(density_value, PL, GCW_1, GCW_2, PSG_W, pore_area)
            operational_gs2, _ = gsmax(density_value, PL, GCW_1, GCW_2, PSG_W, area2)

            # Append values to averages
            avg_pore_length += PL
            avg_gcw_1 += GCW_1
            avg_gcw_2 += GCW_2
            avg_psg += PSG_W
            avg_pore_area += pore_area
            avg_gc_area += guard_area
            avg_gsmax_c += gsmax_circular_mol
            avg_gsmax_e += gsmax_ellipitical_mol
            #avg_operational_gs += operational_gs

            # Excel output in columns: "ID, Length of Pore, GCW_1, GCW_2, PSG(W), Pore Area, GC Area, gsmax circular, gsmax ellpitical, operational_gs"
            excel_output.append("," + str(id) + ", " + str(PL) + ", " + str(GCW_1) + ", " + str(GCW_2) + ", " + str(PSG_W) + ", " \
                                + str(pore_area) + ", " + str(guard_area) + ", " + str(gsmax_circular_mol) + "," + str(gsmax_ellipitical_mol) \
                                + ", " + str(operational_gs))

            # Visualise on image
            draw.text((max_y - 180,min_x - 60), "ID: " + str(id), fill=(255, 0, 0), font=font)
            draw.text((max_y - 180,min_x - 40), "GC: " + "{:.3f}".format(GCW_1) + ", " + "{:.3f}".format(GCW_2), fill=(0, 0, 0), font=font)
            draw.text((max_y - 180,min_x - 20), "PL: " + "{:.3f}".format(PL), fill=(0, 0, 0), font=font)
            draw.text((max_y - 180,min_x), "PSG: " + "{:.3f}".format(PSG_W), fill=(0, 0, 0), font=font)

            # Save individual crop of stomata
            if save_individual:
                # Draw center line over mask
                temp = np.zeros(mask_crop.shape)
                temp = np.copy(mask_crop)
                #temp[guard_height[0], int(m_w / 2)] = 255
                #temp[int(m_h / 2),guard_width[0]] = 255

                # Save mask image
                mask = Image.fromarray(temp).convert('RGB')
                mask.save(working_dir + predict_dir + file_name + "_mask_"  + str(id) + '_sample.png', subsampling=0, quality=100)
        # Check if empty image
        if id == 0: continue
        # Excel output averages in columns: Length of Pore, GCW_1, GCW_2, PSG(W), Pore Area, GC Area
        excel_output.append(",Average, " + str(avg_pore_length / id) + ", " + str(avg_gcw_1 / id) + ", " + str(avg_gcw_2 / id) + ", " + str(avg_psg / id) + ", " \
                            + str(avg_pore_area / id) + ", " + str(avg_gc_area / id) + ", " + str(avg_gsmax_c / id) + ", " + str(avg_gsmax_e / id) \
                            + ", " + str(avg_operational_gs / id))

        # Density of Image
        density = id / ((w*um_per_pixel/1000)*(h*um_per_pixel/1000))

        # Outputs to display on original image
        draw.text((10, 10), "Stomata Count:     " + str(id), fill=(0, 0, 0), font=font)
        draw.text((10, 30), "Stomata Density:   " + str(float("{:.2f}".format(density))), fill=(0, 0, 0), font=font)

        # Save original image with contours and stats
        original_img = Image.blend(original_img, Image.fromarray(mask_np).convert('RGB'), 0.3)
        original_img.save(working_dir + predict_dir + file_name + "_stats.png", subsampling=0, quality=100)

    # Save the csv file
    with open(working_dir + predict_dir + 'morphometry.csv','w') as file:
        for l in excel_output:
            file.write(l)
            file.write('\n')
