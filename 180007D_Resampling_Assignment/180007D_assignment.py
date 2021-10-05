import numpy as np
import cv2
import math

source_image_path = './images/sample1.jpg'
bi_linear_output_image_path = './images/bi_linear.jpg'
nearest_neighbor_output_image_path = './images/nearest_neighbor.jpg'

# opening image as gray scale
source_img = cv2.imread(source_image_path,0)
source_h, source_w = source_img.shape

# Insert new height and width
new_h, new_w = [400, 600]


#bi polar interpolation method
def bipolar_interpolation_method(x, y, original_values):
    x_low = math.floor(x)
    x_high = math.floor(x) + 1
    y_low = math.floor(y)
    y_high = math.floor(y) + 1

    s1, s2, s3, s4 = original_values

    value = s1 * ((x_high - x) * (y - y_low)) + \
    s2 * (x - x_low) * (y - y_low) + \
    s3 * ((x_high - x) * (y_high - y)) + \
    s4 * ((x - x_low) * (y_high - y))

    return value


# calculate d4 distance
def calculate_d4_distance(x, y,original_values):
    x_low = math.floor(x)
    x_high = math.floor(x) + 1
    y_low = math.floor(y)
    y_high = math.floor(y) + 1

    s1, s2, s3, s4 = original_values

    I1_x, I1_y = [x_low, y_high]
    I2_x, I2_y = [x_high, y_high]
    I3_x, I3_y = [x_low, y_low]
    I4_x, I4_y = [x_high, y_low]

    d1 = abs(x-I1_x)+abs(y-I1_y)
    d2 = abs(x-I2_x)+abs(y-I2_y)
    d3 = abs(x-I3_x)+abs(y-I3_y)
    d4 = abs(x-I4_x)+abs(y-I4_y)

    min_d4_distance = min(d1, d2, d3, d4)
    if d1 == min_d4_distance:
        return s1
    elif d2 == min_d4_distance:
        return s2
    elif d3 == min_d4_distance:
        return s3
    else:
        return s4


# resample image using bipolar interpolation method
def resample_image(s_img, new_height, new_width, resampling_method):
    new_grid = np.zeros((new_height, new_width))

    h_factor = source_h / new_height
    w_factor = source_w / new_width
    for i in range(new_height):
        for j in range(new_width):
            new_x = i * h_factor
            new_y = j * w_factor

            # finding indexes of 4 pixels I1, I2, I3, I4
            # I1 - left top , I2 - right top , I3 - left bottom , I4 - right bottom
            x_low = math.floor(new_x)
            x_high = math.floor(new_x) + 1
            y_low = math.floor(new_y)
            y_high = math.floor(new_y) + 1

            I1_x, I1_y = [x_low, y_high]
            I2_x, I2_y = [x_high, y_high]
            I3_x, I3_y = [x_low, y_low]
            I4_x, I4_y = [x_high, y_low]

            # Source image pixel values of neighboring pixels
            s1 = s_img[I1_x, I1_y]
            s2 = s_img[I2_x, I2_y]
            s3 = s_img[I3_x, I3_y]
            s4 = s_img[I4_x, I4_y]

            original_values = [s1, s2, s3, s4]

            new_pixel_value = resampling_method(new_x, new_y, original_values)
            new_grid[i, j] = new_pixel_value

    return new_grid


bli_resampled_img = resample_image(source_img, new_h, new_w, bipolar_interpolation_method)
nearest_neighbor_image = resample_image(source_img, new_h, new_w, calculate_d4_distance)

# writing nearest neighbor resampled image
cv2.imwrite(bi_linear_output_image_path, bli_resampled_img)
cv2.imwrite(nearest_neighbor_output_image_path, nearest_neighbor_image)


