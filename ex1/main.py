import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


"""
To make it easier to check the assignment, change the paths for the folder where 
the images are stored and for the folder where you want to export the results. 
No change is required in other places.
"""
IMAGE_PATH = r"C:\Users\05ita\PycharmProjects\dip_ex\ex1\images"
EXPORT_PATH = r"C:\Users\05ita\PycharmProjects\dip_ex\ex1\images\gen"

# Part A
# Q1
"""
    section a - The answer to this section contains the function bi_linear_interpolation. 
    The function receives a two-dimensional array and returns a two-dimensional array
    representing an interpolation in factor 2 of the image
"""
def bi_linear_interpolation(image_in, factor = 2):
    n, m = image_in.shape
    image_out = np.zeros((2*n-1, 2*m-1), dtype=image_in.dtype)
    n_new, m_new = image_out.shape
    image_out[0:n_new:2, 0:m_new:2] = image_in[0:n, 0:m]
    for i in range(n):
        for j in range(m-1):
            image_out[2*i, 2*j+1] = 0.5*image_in[i, j] + 0.5*image_in[i, j+1]
    for i in range(n-1):
        for j in range(m_new):
            image_out[2*i+1, j] = 0.5*image_out[2*i, j] + 0.5*image_out[2*(i+1), j]
    if factor == 2:
        return image_out
    else:
        return bi_linear_interpolation(image_out, factor/2)

# Q2
# section a - Calculation of the histogram
def histogram(image_in):
    histogram_out = np.zeros(256, dtype=int)
    image_in = image_in.flatten()
    for pixel in image_in:
        histogram_out[pixel] += 1

    return histogram_out

# section b - Calculation of values in contrast stretching
def contrast(image_in):
    n, m = image_in.shape
    image_out = np.zeros((n, m), dtype=image_in.dtype)
    fmin, fmax = np.min(image_in), np.max(image_in)
    for i in range(n):
        for j in range(m):
            image_out[i, j] = 255*(image_in[i, j]-fmin)/(fmax-fmin)

    return image_out

# section c - Calculation the Histogram Equalization
def histogram_equalization(image_in):
    histogram_in = histogram(image_in)
    s_k = np.zeros(256, dtype=int)
    cdf = histogram_in.cumsum()
    for i in range(256):
        s_k[i] = 255*cdf[i]/cdf[255]

    histogram_out = np.zeros(256)
    for i in range(256):
        histogram_out[s_k[i]] = histogram_out[s_k[i]] + histogram_in[i]

    # linear interpolation to regenerate the image
    n, m = image_in.shape
    scale = np.arange(256)
    cdf_normalized = cdf * (255 / cdf[-1])

    image_out = np.interp(image_in.flatten(), scale, cdf_normalized)
    image_out = image_out.reshape(n, m)
    image_out = image_out.astype(np.uint8)

    return histogram_out, image_out

# Part B
# Q1
""""
    section a - Implementation of a function d2conv that receives a two-dimensional array representing
    the image values and the kernel. 
    The function calculates the convolution between the inputs. 
    The function uses auxiliary functions expand_with_zeros and single_sum to present 
    the calculation in a clear way
e"""
def d2conv(grayscale_image, kernel):
    image_in = np.array(grayscale_image)
    k, k_extra = kernel.shape
    padding_factor = int((k-1)/2)
    expanded_image = expand_with_zeros(image_in, padding_factor)

    n, m = image_in.shape
    image_out = np.zeros((n, m), dtype=image_in.dtype)
    for i in range(n):
        for j in range(m):
            image_out[i, j] = single_sum(expanded_image[i:(i+k), j:(j+k)], kernel, k)

    conv_image = Image.fromarray(image_out, mode='L')

    return conv_image

def single_sum(slot, kernel, k):
    sum = 0
    for i in range(k):
        for j in range(k):
            sum += slot[i, j]*kernel[i, j]
    return sum

def expand_with_zeros(array, padding_factor):
    n, m = array.shape
    expanded_array = np.zeros((n + 2*padding_factor, m + 2*padding_factor), dtype=array.dtype)
    expanded_array[padding_factor:-padding_factor, padding_factor:-padding_factor] = array
    return expanded_array

if __name__ == '__main__':

    # Part A
    # Q1
    # a - Creating a function Bi-Linear Interpolation (above)
    # b - Loading an image and using the function we created
    image_name = "peppers.jpg"
    full_path = os.path.join(IMAGE_PATH, image_name)

    image_in = Image.open(full_path)
    grayscale_image = image_in.convert('L')
    image_array = np.array(grayscale_image)

    '''
    image_name = "peppers_p.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    grayscale_image = Image.fromarray(image_array[50:75, 50:75], mode='L')
    grayscale_image.save(full_path)
    #grayscale_image.show()
    '''

    image_name = "peppers_X2.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    grayscale_image = Image.fromarray(bi_linear_interpolation(image_array), mode='L')
    grayscale_image.save(full_path)
    grayscale_image.show()

    '''
    image_name = "peppers_X2_p.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    grayscale_image = Image.fromarray(bi_linear_interpolation(image_array)[100:150, 100:150], mode='L')
    grayscale_image.save(full_path)
    #grayscale_image.show()
    '''

    # c - Using the function we created with 8x magnification
    image_name = "peppers_X8.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    grayscale_image_X8 = Image.fromarray(bi_linear_interpolation(image_array, 8), mode='L')
    grayscale_image_X8.save(full_path)
    grayscale_image_X8.show()

    '''
    image_name = "peppers_X8_p.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    grayscale_image_X8 = Image.fromarray(bi_linear_interpolation(image_array, 8)[400:600, 400:600], mode='L')
    grayscale_image_X8.save(full_path)
    #grayscale_image_X8.show()
    '''

    # Q2
    # a - Histogram display
    image_name = "leafs.jpg"
    full_path = os.path.join(IMAGE_PATH, image_name)
    image_in = Image.open(full_path)
    grayscale_image = image_in.convert('L')
    image_array = np.array(grayscale_image)

    scale = np.arange(256)
    plt.figure()
    plt.xlabel('Cell Value')
    plt.ylabel('Number of shows')
    plt.title('Histogram of leafs.jpg')
    plt.bar(scale, histogram(image_array))
    plt.grid(True)
    plt.show()

    # b - Contrast stretch
    contrast_image = contrast(image_array)

    plt.figure()
    plt.xlabel('Cell Value')
    plt.ylabel('Number of shows')
    plt.title('Contrast Histogram of leafs.jpg')
    plt.bar(scale, histogram(contrast_image))
    plt.grid(True)
    plt.show()

    image_name = "leafs_contrast.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    contrast_image = Image.fromarray(contrast_image, mode='L')
    contrast_image.save(full_path)

    # c - Histogram Equalization
    image_name = "leafs_HE.jpg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    equalization_his, equalization_image = histogram_equalization(image_array)
    equalization_image = Image.fromarray(equalization_image, mode='L')
    equalization_image.save(full_path)

    plt.figure()
    plt.xlabel('Cell Value')
    plt.ylabel('Number of shows')
    plt.title('Histogram equalization of leafs.jpg')
    plt.bar(scale, equalization_his)
    plt.grid(True)
    plt.show()

    # Part B
    # Q1
    # a - Creating a function d2conv (above)
    image_name = "zebra.jpeg"
    full_path = os.path.join(IMAGE_PATH, image_name)
    image_in = Image.open(full_path)
    grayscale_image = image_in.convert('L')

    kernel_3 = np.array([[0.1, 0.1, 0.1],
                       [0.1, 0.2, 0.1],
                       [0.1, 0.1, 0.1]])

    kernel_5 = np.zeros((5, 5))
    n, m = kernel_5.shape
    for i in range(n):
        for j in range(m):
            if i == 0 or i == 4 or j == 0 or j == 4:
                kernel_5[i, j] = 0.035
            elif i == 1 or i == 3 or j == 1 or j == 3:
                kernel_5[i, j] = 0.048
            else:
                kernel_5[i, j] = 0.056
    # print(kernel_5)
    # print(sum(sum(kernel_5)))

    kernel_9 = np.zeros((9, 9))
    n, m = kernel_9.shape
    for i in range(n):
        for j in range(m):
            if i == 0 or i == 8 or j == 0 or j == 8:
                kernel_9[i, j] = 0.005
            elif i == 1 or i == 7 or j == 1 or j == 7:
                kernel_9[i, j] = 0.01
            elif i == 2 or i == 6 or j == 2 or j == 6:
                kernel_9[i, j] = 0.02
            elif i == 3 or i == 5 or j == 3 or j == 5:
                kernel_9[i, j] = 0.03
            else:
                kernel_9[i, j] = 0.04
    # print(kernel_9)
    # print(sum(sum(kernel_9)))

    image_name = "zebra_blurr_k=3.jpeg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    blurr_image = d2conv(grayscale_image, kernel_3)
    blurr_image.save(full_path)

    image_name = "zebra_blurr_k=5.jpeg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    blurr_image = d2conv(grayscale_image, kernel_5)
    blurr_image.save(full_path)

    image_name = "zebra_blurr_k=9.jpeg"
    full_path = os.path.join(EXPORT_PATH, image_name)
    blurr_image = d2conv(grayscale_image, kernel_9)
    blurr_image.save(full_path)
