from PIL.Image import new
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math 

"""
- Numerically integrate with steps of 0.01
- integrate from -1.5 to 0.5
- -0.5 to 0.5
- 0.5 to 1.5
- put those areas in a 1 dimensional matrix
- then create a 3x3 matrix with the transpose of the first matrix (dot product)
    - np.outer
- Divide each value in each cell by the sum of all the cells (for instance if the sum of all 9 is 0.95 divide by 0.95)
- Use the convolution function to overlay the matrix onto the image
- Figure out how ur doing edge control
- use numpy to display
"""
def gaussx(x, mu, sigma):
    coefficient = 1/(sigma*(math.pi*2)**0.5)
    exp = -0.5*(x-mu)**2/sigma**2
    gauss = coefficient*math.exp(exp)
    return gauss

def numintegrate(a,b):
    pointer = a
    interval = 0.00001
    sum = 0
    while pointer <= b:
        gauss = gaussx(pointer,0.0,1.0)
        sum += gauss 
        pointer += interval 
    return sum

def normalize(arr):
    sum = arr.sum()
    new_arr = np.true_divide(arr, sum)
    return new_arr
def outer(arr, arrT):
    combined = np.zeros((arr.shape[1], arrT.shape[0]))
    print(combined.shape)
    for i in range(0,combined.shape[0]):
        for j in range(0,combined.shape[1]):
            combined[i,j] = arrT[i,0] * arr[0,j]
    return combined

def rescale(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    print('max: ', max)
    scaling_value = 255/(max-min)
    new_arr = (arr-min) * scaling_value
    return new_arr

def convolve(img, kernel):
    img_x = img.shape[1]
    img_y = img.shape[0]
	
    kernel_x = kernel.shape[1]
    kernel_y = kernel.shape[0]

    img_border = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    return_matrix = np.zeros((img_y, img_x), dtype="float32")
    for y in np.arange(1, img_y + 1):
        for x in np.arange(1, img_x + 1):
            pointer_region = img_border[y - 1:y + 2, x - 1:x + 2]
            product = pointer_region * kernel
            return_matrix[y - 1, x - 1] = product.sum()
    return_matrix = rescale(return_matrix)
    return return_matrix

img = cv.imread('/Users/MonicaChan/Desktop/AT/CV unit/Day2/lena512.png', cv.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()

arr = np.array([numintegrate(-1.5,-0.5), numintegrate(-0.5,0.5), numintegrate(0.5,1.5)])
arr = np.reshape(arr, (1,3))
arrT = arr.T
kernel = np.outer(arr,arrT)
normalized_kernel = normalize(kernel)

new_img = convolve(img, normalized_kernel)
plt.imshow(new_img)
plt.show()

#Testing against open cv's gaussian blur function
cv_blur = cv.GaussianBlur(img, (3,3), 1)
print(cv_blur-new_img)
plt.imshow(cv_blur)
plt.show()
