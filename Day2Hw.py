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
    interval = 0.01
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

img = cv.imread('lena512.pgm', cv.IMREAD_GRAYSCALE)

arr = np.array([numintegrate(-1.5,-0.5), numintegrate(-0.5,0.5), numintegrate(0.5,1.5)])
arr = np.reshape(arr, (1,3))
arrT = arr.T
print('arrT', arrT)
print('arr', arr)
kernel = outer(arr,arrT)
normalized_kernel = normalize(kernel)
print('kernel', normalized_kernel)





