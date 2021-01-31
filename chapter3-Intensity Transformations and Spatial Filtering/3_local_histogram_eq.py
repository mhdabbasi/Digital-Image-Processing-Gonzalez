import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chapter3-Intensity Transformations and Spatial Filtering/images/lhe.tif',0)
def his_eq(ker):
    his = cv2.calcHist([ker],[0],None,[256],[0,256])
    cdf = his.cumsum()
    m,n = ker.shape[0],ker.shape[1]
    new = cdf[ker]*(255/(m*n))
    new = np.around(new)
    center = new[(m//2)+1,(n//2)+1]
    return center

def local_his_eq(img,ker_size):
    '''
    ker_size should be odd number'''
    img_shape = img.shape
    new_img = np.zeros_like(img)
    img = np.pad(img,((ker_size//2,ker_size//2),(ker_size//2,ker_size//2)))
    for i in range(new_img.shape[0]) :
        for j in range(new_img.shape[1]) :
            new_img[i,j] = his_eq(img[i:i+ker_size,j:j+ker_size])
    return new_img

new_img = local_his_eq(img,5)
plt.imshow(new_img,cmap='gray')
plt.show()
