import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('chapter3/images/test_pattern_blurring.tif',0)

def lp_filter(img,ker_size) :
    '''
    ker_size should be odd number'''
    kernel = np.ones((ker_size,ker_size)) / (ker_size*ker_size)
    n = ker_size//2
    img_shape = img.shape
    new_img = np.zeros_like(img,dtype=int)
    img = np.pad(img,((n,n),(n,n)))
    for i in range(new_img.shape[0]) :
        for j in range(new_img.shape[1]) :
            new_img[i,j] = np.sum(img[i:i+ker_size,j:j+ker_size]*kernel)
    return new_img

#blured2 = cv2.blur(img,(11,11))  -> lp_filter function is in cv2 with name of blur .

blured = lp_filter(img,11)
plt.imshow(blured,cmap='gray')
plt.show()