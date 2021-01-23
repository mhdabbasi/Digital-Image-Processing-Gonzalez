import numpy as np
import cv2 
import matplotlib.pyplot as plt 

img = cv2.imread('chapter3/images/test_pattern_blurring.tif',0)

def Guassian(img,ker_size,k):
    # Comput kernel
    var = ker_size/6
    kernel = np.zeros((ker_size,ker_size))
    center = (ker_size//2)+1
    for s in range(ker_size):
        for t in range(ker_size):
            r2 = (center-s)**2+(center-t)**2
            kernel[s,t] = k*np.e**(-r2/(2*var**2))
    kernel = kernel/np.sum(kernel)

    # convolve kernel on img
    n = ker_size//2
    img_shape = img.shape
    new_img = np.zeros_like(img)
    img = np.pad(img,((n,n),(n,n)),'reflect')
    for i in range(new_img.shape[0]) :
        for j in range(new_img.shape[1]) :
            new_img[i,j] = np.sum(img[i:i+ker_size,j:j+ker_size]*kernel)
    return new_img

#new2 = cv2.GaussianBlur(img,(43,43),43/6) this methos in cv2 is exactly like my func.

new = Guassian(img,43,1)

plt.imshow(new,cmap='gray')
plt.show()