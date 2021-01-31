import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('chapter3-Intensity Transformations and Spatial Filtering/images/lhe.tif',0)
  
def compute_LE(img,ker_size,k0,k1,k2,k3,C) :
    '''
    ker_size should be odd number '''
    Mg = np.mean(img)
    STDg = np.std(img)
    img_shape = img.shape
    new_img = np.zeros_like(img)
    img = np.pad(img,((ker_size//2,ker_size//2),(ker_size//2,ker_size//2)))
    for i in range(new_img.shape[0]) :
        for j in range(new_img.shape[1]) :
            ker = img[i:i+ker_size,j:j+ker_size]
            Mxy = np.mean(ker)
            STDxy = np.std(ker)
            if (k0*Mg<=Mxy and Mxy<=k1*Mg) and (k2*STDg<=STDxy and STDxy<=k3*STDg) :
                new_img[i,j] = C * img[i,j]
            else :
                new_img[i,j] = img[i,j]
    return new_img

new_img = compute_LE(img,ker_size=3,k0=0,k1=0.1,k2=0,k3=0.1,C=22.8)
plt.imshow(new_img,cmap='gray')
plt.show()
