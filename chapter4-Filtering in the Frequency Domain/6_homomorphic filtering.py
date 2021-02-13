import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0462(a)(PET_image).tif',0)

def homomorphic(img,D0,yh,yl,c):
    #0_pad img
    padded_img = np.pad(img,((0,img.shape[0]),(0,img.shape[1])))

    #1_add one to image for ln(0) and then compute ln(img)
    padded_img = padded_img.astype(np.float32)+1
    z = np.log(padded_img)
    

    #2_compute dft of z
    Z = np.fft.fftshift(np.fft.fft2(z))
    
    #3_compute Z*H
    P,Q = Z.shape[:2]
    D = np.zeros_like(Z,dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            D[i,j] = np.sqrt((i-P/2)**2 + (j-Q/2)**2)
    H = (yh-yl)*(1-np.exp(-c*(D/D0)**2)) + yl

    filtered = Z*H

    #4_compute idft result
    shifted = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(shifted)
    result = result[:img.shape[0],:img.shape[1]]

    #5_compute exp and subtract one that i added it at first .
    exp_result = np.exp(result) - 1
    exp_result = np.abs(exp_result)

    return exp_result


result = homomorphic(img,D0=25,yh=1.0,yl=0.4,c=5)

plt.imshow(result,cmap='gray')
plt.show()