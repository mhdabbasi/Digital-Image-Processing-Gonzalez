import numpy as np
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0438(a)(bld_600by600).tif',0)
img = np.pad(img,(1,1)) 

h = np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]])

# compute padded h
hp = np.pad(h,(((602-3)//2+1,(602-3)//2) ,((602-3)//2+1,(602-3)//2)))

# compute dft of h .
H = np.fft.fft2(hp)
H = np.fft.fftshift(H)
H.real = 0

# compute dft of img
img_dft = np.fft.fft2(img)
img_dft = np.fft.fftshift(img_dft)

# compute filtering in frequency domein .
img_dftxH = img_dft * H


# compute idft of img_dftxH
result1 = np.fft.ifft2(img_dftxH)
result1 = np.fft.ifftshift(result1)
result1 = np.abs(result1)

# convolve filter h on img in spatial domain .
result2 = cv2.filter2D(img,-1,h)


# show both results
f,ax = plt.subplots(1,2)
ax[0].imshow(result1,cmap='gray')
ax[1].imshow(result2,cmap='gray')
plt.show()