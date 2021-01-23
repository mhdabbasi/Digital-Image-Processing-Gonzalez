import numpy as np
import cv2
import matplotlib.pyplot as plt 

#### Sharpening using Laplacian(second-derivative)
img = cv2.imread('chapter3/images/blurry_moon.tif',0) 
kernel = np.array([[0, 1,0],
                   [1,-4,1],
                   [0, 1,0]])

Laplacian_img = cv2.filter2D(img,-1,kernel) # this method convolve kernel on img exactly
                                            # like i implement in prev codes .And it use reflect for padding img.
sharped_img = img+(-1)*Laplacian_img

plt.imshow(sharped_img,cmap='gray')
plt.show()

#### Sharpening using masking
img = cv2.imread('chapter3/images/dipxe_text.tif',0)

blurred = cv2.GaussianBlur(img,(31,31),5)
mask = img - blurred
sharped_img = img + 0.05*mask     # I choose k=0.05 becuse my image is smaller than image in book .

plt.imshow(sharped_img,cmap='gray')
plt.show()