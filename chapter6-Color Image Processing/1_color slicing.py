import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

img = cv2.imread('chapter6-Color Image Processing/images/Fig0631(a)(strawberries_coffee_full_color).tif',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def color_slice(img,radius,cordinates):
    sliced = np.copy(img)
    mask = np.sum((sliced-cordinates)**2,axis=2)
    mask = mask>radius**2
    mask = np.repeat(mask,[3],axis=1).reshape(mask.shape[0],mask.shape[1],3)
    sliced = np.where(mask,np.array([0.5,0.5,0.5]),sliced)
    return sliced

img_max = np.max(img)
img = img.astype('float32')/img_max
sliced = color_slice(img,0.1765,[0.6863,0.1608,0.1922])
sliced = np.uint8(sliced*img_max)
plt.imshow(sliced)
plt.show()