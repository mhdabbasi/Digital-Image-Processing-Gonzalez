import cv2
import numpy as np 
import matplotlib.pyplot as plt 

# Because my img has smaller size than 77x548 so my results isn't exactly like results in book .

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0419(a)(barbara).tif',0)

# reduce img to 33% of its orginal size using row/column deletion .
reduced_img = np.zeros((img.shape[0]//3,img.shape[1]//3))

for i in range(reduced_img.shape[0]):
    for j in range(reduced_img.shape[1]):
        reduced_img[i,j] = img[i*3,j*3]

    
# back to its original size by pixel replication .
same_colsize_img = np.repeat(reduced_img,repeats=3,axis=1)
samesize_img = np.repeat(same_colsize_img,repeats=3,axis=0)

# show new img .
plt.imshow(samesize_img,cmap='gray',vmin=0,vmax=255)
plt.show()