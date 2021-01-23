import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chapter3/images/h_e.tif',0)

his = cv2.calcHist([img],[0],None,[256],[0,256])
cdf = his.cumsum()
mn = img.shape[0]*img.shape[1]

new = cdf[img]*(255/mn)
new = np.around(new)
new_his = cv2.calcHist([new],[0],None,[256],[0,256])

#dst = cv2.equalizeHist(img) all i do in prev lines can be done with this method of opencv.(dst == new)
                            # I just do that to show how it works.

plt.imshow(new)
plt.show()

plt.plot(new_his,label='new')
plt.plot(his,'r',label='old')
plt.legend()
plt.show()
