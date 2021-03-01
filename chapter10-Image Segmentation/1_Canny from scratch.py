import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread('chapter10-Image Segmentation/images/Fig1025(a)(building_original).tif',0)

def Canny(img,sigma,ksize,tl,th):
    # 1 smooth image with a Guassian filter 
    smoothed_img = cv2.GaussianBlur(img,(ksize,ksize),sigma)

    # 2 compute gradient magnitude and angle images 
    gx = cv2.Sobel(smoothed_img,cv2.CV_32F,0,1,ksize=3)
    gy = cv2.Sobel(smoothed_img,cv2.CV_32F,1,0,ksize=3)
    angle = np.arctan2(gy,gx) * 180./np.pi  # it's in range (-180,180)
    angle[angle<0] += 180  # now it's in range (0,180)
    M = np.sqrt(gx**2+gy**2)
    
    # 3 apply nonmaxima suppression
    m,n = img.shape
    thinner_M = np.zeros_like(M)
    for i in range(1,m-1):
        for j in range(1,n-1):
            teta = angle[i,j]
            if (157.5<=teta or teta<22.5) and M[i-1,j]<=M[i,j] and M[i+1,j]<=M[i,j] : # 0 angle
                thinner_M[i,j] = M[i,j]
            elif 22.5<=teta<67.5 and M[i+1,j+1]<=M[i,j] and M[i-1,j-1]<=M[i,j] : # 45
                thinner_M[i,j] = M[i,j]
            elif 67.5<=teta<112.5 and M[i,j-1]<=M[i,j] and M[i,j+1]<=M[i,j] : # 90
                thinner_M[i,j] = M[i,j]
            elif 112.5<=teta<157.5 and M[i-1,j+1]<=M[i,j] and M[i+1,j-1]<=M[i,j] : #180
                thinner_M[i,j] = M[i,j]
    
    # 4 thresholding
    Tl = thinner_M.max() * tl
    Th = thinner_M.max() * th
    thinner_M[thinner_M<Tl] = 0
    thinner_M[thinner_M>=Th] = 255
    for i in range(1,m-1):
        for j in range(1,n-1):
            ker = thinner_M[i-1:i+2,j-1:j+2]
            if (ker>Th).any() and thinner_M[i,j]!=0 :
                thinner_M[i,j] = 255
            else:
                thinner_M[i,j] = 0       

    return thinner_M

out1 = Canny(img,4,25,0.04,0.15)
plt.imshow(out1,cmap='gray')
plt.show()