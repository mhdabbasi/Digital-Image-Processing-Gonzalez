import numpy as np 
import cv2
import matplotlib.pyplot as plt 

##### MEAN FILTERS #######
# Geometric works better than arithmetic to reduce noise .
# and contraharmonic is good for salt and pepper noise.  
# Q<0 is well for selt noise and Q>0 is good for pepper noise .
gaus_img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0507(b)(ckt-board-gauss-var-400).tif',0)
ksize = 11
#1-Arithmetic mean filter
arithmean = cv2.blur(gaus_img,ksize)

#2-Geometric mean filter
# np.square(np.sqrt(x)) is equal to np.exp(np.log(x)) .
geomean = np.uint8(np.exp(cv2.boxFilter(np.log(gaus_img), -1, (ksize, ksize))))

#3-Contraharmonic mean filter
Q = 1.5
contrmean = cv2.boxFilter(gaus_img**(Q+1)) / cv2.boxFilter(gaus_img**Q)

##### ORDER-STATISTIC FILTERS #######
# these filters are so easy to implement from scratch .so, i use builtin methods.

#1-Median filter
# it makes less blurring than linear smoothing filters of similar ksize .
# it works well for salt and pepper noise , if Ps and Pp be less than 0.2 .
SP_img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0510(a)(ckt-board-saltpep-prob.pt05).tif',0)
median = cv2.medianBlur(SP_img,ksize)

#2-Midpoint Filter
# It works best for randomly distributed noise, like Gaussian or uniform noise.
from scipy import ndimage
min_img = ndimage.minimum_filter(gaus_img,ksize)
max_img = ndimage.maximum_filter(gaus_img,ksize)
midimg = 0.5*(max_img+min_img)

#3-Alpha-Trimmed Mean Filter
# the alpha-trimmed filter is useful in situations involving multiple
# types of noise, such as a combination of salt-and-pepper and Gaussian noise.
uni_img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0512(a)(ckt-uniform-var-800).tif',0)
d = 6
m,n = uni_img.shape
alphaimg = np.zeros_like(uni_img)
padded_img = np.pad(uni_img,(ksize//2,ksize//2),mode='reflect')
for i in range(m):
    for j in range(n):
        neighbours = padded_img[i:i+ksize,j:j+ksize].flatten()
        neighbours.sort()
        neighbours = neighbours[d:-d]
        alphaimg[i,j] = np.sum(neighbours) / (m*n-d)

#### ADAPTIVE FILTERS #########
# this types of filters is so better than mean filters .
# it better save the edges .

#1-Adaptive, Local Noise Reduction Filter
gaus_img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0513(a)(ckt_gaussian_var_1000_mean_0).tif',0)
noise_var = 0.25
def compute_ALimg(img,ksize,noise_var):
    m,n = img.shape
    ALimg = np.zeros_like(img)
    padded_img = np.pad(img,(ksize//2,ksize//2),mode='reflect')
    for i in range(m):
        for j in range(n):
            neighbours = padded_img[i:i+ksize,j:j+ksize]
            local_var = np.var(neighbours)
            local_avr = np.mean(neighbours)
            if noise_var > local_var :
                ALimg[i,j] = local_avr
            else:
                ALimg[i,j] = neighbours - (noise_var/local_var)*(neighbours-local_avr)
    return ALimg
    
ALimg = compute_ALimg(gaus_img,ksize,noise_var)

#2-Adaptive Median Filter
# I sead Median filter is well for salt-and-pepper for Ps and Pp less than 0.2 .
# but this filter can handle salt-and-pepper with more than 0.2 well too .
SP_img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0514(a)(ckt_saltpep_prob_pt25).tif',0)
def compute_AMimg(img,Smax):
    m,n = img.shape
    AMimg = np.zeros_like(img)
    h = Smax//2
    padded_img = np.pad(img,(h,h),mode='reflect')
    for i in range(m):
        for j in range(n):
            k = 3    # k is kernel_size//2 .It means i start from kernel with size 7x7 .
            neighbours = padded_img[i+h-k:i+h+k,j+h-k:j+h+k]
            while True:
                if np.min(neighbours)<np.median(neighbours) and np.median(neighbours)<np.max(neighbours):
                    if np.min(neighbours)<img[i,j] and img[i,j]<np.max(neighbours):
                        AMimg[i,j] = img[i,j]
                    else:
                        AMimg[i,j] = np.median(neighbours)
                    break
                else:
                    k += 1
                    Snew = k*2+1
                    if Snew <= Smax :
                        neighbours = padded_img[i+h-k:i+h+k,j+h-k:j+h+k]
                    else :
                        AMimg[i,j] = np.median(neighbours)
                        break
    return AMimg
Smax=9
AMimg = compute_AMimg(SP_img,Smax)


plt.imshow(AMimg,cmap='gray')
plt.show()