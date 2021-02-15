import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread('chapter5-Image Restoration and Reconstruction/images/Fig0503(original_pattern).tif',0)
m,n = img.shape[:2]

# add Gaussian noise to img
mean = 10
std = 25
noise = np.random.normal(mean,std,(m,n))
Gau_noisy_img = img + noise

# add Rayleigh noise to img
std = 40
noise = np.random.rayleigh(scale=std,size=(m,n))
Ray_noisy_img = img + noise

# add Gamma noise to img
shape = 2.0
std = 18
noise = np.random.gamma(shape,std,(m,n))
Gam_noisy_img = img + noise

# add Exponential noise to img
std = 26
noise = np.random.exponential(std,(m,n))
Exp_noisy_img = img + noise

# add Uniform noise to img
a,b = 10,100
noise = np.random.uniform(a,b,(m,n))
Uni_noisy_img = img + noise

# add salt and pepper noise to img
number_black = int(m*n*0.05)
number_white = int(m*n*0.05)

m_blacks = np.random.randint(0,m,number_black)
n_blacks = np.random.randint(0,n,number_black)
m_whites = np.random.randint(0,m,number_white)
n_whites = np.random.randint(0,n,number_white)

SP_noisy_img = np.copy(img)
SP_noisy_img[m_blacks,n_blacks] = 0
SP_noisy_img[m_whites,n_whites] = 255

# show noisy img and its histogram
f,ax = plt.subplots(2,1,figsize=(4,8))
ax[0].imshow(SP_noisy_img,cmap='gray',aspect='auto')
ax[1].hist(SP_noisy_img.flatten(),bins=256)
plt.show()