import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0441(a)(characters_test_pattern).tif',0)
D0 = 40
n = 2.25

def DFT(img):
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    return dft

def IDFT(dft):
    idft = np.fft.ifft2(dft)
    return np.abs(idft)

# IDEAL LOWPASS FILTERS
def ILBF(img,D0):
    padded_img = np.pad(img,((0,img.shape[0]),(0,img.shape[1])),mode='reflect')
    D = np.zeros_like(padded_img,dtype=np.float32)
    P,Q = D.shape[:2]
    for u in range(P):
        for v in range(Q):
            D[u,v] = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
    H = np.float32(D <= D0)
    result = IDFT(DFT(padded_img) * H)
    result = result[:img.shape[0],:img.shape[1]]
    return result
    

# GAUSSIAN LOWPASS FILTERS
def GLBF(img,D0):
    padded_img = np.pad(img,((0,img.shape[0]),(0,img.shape[1])),mode='reflect')
    D = np.zeros_like(padded_img,dtype=np.float32)
    P,Q = D.shape[:2]
    for u in range(P):
        for v in range(Q):
            D[u,v] = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
    H = np.exp((-D**2) / (2*(D0**2)))
    result = IDFT(DFT(padded_img) * H)
    result = result[:img.shape[0],:img.shape[1]]
    return result

# BUTTERWORTH LOWPASS FILTERS
def BLBF(img,D0,n):
    padded_img = np.pad(img,((0,img.shape[0]),(0,img.shape[1])),mode='reflect')
    D = np.zeros_like(padded_img,dtype=np.float32)
    P,Q = D.shape[:2]
    for u in range(P):
        for v in range(Q):
            D[u,v] = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
    H = 1 / (1+(D/D0)**(2*n))
    result = IDFT(DFT(padded_img) * H)
    result = result[:img.shape[0],:img.shape[1]]
    return result

# show results
fig = plt.figure(figsize=(10,10))
titles = ['Orginal Image','ILBF','GLBF','BLBF']
for i,image in enumerate([img ,ILBF(img,D0) ,GLBF(img,D0) ,BLBF(img,D0,n)]):
    ax = fig.add_subplot(2,2,i+1)
    ax.imshow(image,cmap='gray')
    ax.set_title(titles[i])
    ax.axis('off')
plt.show()