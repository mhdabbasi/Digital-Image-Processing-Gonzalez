import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0423(a)(rectangle).tif',0)

#print(img.shape) -> (1024,1024)
#img = cv2.resize(img,(49,49))
def DFT(img) :
    spectrum = np.zeros_like(img,dtype=complex)
    M,N = img.shape[:2]
    # to shift DFT
    for x in range(M):
        for y in range(N):
            img[x,y] = img[x,y]*(-1)**(x+y)

    # compute DFT
    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    spectrum[x,y] += img[x,y] * np.exp(-2j*np.pi*(u*x/M+v*y/N))

    return np.abs(spectrum)

# dft = DFT(img)

# so, it takes t(time to compute one value of spectrum[x,y])*M*M*N*N = 0.00018*1024**4 =54975 hour :-/
# BUT,i can use cv2.dft that compute dft of an signal . it use fast fourier transform ,so it's very fast.
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft = np.fft.fftshift(dft)        # to shift dft to center.
dft = np.sqrt(dft[:,:,0]**2+dft[:,:,1]**2)

plt.imshow(dft,cmap='gray')
plt.show()