import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0464(a)(car_75DPI_Moire).tif',0)

def notch(img,n,D0,UVs):
    '''
    UVs = a list of your locations ,like [[50,50],[60,90]] .
    D0 = D0 for corsponding location in UVs .'''
    # compute dft of image
    F = np.fft.fftshift(np.fft.fft2(img))

    #compute H
    H = np.ones_like(img)
    M,N = img.shape[:2]
    for k in range(len(UVs)):
        Dok = np.zeros_like(H) # D+k
        Dnk = np.zeros_like(H) # D-k
        for u in range(M):
            for v in range(N):
                Dok[u,v] = np.sqrt((u-M/2-UVs[k][0])**2 + (v-N/2-UVs[k][1])**2)
                Dnk[u,v] = np.sqrt((u-M/2+UVs[k][0])**2 + (v-N/2+UVs[k][1])**2)
        H = H * ((1/(1+(D0[k]/Dok)**n)) * (1/(1+(D0[k]/Dnk)**n)))
    G = F*H
    g = np.fft.ifft2(np.fft.ifftshift(G))
    return np.abs(g)

result = notch(img,4,[9,9,9,9],[[38,29],[42,-27],[82,-27],[82,29]])

plt.imshow(result,cmap='gray')
plt.show()