import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread('chapter4-Filtering in the Frequency Domain/images/Fig0458(a)(blurry_moon).tif',0)

def laplacian(img):
    #1_Normalize img and pad it
    padded_img = np.pad(img,((0,img.shape[0]),(0,img.shape[1])))
    normalized_img = (padded_img-np.min(padded_img))/(np.max(padded_img)-np.min(padded_img))
    

    #2_compute DFT of normalized_img
    Fuv = np.fft.fftshift(np.fft.fft2(normalized_img))

    #3_Multiply Laplacian filter with frequency data
    D = np.zeros_like(Fuv,dtype=np.float32)
    P,Q = D.shape[:2]
    for u in range(P):
        for v in range(Q):
            D[u,v] = (u-P/2)**2 + (v-Q/2)**2
    H = -4.0*(np.pi**2)*D
    lap_filter = H*Fuv
    lap_filter = np.fft.ifftshift(lap_filter)

    #4_compute IDFT 
    idft_filter = np.fft.ifft2(lap_filter) 
    idft_filter2 = idft_filter[:img.shape[0],:img.shape[1]]
    normalized_idft_filter = idft_filter2 / np.max(idft_filter2)

    #5_Add normalized data and data we filtered with Laplacian filter
    c = -0.5
    normalized_img_without_pad = (img-np.min(img))/(np.max(img)-np.min(img))
    result = normalized_img_without_pad + c*normalized_idft_filter
    result = np.abs(result)
    scaled_result = 255*(result-np.min(result))/(np.max(result)-np.min(result))

    return scaled_result

result = laplacian(img)

plt.imshow(result,cmap='gray')
plt.show()