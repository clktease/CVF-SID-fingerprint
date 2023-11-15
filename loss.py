import torch
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2

def psnr(img, imclean):
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = []
    for i in range(Img.shape[0]):
        ps = compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.0)
        if np.isinf(ps):
            continue
        PSNR.append(ps)
    return sum(PSNR)/len(PSNR)


def ssim(img, imclean):
    img = img.mul(255).clamp(0, 255).round().div(255)
    imclean = imclean.mul(255).clamp(0, 255).round().div(255)
    Img = img.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
    SSIM = []
    for i in range(Img.shape[0]):
        ss = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], multichannel =True)
        SSIM.append(ss)
    return sum(SSIM)/len(SSIM)
print("finish")

mse = nn.MSELoss(reduction='mean') 

    
def loss_aug(clean, clean1, noise_w, noise_w1, noise_b, noise_b1):
    loss1 = mse(clean1,clean)
    loss2 = mse(noise_w1,noise_w)
    loss3 = mse(noise_b1,noise_b)
    loss = loss1 + loss2 + loss3
    return loss

def test(src, model,opt=0):
    
    img1 = (src*255.0).astype('uint8')
    img2 = (model*255.0).astype('uint8')
    sift = cv2.SIFT_create()  
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  
    try:
        matches = flann.knnMatch(des1, des2, k=2)  
    except:
        return 0
    good_matches = []
    temp = []
    for i,(m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:  
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            if abs(pt1[1]-pt2[1]) < 15:
                temp.append([int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])])
                good_matches.append(m)
    num=len(good_matches)
    if opt == 0:
        return temp
    else:
        return 1
def generate_sift(x_train,y_train,batch_size):
    n = np.zeros((batch_size,176,36))
    for i in range(batch_size):
        x = test(x_train[i,0,:,:],y_train[i,0,:,:],0)
        if x == 0:
            continue
        for k in x:
            n[i,k[1],k[0]] = 1
    n = n.reshape([n.shape[0],1,176,36])
    return n
def match_loss(clean,input_clear,input_noisy,batch_size):
    clean_output = clean.cpu().detach().numpy()
    clean_input = input_clear.cpu().detach().numpy()
    input_noisy = input_noisy.cpu().detach().numpy()
    
    temp = 0.0
    p = []
    for i in range(batch_size):
        n = test(clean_output[i,0,:,:],clean_input[i,0,:,:],1)
        n1 = test(input_noisy[i,0,:,:],clean_input[i,0,:,:],1)
        r = n - n1
        if r  > 0:
            temp += 0.0
        elif r == 0:
            temp += 1.0
        elif r < 0:
            temp += 2.0
            
    return temp*0.003
    
def loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2):
    loss1 = mse(input_noisy_pred, input_noisy)
      
    loss2 = mse(clean1,clean)
    loss3 = mse(noise_b3, noise_b)
    loss4 = mse(noise_w2, noise_w)
    loss5 = mse(clean2, clean)
    
    loss6 = mse(clean3, torch.zeros_like(clean3))
    loss7 = mse(noise_w1, torch.zeros_like(noise_w1))
    loss8 = mse(noise_b1, torch.zeros_like(noise_b1))
    loss9 = mse(noise_b2, torch.zeros_like(noise_b2))

    loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
    return loss


if __name__ == '__main__':
    print('loss')