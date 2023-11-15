import cv2
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from sklearn.neighbors import KNeighborsRegressor

def power_spectrum(img):
    
    n = int( math.ceil(img.shape[0] / 2.) * 2 )

    a = np.fft.rfft(img,n, axis=0)
    a = a.real*a.real + a.imag*a.imag
    a = a.sum(axis=1)/a.shape[1]
    f = np.fft.rfftfreq(n)
    ps = 0
    for i in range(f.shape[0]):
        ps += math.log(a[i]) * f[i]

    n = int( math.ceil(img.shape[1] / 2.) * 2 )
    a = np.fft.rfft(img,n,axis=1)
    a = a.real*a.real + a.imag*a.imag
    a = a.sum(axis=0)/a.shape[0]
    f = np.fft.rfftfreq(n)
    for i in range(f.shape[0]):
        ps += math.log(a[i]) * f[i]
    ps = min(1.0,max(0,(ps-250)/100))
        
    return ps
def smudge_noise(img):
    
    #img = cv2.imread('E:/fingerprintd1/testset_nasic9395_v9.1/identify/2_0_140.bmp', 0)
    #print(img.shape)
    contrast = 200
    brightness = 0
    output = img * (contrast/127 + 1) - contrast + brightness 
    output = np.clip(output, 0, 255)
    output = np.uint8(output)

    output = output[3:173,4:32]
    trans = (output <= 30)
    l = label(trans)
    x = np.array([])
    
    if np.bincount(l.ravel())[1:].shape[0] == 0:
        trans = (output <= output.mean())
        l = label(trans)
        out = (l==np.bincount(l.ravel())[1:].argmax()+1).astype(int)
    else:
        out = (l==np.bincount(l.ravel())[1:].argmax()+1).astype(int)

    value = (out==0).sum()/(170*28)
    #print(value)
    return value


def KNN(file_path):
    
    x_train = np.load('./KNN_x_train.npz')
    y_train = np.load('./KNN_y_train.npz')
    x_train = x_train['x_train']
    y_train = y_train['y_train']
    KNN.fit(x_train,y_train)
    KNN = KNeighborsRegressor(n_neighbors=5)
    
    data = np.load('F:/fingerprint/nasic9395_1008_v1_testset_v8/x_train.npz')
    data1 = np.load('F:/fingerprint/nasic9395_1008_v1_testset_v8/y_train.npz')

    x_train = data['x_train']
    y_train = data1['y_train']
    
    file_path= ''
    img = cv2.imread(file_path,0)



    p1 = np.array([
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
    ])
    p2 = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
    ])
    img1 = cv2.filter2D(img, -1, kernel=p1)
    img2 = cv2.filter2D(img, -1, kernel=p2)
    img3 = np.sqrt((img1.astype('int32') * img1.astype('int32')) + (img2.astype('int32') * img2.astype('int32')))
    img = cv2.imread(i,0)
    value1 = power_spectrum(img3)
    value2 = smudge_noise(img)
    metric = np.array([value1,value2])
    p = KNN.predict(np.reshape(metric,(1,-1)))
    
    if p>0.5:
        quality= 1
    else:
        quality=0
    return quality 
if __name__ == '__main__':
    file_path = ''
    KNN(file_path)