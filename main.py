from model import MLDN_model
from trainer import trainer
import numpy as np
import torch
import os
import glob
import cv2

def load_test_data(device):
    
    data = np.load("./dataset/testset_v9/filename.npz")
    fn = data['filename']
    x_test = []
    coord = np.load("./dataset/testset_v9/test_coord.npz")
    coord = coord['coord']
    level = np.load("./dataset/testset_v9/test_level.npz")
    level = level['level']
    for i in range(fn.shape[0]):
        img = cv2.imread('./dataset/testset_v9/testset_nasic9395_v9/identify/'+fn[i],0)
        x_test.append(img)
    
    x_test = np.array(x_test).astype('float16')

    x_test = x_test.reshape([x_test.shape[0],1,176,36]) / 255.0
    x_test = torch.from_numpy(x_test).float().to(device)   
    coord = torch.from_numpy(coord).float().to(device)
    return x_test,coord,fn,level

def test(train = False):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLDN_model()
    x_test,coord,file,level = load_test_data(device)
    if not train:
        model.load_state_dict(torch.load('./weights/mldncoord_best.pt'))
        trainer(model,device)
    else:
        trainer(model,device)
    
    model = model.to(device)
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for i,k in enumerate(range(len(x_test))):
        #print(file[k],level[k])
        noise_wo, noise_bo, clean_output = model(x_test[k:k+1],coord[k],level[k])
        clean_output = clean_output.cpu().detach().numpy()
        img = clean_output[:,:,:]
        img = img.reshape(176,36)
        img = (img  * 255.0).astype('uint8')

        #you can save result with the original filename or replace with the next line to save result with number  
        cv2.imwrite(result_path + '/'+file[k],img.astype(np.uint8))
        #cv2.imwrite(result_path + '/'+str(i)+'.bmp',img.astype(np.uint8))

if __name__ == '__main__':
    print('start test')
    training = False
    test(training)