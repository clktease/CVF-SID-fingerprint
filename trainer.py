import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm
import torch.nn as nn
from model import MLDN_model
from loss import loss_aug, loss_main,ssim,psnr,match_loss,generate_sift,test

mse = nn.MSELoss(reduction='mean') 

def load_train_data(data):
    datatype = data
    if data == 'train':
        data = np.load('./dataset/x_train.npz')
        data1 = np.load('./dataset/y_train.npz')
    elif data == data == 'val':
        data = np.load('./dataset/x_val.npz')
        data1 = np.load('./dataset/y_val.npz')
    elif data == data == 'test':
        data = np.load('./dataset/x_test.npz')
        data1 = np.load('./dataset/y_test.npz')
    fdx = 'x_'+datatype
    fdy = 'y_'+datatype
    x_train = data[fdx]
    x_train = np.array(x_train).astype('float16')
    x_train = x_train.reshape([x_train.shape[0],1,176,36]) / 255.0
    

    y_train = data1[fdy]
    y_train = np.array(y_train).astype('float16')
    y_train = y_train.reshape([y_train.shape[0],1,176,36]) / 255.0

    return x_train,y_train

def train(model,device,x_train,y_train,x_val,y_val,batch_size = 16):
    epoch_num = 50
    level = 0
    lossh = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-09, amsgrad=True)
    for epoch in tqdm.tqdm(range(0,50)):
        for index in range(int(x_train.shape[0] / batch_size ) ):
            
            sift = generate_sift(x_train[index*batch_size:(index+1)*batch_size],y_train[index*batch_size:(index+1)*batch_size],batch_size)
    
            sift = torch.from_numpy(sift)
            sift = sift.float().to(device)
            
            input_noisy = torch.from_numpy(x_train[index*batch_size:(index+1)*batch_size])
            input_clear = torch.from_numpy(y_train[index*batch_size:(index+1)*batch_size])
            
            #input_match = match[batch_indexes]
            
            input_noisy = input_noisy.float().to(device)
            input_clear = input_clear.float().to(device)
            
            
            optimizer.zero_grad()
            
            if index * batch_size<13408:
                level = 1
            else:
                level = 0
            noise_w, noise_b, clean = model(input_noisy,sift,level)
            noise_w1, noise_b1, clean1 = model((clean),sift,level)
            noise_w2, noise_b2, clean2 = model((clean + noise_w),sift,level) 
            noise_w3, noise_b3, clean3 = model((noise_b),sift,level)

            
            #stage3
            #noise_w4, noise_b4, clean4 = model((clean+torch.pow(clean,gamma)*noise_w-noise_b),sift,level) #2
            #noise_w5, noise_b5, clean5 = model((clean-torch.pow(clean,gamma)*noise_w+noise_b),sift,level) #3
            #noise_w6, noise_b6, clean6 = model((clean-torch.pow(clean,gamma)*noise_w-noise_b),sift,level) #4
            #noise_w7, noise_b7, clean7 = model((clean+noise_w+noise_b),sift,level) #5

            
            
            input_noisy_pred = clean+noise_w+noise_b
    
            loss_match  = match_loss(clean,input_clear,input_noisy,batch_size)

        
            loss_rec = mse(clean,input_clear)

            
            loss = loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2)
            #stage3
            #loss_aug1 = loss_aug(clean, clean4, noise_w, noise_w4, noise_b, -noise_b4)
            #loss_aug2 = loss_aug(clean, clean5, noise_w, -noise_w5, noise_b, noise_b5)
            #loss_aug3 = loss_aug(clean, clean6, noise_w, -noise_w6, noise_b, -noise_b6)
            #loss_aug4 = loss_aug(clean, clean7, noise_w, noise_w7, noise_b, noise_b7)    
            
            if level == 1:
                loss_total = loss + loss_rec * 0.5 + loss_match
            else:
                loss_total = loss + loss_rec * 1.25 + loss_match
            #stage3
            #loss_total = loss + 0.1(loss_aug1+loss_aug2+loss_aug3+loss_aug4)+ loss_rec * 0.5 + loss_match
            
            loss_total.backward()
            lossh.append([epoch,loss_total.cpu().detach()])
            optimizer.step()
            if index %30 ==0:
                print(epoch,index/int(x_train.shape[0] / batch_size), loss_total)

        torch.save(model.state_dict(), './weights/mldn_coord/1018_'+str(epoch)+'.pt')  

        Loss = 0
        for index in range(int(x_val.shape[0] / batch_size )):
            sift = generate_sift(x_val[index*batch_size:(index+1)*batch_size],y_val[index*batch_size:(index+1)*batch_size],batch_size)
            sift = torch.from_numpy(sift)
            sift = sift.float().to(device)
            
            input_noisy = torch.from_numpy(x_train[index*batch_size:(index+1)*batch_size])
            input_clear = torch.from_numpy(y_train[index*batch_size:(index+1)*batch_size])
            
            
            input_noisy = input_noisy.float().to(device)
            input_clear = input_clear.float().to(device)
            
            
            optimizer.zero_grad()
            
            if index * batch_size<196:
                level = 1
            else:
                level = 0
            noise_w, noise_b, clean = model(input_noisy,sift,level)
            noise_w1, noise_b1, clean1 = model((clean),sift,level)
            noise_w2, noise_b2, clean2 = model((clean + noise_w),sift,level) 
            noise_w3, noise_b3, clean3 = model((noise_b),sift,level)
         
            
            input_noisy_pred = clean+noise_w+noise_b
    
            loss_match  = match_loss(clean,input_clear,input_noisy,batch_size)

        
            loss_rec = mse(clean,input_clear)

            
            loss = loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2)  
            
            if level == 1:
                loss_total = loss + loss_rec * 0.5 + loss_match
            else:
                loss_total = loss + loss_rec * 1.25 + loss_match
            
            
            lossh.append([epoch,loss_total.cpu().detach()])
        
        print('Val loss', Loss/int(x_val.shape[0] / batch_size ))
    lossh1 = np.array(lossh)
    np.savez('./loss/mldn_coord.npz',lossh = lossh1)
def test(model,device,x_test,y_test,batch_size = 16):
    
    loss = 0
    s = 0
    p = 0
    for index in range(int(x_test.shape[0] / batch_size )):

               
        input_noisy = torch.from_numpy(x_test[index*batch_size:(index+1)*batch_size])
        input_clear = torch.from_numpy(y_test[index*batch_size:(index+1)*batch_size])
            
        input_noisy = input_noisy.float().to(device)
        input_clear = input_clear.float().to(device)
            
        noise_w, noise_b, clean = model(input_noisy)
        loss_clear = mse(clean,input_clear)
        s += ssim(clean,input_clear)
        p += psnr(clean,input_clear)
        loss += loss_clear
    print('MSE: ',loss/int(x_test.shape[0] / batch_size ))
    print('SSIM: ',s/int(x_test.shape[0] / batch_size ))
    print('PSNR: ',p/int(x_test.shape[0] / batch_size ))
def trainer(model,device):
    result_path = './weights/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_path+'mldn_coord/'):
        os.makedirs(result_path+'mldn_coord/')

    x_train,y_train = load_train_data('train')
    x_val,y_val = load_train_data('val')
    
    model = model.to(device)
    train(model,device,x_train,y_train,x_val,y_val,4)
    print('start train')
    # Test
    Test = False
    if Test:
        x_test,y_test = load_train_data('test')
        model.load_state_dict(torch.load('./weights/mldncoord_best.pt'))
        test(model,device,x_test,y_test,4)

if __name__ == '__main__':
    print('trainer')