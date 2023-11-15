import torch.nn.functional as F
import torch.nn as nn
import torch

class AddCoords(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False, skiptile=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.skiptile = skiptile

    def forward(self, input_tensor,match_tensor):
        
        out = torch.cat([input_tensor,match_tensor],dim=1)
        return out

class CoordConv(nn.Module):
    def __init__(self, x_dim, y_dim, with_r):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, 
                                   y_dim=y_dim, 
                                   with_r=with_r,
                                   skiptile=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    def forward(self, input_tensor,match_tensor):
        ret = self.addcoords(input_tensor,match_tensor)
        return ret
        
class GenClean(nn.Module):

    def __init__(self, channels=1, num_of_layers=10):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 32
        self.num_of_layers = num_of_layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=33, out_channels=features, kernel_size=kernel_size, padding=padding)
        self.r1 =  nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0)
        self.addcoords = AddCoords(x_dim=36, 
                                   y_dim=36, 
                                   with_r=0,
                                   skiptile=True)
    def forward(self, x,y):
        x = self.conv1(x)
        x = self.r1(x)
        for _ in range(self.num_of_layers-2):
            x = self.addcoords(x,y)
            x = self.conv2(x)
            x = self.r1(x)
        out = self.conv3(x)
        return out

class GenNoise(nn.Module):
    def __init__(self, NLayer=8, FSize=32):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1   
        m = [nn.Conv2d(1, FSize, kernel_size=kernel_size, padding=padding),
             nn.ReLU(inplace=True)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            m.append(nn.ReLU(inplace=True))        
        self.body = nn.Sequential(*m)
        
        gen_noise_w = []
        for i in range(4):
            gen_noise_w.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            gen_noise_w.append(nn.ReLU(inplace=True))
        
        gen_noise_w.append(nn.Conv2d(FSize, 1, kernel_size=1, padding=0))       
        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        gen_noise_b = []
        for i in range(4):
            gen_noise_b.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            gen_noise_b.append(nn.ReLU(inplace=True))
        
        gen_noise_b.append(nn.Conv2d(FSize, 1, kernel_size=1, padding=0))       
        self.gen_noise_b = nn.Sequential(*gen_noise_b)
        
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.gen_noise_w:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)  
        for m in self.gen_noise_b:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
      
    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        noise_w = self.gen_noise_w(noise)
        noise_b = self.gen_noise_b(noise)       
          
        m_w = torch.mean(torch.mean(noise_w,-1),-1).unsqueeze(-1).unsqueeze(-1)
        noise_w = noise_w-m_w      
        m_b = torch.mean(torch.mean(noise_b,-1),-1).unsqueeze(-1).unsqueeze(-1)
        noise_b = noise_b-m_b

        return noise_w, noise_b

class MLDN_model(nn.Module):
    def __init__(self):
        super().__init__()
        FSize = 64
        self.gen_noise = GenNoise(FSize=FSize)
        self.genclean1 = GenClean()
        self.genclean2 = GenClean()
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x,y,level = 0,weights=None, test=False):                   
        if level == 1:
            clean = self.genclean1(x,y)
        else:
            clean = self.genclean2(x,y)
        noise_w, noise_b = self.gen_noise(x-clean)               
        return noise_w, noise_b, clean