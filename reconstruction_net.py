import copy
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

from my_first_CNN import my_first_CNN, one_layer_CNN, my_CNN_batchnorm
from unet import UNet
from didn_class import DIDN

# sys.path.insert(0, '/media/data/Pierre/Works/THESES/These_Valentin_Debarnot/DEEP_BLUR/Code/alternatives/DPIR')
sys.path.insert(0, '../alternatives/DPIR')
from models.network_unet import UNetRes as Znet
    
dtype = torch.cuda.FloatTensor

def change_momentum(model):
    for name, param in model.named_parameters():
        print(name, param.size())

def imshow_torch(x):
    tmp = x.detach().cpu().numpy()
    plt.imshow(tmp)
    plt.show()

def cg(C, b, nitermax, callback=None):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    #res_hist = []
    if len(b.shape)==3:
        ps_r = torch.abs(r*r).sum(dim=2).sum(dim=1)
        for k in range(nitermax):
            Cp = C(p)
            alpha = ps_r/(p*torch.conj(Cp)).sum(dim=2).sum(dim=1)
            x = x + alpha.unsqueeze(-1).unsqueeze(-1)*p
            r = r - alpha.unsqueeze(-1).unsqueeze(-1)*Cp
            ps_rp1 = torch.abs(r*r).sum(dim=2).sum(dim=1)
            beta = ps_rp1/ps_r
            p = r+beta.unsqueeze(-1).unsqueeze(-1)*p
            ps_r = ps_rp1
            #res_hist.append(ps_r.sum().sqrt().item())
            if callback is not None:
                callback(x)
    else:
        ps_r = torch.abs(r*r).sum(dim=3).sum(dim=2).sum(dim=1)
        for k in range(nitermax):
            Cp = C(p)
            alpha = ps_r/(p*torch.conj(Cp)).sum(dim=3).sum(dim=2).sum(dim=1)
            x = x + alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*p
            r = r - alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*Cp
            ps_rp1 = torch.abs(r*r).sum(dim=3).sum(dim=2).sum(dim=1)
            beta = ps_rp1/ps_r
            p = r+beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*p
            ps_r = ps_rp1
            #res_hist.append(ps_r.sum().sqrt().item())
            if callback is not None:
                callback(x)
    return x #, res_hist

def tikhonov(A, At, y, lamb, nitermax, callback=None):
    # Solves the linear system (A^T A + \lambda I) x = A^T y  ie Cx=b
    def Cop(x):
        return At(A(x))+lamb*x
    b = At(y)
    return cg(Cop, b, nitermax, callback=callback)

## Solves (AtA + lamb I) x = b
def resolvent(A, At, b, lamb, nitermax, callback=None):
    # Solves the linear system (A^T A + \lambda I) x = A^T y  ie Cx=b
    def Cop(x):
        return At(A(x))+lamb*x
    return cg(Cop, b, nitermax, callback=callback)



## My first reconstruction network
class reconstruction_net(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=1):
        super(reconstruction_net, self).__init__()
        self.num_iter = num_iter
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            #self.proxes.append(my_first_CNN2(num_channels=16,bias=True))
            self.proxes.append(my_first_CNN(num_channels=16,bias=True))
            # self.proxes.append(skip(num_input_channels=1, num_output_channels = 1).type(dtype))
            #self.proxes.append(UNet(num_input_channels=1, num_output_channels=1,need_sigmoid=False,pad='replication'))
            #self.proxes.append(ResNet(num_input_channels=1, num_output_channels=1, num_blocks=5, num_channels=8,need_sigmoid=False))

    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        z = y.clone()
        xp = y.clone()
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.proxes[i]( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.99*(x-xp)
            xp = x.clone()
        return x

## My first reconstruction network with initialization
class reconstruction_net_initialization(nn.Module):
    def __init__(self,A,AT,net_denoise,num_iter=10,tau=1):
        super(reconstruction_net_initialization, self).__init__()
        self.num_iter = num_iter
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            net_denoise_tmp = my_first_CNN(num_channels=16,bias=True)
            net_denoise_tmp.load_state_dict(copy.deepcopy(net_denoise.state_dict()))
            #self.proxes.append(my_first_CNN2(num_channels=16,bias=True))
            self.proxes.append(net_denoise_tmp)
            # self.proxes.append(skip(num_input_channels=1, num_output_channels = 1).type(dtype))
            #self.proxes.append(UNet(num_input_channels=1, num_output_channels=1,need_sigmoid=False,pad='replication'))
            #self.proxes.append(ResNet(num_input_channels=1, num_output_channels=1, num_blocks=5, num_channels=8,need_sigmoid=False))

    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        z = y.clone()
        xp = y.clone()
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.proxes[i]( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.99*(x-xp)
            xp = x.clone()
        return x

## My first reconstruction network with initialization
class reconstruction_net_initialization_didn(nn.Module):
    def __init__(self,A,AT,net_denoise,num_iter=10,tau=1):
        super(reconstruction_net_initialization_didn, self).__init__()
        self.num_iter = num_iter
        self.A = A
        self.AT = AT
        self.tau = tau

        # self.prox = DIDN(1, 1, num_chans=32, bias=True)
        # self.prox.load_state_dict(copy.deepcopy(net_denoise.state_dict()))
        self.prox = net_denoise

        # self.proxes = nn.ModuleList() # The "proximal" operators
        # for i in range(num_iter):
        #     net_denoise_tmp = DIDN(1, 1, num_chans=32, bias=True)
        #     net_denoise_tmp.load_state_dict(copy.deepcopy(net_denoise.state_dict()))
        #     #self.proxes.append(my_first_CNN2(num_channels=16,bias=True))
        #     self.proxes.append(net_denoise_tmp)
        #     # self.proxes.append(skip(num_input_channels=1, num_output_channels = 1).type(dtype))
        #     #self.proxes.append(UNet(num_input_channels=1, num_output_channels=1,need_sigmoid=False,pad='replication'))
        #     #self.proxes.append(ResNet(num_input_channels=1, num_output_channels=1, num_blocks=5, num_channels=8,need_sigmoid=False))

    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        z = y.clone()
        xp = y.clone()
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.prox( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.99*(x-xp)
            xp = x.clone()
        return x

## Reconstruction network Proximal gradient descent
class reconstruction_net_MYCNN_APGD(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=0.8,nit_ini=5):
        super(reconstruction_net_MYCNN_APGD, self).__init__()
        self.num_iter = num_iter
        self.nit_ini = nit_ini  
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(my_first_CNN(num_channels=16,bias=True))
            self.proxes[i].init_weights()
             
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)
        x = self.AT(y)
        for i in range(self.nit_ini):
            x = x - self.tau*self.AT(self.A(x)-y)
    
        # "proximal" gradent descent
        z = x.clone()
        xp = x.clone()
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.proxes[i]( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.9*(x-xp)
            xp = x.clone()
        return x

## Reconstruction network Accelerate Proximal gradient descent
class reconstruction_net_MYCNN_PGD(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=0.8,nit_ini=5):
        super(reconstruction_net_MYCNN_PGD, self).__init__()
        self.num_iter = num_iter
        self.nit_ini = nit_ini  
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(my_first_CNN(num_channels=16,bias=True))
            self.proxes[i].init_weights()
             
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)
        x = self.AT(y)
        for i in range(self.nit_ini):
            x = x - self.tau*self.AT(self.A(x)-y)
            
        # "proximal" gradent descent
        for i in range(self.num_iter):
            x = self.proxes[i]( x - self.tau*self.AT(self.A(x)-y) )
        return x

## An unrolled reconstruction network (to handle different input size)
class reconstruction_net_UNET_PGD(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=0.8,nit_ini=5):
        super(reconstruction_net_UNET_PGD, self).__init__()
        self.num_iter = num_iter
        self.nit_ini = nit_ini  
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(UNet(num_input_channels=1, num_output_channels=1,need_sigmoid=True,pad='replication'))
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)
        
        x = self.AT(y)
        for i in range(self.nit_ini):
            x = x - self.tau*self.AT(self.A(x)-y)
    
        # "proximal" gradent descent
        for i in range(self.num_iter):
            x = self.proxes[i]( x - self.tau*self.AT(self.A(x)-y) )
        return x

## An unrolled reconstruction network (to handle different input size)
class reconstruction_net_UNET_DR(nn.Module):
    def __init__(self,A,AT,lamb=1e-2,num_iter=5,nit_ini_cg=10,nit_cg=5):
        super(reconstruction_net_UNET_DR, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(UNet(num_input_channels=1, num_output_channels=1,need_sigmoid=False,pad='replication').cuda())
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        # "proximal" gradent descent
        for i in range(self.num_iter):
            z = self.proxes[i](x)
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
        return z


## An unrolled Douglas Rachford reconstruction network with DIDN (to handle different input size)
class reconstruction_net_DIDN_DR(nn.Module):
    def __init__(self,A,AT,lamb=1e-2,num_iter=5,nit_ini_cg=15,nit_cg=10):
        super(reconstruction_net_DIDN_DR, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        self.DIDN = DIDN(1, 1, num_chans=32, bias=True).cuda() # The "proximal" operators
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        # "proximal" gradent descent
        for i in range(self.num_iter):
            z = self.DIDN(x)
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
        return z
    
## An unrolled Douglas Rachford reconstruction network with Zhang net
class reconstruction_net_DIDN_Zhang(nn.Module):
    def __init__(self,A,AT,lamb=3e-2,sigma=4e-1,num_iter=2,nit_ini_cg=10,nit_cg=8):
        super(reconstruction_net_DIDN_Zhang, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        self.sigma = sigma
        
        self.model_path = "/media/data/Pierre/Works/THESES/These_Valentin_Debarnot/DEEP_BLUR/Code/alternatives/DPIR/ckpts/drunet_gray.pth"
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prox = Znet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        self.prox.load_state_dict(torch.load(self.model_path), strict=True)
        self.prox.train()
        for k, v in self.prox.named_parameters():
            v.requires_grad = True
        self.prox = self.prox.to(device)
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        
        #imshow_torch(x[0,0])
        # "proximal" gradent descent
        for i in range(self.num_iter):
            xx = torch.cat((x, self.sigma*torch.ones_like(x)), dim=1)
            z = self.prox(xx)

            #imshow_torch(z[0,0])
            
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
            
            #imshow_torch(x[0,0])
        return z    
    
## An unrolled Douglas Rachford reconstruction network with Zhang net
class reconstruction_net_DIDN_Zhang2(nn.Module):
    def __init__(self,A,AT,lamb=3e-2,sigma=4e-1,num_iter=2,nit_ini_cg=10,nit_cg=8,model_path="/media/data/Pierre/Works/THESES/These_Valentin_Debarnot/DEEP_BLUR/Code/alternatives/DPIR/ckpts/drunet_gray.pth"):
        super(reconstruction_net_DIDN_Zhang2, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        self.sigma = sigma
        
        self.model_path = model_path
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prox = Znet(in_nc=n_channels, out_nc=n_channels, nc=[16, 32, 64, 128], nb=2, act_mode='R', downsample_mode="avgpool", upsample_mode="convtranspose")
        self.prox.load_state_dict(torch.load(self.model_path), strict=True)
        self.prox.train()
        for k, v in self.prox.named_parameters():
            v.requires_grad = True
        self.prox = self.prox.to(device)
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        
        #imshow_torch(x[0,0])
        # "proximal" gradent descent
        for i in range(self.num_iter):
            # xx = torch.cat((x, self.sigma*torch.ones_like(x)), dim=1)
            z = self.prox(x)

            #imshow_torch(z[0,0])
            
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
            
            #imshow_torch(x[0,0])
        return z    


## An unrolled Douglas Rachford reconstruction network with Zhang net
class reconstruction_net_DR_Zhang(nn.Module):
    def __init__(self,A,AT,lamb=3e-2,num_iter=4,nit_ini_cg=10,nit_cg=5):
        super(reconstruction_net_DR_Zhang, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(Znet(in_nc=n_channels, out_nc=n_channels, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose"))
            
        for i in range(num_iter):
            for k, v in self.proxes[i].named_parameters():
                v.requires_grad = True
            self.proxes[i] = self.proxes[i].to(device)
            self.proxes[i].train()
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Douglas Rachford
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        
        #imshow_torch(x[0,0])
        # "proximal" gradent descent
        for i in range(self.num_iter):
            z = self.proxes[i](x)
            
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r            
            #imshow_torch(x[0,0])
        return z  

## An unrolled Douglas Rachford reconstruction network with Zhang net adaptive to the noise level
class reconstruction_net_DR_Zhang_adaptive(nn.Module):
    def __init__(self,A,AT,lamb=3e-2,num_iter=4,nit_ini_cg=10,nit_cg=5):
        super(reconstruction_net_DR_Zhang_adaptive, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(Znet(in_nc=n_channels+1, out_nc=n_channels, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose"))
            
        for i in range(num_iter):
            for k, v in self.proxes[i].named_parameters():
                v.requires_grad = True
            self.proxes[i] = self.proxes[i].to(device)
            self.proxes[i].train()
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y, sigma): #Douglas Rachford
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        
        #imshow_torch(x[0,0])
        # "proximal" gradent descent
        for i in range(self.num_iter):
            xx = torch.cat((x, sigma*torch.ones_like(x)), dim=1) #Only difference, we feed the noise level
            z = self.proxes[i](xx)
            
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r            
            #imshow_torch(x[0,0])
        return z    

## An unrolled Douglas Rachford reconstruction network with Zhang net
# modify to change inputs
class reconstruction_net_DR_Zhang2(nn.Module):
    def __init__(self,A_op,AT_op,lamb=3e-2,num_iter=4,nit_ini_cg=10,nit_cg=5):
        super(reconstruction_net_DR_Zhang2, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A_op = A_op
        self.AT_op = AT_op
        self.lamb = lamb
        
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(Znet(in_nc=n_channels, out_nc=n_channels, nc=[32, 64, 128, 256], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose"))
            
        for i in range(num_iter):
            for k, v in self.proxes[i].named_parameters():
                v.requires_grad = True
            self.proxes[i] = self.proxes[i].to(device)
            self.proxes[i].train()
            
    # def change_operator(self,A,AT):
    #     self.A = A
    #     self.AT = AT
        
    def forward(self, y, h_t): #Douglas Rachford
        # defining the initial point with a plain gradient descent (change to CG!)  

        self.A = lambda x : self.A_op(x,h_t)
        self.AT = lambda x : self.AT_op(x,h_t)

        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        
        #imshow_torch(x[0,0])
        # "proximal" gradent descent
        for i in range(self.num_iter):
            z = self.proxes[i](x)
            
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r            
            #imshow_torch(x[0,0])
        return z    

## My second reconstruction network (to handle different input size)
class reconstruction_net_basic(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=1,nit_ini=5):
        super(reconstruction_net_basic, self).__init__()
        self.num_iter = num_iter
        self.nit_ini = nit_ini  
        self.A = A
        self.AT = AT
        self.tau = tau

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(one_layer_CNN(num_channels=16,bias=True))
            self.proxes[i].init_weights()
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)
        x = self.AT(y)
        for i in range(self.nit_ini):
            x = x - self.tau*self.AT(self.A(x)-y)
    
        # "proximal" gradent descent
        z = x.clone()
        xp = x.clone()
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.proxes[i]( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.99*(x-xp)
            xp = x.clone()
        return x
    
## My first reconstruction network
class reconstruction_net_batchnorm(nn.Module):
    def __init__(self,A,AT,num_iter=10,tau=1,nit_ini=5):
        super(reconstruction_net_batchnorm, self).__init__()
        self.num_iter = num_iter
        self.A = A
        self.AT = AT
        self.tau = tau
        self.nit_ini = nit_ini  

        self.proxes = nn.ModuleList() # The "proximal" operators
        for i in range(num_iter):
            self.proxes.append(my_CNN_batchnorm(num_channels=16,bias=True))
  
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        x = self.AT(y)
        for i in range(self.nit_ini):
            x = x - self.tau*self.AT(self.A(x)-y)
        z = x.clone()
        xp = x.clone()
        
        for i in range(self.num_iter):
            #x = z - (self.AT(self.A(z)-y) - self.proxes[i](z))
            x = self.proxes[i]( z - self.tau*self.AT(self.A(z)-y) )
            z = x + 0.99*(x-xp)
            xp = x.clone()
        return x
    
## An unrolled reconstruction network (to handle different input size)
class reconstruction_net_plugandplay_DR(nn.Module):
    def __init__(self,A,AT,denoising_network,lamb=1e-1,num_iter=10,nit_ini_cg=10,nit_cg=10):
        super(reconstruction_net_plugandplay_DR, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        self.proxes = denoising_network.cuda()
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        # "proximal" gradent descent
        for i in range(self.num_iter):
            z = self.proxes(x)
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
        return z
    
## An unrolled reconstruction network (to handle different input size)
class reconstruction_net_plugandplay_Zhang(nn.Module):
    def __init__(self,A,AT,lamb=1e-1,sigma=1e-1,num_iter=8,nit_ini_cg=15,nit_cg=15):
        super(reconstruction_net_plugandplay_Zhang, self).__init__()
        self.num_iter = num_iter
        self.nit_ini_cg = nit_ini_cg
        self.nit_cg = nit_cg
        self.A = A
        self.AT = AT
        self.lamb = lamb
        self.model_path = "/media/data/Pierre/Works/THESES/These_Valentin_Debarnot/DEEP_BLUR/Code/alternatives/DPIR/ckpts/drunet_gray.pth"
        n_channels = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prox = Znet(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        self.prox.load_state_dict(torch.load(self.model_path), strict=True)
        self.prox.eval()
        for k, v in self.prox.named_parameters():
            v.requires_grad = False
        self.prox = self.prox.to(device)
        self.sigma = sigma
            
    def change_operator(self,A,AT):
        self.A = A
        self.AT = AT
        
    def forward(self, y): #Prox descent a la Nesterov
        # defining the initial point with a plain gradient descent (change to CG!)  
        w = self.AT(y)
        x = resolvent(self.A, self.AT, w, self.lamb, self.nit_ini_cg)
        # "proximal" gradent descent
        for i in range(self.num_iter):
            xx = torch.cat((x, self.sigma*torch.ones_like(x)), dim=1)

            z = self.prox(xx)
            w = self.AT(y) + self.lamb*(2*z - x)
            r = resolvent(self.A, self.AT, w, self.lamb, self.nit_cg)
            x = x - z + r
        return z

    