# %% Loading libraries
## Libraries
import os
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import reconstruction_net
import generator 

#%% Change this if you don't have a GPU
# CUDA OR NOT CUDA
device = torch.device("cuda") 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

#%% Basic functions
def rescale_01(y_t):
    y_t -= y_t.amin(dim = (2,3),keepdim=True)
    y_t /= y_t.amax(dim = (2,3),keepdim=True)
    return y_t

def SNR(xref,x):
    return -10*np.log10(np.sum((xref-x)**2)/np.sum(xref**2))

#%% Parameters of the simulation
sigma_noise = 5e-2
params = {
    ## Noise level 
    "sigma_noise": 1e-1, # std of additive Gaussian noise
    
    ## Setting up the PSF generator
    "size_image": 400,
    "coeff_zernike": [4,5,6,7,8,9,10], # number of the Zernike coefficients (0 min, 37 max)
    "size_PSF": 31, # Size of PSF
    "Nz": 1,
    "stepZ": 0,
    "pixelSize": 100,
    "NA": 1.41,
    "wavelength": 540,
    "nI": 1.51,
    "min_coeff_zernike": -0.15*np.ones(7), # minimum value of the Zernike coefficients, same size than 'coeff_zernike'
    "max_coeff_zernike": 0.15*np.ones(7) # maximum value of the Zernike coefficients
}
# The PSF generator
psf_generator_t = generator.PSFGenerator2Dzernike_t(params["coeff_zernike"],imgSize=params["size_PSF"],pixelSize=params["pixelSize"],NA=params["NA"],wavelength=params["wavelength"],nI=params["nI"],min_coeff_zernike=params["min_coeff_zernike"],max_coeff_zernike=params["max_coeff_zernike"],res=params["size_PSF"],device=device,torch_type=dtype)

#%% Loading test images
image_path = "dataset/"
files = os.listdir(image_path)

image_tensors = []

# Define the transform to resize and normalize images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale format (1 channel)
    transforms.Resize((params["size_image"], params["size_image"])),  # Resize to target size
    transforms.ToTensor(),  # Convert image to tensor and scale to [0, 1]
])

# Iterate through the files in the directory and load images
for filename in os.listdir(image_path):
    file_path = os.path.join(image_path, filename)

    if os.path.isfile(file_path) and filename.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file_path).convert("L")  # Convert to grayscale
        image_tensor = transform(image)
        image_tensors.append(image_tensor)

# Convert the list of image tensors into a single tensor (batch x 1 x H x W)
x_t = torch.stack(image_tensors).type(dtype).to(device)
x = x_t.detach().cpu().numpy()
batch_size  = x_t.shape[0]

#%% Loading the weights of the DeepBlur models
repo_id = "pweiss/DeepBlur"
weights_file_identification_net = "Identification_Zernike_7Z_400.pth"
weights_file_deblurring_net = "Deblurring_Zernike_7Z_400_DR_Zhang_light_False.pth"

path_identification_net = hf_hub_download(repo_id=repo_id, filename=weights_file_identification_net)
path_deblurring_net = hf_hub_download(repo_id=repo_id, filename=weights_file_deblurring_net)

#%% Defining the forward operators
# forward operator
def A_op2(x,h):
    return F.conv2d(x, h[0][None,None], padding = 'valid')

# adjoint operator
def AT_op2(x,h):
    hf = torch.flip(h,(1,2))
    sh = h.shape[2]
    return F.conv2d(x, hf[0][None,None], padding = sh-1)

#%% Defining the identification network
def resnet18(num_classes=2,pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # For grayscale images
    model.fc = torch.nn.Linear(512,num_classes,bias=True) # to adapt the output
    return model

identification_net = resnet18(num_classes=np.size(params["coeff_zernike"]),pretrained=False).cuda()
identification_net.train()
checkpoint_identification = torch.load(path_identification_net)
identification_net.load_state_dict(checkpoint_identification['model_state_dict'])

#%% Loading the models
# Now, we compare the models
deblur_light = reconstruction_net.reconstruction_net_DR_Zhang([],[])
checkpoint_light = torch.load(path_deblurring_net)
deblur_light.load_state_dict(checkpoint_light['model_state_dict'])
deblur_light.requires_grad_(False)
deblur_light.eval()

#%% Testing the different models

# Defining a random set of operators
kernel_half_size = int((params["size_PSF"]-1)/2)
with torch.no_grad():
    coeff = psf_generator_t.generate_coeffs(batch_size)
    coeff_t = coeff.clone().detach().to(device)
    h_t = psf_generator_t.generate_psf(coeff)
   
    # Applying the model with noise
    A = lambda arg : A_op2(arg,h_t)
    AT = lambda arg : AT_op2(arg,h_t)
    
    y0_t = A(x_t)
    y_t = y0_t + sigma_noise * torch.randn_like(y0_t)
    y = y_t.cpu().detach().numpy()     

    # Identifying the blur    
    gamma_t = identification_net(rescale_01(y_t))
    h_est_t = psf_generator_t.generate_psf(gamma_t)
    A_est = lambda arg : A_op2(arg,h_est_t)
    AT_est = lambda arg : AT_op2(arg,h_est_t)

    # Now deblurring with the true operator
    deblur_light.change_operator(A_est,AT_est)
    deblur_light.change_operator(A_est,AT_est)
    
    # Evaluating the deblurring nets
    xest_t = deblur_light(y_t)
    
    x_est= xest_t.cpu().detach().numpy()        

    x_valid = x[:,:,kernel_half_size:-kernel_half_size,kernel_half_size:-kernel_half_size]
    x_est_valid = x_est[:,:,kernel_half_size:-kernel_half_size,kernel_half_size:-kernel_half_size]
    
    SNR0 = np.zeros(batch_size)
    SNR_restored = np.zeros(batch_size)

    for k in range(batch_size):
        SNR0[k] = SNR(x_valid[k,0], y[k,0])
        SNR_restored[k] = SNR(x_valid[k,0], x_est_valid[k,0])
        
#%%
for i in range(batch_size):
    print("Initial: %1.2f -- Restored: %1.2f" % (SNR0[i],SNR_restored[i]))    
    plt.figure(2*i)
    plt.subplot(1,3,1)
    plt.gray()
    plt.imshow(x_valid[i,0],vmin=0,vmax=1)
    plt.title("True")
    plt.subplot(1,3,2)
    plt.gray()
    plt.imshow(y[i,0],vmin=0,vmax=1)
    plt.title("Blurred: %1.2f" % SNR0[i])
    plt.subplot(1,3,3)
    plt.gray()
    plt.imshow(x_est_valid[i,0],vmin=0,vmax=1)
    plt.title("Restored: %1.2f" % SNR_restored[i])
    plt.show()
    
    plt.figure(2*i+1)
    plt.subplot(1,2,1)
    h_tmp = (h_t[i]/h_t[i].max()).to('cpu')
    h_est_tmp = (h_est_t[i]/h_est_t[i].max()).to('cpu')
    plt.imshow(h_tmp, vmin=0, vmax=1)
    plt.title("True PSF")
    plt.subplot(1,2,2)
    plt.imshow(h_est_tmp, vmin=0, vmax=1)
    plt.title("Recovered PSF")