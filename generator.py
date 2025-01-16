import torch
import numpy as np

def zernikecartesian(coefficient,x,y):
	"""
	------------------------------------------------
	__zernikecartesian__(coefficient,x,y):
	Return combined aberration
	Zernike Polynomials Caculation in Cartesian coordinates
	coefficient: Zernike Polynomials Coefficient from input
	x: x in Cartesian coordinates
	y: y in Cartesian coordinates
	------------------------------------------------

    Z1:Z00 Piston or Bias
Z2:Z11 x Tilt
Z3:Z11 y Tilt
Z4:Z20 Defocus
Z5:Z22 Primary Astigmatism at 45
Z6:Z22 Primary Astigmatism at 0
Z7:Z31 Primary y Coma
Z8:Z31 Primary x Coma
Z9:Z33 y Trefoil
Z10:Z33 x Trefoil
Z11:Z40 Primary Spherical
Z12:Z42 Secondary Astigmatism at 0
Z13:Z42 Secondary Astigmatism at 45
Z14:Z44 x Tetrafoil
Z15:Z44 y Tetrafoil
Z16:Z51 Secondary x Coma
Z17:Z51 Secondary y Coma
Z18:Z53 Secondary x Trefoil
Z19:Z53 Secondary y Trefoil
Z20:Z55 x Pentafoil
Z21:Z55 y Pentafoil
Z22:Z60 Secondary Spherical
Z23:Z62 Tertiary Astigmatism at 45
Z24:Z62 Tertiary Astigmatism at 0
Z25:Z64 Secondary x Trefoil
Z26:Z64 Secondary y Trefoil
Z27:Z66 Hexafoil Y
Z28:Z66 Hexafoil X
Z29:Z71 Tertiary y Coma
Z30:Z71 Tertiary x Coma
Z31:Z73 Tertiary y Trefoil
Z32:Z73 Tertiary x Trefoil
Z33:Z75 Secondary Pentafoil Y
Z34:Z75 Secondary Pentafoil X
Z35:Z77 Heptafoil Y
Z36:Z77 Heptafoil X
Z37:Z80 Tertiary Spherical
	"""
	Z = [0]+coefficient
	r = np.sqrt(x**2 + y**2)
	Z1  =  Z[1]  * 1
	Z2  =  Z[2]  * 2*x
	Z3  =  Z[3]  * 2*y
	Z4  =  Z[4]  * np.sqrt(3)*(2*r**2-1)
	Z5  =  Z[5]  * 2*np.sqrt(6)*x*y
	Z6  =  Z[6]  * np.sqrt(6)*(x**2-y**2)
	Z7  =  Z[7]  * np.sqrt(8)*y*(3*r**2-2)
	Z8  =  Z[8]  * np.sqrt(8)*x*(3*r**2-2)
	Z9  =  Z[9]  * np.sqrt(8)*y*(3*x**2-y**2)
	Z10 =  Z[10] * np.sqrt(8)*x*(x**2-3*y**2)
	Z11 =  Z[11] * np.sqrt(5)*(6*r**4-6*r**2+1)
	Z12 =  Z[12] * np.sqrt(10)*(x**2-y**2)*(4*r**2-3)
	Z13 =  Z[13] * 2*np.sqrt(10)*x*y*(4*r**2-3)
	Z14 =  Z[14] * np.sqrt(10)*(r**4-8*x**2*y**2)
	Z15 =  Z[15] * 4*np.sqrt(10)*x*y*(x**2-y**2)
	Z16 =  Z[16] * np.sqrt(12)*x*(10*r**4-12*r**2+3)
	Z17 =  Z[17] * np.sqrt(12)*y*(10*r**4-12*r**2+3)
	Z18 =  Z[18] * np.sqrt(12)*x*(x**2-3*y**2)*(5*r**2-4)
	Z19 =  Z[19] * np.sqrt(12)*y*(3*x**2-y**2)*(5*r**2-4)
	Z20 =  Z[20] * np.sqrt(12)*x*(16*x**4-20*x**2*r**2+5*r**4)
	Z21 =  Z[21] * np.sqrt(12)*y*(16*y**4-20*y**2*r**2+5*r**4)
	Z22 =  Z[22] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1)
	Z23 =  Z[23] * 2*np.sqrt(14)*x*y*(15*r**4-20*r**2+6)
	Z24 =  Z[24] * np.sqrt(14)*(x**2-y**2)*(15*r**4-20*r**2+6)
	Z25 =  Z[25] * 4*np.sqrt(14)*x*y*(x**2-y**2)*(6*r**2-5)
	Z26 =  Z[26] * np.sqrt(14)*(8*x**4-8*x**2*r**2+r**4)*(6*r**2-5)
	Z27 =  Z[27] * np.sqrt(14)*x*y*(32*x**4-32*x**2*r**2+6*r**4)
	Z28 =  Z[28] * np.sqrt(14)*(32*x**6-48*x**4*r**2+18*x**2*r**4-r**6)
	Z29 =  Z[29] * 4*y*(35*r**6-60*r**4+30*r**2+10)
	Z30 =  Z[30] * 4*x*(35*r**6-60*r**4+30*r**2+10)
	Z31 =  Z[31] * 4*y*(3*x**2-y**2)*(21*r**4-30*r**2+10)
	Z32 =  Z[32] * 4*x*(x**2-3*y**2)*(21*r**4-30*r**2+10)
	Z33 =  Z[33] * 4*(7*r**2-6)*(4*x**2*y*(x**2-y**2)+y*(r**4-8*x**2*y**2))
	Z34 =  Z[34] * (4*(7*r**2-6)*(x*(r**4-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
	Z35 =  Z[35] * (8*x**2*y*(3*r**4-16*x**2*y**2)+4*y*(x**2-y**2)*(r**4-16*x**2*y**2))
	Z36 =  Z[36] * (4*x*(x**2-y**2)*(r**4-16*x**2*y**2)-8*x*y**2*(3*r**4-16*x**2*y**2))
	Z37 =  Z[37] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)
	ZW = 	Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ \
			Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
			Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
			Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
	return ZW


"""
PSF generator based on zernike polynomials.
INPUTs: 
    -coeff_zernike: list of indexes of Zernike polynomials to use. Default is astigmatism.
    -imgSize: size of the PSF.
    -pixelSize: size in nanometer of one pixel.
    -NA: numerical aperture.
    -wavelength: emission wavelength in nanometers.
    -nI: refractive index.
    -nZ: number z steps.
    -stepZ: size of the z steps in nanometers.
OUTPUTs of generate_blur:
    -psf: (imgSize,imgSize,nZ) psf with random Zernike coefficient between 0 and 1.
    -coeff: value of the zernike coefficient used to compute the PSF.

How to use:
    nZ=7
    gen=PSFGenerator2Dzernike(nZ=nZ)
    psf,_=gen.generate_psf()
    plt.figure(1)
    for z in range(nZ):
        plt.imshow(psf[:,:,z])
        plt.title("z={}".format(z))
        plt.pause(0.7)
"""
import galsim
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
class PSFGenerator3Dzernike():
    def __init__(self,coeff_zernike=[5,6],imgSize=17,pixelSize=240,NA=1.41,wavelength=540,nI=1.51,nZ=17,stepZ=100,
    min_coeff_zernike=[0,0],max_coeff_zernike=[0.4,0.4],res=256):
        self.coeff_zernike=coeff_zernike
        self.pixelSize=pixelSize
        self.NA=NA
        self.wavelength=wavelength
        self.nI=nI
        self.min_coeff_zernike=min_coeff_zernike
        self.max_coeff_zernike=max_coeff_zernike
        self.fc=(NA/wavelength)*pixelSize # Cutoff frequency
        self.kb=(nI/wavelength)*pixelSize # wavenumber
        self.nZ=nZ
        self.stepZ=stepZ
        self.imgSize_=imgSize
        self.imgSize=np.maximum(res,imgSize) # Pad for better discretization in Fourier domain
        lin=np.linspace(-0.5,0.5,self.imgSize)
        XX,YY=np.meshgrid(lin,lin)
        # self.rho=np.sqrt(XX**2+YY**2)
        self.rho,self.th=cart2pol(XX,YY)
        self.zer=[]
        for k in range(len(coeff_zernike)):
            coefficient=np.zeros(38)
            coefficient[coeff_zernike[k]]=1
            # zer_=zernikecartesian(coefficient,XX,YY)*(self.rho<=self.fc)
            zer_ = galsim.zernike.Zernike(coefficient)
            zer_ = zer_.evalPolar(self.rho/self.fc,self.th)
            self.zer.append(zer_*(self.rho<=self.fc))

    def generate_psf(self,coeff_=[]):
        if len(coeff_)==0:
            coeff = np.random.rand(len(self.coeff_zernike))*(self.max_coeff_zernike-self.min_coeff_zernike) + self.min_coeff_zernike
            coeff *= np.sign(coeff[0]) # Because a sign change doesn't change the PSF
        else:
            coeff=coeff_
        
        # PSF generation
        W=np.zeros((self.imgSize,self.imgSize))
        for k in range(len(self.coeff_zernike)):
            W+=coeff[k]*self.zer[k]
            
        pupil=np.exp(-2j*np.pi*W)*(self.rho<=self.fc)

        psf=np.zeros((self.imgSize,self.imgSize,self.nZ))
        defocus=np.linspace(-self.nZ/2,self.nZ/2,self.nZ)*self.stepZ/self.pixelSize
        if self.nZ==1:
            defocus = -0.5*np.ones(1)*self.stepZ/self.pixelSize
        for ii in range(self.nZ):
            propKer=np.exp(-1j*2*np.pi*defocus[ii]*(self.kb**2-self.rho**2+0j)**.5)
            propKer[np.isnan(propKer)]=0
            psfA=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(pupil*propKer)))
            psf[:,:,ii]=np.real(psfA*np.conj(psfA))

        # import ipdb; ipdb.set_trace()
        padPre=int(np.ceil((self.imgSize-self.imgSize_)/2))
        padPost=int(np.floor((self.imgSize-self.imgSize_)/2))
        psf=psf[padPre:self.imgSize-padPost,padPre:self.imgSize-padPost,:]
        psf=psf/np.linalg.norm(psf)
        # psf=psf/np.sum(psf)
        return psf, coeff

"""
2D generator implemented in pytorch

"""
class PSFGenerator2Dzernike_t():
    def __init__(self,coeff_zernike=[5,6],imgSize=17,pixelSize=240,NA=1.41,wavelength=540,nI=1.51,
    min_coeff_zernike=[0,0],max_coeff_zernike=[0.4,0.4],res=256,device='cpu',torch_type='torch.float32'):
        self.device = device
        self.torch_type = torch_type
        self.coeff_zernike=coeff_zernike
        self.pixelSize=pixelSize
        self.NA=NA
        self.wavelength=wavelength
        self.nI=nI
        self.min_coeff_zernike=min_coeff_zernike
        self.max_coeff_zernike=max_coeff_zernike
        self.fc=(NA/wavelength)*pixelSize # Cutoff frequency
        self.kb=(nI/wavelength)*pixelSize # wavenumber
        self.imgSize_=imgSize
        self.imgSize=np.maximum(res,imgSize) # Pad for better discretization in Fourier domain
        lin=np.linspace(-0.5,0.5,self.imgSize)
        XX,YY=np.meshgrid(lin,lin)
        # self.rho=np.sqrt(XX**2+YY**2)
        self.rho,self.th=cart2pol(XX,YY)
        self.rho_t = torch.tensor(self.rho).type(self.torch_type).to(self.device)
        self.th_t = torch.tensor(self.th).type(self.torch_type).to(self.device)
        self.indic = (self.rho_t<=self.fc).view(1,self.imgSize,self.imgSize)
        self.zer=[]
        for k in range(len(coeff_zernike)):
            coefficient=np.zeros(38)
            coefficient[coeff_zernike[k]]=1
            # zer_=zernikecartesian(coefficient,XX,YY)*(self.rho<=self.fc)
            zer_ = galsim.zernike.Zernike(coefficient)
            zer_ = zer_.evalPolar(self.rho/self.fc,self.th)
            tmp = torch.tensor(zer_*(self.rho<=self.fc)).type(torch_type).to(device)
            self.zer.append(tmp.view(1,self.imgSize,self.imgSize))      

    def generate_coeffs(self,N):
        coeff = np.zeros((N,len(self.coeff_zernike)))
        for n in range(N):
            coeff_ = np.random.rand(len(self.coeff_zernike))*(self.max_coeff_zernike-self.min_coeff_zernike) + self.min_coeff_zernike
            coeff_ *= np.sign(coeff_[0]) # Because a sign change doesn't change the PSF
            coeff[n] = coeff_
        return torch.tensor(coeff).type(self.torch_type).to(self.device)

    def generate_psf(self,coeff):        
        # PSF generation
        W = (coeff[:,0].view(-1,1,1))*self.zer[0]
        for k in range(1,len(self.coeff_zernike)):
            W+=(coeff[:,k].view(-1,1,1))*self.zer[k]
            
        pupil=torch.exp(-2j*np.pi*W)*self.indic

        psf=torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(pupil)))
        psf=torch.real(psf*torch.conj(psf))

        padPre=int(np.ceil((self.imgSize-self.imgSize_)/2))
        padPost=int(np.floor((self.imgSize-self.imgSize_)/2))
        psf=psf[:,padPre:self.imgSize-padPost,padPre:self.imgSize-padPost]
        #psf=psf/torch.linalg.norm(psf,dim=(1,2),keepdim=True)
        psf=psf/torch.sum(psf,dim=(1,2),keepdim=True)
        return psf

    def generate_multiple_psf(self,coeff,Npsf,N_coeff):
        psfs = []
        coeffs = []
        for i in range(Npsf):
            coeff = self.generate_coeffs(N_coeff)
            psf = self.generate_psf(coeff)
            coeffs.append(coeff)
            psfs.append(psf)
        
        psfs_t = torch.stack(psfs)
        coeffs_t = torch.stack(coeffs)
        
        return psfs_t,coeffs_t
