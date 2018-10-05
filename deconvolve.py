# load libaries
import numpy as np
import igor.igorpy as igor
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy import fftpack
import scipy


def smoothThroughAveragingNeighbors(data,nBox):
    box = np.ones(nBox)/nBox
    data_smooth = signal.convolve(data, box, mode='same')
    return data_smooth



def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


def fwhm(sigma):
  # Computes the FWHM from sigma for a Gaussian
  return 2.0*np.sqrt(2.0*np.log(2.0))*sigma


# load data
fName = 'psf_8um_bead.pxp'
igorData = igor.load(fName)
xProfile = igorData.x_profile.data
yProfile = igorData.y_profile.data
zProfile = igorData.z_profile.data

#yProfile = yProfile[200:400]
#yProfile = smoothThroughAveragingNeighbors(yProfile,9)

# set parameters
dX = 7.09814e-08 # in m
dY = 7.09584e-08
dZ = 1e-06
beadsLength = 8.0e-6 # beads are on averge 8 um wide

# x, y and z axes
x = np.linspace(dX,len(xProfile)*dX,len(xProfile))
y = np.linspace(dY,len(yProfile)*dY,len(yProfile))
z = np.linspace(dZ,len(zProfile)*dZ,len(zProfile))

# create beads wave form
beadsWaveForm = np.ones(int(beadsLength/dZ)) # square shape of the bead
beadsWaveForm = np.concatenate((np.ones(2)*0.,beadsWaveForm,np.ones(2)*0.))

#################################################################
# fit Gaussian to z profile
# define Gaussian
gauss = lambda x, A, sig, mu: A*np.exp(-( (x-mu)/float(sig))**2 )
# define fit-function and cost function
fitfunc = lambda p, x: gauss(x,p[0],p[1],p[2])
errfunc = lambda p, x, y: fitfunc(p,x)-y

p0 = np.array([20.,1.,0.]) # initial parameter guess
p1, success = scipy.optimize.leastsq(errfunc, p0,args=(z*1e6,zProfile)) # fit routine
# calculate Gaussian with fit parameters
fittedGaussian = fitfunc(p1, z*1e6)

#################################################################
# find best Gaussian point-spread function to reproduced z profile

fitfunc2 = lambda p, x: np.convolve(beadsWaveForm, gauss(x,p[0],p[1],p[2]), mode='same')
errfunc2 = lambda p, x, y: fitfunc2(p,x)-y


q0 = np.array([20.,1.,0.]) # initial parameter guess
q1, success = scipy.optimize.leastsq(errfunc2, p0,args=(z*1e6,zProfile)) # fit routine

convolutionZ = fitfunc2(q1, z*1e6)
psf = gauss(z*1e6,q1[0],q1[1],q1[2])


#pdb.set_trace()
#filtered = np.convolve(beadsWaveForm, gauss(np.arange(50),4,25.), mode='same')


#pointSpreadFz1,remainderZ = signal.deconvolve(fittedGaussian,beadsWaveForm)
#pointSpreadFx2 = deconvolve(beadsWaveForm,yProfile)


# #the deconvolution has n = len(xProfile) - len(beadsWaveForm) + 1 points
# n = len(xProfile)-len(beadsWaveForm)+1
# # so we need to expand it by
# s = (len(xProfile)-n)/2
# #on both sides.
# deconv_res = np.zeros(len(xProfile))
# deconv_res[s:len(xProfile)-s-1] = pointSpreadFx
# deconv = deconv_res
# # now deconv contains the deconvolution
# # expanded to the original shape (filled with zeros)


#### Plot ####
fig , ax = plt.subplots(nrows=4,figsize=(6,10))

#ax[0].plot(x*1e6,xProfile,label="x profile", lw=3 )
ax[0].plot(x*1e6,xProfile,label="x profile", lw=3 )
ax[0].plot(y*1e6,yProfile,label="y profile", lw=3 )
ax[1].plot(z*1e6,zProfile,label="z profile", lw=3 )
ax[1].plot(z*1e6,fittedGaussian,label="z fitted, fwhm = %s" % round(fwhm(p1[1]),2), lw=3 )
#ax[0].plot(filtered/np.sum(gauss(np.arange(50),4,25.)),label="convoluted b waveform", lw=3 )
# we need to divide by the sum of the filter window to get the convolution normalized to 1
#ax[1].plot(z*1e6,zProfile, label="z profile" ,  lw=3 )
ax[2].plot(beadsWaveForm,'o-',label="beads waveform",lw=3 )
#ax[2].plot(gauss(np.arange(50),4,25.),label="beads waveform", lw=3 )

ax[1].plot(z*1e6,convolutionZ,label="convolution fit with bead", lw=3 )
#ax[3].plot(np.real(pointSpreadFx2),label="deconvolution", lw=3 )
ax[3].plot(z*1e6,psf,label=r"point-spread function, $\sigma = %s$, fwhm = %s" % (np.round(q1[1],2),round(fwhm(q1[1]),2)), lw=3 )

#ax[0].set_xlabel(r'distance ($\mu$m)')
ax[3].set_xlabel(r'distance ($\mu$m)')

for i in range(len(ax)):
    #ax[i].set_xlim([0, len(bead)])
    #ax[i].set_ylim([-0.07, 1.2])
    ax[i].legend(loc=1, fontsize=11)
    #if i != len(ax)-1 :
    #    ax[i].set_xticklabels([])

plt.savefig(__file__[:-3] + ".pdf")
plt.show()