import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# let the signal be box-like
bead = np.repeat([0., 1., 0.], 100)
# and use a gaussian filter
# the filter should be shorter than the signal
# the filter should be such that it's much bigger then zero everywhere
gauss = np.exp(-( (np.linspace(0,50)-25.)/float(12))**2 )
print gauss.min()  # = 0.013 >> 0

# calculate the convolution (np.convolve and scipy.signal.convolve identical)
# the keywordargument mode="same" ensures that the convolution spans the same
#   shape as the input array.
#filtered = scipy.signal.convolve(signal, gauss, mode='same')
filtered = np.convolve(bead, gauss, mode='same')

deconv,  _ = scipy.signal.deconvolve( filtered, gauss )
#the deconvolution has n = len(signal) - len(gauss) + 1 points
n = len(bead)-len(gauss)+1
# so we need to expand it by
s = (len(bead)-n)/2
#on both sides.
deconv_res = np.zeros(len(bead))
deconv_res[s:len(bead)-s-1] = deconv
deconv = deconv_res
# now deconv contains the deconvolution
# expanded to the original shape (filled with zeros)


#### Plot ####
fig , ax = plt.subplots(nrows=4, figsize=(6,7))

ax[0].plot(bead,            color="#907700", label="bead",     lw=3 )
ax[1].plot(gauss,          color="#68934e", label="point-spread function (Gaussian)", lw=3 )
# we need to divide by the sum of the filter window to get the convolution normalized to 1
ax[2].plot(filtered/np.sum(gauss), color="#325cab", label="convoluted" ,  lw=3 )
ax[3].plot(deconv,         color="#ab4232", label="deconvoluted", lw=3 )

for i in range(len(ax)):
    ax[i].set_xlim([0, len(bead)])
    ax[i].set_ylim([-0.07, 1.2])
    ax[i].legend(loc=1, fontsize=11)
    if i != len(ax)-1 :
        ax[i].set_xticklabels([])

plt.savefig(__file__[:-3] + ".pdf")
plt.show()