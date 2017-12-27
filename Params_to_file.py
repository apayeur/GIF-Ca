#import sys
#sys.path.append('/ufs/guido/lib/python')


from GIF import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np

# Load GIF model
GIF_TYPE = 'iGIF_NP_'
CELL_NAME = 'DRN651_5HT'
PATH_RESULTS = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + CELL_NAME + '/'
GIF_model_path = PATH_RESULTS + CELL_NAME + '_iGIF_NP_ModelParams' + '.pck'
myGIF = GIF.load(GIF_model_path)


myGIF.dt = 1.e-2
(p_eta_support, p_eta) = myGIF.eta.getInterpolatedFilter(myGIF.dt)
p_eta = p_eta.astype('double')

(p_gamma_support, p_gamma) = myGIF.gamma.getInterpolatedFilter(myGIF.dt)
p_gamma = p_gamma.astype('double')




#Output parameters
fname = PATH_RESULTS + 'params.dat'
File = open(fname, 'w')
File.write(str(myGIF.Vt_star) + '\n')
File.write(str(myGIF.Vr)+ '\n')
File.write(str(myGIF.DV)+ '\n')
File.write(str(myGIF.C)+ '\n')
File.write(str(myGIF.gl)+ '\n')
File.write(str(myGIF.El)+ '\n')
File.write(str(myGIF.Tref)+ '\n')
File.close()

#Output eta kernel
fname = PATH_RESULTS + 'eta.dat'
np.savetxt(fname, p_eta, newline='\n')

#Output gama kernel
fname = PATH_RESULTS + 'gamma.dat'
np.savetxt(fname, p_gamma, newline='\n')


#Output theta-related params
# structure of inFileTheta is
#tau_theta
#size of theta_bins
#values for theta bins
# size of theta_i
#values for theta_i
fname = PATH_RESULTS + 'theta.dat'
with open(fname, 'w') as f:
    f.write(str(myGIF.theta_tau) + '\n')
    f.write(str(myGIF.theta_bins.size) + '\n')
    for theta_bin in myGIF.theta_bins :
        f.write(str(theta_bin) + '\n')
    f.write(str(myGIF.theta_i.size) + '\n')
    for val in myGIF.theta_i :
        f.write(str(val) + '\n')
