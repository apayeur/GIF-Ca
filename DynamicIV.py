#############################################################
#   This program computes the dynamic I-V curve following   #
#   the procedure in Badel et al.                           #
#   For convenience, it uses methods from the GLIF fitting  #
#   protocol (Pozzorini et al.).                            #
#############################################################

#############################################################
#   This program computes the dynamic I-V curve following   #
#   the procedure in Badel et al.                           #
#   For convenience, it uses methods from the GLIF fitting  #
#   protocol (Pozzorini et al.).                            #
#############################################################

from Experiment import *
from GIF import *
from Filter_Rect_LogSpaced import *
import scipy.optimize as optimization
from AEC_Badel import *
from Tools import *
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
import os
import neo



save_to_file = True

def func(x, E_L, tau_m, DeltaT, V_T):
    return (1 / tau_m) * (E_L - x + DeltaT * np.exp((x - V_T) / DeltaT))


def compute_MSE(ydata, yapprox):
    return np.sum((ydata - yapprox) ** 2) / len(ydata)

# CellNames['eighth_set'].remove('DRN165_5HT')  # problematic cell
# Pathological cells: 094, 156, 165, 654, 656, 659
experiment_folder = 'tenth_set'
cell_name = 'DRN659_5HT'


path_data = './' + experiment_folder + '/' + cell_name + '/'
path_results = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + cell_name + '/'

# Find extension of data files
file_names = os.listdir(path_data)
for file_name in file_names:
    if '.abf' in file_name:
        ext = '.abf'
        break
    elif '.mat' in file_name:
        ext = '.mat'
        break

# Load AEC data
filename_AEC = path_data + cell_name + '_aec' + ext
(sampling_timeAEC, voltage_traceAEC, current_traceAEC) = load_AEC_data(filename_AEC)

# Create experiment
experiment = Experiment('Experiment 1', sampling_timeAEC)
experiment.setAECTrace(voltage_traceAEC, 10. ** -3, current_traceAEC, 10. ** -12,
                       len(voltage_traceAEC) * sampling_timeAEC, FILETYPE='Array')

# Load training set data and add to experiment object
filename_training = path_data + cell_name + '_training' + ext
(sampling_time, voltage_trace, current_trace, time) = load_training_data(filename_training)
experiment.addTrainingSetTrace(voltage_trace, 10 ** -3, current_trace, 10 ** -12,
                               len(voltage_trace) * sampling_time, FILETYPE='Array')
# Note: once added to experiment, current is converted to nA.

# Create new object to perform AEC
myAEC = AEC_Badel(experiment.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=200.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0,
                              clamp_period=1.0)
myAEC.p_expFitRange = [5.0, 100.0]
myAEC.p_nbRep = 15

# Assign myAEC to experiment and compensate the voltage recordings
experiment.setAEC(myAEC)
experiment.performAEC()

# Detect spikes
# experiment.detectSpikes(0.,6.)

# Get indices of the training trace far from the spikes (as per Badel's method)
discarded_interval = 500.
indicesFar = experiment.trainingset_traces[0].getROI_FarFromSpikes(0., discarded_interval)
VFar = experiment.trainingset_traces[0].V[indicesFar]
IFar = experiment.trainingset_traces[0].I[indicesFar]
tFar = time[indicesFar]

# Some initial values for parameters (extracted from the GIF protocol)
# model = GIF.load(PATH_RESULTS+'iGIF_NP_'+CELL_NAME+'.pck')
C = 0.1  # nF
gl = 0.003  # uS
El = -70.
tm = C / gl
DV = 1.
V_T = -40.

#############################
#   Fit dynamic I-V curve   #
#############################
# Find capacitance value
dVFardt = np.gradient(VFar, sampling_time)
ind = np.where(np.logical_and(VFar >= El - 1., VFar <= El + 1.))
ind = ind[0]
X = np.vstack((dVFardt[ind], IFar[ind]))
C = np.var(IFar[ind]) / np.cov(X)[0, 1]

# We now can compute the vector dV/dt - I_inj/C
dVdt_minus_Iinj_overC = dVFardt - IFar / C

# Find the resting potential
# We restrict the range to voltages<-50 so that we lie on the linear portion of the I-V curve
# Serves as an initial guess for the fitting of the dynamic I-V curve
xdata = VFar[VFar < -50.]
ydata = dVdt_minus_Iinj_overC[VFar < -50]
slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
El = -intercept / slope
print 'E_L = %f' % El

# Perform the fit
ub_fit = -35.
ind = np.logical_and(VFar < ub_fit,
                         dVdt_minus_Iinj_overC > -10.)  # Upper bound must be adapted to the given data, i.e. it must only include the exponential increase of the I-V curve. Typically, the bound is in [-33,-27]mV
xdata = VFar[ind]
ydata = dVdt_minus_Iinj_overC[ind]
xbins = 100
n, bins = np.histogram(xdata, bins=xbins)
sy, bins = np.histogram(xdata, bins=xbins, weights=ydata)  # Compute mean I-V
sy2, bins = np.histogram(xdata, bins=xbins, weights=ydata * ydata)  # Compute std I-V
mean = sy / n
std = np.sqrt(sy2 / n - mean * mean)
p_init = np.array([El, tm, DV, V_T])
popt, pcov = optimization.curve_fit(func, xdata, ydata, p0=p_init)
popt = np.concatenate((popt, np.array([C])), axis=0)
perr = np.sqrt(np.diag(pcov))
perr.resize(1, perr.shape[0])

if save_to_file:
    np.savetxt(path_results + '/params_IV.dat', popt, fmt='%.18e', delimiter='\t',
           newline='\n')  # save params to file
    np.savetxt(path_results + '/perr_IV.dat', perr, fmt='%2.10f', delimiter='\t',
           newline='\n')  # save errors on params to file

#######  Saving and plotting results  ########
# Save fitting curve to file
# centers = (bins[1:] + bins[:-1])/2
# centers.resize(len(centers),1)
vfit = np.arange(-90, -20, 0.1)
vfit.resize(len(vfit), 1)
fit = func(vfit, popt[0], popt[1], popt[2], popt[3])
fit.resize(len(fit), 1)
X = np.concatenate((vfit, fit), axis=1)
if save_to_file:
    np.savetxt(path_results + '/fit_IV.dat', X, delimiter='\t', newline='\n')

# Plot
fig = plt.figure(1, (4, 3))
plt.suptitle(cell_name[3:6], fontsize=11)
plt.plot(VFar, dVdt_minus_Iinj_overC, '.', alpha=0.3)
plt.plot(VFar, np.zeros(len(VFar)), '-k', lw=0.3)
# plt.plot((bins[1:] + bins[:-1])/2, func((bins[1:] + bins[:-1])/2, popt[0], popt[1], popt[2], popt[3]), color='red', lw=3, label='Fit')
plt.plot(vfit, fit, '-r', lw=1, label='Fit')
plt.errorbar((bins[1:] + bins[:-1]) / 2, mean, yerr=std, fmt='ok', fillstyle='none', lw=1,
             label='mean $\pm$ std')
# str_p_mean = '{0:.2f}'.format(1000.*np.mean(IFar[:-1]))
# plt.text(-80., 20, r'$\langle I_\mathrm{inj} \rangle$ = '+str_p_mean+' pA')
str_param = '{0:.0f}'.format(1000. * popt[4])
plt.text(-80., 16, r'$C = $' + str_param + ' pF')
str_param = '{0:.0f}'.format(popt[0])
plt.text(-80., 13, r'$E_L = $' + str_param + ' mV')
str_param = '{0:.0f}'.format(popt[1])
plt.text(-80., 10, r'$\tau_m = $' + str(str_param) + ' ms')
str_param = '{0:.2f}'.format(popt[2])
plt.text(-80., 7, r'$\Delta V = $' + str(str_param) + ' mV')
str_param = '{0:.0f}'.format(popt[3])
plt.text(-80., 4, r'$V_T = $' + str(str_param) + ' mV')
plt.xlim(-90, -20)
plt.ylim(-10, 20)
plt.ylabel('$F(V)$ [mV/ms]', fontsize=11)
plt.xlabel('$V$ [mV]', fontsize=11)
# plt.legend(loc='upper left')
plt.tight_layout()
plt.subplots_adjust(top=0.93)
if save_to_file:
    plt.savefig(path_results + '/DynamicIV.png', format='png')
    plt.close(fig)
else:
    plt.show()
