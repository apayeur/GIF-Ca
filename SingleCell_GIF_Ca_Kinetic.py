#####################################################
#   SingleCell_GIF_Ca_Kinetic.py
#   Process a single neuron and extract GIF-Ca-Kinetic parameters.
#   Output files:
#       (1) CellName_GIF_Ca_Raster.png : Raster of GIF-Ca model vs experiments
#       (2) CellName_GIF_Ca_FitPerformance.dat : Md* Epsilon_V(test) PVar for cell CellName
#       (3) GIF_Ca_FitPerformance.dat : Md* Epsilon_V(test) for all cells
#       (4) CellName_GIF_Ca_ModelParams.pck : Model parameters and filters
#   Output folders for files 1,2 and 4 : './Results/CellName/'
#   Output folder for file 3 : './Results/'
#####################################################
from Experiment import *
from GIF_Ca_Kinetic import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import neo


is_E_Ca_fixed = False
if is_E_Ca_fixed:
    spec_GIF_Ca = 'ECa_fixed_'
else:
    spec_GIF_Ca = 'ECa_free_'


experiment_folder = 'ninth_set'
cell_name = 'DRN651_5HT'


#################################################################################################
# Load data
#################################################################################################

path_data = './' + experiment_folder + '/' + cell_name + '/'
path_results = './Results/' + cell_name + '/'

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
experiment.setAECTrace(voltage_traceAEC, 10.**-3, current_traceAEC, 10.**-12, len(voltage_traceAEC)*sampling_timeAEC, FILETYPE='Array')

# Load training set data and add to experiment object
filename_training = path_data + cell_name + '_training' + ext
(sampling_time, voltage_trace, current_trace, time) = load_training_data(filename_training)
experiment.addTrainingSetTrace(voltage_trace, 10**-3, current_trace, 10**-12, len(voltage_trace)*sampling_time, FILETYPE='Array')
#Note: once added to experiment, current is converted to nA.

# Load test set data
filename_test = path_data + cell_name + '_test' + ext
if filename_test.find('.mat') > 0:
    mat_contents = sio.loadmat(filename_test)
    analogSignals = mat_contents['analogSignals']
    times_test = mat_contents['times'];
    times_test = times_test.reshape(times_test.size)
    times_test = times_test*10.**3
    sampling_time_test = times_test[1] - times_test[0]
    for testnum in range(analogSignals.shape[1]):
        voltage_test = analogSignals[0, testnum, :]
        current_test = analogSignals[1, testnum, :] - 5.
        experiment.addTestSetTrace(voltage_test, 10. ** -3, current_test, 10. ** -12,
                                   len(voltage_test) * sampling_time_test, FILETYPE='Array')
elif filename_test.find('.abf') > 0:
    r = neo.io.AxonIO(filename=filename_test)
    bl = r.read_block()
    times_test = bl.segments[0].analogsignals[0].times.rescale('ms').magnitude
    sampling_time_test = times_test[1] - times_test[0]
    for i in xrange(len(bl.segments)):
        voltage_test = bl.segments[i].analogsignals[0].magnitude
        current_test = bl.segments[i].analogsignals[1].magnitude - 5.
        experiment.addTestSetTrace(voltage_test, 10. ** -3, current_test, 10. ** -12,
                                   len(voltage_test) * sampling_time_test, FILETYPE='Array')


#################################################################################################
# PERFORM ACTIVE ELECTRODE COMPENSATION
#################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(experiment.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=experiment.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]
myAEC.p_nbRep = 15

# Assign myAEC to experiment and compensate the voltage recordings
experiment.setAEC(myAEC)
experiment.performAEC()


#################################################################################################
# FIT GIF-Ca-Kinetic
#################################################################################################

# Create a new object GIF
GIF_Ca_fit = GIF_Ca_Kinetic(sampling_time)

# Define parameters and filter characteristics
GIF_Ca_fit.Tref = 6.0
GIF_Ca_fit.eta = Filter_Rect_LogSpaced()
GIF_Ca_fit.eta.setMetaParameters(length=2000.0, binsize_lb=0.5, binsize_ub=500.0, slope=10.0)
GIF_Ca_fit.gamma = Filter_Rect_LogSpaced()
GIF_Ca_fit.gamma.setMetaParameters(length=2000.0, binsize_lb=2.0, binsize_ub=500.0, slope=5.0)

# Define the ROI of the training set to be used for the fit
for tr in experiment.trainingset_traces:
    tr.setROI([[2000., sampling_time * (len(voltage_trace) - 1) - 2000.]])

# Perform the fit
(var_explained_dV, var_explained_V_GIF_Ca_train) = GIF_Ca_fit.fit(experiment, DT_beforeSpike=5.0,
                                                                  is_E_Ca_fixed=is_E_Ca_fixed)
# Save the model
# GIF_Ca_fit.save(path_results + cell_name + '_GIF_Ca_'+spec_GIF_Ca+'ModelParams' + '.pck')

###################################################################################################
# EVALUATE MODEL PERFORMANCES ON THE TEST SET DATA
###################################################################################################

# predict spike times in test set
prediction = experiment.predictSpikes(GIF_Ca_fit, nb_rep=500)

# Compute epsilon_V
epsilon_V = 0.
local_counter = 0.
for tr in experiment.testset_traces:
    SSE = 0.
    VAR = 0.
    # tr.detectSpikesWithDerivative(threshold=10)
    (time, V_est, eta_sum_est) = GIF_Ca_fit.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())
    indices_tmp = tr.getROI_FarFromSpikes(5., GIF_Ca_fit.Tref)

    SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp]) ** 2)
    VAR += len(indices_tmp) * np.var(tr.V[indices_tmp])
    epsilon_V += 1.0 - SSE / VAR
    local_counter += 1
epsilon_V = epsilon_V / local_counter
print "epsilonV = %f" %epsilon_V

# Compute Md*
Md_star = prediction.computeMD_Kistler(8.0, GIF_Ca_fit.dt*2.)
fname = path_results  + cell_name  + '_GIF_Ca_Kinetic_' + spec_GIF_Ca + 'Raster.png'
kernelForPSTH = 50.0
PVar = prediction.plotRaster(fname, delta=kernelForPSTH)


#################################################################################################
#  PLOT TRAINING AND TEST TRACES, MODEL VS EXPERIMENT
#################################################################################################

#Comparison for training and test sets w/o inactivation
V_training = experiment.trainingset_traces[0].V
I_training = experiment.trainingset_traces[0].I
(time, V, eta_sum, V_t, S) = GIF_Ca_fit.simulate(I_training, V_training[0])
fig = plt.figure(figsize=(10,6), facecolor='white')
plt.subplot(2,1,1)
plt.plot(time/1000, V,'--r', lw=0.5, label='GIF-Ca-Kinetic')
plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
plt.xlim(18,20)
plt.ylim(-80,20)
plt.ylabel('Voltage [mV]')
plt.title('Training')

V_test = experiment.testset_traces[0].V
I_test = experiment.testset_traces[0].I
(time, V, eta_sum, V_t, S) = GIF_Ca_fit.simulate(I_test, V_test[0])
plt.subplot(2,1,2)
plt.plot(time/1000, V,'--r', lw=0.5, label='GIF-Ca-Kinetic')
plt.plot(time/1000, V_test,'black', lw=0.5, label='Data')
plt.xlim(5,7)
plt.ylim(-80,20)
plt.xlabel('Times [s]')
plt.ylabel('Voltage [mV]')
plt.title('Test')
plt.legend()
plt.tight_layout()
plt.savefig(path_results  + cell_name + '_GIF_Ca_Kinetic' + spec_GIF_Ca + 'simulate.png', format='png')
plt.close()

# Figure comparing V_model, V_data and I during training with forced spikes
(time, V, eta_sum) = GIF_Ca_fit.simulateDeterministic_forceSpikes(I_training, V_training[0], experiment.trainingset_traces[0].getSpikeTimes())
fig = plt.figure(figsize=(10,6), facecolor='white')
plt.subplot(2,1,1)
plt.plot(time/1000, I_training,'-b', lw=0.5, label='$I$')
plt.xlim(17,20)
plt.ylabel('Current [nA]')
plt.title('Training')
plt.subplot(2,1,2)
plt.plot(time/1000, V,'-b', lw=0.5, label='GIF-Ca-Kinetic')
plt.plot(time/1000, V_training,'black', lw=0.5, label='Data')
plt.xlim(17,20)
plt.ylim(-75,0)
plt.ylabel('Time [s]')
plt.ylabel('Voltage [mV]')
plt.legend(loc='best')
plt.savefig(path_results  + cell_name + '_GIF_Ca_Kinetic' + spec_GIF_Ca + 'simulateForcedSpikes_Training.png', format='png')
plt.close(fig)


