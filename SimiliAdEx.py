from Experiment import *
from GIF import *
from Filter_Rect_LogSpaced import *
import scipy.optimize as optimization
from AEC_Badel import *
from Tools import *
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
import os


def simulateDeterministic_forceSpikes(I, V0, spks, model, params):
    # Input parameters
    p_T = len(I)
    p_dt = myExp.dt

    # Model parameters
    p_taum = params['taum']
    p_C = params['C']
    p_gl = p_C / p_taum
    p_El = params['El']
    p_DV = params['DV']
    # p_C = model.C
    # p_gl = model.gl
    # p_El = model.El
    p_Vr = model.Vr
    p_Vt_star = params['VT']
    # p_Vt_star = model.Vt_star
    p_Tref = model.Tref
    p_Tref_i = int(model.Tref / p_dt)

    # Model kernel
    (p_eta_support, p_eta) = model.eta.getInterpolatedFilter(p_dt)
    p_eta = p_eta.astype('double')
    p_eta_l = len(p_eta)

    (p_gamma_support, p_gamma) = model.gamma.getInterpolatedFilter(p_dt)
    p_gamma = p_gamma.astype('double')
    p_gamma_l = len(p_gamma)

    # Define arrays
    V = np.array(np.zeros(p_T), dtype="double")
    I = np.array(I, dtype="double")
    spks = np.array(spks, dtype="double")
    spks_i = Tools.timeToIndex(spks, p_dt)

    # Compute adaptation current (sum of eta triggered at spike times in spks)
    eta_sum = np.array(np.zeros(int(p_T + 1.1 * p_eta_l + p_Tref_i)), dtype="double")
    for s in spks_i:
        eta_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_eta_l] += p_eta
    eta_sum = eta_sum[:p_T]

    gamma_sum = np.array(np.zeros(p_T + 2 * p_gamma_l), dtype="double")
    for s in spks_i:
        gamma_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_gamma_l] += p_gamma
    gamma_sum = gamma_sum[:p_T]

    # Set initial condition
    V[0] = V0

    code = """
             #include <math.h>

             int   T_ind      = int(p_T);
             float dt         = float(p_dt);

             float gl         = float(p_gl);
             float C          = float(p_C);
             float El         = float(p_El);
             float Vr         = float(p_Vr);
             float DVf        = float(p_DV);
             float Vt_star    = float(p_Vt_star);
             int   Tref_ind   = int(float(p_Tref)/dt);


             int next_spike = spks_i[0] + Tref_ind;
             int spks_cnt = 0;


             for (int t=0; t<T_ind-1; t++) {
                 V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + gl*DVf*exp((V[t]-Vt_star-gamma_sum[t])/DVf) + I[t] - eta_sum[t] );

                 if ( t == next_spike ) {
                     spks_cnt = spks_cnt + 1;
                     next_spike = spks_i[spks_cnt] + Tref_ind;
                     V[t-1] = 0 ;
                     V[t] = Vr ;
                     t=t-1;
                 }

             }

             """

    vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr', 'p_Tref', 'p_DV', 'p_Vt_star', 'V', 'I', 'eta_sum',
            'gamma_sum', 'spks_i']

    v = weave.inline(code, vars)

    time = np.arange(p_T) * p_dt
    eta_sum = eta_sum[:p_T]

    return (time, V, eta_sum, gamma_sum)

data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set', 'tenth_set']

# For all experiments, extract the cell names
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]
CellNames['eighth_set'].remove('DRN165_5HT')  # problematic cell
CellNames['eighth_set'].remove('DRN094_5HT')  # problematic cell
CellNames['eighth_set'].remove('DRN156_5HT')  # problematic cell
CellNames['seventh_set'].remove('DRN543_5HT')  # problematic cell
CellNames['ninth_set'].remove('DRN654_5HT')  # problematic cell
CellNames['tenth_set'].remove('DRN656_5HT')  # problematic cell

for experiment_folder in data_folders_for_separate_experiments:
    for CELL_NAME in CellNames[experiment_folder]:

        PATH_DATA = './' + experiment_folder + '/' + CELL_NAME + '/'
        PATH_RESULTS = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + CELL_NAME + '/'

        # Find extension of data files
        file_names = os.listdir(PATH_DATA)
        for file_name in file_names:
            if '.abf' in file_name:
                ext = '.abf'
                break
            elif '.mat' in file_name:
                ext = '.mat'
                break


        # Load AEC data
        #Convert .mat file containing voltage and current traces into numpy arrays
        filename_AEC = PATH_DATA + CELL_NAME + '_aec' + ext
        (sampling_timeAEC, voltage_traceAEC, current_traceAEC) = load_AEC_data(filename_AEC)

        # Create experiment
        myExp = Experiment('Experiment 1', sampling_timeAEC)
        myExp.setAECTrace(voltage_traceAEC, 10.**-3, current_traceAEC, 10.**-12, len(voltage_traceAEC)*sampling_timeAEC, FILETYPE='Array')

        # Load training set data and add to experiement object
        filename_training = PATH_DATA + CELL_NAME + '_training' + ext
        (sampling_time, voltage_trace, current_trace, times) = load_training_data(filename_training)
        myExp.addTrainingSetTrace(voltage_trace, 10**-3, current_trace, 10**-12, len(voltage_trace)*sampling_time, FILETYPE='Array')
        #Note: once added to experiment, current is converted to nA.


        # Create new object to perform AEC
        myAEC = AEC_Badel(myExp.dt)

        # Define metaparametres
        myAEC.K_opt.setMetaParameters(length=200.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
        myAEC.p_expFitRange = [5.0,100.0]
        myAEC.p_nbRep = 15

        # Assign myAEC to myExp and compensate the voltage recordings
        myExp.setAEC(myAEC)
        myExp.performAEC()
        for tr in myExp.trainingset_traces :
            tr.setROI([[2000., sampling_time*(len(voltage_trace)-1)-2000.]])


        #Load model
        model = GIF.load(PATH_RESULTS+CELL_NAME+'_GIF_ModelParams.pck')

        #Load params of dynamic I-V curve
        p = np.loadtxt(PATH_RESULTS + 'params_IV.dat')
        params = {'El':p[0], 'taum':p[1], 'DV':p[2], 'VT':p[3], 'C':p[4]}
        #test
        #params['DV'] = 1.5
        #params['VT'] = -38.



        V0 = myExp.trainingset_traces[0].V[0]
        I = myExp.trainingset_traces[0].I
        spikes = myExp.trainingset_traces[0].getSpikeTimes()
        spike_indices = myExp.trainingset_traces[0].getSpikeIndices()
        (t, V, eta_sum, gamma_sum) = simulateDeterministic_forceSpikes(I, V0, spikes, model, params)
        (time, V_GIF, eta_sum) = model.simulateDeterministic_forceSpikes(I, V0, spikes)

        #Compute average prespike traces
        beginning_prespike_interval = 15. # ms
        beginning_prespike_interval_int = int(beginning_prespike_interval/myExp.dt)
        end_prespike_interval = 5.  # ms
        end_prespike_interval_int = int(end_prespike_interval / myExp.dt)

        VNonLin_prespike_average = np.zeros(beginning_prespike_interval_int - end_prespike_interval_int)
        VLin_prespike_average = np.zeros(beginning_prespike_interval_int - end_prespike_interval_int)
        VExp_prespike_average = np.zeros(beginning_prespike_interval_int - end_prespike_interval_int)
        VExp = myExp.trainingset_traces[0].V

        count = 0
        for spk in spike_indices:
            if not np.isnan(V[spk-beginning_prespike_interval_int:spk-end_prespike_interval_int]).any():
                VNonLin_prespike_average += VExp[spk-beginning_prespike_interval_int:spk-end_prespike_interval_int]-V[spk-beginning_prespike_interval_int:spk-end_prespike_interval_int]
                count += 1
            VLin_prespike_average += VExp[spk-beginning_prespike_interval_int:spk-end_prespike_interval_int] - V_GIF[spk-beginning_prespike_interval_int:spk-end_prespike_interval_int]

        VNonLin_prespike_average /=count
        VLin_prespike_average /= len(spike_indices)

        fig = plt.figure(figsize=(4,3))
        plt.plot(np.arange(-beginning_prespike_interval, -end_prespike_interval, myExp.dt), VNonLin_prespike_average - VNonLin_prespike_average[0], '+', label='Lin-Exp')
        plt.plot(np.arange(-beginning_prespike_interval, -end_prespike_interval, myExp.dt), VLin_prespike_average - VLin_prespike_average[0], label='Lin')
        plt.ylabel('$V_\mathrm{data} - V_\mathrm{model}$ [mV]')
        plt.xlabel('Time [ms]')
        plt.legend(loc='best')
        plt.show()

        fig = plt.figure(figsize=(10,6), facecolor='white')
        plt.subplot(2,1,1)
        plt.plot(t/1000, I,'-b', lw=0.5, label='$I$')
        plt.xlim(17,19)
        plt.ylabel('Current [nA]')
        plt.title('Subthreshold dynamics with forced spikes')
        plt.subplot(2,1,2)
        plt.plot(t/1000, V,'-b', lw=0.5, label='Lin-exp')
        plt.plot(t/1000, V_GIF,'-r', lw=0.5, label='Lin')
        plt.plot(time/1000, myExp.trainingset_traces[0].V,'black', lw=0.5, label='Data')
        #plt.plot(t/1000, -gamma_sum,'-', color='orange', lw=0.5, label='$-\gamma$')
        plt.xlim(17,19)
        plt.ylim(-75,0)
        plt.ylabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.legend(loc='best')
        plt.savefig('../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'+CELL_NAME + '/' + CELL_NAME + '_HybridModel.png', format='png')
        plt.close()
        #plt.show()
