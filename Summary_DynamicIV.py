import numpy as np
import matplotlib.pyplot as plt
import os

# List separate experiments in separate folder
# data_folders_for_separate_experiments = ['tenth_set']
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set', 'tenth_set']

# For all experiments, extract the cell names
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if
                                    os.path.isdir(folder_path + name) and '_5HT' in name]
CellNames['eighth_set'].remove('DRN165_5HT')  # problematic cell
CellNames['eighth_set'].remove('DRN094_5HT')  # problematic cell
CellNames['eighth_set'].remove('DRN156_5HT')  # problematic cell
CellNames['seventh_set'].remove('DRN543_5HT')  # problematic cell
CellNames['ninth_set'].remove('DRN654_5HT')  # problematic cell
CellNames['tenth_set'].remove('DRN656_5HT')  # problematic cell


data = np.array([[0,0,0,0,0]])
for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        path_data = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + cell_name + '/'
        path_results = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'
        data = np.concatenate((data, (np.loadtxt(path_data + 'params_IV.dat', delimiter='\n')).reshape((1,5))), axis=0)

EL = data[1:,0]
taum = data[1:,1]
DeltaV = data[1:,2]
V_T = data[1:,3]
C = data[1:,4]





fig = plt.figure(1, figsize=(8,3))
#fig.suptitle('EIF model parameters for 5-HT neurons', y=0.99)
ax1 = fig.add_subplot(141)
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
ax1.boxplot(EL, showmeans=True)
plt.ylabel(r'$E_L$ (mV)')
ax2 = fig.add_subplot(142)
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are of
plt.ylabel(r'$\tau_m$ (ms)')
ax2.boxplot(taum, showmeans=True)
ax3 = fig.add_subplot(143)
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.ylabel(r'$\Delta V$ (mV)')
ax3.boxplot(DeltaV, showmeans=True)
ax4 = fig.add_subplot(144)
ax4.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.ylabel(r'$V_T$ (mV)')
ax4.boxplot(V_T, showmeans=True)
fig.tight_layout()
plt.savefig(path_results+'DynamicIV_Params5HT.png', format='png')
plt.close(fig)
