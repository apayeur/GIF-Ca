import imp
import sys
sys.path.append('../../Code/')
from iGIF_NP import *
from iGIF_Ca_NP import *
from GIF_Ca import *
from GIF import *
import os
import matplotlib.pyplot as plt


model_type = 'iGIF_NP'
data_folders_for_separate_experiments = ['seventh_set', 'eighth_set', 'ninth_set']
path_results = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'


labels = []
CellNames = {}
for experiment_folder in data_folders_for_separate_experiments:
    folder_path = './' + experiment_folder + '/'
    CellNames[experiment_folder] = [name for name in os.listdir(folder_path) if os.path.isdir(folder_path + name) and '_5HT' in name]
CellNames['eighth_set'].remove('DRN157_5HT') # problematic cell
CellNames['eighth_set'].remove('DRN164_5HT') # problematic cell

iGIFs = []
for experiment_folder in data_folders_for_separate_experiments:
    for cell_name in CellNames[experiment_folder]:
        print '#############################################'
        print '##########     process cell %s    ###' % cell_name
        print '#############################################'
        #################################################################################################
        # Load data
        #################################################################################################
        path_data = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + cell_name + '/'

        iGIFs.append(iGIF.load(path_data + cell_name + '_' + model_type+'_ModelParams.pck'))
        labels.append(cell_name[3:6])
#Compare models
iGIF.compareModels(iGIFs, path_results + 'compare_' + model_type + '.png', labels=labels)

#Average model
iGIF.plotAverageModel(iGIFs, path_results + 'average_' + model_type + '.png')


#cell_name = CellNames['eighth_set'][0]
#path_data = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/' + cell_name + '/'
#model = iGIF.load(path_data + cell_name + '_' + model_type+'_ModelParams.pck')
#(V, theta) = model.getNonlinearCoupling()