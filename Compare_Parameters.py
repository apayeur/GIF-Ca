from Experiment import *
from GIF_Ca import *
from iGIF_NP import *
from GIF import *
from AEC_Badel import *
from Tools import *
from Filter_Rect_LogSpaced import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

folder_path = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'
CellNames = [name for name in os.listdir(folder_path) if
             os.path.isdir(folder_path + name) and '_5HT' in name]


def compare_params(model_type='GIF', spec=None):
    # Gather all models
    models = []
    if spec == None:
        for cell_name in CellNames:
            file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_ModelParams.pck'
            print file_path
            if os.path.exists(file_path):
                models.append(GIF.load(file_path))
        params = {'El': [], 'taum': [], 'C': [], 'DV': []}
        for model in models:
            params['El'].append(model.El)
            params['taum'].append(model.C / model.gl)
            params['C'].append(model.C)
            params['DV'].append(model.DV)
    elif spec == 'ECa_fixed':
        for cell_name in CellNames:
            file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_' + spec + '_ModelParams.pck'
            print file_path
            if os.path.exists(file_path):
                models.append(GIF.load(file_path))
        params = {'El': [], 'taum': [], 'C': [], 'DV': [], 'g_Ca': []}
        for model in models:
            params['El'].append(model.El)
            params['taum'].append(model.C / model.gl)
            params['C'].append(model.C)
            params['DV'].append(model.DV)
            params['g_Ca'].append(model.g_Ca)
    elif spec == 'ECa_free':
        for cell_name in CellNames:
            file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_' + spec + '_ModelParams.pck'
            print file_path
            if os.path.exists(file_path):
                models.append(GIF.load(file_path))
        params = {'El': [], 'taum': [], 'C': [], 'DV': [], 'g_Ca': [], 'E_Ca': []}
        for model in models:
            params['El'].append(model.El)
            params['taum'].append(model.C / model.gl)
            params['C'].append(model.C)
            params['DV'].append(model.DV)
            params['g_Ca'].append(model.g_Ca)
            params['E_Ca'].append(model.E_Ca)
    else:
        print 'Wrongful specification...'
    return params



def plot_model_comparison(params, model_type='GIF', spec=None):
    number_of_subplots = len(params.keys())
    xsize = number_of_subplots*8./3
    fig = plt.figure(1, figsize=(xsize, 3))
    #fig.suptitle(model_type + ' model parameters for 5-HT neurons', y=1)
    ax1 = fig.add_subplot(1, number_of_subplots, 1)
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    ax1.boxplot(params['El'], showmeans=True)
    plt.ylabel(r'$E_L$ [mV]')
    ax2 = fig.add_subplot(1, number_of_subplots, 2)
    ax2.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are of
    plt.ylabel(r'$\tau_m$ [ms]')
    ax2.boxplot(params['taum'], showmeans=True)
    ax3 = fig.add_subplot(1, number_of_subplots, 3)
    ax3.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    ax3.set_ylabel(r'$\Delta V$ [mV]')
    ax3.boxplot(params['DV'], showmeans=True)
    ax3.set_ylim(1, 21)
    #plt.show()
    if spec=='ECa_fixed' or spec=='ECa_free':
        ax4 = fig.add_subplot(1, number_of_subplots, 4)
        ax4.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.ylabel(r'$g_\mathrm{Ca}$ [$\mu$S]')
        ax4.boxplot(params['g_Ca'], showmeans=True)
        ax4.set_ylim(-0.02, 0.21)
    if spec=='ECa_free':
        ax5 = fig.add_subplot(1, number_of_subplots, 5)
        ax5.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        plt.ylabel(r'$E_\mathrm{Ca}$ [mV]')
        ax5.boxplot(params['E_Ca'], showmeans=True)
        ax5.set_ylim(-105, 0)
    fig.tight_layout()
    if spec==None:
        plt.savefig(folder_path + model_type + '_paramsBoxPlots.png', format='png')
    else:
        plt.savefig(folder_path + model_type + '_' + spec + '_paramsBoxPlots.png', format='png')
    plt.close(fig)


'''
    fig = plt.figure(3, figsize=(8, 3))
    # fig.suptitle(model_type + ' model parameters for 5-HT neurons', y=1)
    ax1 = fig.add_subplot(131)
    ax1.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    ax1.plot(np.ones(len(params['El'])), params['El'], 'ro', alpha=0.5)
    plt.ylabel(r'$E_L$ [mV]')
    ax2 = fig.add_subplot(132)
    ax2.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are of
    plt.ylabel(r'$\tau_m$ [ms]')
    ax2.plot(np.ones(len(params['taum'])), params['taum'], 'ro', alpha=0.5)
    ax3 = fig.add_subplot(133)
    ax3.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.ylabel(r'$\Delta V$ [mV]')
    ax3.plot(np.ones(len(params['taum'])), params['DV'], 'ro', alpha=0.5)
    fig.tight_layout()
    #plt.show()
    plt.savefig(folder_path + model_type + '_paramsAllSamples.png', format='png')
    plt.close(fig)
'''
def plot_kernel_comparison(params, model_type='GIF', spec=None):
    models = []
    for cell_name in CellNames:
        if spec==None:
            file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_ModelParams.pck'
        else:
            file_path = folder_path + cell_name + '/' + cell_name + '_' + model_type + '_' + spec + '_ModelParams.pck'
        if os.path.exists(file_path):
            models.append(GIF.load(file_path))
    if model_type[0]=='i':
        number_of_subplots = 4
        theta_inf_all = []
    else:
        number_of_subplots = 3
    # Plot kernel
    K_all = []
    eta_all = []
    gamma_all = []
    plt.figure(2, figsize=(8,3))
    K_support = np.linspace(0, 150.0, 300)
    for model in models:
        K = 1. / model.C * np.exp(-K_support / (model.C / model.gl))
        K_all.append(K)
        plt.subplot(1, number_of_subplots, 1)
        plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
        (p_eta_support, p_eta) = model.eta.getInterpolatedFilter(model.dt)
        eta_all.append(p_eta)
        plt.subplot(1, number_of_subplots, 2)
        plt.plot(p_eta_support, p_eta, color='0.3', lw=1, zorder=5)
        plt.subplot(1, number_of_subplots, 3)
        (p_gamma_support, p_gamma) = model.gamma.getInterpolatedFilter(model.dt)
        gamma_all.append(p_gamma)
        plt.plot(p_gamma_support, p_gamma, color='0.3', lw=1, zorder=5)
        if number_of_subplots==4:
            plt.subplot(1, number_of_subplots, number_of_subplots)
            (theta_inf_support, theta_inf) = model.getNonlinearCoupling()
            theta_inf_all.append(theta_inf)
            plt.plot(theta_inf_support, theta_inf, color='0.3', lw=1, zorder=5)

    K_mean = np.mean(K_all, axis=0)
    K_std = np.std(K_all, axis=0)
    eta_mean = np.mean(eta_all, axis=0)
    eta_std = np.std(eta_all, axis=0)
    gamma_mean = np.mean(gamma_all, axis=0)
    gamma_std = np.std(gamma_all, axis=0)
    if number_of_subplots == 4:
        theta_inf_mean = np.mean(theta_inf_all, axis=0)
        theta_inf_std = np.std(theta_inf_all, axis=0)

    plt.subplot(1, number_of_subplots, 1)
    plt.fill_between(K_support, K_mean + K_std, y2=K_mean - K_std, color='gray', zorder=0)
    plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane filter, $\kappa$ [MOhm/ms]')

    plt.subplot(1, number_of_subplots, 2)
    plt.fill_between(p_eta_support, eta_mean + eta_std, y2=eta_mean - eta_std, color='gray', zorder=0)
    plt.plot(p_eta_support, eta_mean, color='red', lw=2, zorder=10)
    plt.xlim(0, 500)
    plt.ylim(0, 0.1)
    plt.xlabel('Time [ms]')
    plt.ylabel('Spike-triggered\nadaptation current, $\eta$ [nA]')

    plt.subplot(1, number_of_subplots, 3)
    plt.fill_between(p_gamma_support, gamma_mean + gamma_std, y2=gamma_mean - gamma_std, color='gray', zorder=0)
    plt.plot(p_gamma_support, gamma_mean, color='red', lw=2, zorder=10)
    plt.xlim(0,250)
    plt.ylim(0, 20)
    plt.xlabel('Time [ms]')
    plt.ylabel('Spike-triggered\nmoving threshold, $\gamma$ [mV]')

    if number_of_subplots==4:
        plt.subplot(1, number_of_subplots, number_of_subplots)
        plt.fill_between(theta_inf_support, theta_inf_mean + theta_inf_std, y2=theta_inf_mean - theta_inf_std, color='gray', zorder=0)
        plt.plot(theta_inf_support, theta_inf_mean, color='red', lw=2, zorder=10)
        plt.plot(np.array([-100,100]), np.array([-100,100]), color='orange', lw=1, zorder=9)
        plt.xlim(-90,-30)
        plt.ylim(-70, -20)
        plt.xlabel('V [mV]')
        plt.ylabel('$V_T^* + \Theta_\infty(V)$ [mV]')

    plt.tight_layout()
    plt.savefig(folder_path + model_type + '_Kernels.png', format='png')
    plt.close()


if __name__ == "__main__":
    model_type = 'iGIF_Ca_NP'
    spec = 'ECa_free'
    params = compare_params(model_type=model_type, spec=spec)
    plot_model_comparison(params, model_type=model_type, spec=spec)
    plot_kernel_comparison(params, model_type=model_type, spec=spec)