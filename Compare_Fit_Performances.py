import numpy as np
import matplotlib.pyplot as plt

#models = ['GIF', 'GIF-Ca\n$E_\mathrm{Ca}$ fixed', 'GIF-Ca\n$E_\mathrm{Ca}$ free', 'GIF-Ca-Kin\n$E_\mathrm{Ca}$ fixed', 'GIF-Ca-Kin\n$E_\mathrm{Ca}$ free']
models = ['GIF', 'iGIF', 'iGIF-Ca']
#models = ['GIF']
folder_path = '../../../Dropbox/Recherches/Raphe/GIF-Ca/Results/'
filenames = ['GIF_FitPerformance_FreeTauRef.dat',
             'iGIF_NP_FitPerformance_FreeTauRef.dat',
             'iGIF_Ca_NP_ECa_free_FitPerformance_FreeTauRef.dat']

data_type_fit =np.dtype([('cell_name', str, 10),
                        ('Md*', float),
                        ('EpsilonV', float),
                        ('PVar', float)])

fit_performance = {}
for i, filename in enumerate(filenames):
    fit_performance[models[i]] = np.loadtxt(folder_path + filename, dtype=data_type_fit, delimiter='\t')

markers = ['o', 's', '^', 'v', '*', 'p']
colors = ['y', 'b', 'r', 'g', 'c', 'm']



fig = plt.figure(1, figsize=(8, 2.5))
ax = fig.add_subplot(131)
for i in xrange(len(fit_performance)):
    ax.plot(2*i*np.ones(len(fit_performance[models[i]])), fit_performance[models[i]]['Md*'], linestyle='None', marker=markers[i], color=colors[i], markersize=6, alpha=0.5)
    ax.errorbar(2*i, fit_performance[models[i]]['Md*'].mean(), yerr=fit_performance[models[i]]['Md*'].std()/np.sqrt(len(fit_performance[models[i]]['Md*'])), fmt='*k', markersize=6, lw=3, alpha=0.7)

#ax.text(-0.2, 1.05, 'A', transform=ax.transAxes,
#        fontsize=15, fontweight='bold', va='top', ha='right')
ax.set_ylim(0, 1)
ax.set_xticks(range(0, 2*len(models),2))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylabel('Similarity\nmeasure, Md$^*$')

ax = fig.add_subplot(132)
for i in xrange(len(fit_performance)):
    ax.plot(2*i*np.ones(len(fit_performance[models[i]])), fit_performance[models[i]]['EpsilonV']*100, linestyle='None', marker=markers[i], color=colors[i], markersize=6, alpha=0.5)
    ax.errorbar(2*i, fit_performance[models[i]]['EpsilonV'].mean()*100, yerr=100*fit_performance[models[i]]['EpsilonV'].std()/np.sqrt(len(fit_performance[models[i]]['EpsilonV'])), fmt='*k', markersize=6, lw=3, alpha=0.7)
#ax.text(-0.2, 1.05, 'B', transform=ax.transAxes,
#        fontsize=15, fontweight='bold', va='top', ha='right')
ax.set_xticks(range(0, 2*(len(models)),2))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylim(0., 100)
ax.set_ylabel('Explained variance\non $V$, $\epsilon_V$')

ax = fig.add_subplot(133)
for i in xrange(len(fit_performance)):
    ax.plot(2*i*np.ones(len(fit_performance[models[i]])), fit_performance[models[i]]['PVar'], linestyle='None', marker=markers[i], color=colors[i], markersize=6, alpha=0.5)
    ax.errorbar(2*i, fit_performance[models[i]]['PVar'].mean(), yerr=fit_performance[models[i]]['PVar'].std()/np.sqrt(len(fit_performance[models[i]]['PVar'])), fmt='*k', markersize=6, lw=3, alpha=0.7)
#ax.text(-0.2, 1.05, 'C', transform=ax.transAxes,
#        fontsize=15, fontweight='bold', va='top', ha='right')
ax.set_ylim(0,100)
ax.set_xticks(range(0, 2*(len(models)),2))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylabel('Explained variance\non PSTH, pVar')
plt.tight_layout()
plt.savefig(folder_path + 'CompareFitPerformances.png', format='png')
plt.close()
