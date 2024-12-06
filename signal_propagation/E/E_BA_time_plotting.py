# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
import scipy.stats
from matplotlib.ticker import MaxNLocator

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

B=0.25
eta = 0.3

degree = load_dict('E_BA_degree'+str(int(100*B)))
time_continuous = load_dict('E_BA_continuous'+str(int(100*B)))
time_stochastic = load_dict('E_BA_stochastic'+str(int(100*B)))
# time_theory_continuous = load_dict('E_BA_theory_continuous'+str(int(100*B)))
# time_theory_stochastic = load_dict('E_BA_theory_stochastic'+str(int(100*B)))

colors = ['#EECB8E', '#DC8910','#83272E']

time_theory_continuous = np.power(degree, -1)
time_theory_stochastic = np.power(degree, -4)/np.min(degree)**(-3)

fig=plt.figure(figsize=(6.5,6))
ax=fig.add_subplot(111)

time_stochastic=[i*np.abs(time_theory_stochastic[-1]/time_stochastic[-1]) for i in time_stochastic]
ax.scatter(degree, time_stochastic,marker='s',color='none', edgecolors=colors[0],s=200,label='Stochastic Simulation')

ax.plot(degree,time_theory_stochastic,color=colors[2],linewidth=4,label='    Predicted by\nStochastic Theory')

time_continuous=[i*np.abs(time_theory_continuous[-1]/time_continuous[-1]) for i in time_continuous]
ax.scatter(degree, time_continuous,marker='^',color='none', edgecolors=colors[1],s=200,label='Continuous Simulation')

ax.plot(degree,time_theory_continuous,color='gray',linewidth=4,label='    Predicted by\nContinuous Theory')

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([10,55],['10','55'],fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$d_i$",fontsize=45)
plt.ylabel(r"$\tau_i$",fontsize=45)
# plt.legend(fontsize=20,loc=2,bbox_to_anchor=(1.1,1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.tight_layout()
plt.xlim([9,55])
plt.xscale('symlog')
plt.yscale('log')
# plt.axis('equal')

plt.savefig('E_BA_time_'+str(int(100*B))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()