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

a=1.5
b=2
eta = 0.3

degree = load_dict('R_ER_degree'+str(int(100*a))+'_'+str(int(100*b)))
time_continuous = load_dict('R_ER_continuous'+str(int(100*a))+'_'+str(int(100*b)))
time_stochastic = load_dict('R_ER_stochastic'+str(int(100*a))+'_'+str(int(100*b)))
# time_theory_continuous = load_dict('R_ER_theory_continuous'+str(int(100*B)))
# time_theory_stochastic = load_dict('R_ER_theory_stochastic'+str(int(100*B)))
Deltas_continuous=load_dict('R_ER_Deltas_continuous'+str(int(100*a))+'_'+str(int(100*b))).tolist()[0]

colors = ['#8DDFBF','#328CA0', '#006837']

time_theory_continuous = np.power(degree, 1/a-1)
time_theory_stochastic = np.power(degree, 2/a-2)/np.min(degree)**(1/a-1)
# time_theory_stochastic = time_theory_stochastic.tolist()[0]

fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(111)

time_stochastic=[i*np.abs(np.max(time_theory_stochastic)/np.max(time_stochastic)) for i in time_stochastic]
ax.scatter(degree, time_stochastic,marker='s',color='none', edgecolors=colors[0],s=200,label='Stochastic Simulation')

ax.plot(degree,time_theory_stochastic,color=colors[2],linewidth=4,label='    Predicted by\nStochastic Theory'+r'$d_i^{\frac{1}{a}-1}$')

time_continuous=[i*np.abs(time_theory_continuous[0]/time_continuous[0]) for i in time_continuous]
ax.scatter(degree, time_continuous,marker='^',color='none', edgecolors=colors[1],s=200,label='Continuous Simulation')

ax.plot(degree,time_theory_continuous,color='gray',linewidth=4,label='    Predicted by\nContinuous Theory'+r'$d_i^{\frac{2}{a}-2}$')

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([10,50],['10','50'],fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$d_i$",fontsize=45)
plt.ylabel(r"$\tau_i$",fontsize=45)
plt.legend(fontsize=20,loc=2,bbox_to_anchor=(1.1,1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
plt.tight_layout()
# plt.xlim([2,55])
plt.xscale('log')
plt.yscale('log')
# plt.axis('equal')

plt.savefig('R_ER_time_'+str(int(100*a))+'_'+str(int(100*b))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()