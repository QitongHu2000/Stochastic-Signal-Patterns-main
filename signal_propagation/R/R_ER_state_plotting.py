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
Deltas_continuous=load_dict('R_ER_Deltas_continuous'+str(int(100*a))+'_'+str(int(100*b))).tolist()[0]

colors = ['#8DDFBF','#328CA0', '#006837']

E_continuous = np.power(degree, -1+1/a)

fig=plt.figure(figsize=(7,6))
ax=fig.add_subplot(111)

Deltas_continuous=[i*np.abs(E_continuous[-1]/Deltas_continuous[-1]) for i in Deltas_continuous]
ax.scatter(degree, Deltas_continuous,marker='s',color='none', edgecolors=colors[0],s=200,label='Simulation')
ax.plot(degree,E_continuous,color=colors[2],linewidth=4,label='Predicted by\n Theory '+r'$d_i^{-1+\frac{1}{a}}$')

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$d_i$",fontsize=45)
plt.ylabel(r"$\mathbb{E}[\Delta x_i(\infty)]$",fontsize=35)
plt.legend(fontsize=20,loc=1,bbox_to_anchor=(1.1,1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.tight_layout()
# plt.xlim([9,55])
plt.xscale('log')
plt.yscale('log')
# plt.axis('equal')

plt.savefig('R_ER_state_'+str(int(100*a))+'_'+str(int(100*b))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()