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

B=0.2
C=0.1
eta = 0.3

degree = load_dict('B_BA_degree'+str(int(100*B))+'_'+str(int(100*C)))
Deltas_continuous=load_dict('B_BA_Deltas_continuous'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]

colors = ['#ABBAE1', '#526D92', '#1B1464']

E_continuous = np.power(degree, -2)

fig=plt.figure(figsize=(6.5,6))
ax=fig.add_subplot(111)

Deltas_continuous=[i*np.abs(E_continuous[-1]/Deltas_continuous[-1]) for i in Deltas_continuous]
ax.scatter(degree, Deltas_continuous,marker='s',color='none', edgecolors=colors[0],s=200,label='Simulation')
ax.plot(degree,-E_continuous,color=colors[2],linewidth=4,label='Predicted by Theory '+r'$d_i^{-2}$')

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
# plt.legend(fontsize=20,loc=2,bbox_to_anchor=(1.1,1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
plt.tight_layout()
plt.xlim([8,60])
plt.ylim([-10**-1.8,-10**-3.8])
plt.xscale('symlog')
plt.yscale('symlog', linthresh=1e-6)
# plt.axis('equal')

plt.savefig('B_BA_state_'+str(int(100*B))+'_'+str(int(100*C))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()