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

xs_continuous=load_dict('B_ER_xs_continuous'+str(int(100*B))+'_'+str(int(100*C)))
ts_continuous=load_dict('B_ER_ts_continuous'+str(int(100*B))+'_'+str(int(100*C)))[0]
xs_stochastic=load_dict('B_ER_xs_stochastic'+str(int(100*B))+'_'+str(int(100*C)))
ts_stochastic=load_dict('B_ER_ts_stochastic'+str(int(100*B))+'_'+str(int(100*C)))[0]

theory_xs_stochastic=np.power(ts_stochastic[1:],1/2)
theory_xs_stochastic=[np.abs(i/np.mean(theory_xs_stochastic)*np.mean(xs_stochastic)/2) for i in theory_xs_stochastic]

colors = ['#ABBAE1', '#526D92', '#1B1464']

fig=plt.figure(figsize=(10.5,6))
ax=fig.add_subplot(111)

# ax.scatter(ts_continuous, xs_continuous,marker='s',color='none', edgecolors=colors[0],s=200,label='Simulation')

ax.scatter(ts_stochastic, np.abs(xs_stochastic),marker='s',color='none', edgecolors=colors[1],s=200,label='Simulation')

ax.plot(ts_stochastic[1:],theory_xs_stochastic,color=colors[2],linewidth=4,label='      Predicted by\nStochastic Theory '+r'$t^{\frac{1}{2}}$')
ax.plot(ts_stochastic[1:],np.ones_like(theory_xs_stochastic)*np.exp(np.mean(np.log(theory_xs_stochastic))),color='gray',linewidth=4,label='      Predicted by\nContinuous Theory '+r'$t^0$')

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel(r"$t$",fontsize=45)
plt.ylabel(r"$\left|\mathbb{E}[\overline{x}(t)]-\overline{x}(0)\right|$",fontsize=35)
plt.legend(fontsize=20,loc=2,bbox_to_anchor=(1.1,1))
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.tight_layout()
plt.ylim([1e-2,1e3])
plt.xscale('log')
plt.yscale('log')
# plt.axis('equal')

plt.savefig('B_ER_state_'+str(int(100*B))+'_'+str(int(100*C))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()